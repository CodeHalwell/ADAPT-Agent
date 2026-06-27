# LangGraph

[LangGraph](https://langchain-ai.github.io/langgraph/) builds agents as **graphs**:
you declare nodes (functions that read and update a shared *state*) and edges
(including conditional edges that route between nodes), then `compile()` the graph
into a runnable object. A *compiled* graph exposes a callable
`invoke(state) -> state` (and `stream`, `ainvoke`, …). That `invoke` method is the
single seam ADAPT-Agent needs.

This guide shows how to **guard** a LangGraph app at runtime and how to **train**
(evaluate + optimize) it offline. It assumes no prior ADAPT-Agent knowledge.

> Examples: [`examples/langgraph/`](../../examples/langgraph/) — a four-step ladder
> from a one-node graph to a governed, optimized router team.

---

## 1. What you wrap

The wrap target is a **compiled** graph — anything exposing a callable
`invoke(state)`:

```python
from langgraph.graph import StateGraph, START, END

def respond(state: dict) -> dict:
    messages = list(state.get("messages", []))
    messages.append({"role": "assistant", "content": "Hello!"})
    return {**state, "messages": messages}

builder = StateGraph(dict)            # state schema (a dict or a TypedDict)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)
app = builder.compile()               # <-- this is what you wrap

app.invoke({"messages": [{"role": "user", "content": "hi"}]})
```

Unlike prompt-based frameworks (Pydantic AI, OpenAI Agents, …), LangGraph is
**state-dict driven**: the adapter passes your payload straight to `invoke` and
screens every string it can find in the returned state. There is no "latest user
message" extraction step — you pass the whole state.

---

## 2. Install & import-safety

```bash
pip install 'adapt-agent[langgraph]'
```

Importing `adapt_agent` (or `LangGraphAdapter`) **never imports langgraph** — the
framework is only needed at runtime when you build and compile your graph. So
`import adapt_agent` stays cheap and dependency-free, and you only install the
extra for the frameworks you actually use.

---

## 3. Guarding a graph

`LangGraphAdapter` wraps the compiled graph and applies the **six-step governance
pipeline** on every `execute` call:

1. **Input screening** — `Firewall` + `AdversarialDefense` scan every string in the
   payload.
2. **Policy enforcement** — `PolicyEnforcer` is evaluated against the extracted
   `AgentState`; only rules whose `action == "block"` block.
3. **Pre-middleware** — an optional `Middleware` pipeline may rewrite the input.
4. **Traced execution** — `graph.invoke(state)` runs, optionally traced by an
   `AgentObserver`.
5. **Post-middleware** — the pipeline may rewrite the result.
6. **Output screening** — the firewall scans every string in the result.

```python
import re
from adapt_agent import Firewall, AdversarialDefense, PolicyEnforcer, AgentObserver
from adapt_agent.adapters import LangGraphAdapter
from adapt_agent.exceptions import SecurityBlockedError

firewall = Firewall(max_content_length=10_000)   # cap input size (DoS guard)
firewall.add_blocked_pattern(r"(?i)ignore (all|previous) instructions")

adapter = LangGraphAdapter(
    firewall=firewall,                # screen inputs/outputs against patterns + length
    defense=AdversarialDefense(),      # prompt-injection / jailbreak detection
    policy_enforcer=PolicyEnforcer(),  # rules over the agent state
    observer=AgentObserver(),          # tracing
    middleware=None,                   # optional pre/post hooks
    agent_id="support-bot",            # label used in traces (default "<framework>-agent")
    block_on_violation=True,           # raise on a threat (default). False = monitor only
)

guarded = adapter.wrap_agent(app)      # returns an object with .execute(state)
```

Every constructor argument is **optional and keyword-only**. With nothing passed,
the adapter is a transparent pass-through; add controls as you need them.

### Executing

```python
state = {"messages": [{"role": "user", "content": "What are your hours?"}]}
try:
    result = guarded.execute(state)    # the graph's output state (a dict)
    print(result)
    print(guarded.get_state())         # the most recently observed AgentState
except SecurityBlockedError as exc:
    print("Blocked:", exc.reason, exc.threats)  # e.g. ['firewall'] or ['prompt_injection']
```

When `block_on_violation=True`, a firewall/defense hit raises
`SecurityBlockedError("Input blocked by security controls", threats)` **before**
`invoke` runs; a fired `block` policy rule raises
`SecurityBlockedError("Input blocked by policy", ["policy:<rule>"])`; and a hit on
the output raises `SecurityBlockedError("Output blocked by security controls", …)`.

### Monitor mode (`block_on_violation=False`)

Execution proceeds and threats are still recorded — inspect them afterwards:

```python
firewall.get_security_events()
defense.get_detected_attacks()
policy.get_violations()
```

This is the recommended way to roll governance out: observe real traffic first,
then flip blocking on.

---

## 4. Policy, observability, trust & taint

**Policy.** In the adapter pipeline the policy is evaluated with `check_state`, so
rule conditions reference **`state`**, not `message`. The extracted `AgentState`
has `messages`, `context` (every non-`messages` key from your payload), and an
optional numeric `trust_score`. The condition language is a *safe sandbox* (no
`eval`, no function calls, no attribute access, **no negative indices**):

```python
policy.add_rule(
    name="flag_external_context",
    description="Warn when a request carries an external-source tag",
    condition="state['context'] != {}",   # use state[...]; [0] not [-1]
    action="warn",                          # warn | block | modify
    severity="medium",
)
```

Use `policy.check_message({...})` separately if you want to test a rule against a
single `message` dict outside the pipeline.

**Observability.** `AgentObserver` records a trace per `execute`; read them with
`observer.get_traces()` (each has `trace_id`, `operation`, `status`). You can also
`log_event` and `record_metric` for custom instrumentation.

**Trust & taint** are standalone primitives (not adapter args). Combine them with
the pipeline — e.g. lower a source's trust when its input trips the defense, or
mark graph state derived from untrusted input as tainted:

```python
from adapt_agent import TrustManager
from adapt_agent.security import TaintTracker, TaintLevel

trust = TrustManager(); trust.update_trust_score("web-form", -0.3, reason="injection")
taint = TaintTracker(); taint.register_source("web-form", "user_input", TaintLevel.HIGH)
taint.mark_tainted("req-1", ["web-form"]); taint.get_taint_level("req-1")  # HIGH
```

See [`examples/langgraph/02_policy_observability_trust.py`](../../examples/langgraph/02_policy_observability_trust.py).

---

## 5. Optimization (the "training" half)

ADAPT-Agent can evaluate a graph against a **golden dataset** and search for a
better configuration of its prompts, models, hyperparameters, and tool/skill
allow-lists, then apply the winner in place.

### What gets introspected

LangGraph introspection is a **best-effort structural walk** of the compiled
graph's nodes:

| Knob | Kind | Notes |
|------|------|-------|
| node prompts (when found on a node/runnable) | `PROMPT` | only when a node exposes a prompt-like attribute |
| bound chat model id | `MODEL` | when a node carries a bound model object |
| bound temperature | `HYPERPARAM` | when present on the bound model |
| tools | `TOOL` | when a node exposes a tools list (drop-one ablation) |

> **Reality check.** Most LangGraph node logic — including the system prompt —
> lives inside a Python **closure**, which no structural walk can see. So for
> LangGraph you almost always **declare the knobs explicitly**. This is the
> idiomatic pattern, not a workaround.

### Declaring knobs explicitly

Keep tunable values on a live object and bind a `Parameter` to it. Because the
node reads that object on every run, rewriting the parameter changes the next run:

```python
from adapt_agent.optimization import Parameter, ParameterKind, OptimizableAgent

CONFIG = {"prompt": "Answer the question."}

def run(question: str) -> str:                 # drives the compiled graph
    return app.invoke({"question": question})["answer"]

prompt_param = Parameter(
    name="answer_node.prompt", kind=ParameterKind.PROMPT,
    getter=lambda: CONFIG["prompt"],
    setter=lambda v: CONFIG.__setitem__("prompt", v),
    component="answer_node",
)
target = OptimizableAgent.from_components(
    components={"graph": app}, runner=run, parameters=[prompt_param], name="my-graph",
)
```

For tool/skill allow-lists, `target.add_tool_parameter(...)` builds a drop-one
ablation search space in one call:

```python
target.add_tool_parameter(
    name="geo.tools", kind=ParameterKind.TOOL,
    getter=lambda: KNOBS.tools, setter=lambda v: setattr(KNOBS, "tools", list(v)),
    candidate_tools=KNOBS.tools,    # -> [full set, drop-one subsets, ...]
)
```

### Evaluate

```python
from adapt_agent.optimization import GoldenDataset, EvaluationHarness, exact_match, LLMJudge

data = GoldenDataset.from_list([{"input": "...", "expected": "..."}, ...])
judge = LLMJudge(my_completion_fn)             # offline stub, or ClaudeJudge(...) etc.
harness = EvaluationHarness(
    metrics=[exact_match(), judge.as_metric("quality")],
    primary_metric="quality",
    failure_threshold=1.0,                      # what counts as a "failure" to learn from
)
report = harness.evaluate(target, data)
report.score, report.aggregate, report.failures()
```

Built-in metrics include `exact_match`, `contains`, `regex_match`, `token_f1`,
`jaccard`, `numeric_close`, `json_subset`, `levenshtein_ratio`.

### The judge (including adversarial mode)

The `LLMJudge` is provider-agnostic — pass a `ModelProvider`, a registered name,
or any `Callable[[str], str]` (great for offline tests). It scores outputs
(`as_metric`), compares candidates (`compare`), critiques failures, and rewrites
prompts (`improve_prompt`). Set `adversarial=True` to make it a **harsh critic**
that assumes the answer is flawed until proven otherwise and hunts edge cases — and
it can propose brand-new tools/skills from observed failures (`suggest_tools`),
surfaced on `result.recommendations`. Untrusted content is fenced and the rubric
is sent via the provider `system` channel, so a malicious agent output cannot hijack
the grade.

```python
from adapt_agent.optimization.judges import ClaudeJudge
judge = ClaudeJudge(model="claude-opus-4-8", adversarial=True)   # ANTHROPIC_API_KEY from env
```

### Optimizers

All share `optimize(target, data, val_dataset=None)` and apply the best config in
place:

- `CoordinateAscentOptimizer` — greedy per-parameter improvement; the flagship for
  prompt/few-shot tuning and where the judge rewrites instructions. Restrict with
  `kinds=(ParameterKind.PROMPT,)`.
- `BootstrapFewShotOptimizer` — few-shot blocks only.
- `GridSearchOptimizer` / `RandomSearchOptimizer` — exhaustive / sampled.
- `EvolutionaryOptimizer` — population-based mutation + selection.
- `PipelineOptimizer` / `make_default_optimizer(harness, judge=...)` — run several
  stages in sequence (few-shot → prompts → models/hparams → tools/skills) under a
  shared `max_evals` hard cap.

```python
from adapt_agent.optimization import make_default_optimizer
result = make_default_optimizer(harness, judge=judge, max_evals=40).optimize(target, data)
print(result.improvement, result.best_config)
for tip in result.recommendations:
    print(tip)        # advisory new tools/skills the judge proposed
```

### Declarative YAML config

The same run can be described in a file and executed with `run_training`:

```python
from adapt_agent.optimization.config import run_training
result = run_training("examples/langgraph/langgraph.train.yaml")
```

See [`langgraph.train.yaml`](../../examples/langgraph/langgraph.train.yaml). Out-of-range
temperature bounds are clamped to the provider's maximum with a warning; unknown
metric/provider/optimizer names raise a clear `TrainingConfigError`.

---

## 6. Multi-agent / orchestration

LangGraph *is* an orchestration framework — a "team" is just a graph with a router
node and conditional edges to specialist nodes:

```python
builder.add_node("router", router)
builder.add_node("geo", geo); builder.add_node("math", math)
builder.add_edge(START, "router")
builder.add_conditional_edges("router", lambda s: s["route"], {"geo": "geo", "math": "math"})
builder.add_edge("geo", END); builder.add_edge("math", END)
team = builder.compile()
```

You can optimize the **whole team as one unit** (wrap the compiled graph and
declare a `Parameter` per specialist prompt — the optimizer tunes them jointly
against end-to-end golden data) or **each specialist individually** (give each its
own runner + sub-task golden set). A good recipe: tune specialists in isolation
first (cheap, isolates which prompt is failing), then a whole-team pass to tune the
router/coordination prompt with the improved specialists in place. The full
worked example is
[`04_multi_agent_and_training.py`](../../examples/langgraph/04_multi_agent_and_training.py).

---

## 7. Common pitfalls & FAQ

- **"`introspect()` found nothing."** Expected for closure-based nodes — declare
  prompts/tools as explicit `Parameter`s (section 5). The structural walk only
  finds knobs a node exposes as attributes.
- **Policy rule never fires.** In the pipeline, conditions run against `state`, not
  `message`; and the safe sandbox rejects negative indices and function calls. Use
  `state['messages'][0]['content']`, not `[-1]`.
- **Wrong wrap target.** Wrap the **compiled** graph (`builder.compile()`), not the
  builder. `wrap_agent` raises `AdapterError` if the object has no callable
  `invoke`.
- **A read-only knob.** If a parameter's setter raises (e.g. a frozen attribute),
  the optimizer logs a warning, marks it non-optimizable, and continues — it never
  crashes the run.
- **Mutating a parameter doesn't change behaviour.** Make sure the node reads the
  *live* object the parameter binds to (a closure over the same `CONFIG`/`KNOBS`),
  not a snapshot captured at build time.
- **Optimization runs are too small.** For a multi-node team, give `max_evals`
  enough budget for the prompt stage to sweep every node (the default pipeline
  splits the budget across four stages under a shared hard cap).

---

## 8. Examples

| File | What it shows |
|------|---------------|
| [`01_basic_guarded.py`](../../examples/langgraph/01_basic_guarded.py) | Smallest guarded graph; safe vs. blocked input. |
| [`02_policy_observability_trust.py`](../../examples/langgraph/02_policy_observability_trust.py) | Full guard stack in monitor mode + trust/taint. |
| [`03_evaluate_and_optimize.py`](../../examples/langgraph/03_evaluate_and_optimize.py) | Evaluate + optimize a single graph (offline). |
| [`04_multi_agent_and_training.py`](../../examples/langgraph/04_multi_agent_and_training.py) | Router team governed + optimized as one unit + YAML training. |

See also the general [Framework Adapters](../adapters.md) and
[Optimization & Evaluation](../optimization.md) guides.
