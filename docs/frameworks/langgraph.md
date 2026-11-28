# LangGraph & LangChain v1

[LangGraph](https://langchain-ai.github.io/langgraph/) builds agents as **graphs**
of nodes over a shared *state*; LangChain v1's `langchain.agents.create_agent`
builds the standard model-and-tools agent loop *as* a LangGraph graph and is the
constructor both projects now recommend. ADAPT-Agent governs either at two levels:

* **inside** the graph -- every model call and every tool call -- through
  LangChain's middleware, LangGraph's `ToolNode` and LangChain callbacks;
* **around** the graph -- the run's input and final result -- with
  `LangGraphAdapter`.

This page is written against **langgraph 1.2** and **langchain 1.5**.

> Examples: [`examples/langgraph/`](https://github.com/CodeHalwell/ADAPT-Agent/tree/main/examples/langgraph/) -- a five-step ladder
> from a one-node graph to tool governance with approvals and stream screening.

---

## 1. Which seam to use

| You have | Use | Covers |
|----------|-----|--------|
| `create_agent(...)` | `middleware=[governance_middleware(...)]` | every model call (input + output) and every tool call (arguments, approvals, results) of that agent |
| a hand-built `StateGraph` with a `ToolNode` | `ToolNode(tools, **governance_tool_wrapper(...))` | every tool call through that node |
| a hand-built `StateGraph`, model screening | `governance_node(gate)` as a node | the state's messages at that point in the graph |
| several agents / sub-agents / nested graphs | `config={"callbacks": [governance_callback_handler(...)]}` | every model and tool call of the run, abort-only |
| any compiled graph, at its boundary | `LangGraphAdapter(...).wrap_agent(graph)` | the run's input and result; `stream`/`astream` screened before release |
| the deprecated `create_react_agent` | `**governance_hooks(...)` + a governed `ToolNode` | model calls via `pre_model_hook`/`post_model_hook` |

They compose: a typical production setup is the middleware on each agent, the
callback handler as a run-wide backstop, and the adapter at the edge.

---

## 2. Install & import-safety

```bash
pip install 'adapt-agent[langgraph]'   # langgraph>=1.0 and langchain>=1.0
```

`create_agent` and its middleware live in the `langchain` package. Importing
`adapt_agent`, `adapt_agent.integrations.langgraph` or `LangGraphAdapter` **never
imports langgraph or langchain**; the middleware class is built when you call
`governance_middleware(...)`.

---

## 3. Governing `create_agent`: `governance_middleware`

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from adapt_agent import Firewall
from adapt_agent.core.governance import ToolPolicy
from adapt_agent.integrations.langgraph import governance_middleware

agent = create_agent(
    model,
    tools=[search, send_email, drop_table],
    middleware=[
        governance_middleware(
            firewall=firewall,                       # model I/O, tool args and results
            tool_policy=ToolPolicy(deny=["drop_*"], require_approval=["send_email"]),
            on_tool_violation="refuse",              # or "raise" (default)
            agent_id="support",
        )
    ],
    checkpointer=InMemorySaver(),                    # only needed for approvals
)
```

It takes the same controls as every other factory (`gate`, `firewall`, `defense`,
`policy_enforcer`, `block_on_violation`, `agent_id`, `tool_policy`) plus
`on_tool_violation`, `screen_output`, `screen_stream`, `approvals`,
`approval_args` and `observer`.

### Model calls

Both halves run inside `wrap_model_call` (and `awrap_model_call` for
`ainvoke`/`astream`):

* **before** the model: the request's messages -- exactly what is about to be
  sent, after any earlier middleware edited them -- are screened with the
  firewall and adversarial defense, and the policy is evaluated. A block raises
  `SecurityBlockedError` and the model is never called. Earlier model turns are
  screened by their text only; their tool-call arguments were already screened
  where they belong.
* **after** the model: the response (text, tool calls, structured response) is
  screened with the output controls *before* it is written to the graph state or
  checkpointed. A block raises `SecurityBlockedError`.

Because this fires on every model call, it screens tool results on their way back
to the model too -- not just the user's first message.

### Tool calls

| Gate verdict | `on_tool_violation="raise"` (default) | `on_tool_violation="refuse"` |
|--------------|----------------------------------------|-------------------------------|
| call **blocked** (denied tool, screened argument, `block` policy rule) | run aborts with `SecurityBlockedError`; the tool never runs | the tool never runs; the model receives an error `ToolMessage` with `decision.reason` (tool + threat labels, no content) and carries on |
| call **asks** (approval list, `require_approval` rule) | `interrupt()` -- see *Approvals*; without a checkpointer, treated as blocked | same |
| result **blocked** (injection in what the tool returned) | run aborts before the next model call | the result is replaced by `Result of tool '<name>' withheld by ADAPT-Agent ...` |
| report-only gate (`block_on_violation=False`) | everything runs; each finding is logged to `observer` with `report_only=True` | same |

`"raise"` is the default because it matches the model-call screens and because a
refusal hands a possibly-injected model another turn to try a different route.
`create_agent`'s `ToolNode` re-raises the error; a hand-built
`ToolNode(handle_tool_errors=True)` would turn it into an error message instead.

### Approvals (`"ask"`)

With a checkpointer (and a `thread_id` in `config`), an ask pauses the run with a
LangGraph `interrupt()`; the tool has not run. The interrupt value is shaped like
`HumanInTheLoopMiddleware`'s, both its per-call and batched forms, so an approval
UI built for that middleware renders it unchanged:

```python
{
    "type": "tool_approval", "tool_call_id": "call-1", "name": "send_email",
    "args": {"to": "..."}, "description": "Tool call 'send_email' needs approval [support]: tool:approval:send_email.",
    "action_requests": [{"name": ..., "args": ..., "description": ...}],
    "review_configs": [{"action_name": "send_email", "allowed_decisions": ["approve", "edit", "reject", "respond"]}],
    "source": "adapt_agent", "agent_id": "support", "threats": ["tool:approval:send_email"],
}
```

Resume with `agent.invoke(Command(resume=<answer>), config)`:

| Answer | Effect |
|--------|--------|
| `{"type": "approve"}`, `True`, `"approve"`, or `{"decisions": [{"type": "approve"}]}` | the tool runs (its result is still screened) |
| `{"type": "reject", "message": "..."}`, `False`, `"reject"` | refusal `ToolMessage` to the model |
| `{"type": "edit", "edited_action": {"name": <same tool>, "args": {...}}}` | the edited arguments are **re-screened**, then run; switching tools is refused |
| `{"type": "respond", "message": "..."}` | the message is returned in the tool's place (screened as a tool result) |
| anything else | treated as a rejection |

`args` are included because an approver cannot judge a call without them, and an
ask is only raised for arguments the firewall and defense already passed (a
content hit is a block, which outranks an ask). `approval_args=False` sends `{}`
instead and offers only approve/reject. Without a checkpointer, or with
`approvals=False`, an ask **fails closed** to a block. Async interrupts
(`ainvoke`/`astream`) need Python 3.11+, as for LangGraph itself.

### Policy state

A fail-closed `PolicyEnforcer` (the default) treats a rule it cannot evaluate as
violated, so the state is built deliberately. In the middleware and the tool
wrapper a rule can rely on:

* `state['messages']` -- the messages being screened;
* every other key of the **graph state** (custom `state_schema` fields; a
  Pydantic or dataclass state is converted to a mapping);
* `state['context']` -- the run's `Runtime.context` (`create_agent(context_schema=...)`
  and `invoke(..., context=...)`) when it is a mapping, dataclass or Pydantic
  model; otherwise the state's own `context` key, else `{}`;
* in a tool-scoped rule, also `tool['name']` and `tool['arguments']`.

```python
policy.add_rule("admins_only", "deleting needs an admin",
                "tool['name'] == 'drop_table' and state['context']['role'] != 'admin'",
                action="block")
agent.invoke(inputs, context=Ctx(role="intern"))   # -> blocked, tool never runs
```

`governance_callback_handler` sees no graph state: its rules get `messages` and an
empty `context` only.

### Streaming

With `stream_mode="messages"`, LangGraph forwards model tokens to the consumer
from a callback **while the model is still generating** -- before any post-model
check can run. Two behaviours:

* `screen_stream=False` (default): tokens stream as usual. A blocked response is
  still kept out of the graph state and the final result, but **tokens already
  streamed have been seen by the consumer**.
* `screen_stream=True`: the model call is tagged `nostream`, so LangGraph skips its
  token stream and delivers the finished message from the model node's output --
  which only happens after output screening has passed it. A blocked response
  never reaches the consumer. **Trade-off:** each model message arrives whole,
  when it is finished, instead of token by token (time-to-first-token becomes
  time-to-last-token). Needs a `BaseChatModel` instance (anything else raises
  `TypeError` rather than silently streaming).

`stream_mode="updates"` and `"values"` never carry a blocked response either way,
because output is screened inside the model node before it writes its update.
`LangGraphAdapter.stream` adds buffering at the boundary (section 6).

### Composing with built-in middleware

LangChain ships `PIIMiddleware`, `HumanInTheLoopMiddleware`,
`ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`, retry and fallback
middleware. Use them alongside governance rather than re-implementing them: PII
redaction and call limits are theirs, threat screening and tool policy are
ADAPT-Agent's. List `governance_middleware` **first** so it is the outermost
wrapper: it then screens what the model and tools finally receive and return,
after the others have acted. Several `governance_middleware` instances can share
one agent if their `agent_id`s differ (`create_agent` requires unique middleware
names).

### MCP tools

MCP tools loaded with `langchain-mcp-adapters` are ordinary LangChain `BaseTool`s
executed by the same `ToolNode`, so `wrap_tool_call` governs them like local tools
-- there is no separate MCP code path in `create_agent`. Match them in a
`ToolPolicy` by the names your MCP client gives them. (`langchain-mcp-adapters` is
not part of ADAPT-Agent's test matrix; the seam is the same `ToolNode` call
exercised by the tests.)

### Sub-agents and supervisors

Middleware is per agent. A sub-agent used as a tool (a tool that calls
`sub_agent.invoke(...)`) or as a handoff target is governed when it carries its own
`governance_middleware`; a block there propagates through the supervisor's tool
call and aborts the whole run (`"raise"`), or comes back to the supervisor as the
sub-agent's answer (`"refuse"`). An agent *without* the middleware is not covered
by its parent's -- use the run-level callback handler (section 5) for that.

---

## 4. Hand-built graphs

```python
from langgraph.prebuilt import ToolNode
from adapt_agent.integrations.langgraph import governance_tool_wrapper, governance_node

tools_node = ToolNode(tools, **governance_tool_wrapper(tool_policy=policy, firewall=fw))
builder.add_node("tools", tools_node)
builder.add_node("guard", governance_node(gate))          # screens the state's messages
```

`governance_tool_wrapper` returns `{"wrap_tool_call": ..., "awrap_tool_call": ...}`
-- the two `ToolNode` arguments -- with exactly the middleware's tool behaviour,
approvals included. `governance_node(gate, direction="input"|"output")` is a node
that screens the state's messages and returns no update; on input it evaluates
policy against the same state shape as above (a Pydantic state included).

---

## 5. Run-level: `governance_callback_handler`

```python
from adapt_agent.integrations.langgraph import governance_callback_handler

backstop = governance_callback_handler(firewall=fw, tool_policy=policy)
supervisor.invoke(inputs, {"callbacks": [backstop]})
```

LangChain callbacks propagate to everything a run starts: each agent of a
supervisor, a sub-agent invoked inside a tool, nested graphs -- with or without
middleware. The handler screens each chat model's input (`on_chat_model_start`)
and output (`on_llm_end`), each tool call (`on_tool_start`) and each tool result
(`on_tool_end`), and **aborts** the run with `SecurityBlockedError` on any block.
It cannot refuse softly or pause for approval (a callback cannot substitute a
result), so an ask is a block here. LangChain logs `Error in
AdaptGovernanceCallbackHandler...` before re-raising; that line is expected.

---

## 6. Wrapping the boundary: `LangGraphAdapter`

`LangGraphAdapter` wraps any compiled graph and applies the six-step pipeline on
every `execute`/`aexecute`: input screening, policy, pre-middleware, traced run,
post-middleware, output screening.

```python
from adapt_agent.adapters import LangGraphAdapter

guarded = LangGraphAdapter(firewall=fw, defense=AdversarialDefense(),
                           policy_enforcer=policy, observer=AgentObserver(),
                           agent_id="support-bot").wrap_agent(agent)
```

**Run configuration.** Extra keyword arguments reach `invoke`/`ainvoke`
unchanged -- `config=`, `context=`, `version=`:

```python
config = {"configurable": {"thread_id": "user-42"}}
out = guarded.execute({"messages": [...]}, config=config)
if out.get("__interrupt__"):                     # paused for approval
    request = out["__interrupt__"][0].value
    out = guarded.execute(Command(resume={"type": "approve"}), config=config)
```

Interrupts are never dropped: with `version="v2"` the `GraphOutput` is unpacked to
the same `{..., "__interrupt__": [...]}` mapping. A `Command` is screened as input
(its resume value and update enter the graph) and policy is evaluated against the
thread's **checkpointed** state plus the `Command`'s `update` -- not against the
`Command` itself, which has none of the keys a rule gates on. Pre-middleware is
not applied to a `Command`.

**Streaming.** `guarded.stream(...)` / `guarded.astream(...)` take the same
arguments as the graph's. The input is screened before the graph starts; every
item is screened with the output controls **before it is yielded**. In
`stream_mode="messages"` the token chunks of one message are held until the
message is complete and released together, so a pattern split across tokens is
still caught -- at the cost of token-by-token delivery. Other modes are screened
item by item. A blocked item raises `SecurityBlockedError` and is never yielded.

A message counts as complete when its last chunk says so (LangChain's
`chunk_position == "last"`, or a provider `finish_reason` / `stop_reason`), when
an `updates` event names the node that produced it or a `values` event arrives,
or when the stream ends. A change of message id is *not* taken as completion:
chunks of two messages interleave when nodes or model calls run in parallel, and
each message is still screened whole. Items are released in arrival order, so
with a model that never marks its last chunk and `stream_mode="messages"` alone,
delivery happens at the end of the stream.

When `block_on_violation=True`, the adapter raises
`SecurityBlockedError("Input blocked by security controls", threats)` before the
graph runs, `"Input blocked by policy"` for a `block` rule, and `"Output blocked by
security controls"` for the result. With `block_on_violation=False` it records and
proceeds -- the recommended way to roll governance out.

---

## 7. Policy rules, observability, trust & taint

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

See [`examples/langgraph/02_policy_observability_trust.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/langgraph/02_policy_observability_trust.py).


## 8. The deprecated `create_react_agent` and `governance_hooks`

`langgraph.prebuilt.create_react_agent` is **deprecated** (moved to
`langchain.agents.create_agent`, to be removed in LangGraph 2.0). `governance_hooks`
returns its `pre_model_hook`/`post_model_hook` arguments and keeps working there:

```python
agent = create_react_agent(model, ToolNode(tools, **governance_tool_wrapper(tool_policy=p)),
                           **governance_hooks(firewall=fw))
```

`create_agent` has **no** hook parameters: `create_agent(..., **governance_hooks(...))`
raises `TypeError: create_agent() got an unexpected keyword argument 'pre_model_hook'`.
For `create_agent`, use `middleware=[governance_middleware(...)]` -- it covers the
hooks' model screening *and* tool calls.

---

## 9. What is and is not covered

Covered, and tested against the real framework (`tests/test_real_langgraph.py`):
denied tools never run (both modes); screened arguments never reach the tool; a
poisoned result never reaches the next model call; asks pause via `interrupt()` and
fail closed without a checkpointer; model input/output blocks; untouched clean
runs; sub-agents used as tools; the run-level callback; report-only recording;
sync and async; `stream_mode="messages"` with and without `screen_stream`; the
adapter's threaded runs, interrupt/resume and governed streams.

Not covered:

* **Token streaming without `screen_stream=True`** -- tokens already streamed
  before a block are out (see *Streaming*).
* **An agent with neither the middleware nor the callback handler** -- middleware
  does not inherit into sub-agents.
* **`ToolMessage.artifact`** -- not sent to the model by LangChain and not
  screened; only `content` (or a `Command`'s whole update) is.
* **The system prompt** -- developer-supplied; the model request's `messages` are
  screened, the `system_message` is not.
* **Custom stream output** (`get_stream_writer()` in a node or tool) -- not a
  model call; screened only by `LangGraphAdapter.stream`.

---

## 10. Optimization (the "training" half)

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
judge = ClaudeJudge(model="claude-opus-5-5", adversarial=True)   # ANTHROPIC_API_KEY from env
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

See [`langgraph.train.yaml`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/langgraph/langgraph.train.yaml). Out-of-range
temperature bounds are clamped to the provider's maximum with a warning; unknown
metric/provider/optimizer names raise a clear `TrainingConfigError`.

---

## 11. Multi-agent / orchestration

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
[`04_multi_agent_and_training.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/langgraph/04_multi_agent_and_training.py).

---

## 12. Common pitfalls & FAQ

- **`TypeError: create_agent() got an unexpected keyword argument 'pre_model_hook'`.**
  `governance_hooks` is for the deprecated `create_react_agent`; use
  `middleware=[governance_middleware(...)]` with `create_agent`.
- **Every request is blocked by policy.** A fail-closed rule referencing a key the
  hook's state does not carry is a violation. Check the key list in *Policy state*;
  pass run-scoped data as `context=` (middleware) or as state keys.
- **An approval never pauses.** Interrupts need a checkpointer and a `thread_id` in
  `config`; without one an ask is refused, by design.
- **"`introspect()` found nothing."** Expected for closure-based nodes — declare
  prompts/tools as explicit `Parameter`s (section 10). The structural walk only
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

## 13. Examples

| File | What it shows |
|------|---------------|
| [`01_basic_guarded.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/langgraph/01_basic_guarded.py) | Smallest guarded graph; safe vs. blocked input. |
| [`02_policy_observability_trust.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/langgraph/02_policy_observability_trust.py) | Full guard stack in monitor mode + trust/taint. |
| [`03_evaluate_and_optimize.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/langgraph/03_evaluate_and_optimize.py) | Evaluate + optimize a single graph (offline). |
| [`04_multi_agent_and_training.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/langgraph/04_multi_agent_and_training.py) | Router team governed + optimized as one unit + YAML training. |
| [`05_tool_governance_middleware.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/langgraph/05_tool_governance_middleware.py) | `create_agent` + `governance_middleware`: denied tool, withheld result, approval interrupt/resume, `screen_stream`, run-level callback. |

See also the general [Framework Adapters](../adapters.md) and
[Optimization & Evaluation](../optimization.md) guides.
