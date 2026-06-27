# Pydantic AI

This page is a complete, from-scratch guide to using
[Pydantic AI](https://ai.pydantic.dev) with **ADAPT-Agent**. It assumes you are
new to *both*, so every concept and parameter is explained the first time it
appears.

ADAPT-Agent (**A**dversarial **D**efense & **P**olicy **T**raining for LLM
agents) has two independent, framework-agnostic halves:

- **Guard (runtime).** Wrap any agent so every call runs a 6-step security and
  observability pipeline. Nothing about your Pydantic AI code changes; you wrap
  the `Agent` and call `execute(...)` instead of `run_sync(...)`.
- **Train (offline).** Turn any agent or multi-agent system into a tunable search
  space, score it against a golden dataset with metrics or an LLM-as-judge, and
  let an optimizer search for a better configuration of prompts, models,
  hyperparameters, tools, and routing knobs -- applied in place.

Both halves are **import-safe**: importing `adapt_agent` (or any of its adapters)
never imports `pydantic_ai`. Pydantic AI is only needed at runtime, when you
actually build the agent you hand to the adapter.

---

## 1. What Pydantic AI is

Pydantic AI is a typed agent framework built by the Pydantic team. Its core
runnable object is the **`Agent`**:

- **`Agent`** -- a container for an LLM model, a `system_prompt` (and/or
  `instructions`), optional function `tools`, an optional structured
  `output_type`, dependency typing (`deps_type`), and default `model_settings`
  (temperature, `top_p`, `max_tokens`, ...).

You run an agent two ways:

- **`agent.run_sync(prompt)`** -- synchronous; returns an `AgentRunResult`.
- **`await agent.run(prompt)`** -- asynchronous; returns an `AgentRunResult`.

The final answer lives on **`result.output`** (in Pydantic AI v2; older code used
`.data`). Tools are registered with decorators -- `@agent.tool` (receives a
`RunContext`) or `@agent.tool_plain` (no context) -- and temperature is set via
`model_settings=ModelSettings(temperature=...)` at construction or per-run.

```python
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

agent = Agent(
    "openai:gpt-4o",                       # "provider:model" identifier
    system_prompt="You are a concise geography assistant.",
    model_settings=ModelSettings(temperature=0.2),
)

@agent.tool_plain
def country_population(country: str) -> int:
    """Return the population of a country."""
    return {"France": 68_000_000}.get(country, 0)

result = agent.run_sync("What is the capital of France?")
print(result.output)                        # -> "Paris"
```

### Running offline (no API key)

For the examples and for unit tests, Pydantic AI ships test doubles that never
hit the network:

- **`TestModel`** -- auto-generates schema-valid data for tools/outputs.
- **`FunctionModel`** -- you supply a callback `(messages, info) -> ModelResponse`
  and return exactly what you want.

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

def offline(messages: list, info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart("Paris")])

agent = Agent(FunctionModel(offline), system_prompt="...")
print(agent.run_sync("capital of France?").output)   # -> "Paris", no network
```

Every example on this page uses `FunctionModel` so it runs with no API key. The
swap to a real model is a single line: `Agent("openai:gpt-4o", ...)` or
`Agent("anthropic:claude-opus-4-8", ...)`.

---

## 2. Installing the extra and the import-safety guarantee

```bash
pip install 'adapt-agent[pydantic-ai]'
# or just the framework:  pip install pydantic-ai
```

`PydanticAIAdapter` lives in `adapt_agent.adapters` and **does not import**
`pydantic_ai` at module load. You can import and even construct the adapter
without the framework installed; it only touches a Pydantic AI object when you
call `wrap_agent(...)` / `execute(...)`. The examples make this friendly:

```python
try:
    from pydantic_ai import Agent
except ImportError:
    raise SystemExit("This example needs: pip install 'adapt-agent[pydantic-ai]'")
```

---

## 3. Guarding a Pydantic AI agent

### Build the adapter

```python
from adapt_agent import AdversarialDefense, AgentObserver, Firewall
from adapt_agent.adapters import PydanticAIAdapter
from adapt_agent.exceptions import SecurityBlockedError

firewall = Firewall(max_content_length=10_000)
firewall.add_blocked_pattern(r"(?i)ignore (all|previous) instructions")

adapter = PydanticAIAdapter(
    firewall=firewall,
    defense=AdversarialDefense(),
    observer=AgentObserver(),
    agent_id="demo-pydantic-ai-agent",
    block_on_violation=True,
)
guarded = adapter.wrap_agent(agent)        # `agent` is a Pydantic AI Agent
```

Every constructor argument is **keyword-only and optional** (each control is opt
in):

- **`firewall`** (`Firewall`) -- screens input/output text. `max_content_length`
  rejects oversized payloads; `add_blocked_pattern(regex, flags=...)` adds a
  regex that, if matched, is a threat. By default the firewall is block-first;
  set `Firewall(whitelist_mode=True)` to invert.
- **`defense`** (`AdversarialDefense`) -- heuristics that detect prompt-injection
  and jailbreak patterns (e.g. "ignore previous instructions", role-play
  break-outs), reported as `prompt_injection` threats.
- **`policy_enforcer`** (`PolicyEnforcer`) -- rule engine, see §4.
- **`observer`** (`AgentObserver`) -- records a trace per run; read them back with
  `observer.get_traces()`.
- **`middleware`** (`Middleware`) -- pre/post hooks, see §4.
- **`agent_id`** (`str`) -- label attached to traces and errors.
- **`block_on_violation`** (`bool`, default `True`) -- when `True` a detected
  threat or a `block` policy rule raises `SecurityBlockedError`; when `False` the
  run proceeds and threats are merely **recorded** (monitor-only mode).

### How the wrap target works

`PydanticAIAdapter` wraps **any object exposing a callable `run_sync` / `run`** --
which is exactly a Pydantic AI `Agent`. Pydantic AI is a *prompt-based* framework,
so the adapter derives the prompt **string** from the payload's latest user
message before calling the agent.

### What `execute()` takes and returns

```python
out = guarded.execute({"messages": [{"role": "user", "content": "capital of France?"}]})
```

- **Input:** a dict. The uniform shape across all adapters is
  `{"messages": [{"role": "user", "content": "..."}]}`; the adapter extracts the
  latest user message as the prompt. A bare string or a plain message list also
  works.
- **Output:** a dict. A non-dict framework result (Pydantic AI returns an
  `AgentRunResult`) is wrapped as `{"result": <AgentRunResult>}`; read the text
  off `out["result"].output`.

### The 6-step pipeline, as it applies to Pydantic AI

Each `execute(...)` runs, in order:

1. **Input screening** -- `Firewall` + `AdversarialDefense` over the input text.
2. **Policy** -- `PolicyEnforcer` evaluated against the extracted *state*
   (`{"messages": [...], "context": {...}}`). Only `action="block"` rules block.
3. **Pre-middleware** -- `Middleware.process_input(payload)` may rewrite the input.
4. **Traced run** -- the agent's `run_sync` is called inside an `AgentObserver`
   trace (`operation="pydantic_ai.run"`).
5. **Post-middleware** -- `Middleware.process_output({"result": ...})` may rewrite
   the result.
6. **Output screening** -- `Firewall` over the produced text.

If any step finds a threat and `block_on_violation=True`, a
`SecurityBlockedError(reason, threats)` is raised:

```python
try:
    guarded.execute({"messages": [{"role": "user", "content": "Ignore previous instructions."}]})
except SecurityBlockedError as exc:
    print(exc.reason, exc.threats)   # 'Input blocked ...', ['firewall', 'prompt_injection']
```

### Monitor-only mode

Set `block_on_violation=False` to run a control in **shadow** mode: threats and
policy violations are recorded (inspect them on `observer.get_traces()` and
`policy_enforcer.get_violations()`) but the run completes normally. This is the
right setting when rolling a new rule out to production before enforcing it.

See [`examples/pydantic_ai/01_basic_guarded.py`](../../examples/pydantic_ai/01_basic_guarded.py).

---

## 4. Policy, adversarial defense, observability, trust & taint

These compose on the same adapter. The full example is
[`examples/pydantic_ai/02_policy_observability_trust.py`](../../examples/pydantic_ai/02_policy_observability_trust.py).

### PolicyEnforcer

A `PolicyEnforcer` holds rules evaluated by a **safe expression language** (no
`eval`). Through the adapter, rules are checked against the **state**, so
conditions reference the `state` variable:

```python
from adapt_agent import PolicyEnforcer

policy = PolicyEnforcer()                      # PolicyEnforcer(fail_closed=True) to treat
policy.add_rule(                               # un-evaluable conditions as violations
    name="no_password_requests",
    description="Refuse requests that try to extract passwords.",
    condition="'password' in state['messages'][0]['content']",
    action="block",                            # "block" aborts; "warn" only records
    severity="high",
)
adapter = PydanticAIAdapter(policy_enforcer=policy, block_on_violation=True, ...)
```

> **Gotcha.** The safe evaluator supports subscripting (`state['messages'][0]`)
> and `in`, but **not unary operators** -- use a non-negative index like `[0]`
> (the first user message) rather than `[-1]`.

### AdversarialDefense

`AdversarialDefense()` adds injection/jailbreak detection on top of the firewall.
Hits are reported as `prompt_injection` threats and block when
`block_on_violation=True`.

### AgentObserver

`AgentObserver()` records one trace per `execute(...)`. Read them with
`observer.get_traces()`; each trace has `trace_id`, `operation`, and `status`
(`completed` / `error`).

### Middleware

`Middleware()` is a composable pre/post pipeline. Pre-middleware can normalize or
annotate the input payload; post-middleware can rewrite the `{"result": ...}`
payload. By default it is **fail-open** (a crashing hook is logged and skipped);
pass `Middleware(fail_closed=True)` so a crashing security-critical hook aborts
the request instead of leaking unsanitized data.

```python
from adapt_agent import Middleware

mw = Middleware(fail_closed=False)
mw.add_pre_middleware(lambda p: {**p}, name="tag_input")
mw.add_post_middleware(lambda p: {**p, "result": p["result"]}, name="stamp_output")
```

### TrustManager & TaintTracker

`TrustManager` tracks a numeric trust score per caller, useful alongside the
security controls:

```python
from adapt_agent.core.trust import TrustManager

trust = TrustManager()                         # initial_trust=0.5 by default
trust.update_trust_score("caller-42", +0.4, reason="verified API key")
trust.update_trust_score("caller-42", -0.5, reason="tried to extract a password")
print(trust.get_trust_score("caller-42"))      # clamped into [min_trust, max_trust]
```

`TaintTracker` (in `adapt_agent.core`) similarly marks untrusted data so you can
reason about provenance. Both are optional and independent of the adapter.

---

## 5. Optimization (the Train half)

ADAPT-Agent can tune a Pydantic AI agent against a golden dataset and apply the
winning configuration **in place**.

### What gets introspected for a Pydantic AI `Agent`

`detect(agent)` returns `"pydantic_ai"` and `introspect(agent)` walks the live
object and returns tunable `Parameter`s bound to working getters/setters:

| Discovered knob | `ParameterKind` | Source on the `Agent` |
|-----------------|-----------------|-----------------------|
| `agent.system_prompt` | `PROMPT` | the `_system_prompts` tuple (or a public `system_prompt`) |
| `agent.model` | `MODEL` | the `model` string, or `model.model_name` on a model object |
| `agent.temperature` | `HYPERPARAM` | `model.temperature` or `model_settings["temperature"]` |
| `agent.top_p` | `HYPERPARAM` | `model.top_p` or `model_settings["top_p"]` |
| `agent.max_tokens` | `HYPERPARAM` | `model.max_tokens` or `model_settings["max_tokens"]` |
| `agent.tools` | `TOOL` | `_function_tools` / `tools` (drop-one ablation when 2+ tools) |

```python
from adapt_agent.optimization.introspection import detect, introspect

print(detect(agent))                           # "pydantic_ai"
for p in introspect(agent):
    print(p.name, p.kind.value, repr(p.value))
```

> **Gotcha.** With an offline `FunctionModel`, the model's `model_name` is a
> read-only property, so the discovered `agent.model` knob has no working setter.
> ADAPT-Agent handles this gracefully: it logs a warning and marks that one knob
> **non-optimizable** rather than crashing the run. (Quiet it in a demo with
> `logging.getLogger("adapt_agent.optimization.parameters").setLevel(logging.ERROR)`.)

### Declaring extra knobs the framework doesn't expose

Anything not auto-discovered (a routing threshold, a model pool, an external
config value) can be declared explicitly with live `getter`/`setter`:

```python
from adapt_agent.optimization import Parameter, ParameterKind

target.add_parameter(Parameter(
    name="orchestrator.n_facts",
    kind=ParameterKind.ROUTING,
    value=ROUTING["n_facts"],
    bounds=(1, 3), step=1,
    getter=lambda: ROUTING["n_facts"],
    setter=lambda v: ROUTING.__setitem__("n_facts", int(v)),
    component="orchestrator",
))
```

Tool/skill *selection* is its own search space via `add_tool_parameter`, which
derives **drop-one ablation** subsets from a candidate pool:

```python
target.add_tool_parameter(
    "writer.tools",
    kind=ParameterKind.TOOL,
    getter=lambda: list(writer_tools["tools"]),
    setter=lambda v: writer_tools.__setitem__("tools", list(v)),
    candidate_tools=["summarize", "format_city"],
    component="writer",
)
```

### Evaluation and the judge

A `GoldenDataset` is a list of `{"input": ..., "expected": ...}` rows (also
`from_jsonl` / `from_json` / `from_csv`). An `EvaluationHarness` scores a target
with one or more metrics:

```python
from adapt_agent.optimization import (
    EvaluationHarness, GoldenDataset, LLMJudge, OptimizableAgent, exact_match,
)

data = GoldenDataset.from_list([
    {"input": "What is the capital of France?", "expected": "Paris"},
    {"input": "What is the capital of Japan?",  "expected": "Tokyo"},
])

judge = LLMJudge(my_completion_fn)             # any Callable[[str], str], offline-able
harness = EvaluationHarness(
    metrics=[exact_match(), judge.as_metric("quality")],
    primary_metric="quality",                  # which metric optimizers chase
    failure_threshold=1.0,                     # rows below this count as "failures"
)
```

The **`LLMJudge`** is provider-agnostic: it accepts a bare `Callable[[str], str]`,
a registered provider name, or a `ModelProvider`. It is used both to **score** and
to **rewrite prompts** from observed failures. For offline examples, back it with
a deterministic stub:

```python
def stub(prompt: str) -> str:
    if "CURRENT INSTRUCTION" in prompt or "FAILURES" in prompt:
        return "Answer with ONLY the capital city name, nothing else."   # a rewrite
    resp = prompt.split("<response>")[1].split("</response>")[0].strip() if "<response>" in prompt else ""
    score = 9 if resp and " " not in resp else 2                          # a grade
    return f'{{"score": {score}, "pass": {str(score >= 6).lower()}, "reasoning": "auto"}}'

judge = LLMJudge(stub)
```

For real runs, swap in `ClaudeJudge(model="claude-opus-4-8")`,
`OpenAIJudge(...)`, etc. Set **`adversarial=True`** to make the judge grade as a
harsh critic that hunts for missing requirements and unsafe behaviour -- a
reward-hack-resistant signal.

### The optimizers

- **`CoordinateAscentOptimizer(harness, judge=..., seed=...)`** -- tunes one knob
  at a time; with a judge it proposes prompt rewrites from failures. Great for a
  single agent.
- **`make_default_optimizer(harness, judge=..., max_evals=..., suggest_tools=...)`**
  -- the full pipeline: bootstrap few-shot -> prompt coordinate-ascent -> grid
  over models/hyperparameters/routing -> tool/skill coordinate-ascent with
  drop-one ablation and judge-driven new-tool **suggestions**.

```python
from adapt_agent.optimization import CoordinateAscentOptimizer, make_default_optimizer

target = OptimizableAgent.from_agent(agent, runner=lambda q: agent.run_sync(q).output)
result = CoordinateAscentOptimizer(harness, judge=judge, seed=0).optimize(target, data)

print(result.improvement, result.best_config)  # winning config, applied in place
for tip in result.recommendations:             # advisory new tools/skills (judge)
    print(tip)
```

`OptimizableAgent.from_agent` wants a `runner` that maps an input string to a
clean output string -- for Pydantic AI, `lambda q: agent.run_sync(q).output` (so
the harness compares text, not an `AgentRunResult`).

### YAML config path

The same run can be declared in a YAML file and executed with `run_training`:

```python
from adapt_agent.optimization.config import run_training
result = run_training("examples/pydantic_ai/pydantic_ai.train.yaml")
```

The schema (`target.entrypoint`/`components`, `dataset`, `judge`, `metrics`,
`optimizer`, `parameters[]`) is documented in
[`pydantic_ai.train.yaml`](../../examples/pydantic_ai/pydantic_ai.train.yaml).
Temperature bounds that exceed a provider's maximum are **clamped with a warning**
rather than crashing. Note that a YAML `parameter` binds to an *attribute* on a
resolved component (`component` + `attr_path`); knobs that live in a free-standing
dict must instead be declared in Python with explicit getters/setters.

See [`examples/pydantic_ai/03_evaluate_and_optimize.py`](../../examples/pydantic_ai/03_evaluate_and_optimize.py).

---

## 6. Multi-agent / orchestration with Pydantic AI

Pydantic AI has **no built-in orchestrator**. The framework documents several
patterns; the two you will reach for most are:

- **Agent delegation** -- one agent calls another *inside a tool*, forwarding
  usage so the parent's `result.usage` includes the child's:

  ```python
  @selector_agent.tool
  async def gather(ctx, count: int) -> list[str]:
      r = await worker_agent.run(f"produce {count} items", usage=ctx.usage)
      return r.output
  ```

- **Programmatic hand-off** -- a plain Python **orchestrator function** routes a
  query to several specialist `Agent`s and combines their results. This is the
  pattern ADAPT-Agent's example 4 uses (a `researcher` and a `writer`).

### Guard the whole system as one unit

Because the adapter only needs an object with `run_sync`, expose your
orchestrator function on a tiny shim and wrap that:

```python
class RunSyncShim:
    def __init__(self, fn): self._fn = fn
    def run_sync(self, prompt): return self._fn(prompt)

guarded = adapter.wrap_agent(RunSyncShim(orchestrator))
guarded.execute({"messages": [{"role": "user", "content": "capital of France?"}]})
```

Now the entire pipeline (route -> researcher -> writer) is screened and traced as
one governed call.

### Optimize the whole system *and* each agent

Use `from_components` to register every specialist by name. ADAPT-Agent
introspects each one and **namespaces** its knobs (`researcher.agent.system_prompt`,
`writer.agent.system_prompt`, ...), so you can optimize the whole system while
still seeing -- and tuning -- per-agent knobs:

```python
target = OptimizableAgent.from_components(
    components={"researcher": researcher, "writer": writer},
    runner=orchestrator,                       # drives the whole system
    name="research-team",
)
result = make_default_optimizer(harness, judge=judge, max_evals=40, seed=0).optimize(target, data)
```

The full example -- guard + whole-system/per-agent optimization + tool ablation +
recommendations + the YAML path -- is
[`examples/pydantic_ai/04_multi_agent_and_training.py`](../../examples/pydantic_ai/04_multi_agent_and_training.py).

> **Note on other frameworks' orchestrators.** Frameworks with a built-in
> orchestrator (e.g. Microsoft Agent Framework's *magentic* workflow, or OpenAI's
> handoffs) need their routing limits / manager declared as explicit `Parameter`s
> because the orchestrator object exposes no introspectable knobs. Pydantic AI's
> orchestrator is just your function, so you declare *its* knobs (like a fact
> count or a routing threshold) explicitly with `add_parameter` -- the
> specialists themselves are introspected automatically.

---

## 7. Common pitfalls / FAQ

- **`result.output` vs `result.data`.** Pydantic AI v2 uses `.output`; older code
  used `.data`. The introspector and examples use v2.
- **Negative indices in policy conditions.** The safe evaluator does not support
  unary operators. Use `state['messages'][0]` (first user message), not `[-1]`.
- **The output is wrapped.** `execute(...)` returns `{"result": <AgentRunResult>}`
  for non-dict framework results; read text off `out["result"].output`.
- **`agent.model` not optimizing.** A read-only `model_name` (as on
  `FunctionModel`) is detected, logged once, and skipped -- not a crash. With a
  real hosted model the knob is bindable.
- **No improvement without a judge or candidates.** Coordinate-ascent over a
  *prompt* knob needs either a judge (to propose rewrites from failures) or
  explicit `candidates`. The offline YAML example grids an explicit candidate
  list; the Python example uses a judge.
- **Offline runs.** Always available via `FunctionModel`/`TestModel` for the agent
  and a deterministic `Callable[[str], str]` for the judge -- no API key needed.
- **Async.** `run` (async) is also supported; the adapter drives Pydantic AI
  synchronously for you, so you can hand it either a sync or async agent.

---

## 8. The examples

A ladder from a tiny guarded agent to a governed, optimizable multi-agent system:

1. [`examples/pydantic_ai/01_basic_guarded.py`](../../examples/pydantic_ai/01_basic_guarded.py)
   -- smallest guarded `Agent`; safe input vs. a blocked injection.
2. [`examples/pydantic_ai/02_policy_observability_trust.py`](../../examples/pydantic_ai/02_policy_observability_trust.py)
   -- policy, defense, observer, middleware, trust; blocking vs. monitor-only.
3. [`examples/pydantic_ai/03_evaluate_and_optimize.py`](../../examples/pydantic_ai/03_evaluate_and_optimize.py)
   -- introspect, evaluate, and optimize one agent with a judge.
4. [`examples/pydantic_ai/04_multi_agent_and_training.py`](../../examples/pydantic_ai/04_multi_agent_and_training.py)
   -- a multi-agent orchestrator: guarded as one unit, optimized whole-system +
   per-agent with `make_default_optimizer`, plus a YAML
   [`pydantic_ai.train.yaml`](../../examples/pydantic_ai/pydantic_ai.train.yaml)
   path via `run_training`.
