# Pydantic AI

This page is a complete, from-scratch guide to using
[Pydantic AI](https://ai.pydantic.dev) with **ADAPT-Agent**. It assumes you are
new to *both*, so every concept and parameter is explained the first time it
appears.

ADAPT-Agent (**A**dversarial **D**efense & **P**olicy **T**raining for LLM
agents) has two independent, framework-agnostic halves:

- **Guard (runtime).** Two bindings that compose. A Pydantic AI **capability**
  (`governance_capability`, §3) governs the run from inside: every model
  request, every tool call and result (MCP tools included), approvals and
  streamed output. The **adapter** (`PydanticAIAdapter`, §4) wraps the run's
  boundary in a 6-step security and observability pipeline: you call
  `execute(...)` instead of `run_sync(...)`.
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
    "openai:gpt-5.6-luna",                 # "provider:model" identifier
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
swap to a real model is a single line: `Agent("openai:gpt-5.6-luna", ...)` or
`Agent("anthropic:claude-opus-5-5", ...)`.

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

## 3. Governing the run from inside: `governance_capability`

The adapter in §4 sees a run's **boundary**: the prompt going in, the answer
coming out. By the time the answer is screened every tool has already run, and
a tool result carrying an injected instruction has already been read by the
model. Pydantic AI (v2, verified against 2.76) exposes the run itself through
**capabilities** -- `pydantic_ai.capabilities.AbstractCapability` objects whose
hooks fire around every model request, tool call and output validation.
`governance_capability(...)` returns one that binds every ADAPT-Agent control
to those hooks. **This is the binding to use**; the adapter composes with it.

```python
from adapt_agent import AdversarialDefense, Firewall, PolicyEnforcer
from adapt_agent.core import ToolPolicy
from adapt_agent.integrations.pydantic_ai import governance_capability

governance = governance_capability(
    firewall=firewall,
    defense=AdversarialDefense(),
    policy_enforcer=policy,
    tool_policy=ToolPolicy(deny=["delete_*"], require_approval=["send_email"]),
    on_tool_violation="raise",          # or "refuse"
    agent_id="support-bot",
)

# Per agent, at construction ...
agent = Agent("openai:gpt-5.6-luna", deps_type=Deps, tools=[...], capabilities=[governance])
# ... or per run, for an agent you did not construct:
result = await other_agent.run(prompt, deps=deps, capabilities=[governance])
```

`governance_capability` takes the same keywords as every integration factory
(`gate`, `firewall`, `defense`, `policy_enforcer`, `block_on_violation`,
`agent_id`, `tool_policy`) plus `on_tool_violation`, `approvals`,
`buffer_streams`, `observer` and `policy_state`. The class itself is
`adapt_agent.integrations.pydantic_ai.GovernanceCapability` (built on first
access, so importing the module never imports `pydantic_ai`).

### What each hook does

| Pydantic AI hook | What ADAPT-Agent does there |
|---|---|
| `before_model_request` | Screens everything about to reach the model as **input** (firewall, adversarial defense, state policy): the user prompt, any `message_history`, every tool result, `ModelRetry` text and externally supplied deferred result. Each message part is screened once per run. |
| `before_tool_execute` | `gate.check_tool_call(name, args, state=...)` on the validated arguments, **before the tool function runs**: `ToolPolicy` lists, firewall/defense over the arguments, tool-scoped policy rules. |
| `after_tool_execute` | `gate.check_tool_result(...)` -- the indirect-injection seam. A blocked result never reaches the next model call. |
| `after_model_request` | Firewall over the model's text. |
| `after_output_validate` | Firewall over a structured `output_type`, field by field. |
| `get_wrapper_model` | With `buffer_streams=True`, holds a *streamed* response until it is screened (see *Streaming*). |
| `wrap_model_request` / `wrap_tool_execute` | One `AgentObserver` span per model request (`pydantic_ai.model_request`) and per tool call (`pydantic_ai.tool:<name>`). |

### Tool verdicts: allow, block, ask

**Block.** `on_tool_violation` picks what a blocked call or result does:

- `"raise"` (default) -- the run aborts with `SecurityBlockedError`; read
  `exc.threats` (`["tool:denied:delete_db"]`, `["firewall"]`,
  `["policy:<rule>"]`). Pydantic AI propagates an exception raised in a tool
  hook unchanged -- it is not converted to a tool failure, and parallel tool
  calls do not wrap it in an `ExceptionGroup` -- so the abort is real. It is the
  default because a refused call lets the model carry on and try another route
  to the same end, and the caller learns nothing unless it reads the transcript.
- `"refuse"` -- raises Pydantic AI's `ToolFailed` with a content-free reason
  (`Tool call 'delete_db' refused by ADAPT-Agent [support-bot]:
  tool:denied:delete_db.`), which the model receives as the tool's failed
  result. The tool does not run. For a blocked *result* the tool has run, but
  its output is discarded: the model sees only the reason.

**Ask.** A tool in `ToolPolicy(require_approval=[...])`, or a policy rule with
`action="require_approval"`, maps to Pydantic AI's **deferred approval**. The
hook raises `ApprovalRequired` *before* the tool runs, so the agent must be
able to return a `DeferredToolRequests`:

```python
from pydantic_ai import DeferredToolRequests, DeferredToolResults

agent = Agent(model, tools=[send_email], output_type=[str, DeferredToolRequests],
              capabilities=[governance])

paused = agent.run_sync("email bob", deps=deps)
requests = paused.output                         # DeferredToolRequests
call = requests.approvals[0]
print(requests.metadata[call.tool_call_id]["adapt_agent"])
# {'reason': "Tool call 'send_email' needs approval [support-bot]: ...", 'threats': [...], ...}

resumed = agent.run_sync(
    message_history=paused.all_messages(),
    deferred_tool_results=DeferredToolResults(approvals={call.tool_call_id: True}),  # or False
    deps=deps,
)
```

On resume the call is **screened again**: an approval satisfies the *ask* and
nothing else, so arguments that would be blocked (including ones the approver
substituted with `ToolApproved(override_args=...)`) are still blocked. A denial
(`False` or `ToolDenied(...)`) never executes the tool; the model is told it was
denied. A `HandleDeferredToolCalls` capability resolving approvals inline works
the same way.

If the run **cannot pause** -- `DeferredToolRequests` is not among the output
types and nothing resolves the call inline -- Pydantic AI aborts before the
tool runs, and the capability re-raises that as `SecurityBlockedError` with the
ask's threats. `approvals=False` turns every ask into a block up front
(`ToolDecision.resolve(approvals_supported=False)`), handled per
`on_tool_violation`.

**Report-only.** With `block_on_violation=False` every call runs and every
result passes; each decision that found something is logged to the `observer`
(`observer.get_logs()`, with `metadata["report_only"] = True` and the threats),
and policy violations are recorded on the enforcer as usual.

### Policy state: what a rule can rely on

A `PolicyEnforcer` is fail-closed by default -- a rule naming a key the state
lacks counts as **violated** -- so the state each hook builds is fixed:

| `deps` passed to the run | Keys a rule can read |
|---|---|
| a mapping | its string keys |
| a dataclass instance | its fields (read shallowly with `getattr`; no deep copy, so clients and connections in `deps` are fine) |
| a Pydantic model | its declared `model_fields` (shallow) |
| `None` or any other object | nothing -- a rule reading `state[...]` will block |

plus `state['messages']` -- the message parts being screened at
`before_model_request` (on the first step the prompt and any history, later
the new tool results), as `{"role": "user" | "tool", "content": ...}` dicts,
the same shape the adapter uses; **empty at the tool hooks**, so a tool-scoped
rule must not index it -- and `state['context']` (`{}` unless `deps` has one).
Tool-scoped rules -- conditions that name `tool` -- also see
`tool['name']` and `tool['arguments']`:

```python
@dataclass
class Deps:
    user: str
    trust_score: float

policy.add_rule("low_trust", "Low-trust sessions are refused",
                "state['trust_score'] < 0.5", action="block")
policy.add_rule("untrusted_email", "No email from low-trust sessions",
                "tool['name'] == 'send_email' and state['trust_score'] < 0.8",
                action="block")
```

State policy is re-evaluated at **every** model request, so a tool that lowers
`deps.trust_score` mid-run is seen by the next step. For any other state shape,
pass `policy_state=lambda ctx: {...}` -- it replaces the `deps` extraction
(`ctx` is Pydantic AI's `RunContext`). The helper
`adapt_agent.integrations.pydantic_ai.deps_state(deps)` shows exactly what a
given `deps` contributes.

### MCP tools

MCP servers attached with `capabilities.MCP(...)` or `toolsets=[MCPToolset(...)]`
surface through the **same** `before_tool_execute` / `after_tool_execute` hooks
as local tools (verified with an in-process MCP server), so a `ToolPolicy`
deny-list, argument screening and result screening apply to them unchanged.
Name patterns match the tool name as the model sees it (`delete_files`, or with
any prefix you configured on the toolset).

### Streaming

Under `run_stream`, `run_stream_events`, `iter` (with `node.stream()`) or an
`event_stream_handler`, model text is delivered **as it arrives** -- before
`after_model_request` can screen the complete response. Unbuffered, a blocked
answer is therefore partly delivered before the run aborts.

`buffer_streams=True` (the default) closes that gap: the capability wraps the
serving model so each streamed response is drained, screened (its text and the
arguments of any structured-output tool) and only then replayed, event for
event, to the consumer. A blocked response is **never delivered**; nothing is
duplicated or reordered for a clean one.

The trade-off is **latency**: the consumer receives each model response in one
burst when it completes, rather than token by token. Non-streamed runs
(`run` / `run_sync`) are unaffected. `buffer_streams=False` restores live
tokens and accepts that a blocked response may be partly shown first; the run
still aborts and the final output is never returned.

(Why not `wrap_run_event_stream`? Overriding it switches every `agent.run()` to
streamed requests, and holding events there leaves Pydantic AI's
`stream_text()` -- which also reads the accumulated raw response -- out of step
with the event stream, so held text is readable early and delivered twice.
Buffering at the model avoids both.)

### Sub-agents and run-level coverage

A capability given to `agent.run(capabilities=[...])` governs **that run**.
Pydantic AI's multi-agent delegation is a tool that calls another agent's
`run`; that inner run does **not** inherit the parent's capabilities. Give the
delegate the capability too -- at construction, or in the call:

```python
@parent.tool
async def ask_specialist(ctx: RunContext[Deps], question: str) -> str:
    r = await specialist.run(question, deps=ctx.deps, usage=ctx.usage,
                             capabilities=[governance])
    return r.output
```

Either way the parent's hooks screen the delegation call's arguments, and the
specialist's answer as a tool result.

### What is and is not covered

- **Covered:** user prompt, message history, tool results (local, MCP,
  delegated, externally supplied deferred results), tool arguments, tool names,
  model text, structured output, streamed output (buffered), policy on `deps`.
- **Not screened:** system prompts and instructions (developer-authored; they
  are not input), and the arguments of a tool call as the model *proposed* them
  in the message history (they are screened when the call executes).
- **Durable execution** (Temporal/DBOS/Prefect wrappers) is untested: those
  integrations restrict capabilities added per run.

### The older seam: `install_governance` / `governance_output_validator`

These register an `@agent.output_validator` -- the final output only. They
remain for existing code. Pydantic AI has no public way to add a capability to
an agent that already exists (`Agent.override(...)` is a scoped context
manager), so `install_governance(agent, ...)` stays output-only; pass the
capability per run instead. Both refuse `policy_enforcer`, `defense` and
`tool_policy` with a `ValueError` naming `governance_capability`, because an
output validator never sees state, input or tool calls.

See [`examples/pydantic_ai/05_tool_governance.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/pydantic_ai/05_tool_governance.py).

---

## 4. Guarding the run's boundary: `PydanticAIAdapter`

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
- **`policy_enforcer`** (`PolicyEnforcer`) -- rule engine, see §5.
- **`observer`** (`AgentObserver`) -- records a trace per run; read them back with
  `observer.get_traces()`.
- **`middleware`** (`Middleware`) -- pre/post hooks, see §5.
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
- **Run arguments:** any other keyword argument is forwarded to the run method
  (`run_sync` for `execute`, `run` for `aexecute`): `deps=`,
  `message_history=`, `model_settings=`, `usage_limits=`,
  `deferred_tool_results=`, `capabilities=`, ...

  ```python
  out = guarded.execute(payload, deps=Deps(user="ada", trust_score=0.9))
  out = await guarded.aexecute(payload, deps=deps, capabilities=[governance])
  ```

  `message_history` is **content**, not configuration -- it is the conversation
  the model reads next -- so the adapter screens it as input before the run.
  `deferred_tool_results` is screened the same way: it carries tool output
  produced outside the run, and approvals that may replace a call's arguments. A
  payload with no prompt of its own (`{}`) plus a `message_history` runs with
  `user_prompt=None`, which is how Pydantic AI continues a conversation (for
  example to resume deferred tool calls), instead of sending the payload's
  `repr` as a new user turn.

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

See [`examples/pydantic_ai/01_basic_guarded.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/pydantic_ai/01_basic_guarded.py).

---

## 5. Policy, adversarial defense, observability, trust & taint

These compose on the same adapter. The full example is
[`examples/pydantic_ai/02_policy_observability_trust.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/pydantic_ai/02_policy_observability_trust.py).

### PolicyEnforcer

A `PolicyEnforcer` holds rules evaluated by a **safe expression language** (no
`eval`). Through the adapter, rules are checked against the **state**, so
conditions reference the `state` variable:

```python
from adapt_agent import PolicyEnforcer

policy = PolicyEnforcer()                      # fail-closed by default: an un-evaluable
policy.add_rule(                               # condition counts as a violation
    name="no_password_requests",
    description="Refuse requests that try to extract passwords.",
    condition="'password' in state['messages'][0]['content']",
    action="block",                            # "block" aborts; "warn" only records
    severity="high",
)
adapter = PydanticAIAdapter(policy_enforcer=policy, block_on_violation=True)
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

## 6. Optimization (the Train half)

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

For real runs, swap in `ClaudeJudge(model="claude-opus-5-5")`,
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
[`pydantic_ai.train.yaml`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/pydantic_ai/pydantic_ai.train.yaml).
Temperature bounds that exceed a provider's maximum are **clamped with a warning**
rather than crashing. Note that a YAML `parameter` binds to an *attribute* on a
resolved component (`component` + `attr_path`); knobs that live in a free-standing
dict must instead be declared in Python with explicit getters/setters.

See [`examples/pydantic_ai/03_evaluate_and_optimize.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/pydantic_ai/03_evaluate_and_optimize.py).

---

## 7. Multi-agent / orchestration with Pydantic AI

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
one governed call -- at its boundary. To govern each specialist's tool calls
and model requests too, give each one the capability (§3); for delegation
inside a tool, pass it to the inner `run(..., capabilities=[governance])`,
since a delegate's run does not inherit the parent's capabilities.

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
[`examples/pydantic_ai/04_multi_agent_and_training.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/pydantic_ai/04_multi_agent_and_training.py).

> **Note on other frameworks' orchestrators.** Frameworks with a built-in
> orchestrator (e.g. Microsoft Agent Framework's *magentic* workflow, or OpenAI's
> handoffs) need their routing limits / manager declared as explicit `Parameter`s
> because the orchestrator object exposes no introspectable knobs. Pydantic AI's
> orchestrator is just your function, so you declare *its* knobs (like a fact
> count or a routing threshold) explicitly with `add_parameter` -- the
> specialists themselves are introspected automatically.

---

## 8. Common pitfalls / FAQ

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
- **Async.** `execute` drives `run_sync`; `await guarded.aexecute(...)` awaits
  `run` in your own event loop. Calling `execute` from inside a running loop
  raises -- use `aexecute` there.
- **An "ask" raised `SecurityBlockedError`, not a pause.** The agent cannot
  return `DeferredToolRequests`: add it to `output_type` (or use a
  `HandleDeferredToolCalls` capability). Until then the call fails closed.
- **`run_stream` output arrives all at once.** That is `buffer_streams=True`
  holding each response until it is screened; see §3 *Streaming*.
- **`install_governance` did not stop a tool.** It is output-only. Use
  `governance_capability` (§3).

---

## 9. The examples

A ladder from a tiny guarded agent to a governed, optimizable multi-agent system:

1. [`examples/pydantic_ai/01_basic_guarded.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/pydantic_ai/01_basic_guarded.py)
   -- smallest guarded `Agent`; safe input vs. a blocked injection.
2. [`examples/pydantic_ai/02_policy_observability_trust.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/pydantic_ai/02_policy_observability_trust.py)
   -- policy, defense, observer, middleware, trust; blocking vs. monitor-only.
3. [`examples/pydantic_ai/03_evaluate_and_optimize.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/pydantic_ai/03_evaluate_and_optimize.py)
   -- introspect, evaluate, and optimize one agent with a judge.
4. [`examples/pydantic_ai/04_multi_agent_and_training.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/pydantic_ai/04_multi_agent_and_training.py)
   -- a multi-agent orchestrator: guarded as one unit, optimized whole-system +
   per-agent with `make_default_optimizer`, plus a YAML
   [`pydantic_ai.train.yaml`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/pydantic_ai/pydantic_ai.train.yaml)
   path via `run_training`.
5. [`examples/pydantic_ai/05_tool_governance.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/pydantic_ai/05_tool_governance.py)
   -- `governance_capability`: denied tools (raise / refuse), a poisoned tool
   result, the approval round trip, policy on `deps`, buffered vs. unbuffered
   streaming, and an MCP tool.
