# Google ADK

This page covers using **ADAPT-Agent** with **Google's Agent Development Kit
(ADK)**, checked against `google-adk` 2.15. It assumes you are new to both.

ADAPT-Agent has two independent halves, both safe to import without the
framework installed:

- **Guard (runtime):** a governance **plugin** that ADK calls for every model
  call and every tool call of every agent in a run, plus an adapter that screens
  the boundary of the run.
- **Train (offline):** turn an agent or multi-agent system into a tunable search
  space and optimize it against a golden dataset, scored by an LLM-as-judge.

Nothing here calls a live model unless you choose to. Every example runs offline
with a scripted model stand-in (no API key, no network).

---

## 1. Quick start: one plugin for the whole run

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import InMemoryRunner

from adapt_agent import Firewall
from adapt_agent.core import ToolPolicy
from adapt_agent.integrations.google_adk import governance_plugin

fw = Firewall()
fw.add_blocked_pattern(r"(?i)ignore (all|previous) instructions")

app = App(
    name="support",
    root_agent=router,          # sub-agents, AgentTools and their tools are covered too
    plugins=[governance_plugin(
        firewall=fw,
        tool_policy=ToolPolicy(deny=["shell"], require_approval=["refund"]),
    )],
)
runner = InMemoryRunner(app=app)
```

That is the whole binding. ADK runs a plugin's callbacks for **every** agent in
the tree -- sub-agents reached by `transfer_to_agent`, agents wrapped in an
`AgentTool`, the nodes of a `Workflow` -- and for every tool, MCP tools
included. Register it on the `App`; `Runner(plugins=...)` still works but is
deprecated upstream.

What it governs:

| Seam | ADK callback | What happens |
| --- | --- | --- |
| User message | `on_user_message_callback` | Screened before it enters the session. `on_block="raise"` aborts the run; `"refuse"` stores a placeholder in its place (so it cannot poison later turns) and answers with the refusal. |
| Model request | `before_model_callback` | The full `llm_request.contents` the model is about to see, tool results included. |
| Model response | `after_model_callback` | The response, before ADK acts on it -- a function call in a blocked response is never executed. |
| Tool call | `before_tool_callback` | `gate.check_tool_call(tool.name, args)`: the `ToolPolicy`, firewall/defense over the arguments, tool-scoped policy rules. |
| Tool result | `after_tool_callback` | `gate.check_tool_result(...)`, before the result reaches the next model call. |

### Blocking modes

Two options, one for model traffic and one for tools, each `"raise"` or
`"refuse"`:

- **`on_block`** (user message, model request, model response) -- default
  `"raise"`: the run aborts with `SecurityBlockedError`. `"refuse"` returns a
  refusal `LlmResponse` (`refusal_text`, with `custom_metadata["adapt_agent"]`
  naming the agent and threats), so one blocked branch answers with the refusal
  and the rest of the tree keeps running.
- **`on_tool_violation`** (tool call, tool result) -- default `"refuse"`: the
  tool **does not run**, and the model receives `{"error": decision.reason}` --
  a one-line message naming the tool and the threat labels, never the content.
  The run continues, and the model can explain or try something else. That is
  ADK's own short-circuit for a before-tool callback, and a model probing a tool
  it should not use is routine enough that aborting the whole run is usually
  the wrong response. A blocked **result** is replaced the same way, so the
  poisoned content never reaches the next model call. `"raise"` aborts.

### Getting the `SecurityBlockedError` back

ADK re-raises any exception from a plugin callback as
`RuntimeError("Error in plugin ...")`, with the original as `__cause__` -- so
`except SecurityBlockedError` around a bare `Runner` catches nothing. Two
helpers:

```python
from adapt_agent.integrations.google_adk import find_security_error, reraise_security_blocks

with reraise_security_blocks():           # works around `for` and `async for`
    async for event in runner.run_async(user_id="u", session_id="s", new_message=msg):
        ...

# or, in an existing handler:
except RuntimeError as exc:
    blocked = find_security_error(exc)    # the SecurityBlockedError, or None
```

`GoogleADKAdapter` (section 4) unwraps it for you.

### Per-agent callbacks

`governance_callbacks(...)` returns the same hooks for one `LlmAgent`:
`before_model_callback`, `after_model_callback`, `before_tool_callback`,
`after_tool_callback`. Use them **in addition to** the plugin when one agent
needs stricter controls -- the specialist reading untrusted mail -- or where
you cannot touch the `App`:

```python
from adapt_agent.integrations.google_adk import governance_callbacks

reader = LlmAgent(
    name="mail_reader", model="gemini-flash-latest", tools=[read_inbox],
    **governance_callbacks(firewall=strict_fw, agent_id="mail_reader", on_block="refuse"),
)
```

Attaching per agent *instead of* the plugin leaves every agent you forgot
ungoverned. The options (`on_block`, `on_tool_violation`, `approvals`,
`buffer_stream`, `stream_holdback`, `observer`, `tool_policy`) mean the same in
both factories. `govern_tools=False` omits the two tool callbacks.

---

## 2. Tool governance, approvals and MCP

### Tool policy and tool-scoped rules

```python
from adapt_agent.core import PolicyEnforcer, ToolPolicy

tool_policy = ToolPolicy(
    allow=None,                      # or an allow-list of globs
    deny=["shell", "mcp_admin_*"],   # never callable
    require_approval=["refund"],     # every call pauses for a human
)

enforcer = PolicyEnforcer()
enforcer.add_rule(
    name="no_wire_when_untrusted",
    description="wire transfers need a trusted session",
    condition="tool['name'] == 'wire' and state['trust_score'] < 0.8",
    action="block",                  # or "require_approval"
)
plugin = governance_plugin(tool_policy=tool_policy, policy_enforcer=enforcer, firewall=fw)
```

A rule's `state` is the ADK **session state** (`tool_context.state`, including
values written earlier in the same invocation). Seed it with
`session_service.create_session(..., state={"trust_score": 0.9})` or write it
from a tool or callback. A fail-closed enforcer (the default) treats a key that
is missing as a violation.

### Approvals ("ask")

An "ask" -- a `require_approval` match or rule -- uses ADK's tool-confirmation
flow, with no setup on the tool:

1. The plugin calls `tool_context.request_confirmation(hint=decision.reason)`.
   The tool does not run; the run ends with an `adk_request_confirmation`
   function call for your client to show a human.
2. The client answers with that call's id as the next message:

   ```python
   answer = types.Content(role="user", parts=[types.Part(
       function_response=types.FunctionResponse(
           id=request.id, name="adk_request_confirmation",
           response={"confirmed": True},          # or False
       ))])
   runner.run_async(user_id="u", session_id="s", new_message=answer)
   ```
3. ADK re-runs the original call. Approved, it runs (and its result is still
   screened); rejected, the model is told it was rejected.

A human can approve an ask, never a block: a call whose arguments trip the
firewall is refused even with a confirmation. The flow fails closed -- the ask
becomes a block -- with `approvals=False`, in a live (bidi-streaming) run, and
where the context has no function-call id. Inside an `AgentTool` the nested
session is discarded, so ADK answers the outer call with an error: the tool
still does not run.

Without the plugin, derive ADK's own static gate from the gate:

```python
from google.adk.tools import FunctionTool
from adapt_agent.integrations.google_adk import gate_confirmation

tool = FunctionTool(refund, require_confirmation=gate_confirmation(gate, "refund"))
```

It asks for anything the gate would not plainly allow (it cannot refuse, so a
blocked call needs a human too). With the plugin installed it is redundant, and
combining the two still asks once.

### MCP tools

An `McpToolset`'s tools are ordinary `BaseTool`s by the time they run, so their
calls and results pass through the plugin's tool callbacks like any other
(verified against a real stdio MCP server in `tests/test_real_google_adk.py`).
To keep a denied MCP tool from even being offered to the model:

```python
from google.adk.tools.mcp_tool import McpToolset
from adapt_agent.integrations.google_adk import mcp_toolset_kwargs

toolset = McpToolset(
    connection_params=params,
    **mcp_toolset_kwargs(tool_policy=tool_policy, tool_name_prefix="gh"),
)
```

The filter matches the name the plugin sees (`"gh_<tool>"` with a prefix), so one
`ToolPolicy` serves both. It does not derive `require_confirmation`: ADK gives a
toolset's confirmation callable the arguments but not the tool name, so a
name-based policy cannot be evaluated there -- approvals for MCP tools come from
the plugin's flow above.

### Built-in ADK controls that compose with this

- `ToolCallIntegrityPlugin(secret_key)` stamps each stored function call with
  an HMAC and refuses one modified in the session store before ADK re-runs it --
  which is what happens to a call on resume after its confirmation. Register it
  alongside when the session store is not fully trusted.
- ADK emits OpenTelemetry spans (`google.adk.telemetry`, `RunConfig.telemetry`);
  ADAPT-Agent's `observer=` records its own span per governed call.

---

## 3. Streaming

Under `RunConfig(streaming_mode=StreamingMode.SSE)` ADK yields every partial
chunk to the caller as it arrives and runs `after_model_callback` on each one
separately. Without buffering, therefore:

- text before a blocked chunk has **already been delivered** when the block
  fires, and
- a pattern **split across chunks** (`"the SEC" + "RET-TOKEN"`) is invisible to
  every per-chunk check; only the final aggregate response catches it, after
  every piece was shown.

`buffer_stream=True` fixes both. The after-model hook keeps a buffer per model
call, screens each partial as part of the aggregate so far, and rewrites the
partial event's content to what is safe to release:

| Option | Delivered to the caller | Latency |
| --- | --- | --- |
| `buffer_stream=False` (default) | every chunk, unscreened as a whole | none |
| `buffer_stream=True` | empty partials; the final event carries the screened answer | the whole response |
| `buffer_stream=True, stream_holdback=N` | everything but the last `N` characters, as it passes | `N` characters |

With a holdback, a match longer than `N` characters can have its beginning
released before it completes: set `N` to at least the longest string your rules
must catch. Because the hook rewrites the events the `Runner` yields, buffering
works for a bare `Runner` as well as through the adapter. Live (bidi) runs are
not buffered.

---

## 4. The adapter: governing the boundary

`GoogleADKAdapter` wraps a run from the outside and applies the six-step
pipeline (input screening, policy, pre-middleware, traced run, post-middleware,
output screening) to the payload going in and the events coming out. It accepts
three wrap targets:

```python
from adapt_agent.adapters import GoogleADKAdapter

adapter = GoogleADKAdapter(firewall=fw, agent_id="support")

# 1. A Runner (or an App, run on an InMemoryRunner the adapter keeps for it):
guarded = adapter.wrap_agent(runner)
out = guarded.execute({"prompt": "hi"}, user_id="u", session_id="chat-1")
# session created on first use; omit session_id for a fresh session per call;
# run_config=, state_delta=, and a ready-made new_message= pass through.

# 2. A callable you write -- the full payload and every execute() kwarg reach it:
def run(payload, *, session_id):
    msg = types.Content(role="user", parts=[types.Part(text=payload["prompt"])])
    return runner.run_async(user_id="u", session_id=session_id, new_message=msg)

guarded = adapter.wrap_agent(run)
guarded.execute({"prompt": "hi"}, session_id="chat-1")
```

`execute` returns `{"result": [<events>]}`; `aexecute` is the async twin and
drives `run_async` in the caller's loop. A block raised by the plugin inside the
run reaches you as `SecurityBlockedError`, threats intact.

The adapter's own screening sees only the boundary. Register the plugin on the
`App` for tool calls, sub-agents and streaming.

Constructor arguments, all keyword-only and optional: `firewall`, `defense`,
`policy_enforcer`, `observer`, `middleware`, `agent_id`, `block_on_violation`
(`False` records instead of raising). Adapter policy rules see
`state['messages']` and `state['context']`.

---

## 5. Policy, adversarial defense, observability, trust, taint

### PolicyEnforcer

Rules use a **safe expression language** (no `eval`): comparisons, `in`, boolean
ops and subscripting, no calls or comprehensions. What a rule's `state` holds
depends on where it is evaluated:

| Hook | `state` keys |
| --- | --- |
| model request (plugin/callbacks) | `messages` (the request `contents`), `context` (`{}`), plus every ADK session-state key; the screened `messages` win over a session key of the same name |
| user message (plugin) | the same, with `messages = [user_message]` |
| tool call | the ADK session state as-is, plus `tool['name']` / `tool['arguments']` for tool-scoped rules |
| adapter | `messages`, `context` from the payload |

A rule that names `tool` is evaluated only at the tool seam. A condition that
cannot be evaluated counts as a violation by default;
`PolicyEnforcer(fail_closed=False)` opts out.

### AdversarialDefense

`AdversarialDefense()` runs heuristic detectors (prompt injection, jailbreak
phrasing) over inputs: the user message, model requests, tool arguments and tool
results.

### AgentObserver

Pass `observer=AgentObserver()` to the plugin or callbacks for one span per
screened call -- operations `google_adk.user_message`, `google_adk.model_request`,
`google_adk.model_response`, `google_adk.tool_call`, `google_adk.tool_result`,
each attributed to `"<agent_id>/<adk agent name>"` and closed as `error` on a
block. The adapter records one `google_adk.run` span. The plugin also keeps its
recent tool decisions on `plugin.decisions` (report-only ones included):

```python
plugin = governance_plugin(tool_policy=policy, block_on_violation=False)
...
[d for d in plugin.decisions if d.report_only]   # what *would* have been refused
```

### TrustManager and TaintTracker (standalone)

```python
from adapt_agent import TaintTracker, TrustManager

trust = TrustManager()
trust.update_trust_score("support", -0.4, reason="emitted a blocked phrase")
trust.is_trusted("support")                # -> False

taint = TaintTracker()
taint.register_source("user_input", source_type="external")
taint.mark_tainted("question_1", source_ids=["user_input"])
taint.propagate_taint("question_1", "answer_1", operation="llm_answer")
taint.is_tainted("answer_1")               # -> True
```

See [`examples/google_adk/02_policy_observability_trust.py`](../../examples/google_adk/02_policy_observability_trust.py).

---

## 6. ADK essentials, for the newcomer

ADK's agent class is `LlmAgent` (aliased `Agent`):

```python
agent = LlmAgent(
    name="capital_agent",            # unique id; used for routing between agents
    model="gemini-flash-latest",     # a model id string, or a BaseLlm object
    description="Answers questions about country capitals.",
    instruction="Reply with only the capital city name.",
    tools=[lookup],                  # callables or BaseTools (FunctionTool, McpToolset, AgentTool)
    sub_agents=[...],                # children reachable by transfer_to_agent
)
```

Leaving `model` out uses ADK's default (`gemini-3.5-flash` in 2.15).
`global_instruction` is deprecated in favour of `GlobalInstructionPlugin`, and
`SequentialAgent` / `ParallelAgent` / `LoopAgent` in favour of
`google.adk.workflow.Workflow` -- the plugin covers both old and new forms.

A run goes through a `Runner` bound to a session service; it yields `Event`s,
and the answer is the one where `event.is_final_response()` is true:

```python
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.genai import types

runner = InMemoryRunner(app=App(name="my_app", root_agent=agent, plugins=[plugin]))
await runner.session_service.create_session(app_name="my_app", user_id="u", session_id="s")
msg = types.Content(role="user", parts=[types.Part(text="What is the capital of France?")])
async for event in runner.run_async(user_id="u", session_id="s", new_message=msg):
    if event.is_final_response():
        print(event.content.parts[0].text)
```

`runner.run(...)` is the synchronous form. Install the extra with
`pip install 'adapt-agent[google-adk]'`; importing any ADAPT-Agent module works
without it, and ADK is imported only when you build the plugin or an ADK type is
needed.

---

## 7. What is and is not covered

Covered by the plugin: every `LlmAgent` model call in the run (sub-agents,
`AgentTool`s, `Workflow` nodes); every tool call and result that goes through
ADK's tool pipeline (function tools, `AgentTool`, `McpToolset`, `transfer_to_agent`,
`Workflow` tool nodes); the user message; streamed output with `buffer_stream`.

Not covered, or covered with limits:

- **Live / bidi streaming** (`run_live`): tool calls go through the same
  callbacks (per the installed source; not exercised by the test suite), but
  partial audio/text is not buffered, and an approval cannot pause the call
  (asks fail closed to blocks).
- **Unbuffered SSE streaming** delivers chunks before the aggregate is screened
  (section 3).
- **An `AgentTool` with `include_plugins=False`** runs its agent without the
  parent's plugins; attach `governance_callbacks` to that agent, or leave the
  default.
- **Function calls already in the history** are not re-screened as *input* on
  later model calls: they were the model's own output, screened when produced
  and again at the tool seam. (With `governance_callbacks(govern_tools=False)`
  they still are.)
- **Non-text content** (images, audio, file data) is not screened.

---

## 8. Optimization ("training")

The second half of ADAPT-Agent turns an agent into a tunable search space, scores
it over a golden dataset, and searches for a better configuration - then applies
the winner back onto the live agent.

### What gets introspected for a Google ADK agent

`from adapt_agent.optimization.introspection import detect, introspect`
recognises an ADK agent and lists its tunable
`Parameter`s. `detect(agent)` returns `"google_adk"`; `introspect(agent)` returns:

| Attribute | `ParameterKind` | Notes |
|---|---|---|
| `instruction`, `global_instruction` | `PROMPT` | only when plain strings (instruction-provider callables are skipped) |
| `model` | `MODEL` | a string id, or a model object's `model` / `model_name` |
| `generate_content_config.temperature` | `HYPERPARAM` | bounds `(0.0, 2.0)` |
| `generate_content_config.top_p` | `HYPERPARAM` | bounds `(0.0, 1.0)` |
| `generate_content_config.max_output_tokens` | `HYPERPARAM` | bounds `(1, 32000)` |
| `tools` | `TOOL` | drop-one ablation candidates when 2+ tools |
| `sub_agents` | `ROUTING` | walked **recursively**; nested knobs namespaced under the parent (`coordinator.geo_agent.instruction`) |

Cycles and shared/diamond sub-agents are handled (an `id()` visited set), and
duplicate parameter names are de-duplicated.

### Declaring extra knobs the framework doesn't expose

For anything not auto-discovered (a higher-level "skill" allow-list, a custom
routing threshold, ...), declare a `Parameter` with a `getter`/`setter` bound to
the live object:

```python
from adapt_agent.optimization import Parameter, ParameterKind

target.add_parameter(Parameter(
    name="coordinator.route_threshold",
    kind=ParameterKind.ROUTING,
    bounds=(0.2, 0.9), step=0.1,
    getter=lambda: coordinator.route_threshold,
    setter=lambda v: setattr(coordinator, "route_threshold", v),
    component="coordinator",
))
```

For tool/skill *selection* there's a one-call convenience that derives drop-one
ablation subsets from a candidate pool:

```python
target.add_tool_parameter(
    "math_agent.skills",
    kind=ParameterKind.SKILL,                 # or ParameterKind.TOOL
    getter=lambda: math_agent.skills,
    setter=lambda s: setattr(math_agent, "skills", list(s)),
    candidate_tools=["arithmetic", "word_problems"],
    component="math_agent",
)
```

### Wrapping the agent

```python
from adapt_agent.optimization import OptimizableAgent

# Single agent (driven through a Runner, so pass an explicit runner):
target = OptimizableAgent.from_agent(agent, runner=run, name="capital-agent")

# Whole system (register every live agent so its knobs are tunable):
target = OptimizableAgent.from_components(
    components={"coordinator": coordinator,
                "math_agent": math_agent,
                "geo_agent": geo_agent},
    runner=run,                               # one callable that drives the tree
    name="adk-team",
)
```

`runner` is a callable `input -> output` that drives the system end to end (build
a `Runner` and drive `run_async`, as in example 1). `components` are the *live* objects
whose knobs are tuned, so applying a candidate config mutates the next run.

### Evaluation

```python
from adapt_agent.optimization import EvaluationHarness, GoldenDataset, exact_match

data = GoldenDataset.from_list([
    {"input": "What is the capital of France?", "expected": "Paris"},
    # ... also GoldenDataset.from_jsonl / from_json / from_csv
])
harness = EvaluationHarness(
    metrics=[exact_match(), judge.as_metric("quality")],
    primary_metric="exact_match",     # the headline score
    failure_threshold=1.0,            # examples below this on the primary metric
)                                     #   are "failures" the proposers learn from
report = harness.evaluate(target, data)
print(report.aggregate)               # {"exact_match": 0.5, "quality": 0.55}
```

Built-in metrics include `exact_match()` and `token_f1()`.

### The judge (including adversarial mode)

An `LLMJudge` wraps any completion callable and is used both as a *scoring metric*
and to *rewrite prompts / propose tools* from failures. It is provider-agnostic;
back it with a deterministic stub to run offline:

```python
from adapt_agent.optimization import LLMJudge

def stub(prompt, system=None):
    # The judge puts rubrics/rewrite requests in `system` and data in `prompt`.
    if "prompt engineer" in (system or ""):
        return "Answer with ONLY the capital city name, nothing else."
    return '{"score": 9, "pass": true, "reasoning": "auto"}'

judge = LLMJudge(stub)
```

For real use, swap in `ClaudeJudge(model="claude-opus-4-8")`, `OpenAIJudge(...)`,
or `GeminiJudge(...)`. Set `adversarial=True` to grade like a harsh critic
(resistant to reward-hacking) and to surface tougher tool/skill suggestions.

> **Provider note.** Anthropic Opus 4.8/4.7 and Fable 5 reject sampling
> parameters; the providers omit or clamp them automatically. Temperature bounds
> declared in optimization are clamped to the provider max with a warning rather
> than crashing.

### The optimizers

```python
from adapt_agent.optimization import CoordinateAscentOptimizer, make_default_optimizer

# Greedy per-parameter search; the flagship for prompt/few-shot tuning.
result = CoordinateAscentOptimizer(harness, judge=judge, seed=0).optimize(target, data)

# The full pipeline: bootstrap few-shot -> prompts (judge-driven) ->
# models/hyperparameters/routing -> tools/skills (drop-one ablation + judge
# tool suggestions). Budget is split across the four stages.
result = make_default_optimizer(
    harness, judge=judge, max_evals=60, seed=0, suggest_tools=True,
).optimize(target, data)

print(result.improvement, result.best_config)
for tip in result.recommendations:    # advisory NEW tools/skills from the judge
    print(tip)
```

`make_default_optimizer` arguments: `judge` (enables judge-driven proposals),
`max_evals` (total evaluation budget, split across stages), `seed`
(reproducibility), `min_improvement` (reject candidates that only beat the
baseline by judge noise), and `suggest_tools` (let the judge propose brand-new
tools/skills, on by default when a judge is present).

> **Budget tip.** A multi-agent tree exposes many knobs. `make_default_optimizer`
> runs prompts in their own stage, so a winning prompt rewrite is reached even
> with many hyperparameters present - but give it enough `max_evals` (60 is a good
> default) so each stage can sweep all of its parameters.

The winning configuration is applied **in place** on the live components, so the
very next `run` reflects it. `result.recommendations` are *advisory only* - they
describe tools/skills the judge thinks would help; nothing is auto-installed.

### The YAML config path

The same run can be declared in a YAML file and executed with `run_training`:

```python
from adapt_agent.optimization.config import run_training
result = run_training("examples/google_adk/google_adk.train.yaml")
```

The schema (see [`google_adk.train.yaml`](../../examples/google_adk/google_adk.train.yaml)):

- **`target.entrypoint`** - `"module:attribute"` of a callable `input -> output`
  that drives the whole system.
- **`target.components`** - named `"module:attribute"` references to the live
  agents to introspect (the same objects the entrypoint runs).
- **`dataset`** - `path` to `.jsonl`/`.json`/`.csv` (+ optional `input_key` /
  `expected_key` overrides).
- **`judge`** - `provider`, `model`, `adversarial`, `scale`, `pass_threshold`,
  `criteria`, `metric_name`.
- **`metrics`** / **`primary_metric`**.
- **`optimizer`** - `type` (`default` | `coordinate_ascent` | `grid` | `random` |
  `evolutionary` | `bootstrap_few_shot`), `max_evals`, `min_improvement`, `seed`,
  `suggest_tools`.
- **`parameters`** - explicit knobs the framework doesn't auto-expose, each bound
  live to a component attribute via `attr` / `attr_path`.

See [`examples/google_adk/03_evaluate_and_optimize.py`](../../examples/google_adk/03_evaluate_and_optimize.py)
and [`examples/google_adk/04_multi_agent_and_training.py`](../../examples/google_adk/04_multi_agent_and_training.py).

---

## 9. Multi-agent systems

A realistic ADK system is a **coordinator** `LlmAgent` with specialist
`sub_agents`, or a `Workflow` of agents. Govern it with **one plugin on the
`App`**: every agent and tool in the tree is covered, and errors and spans name
the agent (`"<agent_id>/<adk agent name>"`). Add `governance_callbacks` to the
one agent that needs stricter rules. The adapter around the runner adds boundary
screening on top; on its own it sees only the text of the run's events, never a
tool call.

### Optimizing the whole system vs. each agent individually

ADAPT-Agent's introspection walks `sub_agents` **recursively**, so a single
`introspect(coordinator)` discovers every agent's knobs, namespaced under the
parent (`coordinator.math_agent.instruction`, `coordinator.geo_agent.temperature`,
...). You then have two complementary moves:

- **Whole system:** `OptimizableAgent.from_components({...all agents...}, runner=run)`
  and let `make_default_optimizer` tune the tree end to end against the dataset.
- **One agent at a time:** wrap a single specialist with `from_agent` and optimize
  it in isolation (useful when one agent's prompt dominates the failures).

When an agent carries 2+ tools, introspection exposes a `TOOL` parameter whose
candidates are drop-one ablation subsets. When a failure cannot be fixed by
tuning existing knobs, the **adversarial judge** flags the gap on
`result.recommendations` -- advisory only; nothing is auto-installed.

---

## 10. Common pitfalls / FAQ

**"I catch `SecurityBlockedError` but get `RuntimeError: Error in plugin ...`."**
ADK wraps plugin exceptions. Use `reraise_security_blocks()` /
`find_security_error()`, or drive the run through `GoogleADKAdapter`.

**"A denied tool did not stop the run."**
That is `on_tool_violation="refuse"` (the default): the tool did not run, and the
model was told. Pass `on_tool_violation="raise"` to abort instead.

**"My run stopped at the model response, not at the tool."**
With output screening on, a firewall hit in a function call's arguments fails the
model response first (under `on_block`), before the call reaches the tool seam.
The tool seam is what applies the `ToolPolicy`, tool-scoped rules, and result
screening.

**"Every request is blocked by policy."**
A fail-closed enforcer treats a rule it cannot evaluate as violated. Check that
the session state carries every key your rules read (section 5).

**"An approval never resumes."**
Answer with a `FunctionResponse` whose `name` is `adk_request_confirmation` and
whose `id` is the id of the *confirmation* call (not the original tool call), in
the same session.

**"`introspect()` returned an empty list / my `instruction` isn't tunable."**
Introspection recognises an ADK agent by having both `instruction` and
`sub_agents`, and exposes only *string* instructions. Declare a custom
`Parameter` otherwise.

---

## 11. The examples

All run offline with the framework installed, and exit 0 with a skip message
without it.

- [`01_basic_guarded.py`](../../examples/google_adk/01_basic_guarded.py)
  - the plugin on an `App`, the adapter around the `Runner`; a safe call and a
    blocked prompt injection.
- [`02_policy_observability_trust.py`](../../examples/google_adk/02_policy_observability_trust.py)
  - policy, defense, observability, middleware, trust, taint; `block_on_violation=False`.
- [`03_evaluate_and_optimize.py`](../../examples/google_adk/03_evaluate_and_optimize.py)
  - introspect + evaluate + optimize a single ADK agent.
- [`04_multi_agent_and_training.py`](../../examples/google_adk/04_multi_agent_and_training.py)
  - coordinator + sub_agents optimized as one unit, plus the YAML
    `run_training` path ([`google_adk.train.yaml`](../../examples/google_adk/google_adk.train.yaml)).
- [`05_tools_approvals_streaming.py`](../../examples/google_adk/05_tools_approvals_streaming.py)
  - a denied tool in a sub-agent, a poisoned tool result, an approval round trip,
    and a buffered SSE stream, all under one plugin.
