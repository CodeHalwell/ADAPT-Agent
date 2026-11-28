# Framework Adapters

Adapters integrate ADAPT-Agent's security and governance primitives with
third-party LLM agent frameworks. Importing an adapter class never imports the
underlying framework — the framework is only needed at runtime when you build the
agent you wrap — so `adapt_agent` stays import-safe without optional dependencies
installed.

## Support matrix

| Framework | Class | Extra | Wrap target |
|-----------|-------|-------|-------------|
| LangGraph | `LangGraphAdapter` | `adapt-agent[langgraph]` | compiled graph (`.invoke`; governed `stream` / `astream`) |
| Microsoft Agent Framework | `MicrosoftAgentFrameworkAdapter` | `adapt-agent[microsoft-agent-framework]` | `Agent` (`.run`; called `ChatAgent` before 1.0), or a workflow's `as_agent()` |
| Google ADK | `GoogleADKAdapter` | `adapt-agent[google-adk]` | a `Runner`, an `App`, or a callable driving a `Runner` |
| Pydantic AI | `PydanticAIAdapter` | `adapt-agent[pydantic-ai]` | `Agent` (`.run_sync` / `.run`) |
| CrewAI | `CrewAIAdapter` | `adapt-agent[crewai]` | `Crew` or `Flow` (`.kickoff` / `.akickoff`), or a standalone `Agent` |
| OpenAI Agents SDK | `OpenAIAgentsAdapter` | `adapt-agent[openai-agents]` | `Agent` (driven via `Runner`; also `run_streamed`) |
| Claude Agent SDK | `ClaudeAgentSDKAdapter` | `adapt-agent[claude-agent]` | the `query` function, or a connected `ClaudeSDKClient` |

```python
from adapt_agent.adapters import (
    BaseAdapter,
    GovernedAdapter,
    LangGraphAdapter,
    MicrosoftAgentFrameworkAdapter,
    GoogleADKAdapter,
    PydanticAIAdapter,
    CrewAIAdapter,
    OpenAIAgentsAdapter,
    ClaudeAgentSDKAdapter,
)
```

Every adapter shares one governance pipeline (see
[The unified pipeline](#the-unified-pipeline)) and the same keyword-only
constructor. Importing an adapter never imports its framework — the framework is
only needed at runtime when you build and run the agent you wrap. Async-only
frameworks (Microsoft Agent Framework, Google ADK, Claude Agent SDK) are driven
synchronously: coroutines are awaited and async event streams are drained to a
list, so `execute` stays synchronous and the firewall can scan every result.

---

## The unified pipeline

All adapters extend `GovernedAdapter`, which applies the same six steps on every
`execute` / `aexecute` call, regardless of framework:

1. **Input screening** — `Firewall` + `AdversarialDefense` scan every string in
   the payload. A payload too deep or too large to read in full is blocked as
   `unscannable` / `payload_too_large` (see [Security](security.md)).
2. **Policy enforcement** — `PolicyEnforcer` is evaluated against the extracted
   `AgentState`; only `action == "block"` rules block. Policy is evaluated (and
   violations recorded) even in report-only mode.
3. **Pre-middleware** — an optional `Middleware` pipeline may rewrite the input.
   Whatever it *added* (retrieved context, a template, a rewritten prompt) is
   screened again before the agent sees it.
4. **Traced execution** — the framework agent runs, optionally traced by an
   `AgentObserver`.
5. **Post-middleware** — the pipeline may rewrite the result.
6. **Output screening** — the firewall scans every string in the result.

A request refused at step 1 or 2 never starts a run, so it opens no trace; it is
reported to the observer with `record_refusal` instead (`observer.get_refusals()`
lists it with its `stage` and threats). A request refused at step 6 closes its
trace as an error. Either way, `get_state()` keeps describing the last request
that was actually served.

An adapter governs the run's **boundary**. By the time it screens the result
every tool has run; to govern the tool calls themselves, add the framework's
[native binding](integrations.md).

The constructor is identical for every adapter and all controls are optional and
keyword-only:

```python
from adapt_agent import Firewall, AdversarialDefense, PolicyEnforcer, AgentObserver, Middleware
from adapt_agent.adapters import PydanticAIAdapter

adapter = PydanticAIAdapter(
    firewall=Firewall(max_content_length=10_000),
    defense=AdversarialDefense(),
    policy_enforcer=PolicyEnforcer(),
    observer=AgentObserver(),
    middleware=None,              # can be added later via inject_middleware
    agent_id="support-bot",       # defaults to "<framework>-agent"
    block_on_violation=True,      # default
    tool_policy=None,             # carried on adapter.gate, for a native binding
    max_payload_chars=4_000_000,  # per-payload screening budget; None disables it
)
```

`tool_policy` does nothing at the boundary on its own — the adapter never sees an
individual tool call. It is carried on `adapter.gate`, so one adapter object can
configure both the boundary and a native binding built from that gate
(`governance_capability(gate=adapter.gate)`, `adapter.gate.check_tool_call(...)`).
`CrewAIAdapter` is the exception: with `hooks=True` it runs CrewAI's tool hooks
itself (below), and it does not take `max_payload_chars`.

## Running: `execute(input, **run_kwargs)`

```python
out = guarded.execute(payload, **run_kwargs)          # sync
out = await guarded.aexecute(payload, **run_kwargs)   # async; same governance
```

**Extra keyword arguments are forwarded** to the framework's run method
unchanged: a LangGraph `config={"configurable": {"thread_id": ...}}`, a Pydantic
AI `deps=` / `message_history=`, an OpenAI Agents `session=` / `context=` /
`run_config=`, a MAF `session=` / `stream=True` / `middleware=[...]`, ADK
`user_id=` / `session_id=` / `run_config=`, Claude `options=`, CrewAI
`input_files=` / `from_checkpoint=`. They configure the run and are not screened
as content.

**The input need not be a dict.** A string, a framework object (a LangGraph
`Command(resume=...)`, an OpenAI Agents `RunState`) or a message list is
screened like any payload and handed to the adapter unchanged. Pre-middleware
processes dicts only, so an adapter with pre-middleware configured refuses a
non-dict input with `AdapterError` rather than skip the stage silently.

A non-dict framework result comes back as `{"result": <result>}`; a dict result
is returned as-is.

### Per-framework wrap targets

The Microsoft Agent Framework extra installs `agent-framework-core`, which is all
the adapter needs. Install one framework extra per environment: the frameworks
constrain each other's dependencies (CrewAI pins `openai<3`, which holds the OpenAI
Agents SDK and Pydantic AI back), which is why `adapt-agent[all]` leaves CrewAI out.

The only thing that differs between frameworks is the object you pass to
`wrap_agent`, how the input is shaped, and what a framework adds on top:

```python
# LangGraph — wrap a compiled graph; the payload dict is passed to .invoke()
LangGraphAdapter().wrap_agent(compiled_graph)

# Microsoft Agent Framework — wrap an Agent (or workflow.as_agent()); .run() is awaited
MicrosoftAgentFrameworkAdapter().wrap_agent(maf_agent)

# Pydantic AI — wrap an Agent; .run_sync()/.run() receive a prompt string
PydanticAIAdapter().wrap_agent(pydantic_agent)

# CrewAI — wrap a Crew; the payload's non-"messages" keys become kickoff(inputs=...)
CrewAIAdapter().wrap_agent(crew)

# OpenAI Agents SDK — wrap an Agent (driven via a lazily-imported Runner)
OpenAIAgentsAdapter().wrap_agent(openai_agent)

# Claude Agent SDK — wrap the query function; the async stream is drained
from claude_agent_sdk import query
ClaudeAgentSDKAdapter().wrap_agent(query)

# Google ADK — wrap a Runner (or an App); execute() takes user_id= / session_id=
GoogleADKAdapter().wrap_agent(runner)
```

For prompt-based frameworks (Microsoft Agent Framework, Pydantic AI, OpenAI
Agents, Claude Agent SDK, and Google ADK when wrapping a `Runner` or `App`) the
adapter derives the prompt from the payload's latest user message, so the same
`{"messages": [...]}` shape works everywhere.

What each adapter adds:

- **LangGraph.** `guarded.stream(...)` / `guarded.astream(...)` take the graph's
  own arguments; the input is screened before the graph starts and every item is
  screened **before it is yielded** (in `stream_mode="messages"` the token chunks
  of one message are held and released together, so a pattern split across
  tokens is still caught). An interrupted run returns its `"__interrupt__"`, and
  `guarded.execute(Command(resume=...), config=config)` resumes it: the `Command`
  is screened as input and policy is evaluated against the thread's
  checkpointed state plus the `Command`'s `update`.
- **OpenAI Agents SDK.** A run paused for approval comes back with
  `out["interruptions"]`; approve or reject on `out["result"].to_state()` and
  pass the state back as `guarded.execute({"input": state})` (a bare `RunState`
  works too). `{"input": [...]}` passes SDK input items through.
  `adapter.run_streamed(agent, payload, **run_kwargs)` runs the input pipeline
  and returns a governed stream that holds each response's text until screened.
- **CrewAI.** `CrewAIAdapter(hooks=True, tool_policy=..., on_tool_violation=..., approve=...)`
  runs each execution inside CrewAI's interception hooks
  ([`governance_hooks`](integrations.md#crewai), CrewAI 1.15+), scoped to that run:
  tool calls and results, model calls and steps are governed too. `tool_policy`,
  `on_tool_violation` and `approve` without `hooks=True` raise `ValueError`
  rather than being accepted and ignored. A standalone `Agent` is asked the
  payload's messages (`kickoff(messages)`); `aexecute` uses the native
  `akickoff`. A `Crew(stream=True)` is drained inside governance and the final
  `CrewOutput` returned only after screening.
- **Google ADK.** Wrapping a `Runner` sends the prompt as `new_message`; the
  session is created on first use (omit `session_id` for a fresh one per call),
  and `run_config=`, `state_delta=` and a ready-made `new_message=` pass through.
  An `App` runs on an `InMemoryRunner` the adapter keeps for it. A
  `SecurityBlockedError` raised by the governance plugin inside the run reaches
  you as itself, not wrapped in ADK's `RuntimeError`. A callable gets the full
  payload and every keyword argument.
- **Microsoft Agent Framework.** A payload whose `messages` hold MAF
  `Message` / `Content` objects is passed to `run` as a message list (plain
  `{"role", "content"}` dicts are still reduced to the latest user prompt). A
  run paused on a tool approval returns `{"result": ..., "user_input_requests": [...]}`;
  resume by passing the approval responses back as messages on the same
  `session=`. A streamed run (`stream=True`) is drained, screened and returned
  as `{"result": <final response>, "updates": [...]}`.
- **Claude Agent SDK.** Besides `query`, a *connected* `ClaudeSDKClient` can be
  wrapped: the adapter sends the prompt and screens everything
  `receive_response()` yields. Use `aexecute` for a client (its connection is
  bound to the loop it was opened in). The stream is drained before anything is
  returned, so a blocked answer is never delivered.

---

## BaseAdapter

All adapters subclass `adapt_agent.adapters.BaseAdapter` (an `ABC`). It defines
the contract:

- `wrap_agent(agent) -> Agent` — wrap a framework agent so it implements the
  ADAPT-Agent `Agent` protocol (`execute`, `get_state`). *(abstract)*
- `extract_state(agent) -> AgentState` — produce a normalized `AgentState`.
  *(abstract)*
- `inject_middleware(agent, middleware) -> Any` — attach a middleware pipeline.
  *(abstract)*
- `validate_agent(agent) -> bool` — compatibility check (overridable).
- `get_framework_name() -> str` — defaults to the class name minus `"Adapter"`.

The constructor takes an optional `config: dict`.

---

## LangGraph integration walkthrough

`LangGraphAdapter` wraps a **compiled** LangGraph graph — or any object exposing
a callable `invoke(state) -> state` — and applies, in order on each
`execute`: input screening → policy enforcement → pre-middleware → traced
`graph.invoke` → post-middleware → output screening.

### Constructing the adapter

Every control is optional and keyword-only:

```python
from adapt_agent import Firewall, AdversarialDefense, PolicyEnforcer, AgentObserver, Middleware
from adapt_agent.adapters import LangGraphAdapter

fw = Firewall(max_content_length=10_000)
fw.add_blocked_pattern(r"(?i)ignore previous instructions")

defense = AdversarialDefense(max_content_length=10_000)
defense.add_attack_pattern("leak the system prompt")

policy = PolicyEnforcer()
policy.add_rule(
    name="no_system_role",
    description="Block injected system-role messages in context",
    condition="state['context'] != {}",  # safe expression; see Policy docs
    action="warn",
    severity="medium",
)

adapter = LangGraphAdapter(
    config={},                 # optional; e.g. {"graph_config": {...}}
    firewall=fw,
    defense=defense,
    policy_enforcer=policy,
    observer=AgentObserver(),
    middleware=None,           # can be added later via inject_middleware
    agent_id="support-bot",
    block_on_violation=True,   # default
)
```

### Wrapping a compiled graph

```python
# Build and compile a LangGraph graph (requires the `langgraph` extra)...
compiled_graph = graph.compile()   # anything with a callable .invoke()

guarded = adapter.wrap_agent(compiled_graph)
```

`wrap_agent` calls `validate_agent`, which checks that the object has a callable
`invoke`. If not, it raises `AdapterError`:

```python
from adapt_agent.exceptions import AdapterError

try:
    adapter.wrap_agent(object())  # no .invoke
except AdapterError as exc:
    print(exc)  # explains you must compile your graph first
```

### Executing with governance

```python
from adapt_agent.exceptions import SecurityBlockedError

state = {"messages": [{"role": "user", "content": "What are your hours?"}]}

try:
    result = guarded.execute(state)
    print(result)             # the graph's output state (a dict)
    print(guarded.get_state())  # most recently observed AgentState
except SecurityBlockedError as exc:
    print("Blocked:", exc.reason)
    print("Threats:", exc.threats)  # e.g. ['firewall', 'prompt_injection']
```

How blocking works when `block_on_violation=True`:

- **Input screening.** Each text extracted from the payload is run through the
  firewall (`check_input`) and defense (`analyze_input`). Any hit raises
  `SecurityBlockedError("Input blocked by security controls", threats)`.
- **Policy.** The state is extracted and `policy_enforcer.check_state` runs. Only
  fired rules whose `action == "block"` cause
  `SecurityBlockedError("Input blocked by policy", ["policy:<rule>"])`.
- **Output screening.** Texts in the result are run through `check_output`; a hit
  raises `SecurityBlockedError("Output blocked by security controls", threats)`.

With `block_on_violation=False`, execution proceeds and threats are still
recorded by the underlying controls (inspect them via `fw.get_security_events()`,
`defense.get_detected_attacks()`, `policy.get_violations()`).

### extract_state

`extract_state` normalizes a raw payload (or a stateful graph exposing
`get_state`) into an `AgentState`. It is best-effort and always returns a
well-formed state:

```python
state = adapter.extract_state(
    {"messages": [{"role": "user", "content": "hi"}], "user_id": 42, "trust_score": 0.7}
)
# {
#   'messages': [{'role': 'user', 'content': 'hi'}],
#   'context': {'user_id': 42, 'trust_score': 0.7},
#   'trust_score': 0.7,            # promoted when numeric
# }
```

Behavior details:

- `messages` is taken from the payload if it is a list, otherwise `[]`.
- Every non-`messages` key is collected under `context`.
- `trust_score` is promoted to a top-level float **only** when numeric.
- `policy_violations` is promoted **only** when it is a list.
- If the object exposes a callable `get_state`, it is called with
  `config.get("graph_config", {})` and `.values` is read from the snapshot.

### inject_middleware

`inject_middleware` attaches a `Middleware` pipeline and returns a freshly
wrapped agent. The pipeline's `process_input` runs before `graph.invoke` and
`process_output` runs after (the result is passed as `{"result": result}`).

```python
from adapt_agent import Middleware

mw = Middleware()

def tag_request(data):
    data.setdefault("kwargs", {})  # data shape depends on call site
    return data

mw.add_pre_middleware(tag_request, name="tag", priority=10)

guarded = adapter.inject_middleware(compiled_graph, mw)
```

`inject_middleware` raises `AdapterError` if `middleware` is not an
`adapt_agent.core.Middleware` instance.

---

## Other frameworks

Every adapter is a thin subclass of `GovernedAdapter`. The base wraps anything
exposing a recognized "run" method (or a plain callable) and applies the full
pipeline, so adding a new framework is usually a few lines:

```python
from adapt_agent.adapters import GovernedAdapter

class MyFrameworkAdapter(GovernedAdapter):
    framework_name = "MyFramework"
    run_method_names = ("run", "invoke")   # tried in order; callable fallback otherwise
    operation = "myframework.run"

    def _prepare_input(self, payload):     # optional: shape the runner's argument
        return payload
```

Because the base accepts any callable, you can also wrap an arbitrary
`run(input) -> output` function for frameworks not listed above without writing a
subclass at all:

```python
from adapt_agent.adapters import GovernedAdapter

guarded = GovernedAdapter(firewall=fw).wrap_agent(lambda payload: my_runtime(payload))
```

## Async: `aexecute`

A governed agent exposes two entry points with identical governance:

```python
result = guarded.execute(payload)          # synchronous callers
result = await guarded.aexecute(payload)   # any async application
```

`execute` drives an async framework by running its coroutine to completion,
which is impossible inside a running event loop — there it raises
`AdapterError`. Since Pydantic AI, the Claude Agent SDK and Microsoft Agent
Framework are async-native, `aexecute` is their normal path.

Prefer it over offloading `execute` to a worker thread. A thread serialises
concurrent requests behind one blocking call and severs `contextvars`, which is
how OpenTelemetry propagates the active span — so the trace parentage is lost.
`aexecute` awaits in the caller's loop and keeps both. A *synchronous* framework
works through `aexecute` too, so an async app can use one entry point for every
agent.

To govern individual agents *inside* a multi-agent graph rather than at its
boundary, see [Native framework hooks](integrations.md).
