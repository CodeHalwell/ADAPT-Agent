# Framework Adapters

Adapters integrate ADAPT-Agent's security and governance primitives with
third-party LLM agent frameworks. Importing an adapter class never imports the
underlying framework — the framework is only needed at runtime when you build the
agent you wrap — so `adapt_agent` stays import-safe without optional dependencies
installed.

## Support matrix

| Framework | Status | Class | Extra |
|-----------|--------|-------|-------|
| LangGraph | **Supported** | `LangGraphAdapter` | `adapt-agent[langgraph]` |
| Semantic Kernel | Experimental / planned | `SemanticKernelAdapter` | `adapt-agent[semantic-kernel]` |
| CrewAI | Experimental / planned | `CrewAIAdapter` | `adapt-agent[crewai]` |

```python
from adapt_agent.adapters import (
    BaseAdapter,
    LangGraphAdapter,
    SemanticKernelAdapter,
    CrewAIAdapter,
)
```

!!! warning "Experimental adapters"
    `SemanticKernelAdapter` and `CrewAIAdapter` are placeholders that define the
    intended interface. Every method raises `NotImplementedError` with a message
    pointing you to `LangGraphAdapter`. They also set `__experimental__ = True`.
    Use LangGraph for a working integration today.

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

## Experimental adapters

```python
from adapt_agent.adapters import CrewAIAdapter, SemanticKernelAdapter

adapter = CrewAIAdapter()
adapter.wrap_agent(some_agent)
# NotImplementedError: The CrewAI adapter is experimental and not yet implemented...

print(CrewAIAdapter.__experimental__)        # True
print(SemanticKernelAdapter().get_framework_name())  # "SemanticKernel"
```
