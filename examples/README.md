# ADAPT-Agent Examples

Runnable, beginner-friendly examples for the core features of ADAPT-Agent.
Each script is self-contained and can be run directly:

```bash
python examples/01_firewall_and_policy.py
python examples/02_adversarial_defense.py
python examples/03_langgraph_guarded_agent.py
python examples/04_observability.py
python examples/05_multi_framework_adapters.py
```

None of these examples require any optional dependencies (including
`langgraph` - see the note below).

## The examples

### `01_firewall_and_policy.py`
Builds a `Firewall` (blocked regex patterns + a `max_content_length` limit) and
a `PolicyEnforcer` (a rule that blocks messages containing a secret). It
demonstrates `check_input`, `check_message`, and `check_state`, and prints the
results plus recorded violations.

### `02_adversarial_defense.py`
Uses `AdversarialDefense` to analyze a benign prompt and several malicious ones:
prompt injection, a jailbreak attempt, and a custom pattern registered via
`add_attack_pattern`. It prints the analysis dictionaries returned by
`analyze_input`.

### `03_langgraph_guarded_agent.py`
Wraps an agent with `LangGraphAdapter`, configured with a `Firewall`,
`AdversarialDefense`, and `AgentObserver`. It calls `execute()` on safe input
(which succeeds) and on a prompt-injection input (which raises
`SecurityBlockedError`, caught and printed), then prints the observer's traces.

> **Note on LangGraph.** This example defines a tiny `FakeGraph` with an
> `invoke(state)` method so it runs **without installing langgraph**. The
> adapter never imports langgraph; it only needs an object with a callable
> `invoke(state)`. To use a real compiled LangGraph graph, build and compile
> your graph and pass it to `adapter.wrap_agent(...)`:
>
> ```python
> from langgraph.graph import StateGraph, START, END
>
> builder = StateGraph(dict)
> builder.add_node("respond", my_node_function)
> builder.add_edge(START, "respond")
> builder.add_edge("respond", END)
> compiled_graph = builder.compile()
>
> guarded = adapter.wrap_agent(compiled_graph)
> guarded.execute({"messages": [{"role": "user", "content": "hi"}]})
> ```
>
> No other code changes are required.

### `04_observability.py`
Uses `AgentObserver` to start/observe a trace, log events, and record metrics,
then prints `get_metric_stats` and `get_traces`. It also shows `AgentEvaluator`
with a custom metric and `compute_aggregate_metrics`.

### `05_multi_framework_adapters.py`
Runs the **same** governed payload through the Pydantic AI, Microsoft Agent
Framework, CrewAI, and Claude Agent SDK adapters using tiny framework-shaped
fakes (so it needs **no** optional dependencies). Demonstrates that every
adapter shares one constructor and one pipeline, that async agents (`.run`
coroutines, `query` async generators) are driven synchronously, and that a
prompt-injection input is blocked identically across all of them.

## Configuration file

### `config.example.json`
A valid configuration file matching the CLI schema (`policy_rules`,
`firewall.blocked_patterns` / `allowed_patterns` / `max_content_length`,
`adversarial.attack_patterns`). Validate it with:

```bash
adapt-agent validate examples/config.example.json
# or, if the console script is not on your PATH:
python -c "import sys; from adapt_agent.cli import main; sys.exit(main())" validate examples/config.example.json
```
