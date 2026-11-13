# ADAPT-Agent Examples

Runnable, beginner-friendly examples for the core features of ADAPT-Agent.
Each script is self-contained and can be run directly:

```bash
python examples/01_firewall_and_policy.py
python examples/02_adversarial_defense.py
python examples/03_langgraph_guarded_agent.py
python examples/04_observability.py
python examples/05_multi_framework_adapters.py
python examples/06_optimize_with_golden_dataset.py
python examples/07_train_from_yaml.py
python examples/08_agent_evals.py
python examples/09_verifying_your_guardrails.py
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

### `06_optimize_with_golden_dataset.py`
Runs the full **optimization** loop end to end with **no API key and no
network**: a `GoldenDataset`, an `OptimizableAgent` wrapping a tiny
orchestrator + specialist system whose behaviour depends on a tunable prompt,
**two** `LLMJudge`s (deterministic offline stubs) -- one scores answers, a
different one rewrites the prompt from failures (`proposer_judge=`), so the
model that writes a candidate is not the one that marks it -- and a
`CoordinateAscentOptimizer` that searches against the training rows and applies
the winner to the live agent only if it holds up on **held-out** rows
(`val_dataset=`, the validation gate). Swap the stubs for two different real
providers for real optimization.

### `07_train_from_yaml.py`
Runs the same kind of optimization from a **declarative training config**
instead of hand-written optimizer code. The script builds the config in-process
(the structure you would normally put in a `train.yaml`; see
[`train.example.yaml`](train.example.yaml)) and runs it offline. This is what
`adapt-agent train train.yaml` does from the command line.

### `08_agent_evals.py`
Runs **one-call evals** (`evaluate_agent`) against agents shaped like each of
the four major frameworks — Pydantic AI, Microsoft Agent Framework, LangGraph,
and Google ADK (via `adk_runner`) — entirely offline. The golden dataset mixes
**per-row checks**: exact text match, a number-within-tolerance check that
passes on "The answer is 42.", a parameterised tolerance, and an LLM-judge row
graded against per-row criteria by a deterministic offline judge stub.
Framework-native results (an `AgentRunResult`, a LangGraph state dict, an ADK
event stream) are unwrapped to final response text automatically. See
[docs/evals.md](../docs/evals.md).

### `09_verifying_your_guardrails.py`
Shows how to check that a guardrail actually does something. `Firewall.dry_run`
reports a decision and its reason without recording anything, and
`get_pattern_hits` exposes a blocked pattern that has never matched.
`PolicyEnforcer.add_rule` rejects a rule it could never evaluate;
`evaluate_rules` dry-runs rules against example states; the enforcer's
**fail-closed** default is contrasted with `fail_closed=False`; and
`vacuous_rules` finds rules that have never seen the data they read. It ends by
showing that a refused request does not become the wrapped agent's state.

## Per-framework example ladders

Each supported framework has its own folder with a 3–4 step ladder that climbs
from a tiny guarded agent to a governed, optimized multi-agent system, plus a
`train.yaml` and a README. Install the matching extra (e.g.
`pip install 'adapt-agent[langgraph]'`) and run any script directly; each guards
its framework import with a friendly hint and runs offline (no API key) where it
exercises ADAPT-Agent itself.

| Framework | Folder | Extra |
|-----------|--------|-------|
| LangGraph | [`langgraph/`](langgraph/) | `adapt-agent[langgraph]` |
| Microsoft Agent Framework | [`microsoft_agent_framework/`](microsoft_agent_framework/) | `adapt-agent[microsoft-agent-framework]` |
| Google ADK | [`google_adk/`](google_adk/) | `adapt-agent[google-adk]` |
| Pydantic AI | [`pydantic_ai/`](pydantic_ai/) | `adapt-agent[pydantic-ai]` |
| CrewAI | [`crewai/`](crewai/) | `adapt-agent[crewai]` |
| OpenAI Agents SDK | [`openai_agents/`](openai_agents/) | `adapt-agent[openai-agents]` |
| Claude Agent SDK | [`claude_agent/`](claude_agent/) | `adapt-agent[claude-agent]` |

Each ladder follows the same shape: `01` basic guarded agent → `02` policy /
observability / trust → `03` evaluate & optimize a single agent → `04` multi-agent
system + declarative YAML training. The verbose per-framework guides live in
[`docs/frameworks/`](../docs/frameworks/).

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
