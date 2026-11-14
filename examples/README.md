# ADAPT-Agent Examples

Runnable examples covering every part of the library. Each script is
self-contained, deterministic and offline: **no optional dependencies, no API
key, no network**. Where a framework or a model is needed, a small stand-in
plays its part and the ADAPT-Agent code is the real thing.

```bash
python examples/01_firewall_and_policy.py
python examples/14_end_to_end_support_desk.py     # everything together
```

## Where to start

| If you want to... | Read |
|---|---|
| See everything working together in one application | [`14_end_to_end_support_desk.py`](14_end_to_end_support_desk.py) |
| Screen input and output, and write policy rules | [`01`](01_firewall_and_policy.py), [`09`](09_verifying_your_guardrails.py) |
| Understand what injection detection does and does not catch | [`02`](02_adversarial_defense.py) |
| Wrap an existing agent | [`03`](03_langgraph_guarded_agent.py), [`05`](05_multi_framework_adapters.py) |
| Govern each agent inside a multi-agent system | [`12`](12_native_framework_hooks.py) |
| Run async, add middleware, or roll out without blocking | [`11`](11_middleware_async_and_rollout.py) |
| Track trust, untrusted data and memory | [`10`](10_trust_taint_and_memory.py) |
| Trace and measure | [`04`](04_observability.py) |
| Score an agent against a golden dataset | [`08`](08_agent_evals.py) |
| Tune prompts, models and parameters | [`06`](06_optimize_with_golden_dataset.py), [`13`](13_optimizer_strategies.py), [`07`](07_train_from_yaml.py) |
| Do any of this from a shell or CI | [`15`](15_cli_tour.py) |

## What each part of the library looks like in use

| Capability | Class / function | Example |
|---|---|---|
| Input/output screening | `Firewall` | 01, 09, 14 |
| Declarative rules | `PolicyEnforcer` | 01, 09, 10, 14 |
| Injection / jailbreak detection | `AdversarialDefense` | 02, 11 |
| Checking a guardrail is not vacuous | `dry_run`, `evaluate_rules`, `vacuous_rules` | 09, 14 |
| Wrapping an agent | `LangGraphAdapter` and the other six adapters | 03, 05 |
| Per-agent hooks | `adapt_agent.integrations.*`, `GovernanceGate` | 12, 14 |
| Async | `aexecute` | 11 |
| Rewriting payloads | `Middleware` | 11, 14 |
| Report-only rollout | `block_on_violation=False` | 11 |
| Trust | `TrustManager` | 10, 14 |
| Provenance of untrusted data | `TaintTracker` | 10, 14 |
| Memory | `MemorySystem` | 10, 14 |
| Tracing and metrics | `AgentObserver`, `AgentEvaluator` | 04, 14 |
| Evals | `evaluate_agent`, `GoldenDataset`, `LLMJudge` | 08, 14 |
| Optimization | the six optimizers, `make_default_optimizer` | 06, 13 |
| Validation gate, independent judge | `val_dataset=`, `proposer_judge=` | 06, 13 |
| Exporting a tuned config | `to_config`, `load_tuned_config` | 13 |
| Training from a config file | `TrainingConfig`, `adapt-agent train` | 07 |
| Runtime performance hints | `AgentOptimizer` | 13 |
| Command line | `adapt-agent ...` | 15 |

## The examples

### `01_firewall_and_policy.py`
Builds a `Firewall` (blocked regex patterns + a `max_content_length` limit) and
a `PolicyEnforcer` (a rule that blocks messages containing a secret). It
demonstrates `check_input`, `check_message`, and `check_state`, and prints the
results plus recorded violations.

### `02_adversarial_defense.py`
`AdversarialDefense`, honestly. Shows what it catches -- including attacks a
person would never see, such as zero-width characters, HTML comments, invisible
CSS, markdown link titles and forged role tags -- and then what it does **not**
catch (base64, look-alike letters, leetspeak, other languages, novel phrasing),
plus a false positive on text that merely discusses an attack. Ends with custom
signatures via `add_attack_pattern` and the attack log.

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

### `10_trust_taint_and_memory.py`
The three bookkeeping components, each with the wiring that turns it into an
enforced control. `TrustManager` scores move with behaviour (a self-reported
score is ignored) and feed a policy rule that blocks. `TaintTracker` follows
untrusted data through a pipeline, shows where a value came from, and gates a
tool call until its arguments have been sanitised. `MemorySystem` shows bounded
storage, search, and consolidation into long-term memory.

### `11_middleware_async_and_rollout.py`
How governance fits around a real application. An ordered `Middleware` pipeline
(redaction in, a disclaimer out), fail-open versus fail-closed when a middleware
crashes, the order of stages inside an adapter, `aexecute` running three
requests concurrently with identical governance, and a **report-only rollout**
(`block_on_violation=False`) that serves everything while recording what would
have been refused.

### `12_native_framework_hooks.py`
Why wrapping a multi-agent system from the outside is not enough: an instruction
planted in a tool result passes an outer wrapper untouched and is caught by a
hook placed before the agent that reads it. Then one shared `GovernanceGate`
bound per agent, and the native seam for LangGraph, CrewAI (crew callbacks and a
task guardrail), Google ADK, Microsoft Agent Framework and Pydantic AI, driven
through framework-shaped stand-ins.

### `13_optimizer_strategies.py`
One ticket-triage agent with four kinds of knob (prompt, model, temperature,
few-shot), and every strategy run against it: grid, random, coordinate ascent,
bootstrap few-shot, evolutionary, and the staged default pipeline. Shows which
strategy can move which knob, the **validation gate** refusing a prompt that
memorised the training tickets, exporting the result with `to_config` and
reloading it into a fresh agent, and `AgentOptimizer`'s runtime hints.

### `14_end_to_end_support_desk.py`
Everything together in one small application: a support desk with a firewall,
adversarial defense and policy at the boundary, a per-agent hook on the
knowledge-base lookup, redacting middleware, trust that falls with refusals,
a refund tool gated on taint, memory, tracing and metrics. It runs startup
checks that prove each guardrail works, a mix of honest and hostile traffic,
prints the audit trail, and finishes with a quality eval of the governed agent.

### `15_cli_tour.py`
Builds a throwaway project and runs each `adapt-agent` command against it:
`validate` on a good and a broken config (the broken one exits non-zero, for
CI), `monitor`, `skills` and `install skill`, then `evaluate` and `optimize`
with a held-out set and a saved config.

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
