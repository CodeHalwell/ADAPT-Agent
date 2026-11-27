# Google ADK + ADAPT-Agent examples

A four-step ladder from a single guarded [Google ADK](https://google.github.io/adk-docs)
agent to a governed, optimized multi-agent system. Each script is self-contained.

ADAPT-Agent has two halves, both framework-agnostic:

- **Guard (runtime):** wrap an agent so every call runs a six-step security +
  observability pipeline (`GoogleADKAdapter`).
- **Train (offline):** turn an agent (or whole system) into a tunable search
  space and optimize it against a golden dataset, scored by an LLM-as-judge.

## The core ADK shape

ADK does not run an agent by calling a method on it. You put the agent in an
`App`, build a `Runner`, open a session, and feed it a message. ADAPT-Agent
plugs in at two places:

```python
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from adapt_agent import Firewall
from adapt_agent.adapters import GoogleADKAdapter
from adapt_agent.integrations.google_adk import governance_plugin

fw = Firewall()
app = App(name="app", root_agent=my_agent,
          plugins=[governance_plugin(firewall=fw)])   # every model + tool call
runner = InMemoryRunner(app=app)

guarded = GoogleADKAdapter(firewall=fw).wrap_agent(runner)   # the boundary
guarded.execute({"prompt": "hi"}, user_id="u", session_id="s")
```

The plugin covers every agent and tool in the tree (sub-agents, `AgentTool`s,
`Workflow` nodes, MCP tools). The adapter accepts a `Runner`, an `App`, or a
callable you write; `user_id` / `session_id` / `run_config` passed to `execute`
reach the run, and a block raised inside it surfaces as `SecurityBlockedError`.

## Running

```bash
pip install 'adapt-agent[google-adk]'
python examples/google_adk/01_basic_guarded.py
python examples/google_adk/02_policy_observability_trust.py
python examples/google_adk/03_evaluate_and_optimize.py
python examples/google_adk/04_multi_agent_and_training.py
python examples/google_adk/05_tools_approvals_streaming.py
```

Every example runs offline (no API key, no network): 01 and 05 drive a real ADK
`Runner` with a scripted `BaseLlm` stand-in, and print a skip message (exit 0)
when the extra is missing; 02-04 use a small ADK-*shaped* stand-in object, which
ADAPT-Agent's introspection recognises by duck typing, so a real agent is a
drop-in replacement.

## The ladder

### `01_basic_guarded.py`
The smallest guarded ADK agent: `governance_plugin` on the `App`, the
`GoogleADKAdapter` around the `Runner`. A safe question is answered; a
prompt-injection input is blocked with `SecurityBlockedError` before the model
is called.

### `02_policy_observability_trust.py`
Adds the full control stack on a Google ADK agent: `PolicyEnforcer` (a block
rule), `AdversarialDefense`, `AgentObserver` (traces), and pre/post `Middleware`.
Shows `block_on_violation=False` (threats recorded, not raised) and the
standalone `TrustManager` and `TaintTracker` helpers. Fully offline.

### `03_evaluate_and_optimize.py`
Wraps a single ADK agent in `OptimizableAgent`, builds a `GoldenDataset`, scores
it with an `EvaluationHarness` (a metric + an offline `LLMJudge.as_metric`), runs
`CoordinateAscentOptimizer`, and prints `introspect(agent)`'s discovered knobs
plus baseline-to-best improvement. Fully offline.

### `04_multi_agent_and_training.py`
A coordinator `LlmAgent` with two specialist `sub_agents`, wrapped as ONE
governed/optimizable unit. `make_default_optimizer` runs the full pipeline
(few-shot to prompts to models/hparams to tools/skills) with the adversarial
judge proposing new tools on `result.recommendations`. Also shows the parallel
**YAML training** path (`run_training`) mirroring
[`google_adk.train.yaml`](./google_adk.train.yaml). Fully offline.

### `05_tools_approvals_streaming.py`
Tool governance under one plugin: a sub-agent with no callbacks of its own is
refused a denied tool (the run continues), a poisoned tool result is replaced
before the next model call, a `require_approval` tool pauses for ADK's
confirmation flow and runs once approved, and a secret split across SSE chunks
is caught on the aggregate with `buffer_stream=True` -- nothing of it delivered.

## What gets introspected for a Google ADK agent

| Attribute | Kind | Notes |
|---|---|---|
| `instruction`, `global_instruction` | PROMPT | only when plain strings (not instruction-provider callables) |
| `model` | MODEL | a string id, or a model object's `model` / `model_name` |
| `generate_content_config.temperature` / `top_p` / `max_output_tokens` | HYPERPARAM | bounds clamped to provider max with a warning |
| `tools` | TOOL | drop-one ablation candidates when 2+ tools |
| `sub_agents` | ROUTING | walked **recursively**; nested knobs namespaced under the parent |
