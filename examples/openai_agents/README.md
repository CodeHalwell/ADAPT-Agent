# ADAPT-Agent x OpenAI Agents SDK examples

A four-step ladder that climbs from a single guarded agent to a governed,
optimizable multi-agent system built on the
[OpenAI Agents SDK](https://openai.github.io/openai-agents-python) (PyPI package
`openai-agents`, imported as `agents`).

```bash
python examples/openai_agents/01_basic_guarded.py
python examples/openai_agents/02_policy_observability_trust.py
python examples/openai_agents/03_evaluate_and_optimize.py
python examples/openai_agents/04_multi_agent_and_training.py
```

Each script **guards its framework import**: run it without the SDK installed and
it prints a friendly install hint and exits cleanly. Install the extra with:

```bash
pip install 'adapt-agent[openai-agents]'   # or: pip install openai-agents
```

Examples 02, 03 and 04 run **fully offline** (no API key, no network): the LLM
judge is backed by a deterministic stub and the agents are driven by a local
runner. Example 01 sends its *safe* prompt to the real OpenAI API; without
`OPENAI_API_KEY` that one call is caught and reported, while the *malicious*
prompt is still blocked before any network call.

## The ladder

### `01_basic_guarded.py`
The smallest real OpenAI `Agent` (just `name` + `instructions`), wrapped with
`OpenAIAgentsAdapter` and a `Firewall`. Shows a safe input passing input
screening and a prompt-injection input raising `SecurityBlockedError` *before*
the SDK's `Runner` is ever invoked.

### `02_policy_observability_trust.py`
Adds the rest of the runtime guard stack: `PolicyEnforcer` (a `warn` rule over
`state['messages']`), `AdversarialDefense`, `AgentObserver` (printed traces),
`Middleware` (pre/post pipeline), and a `TrustManager` side-car. Uses
`block_on_violation=False` so threats are recorded in **monitor mode** rather
than raising.

### `03_evaluate_and_optimize.py`
The training half for a single agent: `detect`/`introspect` discover the agent's
tunable knobs (instructions, model, model_settings, tools, handoffs), an
`OptimizableAgent` wraps it, an `EvaluationHarness` scores it over a
`GoldenDataset` with `exact_match` plus an offline `LLMJudge`, and a
`CoordinateAscentOptimizer` improves the prompt and applies the winner in place
(baseline 0.20 -> best 0.90 on the bundled data).

### `04_multi_agent_and_training.py`
A realistic topology: a **triage** `Agent` with `handoffs` to two specialist
agents, each carrying a `@function_tool`. The whole system is governed as one
unit, then optimized whole-system **and** per-agent with `make_default_optimizer`
(few-shot -> prompts -> models/hparams/routing -> tools/skills) including
drop-one **tool ablation** and judge-driven new-tool **recommendations**. It also
runs the same optimization via the declarative config path
(`run_training`); the project-style YAML lives in
[`openai_agents.train.yaml`](./openai_agents.train.yaml).

## Going live

Everywhere an offline runner is used, swapping in the real SDK is a one-line
change: build `Agent(...)` instances and let the adapter drive them through the
SDK's `Runner` (`Runner.run_sync(agent, prompt)`), or wrap your own runner that
calls `Runner.run`. See the commented "go live" snippets at the bottom of
examples 01 and 02.
