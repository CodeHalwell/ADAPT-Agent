# ADAPT-Agent x OpenAI Agents SDK examples

A five-step ladder that climbs from a single guarded agent to a governed,
optimizable multi-agent system built on the
[OpenAI Agents SDK](https://openai.github.io/openai-agents-python) (PyPI package
`openai-agents`, imported as `agents`).

```bash
python examples/openai_agents/01_basic_guarded.py
python examples/openai_agents/02_policy_observability_trust.py
python examples/openai_agents/03_evaluate_and_optimize.py
python examples/openai_agents/04_multi_agent_and_training.py
python examples/openai_agents/05_tool_calls_handoffs_streaming.py
```

Each script prints what it ran, what passed, what
was blocked and why (the `SecurityBlockedError` threat labels), and exits 0. The
optimization examples print an `OptimizationResult` line such as
`baseline=0.200, best=0.900, ..., significant=False (p=0.0625, n=4)`: the gain is
applied, but on a handful of rows no outcome can pass the paired significance
test, so `improved` stays `False` -- use a larger golden set for a real verdict.

Each script **guards its framework import**: run it without the SDK installed and
it prints a friendly install hint and exits with status 0. Install the extra with:

```bash
pip install 'adapt-agent[openai-agents]'   # or: pip install openai-agents
```

Every example runs **fully offline** (no API key, no network). Examples 01 and
05 drive real SDK `Agent`s through the real `Runner` with the SDK's own
deterministic model double, `agents.testing.ScriptedModel`; 02-04 use a local
runner and a stubbed LLM judge.

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

### `05_tool_calls_handoffs_streaming.py`
Governance *inside* the run, through the SDK's own seams: `govern_tools` puts
every tool call and result through the gate (a denied tool never runs, a poisoned
page never reaches the model), a `ToolPolicy` approval pauses the run and is
resumed through the adapter with `execute({"input": state})`,
`governance_run_hooks` blocks a handoff target's leaking answer, and
`govern_stream` withholds a flagged streamed response while a clean one streams.

## Going live

Everywhere an offline runner or `ScriptedModel` is used, swapping in the real SDK
is a one-line change (drop `model=`; the SDK default is `gpt-5.6-luna`): build `Agent(...)` instances and let the adapter drive them through the
SDK's `Runner` (`Runner.run_sync(agent, prompt)`), or wrap your own runner that
calls `Runner.run`. See the commented "go live" snippets at the bottom of
examples 01 and 02.
