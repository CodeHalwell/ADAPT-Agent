# Microsoft Agent Framework + ADAPT-Agent examples

[Microsoft Agent Framework](https://github.com/microsoft/agent-framework) is
Microsoft's unified successor to Semantic Kernel and AutoGen. Its primary
runnable object is a **`ChatAgent`** (async `.run(prompt)` -> a response with
`.text`), and its multi-agent orchestrator is the **Magentic** workflow built
with `MagenticBuilder(...).build()` and exposed as a single agent via
`workflow.as_agent(name=...)`.

These four examples climb from a single guarded agent to a full Magentic team
that is governed as one unit and optimized offline. **Every example runs with no
API key and no network**: the framework objects are replaced by tiny offline
stand-ins that mirror the exact attribute/method surface ADAPT-Agent relies on,
so swapping in the real classes is a one-line change (shown in each file).

## Install

```bash
pip install 'adapt-agent[microsoft]'   # or: pip install agent-framework
```

`adapt_agent` itself never imports `agent_framework` -- the adapter and the
introspector duck-type, so importing ADAPT-Agent is always safe even without the
extra installed.

## The ladder

| # | File | What it teaches |
|---|------|-----------------|
| 1 | [`01_basic_guarded.py`](./01_basic_guarded.py) | The smallest agent: wrap a `ChatAgent` with a `Firewall` + `AdversarialDefense` + `AgentObserver`; run a safe input, then a prompt-injection input that raises `SecurityBlockedError`. |
| 2 | [`02_policy_observability_trust.py`](./02_policy_observability_trust.py) | Add `PolicyEnforcer` (block + warn rules), `Middleware`, `TrustManager`, `TaintTracker`; run with `block_on_violation=False` so threats are *recorded* without blocking. |
| 3 | [`03_evaluate_and_optimize.py`](./03_evaluate_and_optimize.py) | Wrap one agent in `OptimizableAgent`, see `introspect(agent)` auto-discover its knobs, score with `EvaluationHarness` + an offline `LLMJudge`, and optimize the prompt with `CoordinateAscentOptimizer`. |
| 4 | [`04_magentic_team_and_training.py`](./04_magentic_team_and_training.py) | A Magentic manager + 4 specialists built with `MagenticBuilder`, wrapped as ONE governed unit via `as_agent`, then trained as a whole (5 agents as `components` + explicit ROUTING knobs) and per-agent, with tool/skill ablation, judge `recommendations`, and the YAML `run_training` path ([`magentic.train.yaml`](./magentic.train.yaml)). |

## Run them

```bash
python examples/microsoft_agent_framework/01_basic_guarded.py
python examples/microsoft_agent_framework/02_policy_observability_trust.py
python examples/microsoft_agent_framework/03_evaluate_and_optimize.py
python examples/microsoft_agent_framework/04_magentic_team_and_training.py
```

## What ADAPT-Agent introspects for a `ChatAgent`

`detect(agent)` returns `"microsoft_agent_framework"` and `introspect(agent)`
yields, per agent:

| Knob | `ParameterKind` | Source attribute |
|------|-----------------|------------------|
| `instructions` | `PROMPT` | `agent.instructions` |
| model | `MODEL` | `agent.chat_client.model_id` (or `model` / `ai_model_id` / `deployment_name`) |
| `temperature`, `top_p`, `max_tokens` | `HYPERPARAM` | `agent.chat_client.*` (or the agent itself) |
| `tools` | `TOOL` | `agent.tools` (drop-one ablation candidates) |
| `skills` | `SKILL` | `agent.skills` (drop-one ablation candidates) |

The Magentic **routing limits** (`max_round_count`, `max_stall_count`,
`max_reset_count`) and the **manager** are *not* auto-discovered, and the
`as_agent()` wrapper exposes no knobs at all -- example 4 shows how to declare
those as explicit `ROUTING` parameters whose setter rebuilds the workflow.

See the full teaching page: [`docs/frameworks/microsoft_agent_framework.md`](../../docs/frameworks/microsoft_agent_framework.md).
