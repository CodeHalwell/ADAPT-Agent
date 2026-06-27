# LangGraph examples

A ladder of runnable examples that wrap and optimize [LangGraph](https://langchain-ai.github.io/langgraph/)
agents with ADAPT-Agent. `LangGraphAdapter` wraps a **compiled** graph (anything
with a callable `invoke(state) -> state`); the optimization layer treats the same
graph as a tunable search space.

Install the extra first:

```bash
pip install 'adapt-agent[langgraph]'
```

Each script guards the `langgraph` import and prints a friendly hint if it is not
installed. The examples themselves run **offline** (plain-Python nodes, a
deterministic judge stub) — no model or API key required.

| # | File | What it shows |
|---|------|---------------|
| 1 | [`01_basic_guarded.py`](01_basic_guarded.py) | Smallest one-node graph wrapped with a `Firewall`; a safe input passes, a prompt-injection input raises `SecurityBlockedError` before `invoke`. |
| 2 | [`02_policy_observability_trust.py`](02_policy_observability_trust.py) | Full guard stack — `PolicyEnforcer` (state-based rule), `AdversarialDefense`, `AgentObserver`, `Middleware`, plus standalone `TrustManager`/`TaintTracker` — in `block_on_violation=False` monitor mode. |
| 3 | [`03_evaluate_and_optimize.py`](03_evaluate_and_optimize.py) | Evaluate a graph against a `GoldenDataset` and optimize its prompt with a `CoordinateAscentOptimizer` + offline `LLMJudge`. Shows the declare-an-explicit-`Parameter` pattern LangGraph needs for closure prompts. |
| 4 | [`04_multi_agent_and_training.py`](04_multi_agent_and_training.py) | A supervisor/router graph (geography + math specialists) governed and optimized as one unit with `make_default_optimizer` (tool/skill ablation + adversarial-judge `recommendations`), plus the declarative `run_training` path. |

Run any of them:

```bash
python examples/langgraph/01_basic_guarded.py
python examples/langgraph/02_policy_observability_trust.py
python examples/langgraph/03_evaluate_and_optimize.py
python examples/langgraph/04_multi_agent_and_training.py
```

See [`langgraph.train.yaml`](langgraph.train.yaml) for the file-based declarative
training config, and [`docs/frameworks/langgraph.md`](../../docs/frameworks/langgraph.md)
for the full guide.
