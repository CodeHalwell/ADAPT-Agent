# LangGraph examples

A ladder of runnable examples that wrap and optimize [LangGraph](https://langchain-ai.github.io/langgraph/)
agents with ADAPT-Agent. `LangGraphAdapter` wraps a **compiled** graph (anything
with a callable `invoke(state) -> state`); the optimization layer treats the same
graph as a tunable search space.

Install the extra first:

```bash
pip install 'adapt-agent[langgraph]'
```

The extra installs LangGraph and LangChain v1 (`langchain>=1.0`), which provides
`create_agent` and the middleware example 5 uses.

Each script guards its framework imports, prints a hint and exits 0 if they are
not installed. The examples themselves run **offline** (plain-Python nodes, a
scripted chat model, a deterministic judge stub) — no model or API key required.

| # | File | What it shows |
|---|------|---------------|
| 1 | [`01_basic_guarded.py`](01_basic_guarded.py) | Smallest one-node graph wrapped with a `Firewall`; a safe input passes, a prompt-injection input raises `SecurityBlockedError` before `invoke`. |
| 2 | [`02_policy_observability_trust.py`](02_policy_observability_trust.py) | Full guard stack — `PolicyEnforcer` (state-based rule), `AdversarialDefense`, `AgentObserver`, `Middleware`, plus standalone `TrustManager`/`TaintTracker` — in `block_on_violation=False` monitor mode. |
| 3 | [`03_evaluate_and_optimize.py`](03_evaluate_and_optimize.py) | Evaluate a graph against a `GoldenDataset` and optimize its prompt with a `CoordinateAscentOptimizer` + offline `LLMJudge`. Shows the declare-an-explicit-`Parameter` pattern LangGraph needs for closure prompts. |
| 4 | [`04_multi_agent_and_training.py`](04_multi_agent_and_training.py) | A supervisor/router graph (geography + math specialists) governed and optimized as one unit with `make_default_optimizer` (tool/skill ablation + adversarial-judge `recommendations`), plus the declarative `run_training` path. |
| 5 | [`05_tool_governance_middleware.py`](05_tool_governance_middleware.py) | Governance *inside* a `create_agent` agent with `governance_middleware`: a denied tool refused, a poisoned tool result withheld from the model, an approval `interrupt()` resumed with `Command(resume=...)`, `screen_stream=True` keeping a blocked response out of `stream_mode="messages"`, and a run-level callback handler covering an ungoverned sub-agent. |

Run any of them:

```bash
python examples/langgraph/01_basic_guarded.py
python examples/langgraph/02_policy_observability_trust.py
python examples/langgraph/03_evaluate_and_optimize.py
python examples/langgraph/04_multi_agent_and_training.py
python examples/langgraph/05_tool_governance_middleware.py
```

Each script prints what it ran, what passed, what
was blocked and why (the `SecurityBlockedError` threat labels), and exits 0. The
optimization examples print an `OptimizationResult` line such as
`baseline=0.200, best=0.900, ..., significant=False (p=0.0625, n=4)`: the gain is
applied, but on a handful of rows no outcome can pass the paired significance
test, so `improved` stays `False` -- use a larger golden set for a real verdict.

See [`langgraph.train.yaml`](langgraph.train.yaml) for the file-based declarative
training config, and [`docs/frameworks/langgraph.md`](../../docs/frameworks/langgraph.md)
for the full guide.
