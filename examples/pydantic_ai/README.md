# Pydantic AI examples

Runnable examples that climb from a single guarded [Pydantic AI](https://ai.pydantic.dev)
`Agent` to a governed, optimizable multi-agent system, using ADAPT-Agent's two
halves: **Guard** (runtime security/observability) and **Train** (offline
optimization).

Every example runs **fully offline** -- they use Pydantic AI's `FunctionModel`
test double for the LLM and a deterministic stub for the LLM-as-judge, so no API
key or network is required. Each example also guards the `pydantic_ai` import and
prints a friendly install hint if it is missing.

## Install

```bash
pip install 'adapt-agent[pydantic-ai]'
# or just the framework:  pip install pydantic-ai
```

## The ladder

| # | File | What it teaches |
|---|------|-----------------|
| 1 | [`01_basic_guarded.py`](01_basic_guarded.py) | The smallest real Pydantic AI `Agent`, wrapped with a `Firewall` via `PydanticAIAdapter`. A safe input succeeds; a prompt-injection input raises `SecurityBlockedError`. |
| 2 | [`02_policy_observability_trust.py`](02_policy_observability_trust.py) | The full guard stack: `PolicyEnforcer` (a blocking rule), `AdversarialDefense`, `AgentObserver` traces, `Middleware` (pre/post hooks), and `TrustManager`. Shows `block_on_violation=False` recording threats without blocking. |
| 3 | [`03_evaluate_and_optimize.py`](03_evaluate_and_optimize.py) | Wrap one `Agent` in `OptimizableAgent`, build a `GoldenDataset`, score with `EvaluationHarness` (a metric + an offline `LLMJudge`), run `CoordinateAscentOptimizer`, and print the discovered knobs and the baseline -> best improvement. |
| 4 | [`04_multi_agent_and_training.py`](04_multi_agent_and_training.py) | A multi-agent **orchestrator function** driving two specialist `Agent`s. Guarded as one unit, then optimized whole-system + per-agent with `make_default_optimizer` (tool/skill ablation + judge `recommendations`), and a parallel YAML path via `run_training` + [`pydantic_ai.train.yaml`](pydantic_ai.train.yaml). |

## Run them

```bash
python examples/pydantic_ai/01_basic_guarded.py
python examples/pydantic_ai/02_policy_observability_trust.py
python examples/pydantic_ai/03_evaluate_and_optimize.py
python examples/pydantic_ai/04_multi_agent_and_training.py
```

## Going from offline to real

Each example notes the one-line swap from the offline `FunctionModel` to a hosted
model (e.g. `Agent("openai:gpt-4o")` or `Agent("anthropic:claude-opus-4-8")`) and
from the deterministic judge stub to a real provider
(`LLMJudge(ClaudeJudge(model="claude-opus-4-8"))`). Export the relevant API key
first.

See the full guide: [`docs/frameworks/pydantic_ai.md`](../../docs/frameworks/pydantic_ai.md).
