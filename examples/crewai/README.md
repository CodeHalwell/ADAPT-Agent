# CrewAI examples

Runnable examples pairing [CrewAI](https://docs.crewai.com) with ADAPT-Agent's
two halves: the **runtime guard** (firewall, policy, observability, middleware)
and the **offline trainer** (introspection, evaluation, optimization).

CrewAI is multi-agent by construction: you assemble a `Crew` of `Agent` objects
working through a list of `Task` objects and run it with `crew.kickoff(inputs=...)`.
The `CrewAIAdapter` wraps a whole `Crew` as one governed unit, and the CrewAI
*introspector* flattens that crew into per-agent and per-task tunable knobs the
optimizer can search.

## Install

```bash
pip install 'adapt-agent[crewai]'   # or: pip install crewai
```

Every example **guards the `crewai` import**: run any of them without CrewAI
installed and you get a friendly install hint instead of a traceback. The
`adapt_agent` parts always import. All examples run **offline with no API key** --
the crew model and the LLM judge are deterministic local stubs.

## The ladder

| # | File | What it teaches |
|---|------|-----------------|
| 1 | [`01_basic_guarded.py`](01_basic_guarded.py) | Smallest real crew (one agent, one task) wrapped with a `Firewall`; safe input succeeds, a prompt-injection input raises `SecurityBlockedError`. |
| 2 | [`02_policy_observability_trust.py`](02_policy_observability_trust.py) | Add `PolicyEnforcer` (a block rule), `AdversarialDefense`, `AgentObserver` (traces), and `Middleware`; `block_on_violation=False` records threats without blocking. |
| 3 | [`03_evaluate_and_optimize.py`](03_evaluate_and_optimize.py) | Wrap one crew as an `OptimizableAgent`, score with `EvaluationHarness` (metric + offline `LLMJudge`), run `CoordinateAscentOptimizer`, print baseline→best and the introspected knobs. |
| 4 | [`04_multi_agent_and_training.py`](04_multi_agent_and_training.py) | A two-agent crew (researcher + writer with tools) governed as one unit, optimized with `make_default_optimizer` (tool/skill ablation + adversarial judge recommendations), plus the YAML-config path via `run_training` and [`crewai.train.yaml`](crewai.train.yaml). |

## Run them

```bash
python examples/crewai/01_basic_guarded.py
python examples/crewai/02_policy_observability_trust.py
python examples/crewai/03_evaluate_and_optimize.py
python examples/crewai/04_multi_agent_and_training.py
```

For the full prose walkthrough -- every constructor argument, the 6-step
pipeline, what gets introspected, and the multi-agent/YAML patterns -- see
[`docs/frameworks/crewai.md`](../../docs/frameworks/crewai.md).
