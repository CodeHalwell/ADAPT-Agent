# Optimization reference

Turning an agent into a tunable search space and improving it against a golden
dataset. Optimization reuses the eval stack — the harness that scores your
agent becomes the objective the optimizer maximises.

## The shape of a run

```python
from adapt_agent.optimization import (
    GoldenDataset, EvaluationHarness, LLMJudge, OptimizableAgent,
    CoordinateAscentOptimizer, exact_match,
)

data = GoldenDataset.from_jsonl("golden.jsonl")
train, holdout = data.split(0.8, seed=0)

judge = LLMJudge("claude")
harness = EvaluationHarness([exact_match(), judge.as_metric("quality")],
                            primary_metric="quality")

target = OptimizableAgent.from_agent(my_agent)
result = CoordinateAscentOptimizer(harness, judge=judge, max_evals=60, seed=0).optimize(
    target, train, val_dataset=holdout
)

result.baseline_score, result.best_score, result.improvement
result.best_config          # {parameter_name: value}, already applied in place
result.history              # every Trial evaluated, in order
result.recommendations      # advisory tool/skill suggestions from the judge
```

The winning configuration is applied to the **live** objects, so the agent your
application holds is improved in place. Snapshot/restore first if you need to
undo: `snap = target.snapshot()` … `target.restore(snap)`.

## Wrapping any architecture

```python
# One framework object — introspected for knobs, run method discovered
target = OptimizableAgent.from_agent(agent)

# A whole system: components supply knobs, the runner drives the pipeline
target = OptimizableAgent.from_components(
    components={"researcher": researcher, "writer": writer},
    runner=lambda q: orchestrator.handle(q),
    name="research-writer",
)

# Just a callable, with knobs you declare yourself
target = OptimizableAgent.from_callable(run, parameters=[my_param])
```

Because the runner is opaque, this works identically for a single agent, six
specialists, an orchestrator with sub-agents, or a multi-step workflow spread
across files — as long as the runner closes over the live component objects.

Inspect what was found: `target.describe()`, `target.parameters`,
`target.parameters_of_kind(ParameterKind.PROMPT)`.

## What gets introspected

`ParameterKind`: `PROMPT`, `FEW_SHOT`, `MODEL`, `HYPERPARAM`, `ROUTING`,
`TOOL`, `SKILL`.

| Framework | Discovered knobs |
| --- | --- |
| Pydantic AI | `_system_prompts` / `system_prompt`, `model`, `model_settings` (temperature/top_p/max_tokens), function tools |
| Microsoft Agent Framework | `instructions`, chat-client model + settings, `tools`; Magentic/workflow routing limits |
| Google ADK | `instruction`, `global_instruction`, `model`, `generate_content_config` (temperature/top_p/max_output_tokens), `tools`, `sub_agents` (recursive) |
| LangGraph | best-effort structural walk of `nodes` → node runnable prompts, bound chat models, tool lists |
| CrewAI | agent `role`/`goal`/`backstory`, `llm`, task descriptions, `tools` |
| OpenAI Agents SDK | `instructions`, `model`, `model_settings`, `tools`, `handoffs` |
| Claude Agent SDK | options `system_prompt`, `model`, `allowed_tools` |

Introspection duck-types everything and never imports the framework. It is
necessarily incomplete — prompts buried in closures cannot be reached
structurally. Declare those explicitly:

```python
from adapt_agent.optimization import Parameter, ParameterKind

target.add_parameter(Parameter(
    "router.threshold", ParameterKind.ROUTING,
    value=0.5, bounds=(0.0, 1.0), step=0.1,
    getter=lambda: cfg["threshold"], setter=lambda v: cfg.__setitem__("threshold", v),
))
```

Tools/skills become a real search space with drop-one ablation:

```python
target.add_tool_parameter(
    "researcher.tools",
    getter=lambda: researcher.tools,
    setter=lambda ts: setattr(researcher, "tools", ts),
    candidate_tools=[search, calculator, browser],   # full set + each drop-one subset
)
```

## Optimizers

| Class | Strategy | Good for |
| --- | --- | --- |
| `CoordinateAscentOptimizer` | tune one knob at a time, keep improvements | the default workhorse |
| `GridSearchOptimizer` | exhaustive over candidate lists | few discrete knobs |
| `RandomSearchOptimizer` | sample the space | many knobs, small budget |
| `BootstrapFewShotOptimizer` | build few-shot blocks from successful examples | prompt-heavy tasks |
| `EvolutionaryOptimizer` | population + mutation/crossover | large, rugged spaces |
| `PipelineOptimizer` | run several strategies in sequence | "just make it better" |

`make_default_optimizer(harness, judge=..., max_evals=..., seed=..., verbose=...)`
returns a sensible pipeline. All share
`optimize(target, dataset, val_dataset=None)` and the constructor kwargs
`max_evals`, `seed`, `judge`, `verbose`.

Pass `val_dataset=` to score the winner on held-out data and catch overfitting
to the training split.

## Proposers

Optimizers generate candidates through proposers, which you can mix:
`CandidateProposer` (enumerate declared candidates), `NumericProposer` (bounded
numeric steps), `PromptMutationProposer` (textual mutations), `FewShotProposer`
(bootstrap examples from data), `LLMProposer` (judge rewrites the instruction
from observed failures), `ToolAblationProposer` (drop-one tool subsets),
`LLMToolProposer` (judge proposes *new* tools). `default_proposers(judge=...)`
assembles the standard set.

The `LLMProposer` loop is the interesting one: it takes
`report.failures()` from the harness, asks the judge to critique them, and asks
for a rewritten instruction that fixes those failures without hard-coding the
answers.

## The judge in optimization

Beyond scoring, the judge drives improvement: `improve_prompt(current,
failures)` for rewrites, `compare(a, b, swap=True)` for reference-free
selection between candidates, `critique(...)` for the feedback fed into
rewrites, and — with `LLMJudge(adversarial=True)` — harsher grading that
reserves high scores for genuinely complete answers. `red_team(input, output)`
and `suggest_tools(component, failures, current_tools)` produce advisory
findings surfaced as `result.recommendations`.

## Declarative training config

One file describes the whole run:

```yaml
target: myapp.agents:build()
components:
  researcher: myapp.agents:researcher
  writer: myapp.agents:writer
dataset:
  path: golden.jsonl
  split: 0.8
  seed: 0
metrics: [exact_match]
judge:
  provider: claude
  model: claude-opus-4-8
optimizer:
  name: default
  max_evals: 60
  seed: 0
save_config: best.json
```

```bash
adapt-agent train train.yaml
```

Or from Python: `load_training_config(path)` → `run_training(config)`. Unknown
metric/provider/optimizer names raise `TrainingConfigError`; hyperparameter
bounds outside a provider's allowable range are clamped with a warning rather
than crashing.

## CLI

```bash
adapt-agent optimize "myapp.agents:build()" --data golden.jsonl \
    --metric token_f1 --judge openai --optimizer default --max-evals 60 \
    --extract-output --save-config best.json

adapt-agent optimize myapp.app:orchestrate \
    --component researcher=myapp.agents:researcher \
    --component writer=myapp.agents:writer \
    --data golden.jsonl --metric checks --judge gemini
```

With `--component`, the positional target (or `--runner`) is the entrypoint
that drives the system and the components supply the tunable knobs.

## Practical notes

- **Measure before you tune.** Run an eval first; if the baseline is already
  perfect on your data, the dataset is too easy to optimize against.
- **Hold data back.** `split(0.8)` plus `val_dataset=` is the difference
  between a real improvement and memorised training rows.
- **Read-only knobs are skipped**, not fatal: a discovered parameter whose
  attribute cannot be written is logged and marked non-optimizable.
- **Budget is per-run**: `max_evals` counts harness evaluations, each of which
  runs the whole dataset — reduce with `dataset.sample(n, seed=0)` while
  iterating.
