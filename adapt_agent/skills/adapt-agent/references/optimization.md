# Optimization reference

Turning an agent into a tunable search space and improving it against a golden
dataset. Optimization reuses the eval stack — the harness that scores your
agent becomes the objective the optimizer maximises.

## The shape of a run

```python
from adapt_agent.optimization import (
    GoldenDataset, EvaluationHarness, OptimizableAgent,
    CoordinateAscentOptimizer, exact_match,
)
from adapt_agent.optimization.judges import get_judge

data = GoldenDataset.from_jsonl("golden.jsonl")
train, holdout = data.split(0.8, seed=0)

judge = get_judge("claude")
harness = EvaluationHarness([exact_match(), judge.as_metric("quality")],
                            primary_metric="quality")

target = OptimizableAgent.from_agent(my_agent)
result = CoordinateAscentOptimizer(harness, judge=judge, max_evals=60, seed=0).optimize(
    target, train, val_dataset=holdout
)

result.baseline_score, result.best_score, result.improvement
result.improved             # True only when the gain was statistically significant
result.significant, result.p_value, result.delta   # the evidence behind `improved`
result.best_config          # {parameter_name: value}, already applied in place
result.history              # every Trial evaluated, in order
result.recommendations      # advisory notes (small sample, untuned knobs, judge independence, tools)
```

The winning configuration is applied to the **live** objects, so the agent your
application holds is improved in place. Snapshot/restore first if you need to
undo: `snap = target.snapshot()` … `target.restore(snap)`.

**Read `result.improved`, not `result.improvement`.** `improvement` is a
difference of means; `improved` says whether it was more than noise (next section).
A script that asserted a gain on a toy dataset now sees `improved=False`.

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
| Pydantic AI | instructions (read and written through `SourcedInstruction`), `model`, `model_settings` (temperature/top_p/max_tokens), function tools |
| Microsoft Agent Framework | `instructions`, client model + settings, `tools`; Magentic/workflow routing limits. Reads both layouts: `.chat_client`/`.instructions`, and current releases' `.client` + `default_options["instructions"]` |
| Google ADK | `instruction`, `model`, `generate_content_config` (temperature/top_p/max_output_tokens), `tools`, `sub_agents` (recursive); workflow agents and `Workflow` recognised. The deprecated `global_instruction` is no longer offered |
| LangGraph | prompt, bound chat model and tools for `create_react_agent` and `create_agent` graphs; best-effort structural walk of `nodes` otherwise |
| CrewAI | agent `role`/`goal`/`backstory`, `llm`, task descriptions, `tools` (writes survive kickoff re-rendering) |
| OpenAI Agents SDK | `instructions`, `model`, `model_settings`, `tools`, `handoffs`; agents behind `handoff()` and `as_tool()` |
| Claude Agent SDK | options `system_prompt`, `model`, `effort`, tools. **`permission_mode` and `disallowed_tools` are read-only** and tool lists can only be *narrowed*, so an optimiser can never select a looser permission setting |

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
`max_evals`, `seed`, `judge`, `verbose`, plus the acceptance options below.

### What "improved" means

A candidate replaces the current best only when it wins a **paired statistical
test**, never because its mean is higher: an agent that is right 70% of the time,
scored twice on ten rows, differs by 0.2 about a third of the time with nothing
changed. Every comparison works on per-row deltas (candidate minus incumbent, same
row) and uses an exact sign-flip test (`adapt_agent.optimization.significance`, no
SciPy).

| Option | Default | Meaning |
| --- | --- | --- |
| `alpha` | `0.05` | one-sided significance level a gain must meet |
| `min_improvement` | `1e-3` | minimum mean per-row gain (the effect size) |
| `repeats` | auto | re-measure each comparison this many times (incumbent and challenger alternately). Default: once, or **3 when the baseline is found non-deterministic** (it is measured twice at the start). Setting it skips detection |
| `detect_nondeterminism` | `True` | the extra baseline pass |
| `holdout_fraction` | `0.25` | share held out for the final check when no `val_dataset` is given; needs `min_holdout_rows` (default 10) on each side, i.e. from about 38 rows |
| `allow_small_sample` | `True` | decide under-powered comparisons on the point estimate |

* **The final verdict needs a held-out gain.** With a `val_dataset` (or the
  automatic split) the winner is applied only if it does *better* than the untuned
  agent there. Smaller sets are judged on the training rows, and the result says
  that flatters the winner.
* **Small samples report what they are.** With `n` rows the strongest evidence
  (every row improves) has `p = 0.5**n`, so below five rows nothing can be
  significant at `alpha=0.05`. By default the best guess is still applied so a toy
  set can be tuned, but `improved=False`, `significant=False` and a `Small sample:`
  note appears in `recommendations`. `allow_small_sample=False` applies nothing
  that is not significant.
* **Evidence travels with the result**: `significant`, `delta`, `p_value`,
  `ci_low` / `ci_high`, `n_rows`, `repeats`, `alpha`, `significance_basis`
  (`"held_out"`, `"held_out_auto"`, `"training"`), `nondeterministic`,
  `untuned_parameters`, `rejected_reason`; each `Trial` has its own `delta`,
  `p_value`, `significant`. A non-deterministic target costs `2 * repeats` passes
  per trial.
* **`validation_tolerance` is inert** (a `FutureWarning`): the gate requires a gain,
  not a bounded loss.
* **No knob is skipped silently.** The grid sweeps knob by knob when the full
  product exceeds the budget; any parameter no trial ever varied is warned about,
  listed in `result.untuned_parameters` and in `recommendations`. `max_evals`,
  `repeats` and similar budgets must be positive integers (a `ValueError`
  otherwise). The search RNG is re-seeded on every `optimize()`, so a run is
  reproducible from `seed`.

### The validation gate

Pass `val_dataset=` to score the winner on held-out data and catch overfitting. It
is a gate, not just a report: a winner that shows **no gain over the untuned agent**
on the held-out set is **not applied**: `result.validation_rejected` is `True`,
`result.best_config` is empty, the discarded values are in `result.rejected_config`
and the reason in `result.rejected_reason`. Check that flag before telling a user
their agent was improved. `validation_score` is the applied configuration's
held-out score and `validation_baseline_score` the untuned agent's. An empty
`val_dataset` raises `ValueError`; `validation_gate=False` turns the gate off. A
few-shot block is never scored on the rows it was copied from.

If one `LLMJudge` both rewrites prompts and scores them, pass a second judge on
a different model as `proposer_judge=` (on any optimizer or
`make_default_optimizer`). Otherwise `result.recommendations` gains an
`Independence:` entry: the reported gain may partly be the judge preferring its
own writing.

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
target:
  entrypoint: "myapp.app:run"            # callable input -> output
  components:                            # introspected for tunable knobs
    researcher: "myapp.agents:researcher"
    writer: "myapp.agents:writer"
    workflow: "myapp.app:workflow"       # every `component:` below must be here
dataset:
  path: "golden.jsonl"
  format: jsonl
  val_path: "holdout.jsonl"              # optional held-out split
judge:
  provider: anthropic                    # a registered provider name
  model: claude-opus-5-5
  adversarial: true                      # the judge IS the adversary
metrics: [exact_match, token_f1]
primary_metric: exact_match
optimizer:
  type: default                          # NOT `name`
  max_evals: 60
  seed: 0
  suggest_tools: true                    # let the judge propose new tools
  alpha: 0.05                            # significance level (optional)
  repeats: 3                             # re-measure each comparison (optional)
  holdout_fraction: 0.25                 # automatic held-out split (optional)
parameters:                              # knobs no framework exposes
  - name: workflow.max_round_count
    kind: routing
    component: workflow
    attr: max_round_count
    bounds: [4, 12]
    step: 1
```

Schema notes worth getting right, because each one raises or is silently
ignored: `target` is a **mapping** with `entrypoint` (a bare string raises
`TrainingConfigError`); the optimizer key is `type`, not `name`; a holdout set
is a separate `dataset.val_path` file rather than a `split` fraction; and
`judge.provider` takes a registered provider (`anthropic`, `openai`, `gemini`,
…). `optimizer:` also takes `min_improvement`, `allow_small_sample` and
`detect_nondeterminism`; a setting the default pipeline cannot honour is refused
rather than ignored. The `judge:` section accepts only `provider`, `model`,
`adversarial`, `scale`, `pass_threshold`, `score_is_normalized`, `criteria`,
`metric_name`, `rubric`, `max_failures`, `on_error`, `temperature`, `max_tokens` and
`system`; `base_url` and `api_key` are refused (set keys in the provider's
environment variable). **A config file is trusted code**: `target:` entries are
imported and may be called. Every `parameters[].component` must also name a component declared under
`target.components`, or `adapt-agent train` fails at build time with
`references unknown component` — the config parses either way, so this one
only surfaces when you run it.

```bash
adapt-agent train train.yaml
adapt-agent train train.yaml --save-config tuned.yaml     # or .json; same as optimize
```

Or from Python: `load_training_config(path)` → `run_training(config)`. Unknown
metric/provider/optimizer names raise `TrainingConfigError`; hyperparameter
bounds outside a provider's allowable range are clamped with a warning rather
than crashing.

## CLI

```bash
adapt-agent optimize "myapp.agents:build()" --data golden.jsonl --val-data holdout.jsonl \
    --metric token_f1 --judge openai --optimizer default --max-evals 60 \
    --alpha 0.05 --repeats 3 --holdout-fraction 0.25 \
    --extract-output --save-config best.yaml

adapt-agent optimize myapp.app:orchestrate \
    --component researcher=myapp.agents:researcher \
    --component writer=myapp.agents:writer \
    --data golden.jsonl --metric checks --judge gemini
```

With `--component`, the positional target (or `--runner`) is the entrypoint
that drives the system and the components supply the tunable knobs.

`--alpha`, `--repeats` and `--holdout-fraction` set the acceptance options above
(`--holdout-fraction` applies only when `--val-data` is not given). The run prints
whether the gain is significant, and `--json` reports `improved`, `baseline_score`,
`best_score`, `improvement`, `validation_score`, `n_evals` and `best_config`; on a
toy dataset expect `improved: false`.

`--save-config FILE` writes the winner where `load_tuned_config` reads it back:
`.json` as flat `{"component.knob": value}`, `.yaml` / `.yml` nested by component.
Values that cannot round-trip (live tool lists, callables, SDK objects) are **not
written**: a stand-in string would break the agent when re-applied. They are named
under `_provenance.unexportable` (JSON) or a comment header (YAML) and on the
console. `adapt-agent train --save-config` does the same.

## Practical notes

- **Measure before you tune.** Run an eval first; if the baseline is already
  perfect on your data, the dataset is too easy to optimize against.
- **Hold data back.** `split(0.8)` plus `val_dataset=` is the difference
  between a real improvement and memorised training rows.
- **Read-only knobs are skipped**, not fatal: a discovered parameter whose
  attribute cannot be written is logged and marked non-optimizable.
- **A skipped knob has a fallback, not a silence.** Whatever no trial varied is
  listed in `result.untuned_parameters` and in the run's warnings: look there before
  concluding a prompt "could not be improved".
- **Default model IDs live in `adapt_agent.optimization.providers`**
  (`ANTHROPIC_DEFAULT_MODEL = "claude-opus-5-5"`, `OPENAI_DEFAULT_MODEL`,
  `GEMINI_DEFAULT_MODEL`, `DEFAULT_MODELS`). Pass `model=` to pin one; the default
  can move with a library upgrade.
- **Provider and judge errors are not scores.** A truncated, refused or empty judge
  response, an out-of-range or NaN score, and a configuration error surface as
  errors rather than a plausible number; SDK clients are built with
  `max_retries=0` so the library's own retry is the only one.
- **Budget is per-run**: `max_evals` counts harness evaluations, each of which
  runs the whole dataset — reduce with `dataset.sample(n, seed=0)` while
  iterating.

## Exporting the winning configuration

`result.best_config` is applied to the live components in place and dies with
the process. For a project whose prompts are version-controlled, that is the
wrong end state — and an LLM-rewritten prompt reaching production without a
human reading the diff is worse than losing a tuning run.

```python
result = CoordinateAscentOptimizer(harness, judge=judge).optimize(target, dataset)
result.to_config("specialists/.config/tuned.yaml")
```

The file nests by component, because parameter names follow
`"<component>.<knob>"`:

```yaml
# Tuned by adapt-agent.
# baseline=0.6300 best=0.9100 improvement=+0.2800 validation=0.88 over 60 evals in 412.0s.
# significant=True p=0.0004 n_rows=40 repeats=1 basis=held_out.
# Review this diff before committing: prompts here were machine-written.
researcher:
  system_prompt: |
    You are a careful researcher...
  temperature: 0.2
writer:
  model: gpt-4o
```

So the loop becomes **optimize → diff → review → commit**, and the application
keeps loading prompts from the YAML it always did. `header=False` drops the
comment block; `default_component=` names the section for a parameter with no
`.` prefix. The header records whether the gain was significant; a configuration
the validation gate rejected is never exported.

Round-trip it with `load_tuned_config`, which flattens the file back into the
form `OptimizableAgent.apply` expects:

```python
from adapt_agent.optimization import load_tuned_config

target.apply(load_tuned_config("specialists/.config/tuned.yaml"))
```
