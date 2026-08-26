# Optimization & Evaluation

ADAPT-Agent can **evaluate** any agent against a golden dataset and **optimize**
it -- automatically tuning prompts, few-shot examples, models, hyperparameters,
routing/topology knobs, and tool allow-lists -- regardless of how the agent is
built or which framework it uses.

The same machinery works for a single "mega" agent, six specialist agents, an
orchestrator delegating to sub-agents, or a multi-step workflow, even when the
code is spread across many files (a `src/` package, an `agents/` directory, a
FastAPI backend, ...). Each LLM SDK and agent framework is imported **lazily** --
only when you actually use one -- so you install just the providers and frameworks
your system needs.

> **Just want to score an agent?** [Running Evals](evals.md) covers the
> one-call `evaluate_agent(...)` API: deterministic text/number checks,
> per-row check selection, LLM-as-judge, and automatic unwrapping of
> framework-native outputs. The optimizers below reuse exactly those pieces.

## The pieces

| Concept | Class | Role |
| --- | --- | --- |
| Golden data | `GoldenDataset`, `Example` | inputs + expected outputs (`from_list`/`from_json`/`from_jsonl`/`from_csv`) |
| Tunable knob | `Parameter`, `SearchSpace` | a prompt / model / temperature / few-shot block / routing / tool knob bound to a live object |
| The target | `OptimizableAgent` | your agent code as `run(input)` + a discovered search space |
| Judge | `LLMJudge` (+ `ClaudeJudge`, `OpenAIJudge`, `GeminiJudge`, ...) | model-graded scoring **and** prompt improvement |
| Provider | `ModelProvider` (+ `AnthropicProvider`, `OpenAIProvider`, ...) | provider-agnostic model access |
| Metrics | `exact_match`, `token_f1`, `numeric_close`, `LLMJudge.as_metric()`, ... | scoring in `[0, 1]` |
| Measurement | `EvaluationHarness`, `EvaluationReport` | run the agent over the data, aggregate scores |
| Search | `CoordinateAscentOptimizer`, `GridSearchOptimizer`, `BootstrapFewShotOptimizer`, `EvolutionaryOptimizer`, `PipelineOptimizer` | strategies that find a better configuration |

## Quick start

```python
from adapt_agent.optimization import (
    GoldenDataset, EvaluationHarness, LLMJudge, OptimizableAgent,
    CoordinateAscentOptimizer, exact_match,
)
from adapt_agent.optimization.judges import ClaudeJudge

# 1. Golden dataset (any loader; keys like input/expected/answer are auto-detected)
data = GoldenDataset.from_jsonl("golden.jsonl")

# 2. Wrap your agent. For a single framework object:
agent = OptimizableAgent.from_agent(my_pydantic_ai_agent)
print([p.name for p in agent.parameters])   # discovered knobs

# 3. An LLM judge (provider-agnostic). Used for scoring AND prompt rewrites.
judge = ClaudeJudge(model="claude-opus-4-8")        # ANTHROPIC_API_KEY from env
harness = EvaluationHarness(
    metrics=[exact_match(), judge.as_metric("quality")],
    primary_metric="quality",
)

# 4. Optimize. The best configuration is applied to the live agent in place.
result = CoordinateAscentOptimizer(harness, judge=judge).optimize(agent, data)
print(result)               # baseline vs best, improvement, #evals
print(result.best_config)   # the winning prompt/model/... values
```

## Wrapping any architecture

`OptimizableAgent` separates **how to run** the system from **what to tune**.

* **Single agent / single framework object**

  ```python
  agent = OptimizableAgent.from_agent(my_crew)         # or graph, Agent, options...
  ```

  A framework-native run method (`run_sync`/`invoke`/`kickoff`/`run`) is
  resolved automatically. Frameworks whose agents are **not directly
  runnable** work too: an OpenAI Agents `Agent` is driven through the SDK's
  `Runner.run_sync`, a Claude Agent SDK options object through `query`, and a
  bare Google ADK agent inside a fresh-session `Runner` -- with results
  unwrapped to final response text (see
  `framework_runner` / `openai_agents_runner` / `claude_agent_runner` /
  `adk_runner` in `adapt_agent.optimization.runners`). Pass `runner=` to take
  over the driving yourself.

* **Many components (specialists, orchestrator + sub-agents), code in many files**

  ```python
  from my_app.agents import researcher, writer, editor   # wherever they live
  agent = OptimizableAgent.from_components(
      components={"researcher": researcher, "writer": writer, "editor": editor},
      runner=lambda q: my_orchestrator.handle(q),        # drives the whole system
      name="research-pipeline",
  )
  ```

  Each component is introspected for tunable knobs; because the `runner` closes
  over the *same* live objects, applying a candidate configuration changes what
  the next run does.

* **Custom / opaque knobs** the framework doesn't expose (routing thresholds,
  few-shot blocks, feature flags) are declared explicitly:

  ```python
  from adapt_agent.optimization import Parameter, ParameterKind
  agent.add_parameter(Parameter(
      name="router.threshold", kind=ParameterKind.ROUTING,
      bounds=(0.0, 1.0), getter=lambda: cfg.threshold,
      setter=lambda v: setattr(cfg, "threshold", v),
  ))
  ```

## Going deep into each framework

Introspection turns a live framework object into bound `Parameter`s. It is
**structural** (duck-typed) -- importing the introspectors never imports the
framework.

| Framework | What gets discovered |
| --- | --- |
| **CrewAI** | per-agent `role`/`goal`/`backstory` (prompt), `llm` model + temperature/top_p/max_tokens, `tools`, `max_iter`, `allow_delegation` (a searchable routing switch); per-task `description`/`expected_output` (a task's `name` becomes its component, so exported configs survive task reordering) |
| **OpenAI Agents SDK** | `instructions` and `handoff_description` (both prompts), `model`, `model_settings` (temperature/top_p/max_tokens), `tools`, `handoffs` (routing) -- recurses into handed-off sub-agents, and binds a `Handoff` wrapper's `tool_description` (the text the routing LLM reads) |
| **Google ADK** | `instruction`/`global_instruction`, `model`, `generate_content_config` (temperature/top_p/top_k/max_output_tokens), `tools`, `sub_agents` (routing) -- recurses |
| **Pydantic AI** | system prompt(s), `model`, `model_settings` (temperature/top_p/max_tokens, frequency/presence penalties), tools |
| **Microsoft Agent Framework** | `instructions`, model (a chat-client attribute, or `model_id` in `default_options`), temperature/top_p/max_tokens + frequency/presence penalties, tools, skills |
| **Claude Agent SDK** | `system_prompt` (str or preset `append`), `model`, `allowed_tools`/`disallowed_tools`, `max_turns`, `max_thinking_tokens`, `permission_mode`; each subagent definition in `agents` contributes its `prompt`/`description` (prompts), `model`, `tools` and `skills` |
| **LangGraph** | best-effort structural walk of compiled-graph nodes for prompts, bound model + temperature/top_p/max_tokens, tools (declare extra params for prompts buried in closures) |

```python
from adapt_agent.optimization.introspection import detect, introspect
detect(my_crew)        # -> "crewai"
introspect(my_crew)    # -> [Parameter(...), ...]
```

## LLM-as-judge at every stage

The judge is central. It scores outputs (as a metric), compares candidates, and
**rewrites prompts from failures** (driving the `LLMProposer`).

```python
from adapt_agent.optimization import LLMJudge
judge = LLMJudge(my_provider)                 # provider, callable, or name

verdict = judge.score("What is 2+2?", "4", "4")   # -> JudgeVerdict(score=..., passed=...)
judge.compare("Q", "answer A", "answer B")        # -> "A" | "B" | "tie"
judge.critique("Q", produced, expected)           # actionable feedback
judge.improve_prompt(current_prompt, failures)    # a rewritten instruction
metric = judge.as_metric("quality", criteria="be concise and correct")
```

### Provider-agnostic models

Every judge is backed by a `ModelProvider`. Use a provider-specific judge, or
build one from any provider / callable / registered name.

```python
from adapt_agent.optimization.judges import (
    ClaudeJudge, OpenAIJudge, GeminiJudge, MistralJudge, CohereJudge,
    GroqJudge, TogetherJudge, OllamaJudge, BedrockJudge, HuggingFaceJudge, get_judge,
)

judge = OpenAIJudge(model="gpt-4o-mini")          # OPENAI_API_KEY from env
judge = get_judge("gemini", model="gemini-2.0-flash")

# Or wrap your own client / a deterministic stub (great for tests):
from adapt_agent.optimization import LLMJudge, CallableProvider
judge = LLMJudge(CallableProvider(lambda prompt: "...your model call..."))
```

Built-in providers: `anthropic`, `openai`, `azure_openai`, `gemini`, `mistral`,
`cohere`, `groq`, `together`, `openrouter`, `ollama`, `bedrock`, `huggingface`
(each imports its SDK lazily), plus `callable` and `echo` for offline use.
Register your own with `register_provider(name, MyProvider)`.

## Optimizers

All optimizers share `optimize(agent, dataset, val_dataset=None)` and apply the
best configuration to the live agent before returning.

* **`CoordinateAscentOptimizer`** -- greedy per-parameter improvement via
  proposers; the flagship for prompt / few-shot tuning and where the LLM judge
  rewrites instructions. Restrict with `kinds=(ParameterKind.PROMPT,)`.
* **`BootstrapFewShotOptimizer`** -- coordinate ascent over few-shot blocks only.
* **`GridSearchOptimizer`** / **`RandomSearchOptimizer`** -- exhaustive / sampled
  search over discrete candidate sets and numeric bounds.
* **`EvolutionaryOptimizer`** -- population-based mutation + selection.
* **`PipelineOptimizer`** -- run several stages in sequence, threading the best
  configuration forward. `make_default_optimizer(harness, judge=...)` builds a
  "do all the optimizations" pipeline (few-shot → prompts → models/hyperparams).

```python
from adapt_agent.optimization import make_default_optimizer
result = make_default_optimizer(harness, judge=judge, max_evals=60).optimize(agent, data)
```

## Evaluation reports

`EvaluationHarness.evaluate(agent, dataset)` returns an `EvaluationReport`:

```python
report = harness.evaluate(agent, data)
report.score            # primary-metric aggregate
report.aggregate        # {metric_name: mean}
report.failures()       # examples below threshold (feed these to a proposer)
report.to_dict()        # JSON-friendly summary
```

See `examples/06_optimize_with_golden_dataset.py` for a complete, offline,
runnable walkthrough.

## Exporting the winning configuration

`result.best_config` is applied to the live components in place and dies with
the process. For a project whose prompts are version-controlled that is the
wrong end state — and a machine-rewritten prompt reaching production without a
human reading the diff is worse than losing a tuning run.

```python
result = CoordinateAscentOptimizer(harness, judge=judge).optimize(target, dataset)
result.to_config("specialists/.config/tuned.yaml")
```

Parameter names follow `"<component>.<knob>"`, so the file nests by component:

```yaml
# Tuned by adapt-agent.
# baseline=0.6300 best=0.9100 improvement=+0.2800 over 60 evals.
# Review this diff before committing: prompts here were machine-written.
researcher:
  system_prompt: |
    You are a careful researcher...
  temperature: 0.2
writer:
  model: gpt-4o
```

The loop becomes **optimize → diff → review → commit**, and the application
keeps loading prompts from the YAML it always did. Round-trip it with
`load_tuned_config`:

```python
from adapt_agent.optimization import load_tuned_config

target.apply(load_tuned_config("specialists/.config/tuned.yaml"))
```
