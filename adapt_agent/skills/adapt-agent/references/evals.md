# Evals reference

Full surface for scoring an agent against a golden dataset. The headline call
is `evaluate_agent`; everything below it is available separately when you need
finer control.

## `evaluate_agent`

```python
from adapt_agent.evaluation import evaluate_agent

report = evaluate_agent(
    agent,
    data,
    *,
    metrics=None,                 # what to score with (see "Choosing metrics")
    judge=None,                   # LLMJudge | provider name | prompt->text callable
    judge_criteria=None,          # task-level grading criteria
    judge_rubric=None,            # replace the default grading rubric
    primary_metric=None,          # headline metric name (default: the first)
    input_adapter="auto",         # "auto" | callable | None
    output_extractor=extract_output_text,   # None to score raw outputs
    input_key=None,               # explicit dataset column names
    expected_key=None,
    capture_output=True,          # store outputs on the report
    max_results=10_000,           # cap stored per-example records
    failure_threshold=1.0,        # cutoff used by report.failures()
)
```

`agent` may be a framework object (LangGraph compiled graph, Microsoft
`ChatAgent`, Pydantic AI `Agent`, CrewAI `Crew`, …), a governed adapter, an
`OptimizableAgent`, or any callable — including the one built by `adk_runner`.

`data` may be a `GoldenDataset`, an iterable of dicts / `Example` objects, or a
`.json` / `.jsonl` / `.csv` path.

### The report

```python
report.score              # aggregate of the primary metric, in [0, 1]
report.aggregate          # {metric_name: mean_score}
report.n, report.n_errors # example count / how many raised
report.avg_latency        # mean seconds per example
report.results            # list[ExampleResult]: .index .inputs .output .expected .scores .latency .error
report.failures()         # rows below failure_threshold (or errored)
report.below("judge", 0.5)  # rows below a threshold on one metric, errors not forced in
report.to_dict()          # JSON-friendly summary
```

For continuous metrics set `failure_threshold=0.6` (or pass
`report.failures(threshold=0.6)`), otherwise every imperfect row counts as a
failure.

## Dataset format

Records are dicts. Column names are auto-detected:

- input: `inputs`, `input`, `question`, `prompt`, `query`, `request`
- expected: `expected`, `expected_output`, `answer`, `label`, `target`, `gold`

`output` is deliberately **not** an expected-key: it conventionally holds a
model's own answer, and auto-mapping it would silently treat a prediction as
ground truth. Pass `expected_key="output"` if a file really stores gold there.

Any other keys become `Example.metadata` — which is where `check` and
`criteria` are read from.

```jsonl
{"input": "What is the capital of France?", "expected": "Paris"}
{"input": "What is 6 * 7?", "expected": 42, "check": "numeric_close"}
{"input": "Price it", "expected": 100, "check": {"name": "numeric_close", "tolerance": 5}}
{"input": "Name a founder", "expected": "Belgium", "check": ["contains", "token_f1"]}
{"input": "Write a greeting", "check": "judge", "criteria": "Warm, one sentence."}
```

Build datasets in code with `GoldenDataset.from_list / from_json / from_jsonl /
from_csv`, then `.split(0.8, seed=0)`, `.sample(50, seed=0)`, `.filter(pred)`,
`.shuffled(seed)` — all deterministic when seeded.

## Built-in checks

Each maps `(output, expected)` to `[0, 1]`.

| Name | Behaviour | Options |
| --- | --- | --- |
| `exact_match` | equality after normalising case/whitespace/punctuation | `normalize=True` |
| `contains` | expected is a substring of output | `normalize=True` |
| `regex_match` | output matches a regex (expected value, or a fixed one) | `pattern=None`, `flags=re.IGNORECASE` |
| `numeric_close` | first number in output within tolerance of expected's | `tolerance=1e-6`, `relative=False` |
| `token_f1` | SQuAD-style token-overlap F1 | — |
| `jaccard` | token-set Jaccard similarity | — |
| `json_subset` | fraction of expected dict key/values present in output dict | — |
| `levenshtein_ratio` | normalised edit-distance similarity | — |
| `checks` | per-row dispatch (see below) | `default`, `judge`, `aggregate` |

Import them from `adapt_agent.evaluation` and call the factory:
`numeric_close(tolerance=0.5, relative=True)`.

## Per-row checks

`checks()` is the default metric. It reads `example.metadata["check"]` (or
`"checks"`) and applies the named scorer to that row only:

- a string — a built-in name, or `"judge"` / `"llm_judge"`
- a mapping — `{"name": "numeric_close", "tolerance": 0.5}`; extra keys go to
  the factory
- a list — several checks combined with `min` (default: all must pass) or
  `mean` via `checks(aggregate="mean")`
- rows with no declaration use `default` (`exact_match` unless changed;
  `default=None` makes an undeclared row an error)

```python
from adapt_agent.evaluation import checks
evaluate_agent(agent, data, metrics=checks(default="token_f1", judge=my_judge))
```

## Choosing metrics

| `metrics=` | Behaviour |
| --- | --- |
| omitted | per-row `checks` (or the judge alone if only `judge=` is given) |
| `"exact_match"` / a `Metric` / a callable | that one metric on every row |
| `["contains", token_f1()]` | several metrics; the first is the headline |
| `{"accuracy": "exact_match"}` | mapping renames each metric in the report |
| `"judge"` | the judge grades every row (requires `judge=`) |

A supplied `judge=` is added as an extra `"judge"` metric grading every row —
**unless** the metrics already route it (an explicit `"judge"` entry or a
`checks` dispatcher, which judges only rows that ask for it). This is a cost
control: `metrics="checks"` with a judge spends judge calls only on
judge-declared rows. Use `metrics=["checks", "judge"]` for both.

Bare callables must accept `(output, expected)`; wrap in
`Metric(name, fn, needs_example=True)` to also receive the `Example`.

## LLM-as-judge

```python
from adapt_agent.evaluation import LLMJudge
from adapt_agent.optimization.judges import ClaudeJudge, OpenAIJudge, GeminiJudge

judge = LLMJudge("claude")                      # provider name
judge = LLMJudge(ClaudeJudge(model="claude-opus-4-8"))
judge = LLMJudge(lambda prompt: my_llm(prompt)) # any callable
judge = LLMJudge(my_provider, pass_threshold=0.7, scale=10, adversarial=True)
```

Providers: `anthropic`/`claude`, `openai`, `azure`, `gemini`/`google`,
`mistral`, `cohere`, `groq`, `together`, `openrouter`, `ollama`, `bedrock`,
`huggingface`. Each imports its SDK lazily; construct with
`get_judge(name, model=...)`.

Judge methods beyond scoring: `compare(a, b, swap=True)` for pairwise
preference with position-swap debiasing, `critique(...)` for actionable
feedback, `improve_prompt(current, failures)` for rewrites, `red_team(...)` and
`suggest_tools(...)` for adversarial analysis.

Safety properties worth keeping: agent text is fenced as untrusted data and
rubrics travel in the system prompt (prompt-injection hardening); a failed or
unparseable grading call returns `on_error` (default `0.0`) so a broken judge
never inflates a score; auth errors are re-raised loudly rather than swallowed.

In tests, pass a deterministic stub: a callable taking `(prompt, system=None)`
returning `'{"score": 9, "pass": true, "reasoning": "..."}'`.

## Output extraction

`extract_output_text(value)` unwraps framework-native results to final response
text. It is structural (no framework imports), recursive with bounded depth,
and conservative: strings pass through, `None` becomes `""`, and anything
unrecognised is returned **unchanged** so structured outputs still reach
`json_subset` or a custom metric.

Recognised: Pydantic AI `AgentRunResult` (`.output`/`.data`), OpenAI Agents
`RunResult` (`.final_output`), CrewAI `CrewOutput` (`.raw`), Claude Agent SDK
`ResultMessage`/content blocks, Microsoft `AgentRunResponse`/`ChatMessage`,
Google ADK events and GenAI `Content.parts[*].text`, LangChain messages
(including content-part lists), mappings with conventional keys, and
message/event streams (scanned from the end for the last text).

```python
from adapt_agent.evaluation import register_extractor
register_extractor("my_fw", lambda v: isinstance(v, MyResult), lambda v: v.completion)
```

Pass `output_extractor=None` to score raw outputs.

## Driving the harness directly

```python
from adapt_agent.evaluation import (
    EvaluationHarness, GoldenDataset, checks, extract_output_text, framework_runner,
)

harness = EvaluationHarness(
    [checks(judge=judge), my_metric],
    primary_metric="checks",
    output_extractor=extract_output_text,   # NOT applied unless you pass it
    failure_threshold=0.6,
    capture_output=True,
)
report = harness.evaluate(framework_runner(agent), GoldenDataset.from_jsonl("golden.jsonl"))
```

The harness is non-fatal: an exception on one example is recorded as a
zero-scored error rather than aborting the run. The same harness object is what
optimizers use, so an eval you trust becomes the optimization objective for
free.

## Per-framework notes

| Framework | How to pass it | What happens |
| --- | --- | --- |
| **LangGraph** | the compiled graph | plain strings wrapped into `{"messages": [{"role": "user", ...}]}`; final message text extracted. Dict inputs pass through — put graph-native state in the dataset for custom schemas, or override with `input_adapter=` |
| **Microsoft Agent Framework** | the `ChatAgent` | async `run()` awaited; `AgentRunResponse.text` scored |
| **Google ADK** | `adk_runner(agent_or_runner)` | builds an `InMemoryRunner` around a bare agent (needs `google-adk`), or drives a `Runner` you built; a **fresh session per example**; events drained, final text extracted. Customise with `user_id=`, `app_name=`, `message_factory=` |
| **Pydantic AI** | the `Agent` | `run_sync()` used; `.output` scored. A structured `output_type` survives extraction unchanged — score with `json_subset` |
| **CrewAI** | the `Crew` | `kickoff()` used; `CrewOutput.raw` scored |
| **OpenAI Agents SDK** | a callable driving `Runner` | `RunResult.final_output` scored |
| **Claude Agent SDK** | a callable / governed adapter | message stream drained; `ResultMessage.result` scored |

`framework_runner(agent)` performs the run-method discovery, async
materialisation, input adaptation and extraction on its own if you want the
plain `input -> text` callable without an eval.

Async agents are driven synchronously. Inside an already-running event loop
(notebook, async handler) that raises — run the eval in a worker thread.

## CLI

```bash
adapt-agent evaluate myapp.agents:agent --data golden.jsonl \
    --metric exact_match --extract-output

adapt-agent evaluate "myapp.agents:build()" --data golden.jsonl \
    --metric checks --judge claude --extract-output --json
```

Target is `module:attribute` (`()` calls a factory). `--metric` repeats;
`--extract-output` enables framework unwrapping; `--metric judge` grades every
row. The same flags work on `adapt-agent optimize`.
