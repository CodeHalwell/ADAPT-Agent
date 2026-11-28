# Evals reference

Full surface for scoring an agent against a golden dataset. The headline call
is `evaluate_agent`; everything below it is available separately when you need
finer control.

## `evaluate_agent`

```text
evaluate_agent(
    agent,
    data,
    *,
    metrics=None,                 # what to score with (see "Choosing metrics")
    judge=None,                   # LLMJudge | provider name | prompt->text callable
    judge_criteria=None,          # task-level grading criteria
    judge_rubric=None,            # replace the default grading rubric
    primary_metric=None,          # headline metric name (default: the first)
    input_adapter="auto",         # "auto" | callable | None
    output_extractor="auto",      # extract_output_text / _payload as the metrics need; None = raw
    input_key=None,               # explicit dataset column names
    expected_key=None,
    capture_output=True,          # store outputs on the report
    max_results=10_000,           # cap stored per-example records
    failure_threshold=1.0,        # cutoff used by report.failures()
    concurrency=1,                # rows in flight at once (threads)
    conversation_input="messages",  # how history reaches the agent on multi-turn rows
) -> EvaluationReport
```

`agent` may be a framework object (LangGraph compiled graph, Microsoft Agent
Framework `Agent`, Pydantic AI `Agent`, CrewAI `Crew`, …), a governed adapter, an
`OptimizableAgent`, or any callable — including the one built by `adk_runner`.

`data` may be a `GoldenDataset`, an iterable of dicts / `Example` objects, or a
`.json` / `.jsonl` / `.csv` path.

### The report

```python
report.score              # aggregate of the primary metric, in [0, 1]
report.aggregate          # {metric_name: mean_score}
report.n, report.n_errors # example count / how many errored (agent raised, or a metric failed)
report.n_transient_errors # rows dropped as provider throttling (excluded from the score)
report.n_metric_errors    # rows where a metric raised or returned NaN/inf
report.is_complete        # False when any row was dropped as transient
report.completeness()     # {"rows_total", "rows_scored", "rows_errored",
                          #  "rows_dropped_transient", "rows_metric_errors"}
report.avg_latency, report.latency_p50, report.latency_p95
report.usage              # summed tokens / cost_usd when the framework exposes them
report.results            # list[ExampleResult]: .index .inputs .output .expected .scores
                          #   .latency .error .metric_errors .usage .turns
report.failures()         # rows below failure_threshold (or errored)
report.below("judge", 0.5)  # rows below a threshold on one metric, errors not forced in
report.to_dict()          # JSON-friendly summary
```

For continuous metrics set `failure_threshold=0.6` (or pass
`report.failures(threshold=0.6)`), otherwise every imperfect row counts as a
failure.

**Read completeness before trusting a score.** `rows_scored` is the denominator
of `score` and *includes* errored rows (they score 0). A score over a subset (rows
dropped as transient) can pass a threshold the full dataset would miss; the CLI
labels such an aggregate `PARTIAL`.

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
| `exact_match` | equality after NFKC + case-folding + whitespace, with punctuation stripped **at token edges only** (`"Paris."` = `"paris"`, but `-5` ≠ `5`) | `normalize=True`, `punctuation="edge"` |
| `contains` | expected appears **on word boundaries** (`"no"` is not in `"know"`) | `normalize=True`, `word_boundary=None` |
| `regex_match` | output matches a regex (the row's expected value, or a fixed one). An empty pattern is a **row error**; patterns over 512 chars or with nested quantifiers are refused | `pattern=None`, `flags=re.IGNORECASE` |
| `numeric_close` | the *answer* number in the output (after the last `answer` / `result` / `total` / `=` marker, else the last number) within tolerance | `tolerance=1e-6`, `relative=False`, `which="auto"` |
| `token_f1` | SQuAD-style token-overlap F1 (CJK / Thai per character) | — |
| `jaccard` | token-set Jaccard similarity | — |
| `json_subset` | **recursive** fraction of expected keys matched; `True` ≠ `1`; fenced / embedded JSON is parsed | `strict_types=True` |
| `levenshtein_ratio` | normalised edit-distance similarity | — |
| `checks` | per-row dispatch (see below) | `default`, `judge`, `aggregate` |
| `field_match` | one field of a structured output, reported under its own name | `field`, `check`, `name`, `missing` |
| `tool_called`, `tool_call_args_match`, `tool_sequence`, `no_forbidden_tools`, `tool_call_budget` | **trajectory** checks on the tool calls the agent made (see below) | per metric |

Import them from `adapt_agent.evaluation` and call the factory:
`numeric_close(tolerance=0.5, relative=True)`.

### The defaults changed because the old ones scored wrong answers as right

| Metric | Now | The old behaviour, by option |
| --- | --- | --- |
| `exact_match` | edge punctuation only | every symbol erased, so `-5` = `5`, `C++` = `C`: `exact_match(punctuation="all")` |
| `contains` | word boundary | substring: `contains(word_boundary=False)` |
| `numeric_close` | `which="auto"` | the first number: `numeric_close(which="first")` (`"any"` passes if any number matches) |
| `regex_match` | empty pattern = row error | matched everything (1.0) |
| `json_subset` | recursive, strict bool/int | flat, loose types: `json_subset(strict_types=False)` |
| any metric | NaN / infinite score = **row error** | NaN was clamped to **1.0** |

A metric that raises, or returns NaN / infinity, scores that row with its
`on_error` fallback (0.0) and is recorded in `result.metric_errors` /
`report.n_metric_errors`. It is never silently a pass. If you pinned a golden
score on the old semantics, expect it to move.

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

`checks` is marked **structural**, so the output keeps its structure for a row
asking `field_match`; rows asking for a text check are flattened individually as
they are scored. Mixing `checks` with a plain text metric is the one case that
loses structure for the whole run — pass
`output_extractor=extract_output_payload` there, at the cost of the text metric.

## Scoring structured output

`extract_output_text` flattens a structured answer: a Microsoft
`AgentResponse` is a recognised shape, so it becomes `.text` and the object
is gone. `output_extractor=None` unwraps nothing at all and scores a `repr()`.
Neither is what a structured-output agent needs, so there is a third option:

```python
from adapt_agent.evaluation import extract_output_payload, field_metrics

report = evaluate_agent(agent, data, metrics=field_metrics(["lane", "matter", "action", "pack"]))
report.aggregate    # {"lane": 0.94, "matter": 0.90, "action": 0.63, "pack": 0.0}
```

`extract_output_payload` strips the framework envelope and keeps the payload: a
mapping or sequence passes through, a declared structured output (Pydantic model
or dataclass) becomes a dict, and a JSON string is parsed back into an object
(including a ```` ```json ```` fenced one, which is how most models emit
structured answers). Non-JSON text degrades to text rather than erroring.

Three shapes are easy to get wrong and each used to score 0.0 across the board.
A **LangGraph** graph built with `response_format=` returns a *state* — a real
run's keys are `["messages", "structured_response"]` — and the declared output
is peeled out of it. The state must be recognised as one: `add_messages` coerces
every entry to a `BaseMessage`, so a real `messages` is a non-empty list of
message objects, and an answer that merely has fields of those names keeps every
column. A **single-field answer** like `{"result": "granted"}` is the answer,
not an envelope: only a conventional key holding something *structured* is
peeled. A **list of records** is the answer too — only a message/event stream is
scanned from the end.

`extract_output_text` leaves a declared output unchanged, which is what lets a
per-row `field_match` still see its fields — on the `checks` path the row picks
the metric at runtime, so text extraction is the one in force.

`evaluate_agent` selects it automatically when **every** metric is structural
(`json_subset`, `field_match`). The all-or-nothing rule is deliberate: mixing a
text metric with a structural one keeps text, because a dict scores 0.0 against
`exact_match`. Pass either extractor explicitly to override.

`field_match(field, check="exact_match", name=None, missing=0.0, **options)`
scores one field, reported under the field's name — which is what turns
`aggregate` into a per-column table instead of one blended number that hides
which column moved. `field_metrics([...])` builds one per field. Options go to
the inner check: `field_match("total", check="numeric_close", tolerance=0.01)`.
Rows can request it too: `{"check": {"name": "field_match", "field": "lane"}}`.

A missing field scores `missing` (default `0.0`) — a field the agent failed to
emit is a failure, not a skip.

## Running examples concurrently

`evaluate` is serial by default. That is fine for a 20-row smoke test and
untenable for what optimization needs: `CoordinateAscentOptimizer(max_evals=60)`
over a 113-case split is 6,780 agent calls, each an LLM round-trip.

```python
report = harness.evaluate(agent, dataset, concurrency=8)          # sync agent: threads
report = await harness.aevaluate(agent, dataset, concurrency=8)   # async agent: no threads
```

`evaluate_agent(..., concurrency=8)` takes the same knob. Use `aevaluate` for an
async-native agent — it awaits in your loop, so `contextvars` (tracing spans)
survive and no thread is involved; use `evaluate(concurrency=)` for a sync agent
whose time goes on network I/O.

**On the optimizer path, set it on the harness instead.** An `Optimizer` calls
`harness.evaluate(target, dataset)` with no keyword arguments, so a per-call
`concurrency=` cannot reach the one path that needs it most:

```python
harness = EvaluationHarness(metrics, concurrency=8)   # the per-instance default
CoordinateAscentOptimizer(harness, judge=judge).optimize(target, dataset)
```

A per-call argument still wins where one is given.

Both preserve the guarantees of the serial path: results are reported in
**example-index order** regardless of completion order, a per-example exception
is still a non-fatal zero-scored error, and `max_results` still bounds stored
records while every example is aggregated. `avg_latency` stays the mean
*per-example* latency, not wall-clock/n, so it remains comparable across
concurrency settings.

### Throttling is retried, and never scored

Concurrency multiplies provider rate-limit pressure by the same factor: `N`
in-flight examples is `N` simultaneous LLM calls. Start at 4–8 and raise it
against your own quota.

Transient failures — 429, 5xx, timeouts, dropped connections — are retried with
jittered exponential backoff (three attempts by default), honouring a
`Retry-After` header when the provider sends one. A failure that outlives its
retries is recorded as `transient`, counted in `report.n_transient_errors`, and
then **left out of the score entirely**:

```python
report.score               # unaffected by throttling
report.n_errors            # includes transient failures
report.n_transient_errors  # how many of them were the provider's fault
report.failures()          # excludes them: not evidence about your prompt
```

This is a correctness property, not a convenience. Scoring a throttled example
zero makes it indistinguishable from a bad prompt, and the error is systematic
rather than random: whichever candidate is evaluated while the provider is
busiest scores lowest, so an optimizer can select a prompt for having been lucky
with rate limits. To judge whether a run is trustworthy at all, read
`report.is_complete` — or compare `n_transient_errors` against `n_evaluated`,
never against `n`. `n` counts *stored records* and is capped by `max_results`,
so on a large dataset it can read `n=1, n_transient_errors=4`.

Only errors classified transient are retried — a genuinely broken agent still
fails once and scores zero, as it should.

**An exception raised by the agent under test is a failed row, not a dropped
one**, unless something shows it came from infrastructure: a link in its cause
chain raised by a provider SDK or HTTP client package (`openai`, `anthropic`,
`google`, `httpx`, `requests`, …), an HTTP status attribute, or a status in its
message (`"Error code: 429"`). An agent's own `ToolTimeout("step budget
exceeded")`, or a bare `TimeoutError` from your tool, scores 0 and stays in the
denominator. (It used to be dropped, so an agent failing two rows of three on
its own timeouts printed `1.0000`.) A judge or metric call keeps the generous
classification, since its provider is the harness's own infrastructure. Tune or
replace the policy:

```python
from adapt_agent.optimization import RetryPolicy

EvaluationHarness(metrics, retry=RetryPolicy(attempts=5, max_backoff=60))
EvaluationHarness(metrics, retry=RetryPolicy(attempts=1))   # classify, don't retry
EvaluationHarness(metrics, retry=RetryPolicy(is_transient=my_classifier))
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
from adapt_agent.optimization.judges import ClaudeJudge, OpenAIJudge, GeminiJudge, get_judge

judge = get_judge("claude")                     # alias -> ClaudeJudge (default claude-opus-5-5)
judge = LLMJudge("anthropic")                   # or a registered provider name
judge = ClaudeJudge(model="claude-opus-5-5", pass_threshold=0.7)   # already an LLMJudge
judge = LLMJudge(lambda prompt, system=None: my_llm(prompt, system=system))  # any callable
judge = LLMJudge(my_provider, pass_threshold=0.7, scale=10, adversarial=True)
```

Pin `model=` for a reproducible eval: the default model IDs are `claude-opus-5-5`
(Anthropic), `gpt-5.6-luna` (OpenAI) and `gemini-3.5-flash` (Gemini), and a
library upgrade can move them. Give a callable a `system` parameter, or every
call grades without the rubric (a logged warning, not an error).

`LLMJudge(...)` takes a **registered provider name** — the aliases
(`claude`, `google`) are resolved by `get_judge()`, not by the provider
registry, so `LLMJudge("claude")` raises `KeyError`. `evaluate_agent(judge=...)`
goes through `get_judge()`, so `judge="claude"` is fine there.

Providers: `anthropic`/`claude`, `openai`, `azure`, `gemini`/`google`,
`mistral`, `cohere`, `groq`, `together`, `openrouter`, `ollama`, `bedrock`,
`huggingface`. Each imports its SDK lazily; construct with
`get_judge(name, model=...)`.

Judge methods beyond scoring: `compare(a, b, swap=True)` for pairwise
preference with position-swap debiasing, `critique(...)` for actionable
feedback, `improve_prompt(current, failures)` for rewrites, `red_team(...)` and
`suggest_tools(...)` for adversarial analysis.

Safety properties worth keeping: agent text is fenced as untrusted data and
rubrics travel in the system prompt (prompt-injection hardening); a broken judge
never inflates a score.

**What counts as no verdict.** A score outside the scale, NaN or infinity is an
error, not a clamped 1.0. A truncated, refused or empty completion raises a typed
error (`TruncatedResponseError`, `RefusedResponseError`, `EmptyResponseError`,
all `ProviderResponseError` in `adapt_agent.optimization.providers`) that the
judge records as an error rather than a grade. Through `as_metric` (what
`evaluate_agent(judge=...)` uses) any of these raises `JudgeError` (`.kind`:
`unparseable`, `out_of_range`, `non_finite`, `truncated`, `refused`, `empty`,
`provider_error`), so the harness records a **metric error** on that row
(`report.n_metric_errors`) rather than a plausible-looking `on_error` score. A
direct `judge.score(...)` returns the error verdict instead of raising.
**Auth and configuration errors** (a bad key, a parameter the installed SDK does
not take, no Azure API version) **raise** instead of being scored `0.0`; a
transient 429/5xx from the judge's provider is retried and, if it persists, the
row is dropped as transient rather than scored. `compare(a, b)` returns `"tie"`
unless the model gave an explicit verdict.

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
`ResultMessage`/content blocks, Microsoft `AgentResponse` (`AgentRunResponse` before 1.0) / `Message`,
Google ADK events and GenAI `Content.parts[*].text`, LangChain messages
(including content-part lists), mappings with conventional keys, and
message/event streams (scanned from the end for the last text).

```python
from adapt_agent.evaluation import register_extractor
register_extractor("my_fw", lambda v: isinstance(v, MyResult), lambda v: v.completion)
```

Pass `output_extractor=None` to score raw outputs, or
`extract_output_payload` to keep structure (see *Scoring structured output*).

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
| **Microsoft Agent Framework** | the `Agent` (`ChatAgent` before 1.0) | async `run()` awaited; `AgentResponse.text` scored |
| **Google ADK** | `adk_runner(agent_or_runner)` | builds an `InMemoryRunner` around a bare agent (needs `google-adk`), or drives a `Runner` you built; a **fresh session per example**; events drained, final text extracted. Customise with `user_id=`, `app_name=`, `message_factory=` |
| **Pydantic AI** | the `Agent` | `run_sync()` used; `.output` scored. A structured `output_type` survives extraction unchanged — score with `json_subset` |
| **CrewAI** | the `Crew` | `kickoff()` used; `CrewOutput.raw` scored |
| **OpenAI Agents SDK** | a callable driving `Runner` | `RunResult.final_output` scored |
| **Claude Agent SDK** | a callable / governed adapter | message stream drained; `ResultMessage.result` scored |

`framework_runner(agent)` performs the run-method discovery, async
materialisation, input adaptation and extraction on its own if you want the
plain `input -> text` callable without an eval.

Async agents are driven synchronously by `evaluate`. Inside an already-running
event loop that raises — use `await harness.aevaluate(agent, dataset)` instead,
which awaits in your loop rather than blocking it. A governed agent's
`aexecute` is preferred automatically by `aresolve_runner`.

## Trajectory (tool-call) evals

A correct answer reached without calling the weather tool is a guess; one reached
via `delete_db` is a failure. Trajectory metrics score **what the agent did** and
compose with the text metrics in one run. A row declares its expected trajectory
beside `input` / `expected`:

```jsonl
{"id": "w1", "input": "Weather in Paris?", "expected": "sunny",
 "expected_tools": [{"name": "get_weather", "args": {"city": "Paris"}}],
 "forbidden_tools": ["delete_*"], "max_tool_calls": 3}
{"id": "w2", "input": "Compare Paris and Rome", "check": {"name": "tool_sequence", "mode": "any_order"},
 "expected_tools": ["get_weather", "get_weather"]}
```

| Metric | Scores 1.0 when ... |
| --- | --- |
| `tool_called(name=None, min_times=1)` | `name` was called; without `name`, the fraction of the row's `expected_tools` called |
| `tool_call_args_match(check="exact_match")` | the fraction of expected calls made with **at least** the expected arguments (a subset) |
| `tool_sequence(mode="in_order", check_args=False)` | `"exact"`, `"in_order"`, `"any_order"` or `"no_extra"` over the expected calls |
| `no_forbidden_tools(forbidden=None)` | no call matched a forbidden name or glob (`"delete_*"`) |
| `tool_call_budget(max_calls=None, max_steps=None)` | at most `max_calls` tool calls / `max_steps` model turns |

A row that declares nothing a metric needs is an **error** for that row, never a
silent pass: apply trajectory metrics per row through `checks` when only some rows
have expectations.

```python
from adapt_agent.evaluation import evaluate_agent, no_forbidden_tools, tool_call_args_match, tool_sequence

report = evaluate_agent(
    agent, "golden.jsonl",
    metrics=["exact_match", tool_sequence(mode="in_order"), tool_call_args_match(),
             no_forbidden_tools()],
)
```

The harness keeps each row's **raw** result (before text extraction) and
`extract_trajectory(result)` recovers the calls for LangGraph / LangChain,
Pydantic AI, OpenAI Agents, Google ADK, Microsoft Agent Framework, the Claude
Agent SDK and OpenAI chat-completions shapes. For your own agent return
`{"output": ..., "tool_calls": [{"name": ..., "arguments": {...}, "result": ...}],
"steps": 3}`. `max_steps` refuses to score when a result does not expose its step
count; budget `max_tool_calls` instead.

## Multi-turn evals

A row may be a scripted conversation; every turn is scored and the row's score is
the **mean over its scored turns**:

```jsonl
{"id": "c1", "turns": [
  {"input": "My order is 1234.", "score": false},
  {"input": "Where is it?", "expected": "1234", "check": "contains"},
  {"input": "Cancel it.", "expected_tools": ["cancel_order"], "check": "tool_called"}
]}
```

`conversation_input=` (on `evaluate_agent` / `EvaluationHarness`) decides how
history reaches the agent: `"messages"` (default) passes the conversation so far as
`[{"role": ..., "content": ...}]`; `"text"` passes only the new turn (for an agent
with its own memory; use a fresh agent per row); or a callable
`(history, turn_input) -> payload`. If the agent raises on turn *k* the row is an
error and later turns do not run. `result.turns` holds per-turn detail.

## Gating CI

```bash
adapt evaluate "myapp.agents:build()" --data evals/golden.jsonl \
    --metric checks --extract-output --concurrency 8 \
    --fail-under 0.9 --fail-under tool_sequence=1.0 \
    --fail-on-error --fail-on-incomplete \
    --baseline evals/baseline.json --fail-on-regression \
    --output eval-results.json --junit eval-junit.xml
```

| Flag | Effect |
| --- | --- |
| `--fail-under [METRIC=]SCORE` | exit 3 when a metric's aggregate is below it; a bare number gates every metric; repeatable; an unknown metric is an error |
| `--fail-on-error` | exit 4 if any row errored (agent raised, or a metric failed) |
| `--fail-on-incomplete` | exit 5 if rows were dropped as transient (the score covers part of the data) |
| `--baseline FILE` + `--fail-on-regression` | exit 6 when a metric or a row dropped by more than `--regression-tolerance` (default 0) |
| `--output FILE` / `--junit FILE` | full per-row JSON (the baseline format) / JUnit XML |
| `--concurrency N` | N rows at once |

| Exit | Meaning |
| --- | --- |
| `0` | ran; every gate passed |
| `1` | could not run: bad arguments, the target / dataset / judge failed to load, an empty dataset |
| `2` | usage error (argparse) |
| `3` | `--fail-under` missed |
| `4` | rows errored with `--fail-on-error` — **and always** when every row errored |
| `5` | rows dropped as transient with `--fail-on-incomplete` — **and always** when every row was |
| `6` | regressed against `--baseline` |

When several gates fail the most severe wins (4, then 5, 3, 6) and every failure
is printed. **Without any gate flag, an eval that scores 0% still exits 0** unless
every row errored: add `--fail-under`. Give dataset rows an `id` so rows match
across runs by identity, not position. `--json` prints the summary, completeness,
rows, a `gates` block and the baseline comparison.

## Exporting results and comparing runs

```python
from adapt_agent.evaluation import compare_runs, load_results, to_json, to_junit

to_json(report, "results.json")                        # summary + completeness + every row
to_junit(report, "junit.xml", thresholds={"exact_match": 1.0})
diff = compare_runs(report, load_results("baseline.json"), tolerance=0.02)
diff.metrics            # {"exact_match": {"baseline": 0.92, "current": 0.88, "delta": -0.04}}
diff.regressed_rows     # [{"key": "id:w7", "metric": "exact_match", "baseline": 1.0, "current": 0.0}]
diff.warnings           # partial runs, position-matched rows, a different primary metric
```

## CLI

```bash
adapt-agent evaluate myapp.agents:agent --data golden.jsonl \
    --metric exact_match --extract-output

adapt-agent evaluate "myapp.agents:build()" --data golden.jsonl \
    --metric checks --judge claude --judge-model claude-opus-5-5 --extract-output --json
```

Target is `module:attribute` (`()` calls a factory). `--metric` repeats (any
built-in, including the trajectory metrics); `--extract-output` enables framework
unwrapping; `--judge PROVIDER` adds a `judge` metric and `--metric judge` grades
every row; `--primary` picks the headline. Every run prints completeness
(`rows: 3 total, 3 scored (2 errored, scored 0), 0 dropped as transient, ...`).
The same flags work on `adapt-agent optimize`. **A target is imported and a
trailing `()` is called, so a target or training config is code**: run only ones
you would run as a script. The CLI's `--json` and exit codes are in "Gating CI".
