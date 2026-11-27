# Running Evals

ADAPT-Agent runs **evals** -- scoring an agent against a golden dataset of
inputs and expected outputs -- for agents built with **LangGraph, Microsoft
Agent Framework, Google ADK, Pydantic AI**, CrewAI, the OpenAI Agents SDK, the
Claude Agent SDK, or plain Python callables. Three kinds of scoring compose
freely:

* **Deterministic checks** against a specific output: exact / contained text,
  regex, token-F1, a **number within tolerance**, JSON subset, edit distance.
* **Per-row checks**: each dataset row declares *how* it is scored (`"check":
  "exact_match"` here, `"check": "numeric_close"` there, an LLM-judge
  elsewhere).
* **LLM-as-judge**: model-graded scoring against a rubric and per-row criteria,
  provider-agnostic (Claude, OpenAI, Gemini, Mistral, Ollama, ... or any
  callable), for rows with no single right answer.

Everything is import-light: no agent framework or LLM SDK is imported unless
your agent / judge actually uses it.

## Quick start

```python
from adapt_agent.evaluation import evaluate_agent

report = evaluate_agent(
    my_agent,                     # LangGraph graph, MAF ChatAgent, Pydantic AI Agent, ...
    [
        # exact text match (the default check)
        {"input": "What is the capital of France?", "expected": "Paris"},
        # number match: "The answer is 42." passes against 42
        {"input": "What is 6 * 7?", "expected": 42, "check": "numeric_close"},
        # graded by the LLM judge against per-row criteria
        {"input": "Summarise our refund policy", "check": "judge",
         "criteria": "Mentions the 30-day window and the exceptions."},
    ],
    judge="claude",               # optional; needed only for judge rows
)

print(report.score)               # headline score in [0, 1]
print(report.aggregate)           # {"checks": 0.83, ...}
for failure in report.failures():
    print(failure.inputs, failure.output, failure.expected)
```

`evaluate_agent` does four things in one call:

1. **Loads the dataset** -- a `GoldenDataset`, a list of records, or a
   `.json` / `.jsonl` / `.csv` path (`input_key=` / `expected_key=` override
   column auto-detection).
2. **Runs the agent** -- discovers the framework run method (`run_sync` /
   `invoke` / `kickoff` / `run` / `execute` or a bare callable), awaits
   coroutines and drains async streams, and adapts plain-string inputs for
   LangGraph message-state graphs automatically.
3. **Extracts the output** -- unwraps framework-native results (a Pydantic AI
   `AgentRunResult`, a LangGraph state dict, a Microsoft `AgentRunResponse`,
   Google ADK events, ...) to the final response text so checks compare
   answers, not `repr()`s.
4. **Scores and reports** -- returns the standard
   [`EvaluationReport`](optimization.md#evaluation-reports).

## Built-in checks

Every check maps `(output, expected)` to a score in `[0, 1]`; most are binary.

| Name | Passes when ... |
| --- | --- |
| `exact_match` | output equals expected after normalisation: NFKC, case-folding, whitespace, and *edge* punctuation (`"Paris."` = `"paris"`, but `-5` ≠ `5`) |
| `contains` | expected appears in the output **on word boundaries** (`"no"` is not found in `"know"`) |
| `regex_match` | output matches the expected regex (or a fixed `pattern=`); row-supplied patterns are screened for catastrophic backtracking |
| `numeric_close` | the answer number in the output is within `tolerance` of expected's (`relative=True` for a relative bound); see below for which number |
| `token_f1` | token-overlap F1 (SQuAD-style, continuous; Japanese/Chinese/Thai tokenised per character) |
| `jaccard` | token-set Jaccard similarity (continuous) |
| `json_subset` | fraction of expected keys matched, **recursively** (nested dicts as subsets; `True` ≠ `1`); fenced or embedded JSON is parsed |
| `levenshtein_ratio` | normalised edit-distance similarity (continuous) |
| `checks` | per-row dispatch -- see below |
| `judge` / `llm_judge` | the LLM judge grades the row (needs `judge=`) |
| `tool_called`, `tool_call_args_match`, `tool_sequence`, `no_forbidden_tools`, `tool_call_budget` | trajectory checks on the tool calls the agent made -- see [Trajectory evals](#trajectory-tool-call-evals) |

### Metric semantics (changed in 0.6)

These defaults changed because the old ones scored wrong answers as right.
Each old behaviour is still one option away.

| Metric | Now | Before (still available) |
| --- | --- | --- |
| `exact_match` | punctuation stripped only from token edges; NFKC + case-folding | every symbol erased, so `-5` = `5`, `<=` = `>=`, `C++` = `C`, `$100` = `100%` (`exact_match(punctuation="all")`) |
| `contains` | word-boundary match | substring (`contains(word_boundary=False)`; also the default with `normalize=False`) |
| `numeric_close` | `which="auto"`: the number after the last answer marker (`answer`, `result`, `total`, `=`), else the last number; reads `.5`, `1e3`, `1,000`, `3,14`, Unicode digits | the first number (`numeric_close(which="first")`); `"any"` passes if any number matches |
| `regex_match` | a row with no pattern (`expected` empty/`None`) is an **error**, not a pass; row patterns over 512 characters or with nested quantifiers/alternation under a quantifier (`(a+)+`, `(a\|aa)+`) are refused | the empty pattern matched everything (1.0); `(a+)+$` could hang the run |
| `json_subset` | recursive subset, strict bool/int, fenced/embedded JSON parsed | flat equality per key (`json_subset(strict_types=False)` for loose types) |
| any metric | a NaN / infinite score is an **error for that row** (`report.n_metric_errors`, `result.metric_errors`) | NaN was clamped to **1.0** |

A metric that raises (or returns a non-finite value) on a row scores that row
with its declared `on_error` fallback (`0.0` by default) and is recorded in
`ExampleResult.metric_errors`; it is never silently a pass.

Pass them by name, as factories with options, or mix with your own callables:

```python
from adapt_agent.evaluation import evaluate_agent, numeric_close

report = evaluate_agent(
    my_agent, "golden.jsonl",
    metrics=["contains", numeric_close(tolerance=0.01, relative=True), my_scorer],
    primary_metric="contains",
)
```

## Per-row checks

With the default `metrics="checks"`, each row picks its own scorer via a
`check` field (`checks` in the plural also works); rows without one fall back
to `exact_match`:

```jsonl
{"input": "What is the capital of France?", "expected": "Paris"}
{"input": "What is 6 * 7?", "expected": 42, "check": "numeric_close"}
{"input": "Quote a price near $100", "expected": 100, "check": {"name": "numeric_close", "tolerance": 5}}
{"input": "List the EU founders", "expected": "Belgium", "check": ["contains", "token_f1"]}
{"input": "Write a friendly greeting", "check": "judge", "criteria": "Warm, one sentence."}
```

* A **string** names a built-in.
* A **mapping** passes options to the factory (`{"name": "numeric_close",
  "tolerance": 5}`).
* A **list** applies several checks; they combine with `min` by default (every
  check must pass for a perfect score) -- build the dispatcher yourself with
  `checks(aggregate="mean")` to average instead.
* `"judge"` routes the row to the LLM judge; only those rows spend judge calls.

The dispatcher is a normal metric, so the same dataset works from the CLI
(`--metric checks`) and inside [optimization](optimization.md) loops.

It is marked **structural**, so the output reaches it with its structure intact
— a row asking for `field_match` needs the fields, and which row asks is only
known at call time. Rows asking for a text check are flattened individually as
they are scored, so both kinds work in one dataset. The exception is `checks`
*mixed* with a plain text metric, where the all-or-nothing rule picks text
extraction for the whole run; pass `output_extractor=extract_output_payload`
(imported from `adapt_agent.evaluation`) if a row needs structure there — at the
cost of the text metric, which then scores a mapping.

## LLM-as-judge

Pass `judge=` as a provider name, an
[`LLMJudge`](optimization.md#llm-as-judge-at-every-stage), or any
`prompt -> text` callable:

```python
from adapt_agent.evaluation import evaluate_agent, LLMJudge
from adapt_agent.optimization.judges import ClaudeJudge

# Provider name (uses the provider's recommended model):
report = evaluate_agent(my_agent, data, judge="claude")

# Or configure the judge yourself:
judge = ClaudeJudge(model="claude-opus-5-5", pass_threshold=0.7)   # a ClaudeJudge *is* an LLMJudge
# ...or wrap any provider / completion callable:
judge = LLMJudge(my_completion_fn, pass_threshold=0.7)
report = evaluate_agent(
    my_agent, data,
    judge=judge,
    judge_criteria="Answers must cite the source document.",
)
```

* With **no `metrics`**, a supplied judge grades every row (reference-free:
  rows do not need an `expected` value).
* With `metrics=[...]`, the judge is added as an extra `"judge"` metric
  alongside the deterministic ones.
* With `metrics="checks"`, only rows declaring `{"check": "judge"}` are judged.
* Rows can carry their own `criteria` metadata; `judge_criteria=` sets the
  task-level default and `judge_rubric=` replaces the grading rubric.

Judges are prompt-injection hardened (agent output is fenced as data) and fail
closed (an unreachable judge scores `0.0`, never inflates results). In tests,
pass a deterministic stub callable -- see
[`examples/08_agent_evals.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/08_agent_evals.py).

## Framework notes

The same `evaluate_agent(...)` call works across frameworks; the differences
that matter are below. Full agent-construction walkthroughs live in the
[framework guides](frameworks/index.md).

### LangGraph

```python
graph = builder.compile()                      # anything with .invoke(state)
report = evaluate_agent(graph, "golden.jsonl")
```

Plain-string inputs are wrapped into
`{"messages": [{"role": "user", "content": ...}]}` automatically (the
`MessagesState` convention used by `create_react_agent` and most graphs), and
the final message's text is read back out of the returned state. Dataset
inputs that are already dicts pass through untouched -- put graph-native state
in your dataset for custom schemas, or pass `input_adapter=` / `None` to
override the adaptation.

### Microsoft Agent Framework

```python
agent = ChatAgent(chat_client=..., instructions=...)
report = evaluate_agent(agent, "golden.jsonl")
```

The async `agent.run(...)` coroutine is awaited internally and the
`AgentRunResponse.text` is scored. (Inside an already-running event loop --
e.g. a notebook -- run the eval in a worker thread.)

### Google ADK

ADK agents execute inside a `Runner` with sessions and `google.genai` message
types; `adk_runner` packages all of that as a plain callable:

```python
from adapt_agent.evaluation import adk_runner, evaluate_agent

runner = adk_runner(my_llm_agent)         # builds an InMemoryRunner (needs google-adk)
# ... or wrap a Runner you already configured:
runner = adk_runner(Runner(agent=my_llm_agent, app_name="app", session_service=...))

report = evaluate_agent(runner, "golden.jsonl")
```

Each eval example runs in a **fresh session**, so examples stay independent.
The event stream is drained and the final response text extracted. Customise
with `user_id=`, `app_name=`, or `message_factory=` (how a dataset input
becomes the `new_message`).

### Pydantic AI

```python
agent = Agent("openai:gpt-5", system_prompt=...)
report = evaluate_agent(agent, "golden.jsonl")
```

`agent.run_sync(...)` is used and the `AgentRunResult.output` scored. A
structured (non-text) `output_type` survives extraction unchanged -- score it
with `json_subset` or your own callable.

### CrewAI

```python
report = evaluate_agent(my_crew, "golden.jsonl")   # Crew.kickoff; CrewOutput.raw scored
```

### OpenAI Agents SDK

An OpenAI Agents `Agent` has no run method of its own -- the SDK executes it
with `Runner.run_sync(agent, input)`. Pass the `Agent` straight in and that
machinery is used automatically (`RunResult.final_output` is scored); use
`openai_agents_runner` when you need to customise the run:

```python
from adapt_agent.evaluation import evaluate_agent, openai_agents_runner

report = evaluate_agent(triage_agent, "golden.jsonl")            # needs openai-agents

runner = openai_agents_runner(triage_agent, run_kwargs={"max_turns": 5})
report = evaluate_agent(runner, "golden.jsonl")
```

(`Runner.run_sync` refuses to run inside an already-running event loop -- the
SDK's own constraint -- so drive evals from sync code.)

### Claude Agent SDK

A Claude agent is a `ClaudeAgentOptions` object driven by
`query(prompt=..., options=...)`. Pass the options object straight in -- the
async message stream is drained and the final `ResultMessage` text scored --
or build the runner explicitly with `claude_agent_runner`:

```python
from adapt_agent.evaluation import claude_agent_runner, evaluate_agent

report = evaluate_agent(options, "golden.jsonl")                 # needs claude-agent-sdk

runner = claude_agent_runner(options)   # same machinery, explicit
```

## Output extraction

Extraction is what lets text/number checks work on framework-native results.
It is structural (duck-typed, no SDK imports), recursive, and conservative:
strings pass through, `None` becomes `""`, and **unrecognised values are
returned unchanged** -- so structured outputs still reach `json_subset` or your
custom metrics intact.

```python
from adapt_agent.evaluation import extract_output_text, register_extractor

extract_output_text(agent_run_result)          # "Paris"
extract_output_text({"messages": [...]})       # final message text

# Score raw outputs instead (e.g. judging the full state dict):
report = evaluate_agent(my_agent, data, output_extractor=None)

# Teach the extractor a custom result type (tried before the built-ins):
register_extractor(
    "my_framework",
    lambda v: isinstance(v, MyResult),
    lambda v: v.completion_text,
)
```

## Using the harness directly

`evaluate_agent` is sugar over the same pieces the optimizers use -- compose
them yourself for full control, or to reuse a harness across
[optimization](optimization.md) runs:

```python
from adapt_agent.evaluation import (
    EvaluationHarness, GoldenDataset, checks, extract_output_text, framework_runner,
)

data = GoldenDataset.from_jsonl("golden.jsonl")
harness = EvaluationHarness(
    [checks(judge=my_judge), my_custom_metric],
    primary_metric="checks",
    output_extractor=extract_output_text,
    failure_threshold=0.6,
)
report = harness.evaluate(framework_runner(my_agent), data)
```

## From the CLI

```bash
# Deterministic checks; unwrap framework-native outputs before scoring
adapt-agent evaluate myapp.agents:agent --data golden.jsonl \
    --metric exact_match --extract-output

# Per-row checks + an LLM judge for rows that declare {"check": "judge"}
adapt-agent evaluate "myapp.agents:build()" --data golden.jsonl \
    --metric checks --judge claude --extract-output --json
```

`--extract-output` applies the same extraction as `evaluate_agent`; the target
is `module:attribute` (append `()` to call a factory). Judge routing matches
`evaluate_agent`: with `--metric checks`, only rows declaring
`{"check": "judge"}` spend judge calls -- add `--metric judge` to also grade
every row with the `--judge` provider. The identical flags work on
`adapt-agent optimize`, so the eval that gates your agent is the same eval
that trains it -- see [Optimization & Evaluation](optimization.md), and
[Gating CI](#gating-ci) below for the exit codes.

!!! warning "Targets and configs are code"
    `module:attribute` **imports** the module, which runs its top-level code,
    and a trailing `()` **calls** the object; the current working directory is
    on the import path. A training config's `target:` entries are the same.
    Treat a config file like a script: only run ones you would execute. The
    `judge:` section of a training config accepts a fixed list of keys and
    refuses others -- in particular `base_url` and `api_key` -- so a config
    cannot redirect graded outputs to another host or carry a credential; set
    API keys in the provider's environment variable.

## Running examples concurrently

`evaluate` is serial by default, which is fine for a smoke test and untenable
for optimization: a `CoordinateAscentOptimizer(max_evals=60)` sweep over a
113-case split is 6,780 agent calls, each an LLM round-trip.

```python
report = harness.evaluate(agent, dataset, concurrency=8)          # sync agent: threads
report = await harness.aevaluate(agent, dataset, concurrency=8)   # async agent: no threads
report = evaluate_agent(agent, data, concurrency=8)               # same knob
```

Use `aevaluate` for an async-native agent — it awaits in your loop, so
`contextvars` (tracing spans) survive and no thread is involved. Use
`evaluate(concurrency=)` for a synchronous agent whose time goes on network I/O.

Both keep the serial path's guarantees: results are reported in **example-index
order** regardless of completion order, a per-example exception is still a
non-fatal zero-scored error, and `max_results` still bounds stored records while
every example is aggregated.

## Scoring structured output

`extract_output_text` flattens a structured answer — a Microsoft
`AgentResponse` is a recognised shape, so it becomes `.text` and the object
is gone — while `output_extractor=None` unwraps nothing and scores a `repr()`.
`extract_output_payload` is the middle path: strip the framework envelope, keep
the payload.

```python
from adapt_agent.evaluation import field_metrics

report = evaluate_agent(agent, data, metrics=field_metrics(["lane", "matter", "action", "pack"]))
report.aggregate    # {"lane": 0.94, "matter": 0.90, "action": 0.63, "pack": 0.0}
```

A mapping or sequence passes through, a declared structured output — a Pydantic
model or a dataclass — becomes a dict, and a JSON string is parsed back into an
object (including a fenced ```` ```json ```` block). Non-JSON text degrades to
text rather than erroring.

Two shapes need naming, because both used to score 0.0 across the board:

- A **LangGraph** graph built with `response_format=` returns a *state*, not an
  answer — a real run's keys are exactly `["messages", "structured_response"]`.
  The declared output is peeled out of it. The state has to be recognised as
  one, not merely carry the key: on their own both `structured_response` and
  `messages` are ordinary field names, so what identifies a state is what
  LangGraph guarantees — `add_messages` coerces every entry to a `BaseMessage`,
  so `messages` is a non-empty list of message *objects*. An answer like
  `{"messages": ["audit"], "structured_response": {...}, "request_id": "42"}`
  keeps every column.
- A **list of records** is the answer, not a message stream. A stream is scanned
  from the end for its final message; a list of dataclasses or models is
  returned whole.
- A **single-field answer** like `{"result": "granted"}` or `{"answer": {"city":
  "Paris"}}` is the answer, not an envelope around one. A governed `execute`
  returns a dict result untouched and wraps only a non-dict, so a conventional
  key is an envelope only when what it holds still needs unwrapping — a
  one-column output keeps its column whatever type it holds.

`extract_output_text` leaves a declared output **unchanged** — it is not a
wrapper around text — which is what lets a per-row `field_match` still see its
fields on the `checks` path, where the row decides the metric at runtime and
text extraction is therefore the one in force.

`evaluate_agent` selects it automatically when **every** metric is structural
(`json_subset`, `field_match`); mixing in a text metric keeps text extraction,
because a dict scores 0.0 against `exact_match`.

`field_match(field, check="exact_match", missing=0.0, **options)` scores one
field and reports under its name, which turns `aggregate` into a per-column
table instead of one blended number that hides which column moved. That
`pack: 0.0` is a column the agent never gets right; averaged in, it reads as a
mild dip. A deterministic field check is also cheaper and more correct than
asking a judge whether two labels agree.

## Trajectory (tool-call) evals

A correct answer reached without calling the weather tool is a guess; a
correct answer reached via `delete_db` is a failure. Trajectory metrics score
**what the agent did** -- the tool calls, in order, with their arguments -- and
compose with the text metrics in one run.

A row declares its expected trajectory next to `input` / `expected`, the same
place a per-row `check` lives:

```jsonl
{"id": "w1", "input": "Weather in Paris?", "expected": "sunny",
 "expected_tools": [{"name": "get_weather", "args": {"city": "Paris"}}],
 "forbidden_tools": ["delete_*"], "max_tool_calls": 3}
{"id": "w2", "input": "Compare Paris and Rome", "check": {"name": "tool_sequence", "mode": "any_order"},
 "expected_tools": ["get_weather", "get_weather"]}
```

| Metric | Scores 1.0 when ... |
| --- | --- |
| `tool_called(name=None, min_times=1)` | `name` was called (`min_times`); without `name`, the fraction of the row's `expected_tools` that were called |
| `tool_call_args_match(check="exact_match")` | the fraction of expected calls made with **at least** the expected arguments (a subset; extra arguments are fine). String values use the `check` text metric, so case and edge punctuation do not matter; others compare by equality (`True` ≠ `1`). Use `{"$check": "contains", "value": "paris"}` (any built-in) for a looser match. Each real call satisfies one expectation. |
| `tool_sequence(mode="in_order", check_args=False)` | `"exact"`: exactly the expected calls in order; `"in_order"`: expected calls appear in order, others may be interleaved; `"any_order"`: all expected calls appear (as a multiset); `"no_extra"`: all appear and nothing else was called. `check_args=True` also requires the arguments. |
| `no_forbidden_tools(forbidden=None)` | no call matched a forbidden name or glob (`"delete_*"`) |
| `tool_call_budget(max_calls=None, max_steps=None)` | the run made at most `max_calls` tool calls / `max_steps` model turns (or the row's `max_tool_calls` / `max_steps`) |

Expectations come from the factory arguments when given, else from the row. A
row that declares nothing a metric needs is an **error** for that row (recorded
in `result.metric_errors`), never a silent pass -- apply trajectory metrics
per row with `checks` when only some rows have expectations.

```python
from adapt_agent.evaluation import (
    evaluate_agent, no_forbidden_tools, tool_call_args_match, tool_sequence,
)

report = evaluate_agent(
    my_agent, "golden.jsonl",
    metrics=["exact_match", tool_sequence(mode="in_order"), tool_call_args_match(),
             no_forbidden_tools()],
)
report.aggregate   # {"exact_match": 0.9, "tool_sequence": 0.8, "tool_call_args_match": 0.75, ...}
```

**Where the trajectory comes from.** The harness keeps each row's *raw* result
(before `output_extractor` flattens it to text) and
`extract_trajectory(result)` recovers the calls, duck-typed like the text
extractor:

| Framework | Read from |
| --- | --- |
| LangGraph / LangChain | state `messages`: `AIMessage.tool_calls`, results from `ToolMessage` |
| Pydantic AI | `AgentRunResult.all_messages()`: `ToolCallPart` / `ToolReturnPart` |
| OpenAI Agents SDK | `RunResult.new_items`: `ToolCallItem` / `ToolCallOutputItem` (steps from `raw_responses`) |
| Google ADK | event stream: `content.parts[*].function_call` / `function_response` |
| Microsoft Agent Framework | `AgentResponse.messages[*].contents` of type `function_call` / `function_result` |
| Claude Agent SDK | message stream: `ToolUseBlock` in `AssistantMessage`, `ToolResultBlock` in replies |
| OpenAI chat completions | assistant `tool_calls` (`function.name` / `function.arguments`), `role: "tool"` replies |
| Your own agent | return `{"output": ..., "tool_calls": [{"name": ..., "arguments": {...}, "result": ...}], "steps": 3}` |

Arguments given as a JSON string are parsed; results are paired to calls by
call id. A step is one model response; when a result does not expose its step
count, `max_steps` refuses to score rather than guess (budget tool calls with
`max_tool_calls` instead). For a framework not listed, return the dict form or
pass a `Trajectory` yourself.

## Multi-turn evals

A row may be a **scripted conversation**: a list of user turns, each with its
own expectation. The harness feeds the turns in order, carries the history, and
scores every turn.

```jsonl
{"id": "c1", "turns": [
  {"input": "My order is 1234.", "score": false},
  {"input": "Where is it?", "expected": "1234", "check": "contains"},
  {"input": "Cancel it.", "expected_tools": ["cancel_order"], "check": "tool_called"}
]}
```

* Each turn has an `input`, optionally `expected`, and any per-turn metadata
  (`check`, `criteria`, `expected_tools`, ...). `"score": false` runs a setup
  turn without scoring it. Turns can also be plain strings.
* How history reaches the agent is `conversation_input=` on `evaluate_agent` /
  `EvaluationHarness`: `"messages"` (default) passes the conversation so far as
  `[{"role": "user"|"assistant", "content": ...}]` ending with the new user turn
  (the assistant content is each previous output's extracted text); `"text"`
  passes only the new turn, for an agent that keeps its own memory (use a fresh
  agent per row); or a callable `(history, turn_input) -> payload`.
* The row's score for each metric is the **mean over its scored turns**;
  `result.turns` holds per-turn inputs, outputs, scores and latency.
* If the agent raises on turn *k*, the row is an error (`"turn k: ..."`) and
  later turns are not run.

The user side is scripted, not simulated: an LLM playing the user is out of
scope. Malformed `turns` fail when the dataset loads, before any agent call.

## Latency, tokens and cost

Every row records its latency (`result.latency`); the report adds
`latency_p50` / `latency_p95` (nearest-rank, over every row run) beside
`avg_latency`. Token usage and cost are recorded when the result exposes them
-- `usage` (attribute, key or Pydantic AI's `usage`), the OpenAI Agents SDK's
`context_wrapper.usage`, Microsoft's `usage_details`, LangChain
`usage_metadata`, ADK `usage_metadata`, or Claude's `ResultMessage`
(`total_cost_usd`) -- normalised to `input_tokens`, `output_tokens`,
`total_tokens`, `cost_usd` on `result.usage`, and summed on `report.usage`.
When a judge's provider exposes `last_usage`, its tokens are added as
`judge_input_tokens` etc. -- on the serial path only, since with concurrent
rows the provider's last call may belong to another row.

## What counts as "transient"

A row whose call failed with a *transient infrastructure fault* -- a 429, a
5xx, a provider timeout that outlived its retries -- is excluded from the
score (so a throttled run is not ranked below an unthrottled one), counted in
`n_transient_errors`, and makes `is_complete` false.

**An exception raised by the agent under test is only eligible when something
shows it came from infrastructure**: a link in its cause chain was raised by a
provider SDK or HTTP client package (`openai`, `anthropic`, `google`, `httpx`,
`requests`, ...; see `evaluation.INFRASTRUCTURE_MODULES`), carries an HTTP
status attribute, or names an HTTP status in its message (`"Error code: 429"`,
`"503 Service Unavailable"`). Anything else -- an agent's own
`ToolTimeout("step budget exceeded")`, a bare `TimeoutError` from your tool --
is a **failure of that row**: scored 0, kept in the denominator, reported as an
error. (Before 0.6 such a row was dropped, so an agent failing two rows of
three on its own timeouts printed `1.0000`.) A judge or metric call keeps the
generous classification, since its provider is the harness's own
infrastructure. Pass `retry=RetryPolicy(is_transient=...)` to decide yourself;
a custom classifier is honoured as given.

`report.completeness()` says how much of the dataset a score covers:

```python
{"rows_total": 3, "rows_scored": 3, "rows_errored": 2,
 "rows_dropped_transient": 0, "rows_metric_errors": 0}
```

`rows_scored` is the denominator of `score` and *includes* errored rows (they
score 0). The CLI prints this line for every run, and labels any aggregate
computed over a subset `PARTIAL` next to the number.

## Gating CI

`adapt evaluate` exits non-zero when a gate fails, so it can be a CI step:

| Exit | Meaning |
| --- | --- |
| `0` | ran; every gate passed |
| `1` | could not run: bad arguments, the target / dataset / judge failed to load, an empty dataset |
| `2` | command-line usage error (from argparse) |
| `3` | `--fail-under` missed |
| `4` | `--fail-on-error`: a row errored (agent raised, or a metric failed) -- **and always** when every row errored |
| `5` | `--fail-on-incomplete`: rows were dropped as transient -- **and always** when every row was |
| `6` | `--fail-on-regression`: worse than `--baseline` |

When several gates fail the most severe code wins (4, then 5, 3, 6), and every
failure is printed.

```yaml
# .github/workflows/evals.yml
- name: Agent evals
  run: |
    adapt evaluate "myapp.agents:build()" --data evals/golden.jsonl \
      --metric checks --extract-output --concurrency 8 \
      --fail-under 0.9 --fail-under tool_sequence=1.0 \
      --fail-on-error --fail-on-incomplete \
      --baseline evals/baseline.json --fail-on-regression \
      --output eval-results.json --junit eval-junit.xml
- uses: actions/upload-artifact@v4
  if: always()
  with: {name: eval-results, path: "eval-results.json\neval-junit.xml"}
```

* `--fail-under 0.9` gates **every** metric; `--fail-under exact_match=0.9`
  gates one (and overrides a bare value for it). An unknown metric name is an
  error, so a typo cannot disable a gate.
* `--fail-on-incomplete` is what makes the gate honest under throttling: a
  score over a subset can pass a threshold the full dataset would miss.
* `--concurrency N` runs N rows at once in a thread pool (the harness's
  `evaluate(concurrency=)`); results are identical, in row order.

## Exporting results and comparing runs

```bash
adapt evaluate ... --output results.json     # full per-row results
adapt evaluate ... --junit junit.xml         # one testcase per row x metric
adapt evaluate ... --baseline results.json --fail-on-regression --regression-tolerance 0.02
```

The same functions are importable:

```python
from adapt_agent.evaluation import compare_runs, load_results, to_json, to_junit

to_json(report, "results.json")                   # summary + completeness + every row
to_junit(report, "junit.xml", thresholds={"exact_match": 1.0, "judge": 0.6})
diff = compare_runs(report, load_results("baseline.json"), tolerance=0.02)
diff.metrics            # {"exact_match": {"baseline": 0.92, "current": 0.88, "delta": -0.04}}
diff.regressed_rows     # [{"key": "id:w7", "metric": "exact_match", "baseline": 1.0, "current": 0.0}]
diff.has_regression, diff.warnings
```

* The JSON document holds `summary` (the report's `to_dict()` plus
  completeness, latency percentiles, usage) and one record per stored row:
  inputs, expected, output (each truncated to 4,000 characters), every score,
  error and error type, metric errors, latency, usage, and per-turn detail.
  `--json` on stdout prints the same summary with the rows and the gate
  results.
* JUnit: a case fails below its threshold (`--fail-under` values; default the
  harness's `failure_threshold`, 1.0) with expected / actual / input in the
  body; an agent exception is an `<error>`; a transient row is `<skipped>`.
  The document is capped (5 MB by default): past the cap failure bodies are
  dropped, never cases, so the counts stay exact.
* Rows are matched across runs by their dataset **`id`** -- give rows one.
  Without it they are matched by position, which a `warnings` entry flags,
  because inserting a row shifts every match. A partial run on either side is
  flagged too.
