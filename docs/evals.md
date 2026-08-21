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
| `exact_match` | output equals expected (normalised: case / whitespace / punctuation) |
| `contains` | expected appears as a substring of the output |
| `regex_match` | output matches the expected regex (or a fixed `pattern=`) |
| `numeric_close` | the first number in the output is within `tolerance` of expected's (`relative=True` for a relative bound) |
| `token_f1` | token-overlap F1 (SQuAD-style, continuous) |
| `jaccard` | token-set Jaccard similarity (continuous) |
| `json_subset` | every key/value in the expected dict appears in the output dict |
| `levenshtein_ratio` | normalised edit-distance similarity (continuous) |
| `checks` | per-row dispatch -- see below |
| `judge` / `llm_judge` | the LLM judge grades the row (needs `judge=`) |

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
judge = LLMJudge(ClaudeJudge(model="claude-opus-4-8"), pass_threshold=0.7)
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

### CrewAI, OpenAI Agents SDK, Claude Agent SDK

`Crew.kickoff` results (`CrewOutput.raw`), OpenAI Agents `RunResult`s
(`final_output`), and Claude Agent SDK message streams (`ResultMessage` /
content blocks) are recognised the same way -- pass the `Crew`, a
runner-driving callable, or the governed adapter and go.

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
that trains it -- see [Optimization & Evaluation](optimization.md).

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
  answer — `{"messages": [...], "remaining_steps": N, "structured_response":
  {...}}`. The declared output is peeled out of it; a state without that key
  arrives whole.
- A **single-field answer** like `{"result": "granted"}` is the answer, not an
  envelope around one. Only a conventional key holding something *structured*
  is peeled, so a one-column output keeps its column.

`evaluate_agent` selects it automatically when **every** metric is structural
(`json_subset`, `field_match`); mixing in a text metric keeps text extraction,
because a dict scores 0.0 against `exact_match`.

`field_match(field, check="exact_match", missing=0.0, **options)` scores one
field and reports under its name, which turns `aggregate` into a per-column
table instead of one blended number that hides which column moved. That
`pack: 0.0` is a column the agent never gets right; averaged in, it reads as a
mild dip. A deterministic field check is also cheaper and more correct than
asking a judge whether two labels agree.
