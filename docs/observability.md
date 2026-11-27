# Observability, Evaluation & Optimization

This page covers the three monitoring and quality components:
[`AgentObserver`](#agentobserver),
[`AgentEvaluator`](#agentevaluator), and
[`AgentOptimizer`](#agentoptimizer). All three keep bounded in-memory state;
the observer can also export OpenTelemetry spans.

```python
from adapt_agent import AgentObserver, AgentEvaluator, AgentOptimizer
```

---

## AgentObserver

`adapt_agent.observability.AgentObserver` records **traces** (one per governed
run, with per-trace **events**), **refusals** (requests refused before a run
started), **tool decisions** (the gate's verdict on each tool call), free-form
**logs**, and numeric **metrics**. Everything is held in bounded in-memory
buffers; optionally every trace, refusal and tool decision is also emitted as an
[OpenTelemetry span](#opentelemetry-bridge).

```python
obs = AgentObserver(
    max_logs=1000,
    max_traces=1000,            # also the size of the refusal and tool-decision buffers
    max_metrics=1000,           # per metric-name series
    max_events_per_trace=1000,
    max_metric_names=1000,
    max_text_length=10_000,     # cap for any stored free-text string
    max_error_length=200,       # cap for error text (exception messages, error results)
    redactor=None,              # e.g. adapt_agent.observability.redact_secrets
    otel=False,                 # True, a Tracer, or an OTelBridge -- see below
)
```

Every cap must be a positive integer; anything else raises `ValueError`.

### Traces and events

A trace bundles the events of one operation. Start it, log events into it, then
end it with a status.

```python
obs.start_trace("trace-1", agent_id="agent-007", operation="answer_question")
obs.log_event("trace-1", event_type="input_screened", description="passed firewall")
obs.log_event("trace-1", event_type="llm_call", description="model responded",
              metadata={"tokens": 312})
obs.end_trace("trace-1", status="completed", result={"ok": True})
```

What a trace record holds:

| Field | Meaning |
|---|---|
| `trace_id`, `agent_id`, `operation` | as given (escaped, capped at 256 characters) |
| `span_id` | 16 hex characters; the OpenTelemetry span id when the bridge is on |
| `parent_id` | the `parent_id=` given to `start_trace`, else `None` |
| `otel_trace_id` | 32 hex characters, only when the bridge produced a recording span |
| `start_time`, `end_time` | ISO-8601 UTC wall-clock timestamps (`end_time` is `None` while active) |
| `duration_ms` | measured on a monotonic clock; `None` while active |
| `status` | `"active"` until ended; then the status given to `end_trace` |
| `result` | cleaned copy of the result (see [what is stored](#what-is-stored-and-how)) |
| `threats` | threat labels behind the outcome (`[]` when none) |
| `decision` | `"block"` / `"allow"` when threats were recorded, else `None` |
| `report_only` | the threats were found in report-only mode, nothing was refused |
| `error_type` | for an error status: the exception class, `"blocked"`, or `"_OTHER"` |
| `metadata`, `events` | cleaned copies; events bounded by `max_events_per_trace` |

`end_trace` keeps its original call shape and takes three optional keywords:

```python
obs.end_trace(trace_id, status="error", result="...")          # as before
obs.end_trace(trace_id, status="error", threats=["firewall"])  # structured labels
obs.end_trace(trace_id, status="error", error=exc)             # labels from exc.threats
obs.end_trace(trace_id, threats=["firewall"], report_only=True)
```

With `error=`, the exception's class becomes `error_type`, its message (capped at
`max_error_length`) becomes the result if none was given, and a `threats`
attribute -- `SecurityBlockedError` has one -- supplies the threat labels.
`"error"`, `"failed"` and `"blocked"` are error statuses.

Query traces (optionally filtered, with an efficient most-recent `limit`):

```python
obs.get_traces(agent_id="agent-007")
obs.get_traces(status="active", limit=10)
```

Readers always get **snapshots**: mutating a returned trace changes nothing in
the observer, and a writer updating the trace cannot disturb a reader iterating
it.

**Eviction.** When `max_traces` is exceeded the oldest *completed* trace is
evicted. A still-running trace is evicted only when every stored trace is
active; that is counted (`get_stats()["active_traces_evicted"]`), logged as a
warning, and its OpenTelemetry span is ended with `error.type="evicted"`. A later
`end_trace` for an evicted (or never-started) id is counted in
`end_trace_unknown`. Reusing a trace id replaces the old record and is counted in
`duplicate_trace_ids`.

### Refusals

A request refused on *input* is refused before the governed run starts, so it has
no trace to end. Record it with `record_refusal`; refusals have their own buffer,
so trace counts keep meaning "runs that started":

```python
obs.record_refusal("agent-007", "answer_question", ["prompt_injection"], stage="input")
obs.record_refusal("agent-007", "answer_question", error=exc, stage="policy")  # labels from exc

obs.get_refusals(agent_id="agent-007", limit=20)
# [{'agent_id': 'agent-007', 'stage': 'policy', 'threats': ['policy:low_trust'],
#   'decision': 'block', 'error_type': 'SecurityBlockedError', 'reason': '...', ...}]
```

### Tool decisions

Framework bindings report each tool call and the gate's
[`ToolDecision`](integrations.md) on it. Two forms:

```python
# One verdict, e.g. a block where the tool never runs:
obs.record_tool_decision(agent_id, decision, parent_id=trace_id, tool_call_id=call_id)

# Time the tool call and record both verdicts on it:
with obs.tool_span(agent_id, tool_name, parent_id=trace_id, tool_call_id=call_id) as span:
    decision = gate.check_tool_call(tool_name, arguments)
    span.record(decision, "call")
    if not decision.allowed:
        return decision.reason
    result = run_tool(arguments)
    span.record(gate.check_tool_result(tool_name, result), "result")
```

`decision` is duck-typed: anything with `action` (`"allow"` / `"block"` /
`"ask"`) and optionally `tool_name`, `threats`, `report_only`, `agent_id`. A
`agent_id=None` falls back to `decision.agent_id`. The tool record keeps the most
severe verdict (`block` > `ask` > `allow`), the union of threat labels, the
phases recorded, `duration_ms`, and a `status` of `"completed"`, `"blocked"`,
`"approval_required"` or `"error"` (the block raised). With a known `parent_id`
the verdict is also logged as a `tool_decision` event on that trace. Read them
back with `obs.get_tool_decisions(agent_id=..., tool_name=..., limit=...)`.

### Logs

Logs are independent of traces.

```python
obs.log("info", "monitoring initialised", agent_id="agent-007")
obs.log("error", "downstream timeout", agent_id="agent-007", metadata={"ms": 30000})

obs.get_logs(level="error")
obs.get_logs(agent_id="agent-007", limit=50)
```

### Metrics

Metrics are named numeric series. `record_metric` appends a value; the series is
bounded by `max_metrics` and the number of distinct names by `max_metric_names`.

```python
obs.record_metric("latency_ms", 142.0)
obs.record_metric("latency_ms", 98.5)

obs.get_metric_stats("latency_ms")
# {'count': 2, 'min': 98.5, 'max': 142.0, 'avg': 120.25}
```

A value must be a finite `int` or `float`: `NaN` and `±inf` raise `ValueError`
(one `NaN` would make every later `min`/`max`/`avg` of the series meaningless)
and a non-number -- `bool` included -- raises `TypeError`. `get_metric_stats`
returns `{}` for an unknown or empty metric; the mean is computed with
`math.fsum`.

### What is stored, and how

The observer exists to be read by people and log pipelines, and what it stores is
largely attacker-influenced (prompts, model output, exception text). So:

- **Escaping.** Every stored string -- messages, descriptions, ids, event types,
  levels, statuses, threat labels, results, metadata keys and values at any depth
  -- has every Unicode `Cc` (C0/C1 controls, DEL, ESC, `\x85`, ...), `Zl`
  (U+2028) and `Zp` (U+2029) character escaped (`\n`, `\r`, `\t`, `\xNN`,
  `\uNNNN`), plus the bidirectional-override controls (U+202A-202E,
  U+2066-2069, U+200E/200F, U+061C). A stored string never contains a character
  `str.splitlines()` or a terminal treats as a line break or control sequence, so
  a logged value cannot forge a log line or rewrite a terminal. Filters
  (`get_traces(agent_id=...)`, ...) escape their argument the same way.
- **Length caps.** Free text is capped at `max_text_length`; error text (the
  result of an error-status trace, exception messages, refusal reasons, span
  status descriptions) at `max_error_length` (200 by default) -- exception text
  routinely quotes the prompt or a credential, so it is never kept at full
  length by default. Identifiers are capped at 256 characters. A truncated value
  ends in `...[truncated]`.
- **Bounded copies.** Metadata and results are copied, not referenced: mappings
  become `dict`s and sequences `list`s of at most 100 items, at most 4 levels
  deep (deeper values become `"[truncated: nested too deep]"`). `None`, `bool`,
  `int` and `float` are kept as is; any other object is stored as its escaped
  `str()`.
- **Redaction.** `redactor` is a `str -> str` callable applied to every stored
  free-text string *before* escaping and truncation (identifiers are not
  redacted, so filters keep working). The default stores text escaped and
  capped but otherwise verbatim. `adapt_agent.observability.redact_secrets` is a
  ready-made redactor for common credential shapes (`sk-...`, AWS `AKIA...`,
  GitHub and Slack tokens, `Bearer ...`, `password=...`); a deployment with its
  own secret formats should supply its own. A redactor that raises makes the
  value be stored as `"[redaction failed]"`, never verbatim.

`escape_text` is exported for code that wants the same guarantee elsewhere.

### Thread safety

One observer may be shared by every thread: the governed adapters call it from
worker threads (`asyncio.to_thread`) while monitoring code reads it. All state is
guarded by a single lock held only for a few dict/deque operations; text cleaning
and all OpenTelemetry calls run outside it. Measured on CPython 3.13 the
uncontended cost is about 0.3 µs per `record_metric` and 7 µs per
`start_trace`/`end_trace` pair -- negligible next to a model call.

### Stats

`obs.get_stats()` reports what the observer dropped, replaced or failed to export:
`traces_evicted`, `active_traces_evicted`, `duplicate_trace_ids`,
`end_trace_unknown`, `logs_evicted`, `refusals_recorded`,
`tool_decisions_recorded`, `otel_errors`, and current sizes (`traces_stored`,
`traces_active`, `logs_stored`, `metric_names_stored`). `active_traces_evicted`
and `end_trace_unknown` should stay at zero; if they do not, `max_traces` is
smaller than the number of concurrently running traces, or a caller never ends
its traces.

### OpenTelemetry bridge

Install the extra (`pip install "adapt-agent[otel]"`, which brings
`opentelemetry-api`; add `opentelemetry-sdk` and an exporter to actually ship
spans) and opt in. Without it nothing changes: the default observer never
imports OpenTelemetry, and `otel=True` without the package installed raises
`MissingDependencyError` rather than silently exporting nothing.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

observer = AgentObserver(otel=True)              # global provider
# observer = AgentObserver(otel=provider.get_tracer("adapt_agent"))  # or a given tracer
```

Each `start_trace` opens a span **as a child of the current OpenTelemetry
context** and makes it current until `end_trace`. So:

- inside a framework's own instrumentation (a native hook -- a LangGraph node, a
  Microsoft Agent Framework middleware, a Google ADK callback -- running under
  the framework's agent span), the governance span nests **under** the
  framework's span;
- around a framework (a governed adapter's `execute`/`aexecute`), the
  framework's spans nest **under** the governance span (`asyncio.to_thread`
  copies the context into the worker thread, so this holds for sync frameworks
  driven from `aexecute` too).

```text
invoke_agent framework-agent            <- the framework's span
└── invoke_agent billing                <- AgentObserver.start_trace / end_trace
    ├── chat <model>                    <- the framework's model span
    └── execute_tool shell  [ERROR]     <- record_tool_decision(... block ...)
```

A `parent_id=` naming a trace the observer knows makes that trace's span the
parent instead.

Spans follow the OpenTelemetry **GenAI semantic conventions**:

| Span | Name | Attributes |
|---|---|---|
| trace / refusal | `invoke_agent {agent_id}` | `gen_ai.operation.name="invoke_agent"`, `gen_ai.agent.id`, `gen_ai.agent.name` |
| tool decision | `execute_tool {tool_name}` | `gen_ai.operation.name="execute_tool"`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.agent.id` |

On a failed, blocked or refused run (or a blocked tool call) the span status is
`ERROR` and `error.type` is the exception class, `"blocked"`, or the
convention's fallback `"_OTHER"`; on success the status is left `UNSET`.
Governance attributes use the `adapt_agent.*` namespace:
`adapt_agent.trace_id`, `adapt_agent.operation`, `adapt_agent.status`,
`adapt_agent.stage`, `adapt_agent.decision` (`allow`/`block`/`ask`),
`adapt_agent.threats`, `adapt_agent.policy.rules` (from `policy:<rule>` labels),
and `adapt_agent.report_only`. `log_event` calls become span events (name = the
event type, `adapt_agent.description` plus `adapt_agent.event.<key>` for
metadata), and threat labels are added as an `adapt_agent.threat_detected` event.
Span content goes through the same escaping, caps and redactor as stored text;
prompts and model output are never put on spans.

To change the span vocabulary, pass a bridge:

```python
from adapt_agent.observability.otel import OTelBridge

observer = AgentObserver(otel=OTelBridge(tracer, agent_operation_name="invoke_agent", activate=True))
```

`agent_operation_name` exists because a framework that emits its own
`invoke_agent` span will show two per run; a backend that counts agent
invocations from spans can filter on `adapt_agent.trace_id` or use another name.

**Limits.** `start_trace` and `end_trace` are separate calls, so the span is
made current with a context token. If `end_trace` runs in a different thread or
task than `start_trace`, the span is still ended correctly, but the token cannot
be detached from the other context, which keeps the (ended) span as its current
span until that thread or task finishes. Pass `OTelBridge(..., activate=False)`
if your code starts and ends traces in different contexts. The bridge is
telemetry, not a control: an exception from the OpenTelemetry pipeline is
counted in `otel_errors` and logged, never raised into the request.

---

## AgentEvaluator

`adapt_agent.evaluation.AgentEvaluator` scores agent responses using custom
metrics and aggregates them.

```python
evaluator = AgentEvaluator(max_results=1000)
```

### Registering metrics

A metric is a callable `metric_func(output_data, expected_output) -> float`.

```python
def exact_match(output, expected):
    return 1.0 if output == expected else 0.0

def length_ratio(output, expected):
    return len(str(output)) / max(len(str(expected)), 1)

evaluator.register_metric("exact_match", exact_match)
evaluator.register_metric("length_ratio", length_ratio)
```

### Evaluating responses

`evaluate_response` runs every registered metric and stores the result. A metric
that raises is recorded as `None` for that response (it does not abort the
evaluation).

```python
result = evaluator.evaluate_response(
    agent_id="agent-007",
    input_data="capital of France?",
    output_data="Paris",
    expected_output="Paris",
)
print(result["metrics"])  # {'exact_match': 1.0, 'length_ratio': 1.0}
```

### Aggregating and querying

```python
evaluator.compute_aggregate_metrics()                 # mean of each metric across all results
evaluator.compute_aggregate_metrics(agent_id="agent-007")  # filtered

evaluator.get_evaluation_results(agent_id="agent-007", limit=20)
```

`compute_aggregate_metrics` averages each metric over results where the score is
not `None`. The result store is bounded by `max_results` (oldest dropped).

---

## AgentOptimizer

`adapt_agent.optimization.AgentOptimizer` records performance samples and
suggests improvements.

```python
optimizer = AgentOptimizer(max_metrics=1000, max_suggestions=1000)
```

### Recording performance

`analyze_performance` stores a sample and returns running statistics for that
agent.

```python
stats = optimizer.analyze_performance(
    agent_id="agent-007",
    execution_time=6.2,     # seconds
    token_usage=1500,
    success=True,
)
print(stats)
# {'total_executions': 1, 'avg_execution_time': 6.2, 'success_rate': 1.0}
```

### Generating suggestions

`suggest_optimizations` inspects the recorded samples for an agent and returns
suggestions. Built-in heuristics fire when:

- average execution time exceeds **5.0 seconds** → a `performance` suggestion;
- average token usage exceeds **1000 tokens** → an `efficiency` suggestion.

```python
suggestions = optimizer.suggest_optimizations("agent-007")
for s in suggestions:
    print(s["type"], s["severity"], "-", s["suggestion"], "(", s["value"], ")")
# performance medium - Consider caching frequently accessed data or using faster models ( 6.2 )
# efficiency low - High token usage detected. Consider prompt optimization ( 1500.0 )
```

Both the metrics list and the suggestions list are bounded (`max_metrics`,
`max_suggestions`); oldest entries are evicted when the caps are exceeded.
