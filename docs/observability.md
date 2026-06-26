# Observability, Evaluation & Optimization

This page covers the three monitoring and quality components:
[`AgentObserver`](#agentobserver),
[`AgentEvaluator`](#agentevaluator), and
[`AgentOptimizer`](#agentoptimizer). All three keep bounded in-memory state.

```python
from adapt_agent import AgentObserver, AgentEvaluator, AgentOptimizer
```

---

## AgentObserver

`adapt_agent.observability.AgentObserver` records four kinds of telemetry:
**traces**, per-trace **events**, free-form **logs**, and numeric **metrics**.

```python
obs = AgentObserver(
    max_logs=1000,
    max_traces=1000,
    max_metrics=1000,           # per metric-name series
    max_events_per_trace=1000,
    max_metric_names=1000,
)
```

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

Query traces (optionally filtered, with an efficient most-recent `limit`):

```python
obs.get_traces(agent_id="agent-007")
obs.get_traces(status="active", limit=10)
```

Each trace stores `events` bounded by `max_events_per_trace`; the trace registry
itself is bounded by `max_traces` (oldest evicted).

### Logs

Logs are independent of traces. Both `log` messages and `log_event` descriptions
are truncated to 10,000 characters and have newline/carriage-return characters
escaped — a **log-poisoning** defense.

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

`get_metric_stats` returns `{}` for an unknown or empty metric.

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
