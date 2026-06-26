"""Example 04: Observability and evaluation.

Two helpers are demonstrated:

* ``AgentObserver`` - records traces (spans with events), free-form logs, and
  numeric metrics. You can query traces and per-metric statistics.
* ``AgentEvaluator`` - scores agent responses with custom metric functions and
  aggregates those scores across many evaluations.

Run it with:

    python examples/04_observability.py
"""

from pprint import pprint

from adapt_agent import AgentEvaluator, AgentObserver


def observer_demo() -> None:
    print("=== AgentObserver ===")
    observer = AgentObserver()

    # Start a trace, log a couple of events inside it, then end it.
    trace_id = "trace-001"
    observer.start_trace(trace_id, agent_id="demo-agent", operation="answer_question")
    observer.log_event(trace_id, event_type="tool_call", description="Looked up the weather.")
    observer.log_event(trace_id, event_type="response", description="Generated final answer.")
    observer.end_trace(trace_id, status="completed", result="ok")

    # A free-form log entry (not tied to a trace).
    observer.log("info", "Agent finished a request.", agent_id="demo-agent")

    # Record some latency measurements (in milliseconds).
    for latency in (120.0, 95.5, 210.0, 150.0):
        observer.record_metric("latency_ms", latency)

    print("Metric stats for 'latency_ms':")
    pprint(observer.get_metric_stats("latency_ms"))

    print("\nTraces:")
    for trace in observer.get_traces():
        print(f"  {trace['trace_id']} status={trace['status']} " f"events={len(trace['events'])}")


def evaluator_demo() -> None:
    print("\n=== AgentEvaluator ===")
    evaluator = AgentEvaluator()

    # A custom metric receives (output, expected) and returns a float score.
    # Here: 1.0 for an exact match, otherwise 0.0.
    def exact_match(output, expected) -> float:
        return 1.0 if output == expected else 0.0

    evaluator.register_metric("exact_match", exact_match)

    samples = [
        ("2 + 2", "4", "4"),  # correct
        ("3 * 3", "9", "9"),  # correct
        ("capital of France", "London", "Paris"),  # incorrect
    ]
    for input_data, output, expected in samples:
        evaluator.evaluate_response(
            agent_id="demo-agent",
            input_data=input_data,
            output_data=output,
            expected_output=expected,
        )

    print("Aggregate metrics across all evaluations:")
    pprint(evaluator.compute_aggregate_metrics(agent_id="demo-agent"))


def main() -> None:
    observer_demo()
    evaluator_demo()


if __name__ == "__main__":
    main()
