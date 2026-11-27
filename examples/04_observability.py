"""Example 04: Observability and evaluation.

Three things are demonstrated:

* ``AgentObserver`` - records traces (spans with events, durations and threat
  labels), refusals, tool verdicts, free-form logs, and numeric metrics. Every
  stored string is escaped and length-capped, so a hostile value cannot forge a
  log line.
* The optional OpenTelemetry bridge - when ``opentelemetry-sdk`` is installed
  the same traces become real spans nested under the framework's own; without
  it that part is skipped. The example stays offline either way.
* ``AgentEvaluator`` - scores agent responses with custom metric functions and
  aggregates those scores across many evaluations.

Run it with:

    python examples/04_observability.py
"""

from pprint import pprint

from adapt_agent import AgentEvaluator, AgentObserver
from adapt_agent.core.governance import ToolDecision
from adapt_agent.exceptions import SecurityBlockedError
from adapt_agent.observability import redact_secrets


def observer_demo() -> None:
    print("=== AgentObserver ===")
    # redact_secrets scrubs common credential shapes from stored free text.
    observer = AgentObserver(redactor=redact_secrets)

    # A served request: start a trace, log events, record a tool verdict, end it.
    trace_id = "trace-001"
    observer.start_trace(trace_id, agent_id="demo-agent", operation="answer_question")
    observer.log_event(trace_id, event_type="tool_call", description="Looked up the weather.")
    observer.record_tool_decision(
        "demo-agent", ToolDecision("allow", "get_weather"), parent_id=trace_id
    )
    observer.log_event(trace_id, event_type="response", description="Generated final answer.")
    observer.end_trace(trace_id, status="completed", result="ok")

    # A request whose *output* was blocked: the exception carries the threat
    # labels, and the observer records them together with the error type.
    observer.start_trace("trace-002", agent_id="demo-agent", operation="answer_question")
    blocked = SecurityBlockedError("Output blocked by security controls", ["firewall"])
    observer.end_trace("trace-002", status="error", error=blocked)

    # A request refused on *input* never opened a trace; record it as a refusal.
    observer.record_refusal("demo-agent", "answer_question", ["prompt_injection"], stage="input")

    # A free-form log entry. The forged second line and the API key are stored
    # escaped and redacted.
    observer.log(
        "info",
        "Agent finished a request.\n[ERROR] forged line, key=sk-live-0123456789abcdef",
        agent_id="demo-agent",
    )

    # Record some latency measurements (in milliseconds).
    for latency in (120.0, 95.5, 210.0, 150.0):
        observer.record_metric("latency_ms", latency)

    print("Metric stats for 'latency_ms':")
    pprint(observer.get_metric_stats("latency_ms"))

    print("\nTraces:")
    for trace in observer.get_traces():
        print(
            f"  {trace['trace_id']} status={trace['status']} "
            f"duration_ms={trace['duration_ms']} threats={trace['threats']} "
            f"error_type={trace['error_type']} events={len(trace['events'])}"
        )

    print("\nRefusals:")
    for refusal in observer.get_refusals():
        print(f"  {refusal['agent_id']} stage={refusal['stage']} threats={refusal['threats']}")

    print("\nTool decisions:")
    for tool in observer.get_tool_decisions():
        print(f"  {tool['tool_name']} decision={tool['decision']} status={tool['status']}")

    print("\nStored log message (escaped and redacted):")
    print(" ", observer.get_logs()[-1]["message"])
    print("\nObserver stats:", observer.get_stats())


def otel_demo() -> None:
    print("\n=== OpenTelemetry bridge ===")
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
    except ImportError:
        print(
            "opentelemetry-sdk is not installed; skipping (pip install 'adapt-agent[otel]' opentelemetry-sdk)."
        )
        return

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    framework = provider.get_tracer("my-agent-framework")
    observer = AgentObserver(otel=provider.get_tracer("adapt_agent"))

    # The framework's own agent span is current when governance runs inside it.
    with framework.start_as_current_span("invoke_agent framework-agent"):
        observer.start_trace("otel-001", agent_id="demo-agent", operation="execute")
        observer.record_tool_decision(
            "demo-agent",
            ToolDecision("block", "shell", ("tool:denied:shell",)),
            parent_id="otel-001",
        )
        observer.end_trace("otel-001", status="completed")

    spans = exporter.get_finished_spans()
    names = {span.context.span_id: span.name for span in spans}
    for span in sorted(spans, key=lambda s: s.start_time or 0):
        parent = names.get(span.parent.span_id) if span.parent else None
        print(f"  {span.name!r:32} parent={parent!r:32} status={span.status.status_code.name}")


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
    otel_demo()
    evaluator_demo()


if __name__ == "__main__":
    main()
