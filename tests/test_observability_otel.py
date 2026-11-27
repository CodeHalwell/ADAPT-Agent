"""The optional OpenTelemetry bridge of AgentObserver.

Spans are asserted through the SDK's in-memory exporter on a *local* tracer
provider, so nothing here touches (or depends on) the global provider other
tests or frameworks may have configured.
"""

import asyncio
import logging
import sys
import threading

import pytest

from adapt_agent.core.governance import ToolDecision
from adapt_agent.exceptions import MissingDependencyError, SecurityBlockedError
from adapt_agent.observability import AgentObserver
from adapt_agent.observability import otel as sc

# -- without opentelemetry -----------------------------------------------------


@pytest.fixture
def no_opentelemetry(monkeypatch):
    for name in list(sys.modules):
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "opentelemetry", None)


def test_the_default_observer_never_needs_opentelemetry(no_opentelemetry):
    obs = AgentObserver()
    obs.start_trace("t", "a", "op")
    obs.log_event("t", "e", "d")
    obs.end_trace("t", status="error", threats=["firewall"])
    obs.record_refusal("a", "op", ["x"])
    obs.record_tool_decision("a", ToolDecision("allow", "tool"))
    assert obs.otel is None
    assert obs.get_traces()[0]["threats"] == ["firewall"]
    assert "otel_trace_id" not in obs.get_traces()[0]


def test_asking_for_otel_without_it_installed_fails_loudly(no_opentelemetry):
    with pytest.raises(MissingDependencyError, match=r"adapt-agent\[otel\]"):
        AgentObserver(otel=True)


# -- with the SDK --------------------------------------------------------------

pytest.importorskip("opentelemetry.trace")
sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
from opentelemetry import trace as otel_trace  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode  # noqa: E402


@pytest.fixture(autouse=True)
def _otel_sdk_enabled(monkeypatch):
    """Keep the SDK switched on for these tests whatever the process says.

    ``OTEL_SDK_DISABLED`` is read when a ``TracerProvider`` is built and turns
    it into a no-op. Other test modules set it for the whole process (it is
    CrewAI's telemetry opt-out), which would make every span assertion here
    fail for a reason that has nothing to do with the bridge.
    """
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)


@pytest.fixture
def otel():
    exporter = InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    framework = provider.get_tracer("framework")
    observer = AgentObserver(otel=provider.get_tracer("adapt_agent"))
    return observer, framework, exporter


def _by_name(exporter):
    return {span.name: span for span in exporter.get_finished_spans()}


def test_a_governance_span_nests_under_the_frameworks_current_span(otel):
    observer, framework, exporter = otel
    with framework.start_as_current_span("invoke_agent framework") as outer:
        observer.start_trace("t1", "billing", "execute")
        observer.end_trace("t1")
        # The framework's context is restored once the trace ends.
        assert otel_trace.get_current_span() is outer

    spans = _by_name(exporter)
    gov = spans["invoke_agent billing"]
    assert gov.parent.span_id == spans["invoke_agent framework"].context.span_id
    assert gov.attributes[sc.GEN_AI_OPERATION_NAME] == "invoke_agent"
    assert gov.attributes[sc.GEN_AI_AGENT_ID] == "billing"
    assert gov.attributes[sc.GEN_AI_AGENT_NAME] == "billing"
    assert gov.attributes[sc.ADAPT_TRACE_ID] == "t1"
    assert gov.attributes[sc.ADAPT_OPERATION] == "execute"
    assert gov.status.status_code is StatusCode.UNSET

    record = observer.get_traces()[0]
    assert record["span_id"] == format(gov.context.span_id, "016x")
    assert record["otel_trace_id"] == format(gov.context.trace_id, "032x")


def test_spans_opened_during_a_trace_nest_under_the_governance_span(otel):
    observer, framework, exporter = otel
    observer.start_trace("t1", "a", "execute")
    with framework.start_as_current_span("model call"):
        pass
    observer.end_trace("t1")
    spans = _by_name(exporter)
    assert spans["model call"].parent.span_id == spans["invoke_agent a"].context.span_id
    assert otel_trace.get_current_span() is otel_trace.INVALID_SPAN


def test_an_explicit_parent_trace_is_the_parent_span(otel):
    observer, _framework, exporter = otel
    observer.start_trace("outer", "planner", "execute")
    observer.start_trace("inner", "worker", "execute", parent_id="outer")
    observer.end_trace("inner")
    observer.end_trace("outer")
    spans = _by_name(exporter)
    assert spans["invoke_agent worker"].parent.span_id == (
        spans["invoke_agent planner"].context.span_id
    )


def test_a_blocked_run_is_an_error_span_with_threats_and_events(otel):
    observer, _framework, exporter = otel
    observer.start_trace("t1", "billing", "execute")
    observer.log_event("t1", "input_screened", "passed", metadata={"tokens": 12})
    observer.end_trace(
        "t1",
        status="error",
        error=SecurityBlockedError("Output blocked", ["firewall", "policy:no_pii"]),
    )
    span = _by_name(exporter)["invoke_agent billing"]
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "Output blocked"
    assert span.attributes[sc.ERROR_TYPE] == "SecurityBlockedError"
    assert span.attributes[sc.ADAPT_DECISION] == "block"
    assert tuple(span.attributes[sc.ADAPT_THREATS]) == ("firewall", "policy:no_pii")
    assert tuple(span.attributes[sc.ADAPT_POLICY_RULES]) == ("no_pii",)
    names = [e.name for e in span.events]
    assert names == ["input_screened", sc.EVENT_THREAT_DETECTED]
    assert span.events[0].attributes["adapt_agent.event.tokens"] == 12


def test_an_error_without_an_exception_uses_the_conventions_fallback_type(otel):
    observer, _framework, exporter = otel
    observer.start_trace("t1", "a", "op")
    observer.end_trace("t1", status="error", result="boom")
    span = _by_name(exporter)["invoke_agent a"]
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes[sc.ERROR_TYPE] == sc.ERROR_TYPE_OTHER


def test_report_only_threats_do_not_make_an_error_span(otel):
    observer, _framework, exporter = otel
    observer.start_trace("t1", "a", "op")
    observer.end_trace("t1", threats=["firewall"], report_only=True)
    span = _by_name(exporter)["invoke_agent a"]
    assert span.status.status_code is StatusCode.UNSET
    assert span.attributes[sc.ADAPT_REPORT_ONLY] is True
    assert span.attributes[sc.ADAPT_DECISION] == "allow"


def test_a_refusal_is_a_zero_length_error_span(otel):
    observer, framework, exporter = otel
    with framework.start_as_current_span("request"):
        observer.record_refusal("billing", "execute", ["prompt_injection"], stage="input")
        assert otel_trace.get_current_span().name == "request"  # refusal never activates
    spans = _by_name(exporter)
    span = spans["invoke_agent billing"]
    assert span.parent.span_id == spans["request"].context.span_id
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes[sc.ERROR_TYPE] == sc.ERROR_TYPE_BLOCKED
    assert span.attributes[sc.ADAPT_STAGE] == "input"
    assert tuple(span.attributes[sc.ADAPT_THREATS]) == ("prompt_injection",)
    assert observer.get_refusals()[0]["span_id"] == format(span.context.span_id, "016x")


def test_tool_spans_follow_the_genai_conventions(otel):
    observer, framework, exporter = otel
    observer.start_trace("run", "billing", "execute")
    with observer.tool_span("billing", "search", parent_id="run", tool_call_id="c1") as span:
        span.record(ToolDecision("allow", "search"), "call")
        with framework.start_as_current_span("http GET"):
            pass
    observer.record_tool_decision(
        None,
        ToolDecision("block", "shell", ("tool:denied:shell",), False, "billing"),
        parent_id="run",
    )
    observer.end_trace("run")

    spans = _by_name(exporter)
    run = spans["invoke_agent billing"]
    search = spans["execute_tool search"]
    shell = spans["execute_tool shell"]
    assert search.parent.span_id == run.context.span_id
    assert spans["http GET"].parent.span_id == search.context.span_id
    assert search.attributes[sc.GEN_AI_OPERATION_NAME] == "execute_tool"
    assert search.attributes[sc.GEN_AI_TOOL_NAME] == "search"
    assert search.attributes[sc.GEN_AI_TOOL_CALL_ID] == "c1"
    assert search.attributes[sc.ADAPT_DECISION] == "allow"
    assert search.status.status_code is StatusCode.UNSET
    assert shell.status.status_code is StatusCode.ERROR
    assert shell.attributes[sc.ADAPT_DECISION] == "block"
    assert tuple(shell.attributes[sc.ADAPT_THREATS]) == ("tool:denied:shell",)
    assert [e.name for e in run.events] == ["tool_decision", "tool_decision"]


def test_span_content_is_escaped_and_capped(otel):
    observer, _framework, exporter = otel
    observer.start_trace("t", "a\nb", "op")
    observer.log_event("t", "evt\x1b[2K", "line1\u2028line2")
    observer.end_trace("t", status="error", error=RuntimeError("key=sk-live-1\n" + "x" * 5000))
    span = exporter.get_finished_spans()[0]
    assert span.name == "invoke_agent a\\nb"
    assert span.events[0].name == "evt\\x1b[2K"
    assert span.events[0].attributes[sc.ADAPT_DESCRIPTION] == "line1\\u2028line2"
    assert len(span.status.description) <= observer.max_error_length
    assert "\n" not in span.status.description


def test_ending_a_trace_from_another_thread_ends_the_span_without_errors(otel, caplog):
    """The token cannot be detached from another thread's context (documented).

    Both ends run in their own threads so the starting context -- which keeps
    the span attached -- is discarded with its thread rather than leaking into
    the tests that run after this one on the main thread.
    """
    observer, _framework, exporter = otel
    caplog.set_level(logging.ERROR, logger="opentelemetry.context")
    for target in (observer.start_trace, observer.end_trace):
        args = ("t", "a", "op") if target == observer.start_trace else ("t",)
        worker = threading.Thread(target=target, args=args)
        worker.start()
        worker.join()
    assert otel_trace.get_current_span() is otel_trace.INVALID_SPAN
    assert len(exporter.get_finished_spans()) == 1
    assert not [r for r in caplog.records if "detach" in r.getMessage().lower()]


def test_an_evicted_active_trace_ends_its_span(otel):
    exporter = otel[2]
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    observer = AgentObserver(max_traces=1, otel=provider.get_tracer("t"))
    observer.start_trace("a", "x", "op")
    observer.start_trace("b", "x", "op")
    ended = exporter.get_finished_spans()
    assert len(ended) == 1 and ended[0].attributes[sc.ERROR_TYPE] == "evicted"
    assert observer.get_stats()["active_traces_evicted"] == 1


def test_a_failing_span_processor_never_fails_the_request():
    class Exploding(SimpleSpanProcessor):
        def on_end(self, span):
            raise RuntimeError("exporter down")

    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(Exploding(InMemorySpanExporter()))
    observer = AgentObserver(otel=provider.get_tracer("t"))
    observer.start_trace("t", "a", "op")
    observer.end_trace("t")
    # Whether the SDK swallows the processor error or it reaches the observer
    # (and is counted there), the run itself is recorded and nothing raised.
    assert observer.get_traces()[0]["status"] == "completed"


def test_a_raising_bridge_is_counted_not_propagated():
    class BrokenTracer:
        def start_span(self, *args, **kwargs):
            raise RuntimeError("broken")

    observer = AgentObserver(otel=BrokenTracer())
    observer.start_trace("t", "a", "op")
    observer.end_trace("t")
    assert observer.get_traces()[0]["status"] == "completed"
    assert observer.get_stats()["otel_errors"] == 1


def test_otel_true_uses_the_global_provider_and_tolerates_a_noop_one():
    observer = AgentObserver(otel=True)
    observer.start_trace("t", "a", "op")
    observer.end_trace("t", status="error")
    assert observer.get_traces()[0]["status"] == "error"
    assert len(observer.get_traces()[0]["span_id"]) == 16


def test_an_otel_bridge_instance_is_accepted(otel):
    _observer, framework, exporter = otel
    bridge = sc.OTelBridge(tracer=framework, agent_operation_name="governance")
    observer = AgentObserver(otel=bridge)
    observer.start_trace("t", "a", "op")
    observer.end_trace("t")
    span = exporter.get_finished_spans()[0]
    assert span.name == "governance a"
    assert span.attributes[sc.GEN_AI_OPERATION_NAME] == "governance"
    with pytest.raises(TypeError):
        sc.OTelBridge(tracer=object())


# -- through the real call sites -------------------------------------------------


def test_the_integrations_traced_helper_nests_under_the_framework(otel):
    from adapt_agent.integrations._common import traced

    observer, framework, exporter = otel
    with framework.start_as_current_span("invoke_agent native"):
        with pytest.raises(ValueError):
            with traced(observer, "worker", "hook"):
                raise ValueError("bad")
    spans = _by_name(exporter)
    gov = spans["invoke_agent worker"]
    assert gov.parent.span_id == spans["invoke_agent native"].context.span_id
    assert gov.status.status_code is StatusCode.ERROR


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_a_governed_adapter_parents_the_frameworks_spans(otel, mode):
    from adapt_agent.adapters import LangGraphAdapter

    observer, framework, exporter = otel

    def runner(payload):
        with framework.start_as_current_span("graph step"):
            return {"messages": [{"role": "assistant", "content": "ok"}]}

    agent = LangGraphAdapter(observer=observer, agent_id="lg").wrap_agent(runner)
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    if mode == "sync":
        agent.execute(payload)
    else:
        asyncio.run(agent.aexecute(payload))
    spans = _by_name(exporter)
    assert spans["graph step"].parent.span_id == spans["invoke_agent lg"].context.span_id


# -- the attribute names are the conventions' ------------------------------------


def test_attribute_names_match_the_installed_semantic_conventions():
    gen_ai = pytest.importorskip("opentelemetry.semconv._incubating.attributes.gen_ai_attributes")
    errors = pytest.importorskip("opentelemetry.semconv.attributes.error_attributes")
    assert sc.GEN_AI_OPERATION_NAME == gen_ai.GEN_AI_OPERATION_NAME
    assert sc.GEN_AI_AGENT_ID == gen_ai.GEN_AI_AGENT_ID
    assert sc.GEN_AI_AGENT_NAME == gen_ai.GEN_AI_AGENT_NAME
    assert sc.GEN_AI_TOOL_NAME == gen_ai.GEN_AI_TOOL_NAME
    assert sc.GEN_AI_TOOL_CALL_ID == gen_ai.GEN_AI_TOOL_CALL_ID
    assert sc.ERROR_TYPE == errors.ERROR_TYPE
    ops = gen_ai.GenAiOperationNameValues
    assert sc.OPERATION_INVOKE_AGENT == ops.INVOKE_AGENT.value
    assert sc.OPERATION_EXECUTE_TOOL == ops.EXECUTE_TOOL.value
    assert sc.ERROR_TYPE_OTHER == errors.ErrorTypeValues.OTHER.value
