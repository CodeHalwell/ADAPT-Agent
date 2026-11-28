"""AgentObserver hardening: what is stored, how it is escaped, eviction, locking, metrics."""

import math
import sys
import threading
import unicodedata

import pytest

from adapt_agent.core.governance import ToolDecision
from adapt_agent.exceptions import SecurityBlockedError
from adapt_agent.observability import AgentObserver, escape_text, redact_secrets

_UNSAFE_CATEGORIES = {"Cc", "Zl", "Zp"}


def _all_stored_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _all_stored_strings(k)
            yield from _all_stored_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _all_stored_strings(v)


def _assert_clean(value):
    for text in _all_stored_strings(value):
        assert len(text.splitlines()) <= 1, repr(text)
        assert not any(unicodedata.category(c) in _UNSAFE_CATEGORIES for c in text), repr(text)


# -- 1. timing and ids ---------------------------------------------------------


def test_a_trace_records_timestamps_duration_and_ids():
    obs = AgentObserver()
    obs.start_trace("parent", "a", "op")
    started = obs.start_trace("child", "a", "op", parent_id="parent")
    assert started["parent_id"] == "parent"
    assert len(started["span_id"]) == 16 and int(started["span_id"], 16) >= 0
    assert started["end_time"] is None and started["duration_ms"] is None

    obs.end_trace("child")
    child = obs.get_traces(limit=1)[0]
    assert child["end_time"] is not None
    assert isinstance(child["duration_ms"], float) and child["duration_ms"] >= 0
    assert child["span_id"] != obs.get_traces()[0]["span_id"]


# -- 2. blocked requests -------------------------------------------------------


def test_end_trace_stores_threat_labels_and_derives_them_from_the_error():
    obs = AgentObserver()
    obs.start_trace("t1", "a", "op")
    obs.end_trace("t1", status="error", threats=["firewall", "policy:no_pii"])
    t1 = obs.get_traces()[0]
    assert t1["threats"] == ["firewall", "policy:no_pii"]
    assert t1["decision"] == "block"
    assert t1["error_type"] == "blocked"

    obs.start_trace("t2", "a", "op")
    exc = SecurityBlockedError("Output blocked by security controls", ["prompt_injection"])
    obs.end_trace("t2", status="error", error=exc)
    t2 = obs.get_traces()[1]
    assert t2["threats"] == ["prompt_injection"]
    assert t2["error_type"] == "SecurityBlockedError"
    assert t2["result"] == "Output blocked by security controls"


def test_report_only_threats_are_recorded_without_a_block():
    obs = AgentObserver()
    obs.start_trace("t", "a", "op")
    obs.end_trace("t", status="completed", threats=["firewall"], report_only=True)
    trace = obs.get_traces()[0]
    assert trace["threats"] == ["firewall"]
    assert trace["decision"] == "allow"
    assert trace["report_only"] is True
    assert trace["error_type"] is None


def test_the_old_end_trace_call_shape_still_works():
    obs = AgentObserver()
    obs.start_trace("t", "a", "op")
    obs.end_trace("t", status="error", result="boom")
    trace = obs.get_traces()[0]
    assert trace["status"] == "error" and trace["result"] == "boom"
    assert trace["threats"] == []


def test_a_refusal_before_any_trace_is_recorded_without_opening_a_trace():
    obs = AgentObserver()
    rec = obs.record_refusal("billing", "execute", ["prompt_injection"], stage="input")
    assert rec["threats"] == ["prompt_injection"] and rec["decision"] == "block"
    assert obs.get_traces() == []  # trace counts keep meaning "runs that started"
    assert obs.get_refusals()[0]["agent_id"] == "billing"
    assert obs.get_refusals(agent_id="other") == []
    assert obs.get_stats()["refusals_recorded"] == 1

    exc = SecurityBlockedError("Input blocked by policy", ["policy:low_trust"])
    rec = obs.record_refusal("billing", "execute", error=exc, stage="policy")
    assert rec["threats"] == ["policy:low_trust"]
    assert rec["error_type"] == "SecurityBlockedError"
    assert rec["reason"] == "Input blocked by policy"
    assert len(obs.get_refusals(limit=1)) == 1


# -- 3. escaping, caps, redaction ----------------------------------------------


def test_the_escape_set_covers_every_control_and_separator_in_unicode():
    """Re-derived, not listed: every Cc/Zl/Zp code point is escaped."""
    for code in range(sys.maxunicode + 1):
        char = chr(code)
        if unicodedata.category(char) in _UNSAFE_CATEGORIES:
            escaped = escape_text(char)
            assert escaped != char, f"U+{code:04X} is stored raw"
            assert escaped.isascii() and escaped.isprintable()


def test_bidi_overrides_are_escaped_but_ordinary_text_is_not():
    assert escape_text("a\u202eb\u2066c") == "a\\u202eb\\u2066c"
    text = "naïve café — 日本語 👩\u200d💻 tab-free"
    assert escape_text(text) == text


def test_every_stored_string_is_escaped():
    payload = "x\x0by\x85z\u2028w\u2029v\x1b[2K\x1cu\n\r\t"
    obs = AgentObserver()
    obs.log(payload, payload, agent_id=payload, metadata={payload: [payload, {"k": payload}]})
    obs.start_trace(payload, payload, payload, metadata={"m": payload})
    obs.log_event(payload, payload, payload, metadata={"raw": (payload, payload)})
    obs.end_trace(payload, status=payload, result={"r": payload}, threats=[payload])
    obs.record_refusal(payload, payload, [payload], reason=payload, metadata={"m": payload})
    decision = ToolDecision("block", payload, (payload,), False, payload)
    obs.record_tool_decision(None, decision, metadata={"m": payload})

    _assert_clean(obs.get_logs())
    _assert_clean(obs.get_traces())
    _assert_clean(obs.get_refusals())
    _assert_clean(obs.get_tool_decisions())
    # ...and the filters still find what was stored under a dirty id.
    assert len(obs.get_logs(agent_id=payload, level=payload)) == 1
    assert len(obs.get_traces(agent_id=payload, status=payload)) == 1


def test_exception_text_is_capped_by_default():
    secret = "upstream 500 for prompt='tell me everything' key=sk-live-123\nINFO forged"
    obs = AgentObserver(max_error_length=40)
    obs.start_trace("t", "a", "op")
    obs.end_trace("t", status="error", result=secret + "x" * 10000)
    stored = obs.get_traces()[0]["result"]
    assert len(stored) <= 40 and "\n" not in stored

    default = AgentObserver()
    default.start_trace("t", "a", "op")
    default.end_trace("t", status="error", error=RuntimeError("y" * 5000))
    assert len(default.get_traces()[0]["result"]) <= default.max_error_length


def test_metadata_is_bounded_in_depth_and_width_and_holds_no_caller_objects():
    class Opaque:
        def __str__(self):
            return "opaque\nvalue"

    nested: dict = {"leaf": "x"}
    for _ in range(10):
        nested = {"n": nested}
    obs = AgentObserver()
    obs.log("info", "m", metadata={"deep": nested, "wide": list(range(500)), "obj": Opaque()})
    meta = obs.get_logs()[0]["metadata"]
    assert len(meta["wide"]) == 100
    assert meta["obj"] == "opaque\\nvalue"
    depth, node = 0, meta["deep"]
    while isinstance(node, dict):
        node, depth = node.get("n", node.get("leaf")), depth + 1
    assert depth <= 4 and node == "[truncated: nested too deep]"


def test_metadata_numbers_and_shapes_are_preserved():
    obs = AgentObserver()
    obs.start_trace("t", "a", "op", metadata={"k": "v", "n": 3, "f": 1.5, "b": True, "z": None})
    assert obs.get_traces()[0]["metadata"] == {"k": "v", "n": 3, "f": 1.5, "b": True, "z": None}


def test_the_redaction_hook_runs_on_free_text_and_fails_closed():
    obs = AgentObserver(redactor=redact_secrets)
    obs.log("info", "calling with key=sk-live-abcdef123456 and Bearer abc.def.ghi.jkl")
    message = obs.get_logs()[0]["message"]
    assert "sk-live" not in message and "abc.def" not in message
    assert "[REDACTED]" in message

    def broken(text: str) -> str:
        raise RuntimeError("no")

    failing = AgentObserver(redactor=broken)
    failing.log("info", "the secret is hunter2")
    assert failing.get_logs()[0]["message"] == "[redaction failed]"


def test_a_long_message_is_truncated_after_escaping():
    obs = AgentObserver(max_text_length=50)
    obs.log("info", "\x1b" * 1000)
    assert len(obs.get_logs()[0]["message"]) <= 50


# -- 4. eviction ---------------------------------------------------------------


def test_eviction_drops_completed_traces_before_active_ones():
    obs = AgentObserver(max_traces=2)
    obs.start_trace("long-running", "a", "op")
    obs.start_trace("t2", "a", "op")
    obs.end_trace("t2")
    obs.start_trace("t3", "a", "op")  # evicts t2 (completed), not long-running

    assert [t["trace_id"] for t in obs.get_traces()] == ["long-running", "t3"]
    obs.end_trace("long-running", status="error", result="blocked output")
    assert obs.get_traces()[0]["status"] == "error"
    stats = obs.get_stats()
    assert stats["traces_evicted"] == 1 and stats["active_traces_evicted"] == 0


def test_evicting_an_active_trace_is_counted_never_silent():
    obs = AgentObserver(max_traces=2)
    for tid in ("a", "b", "c"):
        obs.start_trace(tid, "x", "op")
    obs.end_trace("a")
    stats = obs.get_stats()
    assert stats["active_traces_evicted"] == 1
    assert stats["end_trace_unknown"] == 1
    assert stats["traces_active"] == 2


def test_a_reused_trace_id_is_counted():
    obs = AgentObserver()
    obs.start_trace("t", "a", "op")
    obs.start_trace("t", "a", "op")
    assert obs.get_stats()["duplicate_trace_ids"] == 1
    assert len(obs.get_traces()) == 1


# -- 5. concurrency ------------------------------------------------------------


def test_concurrent_writers_and_readers_never_raise():
    obs = AgentObserver(max_traces=50, max_logs=50, max_metric_names=50)
    errors: list[str] = []
    stop = threading.Event()

    def writer(k: int) -> None:
        for i in range(3000):
            tid = f"{k}-{i}"
            try:
                obs.start_trace(tid, "a", "op")
                obs.log_event(tid, "e", "d")
                obs.log("info", "m", agent_id="a")
                obs.record_metric(f"m{i % 80}", 1.0)
                obs.end_trace(tid)
            except Exception as exc:  # pragma: no cover - the failure being tested
                errors.append(repr(exc))

    def reader() -> None:
        while not stop.is_set():
            try:
                for t in obs.get_traces(status="completed"):
                    dict(t)
                obs.get_traces(limit=5)
                obs.get_logs()
                obs.get_metric_stats("m1")
                obs.get_stats()
            except Exception as exc:  # pragma: no cover
                errors.append(repr(exc))

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    try:
        readers = [threading.Thread(target=reader) for _ in range(2)]
        writers = [threading.Thread(target=writer, args=(k,)) for k in range(4)]
        for t in readers + writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in readers:
            t.join()
    finally:
        sys.setswitchinterval(old)

    assert errors == []
    assert len(obs._traces) <= 50 and len(obs._metrics) <= 50
    assert obs.get_stats()["end_trace_unknown"] == 0


def test_readers_get_snapshots_not_live_records():
    obs = AgentObserver()
    snap = obs.start_trace("t", "a", "op")
    snap["status"] = "tampered"
    listed = obs.get_traces()[0]
    listed["events"].append("junk")
    obs.end_trace("t")
    assert obs.get_traces()[0]["status"] == "completed"
    assert obs.get_traces()[0]["events"] == []


# -- 6. metrics ----------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_metric_values_are_refused(bad):
    obs = AgentObserver()
    obs.record_metric("m", 1.0)
    with pytest.raises(ValueError):
        obs.record_metric("m", bad)
    assert obs.get_metric_stats("m") == {"count": 1, "min": 1.0, "max": 1.0, "avg": 1.0}


@pytest.mark.parametrize("bad", [True, "1.0", None])
def test_non_numeric_metric_values_are_refused(bad):
    with pytest.raises(TypeError):
        AgentObserver().record_metric("m", bad)


def test_metric_stats_for_an_empty_or_unknown_series_is_empty():
    obs = AgentObserver()
    assert obs.get_metric_stats("never") == {}
    obs.record_metric("ints", 2)
    assert obs.get_metric_stats("ints")["avg"] == 2.0


def test_metric_mean_does_not_drift():
    obs = AgentObserver(max_metrics=10)
    for v in (1e16, 1.0, -1e16, 1.0):
        obs.record_metric("m", v)
    assert obs.get_metric_stats("m")["avg"] == 0.5


@pytest.mark.parametrize(
    "kwargs",
    [{"max_traces": 0}, {"max_logs": -1}, {"max_text_length": 0}, {"max_metrics": True}],
)
def test_nonsensical_caps_are_refused(kwargs):
    with pytest.raises(ValueError):
        AgentObserver(**kwargs)


def test_a_non_callable_redactor_or_otel_value_is_refused():
    with pytest.raises(TypeError):
        AgentObserver(redactor="scrub")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AgentObserver(otel=object())


# -- tool decisions ------------------------------------------------------------


def test_record_tool_decision_stores_the_verdict_and_logs_it_on_the_parent():
    obs = AgentObserver()
    obs.start_trace("run", "billing", "execute")
    rec = obs.record_tool_decision(
        None,
        ToolDecision("block", "shell", ("tool:denied:shell",), False, "billing"),
        parent_id="run",
        tool_call_id="call-1",
    )
    assert rec["agent_id"] == "billing" and rec["tool_name"] == "shell"
    assert rec["decision"] == "block" and rec["status"] == "blocked"
    assert rec["threats"] == ["tool:denied:shell"] and rec["tool_call_id"] == "call-1"
    event = obs.get_traces()[0]["events"][0]
    assert event["event_type"] == "tool_decision" and event["metadata"]["status"] == "blocked"
    assert obs.get_tool_decisions(tool_name="shell")[0]["span_id"] == rec["span_id"]
    assert obs.get_tool_decisions(agent_id="nobody") == []


def test_tool_span_times_the_call_and_keeps_the_most_severe_verdict():
    obs = AgentObserver()
    with obs.tool_span("a", "search") as span:
        span.record(ToolDecision("allow", "search"), "call")
        span.record(ToolDecision("block", "search", ("firewall",)), "result")
    rec = obs.get_tool_decisions()[0]
    assert rec["phases"] == ["call", "result"]
    assert rec["decision"] == "block" and rec["threats"] == ["firewall"]
    assert rec["duration_ms"] >= 0

    with obs.tool_span("a", "pay") as span:
        span.record(ToolDecision("ask", "pay", ("tool:approval:pay",)))
    assert obs.get_tool_decisions(limit=1)[0]["status"] == "approval_required"

    with obs.tool_span("a", "audit") as span:
        span.record(ToolDecision("allow", "audit", ("firewall",), report_only=True))
    rec = obs.get_tool_decisions(limit=1)[0]
    assert rec["status"] == "completed" and rec["report_only"] is True


def test_a_tool_span_whose_block_raises_is_closed_as_an_error():
    obs = AgentObserver()
    with pytest.raises(KeyError):
        with obs.tool_span("a", "lookup"):
            raise KeyError("missing")
    rec = obs.get_tool_decisions()[0]
    assert rec["status"] == "error" and rec["error_type"] == "KeyError"
