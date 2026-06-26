"""Additional tests for AgentObserver, covering uncovered branches."""

from adapt_agent.observability import AgentObserver


def test_start_and_end_trace_status_and_result():
    """start_trace creates an active trace; end_trace sets status and result."""
    observer = AgentObserver()
    trace = observer.start_trace(
        trace_id="t1", agent_id="agent1", operation="op", metadata={"k": "v"}
    )
    assert trace["status"] == "active"
    assert trace["metadata"] == {"k": "v"}
    assert trace["events"] == []

    observer.end_trace("t1", status="failed", result={"error": "boom"})
    traces = observer.get_traces()
    assert len(traces) == 1
    ended = traces[0]
    assert ended["status"] == "failed"
    assert ended["result"] == {"error": "boom"}
    assert "end_time" in ended


def test_end_trace_unknown_id_is_noop():
    """Ending a non-existent trace does nothing and does not raise."""
    observer = AgentObserver()
    observer.end_trace("missing", status="completed")
    assert observer.get_traces() == []


def test_log_event_appends_sanitizes_and_truncates():
    """log_event appends to the trace, escapes newlines, and truncates to 10000."""
    observer = AgentObserver()
    observer.start_trace(trace_id="t1", agent_id="agent1", operation="op")

    description = "line1\nline2\rline3" + ("y" * 20000)
    observer.log_event(trace_id="t1", event_type="evt", description=description)

    event = observer.get_traces()[0]["events"][0]
    assert event["event_type"] == "evt"
    assert "\n" not in event["description"]
    assert "\r" not in event["description"]
    assert "\\n" in event["description"]
    assert "\\r" in event["description"]
    # Truncated to 10000 chars before escaping expands them.
    assert len(event["description"]) <= 10000 + 4  # escaping may add a couple chars


def test_log_event_unknown_trace_is_noop():
    """Logging an event to an unknown trace does nothing and does not raise."""
    observer = AgentObserver()
    observer.log_event(trace_id="missing", event_type="evt", description="hi")
    assert observer.get_traces() == []


def test_log_sanitizes_truncates_and_bounds():
    """log escapes newlines, truncates to 10000, and is bounded by max_logs."""
    observer = AgentObserver(max_logs=2)

    observer.log(level="info", message="a\nb\rc" + ("z" * 20000), agent_id="agent1")
    first = observer.get_logs()[0]
    assert "\n" not in first["message"]
    assert "\r" not in first["message"]
    assert "\\n" in first["message"]
    assert "\\r" in first["message"]
    assert len(first["message"]) <= 10000 + 4

    observer.log(level="info", message="m2")
    observer.log(level="info", message="m3")

    logs = observer.get_logs()
    assert len(logs) == 2  # bounded by max_logs
    assert logs[0]["message"] == "m2"
    assert logs[1]["message"] == "m3"


def test_record_metric_and_get_metric_stats():
    """record_metric accumulates values; get_metric_stats computes summary stats."""
    observer = AgentObserver()
    for v in (2.0, 4.0, 6.0):
        observer.record_metric("latency", v)

    stats = observer.get_metric_stats("latency")
    assert stats["count"] == 3
    assert stats["min"] == 2.0
    assert stats["max"] == 6.0
    assert stats["avg"] == 4.0


def test_get_metric_stats_unknown_metric_returns_empty():
    """Unknown metrics return an empty dict."""
    observer = AgentObserver()
    assert observer.get_metric_stats("nope") == {}


def test_get_traces_filters_with_and_without_limit():
    """get_traces filters by agent_id and status, with and without a limit."""
    observer = AgentObserver()
    observer.start_trace(trace_id="t1", agent_id="a1", operation="op")
    observer.start_trace(trace_id="t2", agent_id="a2", operation="op")
    observer.start_trace(trace_id="t3", agent_id="a1", operation="op")
    observer.end_trace("t1", status="completed")
    observer.end_trace("t3", status="completed")
    # t2 remains active.

    # Filter by agent_id (no limit).
    a1_traces = observer.get_traces(agent_id="a1")
    assert {t["trace_id"] for t in a1_traces} == {"t1", "t3"}

    # Filter by status (no limit).
    active = observer.get_traces(status="active")
    assert {t["trace_id"] for t in active} == {"t2"}

    # Filter by both agent_id and status (no limit).
    a1_completed = observer.get_traces(agent_id="a1", status="completed")
    assert {t["trace_id"] for t in a1_completed} == {"t1", "t3"}

    # Filter by agent_id with a limit (most recent first by insertion order).
    limited = observer.get_traces(agent_id="a1", limit=1)
    assert len(limited) == 1
    assert limited[0]["trace_id"] == "t3"

    # Filter by status with a limit.
    completed_limited = observer.get_traces(status="completed", limit=1)
    assert len(completed_limited) == 1
    assert completed_limited[0]["trace_id"] == "t3"


def test_get_logs_filters_with_and_without_limit():
    """get_logs filters by level and agent_id, with and without a limit."""
    observer = AgentObserver()
    observer.log(level="info", message="i1", agent_id="a1")
    observer.log(level="error", message="e1", agent_id="a1")
    observer.log(level="info", message="i2", agent_id="a2")
    observer.log(level="error", message="e2", agent_id="a1")

    # Filter by level (no limit).
    errors = observer.get_logs(level="error")
    assert [log["message"] for log in errors] == ["e1", "e2"]

    # Filter by agent_id (no limit).
    a1_logs = observer.get_logs(agent_id="a1")
    assert [log["message"] for log in a1_logs] == ["i1", "e1", "e2"]

    # Filter by both level and agent_id (no limit).
    a1_errors = observer.get_logs(level="error", agent_id="a1")
    assert [log["message"] for log in a1_errors] == ["e1", "e2"]

    # Filter by level with a limit (most recent).
    limited_errors = observer.get_logs(level="error", limit=1)
    assert len(limited_errors) == 1
    assert limited_errors[0]["message"] == "e2"

    # Filter by agent_id with a limit.
    limited_a1 = observer.get_logs(agent_id="a1", limit=1)
    assert len(limited_a1) == 1
    assert limited_a1[0]["message"] == "e2"


def test_max_traces_bounding():
    """The trace store is bounded by max_traces, evicting the oldest."""
    observer = AgentObserver(max_traces=3)
    for i in range(5):
        observer.start_trace(trace_id=str(i), agent_id="a1", operation="op")

    traces = observer.get_traces()
    assert len(traces) == 3
    assert {t["trace_id"] for t in traces} == {"2", "3", "4"}


def test_max_events_per_trace_bounding():
    """Events within a trace are bounded by max_events_per_trace."""
    observer = AgentObserver(max_events_per_trace=2)
    observer.start_trace(trace_id="t1", agent_id="a1", operation="op")
    for i in range(5):
        observer.log_event(trace_id="t1", event_type="evt", description=str(i))

    events = observer.get_traces()[0]["events"]
    assert len(events) == 2
    assert [e["description"] for e in events] == ["3", "4"]


def test_max_metric_names_bounding():
    """The number of distinct metric names is bounded by max_metric_names."""
    observer = AgentObserver(max_metric_names=2)
    observer.record_metric("m1", 1.0)
    observer.record_metric("m2", 1.0)
    observer.record_metric("m3", 1.0)

    # Oldest metric name evicted once a new name overflows the cap.
    assert observer.get_metric_stats("m1") == {}
    assert observer.get_metric_stats("m3")["count"] == 1


def test_max_metrics_bounding_per_name():
    """Values per metric name are bounded by max_metrics."""
    observer = AgentObserver(max_metrics=3)
    for v in range(10):
        observer.record_metric("m", float(v))

    stats = observer.get_metric_stats("m")
    assert stats["count"] == 3
    # Oldest values evicted; newest retained.
    assert stats["max"] == 9.0
    assert stats["min"] == 7.0
