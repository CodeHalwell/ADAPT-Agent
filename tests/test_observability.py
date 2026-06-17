from adapt_agent.observability import AgentObserver


def test_get_logs_limit():
    observer = AgentObserver(max_logs=100)
    for i in range(10):
        observer.log(level="info", message=f"test{i}", agent_id="agent1")

    logs = observer.get_logs(limit=3)
    assert len(logs) == 3
    assert logs[0]["message"] == "test7"
    assert logs[2]["message"] == "test9"


def test_get_traces_limit():
    observer = AgentObserver()
    for i in range(5):
        observer.start_trace(trace_id=str(i), agent_id="agent1", operation="op")

    traces = observer.get_traces(limit=2)
    assert len(traces) == 2
    assert traces[0]["trace_id"] == "3"
    assert traces[1]["trace_id"] == "4"


def test_log_poisoning_prevention():
    observer = AgentObserver()
    observer.log(level="info", message="Line 1\nLine 2\rLine 3")
    logs = observer.get_logs()
    assert logs[0]["message"] == "Line 1\\nLine 2\\rLine 3"

    observer.start_trace(trace_id="1", agent_id="agent1", operation="op")
    observer.log_event(trace_id="1", event_type="test", description="Desc 1\nDesc 2\rDesc 3")
    traces = observer.get_traces()
    assert traces[0]["events"][0]["description"] == "Desc 1\\nDesc 2\\rDesc 3"
