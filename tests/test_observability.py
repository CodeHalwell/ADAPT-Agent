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

    # Test log poisoning prevention
    malicious_log = "user login\n[ERROR] system compromised"
    observer.log(level="info", message=malicious_log, agent_id="agent1")
    logs = observer.get_logs(limit=1)
    assert "\n" not in logs[0]["message"]
    assert "\\n" in logs[0]["message"]

    # Test event poisoning prevention
    observer.start_trace(trace_id="trace1", agent_id="agent1", operation="op")
    malicious_event = "started task\r\n[CRITICAL] deleted all files"
    observer.log_event(trace_id="trace1", event_type="task", description=malicious_event)
    traces = observer.get_traces(limit=1)
    event = traces[0]["events"][0]
    assert "\n" not in event["description"]
    assert "\r" not in event["description"]
    assert "\\n" in event["description"]
    assert "\\r" in event["description"]
