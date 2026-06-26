"""Tests for the AgentOptimizer optimization tools."""

from adapt_agent.optimization import AgentOptimizer


def test_analyze_performance_computes_statistics():
    """Statistics are computed correctly across multiple calls for an agent."""
    optimizer = AgentOptimizer()

    optimizer.analyze_performance("agent_a", execution_time=2.0, success=True)
    optimizer.analyze_performance("agent_a", execution_time=4.0, success=False)
    stats = optimizer.analyze_performance("agent_a", execution_time=6.0, success=True)

    assert stats["total_executions"] == 3
    assert stats["avg_execution_time"] == (2.0 + 4.0 + 6.0) / 3
    assert stats["success_rate"] == 2 / 3


def test_analyze_performance_isolates_agents():
    """Statistics only aggregate metrics for the requested agent_id."""
    optimizer = AgentOptimizer()

    optimizer.analyze_performance("agent_a", execution_time=1.0, success=True)
    optimizer.analyze_performance("agent_b", execution_time=10.0, success=False)
    stats = optimizer.analyze_performance("agent_a", execution_time=3.0, success=True)

    assert stats["total_executions"] == 2
    assert stats["avg_execution_time"] == 2.0
    assert stats["success_rate"] == 1.0


def test_suggest_optimizations_slow_execution():
    """A slow-execution suggestion is produced when avg time > 5.0."""
    optimizer = AgentOptimizer()
    optimizer.analyze_performance("agent_a", execution_time=6.0)
    optimizer.analyze_performance("agent_a", execution_time=8.0)

    suggestions = optimizer.suggest_optimizations("agent_a")

    perf = [s for s in suggestions if s["metric"] == "execution_time"]
    assert len(perf) == 1
    assert perf[0]["type"] == "performance"
    assert perf[0]["value"] == 7.0


def test_suggest_optimizations_high_token_usage():
    """A high-token suggestion is produced when avg token usage > 1000."""
    optimizer = AgentOptimizer()
    optimizer.analyze_performance("agent_a", execution_time=1.0, token_usage=1500)
    optimizer.analyze_performance("agent_a", execution_time=1.0, token_usage=2500)

    suggestions = optimizer.suggest_optimizations("agent_a")

    tokens = [s for s in suggestions if s["metric"] == "token_usage"]
    assert len(tokens) == 1
    assert tokens[0]["type"] == "efficiency"
    assert tokens[0]["value"] == 2000.0
    # Execution time is below threshold, so no perf suggestion.
    assert all(s["metric"] != "execution_time" for s in suggestions)


def test_suggest_optimizations_both_suggestions():
    """Both suggestions appear when both thresholds are exceeded."""
    optimizer = AgentOptimizer()
    optimizer.analyze_performance("agent_a", execution_time=10.0, token_usage=5000)

    suggestions = optimizer.suggest_optimizations("agent_a")
    metrics = {s["metric"] for s in suggestions}
    assert metrics == {"execution_time", "token_usage"}


def test_suggest_optimizations_unknown_agent_returns_empty():
    """An unknown agent yields no suggestions."""
    optimizer = AgentOptimizer()
    optimizer.analyze_performance("agent_a", execution_time=10.0, token_usage=5000)

    assert optimizer.suggest_optimizations("does_not_exist") == []


def test_suggest_optimizations_below_thresholds_returns_empty():
    """No suggestions when metrics are below thresholds."""
    optimizer = AgentOptimizer()
    optimizer.analyze_performance("agent_a", execution_time=1.0, token_usage=100)
    optimizer.analyze_performance("agent_a", execution_time=2.0, token_usage=200)

    assert optimizer.suggest_optimizations("agent_a") == []


def test_suggest_optimizations_token_usage_none_handled():
    """token_usage=None produces no token suggestion even with slow execution."""
    optimizer = AgentOptimizer()
    optimizer.analyze_performance("agent_a", execution_time=9.0, token_usage=None)
    optimizer.analyze_performance("agent_a", execution_time=9.0)  # defaults to None

    suggestions = optimizer.suggest_optimizations("agent_a")
    assert all(s["metric"] != "token_usage" for s in suggestions)
    # Slow execution suggestion should still be present.
    assert any(s["metric"] == "execution_time" for s in suggestions)


def test_max_metrics_bounding_drops_oldest():
    """The internal metrics list never exceeds max_metrics."""
    optimizer = AgentOptimizer(max_metrics=3)

    for i in range(6):
        optimizer.analyze_performance("agent_a", execution_time=float(i))

    assert len(optimizer._metrics) == 3
    assert [m["execution_time"] for m in optimizer._metrics] == [3.0, 4.0, 5.0]


def test_max_suggestions_bounding():
    """Stored suggestions are bounded by max_suggestions, keeping newest."""
    optimizer = AgentOptimizer(max_suggestions=2)

    # Each call produces 2 suggestions (slow + high tokens). Because suggestions
    # are now stored per agent and replaced (not appended) on each call, the
    # buffer for a single agent never grows past the latest 2.
    for _ in range(3):
        optimizer.analyze_performance("agent_a", execution_time=10.0, token_usage=5000)
        optimizer.suggest_optimizations("agent_a")

    assert len(optimizer._optimization_suggestions) == 2


def test_suggestions_deduped_per_agent_not_appended():
    """Repeated suggest_optimizations for one agent does not accumulate dupes."""
    optimizer = AgentOptimizer()
    optimizer.analyze_performance("agent_a", execution_time=10.0, token_usage=5000)

    # Call many times; the stored buffer should reflect only the latest set,
    # not 10x duplicates.
    for _ in range(10):
        optimizer.suggest_optimizations("agent_a")

    stored = optimizer._optimization_suggestions
    assert len(stored) == 2
    assert {s["metric"] for s in stored} == {"execution_time", "token_usage"}


def test_suggestions_stored_separately_per_agent():
    """Suggestions for different agents are kept independently."""
    optimizer = AgentOptimizer()
    optimizer.analyze_performance("agent_a", execution_time=10.0, token_usage=5000)
    optimizer.analyze_performance("agent_b", execution_time=10.0)

    optimizer.suggest_optimizations("agent_a")
    optimizer.suggest_optimizations("agent_a")  # replace, not append
    optimizer.suggest_optimizations("agent_b")

    stored = optimizer._optimization_suggestions
    # agent_a: 2 (slow + tokens), agent_b: 1 (slow only) = 3 total, no dupes.
    assert len(stored) == 3
