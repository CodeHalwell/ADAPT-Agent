"""Tests for the AgentEvaluator evaluation framework."""

from adapt_agent.evaluation import AgentEvaluator


def test_register_metric_and_evaluate_response_computes_metrics():
    """Registered metrics are computed and stored on the result."""
    evaluator = AgentEvaluator()
    evaluator.register_metric("exact_match", lambda out, exp: 1.0 if out == exp else 0.0)
    evaluator.register_metric("length", lambda out, exp: float(len(out)))

    result = evaluator.evaluate_response(
        agent_id="agent_a",
        input_data="hi",
        output_data="hello",
        expected_output="hello",
    )

    assert result["agent_id"] == "agent_a"
    assert result["input"] == "hi"
    assert result["output"] == "hello"
    assert result["expected"] == "hello"
    assert result["metrics"]["exact_match"] == 1.0
    assert result["metrics"]["length"] == 5.0


def test_metric_that_raises_is_caught_and_recorded_as_none():
    """A metric function that raises is caught and recorded as None."""
    evaluator = AgentEvaluator()

    def boom(out, exp):
        raise ValueError("kaboom")

    evaluator.register_metric("boom", boom)
    evaluator.register_metric("ok", lambda out, exp: 1.0)

    result = evaluator.evaluate_response(
        agent_id="agent_a",
        input_data="x",
        output_data="y",
    )

    assert result["metrics"]["boom"] is None
    assert result["metrics"]["ok"] == 1.0


def test_evaluate_response_stores_results():
    """evaluate_response appends to the stored results."""
    evaluator = AgentEvaluator()
    evaluator.register_metric("score", lambda out, exp: 1.0)

    evaluator.evaluate_response("a", "i", "o")
    evaluator.evaluate_response("a", "i", "o")

    assert len(evaluator.get_evaluation_results()) == 2


def test_compute_aggregate_metrics_averages_across_evaluations():
    """Aggregate metrics average the per-evaluation scores."""
    evaluator = AgentEvaluator()
    scores = iter([0.0, 1.0, 0.5])
    evaluator.register_metric("score", lambda out, exp: next(scores))

    evaluator.evaluate_response("a", "i", "o")
    evaluator.evaluate_response("a", "i", "o")
    evaluator.evaluate_response("a", "i", "o")

    aggregate = evaluator.compute_aggregate_metrics()
    assert aggregate["score"] == (0.0 + 1.0 + 0.5) / 3


def test_compute_aggregate_metrics_filters_by_agent_id():
    """Aggregate metrics only consider the requested agent_id."""
    evaluator = AgentEvaluator()
    scores = iter([0.2, 0.8, 1.0])
    evaluator.register_metric("score", lambda out, exp: next(scores))

    evaluator.evaluate_response("agent_a", "i", "o")  # 0.2
    evaluator.evaluate_response("agent_b", "i", "o")  # 0.8
    evaluator.evaluate_response("agent_a", "i", "o")  # 1.0

    agg_a = evaluator.compute_aggregate_metrics(agent_id="agent_a")
    assert agg_a["score"] == (0.2 + 1.0) / 2

    agg_b = evaluator.compute_aggregate_metrics(agent_id="agent_b")
    assert agg_b["score"] == 0.8


def test_compute_aggregate_metrics_empty_returns_empty_dict():
    """No results yields an empty aggregate."""
    evaluator = AgentEvaluator()
    assert evaluator.compute_aggregate_metrics() == {}
    assert evaluator.compute_aggregate_metrics(agent_id="nope") == {}


def test_compute_aggregate_metrics_ignores_none_scores():
    """None metric values are excluded from the average."""
    evaluator = AgentEvaluator()
    calls = iter([1.0, "raise", 3.0])

    def metric(out, exp):
        val = next(calls)
        if val == "raise":
            raise RuntimeError("nope")
        return val

    evaluator.register_metric("score", metric)
    evaluator.evaluate_response("a", "i", "o")  # 1.0
    evaluator.evaluate_response("a", "i", "o")  # None (raised)
    evaluator.evaluate_response("a", "i", "o")  # 3.0

    agg = evaluator.compute_aggregate_metrics()
    assert agg["score"] == (1.0 + 3.0) / 2


def test_get_evaluation_results_without_limit():
    """Without a limit, all results are returned in chronological order."""
    evaluator = AgentEvaluator()
    evaluator.register_metric("idx", lambda out, exp: float(out))

    for i in range(5):
        evaluator.evaluate_response("a", "i", i)

    results = evaluator.get_evaluation_results()
    assert len(results) == 5
    assert [r["output"] for r in results] == [0, 1, 2, 3, 4]


def test_get_evaluation_results_with_limit_returns_most_recent_in_order():
    """With a limit, the most recent N results are returned chronologically."""
    evaluator = AgentEvaluator()

    for i in range(5):
        evaluator.evaluate_response("a", "i", i)

    results = evaluator.get_evaluation_results(limit=2)
    assert len(results) == 2
    # Most recent two (3, 4) in chronological order.
    assert [r["output"] for r in results] == [3, 4]


def test_get_evaluation_results_filters_by_agent_id():
    """agent_id filter restricts results, both with and without limit."""
    evaluator = AgentEvaluator()

    evaluator.evaluate_response("agent_a", "i", 0)
    evaluator.evaluate_response("agent_b", "i", 1)
    evaluator.evaluate_response("agent_a", "i", 2)
    evaluator.evaluate_response("agent_b", "i", 3)
    evaluator.evaluate_response("agent_a", "i", 4)

    no_limit = evaluator.get_evaluation_results(agent_id="agent_a")
    assert [r["output"] for r in no_limit] == [0, 2, 4]

    limited = evaluator.get_evaluation_results(agent_id="agent_a", limit=2)
    assert [r["output"] for r in limited] == [2, 4]


def test_max_results_bounding_drops_oldest():
    """The internal results list never exceeds max_results; oldest are dropped."""
    evaluator = AgentEvaluator(max_results=3)

    for i in range(6):
        evaluator.evaluate_response("a", "i", i)

    assert len(evaluator._evaluation_results) == 3
    # Oldest dropped, newest retained, in chronological order.
    assert [r["output"] for r in evaluator._evaluation_results] == [3, 4, 5]
