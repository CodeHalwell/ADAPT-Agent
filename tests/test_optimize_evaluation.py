"""Offline, deterministic tests for ``adapt_agent.optimization.evaluation``."""

import pytest

from adapt_agent.optimization.dataset import Example, GoldenDataset
from adapt_agent.optimization.evaluation import (
    EvaluationHarness,
    EvaluationReport,
    ExampleResult,
    resolve_runner,
)
from adapt_agent.optimization.metrics import Metric

# -- resolve_runner -----------------------------------------------------------


def test_resolve_runner_prefers_run_attr():
    calls = []

    class WithRun:
        def run(self, x):
            calls.append(x)
            return f"run:{x}"

        def __call__(self, x):  # pragma: no cover - should not be used
            return "call"

        def execute(self, x):  # pragma: no cover - should not be used
            return "execute"

    obj = WithRun()
    runner = resolve_runner(obj)
    assert runner("a") == "run:a"
    assert calls == ["a"]


def test_resolve_runner_plain_callable():
    def fn(x):
        return x * 2

    assert resolve_runner(fn) is fn
    assert resolve_runner(fn)(3) == 6


def test_resolve_runner_execute_attr():
    class WithExecute:
        def execute(self, x):
            return f"exec:{x}"

    runner = resolve_runner(WithExecute())
    assert runner("z") == "exec:z"


def test_resolve_runner_non_callable_run_falls_through_to_execute():
    class Weird:
        run = "not callable"

        def execute(self, x):
            return f"exec:{x}"

    runner = resolve_runner(Weird())
    assert runner("q") == "exec:q"


def test_resolve_runner_invalid_raises_typeerror():
    class NoRun:
        pass

    with pytest.raises(TypeError):
        resolve_runner(NoRun())


def test_resolve_runner_framework_methods():
    # LangGraph (invoke), CrewAI (kickoff), Pydantic AI (run_sync) work directly.
    class Graph:
        def invoke(self, x):
            return {"state": x}

    class Crew:
        def kickoff(self, x):
            return f"crew:{x}"

    class PydanticAgent:
        def run_sync(self, x):
            return f"sync:{x}"

    assert resolve_runner(Graph())("q") == {"state": "q"}
    assert resolve_runner(Crew())("q") == "crew:q"
    assert resolve_runner(PydanticAgent())("q") == "sync:q"


def test_resolve_runner_prefers_run_sync_over_async_run():
    # An object exposing both run_sync (sync) and run (async) uses run_sync.
    class Dual:
        def run_sync(self, x):
            return f"sync:{x}"

        async def run(self, x):  # pragma: no cover - must not be chosen
            return f"async:{x}"

    assert resolve_runner(Dual())("q") == "sync:q"


def test_resolve_runner_resolves_async_run_method():
    # An async-only run method is driven synchronously (coroutine awaited).
    class AsyncAgent:
        async def run(self, x):
            return f"async:{x}"

    assert resolve_runner(AsyncAgent())("q") == "async:q"


# -- metric helpers -----------------------------------------------------------


def _const_metric(name, value):
    return Metric(name, lambda output, expected: value)


def _echo_match():
    return Metric("echo", lambda output, expected: 1.0 if output == expected else 0.0)


# -- EvaluationHarness construction -------------------------------------------


def test_harness_single_metric():
    h = EvaluationHarness(_echo_match())
    assert [m.name for m in h.metrics] == ["echo"]
    assert h.primary_metric == "echo"


def test_harness_list_of_metrics_and_default_primary():
    h = EvaluationHarness([_const_metric("a", 0.5), _const_metric("b", 1.0)])
    assert [m.name for m in h.metrics] == ["a", "b"]
    assert h.primary_metric == "a"  # first one by default


def test_harness_bare_callable_coerced_uses_function_name():
    def my_metric(output, expected):
        return 1.0

    h = EvaluationHarness(my_metric)
    assert h.metrics[0].name == "my_metric"


def test_harness_dict_mapping_key_becomes_name():
    m = Metric("original", lambda o, e: 1.0)
    h = EvaluationHarness({"renamed": m, "bare": lambda o, e: 0.0})
    names = [x.name for x in h.metrics]
    assert "renamed" in names
    assert "bare" in names
    assert "original" not in names


def test_harness_dict_preserves_needs_example_flag():
    m = Metric("orig", lambda o, e, ex: 1.0, needs_example=True)
    h = EvaluationHarness({"keyname": m})
    renamed = h.metrics[0]
    assert renamed.name == "keyname"
    assert renamed.needs_example is True


def test_harness_explicit_primary_metric():
    h = EvaluationHarness([_const_metric("a", 0.1), _const_metric("b", 0.9)], primary_metric="b")
    assert h.primary_metric == "b"


def test_harness_invalid_primary_raises():
    with pytest.raises(ValueError):
        EvaluationHarness([_const_metric("a", 1.0)], primary_metric="nope")


def test_harness_duplicate_names_raises():
    with pytest.raises(ValueError):
        EvaluationHarness([_const_metric("dup", 1.0), _const_metric("dup", 0.0)])


def test_harness_empty_metrics_raises():
    with pytest.raises(ValueError):
        EvaluationHarness([])
    with pytest.raises(ValueError):
        EvaluationHarness(None)


# -- evaluate() ---------------------------------------------------------------


def _dataset():
    return GoldenDataset(
        [
            Example(inputs="hi", expected="hi"),
            Example(inputs="yo", expected="nope"),
        ]
    )


def test_evaluate_aggregate_means_and_per_example_results():
    h = EvaluationHarness(_echo_match())
    report = h.evaluate(lambda x: x, _dataset())
    # First example matches (1.0), second does not (0.0) -> mean 0.5
    assert report.aggregate["echo"] == 0.5
    assert report.score == 0.5
    assert report.n == 2
    assert report.n_errors == 0
    assert all(isinstance(r, ExampleResult) for r in report.results)
    assert report.results[0].scores["echo"] == 1.0
    assert report.results[1].scores["echo"] == 0.0
    assert report.results[0].output == "hi"
    assert report.results[0].inputs == "hi"
    assert report.results[0].index == 0
    assert report.results[0].primary_ok is True


def test_evaluate_multiple_metrics_aggregate():
    h = EvaluationHarness([_const_metric("a", 0.25), _const_metric("b", 0.75)])
    report = h.evaluate(lambda x: x, _dataset())
    assert report.aggregate["a"] == 0.25
    assert report.aggregate["b"] == 0.75


def test_evaluate_runner_raises_records_error_scores_zero():
    def boom(x):
        raise RuntimeError("kaboom")

    h = EvaluationHarness(_const_metric("a", 1.0))
    report = h.evaluate(boom, _dataset())
    assert report.n_errors == 2
    assert report.aggregate["a"] == 0.0
    for r in report.results:
        assert r.error is not None
        assert "kaboom" in r.error
        assert r.scores["a"] == 0.0
        assert r.output is None
        assert r.primary_ok is False


def test_evaluate_metric_raises_scores_zero_but_no_error():
    def bad_metric(output, expected):
        raise ValueError("metric blew up")

    h = EvaluationHarness(Metric("bad", bad_metric))
    report = h.evaluate(lambda x: x, _dataset())
    # Runner did not error, so n_errors stays 0; metric scores 0.
    assert report.n_errors == 0
    assert report.aggregate["bad"] == 0.0
    for r in report.results:
        assert r.error is None
        assert r.scores["bad"] == 0.0


def test_evaluate_object_with_run():
    class Agent:
        def run(self, x):
            return x.upper()

    h = EvaluationHarness(Metric("upper", lambda o, e: 1.0 if o == e else 0.0))
    ds = GoldenDataset([Example(inputs="ab", expected="AB")])
    report = h.evaluate(Agent(), ds)
    assert report.score == 1.0


def test_evaluate_all_metrics_present_even_when_no_examples():
    h = EvaluationHarness([_const_metric("a", 1.0), _const_metric("b", 1.0)])
    report = h.evaluate(lambda x: x, GoldenDataset([]))
    assert report.aggregate == {"a": 0.0, "b": 0.0}
    assert report.n == 0


# -- capture_output / max_results --------------------------------------------


def test_capture_output_false_drops_io():
    h = EvaluationHarness(_echo_match(), capture_output=False)
    report = h.evaluate(lambda x: x, _dataset())
    for r in report.results:
        assert r.inputs is None
        assert r.output is None
    # Scores still computed from the live output.
    assert report.results[0].scores["echo"] == 1.0


def test_capture_output_false_on_error_path():
    def boom(x):
        raise RuntimeError("x")

    h = EvaluationHarness(_const_metric("a", 1.0), capture_output=False)
    report = h.evaluate(boom, _dataset())
    for r in report.results:
        assert r.inputs is None
        assert r.output is None
        assert r.error is not None


def test_max_results_bounds_stored_results_but_not_aggregate():
    ds = GoldenDataset([Example(inputs=str(i), expected=str(i)) for i in range(5)])
    h = EvaluationHarness(_echo_match(), max_results=2)
    report = h.evaluate(lambda x: x, ds)
    # Only 2 results stored ...
    assert len(report.results) == 2
    assert report.n == 2
    # ... but aggregate computed over all 5 (all match -> 1.0).
    assert report.aggregate["echo"] == 1.0


# -- EvaluationReport API -----------------------------------------------------


def test_report_failures_default_primary_threshold():
    h = EvaluationHarness(_echo_match())
    report = h.evaluate(lambda x: x, _dataset())
    fails = report.failures()
    # Second example scored 0.0 < 1.0 default threshold.
    assert len(fails) == 1
    assert fails[0].index == 1


def test_report_failures_custom_threshold():
    h = EvaluationHarness(_const_metric("a", 0.6))
    report = h.evaluate(lambda x: x, _dataset())
    assert report.failures(threshold=0.5) == []  # 0.6 >= 0.5
    assert len(report.failures(threshold=0.7)) == 2  # 0.6 < 0.7


def test_harness_failure_threshold_default_is_one():
    # Unconfigured harness keeps the historical "perfect-only passes" behaviour.
    h = EvaluationHarness(_const_metric("a", 0.6))
    report = h.evaluate(lambda x: x, _dataset())
    assert report.failure_threshold == 1.0
    # 0.6 < 1.0 -> every example is a failure by default.
    assert len(report.failures()) == 2


def test_harness_configurable_failure_threshold_propagates_to_report():
    # A tunable threshold avoids flooding failures() for continuous metrics.
    h = EvaluationHarness(_const_metric("a", 0.6), failure_threshold=0.5)
    report = h.evaluate(lambda x: x, _dataset())
    assert report.failure_threshold == 0.5
    # 0.6 >= 0.5 -> no failures when relying on the harness default.
    assert report.failures() == []
    # An explicit threshold still overrides the harness default.
    assert len(report.failures(threshold=0.7)) == 2


def test_harness_failure_threshold_does_not_drop_errors():
    # Errored examples are always reported regardless of the threshold.
    def runner(x):
        if x == "yo":
            raise RuntimeError("boom")
        return x

    h = EvaluationHarness(_const_metric("a", 1.0), failure_threshold=0.0)
    report = h.evaluate(runner, _dataset())
    fails = report.failures()
    assert len(fails) == 1
    assert fails[0].error is not None


def test_report_failure_threshold_default_when_constructed_directly():
    # A report built directly (not via the harness) defaults to 1.0.
    rep = EvaluationReport(aggregate={"a": 0.5}, primary_metric="a")
    assert rep.failure_threshold == 1.0


def test_report_below_named_metric_excludes_errors():
    def runner(x):
        if x == "yo":
            raise RuntimeError("boom")
        return x

    h = EvaluationHarness([_echo_match(), _const_metric("always", 1.0)])
    report = h.evaluate(runner, _dataset())
    # 'always' scores 1.0 for the surviving example and 0.0 for the errored one.
    below = report.below("always", 0.5)
    # Only the errored example scores below 0.5; it is included via its 0.0 score.
    assert len(below) == 1
    assert below[0].error is not None
    # Nothing is below 0.0.
    assert report.below("always", 0.0) == []


def test_harness_cache_flag_is_noop_and_reevaluates():
    # The reserved cache flag must not change behaviour: the agent is re-run.
    calls = []

    def runner(x):
        calls.append(x)
        return x

    h = EvaluationHarness(_echo_match(), cache=True)
    assert h.cache is True
    h.evaluate(runner, _dataset())
    h.evaluate(runner, _dataset())
    # Each evaluate ran both examples again (no memoization).
    assert len(calls) == 4


def test_report_failures_specific_metric_and_errors_included():
    def runner(x):
        if x == "yo":
            raise RuntimeError("boom")
        return x

    h = EvaluationHarness([_echo_match(), _const_metric("always", 1.0)])
    report = h.evaluate(runner, _dataset())
    # On 'always' metric nothing is below threshold, but the errored example
    # is always included.
    fails = report.failures(metric="always")
    assert len(fails) == 1
    assert fails[0].error is not None


def test_report_score_absent_primary_returns_zero():
    rep = EvaluationReport(aggregate={"a": 0.5}, primary_metric="missing")
    assert rep.score == 0.0


def test_report_avg_latency():
    r1 = ExampleResult(0, None, None, None, {}, latency=1.0)
    r2 = ExampleResult(1, None, None, None, {}, latency=3.0)
    rep = EvaluationReport(aggregate={}, primary_metric="x", results=[r1, r2], total_latency=4.0)
    assert rep.avg_latency == 2.0


def test_report_avg_latency_empty():
    rep = EvaluationReport(aggregate={}, primary_metric="x")
    assert rep.avg_latency == 0.0


def test_report_to_dict():
    h = EvaluationHarness(_echo_match())
    report = h.evaluate(lambda x: x, _dataset())
    d = report.to_dict()
    assert d == {
        "score": report.score,
        "primary_metric": "echo",
        "aggregate": {"echo": 0.5},
        "n": 2,
        "n_errors": 0,
        "n_transient_errors": 0,
        "is_complete": True,
        "avg_latency": report.avg_latency,
    }
    # aggregate is a copy, not the same object.
    assert d["aggregate"] is not report.aggregate


def test_report_repr_contains_key_fields():
    h = EvaluationHarness(_echo_match())
    report = h.evaluate(lambda x: x, _dataset())
    text = repr(report)
    assert "EvaluationReport" in text
    assert "echo" in text
    assert "n=2" in text


# -- output_extractor ----------------------------------------------------------


def test_harness_output_extractor_unwraps_before_scoring():
    from adapt_agent.optimization.extractors import extract_output_text
    from adapt_agent.optimization.metrics import exact_match

    class WrappedResult:
        def __init__(self, output):
            self.output = output

        def all_messages(self):  # Pydantic-AI-shaped
            return []

    harness = EvaluationHarness(exact_match(), output_extractor=extract_output_text)
    data = GoldenDataset([Example(inputs="q", expected="Paris")])
    report = harness.evaluate(lambda q: WrappedResult("Paris"), data)
    assert report.score == 1.0
    # The stored output is the extracted text, not the wrapper object.
    assert report.results[0].output == "Paris"


def test_harness_without_extractor_scores_raw_output():
    from adapt_agent.optimization.metrics import exact_match

    class WrappedResult:
        def __init__(self, output):
            self.output = output

        def all_messages(self):
            return []

    harness = EvaluationHarness(exact_match())
    data = GoldenDataset([Example(inputs="q", expected="Paris")])
    report = harness.evaluate(lambda q: WrappedResult("Paris"), data)
    assert report.score == 0.0  # repr(WrappedResult) never equals "Paris"


def test_harness_output_extractor_errors_are_non_fatal():
    from adapt_agent.optimization.metrics import contains

    def broken_extractor(value):
        raise RuntimeError("boom")

    harness = EvaluationHarness(contains(), output_extractor=broken_extractor)
    data = GoldenDataset([Example(inputs="q", expected="Paris")])
    report = harness.evaluate(lambda q: "Paris is the capital", data)
    # The raw output is scored instead of crashing the evaluation.
    assert report.n_errors == 0
    assert report.score == 1.0


def test_harness_output_extractor_applies_per_example_checks():
    from adapt_agent.optimization.extractors import extract_output_text
    from adapt_agent.optimization.metrics import checks

    harness = EvaluationHarness(checks(), output_extractor=extract_output_text)
    data = GoldenDataset(
        [
            Example(inputs="capital of France?", expected="Paris"),
            Example(
                inputs="2+2?",
                expected="4",
                metadata={"check": "numeric_close"},
            ),
        ]
    )

    def agent(question):
        return {"output": "Paris" if "France" in question else "the answer is 4"}

    report = harness.evaluate(agent, data)
    assert report.aggregate["checks"] == 1.0
