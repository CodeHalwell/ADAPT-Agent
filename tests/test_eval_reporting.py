"""Harness completeness, agent-failure classification, multi-turn rows, usage and export."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from adapt_agent.optimization import reporting
from adapt_agent.optimization.dataset import Example, GoldenDataset
from adapt_agent.optimization.evaluation import EvaluationHarness, is_infrastructure_error
from adapt_agent.optimization.metrics import exact_match
from adapt_agent.optimization.retry import RetryPolicy

_FAST = RetryPolicy(attempts=2, initial_backoff=0.0, jitter=0.0)


def _data(n: int = 3, **extra: object) -> GoldenDataset:
    return GoldenDataset.from_list(
        [{"input": f"q{i}", "expected": f"a{i}", "id": f"r{i}", **extra} for i in range(n)]
    )


# -- an agent's own failure is a failure ----------------------------------------------


class ToolTimeout(Exception):
    """An agent-side exception whose *name* sounds transient."""


def test_an_agent_raising_its_own_timeout_is_scored_not_dropped() -> None:
    def agent(x: str) -> str:
        if x != "q0":
            raise ToolTimeout("agent exceeded its step budget")
        return "a0"

    report = EvaluationHarness([exact_match()], retry=_FAST).evaluate(agent, _data())
    assert report.n_transient_errors == 0
    assert report.is_complete
    assert report.score == pytest.approx(1 / 3)
    assert report.completeness() == {
        "rows_total": 3,
        "rows_scored": 3,
        "rows_errored": 2,
        "rows_dropped_transient": 0,
        "rows_metric_errors": 0,
    }
    assert report.results[1].error_type == "ToolTimeout"


@pytest.mark.parametrize(
    "exc",
    [
        TimeoutError("tool took too long"),
        ConnectionError("my mock server is down"),
        RuntimeError("rate limit of my own tool budget reached"),
    ],
)
def test_agent_failures_without_infrastructure_evidence_count_against_the_agent(
    exc: Exception,
) -> None:
    def agent(_: str) -> str:
        raise exc

    report = EvaluationHarness([exact_match()], retry=_FAST).evaluate(agent, _data(2))
    assert report.n_transient_errors == 0
    assert report.n_agent_errors == 2
    assert report.score == 0.0


class _ProviderError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_provider_shaped_agent_failures_are_still_excused() -> None:
    def agent(_: str) -> str:
        raise RuntimeError("agent invocation failed") from _ProviderError("busy", 429)

    report = EvaluationHarness([exact_match()], retry=_FAST).evaluate(agent, _data(2))
    assert report.n_transient_errors == 2
    assert not report.is_complete


def test_infrastructure_evidence() -> None:
    assert is_infrastructure_error(RuntimeError("Error code: 529 - overloaded"))
    assert is_infrastructure_error(_ProviderError("x", 503))
    assert not is_infrastructure_error(ToolTimeout("timeout"))
    assert not is_infrastructure_error(RuntimeError("expected 429 items, got 3"))
    fake_sdk = type("APITimeoutError", (Exception,), {"__module__": "openai._exceptions"})
    assert is_infrastructure_error(fake_sdk("timed out"))


def test_a_custom_classifier_is_trusted_as_given() -> None:
    policy = RetryPolicy(attempts=1, is_transient=lambda exc: isinstance(exc, ToolTimeout))

    def agent(_: str) -> str:
        raise ToolTimeout("x")

    report = EvaluationHarness([exact_match()], retry=policy).evaluate(agent, _data(2))
    assert report.n_transient_errors == 2


# -- latency and usage ---------------------------------------------------------------


def test_latency_percentiles_and_usage_are_aggregated() -> None:
    def agent(x: str) -> dict:
        return {"output": "a" + x[1:], "usage": {"input_tokens": 10, "output_tokens": 2}}

    report = EvaluationHarness([exact_match()], output_extractor=lambda r: r["output"]).evaluate(
        agent, _data(4)
    )
    assert report.score == 1.0
    assert report.usage == {"input_tokens": 40.0, "output_tokens": 8.0}
    assert report.results[0].usage == {"input_tokens": 10.0, "output_tokens": 2.0}
    assert 0.0 < report.latency_p50 <= report.latency_p95


def test_judge_last_usage_is_attributed_per_row_on_the_serial_path() -> None:
    from adapt_agent.optimization.judge import LLMJudge

    class Provider:
        last_usage = None

        def __call__(self, prompt: str) -> str:
            self.last_usage = {"input_tokens": 100, "output_tokens": 5}
            return '{"score": 8, "reasoning": "ok"}'

    judge = LLMJudge(Provider())
    report = EvaluationHarness([judge.as_metric("judge")]).evaluate(lambda x: "x", _data(2))
    assert report.usage == {"judge_input_tokens": 200.0, "judge_output_tokens": 10.0}


# -- multi-turn rows -----------------------------------------------------------------


def test_multi_turn_rows_carry_history_and_score_per_turn() -> None:
    seen: list[list[dict]] = []

    def chat(messages: list[dict]) -> str:
        seen.append(messages)
        return "hello" if len(messages) == 1 else f"you said {messages[-1]['content']}"

    data = GoldenDataset.from_list(
        [
            {
                "id": "conv",
                "turns": [
                    {"input": "hi", "expected": "hello"},
                    {"input": "ping", "expected": "you said ping"},
                    {"input": "bye", "expected": "nope"},
                ],
            }
        ]
    )
    report = EvaluationHarness([exact_match()]).evaluate(chat, data)
    assert report.score == pytest.approx(2 / 3)
    turns = report.results[0].turns
    assert [t["scores"]["exact_match"] for t in turns] == [1.0, 1.0, 0.0]
    assert seen[1] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "ping"},
    ]


def test_multi_turn_text_mode_setup_turns_and_per_turn_checks() -> None:
    from adapt_agent.optimization.metrics import checks

    memory: list[str] = []

    def stateful(text: str) -> str:
        memory.append(text)
        return f"{len(memory)} items"

    data = GoldenDataset.from_list(
        [
            {
                "turns": [
                    {"input": "remember x", "score": False},
                    {"input": "count?", "expected": 2, "check": "numeric_close"},
                ]
            }
        ]
    )
    report = EvaluationHarness([checks()], conversation_input="text").evaluate(stateful, data)
    assert report.score == 1.0
    assert memory == ["remember x", "count?"]
    assert report.results[0].turns[0]["scored"] is False


def test_an_agent_failure_mid_conversation_fails_the_row() -> None:
    def chat(messages: list[dict]) -> str:
        if len(messages) > 1:
            raise ValueError("lost context")
        return "hello"

    data = GoldenDataset.from_list(
        [{"turns": [{"input": "hi", "expected": "hello"}, {"input": "again", "expected": "x"}]}]
    )
    report = EvaluationHarness([exact_match()]).evaluate(chat, data)
    assert report.score == 0.0
    assert report.results[0].error.startswith("turn 1:")
    assert len(report.results[0].turns) == 1


def test_async_multi_turn_matches_sync() -> None:
    import asyncio

    async def chat(messages: list[dict]) -> str:
        return messages[-1]["content"].upper()

    data = GoldenDataset.from_list(
        [{"turns": [{"input": "a", "expected": "A"}, {"input": "b", "expected": "c"}]}]
    )
    report = asyncio.run(EvaluationHarness([exact_match()]).aevaluate(chat, data))
    assert report.score == 0.5


@pytest.mark.parametrize("turns", [[], "hi", [{"expected": "x"}], [42]])
def test_malformed_conversations_fail_at_load(turns: object) -> None:
    with pytest.raises(ValueError):
        Example.from_record({"turns": turns})


def test_conversation_input_is_validated() -> None:
    with pytest.raises(ValueError):
        EvaluationHarness([exact_match()], conversation_input="xml")


# -- export --------------------------------------------------------------------------


def _mixed_report() -> object:
    def agent(x: str) -> str:
        if x == "q2":
            raise ValueError("boom <&>")
        return "a0" if x == "q0" else "wrong\x01"

    return EvaluationHarness([exact_match()]).evaluate(agent, _data())


def test_results_document_round_trips_through_json(tmp_path) -> None:
    report = _mixed_report()
    path = tmp_path / "out" / "results.json"
    reporting.to_json(report, path, metadata={"target": "t"})
    doc = reporting.load_results(path)
    assert doc["schema_version"] == reporting.RESULTS_SCHEMA_VERSION
    assert doc["summary"]["completeness"]["rows_errored"] == 1
    assert [r["id"] for r in doc["rows"]] == ["r0", "r1", "r2"]
    assert doc["rows"][2]["error_type"] == "ValueError"
    assert doc["metadata"] == {"target": "t"}


def test_load_results_rejects_other_json(tmp_path) -> None:
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"score": 1}))
    with pytest.raises(ValueError, match="not an evaluation results file"):
        reporting.load_results(path)


def test_junit_has_one_case_per_row_and_metric_with_details(tmp_path) -> None:
    report = _mixed_report()
    xml = reporting.to_junit(report, tmp_path / "j.xml")
    root = ET.fromstring(xml.split("\n", 1)[1])
    suite = root.find("testsuite")
    assert suite is not None
    assert suite.get("tests") == "3"
    assert suite.get("failures") == "1"
    assert suite.get("errors") == "1"
    failure = suite.find("testcase/failure")
    assert failure is not None and "expected: a1" in (failure.text or "")
    assert "\x01" not in xml


def test_junit_size_cap_keeps_every_case() -> None:
    data = GoldenDataset.from_list([{"input": "q" * 500, "expected": "x" * 500}] * 50)
    report = EvaluationHarness([exact_match()]).evaluate(lambda x: "y" * 500, data)
    xml = reporting.to_junit(report, max_bytes=5_000)
    suite = ET.fromstring(xml.split("\n", 1)[1]).find("testsuite")
    assert suite is not None
    assert suite.get("tests") == "50" and suite.get("failures") == "50"
    assert suite.get("truncated") == "true"
    assert len(xml) < 60_000


def test_junit_marks_transient_rows_skipped() -> None:
    def agent(_: str) -> str:
        raise _ProviderError("busy", 503)

    report = EvaluationHarness([exact_match()], retry=_FAST).evaluate(agent, _data(1))
    xml = reporting.to_junit(report)
    assert "<skipped" in xml


# -- comparison ----------------------------------------------------------------------


def test_compare_runs_reports_deltas_and_moved_rows() -> None:
    harness = EvaluationHarness([exact_match()])
    base = harness.evaluate(lambda x: "a" + x[1:], _data())
    cur = harness.evaluate(lambda x: "a0" if x == "q0" else "no", _data())
    comparison = reporting.compare_runs(cur, reporting.results_document(base))
    assert comparison.metrics["exact_match"]["delta"] == pytest.approx(-2 / 3)
    assert comparison.regressed_metrics == ["exact_match"]
    assert {r["key"] for r in comparison.regressed_rows} == {"id:r1", "id:r2"}
    assert comparison.has_regression
    assert not comparison.warnings
    reverse = reporting.compare_runs(base, cur)
    assert not reverse.has_regression and len(reverse.improved_rows) == 2


def test_compare_runs_tolerance_and_position_matching_warning() -> None:
    harness = EvaluationHarness([exact_match()])
    rows = GoldenDataset.from_list([{"input": "q", "expected": "a"}])
    a = harness.evaluate(lambda x: "a", rows)
    b = harness.evaluate(lambda x: "b", rows)
    assert not reporting.compare_runs(b, a, tolerance=1.0).has_regression
    assert any("position" in w for w in reporting.compare_runs(b, a).warnings)


def test_a_metric_that_raises_is_counted_as_a_row_error() -> None:
    from adapt_agent.optimization.metrics import Metric

    class JudgeError(Exception):
        pass

    def broken(output: object, expected: object) -> float:
        raise JudgeError("could not grade this row")

    report = EvaluationHarness([exact_match(), Metric("judge", broken, on_error=0.7)]).evaluate(
        lambda x: "a" + x[1:], _data(2)
    )
    assert report.n_metric_errors == 2
    assert report.completeness()["rows_metric_errors"] == 2
    assert report.results[0].metric_errors["judge"].startswith("JudgeError:")
    assert report.aggregate["judge"] == 0.7  # the declared fallback, but flagged
    doc = reporting.results_document(report)
    assert doc["summary"]["n_metric_errors"] == 2
