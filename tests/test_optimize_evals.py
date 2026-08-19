"""Offline end-to-end tests for ``adapt_agent.optimization.evals.evaluate_agent``.

These exercise the headline scenario: evaluate agents built with LangGraph,
Microsoft Agent Framework, Google ADK, and Pydantic AI against a golden dataset
using deterministic checks (text / number match) and an LLM-as-judge -- all with
framework-shaped stubs, so no framework or LLM SDK is required.
"""

import json

import pytest

from adapt_agent.evaluation import evaluate_agent  # re-exported "evals" namespace
from adapt_agent.optimization import (
    EvaluationReport,
    Example,
    GoldenDataset,
    LLMJudge,
    exact_match,
)
from adapt_agent.optimization.runners import adk_runner

# -- framework-shaped stub agents ------------------------------------------------

_QA = {
    "What is the capital of France?": "Paris",
    "What is 6 * 7?": "The answer is 42.",
}


def _answer(question: str) -> str:
    return _QA.get(question, "I do not know.")


class PydanticAIStyleAgent:
    """Shape of a Pydantic AI ``Agent``: ``run_sync`` returning an ``AgentRunResult``."""

    class _Result:
        def __init__(self, output):
            self.output = output

        def all_messages(self):  # pragma: no cover - presence is what matters
            return []

    def run_sync(self, question):
        return self._Result(_answer(question))


class MAFStyleAgent:
    """Shape of a Microsoft Agent Framework ``ChatAgent``: async ``run`` -> response."""

    class _Message:
        def __init__(self, text):
            self.role = "assistant"
            self.text = text
            self.contents = []

    class _Response:
        def __init__(self, text):
            self.messages = [MAFStyleAgent._Message(text)]
            self.text = text

    async def run(self, question):
        return self._Response(_answer(question))


class LangGraphStyleGraph:
    """Shape of a compiled LangGraph graph: ``invoke(state) -> state``."""

    def __init__(self):
        self.nodes = {}

    def invoke(self, state):
        question = state["messages"][-1]["content"]
        reply = {"role": "assistant", "content": _answer(question)}
        return {"messages": [*state["messages"], reply]}


class ADKStyleRunner:
    """Shape of a Google ADK ``Runner``: kwargs ``run`` yielding events."""

    class _Part:
        def __init__(self, text=None):
            self.text = text

    class _Content:
        def __init__(self, parts, role="model"):
            self.parts = parts
            self.role = role

    class _Event:
        def __init__(self, text):
            self.content = ADKStyleRunner._Content([ADKStyleRunner._Part(text)])
            self.author = "agent"

    class _Sessions:
        def create_session(self, *, app_name, user_id, session_id):
            return object()

    def __init__(self):
        self.app_name = "qa-app"
        self.session_service = self._Sessions()

    def run(self, *, user_id, session_id, new_message):
        yield self._Event(None)
        yield self._Event(_answer(new_message))


_DATASET = [
    {"input": "What is the capital of France?", "expected": "Paris"},
    {"input": "What is 6 * 7?", "expected": 42, "check": "numeric_close"},
]


@pytest.mark.parametrize(
    "make_agent",
    [
        PydanticAIStyleAgent,
        MAFStyleAgent,
        LangGraphStyleGraph,
        lambda: adk_runner(ADKStyleRunner(), message_factory=lambda s: s),
    ],
    ids=["pydantic_ai", "microsoft_agent_framework", "langgraph", "google_adk"],
)
def test_evaluate_agent_across_framework_shapes(make_agent):
    report = evaluate_agent(make_agent(), _DATASET)
    assert isinstance(report, EvaluationReport)
    assert report.n == 2
    assert report.n_errors == 0
    # Row 1 passes exact_match (the default check); row 2 passes numeric_close
    # even though the produced text is "The answer is 42."
    assert report.score == 1.0
    assert report.aggregate["checks"] == 1.0


def test_evaluate_agent_default_check_is_exact_match():
    report = evaluate_agent(lambda q: "wrong answer", [{"input": "q", "expected": "right"}])
    assert report.score == 0.0


def test_evaluate_agent_named_metrics():
    report = evaluate_agent(
        lambda q: "Paris is the capital",
        [{"input": "q", "expected": "Paris"}],
        metrics=["contains", "exact_match"],
    )
    assert report.aggregate["contains"] == 1.0
    assert report.aggregate["exact_match"] == 0.0
    assert report.primary_metric == "contains"
    assert report.score == 1.0


def test_evaluate_agent_primary_metric_override():
    report = evaluate_agent(
        lambda q: "Paris is the capital",
        [{"input": "q", "expected": "Paris"}],
        metrics=["contains", "exact_match"],
        primary_metric="exact_match",
    )
    assert report.score == 0.0


def test_evaluate_agent_metric_objects_and_callables():
    def always_half(output, expected):
        return 0.5

    report = evaluate_agent(
        lambda q: "x",
        [{"input": "q", "expected": "x"}],
        metrics=[exact_match(), always_half],
    )
    assert report.aggregate["exact_match"] == 1.0
    assert report.aggregate["always_half"] == 0.5


def test_evaluate_agent_metrics_mapping_renames():
    report = evaluate_agent(
        lambda q: "4",
        [{"input": "q", "expected": "4"}],
        metrics={"accuracy": "exact_match"},
    )
    assert report.aggregate == {"accuracy": 1.0}


def test_evaluate_agent_unknown_metric_name():
    with pytest.raises(KeyError):
        evaluate_agent(lambda q: q, [{"input": "q", "expected": "q"}], metrics=["nope"])


def test_evaluate_agent_bad_metric_spec():
    with pytest.raises(TypeError):
        evaluate_agent(lambda q: q, [{"input": "q", "expected": "q"}], metrics=[123])


# -- LLM-as-judge ------------------------------------------------------------------


def _stub_judge_completion(prompt: str, system: str | None = None) -> str:
    """Deterministic judge: high score iff the fenced response mentions Paris."""
    response = ""
    if "<response>" in prompt:
        response = prompt.split("<response>")[1].split("</response>")[0]
    score = 9 if "Paris" in response else 1
    return json.dumps({"score": score, "pass": score >= 6, "reasoning": "stub"})


def test_evaluate_agent_judge_only():
    report = evaluate_agent(
        lambda q: "Paris, of course",
        [{"input": "capital of France?"}],  # unlabeled: reference-free judging
        judge=_stub_judge_completion,
    )
    assert report.primary_metric == "judge"
    assert report.score == pytest.approx(0.9)


def test_evaluate_agent_judge_instance_plus_metrics():
    judge = LLMJudge(_stub_judge_completion)
    report = evaluate_agent(
        lambda q: "Paris",
        [{"input": "capital?", "expected": "Paris"}],
        metrics=["exact_match"],
        judge=judge,
    )
    assert report.aggregate["exact_match"] == 1.0
    assert report.aggregate["judge"] == pytest.approx(0.9)
    assert report.primary_metric == "exact_match"


def test_evaluate_agent_judge_named_in_metrics_not_duplicated():
    report = evaluate_agent(
        lambda q: "Paris",
        [{"input": "capital?", "expected": "Paris"}],
        metrics=["judge", "exact_match"],
        judge=_stub_judge_completion,
    )
    assert set(report.aggregate) == {"judge", "exact_match"}
    assert report.primary_metric == "judge"


def test_evaluate_agent_judge_metric_without_judge_raises():
    with pytest.raises(ValueError):
        evaluate_agent(lambda q: q, [{"input": "q"}], metrics=["judge"])


def test_evaluate_agent_per_row_judge_check():
    report = evaluate_agent(
        lambda q: "Paris" if "France" in q else "4",
        [
            {"input": "What is 2+2?", "expected": "4"},
            {"input": "Describe France's capital", "check": "judge"},
        ],
        judge=_stub_judge_completion,
        metrics="checks",
    )
    # Row 1: exact_match (default). Row 2: judge scores 0.9 -> mean is 0.95.
    assert report.aggregate["checks"] == pytest.approx(0.95)


def test_evaluate_agent_bad_judge_type():
    with pytest.raises(TypeError):
        evaluate_agent(lambda q: q, [{"input": "q"}], judge=123)


# -- dataset coercion ------------------------------------------------------------


def test_evaluate_agent_accepts_golden_dataset_and_examples():
    data = GoldenDataset([Example(inputs="q", expected="a")])
    report = evaluate_agent(lambda q: "a", data)
    assert report.score == 1.0


def test_evaluate_agent_loads_jsonl_path(tmp_path):
    path = tmp_path / "golden.jsonl"
    rows = [
        {"input": "capital of France?", "expected": "Paris"},
        {"input": "6*7?", "expected": "42", "check": "numeric_close"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    report = evaluate_agent(lambda q: "Paris" if "France" in q else "it's 42", str(path))
    assert report.score == 1.0


def test_evaluate_agent_loads_csv_with_explicit_keys(tmp_path):
    path = tmp_path / "golden.csv"
    path.write_text("question,gold\n2+2?,4\n", encoding="utf-8")
    report = evaluate_agent(lambda q: "4", path, input_key="question", expected_key="gold")
    assert report.score == 1.0


def test_evaluate_agent_rejects_unknown_extension(tmp_path):
    path = tmp_path / "golden.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_agent(lambda q: q, str(path))


def test_evaluate_agent_rejects_non_dataset():
    with pytest.raises(TypeError):
        evaluate_agent(lambda q: q, 42)


# -- extraction and adaptation toggles ----------------------------------------------


def test_evaluate_agent_output_extractor_none_scores_raw():
    report = evaluate_agent(
        lambda q: {"answer": {"nested": True}},
        [{"input": "q", "expected": {"nested": True}}],
        metrics=[exact_match(normalize=False)],
        output_extractor=None,
    )
    # Raw dict output compared structurally: {"answer": ...} != {"nested": True}.
    assert report.score == 0.0

    report = evaluate_agent(
        lambda q: {"nested": True},
        [{"input": "q", "expected": {"nested": True}}],
        metrics=[exact_match(normalize=False)],
        output_extractor=None,
    )
    assert report.score == 1.0


def test_evaluate_agent_agent_errors_are_recorded_not_fatal():
    def flaky(question):
        raise RuntimeError("boom")

    report = evaluate_agent(flaky, [{"input": "q", "expected": "a"}])
    assert report.n_errors == 1
    assert report.score == 0.0
    assert report.results[0].error is not None


def test_evaluate_agent_report_failures_surface_wrong_rows():
    report = evaluate_agent(
        lambda q: "Paris" if "France" in q else "no idea",
        [
            {"input": "capital of France?", "expected": "Paris"},
            {"input": "capital of Japan?", "expected": "Tokyo"},
        ],
    )
    failures = report.failures()
    assert len(failures) == 1
    assert failures[0].expected == "Tokyo"


def test_evaluate_agent_checks_metric_does_not_double_judge():
    calls = []

    def counting_judge(prompt, system=None):
        calls.append(prompt)
        return '{"score": 9, "pass": true, "reasoning": "ok"}'

    report = evaluate_agent(
        lambda q: "Paris",
        [
            {"input": "capital?", "expected": "Paris"},
            {"input": "describe France", "check": "judge"},
        ],
        metrics="checks",
        judge=counting_judge,
    )
    # No auto-appended "judge" metric: checks routes the judge itself, so only
    # the row that declared a judge check spent a judge call.
    assert set(report.aggregate) == {"checks"}
    assert len(calls) == 1
