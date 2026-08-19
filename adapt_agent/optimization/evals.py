"""One-call agent evals: run any framework agent against a golden dataset.

:func:`evaluate_agent` is the high-level entry point for the common eval loop --
"here is my agent, here is my dataset of inputs and expected outputs, score it"
-- across every supported framework (LangGraph, Microsoft Agent Framework,
Google ADK, Pydantic AI, CrewAI, OpenAI Agents SDK, Claude Agent SDK) and plain
callables. It composes the lower-level pieces so a single call:

1. loads the dataset (a :class:`~adapt_agent.optimization.dataset.GoldenDataset`,
   a list of records, or a ``.json`` / ``.jsonl`` / ``.csv`` path),
2. wraps the agent via
   :func:`~adapt_agent.optimization.runners.framework_runner` (framework run
   method discovery, async resolution, LangGraph input adaptation),
3. unwraps framework-native results to final response text via
   :func:`~adapt_agent.optimization.extractors.extract_output_text`,
4. scores with deterministic metrics (exact/contains/regex/numeric/...), the
   per-example :func:`~adapt_agent.optimization.metrics.checks` dispatcher,
   and/or an LLM-as-judge, and
5. returns the standard
   :class:`~adapt_agent.optimization.evaluation.EvaluationReport`.

Example::

    from adapt_agent.evaluation import evaluate_agent

    report = evaluate_agent(
        agent,                               # any supported framework object
        [
            {"input": "What is the capital of France?", "expected": "Paris"},
            {"input": "What is 6 * 7?", "expected": 42,
             "check": "numeric_close"},
            {"input": "Summarise our refund policy", "check": "judge",
             "criteria": "Mentions the 30-day window."},
        ],
        judge=my_judge,                      # optional LLM-as-judge
    )
    print(report.score, report.aggregate)

Everything here is offline-testable: no LLM SDK or agent framework is imported
by this module.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from adapt_agent.optimization.dataset import GoldenDataset
from adapt_agent.optimization.evaluation import EvaluationHarness, EvaluationReport
from adapt_agent.optimization.extractors import extract_output_text
from adapt_agent.optimization.judge import LLMJudge
from adapt_agent.optimization.metrics import (
    BUILTIN_METRICS,
    Metric,
    checks,
    coerce_metric,
    get_metric,
)
from adapt_agent.optimization.runners import AUTO, framework_runner

#: Metric names routed to the LLM-as-judge instead of a deterministic built-in.
_JUDGE_NAMES = ("judge", "llm_judge")


def evaluate_agent(
    agent: Any,
    data: GoldenDataset | Iterable[Any] | str | Path,
    *,
    metrics: Any = None,
    judge: Any = None,
    judge_criteria: str | None = None,
    judge_rubric: str | None = None,
    primary_metric: str | None = None,
    input_adapter: Callable[[Any], Any] | str | None = AUTO,
    output_extractor: Callable[[Any], Any] | None = extract_output_text,
    input_key: str | None = None,
    expected_key: str | None = None,
    capture_output: bool = True,
    max_results: int = 10_000,
    failure_threshold: float = 1.0,
) -> EvaluationReport:
    """Evaluate any agent against a golden dataset and return the report.

    Args:
        agent: The system under test: a framework object (LangGraph compiled
            graph, Microsoft Agent Framework ``ChatAgent``, Pydantic AI
            ``Agent``, CrewAI ``Crew``, ...), a governed adapter, an
            :class:`~adapt_agent.optimization.target.OptimizableAgent`, or any
            callable -- including the callable built by
            :func:`~adapt_agent.optimization.runners.adk_runner` for Google ADK.
        data: A :class:`~adapt_agent.optimization.dataset.GoldenDataset`, an
            iterable of records/:class:`Example` s, or a dataset file path
            (``.json`` / ``.jsonl`` / ``.csv``).
        metrics: What to score. A metric name (``"exact_match"``,
            ``"numeric_close"``, ``"checks"``, ``"judge"``, ...), a
            :class:`~adapt_agent.optimization.metrics.Metric`, a bare callable,
            or a list mixing any of these. Defaults to the per-example
            :func:`~adapt_agent.optimization.metrics.checks` dispatcher (rows
            choose their own check via ``metadata["check"]``, falling back to
            ``exact_match``) -- or to the judge alone when only ``judge`` is
            supplied.
        judge: Optional LLM-as-judge: an
            :class:`~adapt_agent.optimization.judge.LLMJudge`, a provider name
            (``"claude"``, ``"openai"``, ``"gemini"``, ...), or a bare
            completion callable ``prompt -> text``. Added as a ``"judge"``
            metric grading every row -- unless ``metrics`` already routes it
            (an explicit ``"judge"`` entry, or a ``"checks"`` dispatcher,
            which judges only the rows declaring ``{"check": "judge"}``).
        judge_criteria: Task-level grading criteria passed to the judge metric
            (rows may override via ``metadata["criteria"]``).
        judge_rubric: Rubric override for the judge metric.
        primary_metric: Headline metric name; defaults to the first metric.
        input_adapter: Forwarded to
            :func:`~adapt_agent.optimization.runners.framework_runner`:
            ``"auto"`` (default) adapts plain strings for LangGraph graphs;
            pass a callable to customise or ``None`` to disable.
        output_extractor: Unwraps each framework-native result before scoring;
            :func:`~adapt_agent.optimization.extractors.extract_output_text` by
            default. Pass ``None`` to score raw outputs (e.g. for
            ``json_subset`` over structured state).
        input_key: Explicit input column name for record/file datasets.
        expected_key: Explicit expected/gold column name for record/file
            datasets.
        capture_output: Store per-example outputs on the report.
        max_results: Cap on stored per-example results.
        failure_threshold: Default cutoff for ``report.failures()``.

    Returns:
        The :class:`~adapt_agent.optimization.evaluation.EvaluationReport`.
    """
    dataset = _coerce_dataset(data, input_key=input_key, expected_key=expected_key)
    judge_obj = _coerce_judge(judge)
    metric_list = _resolve_metrics(metrics, judge_obj, criteria=judge_criteria, rubric=judge_rubric)
    harness = EvaluationHarness(
        metric_list,
        primary_metric=primary_metric,
        capture_output=capture_output,
        max_results=max_results,
        failure_threshold=failure_threshold,
        output_extractor=output_extractor,
    )
    runner = framework_runner(agent, input_adapter=input_adapter, output_extractor=None)
    return harness.evaluate(runner, dataset)


# -- coercion helpers -----------------------------------------------------------


def _coerce_dataset(data: Any, *, input_key: str | None, expected_key: str | None) -> GoldenDataset:
    """Accept a GoldenDataset, a file path, or an iterable of records."""
    if isinstance(data, GoldenDataset):
        return data
    if isinstance(data, (str, Path)):
        path = str(data)
        lower = path.lower()
        if lower.endswith(".jsonl"):
            return GoldenDataset.from_jsonl(path, input_key=input_key, expected_key=expected_key)
        if lower.endswith(".json"):
            return GoldenDataset.from_json(path, input_key=input_key, expected_key=expected_key)
        if lower.endswith(".csv"):
            return GoldenDataset.from_csv(path, input_key=input_key, expected_key=expected_key)
        raise ValueError(f"Unsupported dataset extension for {path!r} (use .json/.jsonl/.csv)")
    if isinstance(data, Iterable):
        return GoldenDataset.from_list(list(data), input_key=input_key, expected_key=expected_key)
    raise TypeError(
        f"Cannot build a dataset from {type(data)!r}: pass a GoldenDataset, a list "
        "of records, or a .json/.jsonl/.csv path"
    )


def _coerce_judge(judge: Any) -> Any:
    """Coerce ``judge`` into something exposing ``as_metric`` (or ``None``)."""
    if judge is None:
        return None
    if callable(getattr(judge, "as_metric", None)):
        return judge
    if isinstance(judge, str):
        from adapt_agent.optimization.judges import get_judge

        return get_judge(judge)
    if callable(judge):
        return LLMJudge(judge)
    raise TypeError(
        f"judge must be an LLMJudge, a provider name, or a completion callable, "
        f"got {type(judge)!r}"
    )


def _resolve_metrics(
    metrics: Any, judge_obj: Any, *, criteria: str | None, rubric: str | None
) -> list[Metric]:
    """Build the final metric list from user metrics + the optional judge."""

    def _judge_metric() -> Metric:
        if judge_obj is None:
            raise ValueError(
                "A 'judge' metric was requested but no judge was supplied: pass "
                "judge=<LLMJudge | provider name | completion callable>"
            )
        return coerce_metric(judge_obj.as_metric("judge", criteria=criteria, rubric=rubric))

    def _resolve_one(spec: Any) -> Metric:
        if isinstance(spec, Metric):
            return spec
        if isinstance(spec, str):
            if spec in _JUDGE_NAMES:
                return _judge_metric()
            if spec == "checks":
                return checks(judge=judge_obj)
            if spec in BUILTIN_METRICS:
                return get_metric(spec)
            raise KeyError(
                f"Unknown metric {spec!r}. Available: {sorted(BUILTIN_METRICS)} "
                f"plus {list(_JUDGE_NAMES)}"
            )
        if callable(spec):
            return coerce_metric(spec)
        raise TypeError(f"Unsupported metric specification: {spec!r}")

    def _routes_judge(spec: Any) -> bool:
        """Whether a raw spec already involves the judge (or routes it per-row).

        Decided on the *specification*, before any mapping-key renaming, so
        ``metrics={"accuracy": "checks"}`` still counts as judge-routing.
        """
        if isinstance(spec, Metric):
            return spec.name in (*_JUDGE_NAMES, "checks")
        if isinstance(spec, str):
            return spec in _JUDGE_NAMES or spec == "checks"
        return False

    if metrics is None:
        # Judge-only evals grade every row with the judge; otherwise default to
        # the per-example checks dispatcher (rows fall back to exact_match).
        return [_judge_metric()] if judge_obj is not None else [checks(judge=judge_obj)]

    if isinstance(metrics, dict):
        # A mapping renames each metric after its key (harness convention).
        raw_specs: list[Any] = list(metrics.values())
        resolved = []
        for name, spec in metrics.items():
            metric = _resolve_one(spec)
            resolved.append(Metric(name, metric.fn, needs_example=metric.needs_example))
    else:
        raw_specs = list(metrics) if isinstance(metrics, (list, tuple)) else [metrics]
        resolved = [_resolve_one(spec) for spec in raw_specs]
    # A supplied judge also grades every row -- unless a metric already routes
    # it: an explicit "judge" entry, or a "checks" dispatcher (which judges
    # exactly the rows that declare a judge check; grading every row anyway
    # would burn judge calls the dataset opted out of).
    if judge_obj is not None and not any(_routes_judge(spec) for spec in raw_specs):
        resolved.append(_judge_metric())
    return resolved


__all__ = ["evaluate_agent"]
