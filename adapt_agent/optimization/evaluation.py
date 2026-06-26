"""Evaluation harness: run an agent across a golden dataset and score it.

The :class:`EvaluationHarness` is the measurement engine shared by every
optimizer. Given an agent (anything runnable) and a
:class:`~adapt_agent.optimization.dataset.GoldenDataset`, it runs each example,
applies a set of metrics (including any LLM-as-judge metric), captures latency
and errors, and returns a structured :class:`EvaluationReport` whose ``score`` is
the aggregate of the designated primary metric.

Running is non-fatal: an exception on one example is recorded as a zero-scored
error rather than aborting the whole evaluation, so a single bad candidate never
crashes an optimization loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, cast

from adapt_agent.optimization.dataset import Example, GoldenDataset
from adapt_agent.optimization.metrics import Metric, MetricFn, coerce_metric

logger = logging.getLogger(__name__)


# Framework-native "run" methods, sync entrypoints first so frameworks that
# expose both (e.g. Pydantic AI ``run_sync`` + async ``run``) use the sync one.
# Covers LangGraph (``invoke``), CrewAI (``kickoff``), Pydantic AI (``run_sync``),
# the OptimizableAgent / Microsoft Agent Framework (``run``).
_RUN_METHOD_NAMES = ("run_sync", "invoke", "kickoff", "run")


def resolve_runner(agent: Any) -> Callable[[Any], Any]:
    """Return a ``Callable[[input], output]`` from a variety of agent shapes.

    Resolution order: a recognized framework run method (``run_sync``, ``invoke``,
    ``kickoff``, ``run`` -- so a raw LangGraph graph, CrewAI ``Crew`` or Pydantic
    AI ``Agent`` works directly), then a plain callable, then a governed agent's
    ``execute``. Method-driven runners have their result *resolved* -- coroutines
    are awaited and async/sync generators are drained -- so async-native agents
    can be driven synchronously. A plain callable is returned unwrapped. Raises
    :class:`TypeError` when nothing runnable is found.
    """
    for name in _RUN_METHOD_NAMES:
        method = getattr(agent, name, None)
        if callable(method):
            return _resolving_runner(method)
    if callable(agent):
        return cast("Callable[[Any], Any]", agent)
    execute = getattr(agent, "execute", None)
    if callable(execute):
        return _resolving_runner(execute)
    raise TypeError(
        "Cannot evaluate object: expected a callable, an object with a callable "
        "run method (run_sync/invoke/kickoff/run), or a governed agent with "
        "`execute`. Pass an explicit runner if your framework differs."
    )


def _resolving_runner(method: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Wrap a framework run method so sync/async results are materialized."""
    # Imported lazily; reuses the adapters' result resolver (coroutines awaited,
    # async/sync generators drained). Importing it pulls in no framework SDK.
    from adapt_agent.adapters._governed import _resolve_result

    def _runner(input_data: Any) -> Any:
        return _resolve_result(method(input_data))

    return _runner


@dataclass
class ExampleResult:
    """Per-example outcome captured during evaluation."""

    index: int
    inputs: Any
    output: Any
    expected: Any
    scores: dict[str, float]
    latency: float
    error: str | None = None

    @property
    def primary_ok(self) -> bool:
        return self.error is None


@dataclass
class EvaluationReport:
    """Aggregate evaluation outcome over a dataset.

    Args:
        aggregate: Mean of each metric across non-error examples.
        primary_metric: Name of the metric whose aggregate is the headline score.
        results: Per-example :class:`ExampleResult` records.
        n_errors: Number of examples that raised during execution.
        total_latency: Wall-clock seconds summed across example runs.
    """

    aggregate: dict[str, float]
    primary_metric: str
    results: list[ExampleResult] = field(default_factory=list)
    n_errors: int = 0
    total_latency: float = 0.0

    @property
    def score(self) -> float:
        """The headline score (aggregate of the primary metric, 0.0 if absent)."""
        return self.aggregate.get(self.primary_metric, 0.0)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.n if self.results else 0.0

    def failures(self, *, metric: str | None = None, threshold: float = 1.0) -> list[ExampleResult]:
        """Return examples scoring below ``threshold`` on a metric (or errored).

        Defaults to the primary metric. Useful for feeding an LLM proposer the
        cases an instruction still gets wrong.
        """
        name = metric or self.primary_metric
        out: list[ExampleResult] = []
        for r in self.results:
            if r.error is not None or r.scores.get(name, 0.0) < threshold:
                out.append(r)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "primary_metric": self.primary_metric,
            "aggregate": dict(self.aggregate),
            "n": self.n,
            "n_errors": self.n_errors,
            "avg_latency": self.avg_latency,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        metrics = ", ".join(f"{k}={v:.3f}" for k, v in sorted(self.aggregate.items()))
        return (
            f"EvaluationReport(score={self.score:.3f} [{self.primary_metric}], "
            f"n={self.n}, errors={self.n_errors}, {metrics})"
        )


class EvaluationHarness:
    """Runs an agent across a dataset and scores it with one or more metrics.

    Args:
        metrics: Metrics to apply. A single metric, a list, or a name->metric
            mapping. Bare callables are wrapped automatically.
        primary_metric: Name of the headline metric. Defaults to the first
            metric supplied.
        capture_output: Store each produced output on the result records (set
            ``False`` to keep reports small for very large datasets).
        max_results: Cap on stored per-example results (bounds memory on huge
            datasets); aggregates are computed over everything regardless.
    """

    def __init__(
        self,
        metrics: Metric | MetricFn | list | dict | None = None,
        *,
        primary_metric: str | None = None,
        capture_output: bool = True,
        max_results: int = 10_000,
    ):
        self.metrics: list[Metric] = self._normalize_metrics(metrics)
        if not self.metrics:
            raise ValueError("EvaluationHarness requires at least one metric")
        names = [m.name for m in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate metric names: {names}")
        self.primary_metric = primary_metric or names[0]
        if self.primary_metric not in names:
            raise ValueError(f"primary_metric {self.primary_metric!r} not among metrics {names}")
        self.capture_output = capture_output
        self.max_results = max_results

    @staticmethod
    def _normalize_metrics(metrics: Any) -> list[Metric]:
        if metrics is None:
            return []
        if isinstance(metrics, dict):
            normalized: list[Metric] = []
            for name, m in metrics.items():
                if isinstance(m, Metric):
                    # Honour the mapping key as the metric's reporting name.
                    normalized.append(Metric(name, m.fn, needs_example=m.needs_example))
                else:
                    normalized.append(Metric(name, m))
                # Note: a bare callable mapped under ``name`` becomes an
                # example-unaware metric (signature ``(output, expected)``).
            return normalized
        if isinstance(metrics, (list, tuple)):
            return [coerce_metric(m) for m in metrics]
        return [coerce_metric(metrics)]

    def evaluate(self, agent: Any, dataset: GoldenDataset) -> EvaluationReport:
        """Run ``agent`` over ``dataset`` and return an :class:`EvaluationReport`."""
        runner = resolve_runner(agent)
        results: list[ExampleResult] = []
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        n_errors = 0
        total_latency = 0.0

        for index, example in enumerate(dataset):
            result = self._run_one(runner, index, example)
            total_latency += result.latency
            if result.error is not None:
                n_errors += 1
            for name, score in result.scores.items():
                sums[name] = sums.get(name, 0.0) + score
                counts[name] = counts.get(name, 0) + 1
            if len(results) < self.max_results:
                results.append(result)

        aggregate = {name: sums[name] / counts[name] for name in sums if counts[name]}
        # Ensure every metric appears even if all examples errored before scoring.
        for m in self.metrics:
            aggregate.setdefault(m.name, 0.0)
        return EvaluationReport(
            aggregate=aggregate,
            primary_metric=self.primary_metric,
            results=results,
            n_errors=n_errors,
            total_latency=total_latency,
        )

    def _run_one(self, runner: Callable[[Any], Any], index: int, example: Example) -> ExampleResult:
        start = time.perf_counter()
        try:
            output = runner(example.inputs)
            latency = time.perf_counter() - start
        except Exception as exc:  # non-fatal: record and score zero
            latency = time.perf_counter() - start
            logger.warning("Agent raised on example %d: %s", index, exc)
            return ExampleResult(
                index=index,
                inputs=example.inputs if self.capture_output else None,
                output=None,
                expected=example.expected,
                scores={m.name: 0.0 for m in self.metrics},
                latency=latency,
                error=str(exc),
            )

        scores: dict[str, float] = {}
        for metric in self.metrics:
            try:
                scores[metric.name] = metric(output, example.expected, example)
            except Exception as exc:
                logger.warning("Metric %s raised on example %d: %s", metric.name, index, exc)
                scores[metric.name] = 0.0

        return ExampleResult(
            index=index,
            inputs=example.inputs if self.capture_output else None,
            output=output if self.capture_output else None,
            expected=example.expected,
            scores=scores,
            latency=latency,
        )


__all__ = ["EvaluationHarness", "EvaluationReport", "ExampleResult", "resolve_runner"]
