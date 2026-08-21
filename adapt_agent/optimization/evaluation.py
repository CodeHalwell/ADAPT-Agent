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

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, cast

from adapt_agent.optimization.dataset import Example, GoldenDataset
from adapt_agent.optimization.metrics import Metric, MetricFn, coerce_metric
from adapt_agent.optimization.retry import (
    DEFAULT_RETRY_POLICY,
    RetryPolicy,
    retries_already_exhausted,
)

logger = logging.getLogger(__name__)


# Framework-native "run" methods, sync entrypoints first so frameworks that
# expose both (e.g. Pydantic AI ``run_sync`` + async ``run``) use the sync one.
# Covers LangGraph (``invoke``), CrewAI (``kickoff``), Pydantic AI (``run_sync``),
# the OptimizableAgent / Microsoft Agent Framework (``run``).
_RUN_METHOD_NAMES = ("run_sync", "invoke", "kickoff", "run")

# The same list for the async path, with the preference *inverted*: a framework
# exposing both styles should be driven by its async entry point so the event
# loop is never blocked. ``aexecute`` leads because a governed agent exposes it
# alongside ``execute`` (see adapt_agent.adapters._governed).
_ARUN_METHOD_NAMES = (
    "aexecute",
    "arun",
    "ainvoke",
    "kickoff_async",
    "run",
    "invoke",
    "kickoff",
    "run_sync",
    "execute",
)


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


def aresolve_runner(agent: Any) -> Callable[[Any], Awaitable[Any]]:
    """Return an ``async`` ``Callable[[input], output]`` from an agent.

    The async twin of :func:`resolve_runner`. Two differences matter:

    * **Async entry points win.** A framework exposing both styles (Pydantic AI
      ``run``/``run_sync``, LangGraph ``ainvoke``/``invoke``) is driven by the
      async one, so the caller's event loop is never blocked. A governed agent's
      :meth:`aexecute <adapt_agent.adapters._governed._GovernedAgent.aexecute>`
      is preferred over its ``execute``.
    * **The result is awaited in the caller's loop**, so concurrency is real and
      ``contextvars`` (tracing spans, request-scoped state) are preserved.

    A purely synchronous agent works too: there is simply nothing to await, and
    it is called directly. Note that such an agent *blocks the loop* for the
    duration of its call -- for a sync agent prefer
    :meth:`EvaluationHarness.evaluate` with ``concurrency``, which uses threads.
    """
    for name in _ARUN_METHOD_NAMES:
        method = getattr(agent, name, None)
        if callable(method):
            return _aresolving_runner(method)
    if callable(agent):
        return _aresolving_runner(cast("Callable[[Any], Any]", agent))
    raise TypeError(
        "Cannot evaluate object: expected a callable, an object with a callable "
        "run method (aexecute/arun/ainvoke/run/invoke/kickoff/run_sync), or a "
        "governed agent. Pass an explicit runner if your framework differs."
    )


def _resolving_runner(method: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Wrap a framework run method so sync/async results are materialized."""
    # Imported lazily; reuses the adapters' result resolver (coroutines awaited,
    # async/sync generators drained). Importing it pulls in no framework SDK.
    from adapt_agent.adapters._governed import _resolve_result

    def _runner(input_data: Any) -> Any:
        return _resolve_result(method(input_data))

    return _runner


def _aresolving_runner(method: Callable[[Any], Any]) -> Callable[[Any], Awaitable[Any]]:
    """Wrap a run method so its result is awaited/drained in the caller's loop."""
    from adapt_agent.adapters._governed import _aresolve_result

    async def _runner(input_data: Any) -> Any:
        return await _aresolve_result(method(input_data))

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
    #: Metrics whose own provider call failed transiently on this row. Only
    #: these lose their score; the rest of the row still counts, so a throttled
    #: secondary judge cannot erase a primary that computed fine.
    transient_metrics: tuple[str, ...] = ()
    #: True when this example failed with a *transient* provider error (429,
    #: 5xx, timeout) that survived every retry. Such a result says nothing about
    #: the agent's quality, so it is kept out of the aggregate and out of
    #: :meth:`EvaluationReport.failures` -- see :attr:`EvaluationReport.n_transient_errors`.
    transient: bool = False
    #: How many times the agent was called for this example (1 = no retry).
    attempts: int = 1

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
        n_transient_errors: How many of ``n_errors`` were transient provider
            faults (throttling, 5xx, timeouts) that outlived their retries.
            These are *excluded from* :attr:`aggregate`, so a run that hit rate
            limits scores the same as one that did not -- otherwise whichever
            candidate was evaluated while the provider was busiest would score
            lowest, and an optimizer would select for luck. Compare it against
            ``n`` to judge whether a run is trustworthy at all.
        total_latency: Wall-clock seconds summed across example runs.
    """

    aggregate: dict[str, float]
    primary_metric: str
    results: list[ExampleResult] = field(default_factory=list)
    n_errors: int = 0
    n_transient_errors: int = 0
    #: Rows actually evaluated. Differs from :attr:`n` once a dataset
    #: exceeds ``max_results``, which bounds *stored records* only -- every
    #: row is still run and aggregated. Use this as the denominator for the
    #: error counts, which are also totals: ``n`` would give impossible
    #: summaries like ``n=1, n_transient_errors=4``.
    n_evaluated: int = 0
    total_latency: float = 0.0
    #: Default threshold used by :meth:`failures` when none is supplied. Set by
    #: the producing :class:`EvaluationHarness` from its ``failure_threshold``;
    #: ``1.0`` reproduces the historical "anything short of perfect is a failure"
    #: behaviour. Lower it for continuous-score metrics so :meth:`failures`
    #: isn't flooded.
    failure_threshold: float = 1.0

    @property
    def score(self) -> float:
        """The headline score (aggregate of the primary metric, 0.0 if absent)."""
        return self.aggregate.get(self.primary_metric, 0.0)

    def __post_init__(self) -> None:
        # `n_evaluated` is set explicitly by the harness, which knows the true
        # total even when `max_results` bounded what it kept. A report built by
        # hand -- the dataclass is public -- has only its records to go on, and
        # leaving the count at 0 would silently zero `avg_latency` and
        # `n_scored` for those callers.
        if not self.n_evaluated:
            self.n_evaluated = len(self.results)

    @property
    def n(self) -> int:
        """How many per-example records are *stored* (bounded by ``max_results``).

        For the number of rows actually run, use :attr:`n_evaluated`.
        """
        return len(self.results)

    @property
    def n_scored(self) -> int:
        """Rows that contributed a score: evaluated, minus transient dropouts."""
        return max(0, self.n_evaluated - self.n_transient_errors)

    @property
    def is_complete(self) -> bool:
        """Whether every example actually produced a score.

        ``False`` means at least one row was dropped for a transient provider
        failure, so :attr:`score` is a mean over a *subset* of the dataset.
        Two such scores are not comparable with each other, nor with a complete
        one -- see :meth:`Optimizer._record`, which refuses to crown an
        incomplete trial for exactly that reason.
        """
        return self.n_transient_errors == 0

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.n_evaluated if self.n_evaluated else 0.0

    def failures(
        self, *, metric: str | None = None, threshold: float | None = None
    ) -> list[ExampleResult]:
        """Return examples scoring below ``threshold`` on a metric (or errored).

        Defaults to the primary metric. Useful for feeding an LLM proposer the
        cases an instruction still gets wrong.

        When ``threshold`` is ``None`` it falls back to :attr:`failure_threshold`
        (set by the producing harness, default ``1.0``). Supplying ``threshold``
        explicitly always wins. Lower the threshold for continuous-score metrics
        so this doesn't return every imperfect example.

        Examples that failed transiently (see :attr:`n_transient_errors`) are
        omitted: this list is what gets handed to an LLM proposer as "cases your
        instruction still gets wrong", and "the provider throttled you" is not
        evidence about the instruction.
        """
        name = metric or self.primary_metric
        cutoff = self.failure_threshold if threshold is None else threshold
        out: list[ExampleResult] = []
        for r in self.results:
            if r.transient:
                continue
            if r.error is not None or r.scores.get(name, 0.0) < cutoff:
                out.append(r)
        return out

    def below(self, metric: str, threshold: float) -> list[ExampleResult]:
        """Return examples whose ``metric`` score is below ``threshold``.

        Unlike :meth:`failures`, this looks only at the named metric's score and
        does **not** force-include errored examples (an errored example simply
        scores ``0.0`` and is included if that is below ``threshold``).
        """
        return [r for r in self.results if r.scores.get(metric, 0.0) < threshold]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "primary_metric": self.primary_metric,
            "aggregate": dict(self.aggregate),
            "n": self.n,
            "n_evaluated": self.n_evaluated,
            "n_errors": self.n_errors,
            "n_transient_errors": self.n_transient_errors,
            "is_complete": self.is_complete,
            "avg_latency": self.avg_latency,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        metrics = ", ".join(f"{k}={v:.3f}" for k, v in sorted(self.aggregate.items()))
        return (
            f"EvaluationReport(score={self.score:.3f} [{self.primary_metric}], "
            f"n={self.n}, errors={self.n_errors}"
            + (f" ({self.n_transient_errors} transient)" if self.n_transient_errors else "")
            + f", {metrics})"
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
        failure_threshold: Default cutoff for
            :meth:`EvaluationReport.failures` on reports this harness produces.
            ``1.0`` (the default) keeps the historical "only a perfect score
            passes" behaviour; lower it for continuous-score metrics so the
            failure set used by proposers isn't flooded with near-misses.
        cache: Reserved flag for future result memoization. Currently a no-op:
            ``evaluate`` always re-runs the agent so results stay correct even
            for stateful/impure agents. Kept so callers can opt in later without
            a signature change; setting it does **not** enable any caching today.
        output_extractor: Optional post-processor applied to each raw agent
            output before scoring (and before storing on the result record).
            Pass :func:`adapt_agent.optimization.extractors.extract_output_text`
            to unwrap framework-native results (Pydantic AI ``AgentRunResult``,
            LangGraph state, ADK events, ...) into final response text so
            text/number metrics compare against the answer rather than a
            ``repr``. ``None`` (the default) keeps raw outputs. Extractor
            errors are non-fatal: the raw output is scored instead.
    """

    def __init__(
        self,
        metrics: Metric | MetricFn | list | dict | None = None,
        *,
        primary_metric: str | None = None,
        capture_output: bool = True,
        max_results: int = 10_000,
        failure_threshold: float = 1.0,
        cache: bool = False,
        output_extractor: Callable[[Any], Any] | None = None,
        concurrency: int = 1,
        retry: RetryPolicy | None = None,
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
        self.failure_threshold = failure_threshold
        # Reserved: no caching is performed today (see class docstring).
        self.cache = cache
        self.output_extractor = output_extractor
        self.concurrency = max(1, concurrency)
        self.retry = DEFAULT_RETRY_POLICY if retry is None else retry

    @staticmethod
    def _normalize_metrics(metrics: Any) -> list[Metric]:
        if metrics is None:
            return []
        if isinstance(metrics, dict):
            normalized: list[Metric] = []
            for name, m in metrics.items():
                if isinstance(m, Metric):
                    # Honour the mapping key as the metric's reporting name.
                    normalized.append(
                        Metric(
                            name,
                            m.fn,
                            needs_example=m.needs_example,
                            structural=m.structural,
                        )
                    )
                else:
                    normalized.append(Metric(name, m))
                # Note: a bare callable mapped under ``name`` becomes an
                # example-unaware metric (signature ``(output, expected)``).
            return normalized
        if isinstance(metrics, (list, tuple)):
            return [coerce_metric(m) for m in metrics]
        return [coerce_metric(metrics)]

    def evaluate(
        self, agent: Any, dataset: GoldenDataset, *, concurrency: int | None = None
    ) -> EvaluationReport:
        """Run ``agent`` over ``dataset`` and return an :class:`EvaluationReport`.

        Args:
            agent: Anything :func:`resolve_runner` accepts.
            dataset: The golden dataset to score against.
            concurrency: How many examples to run at once. ``None`` (the
                default) uses the harness's own ``concurrency``, itself ``1``
                unless set, which keeps the historical strictly-serial
                behaviour. Higher values
                run examples in a thread pool, which is the right tool for a
                *synchronous* agent whose time goes on network I/O -- an LLM
                round-trip per example. For an async-native agent prefer
                :meth:`aevaluate`, which needs no threads.

        Ordering is by example index regardless of completion order, and the
        per-example non-fatal error handling is identical on both paths.
        """
        runner = resolve_runner(agent)
        concurrency = self.concurrency if concurrency is None else concurrency
        if concurrency <= 1:
            return self._build_report(
                self._run_one(runner, index, example) for index, example in enumerate(dataset)
            )
        return self._build_report(self._threaded_results(runner, dataset, concurrency))

    def _threaded_results(
        self, runner: Callable[[Any], Any], dataset: GoldenDataset, concurrency: int
    ) -> Iterator[ExampleResult]:
        """Run examples in a thread pool, yielding results in index order.

        Deliberately not ``ThreadPoolExecutor.map``: that submits the *whole*
        iterable up front (``fs = [self.submit(fn, *args) for args in ...]``),
        holding one future per example and consuming the dataset eagerly. This
        keeps at most ``concurrency`` runs in flight and pulls from the dataset
        lazily, so memory stays bounded on a large dataset -- the same property
        :meth:`aevaluate` gets from its worker pool.

        Refilling waits on *whichever* future finishes first rather than on the
        oldest. Waiting on the oldest would idle the rest of the pool behind one
        slow example -- and with LLM latency the spread is wide, so that
        collapses the concurrency actually achieved. Results therefore arrive
        out of order, which is fine: the accumulator restores index order.
        """
        examples = enumerate(dataset)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            pending = {
                pool.submit(self._run_one, runner, index, example)
                for index, example in islice(examples, concurrency)
            }
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    nxt = next(examples, None)
                    if nxt is not None:
                        pending.add(pool.submit(self._run_one, runner, nxt[0], nxt[1]))
                    yield future.result()

    async def aevaluate(
        self, agent: Any, dataset: GoldenDataset, *, concurrency: int | None = None
    ) -> EvaluationReport:
        """Async twin of :meth:`evaluate`, for async-native agents.

        Runs up to ``concurrency`` examples at once against the caller's event
        loop -- no threads, so ``contextvars`` (tracing spans) are preserved and
        a governed agent's :meth:`aexecute` path is used. ``None`` takes the
        harness's own ``concurrency``.

        This is the difference between an eval you run once and one you run
        often: a coordinate-ascent sweep is ``max_evals x len(dataset)`` LLM
        round-trips, which is untenable strictly serially.

        Results are aggregated as they complete but **reported in index order**.
        Memory stays bounded whatever the dataset size: exactly ``concurrency``
        tasks are alive at a time (a worker pool over the dataset, not one task
        per example), and records past ``max_results`` are aggregated then
        dropped rather than accumulated.
        """
        runner = aresolve_runner(agent)
        concurrency = self.concurrency if concurrency is None else concurrency
        accumulator = _Accumulator(self)
        examples = enumerate(dataset)

        async def _worker() -> None:
            while True:
                # Safe without a lock: ``next`` on a shared iterator contains no
                # await, so the event loop cannot switch coroutines mid-call.
                try:
                    index, example = next(examples)
                except StopIteration:
                    return
                accumulator.add(await self._arun_one(runner, index, example))

        workers = [asyncio.ensure_future(_worker()) for _ in range(max(1, concurrency))]
        try:
            await asyncio.gather(*workers)
        except BaseException:
            # A cancellation (or an error escaping a worker) must not leave the
            # remaining workers running detached.
            for worker in workers:
                worker.cancel()
            raise
        return accumulator.report()

    # -- per-example execution -------------------------------------------------

    def _run_one(self, runner: Callable[[Any], Any], index: int, example: Example) -> ExampleResult:
        start = time.perf_counter()
        policy = self.retry
        attempt = 1
        while True:
            try:
                output = runner(example.inputs)
            except Exception as exc:  # non-fatal: retry if transient, else record
                if policy.should_retry(exc, attempt):
                    delay = policy.delay_for(exc, attempt)
                    logger.info(
                        "Transient error on example %d (attempt %d/%d), retrying in %.2fs: %s",
                        index,
                        attempt,
                        policy.attempts,
                        delay,
                        exc,
                    )
                    # A thread-pool worker, or the caller's own thread on the
                    # serial path: blocking here is what we want, and it is the
                    # backpressure that lets the provider recover.
                    time.sleep(delay)
                    attempt += 1
                    continue
                return self._error_result(
                    index, example, time.perf_counter() - start, exc, attempts=attempt
                )
            return self._score_one(
                index, example, output, time.perf_counter() - start, attempts=attempt
            )

    async def _arun_one(
        self, runner: Callable[[Any], Awaitable[Any]], index: int, example: Example
    ) -> ExampleResult:
        start = time.perf_counter()
        policy = self.retry
        attempt = 1
        while True:
            try:
                output = await runner(example.inputs)
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # non-fatal: retry if transient, else record
                if not policy.should_retry(exc, attempt):
                    return self._error_result(
                        index, example, time.perf_counter() - start, exc, attempts=attempt
                    )
                delay = policy.delay_for(exc, attempt)
                logger.info(
                    "Transient error on example %d (attempt %d/%d), retrying in %.2fs: %s",
                    index,
                    attempt,
                    policy.attempts,
                    delay,
                    exc,
                )
                # `asyncio.sleep`, never `time.sleep`: this coroutine is sharing
                # the caller's loop with every other in-flight example.
                await asyncio.sleep(delay)
                attempt += 1
        latency = time.perf_counter() - start
        # Scoring goes to a worker thread. Metrics are synchronous by contract,
        # and an LLM judge's provider call is a *network* round trip: run inline
        # here, the agent calls overlap but their judging serialises on the loop
        # and stalls every other task. `asyncio.to_thread` propagates
        # `contextvars`, so a judge still sees the caller's tracing context.
        return await asyncio.to_thread(self._score_one, index, example, output, latency, attempt)

    def _error_result(
        self, index: int, example: Example, latency: float, exc: Exception, *, attempts: int = 1
    ) -> ExampleResult:
        transient = self.retry.should_retry(exc, 0)
        if transient:
            logger.warning(
                "Example %d failed transiently after %d attempt(s); excluded from the "
                "score rather than counted against the agent: %s",
                index,
                attempts,
                exc,
            )
        else:
            logger.warning("Agent raised on example %d: %s", index, exc)
        return ExampleResult(
            index=index,
            inputs=example.inputs if self.capture_output else None,
            output=None,
            expected=example.expected,
            scores={m.name: 0.0 for m in self.metrics},
            latency=latency,
            error=str(exc),
            transient=transient,
            attempts=attempts,
        )

    def _score_one(
        self, index: int, example: Example, output: Any, latency: float, attempts: int = 1
    ) -> ExampleResult:
        """Apply the output extractor and every metric. Shared by both paths."""
        if self.output_extractor is not None:
            try:
                output = self.output_extractor(output)
            except Exception as exc:  # non-fatal: score the raw output instead
                logger.warning("Output extractor raised on example %d: %s", index, exc)

        scores: dict[str, float] = {}
        transient_metrics: list[str] = []
        for metric in self.metrics:
            score, metric_transient = self._score_with_metric(metric, output, example, index)
            scores[metric.name] = score
            if metric_transient:
                transient_metrics.append(metric.name)

        # Only the *primary* makes the whole row unusable: it is the number the
        # optimizer compares, so a gap there means the row cannot be ranked. A
        # throttled secondary loses its own sample and nothing else -- marking
        # the row transient for it would delete a primary score that computed
        # perfectly well, and (with the completeness gate) could abort a run.
        transient = self.primary_metric in transient_metrics

        return ExampleResult(
            index=index,
            inputs=example.inputs if self.capture_output else None,
            output=output if self.capture_output else None,
            expected=example.expected,
            scores=scores,
            latency=latency,
            error="transient metric failure" if transient else None,
            transient=transient,
            transient_metrics=tuple(transient_metrics),
            attempts=attempts,
        )

    def _score_with_metric(
        self, metric: Metric, output: Any, example: Example, index: int
    ) -> tuple[float, bool]:
        """Apply one metric, retrying it on transient failures.

        A metric is free to be a network call -- an LLM judge is the documented
        default for open-ended tasks -- so throttling *here* biases candidate
        selection exactly as throttling on the agent call does. It therefore
        gets the same treatment, and through the same configured policy: a
        custom ``RetryPolicy(is_transient=...)`` governs metric calls too, and a
        provider-backed metric that is not :class:`LLMJudge` still gets its
        retries rather than one attempt.

        Returns ``(score, transient)``. A transient failure that outlives its
        retries scores ``0.0`` *and* flags the row, so the caller can drop it
        from the aggregate instead of counting it against the agent.
        """
        policy = self.retry
        attempt = 1
        while True:
            try:
                return metric(output, example.expected, example), False
            except Exception as exc:
                # A metric that runs its own retry loop (LLMJudge does) stamps
                # the error when its budget is spent. Retrying here as well
                # would multiply the budgets and reset the backoff, adding
                # load exactly while the provider is throttling.
                if retries_already_exhausted(exc):
                    logger.warning(
                        "Metric %s failed transiently on example %d (its own retries "
                        "were already spent); excluded from the score: %s",
                        metric.name,
                        index,
                        exc,
                    )
                    return 0.0, True
                if policy.should_retry(exc, attempt):
                    delay = policy.delay_for(exc, attempt)
                    logger.info(
                        "Metric %s hit a transient error on example %d (attempt %d/%d), "
                        "retrying in %.2fs: %s",
                        metric.name,
                        index,
                        attempt,
                        policy.attempts,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                # `should_retry` is False either because the error is not
                # transient or because the attempts are spent; only the second
                # marks the row, so ask the policy's own classifier which it is.
                if policy.should_retry(exc, 0):
                    logger.warning(
                        "Metric %s failed transiently on example %d after %d attempt(s); "
                        "excluded from the score rather than counted against the agent: %s",
                        metric.name,
                        index,
                        attempt,
                        exc,
                    )
                    return 0.0, True
                logger.warning("Metric %s raised on example %d: %s", metric.name, index, exc)
                return 0.0, False

    def _build_report(self, results_iter: Any) -> EvaluationReport:
        """Aggregate a stream of per-example results into a report."""
        accumulator = _Accumulator(self)
        for result in results_iter:
            accumulator.add(result)
        return accumulator.report()


class _Accumulator:
    """Running aggregation over per-example results.

    Shared by the serial, threaded and async paths so all three produce
    identical reports. Records are kept by *index* rather than by arrival, so
    results may be added out of order (as they do on the async path) and still
    come back index-ordered -- while never holding more than ``max_results`` of
    them.
    """

    def __init__(self, harness: EvaluationHarness):
        self._harness = harness
        self._results: list[ExampleResult] = []
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._unordered = False
        self._n_errors = 0
        self._n_transient_errors = 0
        self._n_evaluated = 0
        self._total_latency = 0.0

    def add(self, result: ExampleResult) -> None:
        self._n_evaluated += 1
        self._total_latency += result.latency
        if result.error is not None:
            self._n_errors += 1
        if result.transient:
            # A throttled example carries a zero score it did not earn. Folding
            # that into the mean is what lets an optimizer prefer whichever
            # candidate happened to run when the provider was least busy, so the
            # record is kept (and counted) but never scored.
            self._n_transient_errors += 1
        else:
            skip = set(result.transient_metrics)
            for name, score in result.scores.items():
                if name in skip:
                    # This metric's own provider call failed; its zero is not a
                    # measurement. Other metrics on the row are unaffected.
                    continue
                self._sums[name] = self._sums.get(name, 0.0) + score
                self._counts[name] = self._counts.get(name, 0) + 1
        if result.index < self._harness.max_results:
            if self._results and result.index < self._results[-1].index:
                self._unordered = True
            self._results.append(result)

    def report(self) -> EvaluationReport:
        if self._unordered:
            self._results.sort(key=lambda r: r.index)
        aggregate = {n: self._sums[n] / self._counts[n] for n in self._sums if self._counts[n]}
        # Ensure every metric appears even if all examples errored before scoring.
        for metric in self._harness.metrics:
            aggregate.setdefault(metric.name, 0.0)
        return EvaluationReport(
            aggregate=aggregate,
            primary_metric=self._harness.primary_metric,
            results=self._results,
            n_errors=self._n_errors,
            n_transient_errors=self._n_transient_errors,
            n_evaluated=self._n_evaluated,
            total_latency=self._total_latency,
            failure_threshold=self._harness.failure_threshold,
        )


__all__ = [
    "EvaluationHarness",
    "EvaluationReport",
    "ExampleResult",
    "RetryPolicy",
    "aresolve_runner",
    "resolve_runner",
]
