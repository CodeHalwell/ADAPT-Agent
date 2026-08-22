"""Tests for transient-error retry and the harness-level concurrency default.

The bug these guard against is a *measurement* bug, not a crash: under
concurrency, provider throttling is expected, and scoring a throttled example
zero makes it indistinguishable from a bad prompt. Worse, it biases
systematically -- whichever candidate is evaluated while the provider is busiest
scores lowest -- so an optimizer can select a prompt for having been lucky.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from adapt_agent.optimization.dataset import Example, GoldenDataset
from adapt_agent.optimization.evaluation import EvaluationHarness
from adapt_agent.optimization.metrics import exact_match
from adapt_agent.optimization.retry import (
    DEFAULT_RETRY_POLICY,
    RetryPolicy,
    is_transient_error,
    retry_after_seconds,
)

#: Backoff small enough to keep the suite fast, deterministic (no jitter).
FAST = RetryPolicy(attempts=3, initial_backoff=0.001, jitter=0.0)


def _dataset() -> GoldenDataset:
    return GoldenDataset([Example(inputs=x, expected="ok") for x in ("a", "boom", "c")])


# -- classification -----------------------------------------------------------


class _Response:
    def __init__(self, status: int, headers: dict | None = None) -> None:
        self.status_code = status
        self.headers = headers or {}


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 504])
def test_transient_http_statuses_are_recognised(status: int) -> None:
    exc = RuntimeError("upstream said no")
    exc.status_code = status  # type: ignore[attr-defined]
    assert is_transient_error(exc) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_error_statuses_are_not_transient(status: int) -> None:
    """A 400 means *your request* is wrong; retrying it just wastes quota."""
    exc = RuntimeError("bad request")
    exc.status_code = status  # type: ignore[attr-defined]
    assert is_transient_error(exc) is False


def test_status_on_an_attached_response_is_found() -> None:
    exc = RuntimeError("nope")
    exc.response = _Response(429)  # type: ignore[attr-defined]
    assert is_transient_error(exc) is True


def test_an_explicit_status_beats_a_misleading_message() -> None:
    """A 400 whose text happens to mention a timeout is still not transient."""
    exc = RuntimeError("request timed out while validating")
    exc.status_code = 400  # type: ignore[attr-defined]
    assert is_transient_error(exc) is False


@pytest.mark.parametrize(
    "name",
    ["RateLimitError", "APITimeoutError", "ServiceUnavailableError", "ThrottlingException"],
)
def test_provider_exception_type_names_are_recognised(name: str) -> None:
    """No provider SDK is imported; classification is by duck-typed name."""
    exc = type(name, (Exception,), {})("something went wrong")
    assert is_transient_error(exc) is True


@pytest.mark.parametrize(
    "message",
    [
        "429 Too Many Requests",
        "Rate limit reached for gpt-4o",
        "The server is overloaded, please try again",
        "503 Service Unavailable",
        "Connection reset by peer",
    ],
)
def test_transient_messages_are_recognised(message: str) -> None:
    assert is_transient_error(RuntimeError(message)) is True


def test_an_ordinary_agent_failure_is_not_transient() -> None:
    assert is_transient_error(ValueError("the model returned malformed JSON")) is False


def test_a_bare_number_in_a_message_does_not_trip_classification() -> None:
    """`429` is matched as a word, so a token count can't masquerade as throttling."""
    assert is_transient_error(ValueError("produced 1429 tokens, over budget")) is False


def test_stop_signals_are_never_transient() -> None:
    assert is_transient_error(KeyboardInterrupt()) is False
    assert is_transient_error(SystemExit()) is False
    assert is_transient_error(asyncio.CancelledError()) is False


def test_retry_after_is_read_from_attribute_and_header() -> None:
    attr = RuntimeError("slow down")
    attr.retry_after = 2.5  # type: ignore[attr-defined]
    assert retry_after_seconds(attr) == 2.5

    header = RuntimeError("slow down")
    header.response = _Response(429, {"retry-after": "7"})  # type: ignore[attr-defined]
    assert retry_after_seconds(header) == 7.0


def test_an_unparseable_or_absurd_retry_after_is_ignored() -> None:
    http_date = RuntimeError("slow down")
    http_date.retry_after = "Wed, 21 Oct 2026 07:28:00 GMT"  # type: ignore[attr-defined]
    assert retry_after_seconds(http_date) is None

    absurd = RuntimeError("slow down")
    absurd.retry_after = 99_999  # type: ignore[attr-defined]
    assert retry_after_seconds(absurd) is None


def test_retry_after_is_preferred_over_computed_backoff_but_capped() -> None:
    policy = RetryPolicy(initial_backoff=0.5, max_backoff=5.0, jitter=0.0)
    exc = RuntimeError("slow down")
    exc.retry_after = 3.0  # type: ignore[attr-defined]
    assert policy.delay_for(exc, 1) == 3.0

    exc.retry_after = 120.0  # type: ignore[attr-defined]
    assert policy.delay_for(exc, 1) == 5.0


def test_backoff_grows_and_is_capped() -> None:
    policy = RetryPolicy(initial_backoff=1.0, multiplier=2.0, max_backoff=5.0, jitter=0.0)
    plain = RuntimeError("429")
    assert [policy.delay_for(plain, n) for n in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 5.0]


def test_a_broken_custom_classifier_does_not_abort_the_run() -> None:
    def explode(exc: BaseException) -> bool:
        raise RuntimeError("classifier is broken")

    policy = RetryPolicy(attempts=3, is_transient=explode)
    assert policy.should_retry(RuntimeError("429"), 1) is False


# -- harness behaviour --------------------------------------------------------


def test_a_transient_failure_is_retried_and_recovers() -> None:
    state = {"n": 0}

    def flaky(inputs):
        if inputs == "boom":
            state["n"] += 1
            if state["n"] < 3:
                raise RuntimeError("429 Too Many Requests")
        return "ok"

    report = EvaluationHarness([exact_match()], retry=FAST).evaluate(flaky, _dataset())
    assert report.score == 1.0
    assert report.n_errors == 0
    assert [r.attempts for r in report.results] == [1, 3, 1]


def test_an_exhausted_transient_failure_is_excluded_from_the_score() -> None:
    """The measurement bug: without this the score is 2/3, not 2/2.

    A throttled example contributes a zero it did not earn, so the candidate
    that happened to run while the provider was busiest scores lowest.
    """

    def throttled(inputs):
        if inputs == "boom":
            raise RuntimeError("429 Too Many Requests")
        return "ok"

    report = EvaluationHarness([exact_match()], retry=FAST).evaluate(throttled, _dataset())
    assert report.score == 1.0
    assert report.n_errors == 1
    assert report.n_transient_errors == 1
    assert report.to_dict()["n_transient_errors"] == 1
    # ...and it is not offered to a proposer as a case the prompt gets wrong.
    assert report.failures() == []
    assert report.results[1].transient is True
    assert report.results[1].attempts == 3


def test_a_broken_agent_still_scores_zero_and_is_not_retried() -> None:
    """Retrying is for the provider's faults, not the agent's."""
    calls = {"n": 0}

    def broken(inputs):
        calls["n"] += 1
        if inputs == "boom":
            raise ValueError("the model returned malformed JSON")
        return "ok"

    report = EvaluationHarness([exact_match()], retry=FAST).evaluate(broken, _dataset())
    assert report.score == pytest.approx(2 / 3)
    assert report.n_errors == 1
    assert report.n_transient_errors == 0
    assert len(report.failures()) == 1
    assert calls["n"] == 3, "a non-transient failure must not be retried"


def test_attempts_one_still_classifies_without_retrying() -> None:
    calls = {"n": 0}

    def throttled(inputs):
        calls["n"] += 1
        if inputs == "boom":
            raise RuntimeError("429 Too Many Requests")
        return "ok"

    harness = EvaluationHarness([exact_match()], retry=RetryPolicy(attempts=1))
    report = harness.evaluate(throttled, _dataset())
    assert calls["n"] == 3, "attempts=1 means one try per example"
    assert report.n_transient_errors == 1
    assert report.score == 1.0


def test_the_default_harness_retries_transient_errors() -> None:
    """Opt-out, not opt-in: the silent-corruption default is the one being fixed."""
    assert DEFAULT_RETRY_POLICY.attempts >= 2
    assert EvaluationHarness([exact_match()]).retry is DEFAULT_RETRY_POLICY


def test_async_transient_failure_is_retried_without_blocking_the_loop() -> None:
    delay = 0.05

    class Agent:
        def __init__(self) -> None:
            self.n = 0

        async def run(self, inputs):
            if inputs == "boom":
                self.n += 1
                if self.n < 2:
                    raise RuntimeError("429 Too Many Requests")
            return "ok"

    harness = EvaluationHarness(
        [exact_match()], retry=RetryPolicy(attempts=3, initial_backoff=delay, jitter=0.0)
    )

    async def main():
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(delay / 10)
                ticks += 1

        beat = asyncio.ensure_future(heartbeat())
        report = await harness.aevaluate(Agent(), _dataset())
        beat.cancel()
        with __import__("contextlib").suppress(asyncio.CancelledError):
            await beat
        return report, ticks

    report, ticks = asyncio.run(main())
    assert report.score == 1.0
    assert [r.attempts for r in report.results] == [1, 2, 1]
    assert ticks >= 5, "the backoff blocked the event loop (time.sleep instead of asyncio.sleep)"


def test_async_exhausted_transient_failure_is_excluded_from_the_score() -> None:
    class Agent:
        async def run(self, inputs):
            if inputs == "boom":
                raise RuntimeError("429 Too Many Requests")
            return "ok"

    harness = EvaluationHarness([exact_match()], retry=FAST)
    report = asyncio.run(harness.aevaluate(Agent(), _dataset()))
    assert report.score == 1.0
    assert report.n_transient_errors == 1
    assert report.failures() == []


# -- concurrency default ------------------------------------------------------


def _tracking_agent(delay: float = 0.05):
    """Return an agent plus a dict recording the peak number in flight."""
    stats = {"live": 0, "peak": 0}
    lock = threading.Lock()

    def agent(inputs):
        with lock:
            stats["live"] += 1
            stats["peak"] = max(stats["peak"], stats["live"])
        time.sleep(delay)
        with lock:
            stats["live"] -= 1
        return "ok"

    return agent, stats


def test_harness_concurrency_reaches_a_call_site_that_passes_no_kwargs() -> None:
    """This is the optimizer path.

    `Optimizer` calls `self.harness.evaluate(target, dataset)` with no keyword
    arguments, so a per-call `concurrency=` can never reach it -- and that is
    exactly the path that needs it, being `max_evals x len(dataset)` round trips
    rather than a single pass.
    """
    dataset = GoldenDataset([Example(inputs=str(i), expected="ok") for i in range(8)])

    agent, stats = _tracking_agent()
    EvaluationHarness([exact_match()], concurrency=4).evaluate(agent, dataset)
    assert stats["peak"] > 1, "the harness-level concurrency default never reached evaluate()"
    assert stats["peak"] <= 4

    agent, stats = _tracking_agent()
    EvaluationHarness([exact_match()]).evaluate(agent, dataset)
    assert stats["peak"] == 1, "the default must stay strictly serial"


def test_a_per_call_concurrency_overrides_the_instance_default() -> None:
    dataset = GoldenDataset([Example(inputs=str(i), expected="ok") for i in range(6)])
    agent, stats = _tracking_agent()
    EvaluationHarness([exact_match()], concurrency=4).evaluate(agent, dataset, concurrency=1)
    assert stats["peak"] == 1


def test_harness_concurrency_reaches_aevaluate_with_no_kwargs() -> None:
    delay = 0.05
    dataset = GoldenDataset([Example(inputs=str(i), expected="ok") for i in range(6)])

    class Agent:
        async def run(self, inputs):
            await asyncio.sleep(delay)
            return "ok"

    harness = EvaluationHarness([exact_match()], concurrency=6)
    started = time.perf_counter()
    report = asyncio.run(harness.aevaluate(Agent(), dataset))
    elapsed = time.perf_counter() - started

    assert report.score == 1.0
    assert elapsed < delay * len(dataset) * 0.6, "aevaluate ignored the harness concurrency default"


def test_a_non_positive_concurrency_is_clamped_to_serial() -> None:
    assert EvaluationHarness([exact_match()], concurrency=0).concurrency == 1
    assert EvaluationHarness([exact_match()], concurrency=-4).concurrency == 1


# -- transient failures in the judge / metric ---------------------------------
#
# Retrying only the agent call covers half the problem. An LLM judge is the
# documented metric for open-ended tasks, and its provider call is a network
# round trip too -- so a 429 there biases candidate selection exactly as one on
# the agent call does, just by a different route.


def _judge_harness(judge_fn, *, retry=FAST):
    from adapt_agent.optimization.judge import LLMJudge

    judge = LLMJudge(judge_fn, retry=retry)
    return EvaluationHarness([judge.as_metric()], retry=retry)


GOOD_VERDICT = '{"score": 10, "reasoning": "good"}'


def test_a_transient_judge_failure_is_retried() -> None:
    calls = {"n": 0}

    def judge_fn(prompt, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429 Too Many Requests")
        return GOOD_VERDICT

    report = _judge_harness(judge_fn).evaluate(lambda x: "ok", _dataset())
    assert report.score == 1.0
    assert report.n_transient_errors == 0
    assert calls["n"] == 4, "the throttled judge call was not retried"


def test_an_exhausted_transient_judge_failure_is_excluded_from_the_score() -> None:
    """Previously the judge swallowed this into `on_error` and scored it 0.0."""

    def judge_fn(prompt, **kwargs):
        if "THROTTLE" in prompt:
            raise RuntimeError("429 Too Many Requests")
        return GOOD_VERDICT

    dataset = GoldenDataset([Example(inputs=x, expected="ok") for x in ("a", "THROTTLE", "c")])
    report = _judge_harness(judge_fn).evaluate(lambda x: "ok", dataset)
    assert report.score == 1.0, "a throttled judge dragged the candidate's score down"
    assert report.n_transient_errors == 1
    assert report.results[1].transient is True
    assert report.failures() == []


def test_a_judge_that_reliably_fails_is_still_a_real_failure() -> None:
    """Retrying is for the provider's faults, not a broken judge configuration."""
    report = _judge_harness(lambda p, **k: "not json at all").evaluate(lambda x: "ok", _dataset())
    assert report.score == 0.0
    assert report.n_transient_errors == 0
    assert len(report.failures()) == 3


def test_auth_errors_from_the_judge_still_fail_loudly() -> None:
    """A bad key must not be retried into a run of silent zeros."""
    from adapt_agent.optimization.judge import LLMJudge

    class AuthenticationError(Exception):
        pass

    def judge_fn(prompt, **kwargs):
        raise AuthenticationError("bad key")

    with pytest.raises(AuthenticationError):
        LLMJudge(judge_fn, retry=FAST).score("i", "o")


# -- optimizer: an incomplete trial cannot win --------------------------------


def test_an_incomplete_trial_is_not_eligible_to_win() -> None:
    """Dropping throttled rows stops throttling *penalising* a candidate.

    On its own it lets throttling *reward* one instead: a candidate that answers
    one easy row and is throttled on a hard row scores 1.0 over a single row and
    would beat a fully-evaluated 0.9. The score is a mean over a different
    subset, so it is not comparable at all.
    """
    from adapt_agent.optimization.optimizers import Optimizer

    dataset = GoldenDataset(
        [Example(inputs="easy", expected="ok"), Example(inputs="hard", expected="ok")]
    )

    def complete_but_imperfect(inputs):
        return "ok" if inputs == "easy" else "wrong"

    def throttled_on_the_hard_row(inputs):
        if inputs == "hard":
            raise RuntimeError("429 Too Many Requests")
        return "ok"

    harness = EvaluationHarness([exact_match()], retry=RetryPolicy(attempts=1))
    complete = harness.evaluate(complete_but_imperfect, dataset)
    partial = harness.evaluate(throttled_on_the_hard_row, dataset)

    assert complete.is_complete is True and complete.score == 0.5
    assert partial.is_complete is False and partial.score == 1.0

    class _State:
        def __init__(self):
            self.history = []
            self.best_score = 0.0
            self.best_config = {}
            self.best_report = None

    class _Probe(Optimizer):
        strategy_name = "probe"
        min_improvement = 0.0

        def __init__(self):
            self.verbose = False

        def optimize(self, *args, **kwargs):  # pragma: no cover - not exercised
            raise NotImplementedError

    optimizer, state = _Probe(), _State()
    assert optimizer._record(state, {"c": 1}, partial) is False, "a partial trial won"
    assert state.best_score == 0.0
    assert optimizer._record(state, {"c": 2}, complete) is True
    assert state.best_score == 0.5
    assert [(t.score, t.complete, t.accepted) for t in state.history] == [
        (1.0, False, False),
        (0.5, True, True),
    ]


# -- completeness reaches every path that ranks on score ----------------------
#
# Guarding `_record` alone is not enough: it only protects the *global best*.
# An incomplete report reaching any other comparison lets throttling steer the
# search from a different direction.


def _throttle_on(*inputs):
    marked = set(inputs)

    def agent(payload):
        if payload in marked:
            raise RuntimeError("429 Too Many Requests")
        return "ok"

    return agent


def test_an_incomplete_baseline_is_re_run_before_seeding_the_search() -> None:
    """The baseline never passes through `_record`, and everything is measured
    against it. A throttled baseline sets `best_score` over its survivors, so no
    fully-evaluated candidate can beat it and the search returns the starting
    config -- indistinguishable from "nothing improved on your prompt".
    """
    from adapt_agent.optimization.optimizers import Optimizer

    dataset = GoldenDataset(
        [Example(inputs="easy", expected="ok"), Example(inputs="hard", expected="ok")]
    )
    evaluations = {"n": 0}

    class _Harness(EvaluationHarness):
        def evaluate(self, agent, data, **kwargs):
            evaluations["n"] += 1
            # Throttled the first time, healthy on the re-run.
            target = _throttle_on("hard") if evaluations["n"] == 1 else (lambda p: "ok")
            return super().evaluate(target, data, **kwargs)

    class _Probe(Optimizer):
        strategy_name = "probe"

        def __init__(self, harness):
            self.harness = harness
            self.verbose = False

        def optimize(self, *args, **kwargs):  # pragma: no cover - not exercised
            raise NotImplementedError

    harness = _Harness([exact_match()], retry=RetryPolicy(attempts=1))
    report = _Probe(harness)._evaluate_baseline(object(), dataset)

    assert evaluations["n"] == 2, "an incomplete baseline was accepted without a re-run"
    assert report.is_complete is True
    assert report.score == 1.0


def test_a_baseline_that_stays_incomplete_aborts_the_run() -> None:
    """A logged warning is too easy to lose in a long run's output.

    The failure it precedes is a wrong answer wearing the costume of a valid
    one: an inflated baseline is unbeatable, so the search ends reporting the
    starting configuration, which is indistinguishable from "nothing improved".
    """
    from adapt_agent.exceptions import IncompleteEvaluationError
    from adapt_agent.optimization.optimizers import Optimizer

    dataset = GoldenDataset(
        [Example(inputs="easy", expected="ok"), Example(inputs="hard", expected="ok")]
    )

    class _Probe(Optimizer):
        strategy_name = "probe"

        def __init__(self, harness):
            self.harness = harness
            self.verbose = False

        def optimize(self, *args, **kwargs):  # pragma: no cover - not exercised
            raise NotImplementedError

    harness = EvaluationHarness([exact_match()], retry=RetryPolicy(attempts=1))
    with pytest.raises(IncompleteEvaluationError, match="baseline"):
        _Probe(harness)._evaluate_baseline(_throttle_on("hard"), dataset)


def test_an_incomplete_candidate_is_not_bred_from_in_an_evolutionary_search() -> None:
    """`_record` bars it from winning; this bars it from parenting.

    Survivors are chosen from `_score_population`'s list, so leaving an
    incomplete candidate in it lets a score measured over an easier subset
    direct the whole search.
    """
    from adapt_agent.optimization.optimizers import EvolutionaryOptimizer, _SearchState

    dataset = GoldenDataset(
        [Example(inputs="easy", expected="ok"), Example(inputs="hard", expected="ok")]
    )

    class _Harness(EvaluationHarness):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0

        def evaluate(self, agent, data, **kwargs):
            self.calls += 1
            # First config is throttled on the hard row (scores 1.0 over one
            # row); the second is fully evaluated and imperfect (0.5).
            target = (
                _throttle_on("hard")
                if self.calls == 1
                else (lambda p: "ok" if p == "easy" else "wrong")
            )
            return super().evaluate(target, data, **kwargs)

    class _Target:
        """The bare surface `_eval_config` touches."""

        def restore(self, snapshot):
            pass

        def apply(self, config):
            pass

    harness = _Harness([exact_match()], retry=RetryPolicy(attempts=1))
    optimizer = EvolutionaryOptimizer(harness=harness)
    optimizer.verbose = False
    optimizer._baseline_snapshot = {}
    state = _SearchState(
        best_config={}, best_score=0.0, best_report=None, baseline_snapshot={}, history=[]
    )

    scored = optimizer._score_population(
        _Target(), dataset, state, [{"p": "throttled"}, {"p": "complete"}]
    )

    assert [cfg for cfg, _ in scored] == [
        {"p": "complete"}
    ], "the incomplete candidate survived into the breeding pool"
    # It is still recorded in the history, just not ranked.
    assert len(state.history) == 2
    assert [t.complete for t in state.history] == [False, True]


# -- metric retry honours the configured policy -------------------------------


def test_a_provider_backed_custom_metric_is_retried() -> None:
    """Not just `LLMJudge`: any metric may be a network call."""
    from adapt_agent.optimization.metrics import Metric

    calls = {"n": 0}

    def flaky(output, expected):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("429 Too Many Requests")
        return 1.0

    report = EvaluationHarness([Metric("custom", flaky)], retry=FAST).evaluate(
        lambda x: "ok", _dataset()
    )
    assert calls["n"] > 3, "a transient metric failure was classified but never retried"
    assert report.score == 1.0
    assert report.n_transient_errors == 0


def test_a_custom_classifier_governs_metric_failures_too() -> None:
    """`RetryPolicy(is_transient=...)` must not be bypassed for metrics."""
    from adapt_agent.optimization.metrics import Metric

    policy = RetryPolicy(
        attempts=2,
        initial_backoff=0.001,
        jitter=0.0,
        is_transient=lambda exc: "SQUELCH" in str(exc),
    )

    def squelch(output, expected):
        raise RuntimeError("SQUELCH: provider hiccup")

    report = EvaluationHarness([Metric("custom", squelch)], retry=policy).evaluate(
        lambda x: "ok", _dataset()
    )
    assert report.n_transient_errors == 3, "the custom classifier was ignored"

    def plain_429(output, expected):
        raise RuntimeError("429 Too Many Requests")

    # ...and the converse: the default classifier no longer applies.
    narrow = EvaluationHarness([Metric("custom", plain_429)], retry=policy).evaluate(
        lambda x: "ok", _dataset()
    )
    assert narrow.n_transient_errors == 0
    assert len(narrow.failures()) == 3


# -- nested retry budgets must not multiply -----------------------------------


def test_a_judge_and_the_harness_do_not_each_spend_a_retry_budget() -> None:
    """Three attempts at two layers is nine provider calls for one row.

    Worse than wasteful: the backoff resets between the layers, so the load
    lands hardest exactly while the provider is throttling.
    """
    from adapt_agent.optimization.judge import LLMJudge

    calls = {"n": 0}

    def always_throttled(prompt, **kwargs):
        calls["n"] += 1
        raise RuntimeError("429 Too Many Requests")

    policy = RetryPolicy(attempts=3, initial_backoff=0.001, jitter=0.0)
    harness = EvaluationHarness(
        [LLMJudge(always_throttled, retry=policy).as_metric()], retry=policy
    )
    report = harness.evaluate(lambda x: "ok", GoldenDataset([Example(inputs="a", expected="ok")]))

    assert calls["n"] == 3, f"retried at both layers: {calls['n']} provider calls for one row"
    assert report.n_transient_errors == 1
    assert report.failures() == []


def test_a_metric_without_its_own_retries_is_still_retried_by_the_harness() -> None:
    """The marker must not switch retrying off for ordinary metrics."""
    from adapt_agent.optimization.metrics import Metric

    calls = {"n": 0}

    def flaky(output, expected):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("429 Too Many Requests")
        return 1.0

    EvaluationHarness([Metric("custom", flaky)], retry=FAST).evaluate(
        lambda x: "ok", GoldenDataset([Example(inputs="a", expected="ok")])
    )
    assert calls["n"] == 3


def test_a_custom_classifier_reaches_the_judge() -> None:
    """`RetryPolicy(is_transient=...)` governs the judge's own classification.

    Gating on the module-level default meant a custom classifier never reached
    it. Observable two ways, since `score()` keeps its `on_error` fallback: the
    retry count, and whether the metric adapter propagates.
    """
    from adapt_agent.optimization.judge import LLMJudge

    policy = RetryPolicy(
        attempts=2,
        initial_backoff=0.001,
        jitter=0.0,
        is_transient=lambda exc: "SQUELCH" in str(exc),
    )

    calls = {"n": 0}

    def squelch(prompt, **kwargs):
        calls["n"] += 1
        raise RuntimeError("SQUELCH: provider hiccup")

    # Classified transient by the custom policy -> retried, then excluded.
    judge = LLMJudge(squelch, retry=policy)
    report = EvaluationHarness([judge.as_metric()], retry=policy).evaluate(
        lambda x: "ok", GoldenDataset([Example(inputs="a", expected="ok")])
    )
    assert calls["n"] == 2, "the custom classifier never reached the judge"
    assert report.n_transient_errors == 1

    # ...and the converse: a 429 is *not* transient under that policy, so it is
    # tried once and collapses to on_error rather than being excluded.
    calls["n"] = 0

    def plain_429(prompt, **kwargs):
        calls["n"] += 1
        raise RuntimeError("429 Too Many Requests")

    other = LLMJudge(plain_429, retry=policy)
    other_report = EvaluationHarness([other.as_metric()], retry=policy).evaluate(
        lambda x: "ok", GoldenDataset([Example(inputs="a", expected="ok")])
    )
    assert calls["n"] == 1
    assert other_report.n_transient_errors == 0


# -- report denominators ------------------------------------------------------


def test_error_counts_have_a_denominator_that_can_hold_them() -> None:
    """`max_results` bounds *stored records*, not rows run.

    Counting transient errors across the whole dataset while `n` stopped at the
    cap produced impossible summaries -- `n=1, n_transient_errors=4` -- and made
    the optimizer's `n - n_transient_errors` logging go negative.
    """
    dataset = GoldenDataset([Example(inputs=str(i), expected="ok") for i in range(4)])

    def throttled(payload):
        raise RuntimeError("429 Too Many Requests")

    report = EvaluationHarness(
        [exact_match()], retry=RetryPolicy(attempts=1), max_results=1
    ).evaluate(throttled, dataset)

    assert report.n == 1, "the storage cap should still bound the records"
    assert report.n_evaluated == 4
    assert report.n_transient_errors == 4
    assert report.n_scored == 0
    assert report.n_evaluated >= report.n_transient_errors
    assert report.to_dict()["n_evaluated"] == 4


def test_avg_latency_is_per_evaluated_row_not_per_stored_record() -> None:
    dataset = GoldenDataset([Example(inputs=str(i), expected="ok") for i in range(4)])
    report = EvaluationHarness([exact_match()], max_results=1).evaluate(lambda x: "ok", dataset)
    assert report.n == 1
    assert report.n_evaluated == 4
    assert report.avg_latency == pytest.approx(report.total_latency / 4)


# -- transient failures are scoped to what actually failed --------------------


def test_a_throttled_secondary_metric_does_not_erase_the_primary() -> None:
    """Marking the whole row transient deleted a primary score that computed fine.

    With the completeness gate in place that is not just a lost sample: the row
    counts as unscored, so a run graded by `exact_match` plus a secondary judge
    could be rejected -- or abort at the baseline -- because the *judge* was
    throttled.
    """
    from adapt_agent.optimization.metrics import Metric

    def throttled(output, expected):
        raise RuntimeError("429 Too Many Requests")

    harness = EvaluationHarness(
        [exact_match(), Metric("secondary", throttled)], retry=RetryPolicy(attempts=1)
    )
    report = harness.evaluate(lambda x: "ok", _dataset())

    assert report.aggregate["exact_match"] == 1.0
    assert report.score == 1.0
    assert report.n_transient_errors == 0
    assert report.is_complete is True
    assert report.results[0].transient_metrics == ("secondary",)


def test_a_throttled_primary_metric_still_makes_the_row_unusable() -> None:
    """The primary is the number the optimizer ranks on, so a gap there counts."""
    from adapt_agent.optimization.metrics import Metric

    def throttled(output, expected):
        raise RuntimeError("429 Too Many Requests")

    harness = EvaluationHarness(
        [Metric("primary", throttled), exact_match()], retry=RetryPolicy(attempts=1)
    )
    report = harness.evaluate(lambda x: "ok", _dataset())

    assert report.n_transient_errors == 3
    assert report.is_complete is False


def test_a_standalone_judge_keeps_its_documented_fallback() -> None:
    """Propagation is for the metric adapter only.

    `score()`, `critique()` and friends have no harness behind them to catch an
    exception, so turning them into raisers was an unannounced breaking change.
    """
    from adapt_agent.optimization.judge import LLMJudge

    def throttled(prompt, **kwargs):
        raise RuntimeError("429 Too Many Requests")

    judge = LLMJudge(throttled, retry=RetryPolicy(attempts=1), on_error=0.0)
    assert judge.score("i", "o").score == 0.0
    assert judge.critique("i", "o") == ""


def test_the_metric_adapter_still_propagates_so_the_row_is_excluded() -> None:
    from adapt_agent.optimization.judge import LLMJudge

    def throttled(prompt, **kwargs):
        raise RuntimeError("429 Too Many Requests")

    judge = LLMJudge(throttled, retry=RetryPolicy(attempts=1))
    report = EvaluationHarness([judge.as_metric()], retry=RetryPolicy(attempts=1)).evaluate(
        lambda x: "ok", _dataset()
    )
    assert report.n_transient_errors == 3, "the metric adapter stopped propagating"
    assert report.failures() == []


# -- partial aggregates and validation are visible, not silent ----------------


def test_a_partial_secondary_aggregate_is_visible() -> None:
    """`is_complete` speaks for the primary, so a secondary needs its own signal.

    Scoping transient failures to the failing metric stopped a throttled
    secondary erasing the primary -- but left its mean computed over whichever
    rows survived, in a report claiming `is_complete=True` and zero transient
    errors.
    """
    from adapt_agent.optimization.metrics import Metric

    dataset = GoldenDataset([Example(inputs=str(i), expected="ok") for i in range(4)])
    calls = {"n": 0}

    def flaky(output, expected):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("429 Too Many Requests")
        return 1.0

    report = EvaluationHarness(
        [exact_match(), Metric("secondary", flaky)], retry=RetryPolicy(attempts=1)
    ).evaluate(lambda x: "ok", dataset)

    assert report.is_complete is True, "the primary scored fine"
    assert report.n_evaluated == 4
    assert report.metric_samples == {"exact_match": 4, "secondary": 2}
    assert report.transient_by_metric == {"secondary": 2}
    assert report.partial_metrics == ["secondary"]
    assert report.to_dict()["partial_metrics"] == ["secondary"]


def test_nothing_is_partial_when_no_metric_was_throttled() -> None:
    dataset = GoldenDataset([Example(inputs=str(i), expected="ok") for i in range(3)])
    report = EvaluationHarness([exact_match()]).evaluate(lambda x: "ok", dataset)
    assert report.partial_metrics == []
    assert report.metric_samples == {"exact_match": 3}


def test_an_incomplete_validation_pass_is_reported_not_hidden() -> None:
    """Validation does not steer the search, so it re-runs but never aborts.

    It must not be reported as if it were whole, though: it is the number a
    user reads to decide whether the tuned config generalises.
    """
    from adapt_agent.optimization.optimizers import Optimizer

    dataset = GoldenDataset(
        [Example(inputs="easy", expected="ok"), Example(inputs="hard", expected="ok")]
    )

    class _Probe(Optimizer):
        strategy_name = "probe"

        def __init__(self, harness):
            self.harness = harness
            self.verbose = False

        def optimize(self, *args, **kwargs):  # pragma: no cover - not exercised
            raise NotImplementedError

    harness = EvaluationHarness([exact_match()], retry=RetryPolicy(attempts=1))
    report = _Probe(harness)._evaluate_validation(_throttle_on("hard"), dataset)

    assert report.is_complete is False, "a partial validation is still returned"
    assert report.n_transient_errors == 1


def test_a_validation_pass_that_recovers_on_the_re_run_is_complete() -> None:
    from adapt_agent.optimization.optimizers import Optimizer

    dataset = GoldenDataset(
        [Example(inputs="easy", expected="ok"), Example(inputs="hard", expected="ok")]
    )
    evaluations = {"n": 0}

    class _Harness(EvaluationHarness):
        def evaluate(self, agent, data, **kwargs):
            evaluations["n"] += 1
            target = _throttle_on("hard") if evaluations["n"] == 1 else (lambda p: "ok")
            return super().evaluate(target, data, **kwargs)

    class _Probe(Optimizer):
        strategy_name = "probe"

        def __init__(self, harness):
            self.harness = harness
            self.verbose = False

        def optimize(self, *args, **kwargs):  # pragma: no cover - not exercised
            raise NotImplementedError

    harness = _Harness([exact_match()], retry=RetryPolicy(attempts=1))
    report = _Probe(harness)._evaluate_validation(object(), dataset)

    assert evaluations["n"] == 2
    assert report.is_complete is True


def _tunable_agent(throttle_on: set[str], *, throttle_times: int | None = None):
    """A minimal optimizable agent that throttles on the named inputs.

    One prompt knob with two candidates, so the search has something to do and
    `optimize()` runs end to end. `throttle_times` bounds how many times each
    named input throttles before it starts succeeding -- that is what separates
    a re-running validation path from a single-shot one.
    """
    from adapt_agent.optimization.parameters import Parameter, ParameterKind
    from adapt_agent.optimization.target import wrap

    state = {"prompt": "BAD"}
    thrown: dict[str, int] = {}

    def runner(question: str) -> str:
        if question in throttle_on:
            seen = thrown.get(question, 0)
            if throttle_times is None or seen < throttle_times:
                thrown[question] = seen + 1
                raise RuntimeError("429 Too Many Requests")
        return "ok" if state["prompt"] == "GOOD" else "wrong"

    parameter = Parameter(
        name="prompt",
        kind=ParameterKind.PROMPT,
        candidates=["BAD", "GOOD"],
        getter=lambda: state["prompt"],
        setter=lambda v: state.__setitem__("prompt", v),
    )
    return wrap(runner, runner=runner, parameters=[parameter])


def _build_optimizer(name: str, harness):
    """`GridSearchOptimizer` directly, and the same wrapped in a pipeline.

    Both override `optimize()`, so both need their own validation wiring.
    """
    import adapt_agent.optimization.optimizers as optimizers

    grid = optimizers.GridSearchOptimizer(harness, seed=0)
    if name == "GridSearchOptimizer":
        return grid
    return optimizers.PipelineOptimizer(harness, [grid])


@pytest.mark.parametrize("optimizer_name", ["GridSearchOptimizer", "PipelineOptimizer"])
def test_optimize_surfaces_an_incomplete_validation_end_to_end(optimizer_name: str) -> None:
    """The wiring, not just the helper.

    `_evaluate_validation` had its own tests while both `optimize()` methods
    still called `harness.evaluate` directly -- so the helper was correct and
    unreachable, which is the shape of the original finding. This drives the
    public entry point and reads the flag off the result.
    """
    train = GoldenDataset([Example(inputs="q0", expected="ok")])
    val = GoldenDataset(
        [Example(inputs="v_ok", expected="ok"), Example(inputs="v_throttled", expected="ok")]
    )
    agent = _tunable_agent({"v_throttled"})
    harness = EvaluationHarness([exact_match()], retry=RetryPolicy(attempts=1))

    result = _build_optimizer(optimizer_name, harness).optimize(agent, train, val_dataset=val)

    assert result.validation_complete is False, "a partial validation reported as whole"
    assert result.validation_score is not None, "the partial score is still returned"


@pytest.mark.parametrize("optimizer_name", ["GridSearchOptimizer", "PipelineOptimizer"])
def test_optimize_reports_a_clean_validation_as_complete(optimizer_name: str) -> None:
    train = GoldenDataset([Example(inputs="v_ok", expected="ok")])
    val = GoldenDataset([Example(inputs="v_ok", expected="ok")])
    agent = _tunable_agent(set())
    harness = EvaluationHarness([exact_match()], retry=RetryPolicy(attempts=1))

    result = _build_optimizer(optimizer_name, harness).optimize(agent, train, val_dataset=val)

    assert result.validation_complete is True


@pytest.mark.parametrize("optimizer_name", ["GridSearchOptimizer", "PipelineOptimizer"])
def test_optimize_re_runs_a_throttled_validation_pass(optimizer_name: str) -> None:
    """The assertion that separates the two paths.

    A permanently throttled hold-out reports `validation_complete=False` whether
    `optimize()` calls `_evaluate_validation` or `harness.evaluate` directly, so
    that alone does not pin the wiring. Only the re-run does: this row fails once
    and then succeeds, so a single-shot call reports it partial forever.
    """
    train = GoldenDataset([Example(inputs="v_ok", expected="ok")])
    val = GoldenDataset(
        [Example(inputs="v_ok", expected="ok"), Example(inputs="v_flaky", expected="ok")]
    )
    # `attempts=1` means no in-harness retry, so recovery can only come from the
    # optimizer re-running the whole pass.
    agent = _tunable_agent({"v_flaky"}, throttle_times=1)
    harness = EvaluationHarness([exact_match()], retry=RetryPolicy(attempts=1))

    result = _build_optimizer(optimizer_name, harness).optimize(agent, train, val_dataset=val)

    assert result.validation_complete is True, "validation did not go through _evaluate_validation"


# -- exclusion is per metric in *both* directions ------------------------------


def test_a_throttled_primary_does_not_discard_a_good_secondary() -> None:
    """The mirror of the round-five fix, and the same mistake in reverse.

    Scoping transient failures to the failing metric stopped a throttled
    *secondary* erasing the primary. The whole-row branch still discarded every
    metric when the *primary* was the one throttled -- so a secondary that
    measured all four rows perfectly reported a mean of 0.0 over no samples.
    An unearned zero is not a measurement; neither is discarding an earned one.
    """
    from adapt_agent.optimization.metrics import Metric

    def throttled(output, expected):
        raise RuntimeError("429 Too Many Requests")

    dataset = GoldenDataset([Example(inputs=str(i), expected="ok") for i in range(4)])
    report = EvaluationHarness(
        [Metric("primary", throttled), exact_match()], retry=RetryPolicy(attempts=1)
    ).evaluate(lambda x: "ok", dataset)

    assert report.aggregate["exact_match"] == 1.0, "a valid measurement was thrown away"
    assert report.metric_samples == {"exact_match": 4}
    assert report.transient_by_metric == {"primary": 4}
    assert report.partial_metrics == ["primary"]
    # The primary is still unmeasurable, so the run is still not comparable.
    assert report.is_complete is False
    assert report.n_transient_errors == 4


def test_a_throttled_agent_call_loses_every_metric() -> None:
    """No output means nothing for any metric to measure.

    `transient_metrics` is the single source of truth for per-metric exclusion,
    so a whole-row failure has to name them all rather than leave the
    accumulator inferring it from the row flag.
    """
    from adapt_agent.optimization.metrics import Metric

    def throttled_agent(payload):
        raise RuntimeError("429 Too Many Requests")

    dataset = GoldenDataset([Example(inputs=str(i), expected="ok") for i in range(4)])
    report = EvaluationHarness(
        [exact_match(), Metric("secondary", lambda o, e: 1.0)], retry=RetryPolicy(attempts=1)
    ).evaluate(throttled_agent, dataset)

    assert report.metric_samples == {}, "a metric was credited for a row that never ran"
    assert report.transient_by_metric == {"exact_match": 4, "secondary": 4}
    assert report.is_complete is False


def test_a_permanent_agent_failure_still_earns_its_zeros() -> None:
    """The exclusion is for *transient* failures only -- a broken agent scores 0."""

    def broken(payload):
        raise ValueError("bad prompt")

    dataset = GoldenDataset([Example(inputs=str(i), expected="ok") for i in range(4)])
    report = EvaluationHarness([exact_match()], retry=RetryPolicy(attempts=1)).evaluate(
        broken, dataset
    )

    assert report.aggregate["exact_match"] == 0.0
    assert report.metric_samples == {"exact_match": 4}
    assert report.n_errors == 4
    assert report.n_transient_errors == 0
    assert report.is_complete is True
    assert len(report.failures()) == 4


def test_validation_completeness_survives_serialisation() -> None:
    """`validation_complete` exists to be read later, so it has to be written down.

    A persisted result or a provenance header outlives the run's logs; a partial
    validation score that serialises identically to a whole one is back to being
    invisible.
    """
    from adapt_agent.optimization.optimizers import OptimizationResult

    partial = OptimizationResult(
        best_config={"p": "x"},
        best_score=1.0,
        baseline_score=0.0,
        baseline_config={},
        history=[],
        validation_score=0.5,
        validation_complete=False,
    )
    assert partial.to_dict()["validation_complete"] is False
    assert partial._provenance()["validation_complete"] is False

    whole = OptimizationResult(
        best_config={},
        best_score=1.0,
        baseline_score=0.0,
        baseline_config={},
        history=[],
        validation_score=0.5,
    )
    assert whole.to_dict()["validation_complete"] is True
    assert whole._provenance()["validation_complete"] is True
