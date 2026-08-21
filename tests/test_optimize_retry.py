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
