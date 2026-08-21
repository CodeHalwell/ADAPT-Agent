"""Tests for the async execution path and the harness concurrency knob.

The point of ``aexecute`` is not merely "it also works with await" -- it is that
the two properties a worker thread destroys, *real concurrency* and *contextvar
propagation*, survive. Both are asserted here rather than assumed.
"""

from __future__ import annotations

import asyncio
import contextvars
import gc
import threading
import time

import pytest

from adapt_agent.adapters import LangGraphAdapter, MicrosoftAgentFrameworkAdapter
from adapt_agent.exceptions import AdapterError, SecurityBlockedError
from adapt_agent.optimization.dataset import Example, GoldenDataset
from adapt_agent.optimization.evaluation import EvaluationHarness, aresolve_runner
from adapt_agent.optimization.metrics import exact_match
from adapt_agent.security import Firewall

TRACE: contextvars.ContextVar[str] = contextvars.ContextVar("trace", default="none")


def _payload(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


class AsyncAgent:
    """An async-native agent, like a MAF ChatAgent or a Pydantic AI Agent."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.seen_trace: list[str] = []

    async def run(self, prompt):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.seen_trace.append(TRACE.get())
        return f"echo:{prompt}"


class SyncAgent:
    def __init__(self, delay: float = 0.0):
        self.delay = delay

    def invoke(self, payload):
        if self.delay:
            time.sleep(self.delay)
        return {"messages": [{"role": "assistant", "content": "ok"}]}


def _guarded(agent, **kwargs):
    return MicrosoftAgentFrameworkAdapter(**kwargs).wrap_agent(agent)


# -- aexecute ------------------------------------------------------------------


def test_aexecute_runs_inside_a_running_loop_where_execute_cannot():
    guarded = _guarded(AsyncAgent())

    async def main():
        result = await guarded.aexecute(_payload("hello"))
        with pytest.raises(AdapterError, match="aexecute"):
            guarded.execute(_payload("hello"))
        return result

    assert asyncio.run(main()) == {"result": "echo:hello"}


def test_execute_still_works_for_sync_callers():
    """The sync path must be untouched by the async addition."""
    guarded = _guarded(AsyncAgent())
    assert guarded.execute(_payload("hi")) == {"result": "echo:hi"}


def test_a_coroutine_returning_a_stream_is_drained():
    """An async run method is often a coroutine that *returns* a stream.

    Stopping at the first await handed the live generator to output screening,
    which then found no text to screen, and to the caller, which got a generator
    where the envelope documents a materialised list.
    """

    class Agent:
        def invoke(self, payload):
            raise AssertionError("aexecute must not use the sync entry point")

        async def ainvoke(self, payload):
            async def stream():
                yield "part one"
                yield "part two"

            return stream()

    guarded = LangGraphAdapter().wrap_agent(Agent())
    assert asyncio.run(guarded.aexecute({"messages": []}))["result"] == ["part one", "part two"]


def test_aexecute_does_not_block_the_loop_on_a_sync_agent():
    """A framework with no async entry point still reaches `aexecute`.

    Calling the sync runner inline did the work *before* the first await, so the
    whole loop stalled -- measured at zero heartbeat ticks and three concurrent
    calls serialising. That defeats the concurrency this entry point exists for.
    """
    delay = 0.05

    class SyncAgent:
        def invoke(self, payload):
            time.sleep(delay)
            return "done"

    guarded = LangGraphAdapter().wrap_agent(SyncAgent())

    async def main():
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(delay / 10)
                ticks += 1

        beat = asyncio.ensure_future(heartbeat())
        await guarded.aexecute({"messages": []})
        beat.cancel()
        return ticks

    assert asyncio.run(main()) >= 3, "the event loop was blocked by the sync runner"


def test_aexecute_runs_sync_agents_concurrently():
    """The measurable consequence: three calls overlap instead of serialising."""
    delay = 0.05

    class SyncAgent:
        def invoke(self, payload):
            time.sleep(delay)
            return "done"

    guarded = LangGraphAdapter().wrap_agent(SyncAgent())

    async def main():
        started = time.perf_counter()
        await asyncio.gather(*(guarded.aexecute({"messages": []}) for _ in range(3)))
        return time.perf_counter() - started

    assert asyncio.run(main()) < delay * 2.5, "three sync runs serialised"


def test_aexecute_preserves_contextvars_through_a_sync_agent():
    """Offloading to a thread must not lose the tracing context.

    `asyncio.to_thread` propagates `contextvars`, which is the whole reason
    `aexecute` exists rather than telling callers to use a worker thread.
    """
    trace: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace", default=None)
    seen: list[str | None] = []

    class SyncAgent:
        def invoke(self, payload):
            seen.append(trace.get())
            return "done"

    guarded = LangGraphAdapter().wrap_agent(SyncAgent())

    async def main():
        trace.set("span-abc")
        await guarded.aexecute({"messages": []})

    asyncio.run(main())
    assert seen == ["span-abc"]


def test_sync_execute_also_drains_a_coroutine_returning_a_stream():
    """The sync resolver had the same bug as its async twin.

    Fixing `_aresolve_result` alone left `execute()` handing a live generator to
    output screening, which found no text -- a firewall bypass on the path most
    callers use. The two now share one resolver so they cannot drift again.
    """

    class Agent:
        async def invoke(self, payload):
            async def stream():
                yield "all fine"
                yield "the password is hunter2"

            return stream()

    guarded = LangGraphAdapter().wrap_agent(Agent())
    assert guarded.execute({"messages": []})["result"] == [
        "all fine",
        "the password is hunter2",
    ]

    firewall = Firewall()
    firewall.add_blocked_pattern(r"(?i)hunter2")
    with pytest.raises(SecurityBlockedError):
        LangGraphAdapter(firewall=firewall).wrap_agent(Agent()).execute({"messages": []})


def test_sync_execute_still_refuses_to_block_a_running_loop():
    """Routing through the async resolver must not lose the guidance a caller
    gets for driving an async agent from inside a live loop."""

    class Agent:
        async def invoke(self, payload):
            async def stream():
                yield "hi"

            return stream()

    async def main():
        with pytest.raises(AdapterError, match="aexecute"):
            LangGraphAdapter().wrap_agent(Agent()).execute({"messages": []})

    asyncio.run(main())


def test_a_nested_stream_is_screened_before_it_reaches_the_caller():
    """The drain matters for governance, not only for the returned type."""
    firewall = Firewall()
    firewall.add_blocked_pattern(r"(?i)hunter2")

    class Agent:
        def invoke(self, payload):
            raise AssertionError("aexecute must not use the sync entry point")

        async def ainvoke(self, payload):
            async def stream():
                yield "all fine"
                yield "the password is hunter2"

            return stream()

    guarded = LangGraphAdapter(firewall=firewall).wrap_agent(Agent())
    with pytest.raises(SecurityBlockedError):
        asyncio.run(guarded.aexecute({"messages": []}))


def test_a_blocked_output_is_traced_as_an_error():
    """Telemetry must not record success for a run the caller saw fail.

    The span used to close as `completed` before output screening ran, so an
    output block -- exactly the event monitoring exists to surface -- was
    invisible in the trace.
    """

    class Observer:
        def __init__(self):
            self.events = []

        def start_trace(self, trace_id, agent_id, operation):
            self.events.append("start")

        def end_trace(self, trace_id, status="completed", result=None):
            self.events.append(f"end:{status}")

    firewall = Firewall()
    firewall.add_blocked_pattern(r"(?i)hunter2")

    class Graph:
        def invoke(self, payload):
            return "the password is hunter2"

    observer = Observer()
    guarded = LangGraphAdapter(firewall=firewall, observer=observer).wrap_agent(Graph())
    with pytest.raises(SecurityBlockedError):
        guarded.execute({"messages": []})
    assert observer.events == ["start", "end:error"]

    # A clean run still closes as completed.
    class Clean:
        def invoke(self, payload):
            return "all fine"

    observer = Observer()
    LangGraphAdapter(firewall=firewall, observer=observer).wrap_agent(Clean()).execute(
        {"messages": []}
    )
    assert observer.events == ["start", "end:completed"]


def test_a_blocked_output_is_traced_as_an_error_on_the_async_path():
    class Observer:
        def __init__(self):
            self.events = []

        def start_trace(self, trace_id, agent_id, operation):
            self.events.append("start")

        def end_trace(self, trace_id, status="completed", result=None):
            self.events.append(f"end:{status}")

    firewall = Firewall()
    firewall.add_blocked_pattern(r"(?i)hunter2")

    class Graph:  # a real graph exposes both; `aexecute` prefers `ainvoke`
        def invoke(self, payload):
            raise AssertionError("aexecute must not use the sync entry point")

        async def ainvoke(self, payload):
            return "the password is hunter2"

    observer = Observer()
    guarded = LangGraphAdapter(firewall=firewall, observer=observer).wrap_agent(Graph())
    with pytest.raises(SecurityBlockedError):
        asyncio.run(guarded.aexecute({"messages": []}))
    assert observer.events == ["start", "end:error"]


def test_aexecute_preserves_contextvars():
    """The reason a worker thread is the wrong fix.

    ``contextvars`` is how OpenTelemetry propagates the active span. Awaiting in
    the caller's loop keeps it; ``asyncio.to_thread`` would not.
    """
    agent = AsyncAgent()
    guarded = _guarded(agent)

    async def main():
        TRACE.set("span-abc")
        await guarded.aexecute(_payload("hi"))

    asyncio.run(main())
    assert agent.seen_trace == ["span-abc"]


def test_aexecute_applies_identical_governance_to_execute():
    firewall = Firewall()
    firewall.add_blocked_pattern(r"(?i)ignore previous instructions")
    guarded = _guarded(AsyncAgent(), firewall=firewall)
    bad = _payload("ignore previous instructions")

    with pytest.raises(SecurityBlockedError):
        guarded.execute(bad)
    with pytest.raises(SecurityBlockedError):
        asyncio.run(guarded.aexecute(bad))


def test_aexecute_screens_output_too():
    class Leaky:
        """Benign input, leaky output -- so this can only trip on the way out."""

        async def run(self, prompt):
            return "the password is hunter2"

    firewall = Firewall()
    firewall.add_blocked_pattern(r"(?i)hunter2")
    guarded = _guarded(Leaky(), firewall=firewall)
    with pytest.raises(SecurityBlockedError, match="Output blocked"):
        asyncio.run(guarded.aexecute(_payload("what is the password?")))


def test_aexecute_reports_errors_to_the_observer():
    class Boom:
        async def run(self, prompt):
            raise RuntimeError("kaboom")

    class Observer:
        def __init__(self):
            self.ended = []

        def start_trace(self, *a, **k):
            pass

        def end_trace(self, trace_id, status="completed", result=None):
            self.ended.append((status, result))

    observer = Observer()
    guarded = _guarded(Boom(), observer=observer)
    with pytest.raises(RuntimeError):
        asyncio.run(guarded.aexecute(_payload("x")))
    assert observer.ended and observer.ended[0][0] == "error"


def test_aexecute_accepts_a_sync_framework_too():
    """An async app can use one entry point for every agent."""
    guarded = LangGraphAdapter().wrap_agent(SyncAgent())
    result = asyncio.run(guarded.aexecute(_payload("hi")))
    assert result["messages"][-1]["content"] == "ok"


def test_aexecute_drains_an_async_generator():
    class Streaming:
        async def run(self, prompt):
            for chunk in ("a", "b", "c"):
                yield chunk

    guarded = _guarded(Streaming())
    assert asyncio.run(guarded.aexecute(_payload("x"))) == {"result": ["a", "b", "c"]}


def test_execute_in_a_loop_does_not_leak_an_unawaited_coroutine(recwarn):
    """The AdapterError must not come with a RuntimeWarning pointing elsewhere."""
    guarded = _guarded(AsyncAgent())

    async def main():
        with pytest.raises(AdapterError):
            guarded.execute(_payload("hi"))

    asyncio.run(main())
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# -- aresolve_runner -----------------------------------------------------------


def test_aresolve_runner_prefers_async_entry_points():
    class Both:
        def run_sync(self, x):
            return "sync"

        async def run(self, x):
            return "async"

    assert asyncio.run(aresolve_runner(Both())("q")) == "async"


def test_aresolve_runner_prefers_aexecute_on_a_governed_agent():
    guarded = _guarded(AsyncAgent())
    assert asyncio.run(aresolve_runner(guarded)(_payload("hi"))) == {"result": "echo:hi"}


def test_aresolve_runner_rejects_non_runnables():
    with pytest.raises(TypeError, match="Cannot evaluate object"):
        aresolve_runner(object())


# -- harness concurrency -------------------------------------------------------


def _dataset(n: int) -> GoldenDataset:
    return GoldenDataset.from_list([{"input": f"q{i}", "expected": f"a{i}"} for i in range(n)])


class SlowAsyncScorer:
    def __init__(self, delay: float = 0.02, fail_on: str | None = None):
        self.delay, self.fail_on = delay, fail_on
        self.in_flight = 0
        self.peak = 0

    async def run(self, x):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            if x == self.fail_on:
                raise RuntimeError("boom")
            return "a" + x[1:]
        finally:
            self.in_flight -= 1


def test_aevaluate_runs_examples_concurrently():
    agent = SlowAsyncScorer()
    harness = EvaluationHarness([exact_match()])
    report = asyncio.run(harness.aevaluate(agent, _dataset(12), concurrency=6))
    assert report.score == 1.0
    assert agent.peak > 1, "examples did not overlap"
    assert agent.peak <= 6, "semaphore did not bound concurrency"


def test_aevaluate_serial_by_default():
    agent = SlowAsyncScorer()
    harness = EvaluationHarness([exact_match()])
    asyncio.run(harness.aevaluate(agent, _dataset(5)))
    assert agent.peak == 1


def test_aevaluate_reports_in_index_order_not_completion_order():
    class Jittered:
        async def run(self, x):
            # Later examples finish first, so completion order != index order.
            await asyncio.sleep(0.02 - 0.002 * int(x[1:]))
            return "a" + x[1:]

    harness = EvaluationHarness([exact_match()])
    report = asyncio.run(harness.aevaluate(Jittered(), _dataset(8), concurrency=8))
    assert [r.index for r in report.results] == list(range(8))
    assert report.score == 1.0


def test_aevaluate_keeps_per_example_errors_non_fatal():
    harness = EvaluationHarness([exact_match()])
    report = asyncio.run(
        harness.aevaluate(SlowAsyncScorer(fail_on="q3"), _dataset(6), concurrency=3)
    )
    assert report.n_errors == 1
    assert report.n == 6
    assert [r.index for r in report.results if r.error] == [3]


def test_aevaluate_respects_max_results_while_aggregating_everything():
    harness = EvaluationHarness([exact_match()], max_results=3)
    report = asyncio.run(harness.aevaluate(SlowAsyncScorer(), _dataset(10), concurrency=5))
    assert [r.index for r in report.results] == [0, 1, 2]
    assert report.aggregate["exact_match"] == 1.0  # all ten were scored


def test_aevaluate_handles_an_empty_dataset():
    harness = EvaluationHarness([exact_match()])
    report = asyncio.run(harness.aevaluate(SlowAsyncScorer(), _dataset(0), concurrency=4))
    assert report.n == 0
    assert report.aggregate["exact_match"] == 0.0


def test_evaluate_with_threads_matches_serial_results():
    """The threaded path must agree with the serial one, ordering included."""

    class Sync:
        def invoke(self, x):
            time.sleep(0.005)
            return "a" + x[1:]

    harness = EvaluationHarness([exact_match()])
    serial = harness.evaluate(Sync(), _dataset(10))
    threaded = harness.evaluate(Sync(), _dataset(10), concurrency=5)
    assert serial.aggregate == threaded.aggregate
    assert [r.index for r in threaded.results] == list(range(10))


def test_evaluate_threads_actually_overlap():
    class Sync:
        def __init__(self):
            self.in_flight = 0
            self.peak = 0

        def invoke(self, x):
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            time.sleep(0.02)
            self.in_flight -= 1
            return "a" + x[1:]

    agent = Sync()
    EvaluationHarness([exact_match()]).evaluate(agent, _dataset(8), concurrency=4)
    assert agent.peak > 1


def test_avg_latency_is_per_example_not_wall_clock_over_n():
    """Concurrency must not deflate the reported latency.

    ``total_latency`` sums per-example durations, so ``avg_latency`` stays the
    mean time one example took -- comparable across concurrency settings, which
    wall-clock/n would not be.
    """
    delay = 0.02
    harness = EvaluationHarness([exact_match()])
    report = asyncio.run(
        harness.aevaluate(SlowAsyncScorer(delay=delay), _dataset(8), concurrency=8)
    )
    # The discriminating comparison is against wall-clock/n, which for eight
    # overlapped examples would be about `delay / 8`. A tight two-sided window
    # round-trips that meaning into a load-sensitive assertion -- a busy machine
    # stretches each sleep -- so bound it where the meaning actually lives: far
    # above wall-clock/n, and not absurdly above `delay`.
    assert report.avg_latency > delay * 0.8, "avg_latency looks like wall-clock/n"
    assert report.avg_latency < delay * 5
    # Eight examples overlapped, so the summed latency far exceeds wall clock.
    assert report.total_latency > delay * 4


def test_execute_in_a_loop_does_not_leak_an_async_generator(recwarn):
    """An async generator has no sync ``close()`` -- assert none is needed.

    ``_resolve_result`` raises before iterating, so the generator is never
    started and holds no suspended frame to finalize.
    """

    class Streaming:
        async def run(self, prompt):
            yield "a"
            yield "b"

    guarded = _guarded(Streaming())

    async def main():
        with pytest.raises(AdapterError):
            guarded.execute(_payload("hi"))

    asyncio.run(main())
    gc.collect()
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_threaded_evaluate_consumes_the_dataset_lazily():
    """Bounded submission: at most ``concurrency`` examples are pulled ahead.

    ``ThreadPoolExecutor.map`` would consume the whole iterable up front, so
    this guards the memory-bounding property rather than assuming it.
    """
    pulled = []

    class LazyDataset:
        def __iter__(self):
            for i in range(40):
                pulled.append(i)
                yield Example(inputs=f"q{i}", expected=f"a{i}")

    pulled_when_first_ran = []

    class Slow:
        def invoke(self, x):
            # How much of the dataset had been consumed by the time the first
            # example started? Eager submission would already have drained it.
            pulled_when_first_ran.append(len(pulled))
            time.sleep(0.01)
            return "a" + x[1:]

    harness = EvaluationHarness([exact_match()])
    report = harness.evaluate(Slow(), LazyDataset(), concurrency=4)
    assert report.score == 1.0
    assert [r.index for r in report.results] == list(range(40))
    assert len(pulled) == 40, "every example must eventually run"
    # The real guard: only a bounded look-ahead had been pulled when work began.
    # `ThreadPoolExecutor.map` would report 40 here.
    assert pulled_when_first_ran[0] <= 8, (
        f"{pulled_when_first_ran[0]} examples were consumed before the first "
        "one ran -- the dataset is being materialised eagerly"
    )


def test_threaded_evaluate_bounds_in_flight_work():
    class Counting:
        def __init__(self):
            self.in_flight = 0
            self.peak = 0
            self._lock = threading.Lock()

        def invoke(self, x):
            with self._lock:
                self.in_flight += 1
                self.peak = max(self.peak, self.in_flight)
            time.sleep(0.01)
            with self._lock:
                self.in_flight -= 1
            return "a" + x[1:]

    agent = Counting()
    EvaluationHarness([exact_match()]).evaluate(agent, _dataset(30), concurrency=4)
    assert 1 < agent.peak <= 4


def test_aexecute_prefers_the_frameworks_async_entry_point():
    """`aexecute` must not call the sync twin -- that blocks the loop it shares.

    LangGraph, CrewAI, Pydantic AI and the OpenAI Agents SDK all expose both
    styles, and the adapters' `run_method_names` are sync-first by design.
    """
    from adapt_agent.adapters import CrewAIAdapter, PydanticAIAdapter

    calls = []

    class Graph:
        def invoke(self, x):
            calls.append("invoke")
            return {"messages": []}

        async def ainvoke(self, x):
            calls.append("ainvoke")
            return {"messages": []}

    guarded = LangGraphAdapter().wrap_agent(Graph())
    asyncio.run(guarded.aexecute({"messages": []}))
    assert calls == ["ainvoke"]

    calls.clear()
    guarded.execute({"messages": []})
    assert calls == ["invoke"], "the sync path must be unchanged"

    calls.clear()

    class Crew:
        def kickoff(self, x):
            calls.append("kickoff")
            return "r"

        async def kickoff_async(self, x):
            calls.append("kickoff_async")
            return "r"

    asyncio.run(CrewAIAdapter().wrap_agent(Crew()).aexecute({"messages": []}))
    assert calls == ["kickoff_async"]

    calls.clear()

    class PydanticAgent:
        def run_sync(self, prompt):
            calls.append("run_sync")
            return "r"

        async def run(self, prompt):
            calls.append("run")
            return "r"

    asyncio.run(PydanticAIAdapter().wrap_agent(PydanticAgent()).aexecute(_payload("hi")))
    assert calls == ["run"]


def test_aexecute_falls_back_to_the_sync_runner_when_there_is_no_async_one():
    calls = []

    class SyncOnly:
        def invoke(self, x):
            calls.append("invoke")
            return {"messages": []}

    asyncio.run(LangGraphAdapter().wrap_agent(SyncOnly()).aexecute({"messages": []}))
    assert calls == ["invoke"]


def test_cancelling_aexecute_closes_the_observer_span():
    """`CancelledError` derives from BaseException, so `except Exception` misses
    it and would leave the span open for the life of the process."""

    class Observer:
        def __init__(self):
            self.events = []

        def start_trace(self, trace_id, agent_id, operation):
            self.events.append("start")

        def end_trace(self, trace_id, status="completed", result=None):
            self.events.append(f"end:{status}")

    class Slow:
        async def run(self, prompt):
            await asyncio.sleep(10)

    observer = Observer()
    guarded = _guarded(Slow(), observer=observer)

    async def main():
        task = asyncio.ensure_future(guarded.aexecute(_payload("hi")))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert observer.events == ["start", "end:error"]


def test_threaded_pool_refills_from_any_completed_future():
    """One slow example must not idle the rest of the pool behind it.

    Waiting on the *oldest* future blocks refilling even after the other
    `concurrency - 1` finish, which collapses the achieved concurrency exactly
    when latency is variable -- i.e. on every real LLM workload.
    """

    class SlowFirst:
        def invoke(self, x):
            time.sleep(0.30 if x == "q0" else 0.02)
            return "a" + x[1:]

    harness = EvaluationHarness([exact_match()])
    started = time.perf_counter()
    report = harness.evaluate(SlowFirst(), _dataset(24), concurrency=4)
    elapsed = time.perf_counter() - started

    assert report.score == 1.0
    assert [r.index for r in report.results] == list(range(24)), "ordering must survive"
    # With refill the run is bounded by the slow example (~0.30s) plus the rest
    # spread over the other workers; head-of-line blocking pushes it past 0.45s.
    assert elapsed < 0.45, f"pool stalled behind the slow example ({elapsed:.2f}s)"


def test_openai_agents_direct_wrap_gets_an_async_sdk_runner():
    """An SDK `Agent` exposes neither run nor run_sync, so the async preference
    list cannot match it -- without an override `aexecute` falls back to the
    synchronous `Runner.run_sync` lambda and blocks the loop."""
    from adapt_agent.adapters import OpenAIAgentsAdapter

    class SDKAgent:
        name = "triage"

    guarded = OpenAIAgentsAdapter().wrap_agent(SDKAgent())
    assert guarded._arunner is not guarded._runner
