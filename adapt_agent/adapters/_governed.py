"""Shared governance machinery for ADAPT-Agent framework adapters.

This module factors out the framework-agnostic parts of an adapter so that each
concrete adapter (LangGraph, CrewAI, Pydantic AI, ...) only has to declare a few
small details: which method runs the underlying agent, how to shape the input,
and a human-readable framework name.

The :class:`GovernedAdapter` applies, in order, on every ``execute`` call:

1. **Input screening** -- :class:`~adapt_agent.security.Firewall` and
   :class:`~adapt_agent.adversarial.AdversarialDefense` scan every string found
   in the payload.
2. **Policy enforcement** -- a :class:`~adapt_agent.core.PolicyEnforcer` is
   evaluated against the extracted :class:`~adapt_agent.core.types.AgentState`.
3. **Pre-middleware** -- a :class:`~adapt_agent.core.Middleware` pipeline may
   rewrite the payload.
4. **Traced execution** -- the framework agent runs, optionally traced by an
   :class:`~adapt_agent.observability.AgentObserver`.
5. **Post-middleware** -- the pipeline may rewrite the result.
6. **Output screening** -- the firewall scans every string in the result.

The design is intentionally *structural*: importing an adapter never imports the
underlying framework. The framework is only needed at runtime to build the agent
you pass to :meth:`wrap_agent`. Adapters wrap anything that exposes a recognized
"run" method -- or any plain callable -- so they work without the optional
dependency installed and are trivial to unit-test.

Async support
-------------
Several modern frameworks (Pydantic AI, the Claude Agent SDK, Microsoft Agent
Framework) are async-native, so governance has to work on both call styles.
Every wrapped agent therefore exposes **two** entry points:

* :meth:`_GovernedAgent.execute` -- synchronous. The framework result is
  resolved transparently: a coroutine is run to completion, an async generator
  (or any async iterator) is drained into a list, and a sync generator is
  likewise materialised. Called from inside a running event loop, blocking on a
  coroutine is impossible, so a clear :class:`AdapterError` is raised.
* :meth:`_GovernedAgent.aexecute` -- ``await``-able, and the correct entry point
  for any async application. It awaits the framework in the *caller's* loop, so
  concurrent requests stay concurrent and ``contextvars`` (how OpenTelemetry
  propagates the active span, among other things) are preserved. Offloading
  ``execute`` to a worker thread achieves neither.

The two share every governance stage; only the framework call itself differs.
The governance stages are pure synchronous CPU work -- firewall regexes, the
policy sandbox, middleware -- so there is no async variant of them to write.
"""

import asyncio
import inspect
import uuid
from collections.abc import AsyncIterable, Callable
from typing import Any, cast

from adapt_agent.adapters.base import BaseAdapter
from adapt_agent.adversarial import AdversarialDefense
from adapt_agent.core.governance import GovernanceGate
from adapt_agent.core.governance import extract_prompt as _extract_prompt
from adapt_agent.core.governance import extract_texts as _extract_texts
from adapt_agent.core.middleware import Middleware
from adapt_agent.core.policy import PolicyEnforcer
from adapt_agent.core.types import Agent, AgentState
from adapt_agent.exceptions import AdapterError, SecurityBlockedError
from adapt_agent.observability import AgentObserver
from adapt_agent.security.firewall import Firewall

#: A callable that runs a framework agent given its prepared input. Accepts
#: arbitrary call signatures (some adapters call it with keyword arguments).
Runner = Callable[..., Any]


def _resolve_result(value: Any) -> Any:
    """Materialise a sync/async framework result into a concrete value.

    * Coroutines are run to completion.
    * Async generators / async iterators are drained into a list.
    * Sync generators are materialised into a list.

    Raises:
        AdapterError: If an awaitable is produced while an event loop is already
            running on this thread (where blocking is unsafe).
    """
    # ``AsyncIterable`` covers both ``async def`` generators and custom async
    # iterators (``__aiter__``/``__anext__``), which streaming SDKs often return.
    if isinstance(value, AsyncIterable):
        return _run_coro(_drain_async_gen(value), value)
    if inspect.isawaitable(value):
        # Through the async resolver rather than a bare await, for two reasons.
        # The awaited value may itself be a stream -- an async run method often
        # *returns* one rather than being one -- and draining it has to happen
        # in the **same** loop, because `_run_coro` opens a fresh one per call
        # and a generator created in the first is dead in the second. Sharing
        # the resolver also stops the sync and async paths drifting apart, which
        # is how this bug reached only one of them.
        return _run_coro(_aresolve_result(value), value)
    if inspect.isgenerator(value):
        return list(value)
    return value


async def _aresolve_result(value: Any) -> Any:
    """Await/drain a framework result *inside the caller's event loop*.

    The async counterpart of :func:`_resolve_result`. Because the caller already
    owns a running loop, nothing is blocked and no new loop is created -- so
    ``contextvars`` set by the caller (OpenTelemetry's active span, request-scoped
    state) remain visible to the framework call.
    """
    if isinstance(value, AsyncIterable):
        return [item async for item in value]
    if inspect.isawaitable(value):
        # Recursively: an async run method is often a coroutine that *returns* a
        # stream rather than being one, and stopping at the first await handed
        # the live generator to output screening -- which found no text, and to
        # the caller, which got a generator where the envelope documents a list.
        return await _aresolve_result(await value)
    if inspect.isgenerator(value):
        # In a worker: a sync generator from a streaming SDK does blocking work
        # *between* yields, so draining it here would stall the loop even though
        # creating it did not.
        return await asyncio.to_thread(list, value)
    return value


async def _await(value: Any) -> Any:
    return await value


async def _drain_async_gen(gen: Any) -> list[Any]:
    return [item async for item in gen]


def _run_coro(coro: Any, source: Any = None) -> Any:
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is not None:
        coro.close()
        # Close the framework's own coroutine/async generator too. It was created
        # by calling the run method and is now never awaited; without this Python
        # emits a spurious "coroutine was never awaited" RuntimeWarning that
        # points at the framework rather than at the real problem below.
        _close_unawaited(source)
        raise AdapterError(
            "Cannot synchronously run an async agent from inside a running event "
            "loop. Use `await agent.aexecute(...)` (the async twin of execute, "
            "with identical governance), or call the framework's native async API "
            "directly. Offloading execute() to a worker thread also works but "
            "serialises concurrent requests and drops contextvars, so tracing "
            "context is lost."
        )
    return asyncio.run(coro)


def _close_unawaited(value: Any) -> None:
    """Release a coroutine that will now never be awaited.

    A coroutine must be closed explicitly or Python emits "coroutine was never
    awaited", pointing at the framework's run method rather than at the real
    problem (a sync call from inside a running loop).

    An **async generator has no synchronous ``close()``** -- only ``aclose()``,
    itself a coroutine that cannot be awaited from here. That is deliberately
    fine: one reaching this path has never been started, because
    :func:`_resolve_result` raises before iterating it, so it holds no suspended
    frame and needs no finalization. Scheduling an ``aclose()`` task into a loop
    that is about to unwind would trade a warning we do not have for a
    "Task was destroyed but it is pending" that we would. Asserted by
    ``test_execute_in_a_loop_does_not_leak_an_async_generator``.
    """
    if value is None:
        return
    close = getattr(value, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # pragma: no cover - close() should not raise
            pass


class GovernedEnvelope(dict):  # noqa: UP006 - a dict subclass, deliberately untyped
    """The ``{"result": <payload>}`` wrapper `execute` puts around a non-dict.

    A plain ``dict`` in every respect that matters -- equality, ``isinstance``,
    JSON encoding, key access -- but *identifiable*, which a plain dict is not.
    Whether ``{"result": [...]}`` is a wrapper or a genuine one-field answer
    cannot be decided from its shape: both readings occur, and guessing wrongly
    either deletes the answer's only column or leaves the envelope in front of
    it. Since this wrapper is our own construct, it says so, and
    `extract_output_payload` stops guessing.

    The marker is an attribute rather than a key, so the mapping a caller sees
    is unchanged; extraction duck-types it, so nothing imports this module.
    """

    __adapt_governed_envelope__ = True

    def __init__(self, *, result: Any) -> None:
        super().__init__(result=result)


class GovernedAdapter(BaseAdapter):
    """Base adapter that wraps a framework agent with ADAPT-Agent governance.

    Subclasses customise behaviour by overriding the class attributes below and,
    where a framework needs special handling, the :meth:`_resolve_runner`,
    :meth:`_prepare_input` and :meth:`_call_runner` hooks.

    Class attributes:
        framework_name: Human-readable framework name (used by
            :meth:`get_framework_name` and default ``agent_id``).
        run_method_names: Method names to look for on the wrapped object, in
            priority order. The first callable match becomes the runner. If none
            match and the object itself is callable, the object is used directly.
        operation: Trace operation label recorded by the observer.

    Args:
        config: Optional adapter configuration dictionary.
        firewall: Optional :class:`Firewall` used to screen inputs/outputs.
        defense: Optional :class:`AdversarialDefense` used on inputs.
        policy_enforcer: Optional :class:`PolicyEnforcer` evaluated against the
            extracted agent state before execution.
        observer: Optional :class:`AgentObserver` used to trace executions.
        middleware: Optional :class:`Middleware` pipeline applied to the input
            payload (pre) and the result payload (post).
        agent_id: Stable identifier used in traces and policy checks. Defaults to
            ``"<framework>-agent"``.
        block_on_violation: When ``True`` (default), a firewall/defense hit or a
            ``block`` policy action raises :class:`SecurityBlockedError`. When
            ``False`` the execution proceeds but threats are still recorded.
    """

    framework_name: str = "Governed"
    run_method_names: tuple[str, ...] = ("invoke",)
    #: Run methods preferred by :meth:`_GovernedAgent.aexecute`, async-native
    #: first. A framework exposing both styles (LangGraph ``ainvoke``/``invoke``,
    #: CrewAI ``kickoff_async``/``kickoff``, Pydantic AI ``run``/``run_sync``)
    #: must be driven by the async one there, or ``aexecute`` blocks the very
    #: event loop it exists to cooperate with. Empty falls back to
    #: :attr:`run_method_names`.
    async_run_method_names: tuple[str, ...] = ()
    operation: str = "invoke"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        firewall: Firewall | None = None,
        defense: AdversarialDefense | None = None,
        policy_enforcer: PolicyEnforcer | None = None,
        observer: AgentObserver | None = None,
        middleware: Middleware | None = None,
        agent_id: str | None = None,
        block_on_violation: bool = True,
    ):
        super().__init__(config)
        self.firewall = firewall
        self.defense = defense
        self.policy_enforcer = policy_enforcer
        self.observer = observer
        self.middleware = middleware
        self.agent_id = agent_id or f"{self.framework_name.lower().replace(' ', '-')}-agent"
        self.block_on_violation = block_on_violation

    # -- framework-specific hooks ---------------------------------------------

    def _resolve_runner(self, agent: Any) -> Runner | None:
        """Return a callable that runs ``agent`` given prepared input, or ``None``.

        The default looks for the first callable attribute named in
        :attr:`run_method_names`, then falls back to the object itself if it is
        callable. Subclasses may override to support runner/agent frameworks.
        """
        for name in self.run_method_names:
            candidate = getattr(agent, name, None)
            if callable(candidate):
                return cast(Runner, candidate)
        if callable(agent):
            return cast(Runner, agent)
        return None

    def _resolve_async_runner(self, agent: Any) -> Runner | None:
        """Return the runner :meth:`aexecute` should use, preferring async.

        Falls back to :meth:`_resolve_runner` when the adapter declares no
        async-specific methods or the object exposes none of them -- a purely
        synchronous framework is still callable from ``aexecute``.
        """
        for name in self.async_run_method_names:
            candidate = getattr(agent, name, None)
            if callable(candidate):
                return cast(Runner, candidate)
        return self._resolve_runner(agent)

    def _prepare_input(self, payload: dict[str, Any]) -> Any:
        """Transform the (post-middleware) payload into the runner's argument.

        The default passes the payload through unchanged (state-dict frameworks
        such as LangGraph). Prompt-based frameworks override this to return a
        string via :func:`_extract_prompt`.
        """
        return payload

    def _call_runner(self, runner: Runner, payload: dict[str, Any]) -> Any:
        """Invoke ``runner`` with the prepared input and return the raw result."""
        return runner(self._prepare_input(payload))

    # -- BaseAdapter contract --------------------------------------------------

    def validate_agent(self, agent: Any) -> bool:
        """Return ``True`` if a runner can be resolved from ``agent``."""
        return self._resolve_runner(agent) is not None

    def wrap_agent(self, agent: Any) -> Agent:
        """Wrap a framework agent with ADAPT-Agent governance.

        Raises:
            AdapterError: If no runnable entry point can be resolved.
        """
        if not self.validate_agent(agent):
            # An adapter with no run_method_names wraps a *callable* by design
            # (Google ADK: a run needs session/user arguments, so there is no
            # zero-argument method to bind). Saying "expected one of ''" there
            # reads as a broken adapter rather than as the wrong argument.
            expected = (
                f"Expected one of {', '.join(self.run_method_names)!r} or a plain callable."
                if self.run_method_names
                else (
                    "This adapter takes a plain callable -- one you write that performs "
                    "the run and returns (or yields) its result -- because the framework's "
                    "own entry point needs arguments the adapter cannot supply."
                )
            )
            raise AdapterError(
                f"{type(self).__name__}.wrap_agent could not find a runnable entry "
                f"point on the supplied {type(agent).__name__}. {expected} "
                f"See the adapter docstring for the object to wrap."
            )
        return _GovernedAgent(agent, self)

    def extract_state(self, agent: Any) -> AgentState:
        """Extract a normalized :class:`AgentState` from a state payload or agent.

        Accepts either a state mapping (e.g. ``{"messages": [...], ...}``) or a
        stateful object exposing ``get_state``. Extraction is best-effort and
        always returns a well-formed :class:`AgentState`.
        """
        raw: Any = agent
        get_state = getattr(agent, "get_state", None)
        if callable(get_state):
            try:
                snapshot = get_state(self.config.get("graph_config", {}))
                raw = getattr(snapshot, "values", snapshot)
            except Exception:
                raw = {}

        if not isinstance(raw, dict):
            raw = {}

        messages = raw.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        context = {k: v for k, v in raw.items() if k != "messages"}

        state: AgentState = {"messages": messages, "context": context}
        if "trust_score" in raw and isinstance(raw["trust_score"], (int, float)):
            state["trust_score"] = float(raw["trust_score"])
        if "policy_violations" in raw and isinstance(raw["policy_violations"], list):
            state["policy_violations"] = raw["policy_violations"]
        return state

    def inject_middleware(self, agent: Any, middleware: Any) -> Any:
        """Attach a middleware pipeline and return a freshly wrapped agent."""
        if not isinstance(middleware, Middleware):
            raise AdapterError("inject_middleware expects an adapt_agent.core.Middleware instance.")
        self.middleware = middleware
        return self.wrap_agent(agent)

    def get_framework_name(self) -> str:
        """Return the human-readable framework name."""
        return self.framework_name

    # -- internal helpers shared with the wrapped agent ------------------------

    @property
    def gate(self) -> GovernanceGate:
        """A :class:`GovernanceGate` over this adapter's currently-set controls.

        Built per access rather than cached, so reassigning ``adapter.firewall``
        (or any other control) after construction takes effect immediately.
        """
        return GovernanceGate(
            firewall=self.firewall,
            defense=self.defense,
            policy_enforcer=self.policy_enforcer,
            block_on_violation=self.block_on_violation,
            agent_id=self.agent_id,
        )

    def _screen_input(self, payload: Any) -> list[str]:
        """Run firewall + adversarial defense over a payload, returning threats."""
        return self.gate.scan_input(payload)

    def _screen_output(self, payload: Any) -> list[str]:
        """Run firewall over an output payload, returning threats."""
        return self.gate.scan_output(payload)


class _GovernedAgent:
    """A framework agent wrapped with ADAPT-Agent governance.

    Implements the :class:`~adapt_agent.core.types.Agent` protocol (``execute``
    and ``get_state``).
    """

    def __init__(self, agent: Any, adapter: GovernedAdapter):
        self._agent = agent
        self._adapter = adapter
        # ``wrap_agent`` validates the agent first, so a runner always resolves.
        self._runner: Runner = cast(Runner, adapter._resolve_runner(agent))
        # Resolved separately so `aexecute` uses the framework's async entry
        # point where one exists, rather than blocking on its sync twin.
        self._arunner: Runner = cast(Runner, adapter._resolve_async_runner(agent))
        self._last_state: AgentState = {"messages": [], "context": {}}

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the wrapped agent with governance applied, synchronously.

        Order of operations: input screening -> policy check -> pre-middleware
        -> traced run -> post-middleware -> output screening.

        An async-native framework is driven by running its coroutine to
        completion, which is impossible from inside a running event loop. In an
        async application use :meth:`aexecute` instead -- it applies exactly the
        same governance.

        Raises:
            SecurityBlockedError: If a control blocks the input/output (and
                ``block_on_violation`` is enabled).
            AdapterError: If the framework is async and an event loop is already
                running on this thread.
        """
        trace_id, payload = self._before(input_data)
        try:
            raw = self._adapter._call_runner(self._runner, payload)
            result = _resolve_result(raw)
        except BaseException as exc:  # close the span on KeyboardInterrupt too
            self._trace_error(trace_id, exc)
            raise
        return self._after(trace_id, result)

    async def aexecute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the wrapped agent with governance applied, asynchronously.

        The async twin of :meth:`execute`, and the right entry point for an
        async application: the framework is awaited in the caller's own event
        loop, so concurrent requests stay concurrent and ``contextvars`` -- the
        mechanism OpenTelemetry uses to propagate the active span -- survive the
        call. Running :meth:`execute` in a worker thread preserves neither.

        Every governance stage is identical to :meth:`execute`; only the
        framework call itself is awaited rather than blocked on. A *synchronous*
        framework works here too (nothing to await is simply passed through), so
        an async app can use this entry point uniformly.

        Raises:
            SecurityBlockedError: If a control blocks the input/output (and
                ``block_on_violation`` is enabled).
        """
        trace_id, payload = self._before(input_data)
        try:
            raw = await self._acall_runner(payload)
            result = await _aresolve_result(raw)
        except BaseException as exc:
            # BaseException, not Exception: `asyncio.CancelledError` derives from
            # BaseException, so a cancelled request would otherwise leave the
            # observer span open forever.
            self._trace_error(trace_id, exc)
            raise
        return self._after(trace_id, result)

    async def _acall_runner(self, payload: dict[str, Any]) -> Any:
        """Invoke the async-preferred runner without blocking the caller's loop.

        A framework with no async entry point still reaches `aexecute` -- the
        resolver falls back to the sync runner so an async app can use one entry
        point uniformly. But calling it does the work *before* the first await,
        which blocks the whole loop: measured with a slow sync agent, a
        heartbeat task got zero ticks and three concurrent calls serialised
        (0.45s for 3 x 150ms). That defeats the concurrency this method exists
        for, and this docstring promises.

        A declared coroutine or async-generator function returns immediately, so
        it is called inline. Anything else goes to a worker thread --
        ``asyncio.to_thread`` propagates ``contextvars``, so an active
        OpenTelemetry span still reaches the framework. A runner that merely
        *returns* a coroutine (some adapters resolve one via a lambda) takes the
        thread hop too and pays only its cost, because it returns at once.
        """
        runner = self._arunner
        if inspect.iscoroutinefunction(runner) or inspect.isasyncgenfunction(runner):
            return self._adapter._call_runner(runner, payload)
        return await asyncio.to_thread(self._adapter._call_runner, runner, payload)

    # -- shared governance stages ---------------------------------------------
    # execute() and aexecute() differ only in how the framework result is
    # materialised; keeping the stages here means governance can never drift
    # between the two call styles.

    def _before(self, input_data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Stages 1-3: input screening, policy, pre-middleware. Opens the trace."""
        adapter = self._adapter
        trace_id = uuid.uuid4().hex

        # 1. Input screening.
        in_threats = adapter._screen_input(input_data)
        if in_threats and adapter.block_on_violation:
            raise SecurityBlockedError("Input blocked by security controls", in_threats)

        # 2. Policy enforcement against the extracted state.
        #
        # Policy is evaluated *whatever* the blocking mode: `check_state` is what
        # records violations and fires warn/log handlers, so gating the call on
        # `block_on_violation` would turn the documented report-only rollout into
        # silence -- no auditing at all, rather than auditing without refusal.
        # Only the refusal itself is conditional.
        self._last_state = adapter.extract_state(input_data)
        blocking = adapter.gate.policy_violations(self._last_state)
        if blocking and adapter.block_on_violation:
            raise SecurityBlockedError("Input blocked by policy", [f"policy:{v}" for v in blocking])

        # 3. Pre-middleware.
        payload = input_data
        if adapter.middleware is not None:
            payload = adapter.middleware.process_input(input_data)

        # 4. Traced execution begins (the run itself is the caller's job).
        if adapter.observer is not None:
            adapter.observer.start_trace(trace_id, adapter.agent_id, adapter.operation)
        return trace_id, payload

    def _trace_error(self, trace_id: str, exc: BaseException) -> None:
        if self._adapter.observer is not None:
            self._adapter.observer.end_trace(trace_id, status="error", result=str(exc))

    def _after(self, trace_id: str, result: Any) -> dict[str, Any]:
        """Stages 5-6: post-middleware, output screening, then close the trace.

        The trace closes *last*, and as an error if either stage raises. Closing
        it first recorded a successful execution for a run the caller saw fail:
        an output block is exactly the event monitoring exists to surface, and
        the runner's own ``try`` no longer covers this method.
        """
        adapter = self._adapter
        try:
            # 5. Post-middleware.
            if adapter.middleware is not None:
                wrapped = adapter.middleware.process_output({"result": result})
                result = wrapped["result"]

            # 6. Output screening.
            out_threats = adapter._screen_output(result)
            if out_threats and adapter.block_on_violation:
                raise SecurityBlockedError("Output blocked by security controls", out_threats)
        except BaseException as exc:  # CancelledError derives from BaseException
            self._trace_error(trace_id, exc)
            raise

        if adapter.observer is not None:
            adapter.observer.end_trace(trace_id, status="completed")

        # Track state from the actual returned payload. Non-dict framework
        # results (AgentRunResult, CrewOutput, ...) are wrapped in {"result": ...}
        # so get_state() reflects the latest execution rather than stale input.
        output_payload = result if isinstance(result, dict) else GovernedEnvelope(result=result)
        self._last_state = adapter.extract_state(output_payload)
        return output_payload

    def get_state(self) -> AgentState:
        """Return the most recently observed agent state."""
        return self._last_state


__all__ = [
    "GovernedAdapter",
    "GovernedEnvelope",
    "_GovernedAgent",
    "_aresolve_result",
    "_extract_prompt",
    "_extract_texts",
]
