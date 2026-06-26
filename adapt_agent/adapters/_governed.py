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
Several modern frameworks expose async-only entry points. ``execute`` therefore
resolves the framework result transparently:

* a coroutine is run to completion;
* an async generator (or any async iterator) is drained into a list;
* a sync generator is likewise materialised into a list.

When called from inside a running event loop, blocking on a coroutine is not
possible; in that case a clear :class:`AdapterError` is raised pointing the
caller at the framework's native async API.
"""

import asyncio
import inspect
import uuid
from typing import Any, Callable, Optional, cast

from adapt_agent.adapters.base import BaseAdapter
from adapt_agent.adversarial import AdversarialDefense
from adapt_agent.core.middleware import Middleware
from adapt_agent.core.policy import PolicyEnforcer
from adapt_agent.core.types import Agent, AgentState
from adapt_agent.exceptions import AdapterError, SecurityBlockedError
from adapt_agent.observability import AgentObserver
from adapt_agent.security.firewall import Firewall

#: A callable that runs a framework agent given its prepared input. Accepts
#: arbitrary call signatures (some adapters call it with keyword arguments).
Runner = Callable[..., Any]

#: Attribute names that commonly hold human-readable text on framework result
#: and message objects (Pydantic AI ``.output``, Microsoft Agent Framework
#: ``.text``, CrewAI ``.raw``, OpenAI Agents ``.final_output``, Claude
#: ``ResultMessage.result`` / ``TextBlock.text``, LangChain ``.content`` ...).
_TEXT_ATTRS = (
    "content",
    "text",
    "output",
    "final_output",
    "raw",
    "result",
    "data",
)

#: Attribute names that hold *containers* of further text-bearing objects
#: (e.g. ``Content.parts`` in Google ADK, message lists, CrewAI ``tasks_output``).
#: These are recursed into even when they are framework objects rather than
#: plain dicts/lists, so screening reaches deeply-structured results.
_RECURSE_ATTRS = (
    "content",
    "parts",
    "messages",
    "tasks_output",
)


def _extract_texts(data: Any) -> list[str]:
    """Best-effort extraction of human-readable text from an arbitrary payload.

    Adapter payloads are typically dicts that may contain a ``messages`` list or
    arbitrary string fields, but framework result objects expose their text via
    attributes instead (see :data:`_TEXT_ATTRS`). We collect every string we can
    reach so the security controls can scan it without assuming a fixed schema.
    """
    texts: list[str] = []

    def _walk(value: Any, depth: int = 0) -> None:
        if depth > 6:  # bound recursion (defensive against pathological nesting)
            return
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                _walk(v, depth + 1)
        elif isinstance(value, (list, tuple)):
            for v in value:
                _walk(v, depth + 1)
        else:
            # Framework message / result objects expose their text via one of a
            # handful of well-known attributes. Recurse into anything walkable.
            for attr in _TEXT_ATTRS:
                inner = getattr(value, attr, None)
                if isinstance(inner, str):
                    texts.append(inner)
                elif isinstance(inner, (dict, list, tuple)):
                    _walk(inner, depth + 1)
            # Structured content containers (e.g. genai ``Content.parts``) hold
            # further objects whose text we still want to scan.
            for attr in _RECURSE_ATTRS:
                inner = getattr(value, attr, None)
                if inner is not None and not isinstance(inner, (str, int, float, bool)):
                    _walk(inner, depth + 1)

    _walk(data)
    return texts


def _extract_prompt(payload: Any) -> str:
    """Derive a single prompt string from an adapter payload.

    Many frameworks (Pydantic AI, OpenAI Agents, Microsoft Agent Framework,
    Claude Agent SDK) run from a string prompt rather than a state dict. This
    helper picks the most plausible prompt: the latest user message, then a
    common prompt-like key, then a string fallback.
    """
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            for message in reversed(messages):
                if isinstance(message, dict):
                    role, content = message.get("role"), message.get("content")
                else:
                    role = getattr(message, "role", None)
                    content = getattr(message, "content", None)
                if isinstance(content, str) and (role == "user" or role is None):
                    return content
            last = messages[-1]
            last_content = last.get("content") if isinstance(last, dict) else getattr(
                last, "content", None
            )
            if isinstance(last_content, str):
                return last_content
        for key in ("prompt", "input", "query", "text"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
    return str(payload)


def _resolve_result(value: Any) -> Any:
    """Materialise a sync/async framework result into a concrete value.

    * Coroutines are run to completion.
    * Async generators / async iterators are drained into a list.
    * Sync generators are materialised into a list.

    Raises:
        AdapterError: If an awaitable is produced while an event loop is already
            running on this thread (where blocking is unsafe).
    """
    if inspect.isasyncgen(value):
        return _run_coro(_drain_async_gen(value))
    if inspect.isawaitable(value):
        return _run_coro(_await(value))
    if inspect.isgenerator(value):
        return list(value)
    return value


async def _await(value: Any) -> Any:
    return await value


async def _drain_async_gen(gen: Any) -> list[Any]:
    return [item async for item in gen]


def _run_coro(coro: Any) -> Any:
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is not None:
        coro.close()
        raise AdapterError(
            "Cannot synchronously run an async agent from inside a running event "
            "loop. Call the framework's native async API directly, or run "
            "execute() in a worker thread."
        )
    return asyncio.run(coro)


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
    operation: str = "invoke"

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        *,
        firewall: Optional[Firewall] = None,
        defense: Optional[AdversarialDefense] = None,
        policy_enforcer: Optional[PolicyEnforcer] = None,
        observer: Optional[AgentObserver] = None,
        middleware: Optional[Middleware] = None,
        agent_id: Optional[str] = None,
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

    def _resolve_runner(self, agent: Any) -> Optional[Runner]:
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
            raise AdapterError(
                f"{type(self).__name__}.wrap_agent could not find a runnable entry "
                f"point on the supplied object. Expected one of "
                f"{', '.join(self.run_method_names)!r} or a plain callable. See the "
                f"adapter docstring for the object to wrap."
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

    def _screen_input(self, payload: Any) -> list[str]:
        """Run firewall + adversarial defense over a payload, returning threats."""
        threats: list[str] = []
        for text in _extract_texts(payload):
            if self.firewall is not None and not self.firewall.check_input(text):
                threats.append("firewall")
            if self.defense is not None:
                analysis = self.defense.analyze_input(text)
                if not analysis["is_safe"]:
                    threats.extend(analysis["threats_detected"])
        return threats

    def _screen_output(self, payload: Any) -> list[str]:
        """Run firewall over an output payload, returning threats."""
        threats: list[str] = []
        if self.firewall is None:
            return threats
        for text in _extract_texts(payload):
            if not self.firewall.check_output(text):
                threats.append("firewall")
        return threats


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
        self._last_state: AgentState = {"messages": [], "context": {}}

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the wrapped agent with governance applied.

        Order of operations: input screening -> policy check -> pre-middleware
        -> traced run -> post-middleware -> output screening.

        Raises:
            SecurityBlockedError: If a control blocks the input/output (and
                ``block_on_violation`` is enabled).
        """
        adapter = self._adapter
        trace_id = uuid.uuid4().hex

        # 1. Input screening.
        in_threats = adapter._screen_input(input_data)
        if in_threats and adapter.block_on_violation:
            raise SecurityBlockedError("Input blocked by security controls", in_threats)

        # 2. Policy enforcement against the extracted state.
        self._last_state = adapter.extract_state(input_data)
        if adapter.policy_enforcer is not None:
            violations = adapter.policy_enforcer.check_state(self._last_state)
            if violations and adapter.block_on_violation:
                blocking = []
                for v in violations:
                    rule = adapter.policy_enforcer.get_rule(v)
                    if rule is not None and rule.get("action") == "block":
                        blocking.append(v)
                if blocking:
                    raise SecurityBlockedError(
                        "Input blocked by policy", [f"policy:{v}" for v in blocking]
                    )

        # 3. Pre-middleware.
        payload = input_data
        if adapter.middleware is not None:
            payload = adapter.middleware.process_input(input_data)

        # 4. Traced execution.
        if adapter.observer is not None:
            adapter.observer.start_trace(trace_id, adapter.agent_id, adapter.operation)
        try:
            raw = adapter._call_runner(self._runner, payload)
            result = _resolve_result(raw)
        except Exception as exc:
            if adapter.observer is not None:
                adapter.observer.end_trace(trace_id, status="error", result=str(exc))
            raise
        if adapter.observer is not None:
            adapter.observer.end_trace(trace_id, status="completed")

        # 5. Post-middleware.
        if adapter.middleware is not None:
            wrapped = adapter.middleware.process_output({"result": result})
            result = wrapped["result"]

        # 6. Output screening.
        out_threats = adapter._screen_output(result)
        if out_threats and adapter.block_on_violation:
            raise SecurityBlockedError("Output blocked by security controls", out_threats)

        if isinstance(result, dict):
            self._last_state = adapter.extract_state(result)
        return result if isinstance(result, dict) else {"result": result}

    def get_state(self) -> AgentState:
        """Return the most recently observed agent state."""
        return self._last_state


__all__ = ["GovernedAdapter", "_GovernedAgent", "_extract_texts", "_extract_prompt"]
