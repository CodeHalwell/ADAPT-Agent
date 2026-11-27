"""LangGraph adapter for ADAPT-Agent.

This adapter wraps a compiled LangGraph graph -- a ``langchain.agents.create_agent``
agent, a hand-built ``StateGraph``, or any object exposing an ``invoke`` method --
with ADAPT-Agent's security and governance primitives: a
:class:`~adapt_agent.security.Firewall`, an
:class:`~adapt_agent.adversarial.AdversarialDefense`, a
:class:`~adapt_agent.core.PolicyEnforcer`, an
:class:`~adapt_agent.observability.AgentObserver` and a
:class:`~adapt_agent.core.Middleware` pipeline.

It governs the *boundary* of a run: the input before it, the result after it.
Individual model and tool calls inside the graph are governed by
:func:`adapt_agent.integrations.langgraph.governance_middleware`; the two
compose.

Run configuration
-----------------
Extra keyword arguments to ``execute`` / ``aexecute`` / ``stream`` / ``astream``
reach ``invoke`` / ``ainvoke`` / ``stream`` / ``astream`` unchanged, so a
checkpointed thread and a human-in-the-loop cycle work through the wrapper::

    config = {"configurable": {"thread_id": "t1"}}
    out = guarded.execute({"messages": [...]}, config=config)
    if out.get("__interrupt__"):                 # paused for approval
        out = guarded.execute(Command(resume={"type": "approve"}), config=config)

Interrupts are never dropped: a ``version="v2"`` ``GraphOutput`` is unpacked
to the same ``{..., "__interrupt__": [...]}`` mapping ``version="v1"`` returns.

The adapter is intentionally *structural*: importing this module never imports
``langgraph``.

Example
-------
>>> from adapt_agent.adapters import LangGraphAdapter
>>> from adapt_agent.security import Firewall
>>> fw = Firewall(max_content_length=10_000)
>>> fw.add_blocked_pattern(r"(?i)ignore previous instructions")
>>> adapter = LangGraphAdapter(firewall=fw)
>>> guarded = adapter.wrap_agent(compiled_graph)   # doctest: +SKIP
>>> guarded.execute({"messages": [{"role": "user", "content": "hi"}]})  # doctest: +SKIP
"""

import inspect
import uuid
from collections.abc import AsyncIterator, Iterator
from contextvars import ContextVar
from typing import Any

from adapt_agent.adapters._governed import GovernedAdapter, Runner, _extract_texts, _GovernedAgent
from adapt_agent.core.types import Agent
from adapt_agent.exceptions import AdapterError, SecurityBlockedError

#: The key under which LangGraph ``version="v1"`` reports pending interrupts.
_INTERRUPT_KEY = "__interrupt__"

#: The run's ``config`` while the input stages run, so a ``Command`` can be
#: judged against its thread's checkpoint. A context variable rather than an
#: attribute: one wrapped graph serves concurrent requests on different threads.
_RUN_CONFIG: ContextVar[Any] = ContextVar("adapt_langgraph_run_config", default=None)


def _is_command(value: Any) -> bool:
    """A LangGraph ``Command``, duck-typed so LangGraph is not imported."""
    return type(value).__name__ == "Command" and hasattr(value, "resume")


def _unpack_graph_output(result: Any) -> Any:
    """Turn a ``version="v2"`` ``GraphOutput`` into the v1 mapping, interrupts kept.

    A ``GraphOutput`` is not a dict, so the shared stages would wrap it in a
    ``{"result": ...}`` envelope whose interrupts nobody looks for -- a paused
    run then reads as a finished one. Duck-typed so LangGraph is not imported.
    """
    if type(result).__name__ != "GraphOutput" or not hasattr(result, "interrupts"):
        return result
    value = getattr(result, "value", None)
    interrupts = list(getattr(result, "interrupts", ()) or ())
    if isinstance(value, dict):
        unpacked = dict(value)
        if interrupts:
            unpacked[_INTERRUPT_KEY] = interrupts
        return unpacked
    if interrupts:
        return {"result": value, _INTERRUPT_KEY: interrupts}
    return value


async def _aunpack(awaitable: Any) -> Any:
    return _unpack_graph_output(await awaitable)


class LangGraphAdapter(GovernedAdapter):
    """Adapter for integrating ADAPT-Agent with LangGraph agents.

    Wrap a **compiled** LangGraph graph -- or any object exposing a callable
    ``invoke(state) -> state``. See :class:`~adapt_agent.adapters._governed.GovernedAdapter`
    for the full constructor signature; every security control is optional and
    keyword-only. The wrapped agent adds governed :meth:`_LangGraphGovernedAgent.stream`
    and :meth:`_LangGraphGovernedAgent.astream`.
    """

    framework_name = "LangGraph"
    run_method_names = ("invoke",)
    async_run_method_names = ("ainvoke", "invoke")
    operation = "langgraph.invoke"

    def _call_runner(self, runner: Runner, payload: Any, **run_kwargs: Any) -> Any:
        """Run the graph, forwarding ``config=`` / ``context=`` / ``version=``.

        The payload may be a state mapping or a LangGraph ``Command`` (a
        ``Command(resume=...)`` answering an interrupt); either goes to the
        graph as-is.
        """
        raw = runner(self._prepare_input(payload), **run_kwargs)
        if inspect.isawaitable(raw):
            return _aunpack(raw)
        return _unpack_graph_output(raw)

    def wrap_agent(self, agent: Any) -> Agent:
        """Wrap a compiled graph; the result also offers governed streaming."""
        super().wrap_agent(agent)  # validates, raising AdapterError for a non-graph
        return _LangGraphGovernedAgent(agent, self)


class _LangGraphGovernedAgent(_GovernedAgent):
    """A governed graph with ``stream`` / ``astream`` that screen before releasing.

    The input goes through the same stages as :meth:`execute` (screening,
    policy, pre-middleware) before the graph starts. Each streamed item is then
    screened with the output controls *before it is yielded*:

    * ``stream_mode="messages"``: token chunks are held per message (grouped by
      message id) and released together once the whole message has passed. A
      blocked pattern split across two tokens is therefore still caught; the
      cost is that the consumer receives each message whole instead of token
      by token.
    * every other mode (``"updates"``, ``"values"``, ``"custom"``, ...): each
      item is screened and released on its own.

    A blocked item raises :class:`~adapt_agent.exceptions.SecurityBlockedError`
    (when ``block_on_violation`` is set) and is never yielded; everything
    released before it had passed. Post-middleware does not apply to streams --
    it rewrites one result, and a stream has none.
    """

    # -- resuming a checkpointed thread -------------------------------------------

    def execute(self, input_data: Any, **run_kwargs: Any) -> dict[str, Any]:
        token = _RUN_CONFIG.set(run_kwargs.get("config"))
        try:
            return super().execute(input_data, **run_kwargs)
        finally:
            _RUN_CONFIG.reset(token)

    async def aexecute(self, input_data: Any, **run_kwargs: Any) -> dict[str, Any]:
        token = _RUN_CONFIG.set(run_kwargs.get("config"))
        try:
            return await super().aexecute(input_data, **run_kwargs)
        finally:
            _RUN_CONFIG.reset(token)

    def _before(self, input_data: Any) -> tuple[str, Any]:
        """The shared input stages, with a ``Command`` judged against its thread.

        A ``Command(resume=...)`` carries an answer, not a conversation: the
        state a policy rule gates on lives in the thread's checkpoint. Evaluated
        against the ``Command`` itself every state rule sees missing keys, and a
        fail-closed enforcer then refuses every resume. So the policy state is
        the checkpointed values (plus the ``Command``'s own ``update``); the
        ``Command`` is still screened as input, since its resume value and
        update are about to enter the graph.

        Pre-middleware rewrites state payloads and cannot process a
        ``Command``. A resume through an adapter that has pre-middleware
        configured is therefore refused, as for any other non-dict input:
        skipping a configured stage would be a control that is accepted and
        silently does nothing.
        """
        if not _is_command(input_data):
            return super()._before(input_data)
        adapter = self._adapter
        if adapter.middleware is not None and adapter.middleware.has_input_stages:
            raise AdapterError(
                f"{type(adapter).__name__} has pre-middleware configured, which "
                "processes dict payloads, but execute() received a Command. Resume "
                "the thread through an adapter without pre-middleware."
            )
        threats = adapter._screen_input(input_data)
        if threats and adapter.block_on_violation:
            self._refused(threats, stage="input")
            raise SecurityBlockedError("Input blocked by security controls", threats)
        state = adapter.extract_state(self._thread_state(input_data))
        blocking = adapter.gate.policy_violations(state)
        if blocking and adapter.block_on_violation:
            policy_threats = [f"policy:{v}" for v in blocking]
            self._refused(policy_threats, stage="policy")
            raise SecurityBlockedError("Input blocked by policy", policy_threats)
        trace_id = uuid.uuid4().hex
        if adapter.observer is not None:
            adapter.observer.start_trace(trace_id, adapter.agent_id, adapter.operation)
        return trace_id, input_data

    def _thread_state(self, command: Any) -> dict[str, Any]:
        values: dict[str, Any] = {}
        get_state = getattr(self._agent, "get_state", None)
        config = _RUN_CONFIG.get()
        if callable(get_state) and config is not None:
            try:
                snapshot = get_state(config)
            except Exception:
                snapshot = None  # no checkpointer: nothing to read, rules fail closed
            current = getattr(snapshot, "values", None)
            if isinstance(current, dict):
                values.update(current)
        update = getattr(command, "update", None)
        if isinstance(update, dict):
            values.update(update)
        return values

    # -- streaming -----------------------------------------------------------------

    def stream(self, input_data: Any, **run_kwargs: Any) -> Iterator[Any]:
        """Governed ``graph.stream(...)``. Input is screened before this returns.

        Raises:
            SecurityBlockedError: From the input stages immediately; from an
                item while iterating.
            AdapterError: If the wrapped object has no ``stream`` method.
        """
        method = getattr(self._agent, "stream", None)
        if not callable(method):
            raise AdapterError("The wrapped LangGraph object has no stream() method.")
        token = _RUN_CONFIG.set(run_kwargs.get("config"))
        try:
            trace_id, payload = self._before(input_data)
        finally:
            _RUN_CONFIG.reset(token)
        try:
            source = method(self._adapter._prepare_input(payload), **run_kwargs)
        except BaseException as exc:
            self._trace_error(trace_id, exc)
            raise
        return self._screen_sync(trace_id, source, run_kwargs)

    def astream(self, input_data: Any, **run_kwargs: Any) -> AsyncIterator[Any]:
        """Governed ``graph.astream(...)``: ``async for item in agent.astream(...)``.

        Input is screened when this is called, before the graph starts.

        Raises:
            SecurityBlockedError: From the input stages immediately; from an
                item while iterating.
            AdapterError: If the wrapped object has no ``astream`` method.
        """
        method = getattr(self._agent, "astream", None)
        if not callable(method):
            raise AdapterError("The wrapped LangGraph object has no astream() method.")
        token = _RUN_CONFIG.set(run_kwargs.get("config"))
        try:
            trace_id, payload = self._before(input_data)
        finally:
            _RUN_CONFIG.reset(token)
        try:
            source = method(self._adapter._prepare_input(payload), **run_kwargs)
        except BaseException as exc:
            self._trace_error(trace_id, exc)
            raise
        return self._screen_async(trace_id, source, run_kwargs)

    # -- screening ---------------------------------------------------------------

    def _screen_sync(self, trace_id: str, source: Any, run_kwargs: dict[str, Any]) -> Iterator[Any]:
        buffer = _MessageBuffer(self, run_kwargs)
        try:
            for item in source:
                yield from buffer.push(item)
            yield from buffer.flush()
        except BaseException as exc:
            self._trace_error(trace_id, exc)
            raise
        finally:
            close = getattr(source, "close", None)
            if callable(close):
                close()
        self._close_trace(trace_id)

    async def _screen_async(
        self, trace_id: str, source: Any, run_kwargs: dict[str, Any]
    ) -> AsyncIterator[Any]:
        buffer = _MessageBuffer(self, run_kwargs)
        try:
            async for item in source:
                for released in buffer.push(item):
                    yield released
            for released in buffer.flush():
                yield released
        except BaseException as exc:
            self._trace_error(trace_id, exc)
            raise
        finally:
            aclose = getattr(source, "aclose", None)
            if callable(aclose):
                await aclose()
        self._close_trace(trace_id)

    def _close_trace(self, trace_id: str) -> None:
        if self._adapter.observer is not None:
            self._adapter.observer.end_trace(trace_id, status="completed")

    def _screen_items(self, items: list[Any]) -> None:
        threats = self._adapter._screen_output(items)
        if threats and self._adapter.block_on_violation:
            raise SecurityBlockedError("Output blocked by security controls", threats)


class _MessageBuffer:
    """Holds ``messages``-mode chunks of one message until it is complete.

    Item shapes follow ``graph.stream``'s arguments: ``data`` for one mode,
    ``(mode, data)`` for a list of modes, with a leading namespace when
    ``subgraphs=True``; ``version="v2"`` yields ``{"type", "ns", "data"}``
    parts. A ``messages`` datum is ``(chunk, metadata)``.
    """

    def __init__(self, agent: _LangGraphGovernedAgent, run_kwargs: dict[str, Any]):
        self._agent = agent
        mode = run_kwargs.get("stream_mode", "values")
        self._modes_listed = isinstance(mode, (list, tuple))
        self._single_mode = None if self._modes_listed else mode
        self._subgraphs = bool(run_kwargs.get("subgraphs", False))
        self._v2 = run_kwargs.get("version") == "v2"
        self._pending: list[Any] = []
        self._key: Any = None

    def _classify(self, item: Any) -> tuple[Any, Any, Any]:
        """``(namespace, mode, data)`` for one streamed item."""
        if self._v2 and isinstance(item, dict) and "type" in item:
            return tuple(item.get("ns") or ()), item.get("type"), item.get("data")
        ns: Any = ()
        rest = item
        if self._subgraphs and isinstance(rest, tuple) and len(rest) >= 2:
            # `(ns, data)` for one mode, `(ns, mode, data)` for several.
            ns, rest = rest[0], (rest[1:] if self._modes_listed else rest[1])
        if self._modes_listed and isinstance(rest, tuple) and len(rest) == 2:
            return ns, rest[0], rest[1]
        return ns, self._single_mode, rest

    def push(self, item: Any) -> list[Any]:
        ns, mode, data = self._classify(item)
        if mode == "messages" and isinstance(data, tuple) and data:
            key = (ns, getattr(data[0], "id", None))
            if self._pending and key == self._key and key[1] is not None:
                self._pending.append(item)
                return []
            released = self.flush()
            self._pending, self._key = [item], key
            return released
        released = self.flush()
        self._agent._screen_items([data])
        released.append(item)
        return released

    def flush(self) -> list[Any]:
        if not self._pending:
            return []
        pending, self._pending, self._key = self._pending, [], None
        # The chunks' text, joined: a pattern split across tokens is matched.
        chunks = [self._classify(item)[2][0] for item in pending]
        joined = "".join(c for c in (_chunk_text(chunk) for chunk in chunks) if c)
        self._agent._screen_items([joined, chunks])
        return pending


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    return "".join(_extract_texts(content)) if content is not None else ""


__all__ = ["LangGraphAdapter", "_extract_texts"]
