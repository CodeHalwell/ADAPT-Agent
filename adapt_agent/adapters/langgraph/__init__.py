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
      message id) and released together once the message is complete and has
      passed, in arrival order -- also when the chunks of several messages
      interleave. A blocked pattern split across two tokens is therefore still
      caught; the cost is that the consumer receives each message whole instead
      of token by token. See ``_MessageBuffer`` for what counts as complete.
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


#: LangChain message classes that are whole messages rather than stream chunks.
_WHOLE_MESSAGE_TYPES = frozenset(
    {"AIMessage", "ToolMessage", "HumanMessage", "SystemMessage", "FunctionMessage", "ChatMessage"}
)


class _Held:
    """One streamed item waiting in the buffer, and whether it may be released."""

    __slots__ = ("item", "ready")

    def __init__(self, item: Any, *, ready: bool = False):
        self.item = item
        self.ready = ready


class _MessageBuffer:
    """Holds ``messages``-mode chunks until their message is complete and screened.

    A message is screened as a whole -- its chunks' text joined -- because a
    pattern split across tokens is invisible to any chunk-by-chunk check. So no
    chunk of a message may be released before that message is known to be
    complete, and "the next chunk belongs to a different message" is **not**
    evidence of that: parallel nodes, or one node awaiting two models at once,
    interleave their chunks. Releasing on a change of message id let
    ``(a, "TOP"), (b, "hello"), (a, "SECRET")`` through with a firewall
    blocking ``TOPSECRET``, because neither half of ``a`` was ever screened
    beside the other.

    Chunks are therefore kept per ``(namespace, message id)`` until one of
    these says the message is finished:

    * the chunk itself -- LangChain marks the final chunk
      (``chunk_position == "last"``), a provider's ``finish_reason`` /
      ``stop_reason`` arrives in ``response_metadata``, or the item is a whole
      message rather than a chunk;
    * an ``updates`` event naming the node that produced it, or a ``values``
      event in its namespace (the step that produced it has ended);
    * the end of the stream.

    Items are released strictly in arrival order: an unfinished message at the
    head holds back whatever arrived after it. With a model that never marks
    its last chunk and a stream of ``messages`` alone, that means delivery at
    the end of the stream -- screening before release costs latency, and this
    is where the cost is paid.

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
        #: Every item not yet released, in arrival order.
        self._queue: list[_Held] = []
        #: Unfinished messages: key -> (producing node, its held chunks).
        self._open: dict[Any, tuple[Any, list[_Held]]] = {}

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
        """Take one streamed item; return the items that may now be released."""
        ns, mode, data = self._classify(item)
        if mode == "messages" and isinstance(data, tuple) and data:
            chunk = data[0]
            metadata = data[1] if len(data) > 1 else None
            key = (_hashable(ns), getattr(chunk, "id", None))
            held = _Held(item)
            self._queue.append(held)
            node = metadata.get("langgraph_node") if isinstance(metadata, dict) else None
            self._open.setdefault(key, (node, []))[1].append(held)
            if _is_final_chunk(chunk):
                self._close(key)
            return self._drain()
        # Not a message chunk: screened on its own, and released in its turn.
        self._agent._screen_items([data])
        for key in self._finished_by(_hashable(ns), mode, data):
            self._close(key)
        self._queue.append(_Held(item, ready=True))
        return self._drain()

    def flush(self) -> list[Any]:
        """The stream has ended, so every message still open is complete."""
        for key in list(self._open):
            self._close(key)
        return self._drain()

    def _finished_by(self, ns: Any, mode: Any, data: Any) -> list[Any]:
        """Open messages that a non-message event shows to be complete."""
        if mode == "values":
            return [key for key in self._open if key[0] == ns]
        if mode == "updates" and isinstance(data, dict):
            return [
                key
                for key, (node, _) in self._open.items()
                if key[0] == ns and node is not None and node in data
            ]
        return []

    def _close(self, key: Any) -> None:
        """Screen a finished message as a whole and mark its chunks releasable.

        Raises:
            SecurityBlockedError: If the message is blocked; none of its chunks
                has been released, and none will be.
        """
        _, held = self._open.pop(key)
        chunks = [self._classify(h.item)[2][0] for h in held]
        # The chunks' text, joined: a pattern split across tokens is matched.
        joined = "".join(c for c in (_chunk_text(chunk) for chunk in chunks) if c)
        # A streamed tool call carries its arguments the same way -- a JSON
        # string in fragments, with `content` empty -- so they are assembled
        # per call before screening, or each fragment is judged alone.
        self._agent._screen_items([joined, *_tool_call_arguments(chunks), chunks])
        for h in held:
            h.ready = True

    def _drain(self) -> list[Any]:
        """Release the screened items at the head of the queue, in order."""
        released = 0
        while released < len(self._queue) and self._queue[released].ready:
            released += 1
        ready, self._queue = self._queue[:released], self._queue[released:]
        return [h.item for h in ready]


def _hashable(ns: Any) -> Any:
    """A namespace as a dict-key-safe value (LangGraph yields tuples; v2, lists)."""
    return tuple(ns) if isinstance(ns, list) else ns


def _is_final_chunk(chunk: Any) -> bool:
    """Whether a ``messages`` datum is known to be the end of its message.

    Positive evidence only. An object that says nothing about its position is
    treated as an unfinished chunk, which costs latency; the opposite guess
    would release a message in pieces.
    """
    if getattr(chunk, "chunk_position", None) == "last":
        return True
    if type(chunk).__name__ in _WHOLE_MESSAGE_TYPES:
        return True
    metadata = getattr(chunk, "response_metadata", None)
    if isinstance(metadata, dict):
        return bool(metadata.get("finish_reason") or metadata.get("stop_reason"))
    return False


def _tool_call_arguments(chunks: list[Any]) -> list[str]:
    """Each streamed tool call's argument string, reassembled from its fragments.

    LangChain streams a tool call as ``tool_call_chunks`` entries whose ``args``
    are consecutive pieces of one JSON string, tied together by ``index`` (the
    ``id`` and ``name`` arrive on the first piece only). Calls are kept apart:
    joining the fragments of two parallel calls would screen text no tool was
    ever going to receive.
    """
    assembled: dict[Any, list[str]] = {}
    for chunk in chunks:
        pieces = getattr(chunk, "tool_call_chunks", None)
        if not isinstance(pieces, (list, tuple)):
            continue
        for position, piece in enumerate(pieces):
            fragment = _piece_field(piece, "args")
            index = _piece_field(piece, "index")
            call_id = _piece_field(piece, "id")
            if not isinstance(fragment, str) or not fragment:
                continue
            # `index` is the stable key; a provider that omits it sends an id on
            # every piece, and failing both the position within the chunk is
            # the best evidence of which call a fragment continues.
            key = ("index", index) if index is not None else ("id", call_id or position)
            assembled.setdefault(key, []).append(fragment)
    return ["".join(fragments) for fragments in assembled.values()]


def _piece_field(piece: Any, name: str) -> Any:
    """A field of one ``tool_call_chunks`` entry, a dict or an object."""
    if isinstance(piece, dict):
        return piece.get(name)
    try:
        return getattr(piece, name, None)
    except Exception:
        return None


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    return "".join(_extract_texts(content)) if content is not None else ""


__all__ = ["LangGraphAdapter", "_extract_texts"]
