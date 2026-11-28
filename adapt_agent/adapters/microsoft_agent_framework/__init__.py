"""Microsoft Agent Framework adapter for ADAPT-Agent.

`Microsoft Agent Framework <https://github.com/microsoft/agent-framework>`_ is
Microsoft's unified successor to Semantic Kernel and AutoGen. Its runnable
object is an ``agent_framework.Agent`` (called ``ChatAgent`` before 1.0),
built from a chat client -- ``Agent(client, instructions, tools=[...])`` or
``client.as_agent(...)``. ``agent.run(messages, ...)`` returns an awaitable
that resolves to an ``AgentResponse`` with the final text on ``.text``, or,
with ``stream=True``, a ``ResponseStream`` of ``AgentResponseUpdate``.

This adapter wraps an ``Agent`` (or a workflow's ``as_agent()``, or anything
exposing a callable ``run`` / ``run_sync``) and applies ADAPT-Agent's
governance pipeline at the boundary of the run. For governance *inside* the run
-- every model call and every tool call -- install
:func:`adapt_agent.integrations.agent_framework.governance_bundle` on the agent;
the two compose.

Extra keyword arguments to ``execute`` / ``aexecute`` are forwarded to
``agent.run``: ``session=`` (an ``AgentSession``, required to resume an
approval), ``stream=True``, ``middleware=[...]`` (run-level middleware),
``tools=``, ``options=`` and friends.

Approvals round-trip through the adapter. A run that paused on a tool approval
returns ``{"result": <AgentResponse>, "user_input_requests": [...]}``; resume by
passing the responses back as messages on the same session::

    paused = guarded.execute({"prompt": "deploy"}, session=session)
    answers = [r.to_function_approval_response(True) for r in paused["user_input_requests"]]
    guarded.execute({"messages": [Message(role="user", contents=answers)]}, session=session)

A payload whose ``messages`` hold framework objects (``Message`` / ``Content``)
is passed to ``run`` as a message list; a payload of plain ``{"role",
"content"}`` dicts is still reduced to its latest user prompt, as before.
Importing this module never imports ``agent_framework``.

Example
-------
>>> from agent_framework import Agent                              # doctest: +SKIP
>>> agent = Agent(client, "You are concise.", name="assistant")    # doctest: +SKIP
>>> from adapt_agent.adapters import MicrosoftAgentFrameworkAdapter
>>> from adapt_agent.security import Firewall
>>> adapter = MicrosoftAgentFrameworkAdapter(firewall=Firewall())
>>> guarded = adapter.wrap_agent(agent)                            # doctest: +SKIP
>>> guarded.execute({"messages": [{"role": "user", "content": "hi"}]})  # doctest: +SKIP
"""

import inspect
from collections.abc import AsyncIterable, Generator
from typing import Any

from adapt_agent.adapters._governed import (
    GovernedAdapter,
    GovernedEnvelope,
    Runner,
    _close_unawaited,
    _extract_prompt,
)


def _is_framework_object(value: Any) -> bool:
    """A message entry that is not a plain dict -- a MAF ``Message``/``Content``/str."""
    return not isinstance(value, dict)


def _to_message(value: Any) -> Any:
    """Convert a ``{"role", "content"}`` dict to a MAF ``Message`` where possible.

    ``agent.run`` accepts strings, ``Content`` and ``Message`` objects only; a
    dict mixed into a list of real messages would be rejected. Anything that is
    not such a dict, or any dict when MAF is not importable, passes through.
    """
    if not isinstance(value, dict) or "role" not in value:
        return value
    try:
        from agent_framework import Message
    except Exception:  # pragma: no cover - exercised without the SDK
        return value
    content = value.get("content")
    contents = content if isinstance(content, list) else [content if content is not None else ""]
    return Message(role=value["role"], contents=contents)


class _SurfacedRun:
    """Await a MAF run and surface what a caller must act on.

    Wraps the awaitable (or stream) ``agent.run`` returned. On completion the
    result is the bare ``AgentResponse`` -- unchanged for every ordinary run --
    or, when the run paused for approval, an envelope that also carries
    ``user_input_requests``: a paused run's ``.text`` is empty, and a caller
    that reads only the text would never learn there was a request to answer.

    A stream is drained here and its final response returned with the updates,
    so output screening sees the whole response before ``execute`` returns it.

    ``close()`` releases the wrapped coroutine when this object is never
    awaited (``execute`` called from inside an event loop), which would
    otherwise warn "coroutine was never awaited" against the framework.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __await__(self) -> Generator[Any, None, Any]:
        return self._resolve().__await__()

    async def _resolve(self) -> Any:
        inner = self._inner
        finalize = getattr(inner, "get_final_response", None)
        if isinstance(inner, AsyncIterable) and callable(finalize):
            updates = [update async for update in inner]
            final = await finalize()
            envelope = GovernedEnvelope(result=final)
            envelope["updates"] = updates
            return self._with_requests(final, envelope)
        if isinstance(inner, AsyncIterable):
            return [item async for item in inner]
        response = await inner
        return self._with_requests(response, None)

    @staticmethod
    def _with_requests(response: Any, envelope: GovernedEnvelope | None) -> Any:
        requests = list(getattr(response, "user_input_requests", None) or [])
        if not requests:
            return envelope if envelope is not None else response
        envelope = envelope if envelope is not None else GovernedEnvelope(result=response)
        envelope["user_input_requests"] = requests
        return envelope

    def close(self) -> None:
        _close_unawaited(self._inner)


class MicrosoftAgentFrameworkAdapter(GovernedAdapter):
    """Adapter for integrating ADAPT-Agent with Microsoft Agent Framework agents.

    Wrap an ``agent_framework.Agent`` (``ChatAgent`` in pre-1.0 releases), a
    workflow's ``as_agent()``, or any object exposing a callable ``run`` /
    ``run_sync``. See the module docstring for how payloads, keyword arguments
    and approvals map onto ``agent.run``, and
    :class:`~adapt_agent.adapters._governed.GovernedAdapter` for the full
    constructor signature.
    """

    framework_name = "Microsoft Agent Framework"
    run_method_names = ("run", "run_sync")
    operation = "agent_framework.run"

    def _prepare_input(self, payload: Any) -> Any:
        """A message list when the payload carries framework objects, else a prompt.

        Flattening every payload to its latest user string made it impossible
        to resume an approval -- the approval response is a ``Content`` inside a
        ``Message``, with no text to extract -- and dropped any multi-part
        message a caller built.
        """
        messages: Any
        if isinstance(payload, (list, tuple)):
            messages = payload
        elif isinstance(payload, dict):
            messages = payload.get("messages")
        else:
            messages = None
        if isinstance(messages, (list, tuple)) and any(_is_framework_object(m) for m in messages):
            return [_to_message(m) for m in messages]
        if messages is not None and not isinstance(messages, (list, tuple, dict, str)):
            return messages  # a single framework Message
        return _extract_prompt(payload)

    def _call_runner(self, runner: Runner, payload: Any, **run_kwargs: Any) -> Any:
        result = runner(self._prepare_input(payload), **run_kwargs)
        if inspect.isawaitable(result) or isinstance(result, AsyncIterable):
            return _SurfacedRun(result)
        return result


__all__ = ["MicrosoftAgentFrameworkAdapter"]
