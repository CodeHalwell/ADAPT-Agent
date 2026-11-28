"""Offline plumbing shared by the Microsoft Agent Framework examples.

The examples drive *real* ``agent_framework.Agent`` objects. The only thing
replaced is the model: :class:`OfflineChatClient` is a genuine MAF chat client
(a ``BaseChatClient`` with MAF's function-invocation and chat-middleware
layers, exactly as the shipped clients are built) whose replies come from a
Python function instead of a network call. Everything else -- the tool loop,
middleware, sessions, approvals, streaming, workflows -- is the framework's own
code. Swap in ``OpenAIChatClient(...)`` (``pip install agent-framework-openai``)
and nothing else changes.

``require_framework()`` lets an example exit cleanly, with an install hint,
when the framework is not installed.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from typing import Any


def require_framework() -> None:
    """Exit 0 with a hint when ``agent_framework`` is not installed."""
    if importlib.util.find_spec("agent_framework") is None:
        print(
            "This example drives real Microsoft Agent Framework objects.\n"
            "Install it with: pip install 'adapt-agent[microsoft-agent-framework]'\n"
            "(or: pip install agent-framework-core). Skipping."
        )
        sys.exit(0)


#: A responder sees the messages the model was sent and the run's options
#: (``options["instructions"]`` is the agent's system prompt) and returns the
#: reply: a string, or a list of ``Content`` (e.g. a function call).
Responder = Callable[[list[Any], dict[str, Any]], Any]


def offline_client(responder: Responder) -> Any:
    """Build an :class:`OfflineChatClient` around ``responder``."""
    from agent_framework import (
        BaseChatClient,
        ChatMiddlewareLayer,
        ChatResponse,
        ChatResponseUpdate,
        Content,
        FunctionInvocationLayer,
        Message,
    )

    class OfflineChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):
        """A real MAF chat client whose 'model' is a local function."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.calls: list[list[Any]] = []

        def _inner_get_response(self, *, messages, stream, options, **kwargs):  # type: ignore[override]
            self.calls.append(list(messages))
            reply = responder(list(messages), dict(options or {}))
            contents = reply if isinstance(reply, list) else [Content.from_text(str(reply))]
            if stream:

                async def _updates():
                    for item in contents:
                        if item.type == "text":
                            for word in item.text.split(" "):
                                yield ChatResponseUpdate(
                                    role="assistant", contents=[Content.from_text(word + " ")]
                                )
                        else:
                            yield ChatResponseUpdate(role="assistant", contents=[item])

                return self._build_response_stream(_updates())

            async def _response():
                return ChatResponse(
                    messages=[Message(role="assistant", contents=contents)], response_id="offline"
                )

            return _response()

    return OfflineChatClient()


def scripted(*replies: Any) -> Responder:
    """A responder that plays back ``replies`` in order (then repeats the last)."""
    queue = list(replies)

    def respond(messages: list[Any], options: dict[str, Any]) -> Any:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return respond


def last_user_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if getattr(message, "role", None) == "user" and message.text:
            return str(message.text)
    return ""


def tool_results(messages: list[Any]) -> list[str]:
    """The tool results the model was shown on this call."""
    return [
        str(content.result)
        for message in messages
        for content in message.contents
        if content.type == "function_result"
    ]


def call(name: str, call_id: str = "call-1", **arguments: Any) -> list[Any]:
    """A model reply that asks for one tool call."""
    from agent_framework import Content

    return [Content.from_function_call(call_id=call_id, name=name, arguments=arguments)]


def run_sync(awaitable: Any) -> Any:
    """Drive ``agent.run(...)`` from synchronous code.

    ``Agent.run`` returns an awaitable rather than being a coroutine function,
    and ``asyncio.run`` accepts only coroutines -- hence the wrapper.
    """
    import asyncio

    async def _go() -> Any:
        return await awaitable

    return asyncio.run(_go())
