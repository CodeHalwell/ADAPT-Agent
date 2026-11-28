"""Claude Agent SDK adapter for ADAPT-Agent.

The `Claude Agent SDK <https://github.com/anthropics/claude-agent-sdk-python>`_
exposes ``query(prompt=..., options=...)``, an async generator that yields
message objects (``AssistantMessage`` with ``TextBlock`` content, terminated by
a ``ResultMessage``).

This adapter wraps the ``query`` function (or any callable accepting a
``prompt`` keyword) and applies ADAPT-Agent's governance pipeline. The async
stream is drained to a list of messages, so ``execute`` stays synchronous and
the firewall can scan every text block in the response. Importing this module
never imports ``claude_agent_sdk``.

Example
-------
>>> from claude_agent_sdk import query             # doctest: +SKIP
>>> from adapt_agent.adapters import ClaudeAgentSDKAdapter
>>> from adapt_agent.security import Firewall
>>> adapter = ClaudeAgentSDKAdapter(firewall=Firewall())
>>> guarded = adapter.wrap_agent(query)            # doctest: +SKIP
>>> guarded.execute({"messages": [{"role": "user", "content": "hi"}]})  # doctest: +SKIP
"""

from collections.abc import AsyncIterator, Callable
from typing import Any, cast

from adapt_agent.adapters._governed import GovernedAdapter, Runner, _extract_prompt


def _is_sdk_client(agent: Any) -> bool:
    """Whether ``agent`` looks like a ``ClaudeSDKClient`` (duck-typed, no import).

    A client's ``query`` only *sends* the prompt and returns ``None``; the answer
    arrives from ``receive_response``. Resolving it like the module-level
    ``query`` function would run nothing the adapter can screen.
    """
    return (
        callable(getattr(agent, "receive_response", None))
        and callable(getattr(agent, "query", None))
        and callable(getattr(agent, "connect", None))
    )


def _client_runner(client: Any) -> Callable[..., AsyncIterator[Any]]:
    async def run(*, prompt: str, **run_kwargs: Any) -> AsyncIterator[Any]:
        # ``run_kwargs`` are the client's own ``query`` arguments (``session_id``).
        await client.query(prompt, **run_kwargs)
        async for message in client.receive_response():
            yield message

    return run


class ClaudeAgentSDKAdapter(GovernedAdapter):
    """Adapter for integrating ADAPT-Agent with the Claude Agent SDK.

    Wrap the SDK's ``query`` function (or any callable accepting a ``prompt``
    keyword). The async message stream is drained synchronously. See
    :class:`~adapt_agent.adapters._governed.GovernedAdapter` for the full
    constructor signature.

    What can be wrapped:

    * the module-level ``query`` function (or any callable taking ``prompt=``);
    * a *connected* ``ClaudeSDKClient`` -- the adapter sends the prompt and
      screens everything ``receive_response()`` yields. Use ``aexecute`` for a
      client: its connection is bound to the event loop it was opened in, and
      ``execute`` drains in a fresh one.

    Extra keyword arguments to ``execute``/``aexecute`` are forwarded:
    ``options=ClaudeAgentOptions(...)`` for ``query``, ``session_id=...`` for a
    client. The stream is **drained before anything is returned**, so the
    firewall screens the complete answer and a blocked one is never delivered --
    the price is that no partial output reaches the caller early. Tool calls
    inside the run are governed by
    :func:`~adapt_agent.integrations.claude_agent.governance_hooks`, which you
    put on ``options.hooks``; this adapter sees only the messages that come back.
    """

    framework_name = "Claude Agent SDK"
    run_method_names = ("query", "run")
    operation = "claude_agent.query"

    def _resolve_runner(self, agent: Any) -> Runner | None:
        if _is_sdk_client(agent):
            return cast(Runner, _client_runner(agent))
        return super()._resolve_runner(agent)

    def _prepare_input(self, payload: dict[str, Any]) -> Any:
        return _extract_prompt(payload)

    def _call_runner(self, runner: Runner, payload: dict[str, Any], **run_kwargs: Any) -> Any:
        return runner(prompt=self._prepare_input(payload), **run_kwargs)


__all__ = ["ClaudeAgentSDKAdapter"]
