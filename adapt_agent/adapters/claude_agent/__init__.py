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

from typing import Any

from adapt_agent.adapters._governed import GovernedAdapter, Runner, _extract_prompt


class ClaudeAgentSDKAdapter(GovernedAdapter):
    """Adapter for integrating ADAPT-Agent with the Claude Agent SDK.

    Wrap the SDK's ``query`` function (or any callable accepting a ``prompt``
    keyword). The async message stream is drained synchronously. See
    :class:`~adapt_agent.adapters._governed.GovernedAdapter` for the full
    constructor signature.
    """

    framework_name = "Claude Agent SDK"
    run_method_names = ("query", "run")
    operation = "claude_agent.query"

    def _prepare_input(self, payload: dict[str, Any]) -> Any:
        return _extract_prompt(payload)

    def _call_runner(self, runner: Runner, payload: dict[str, Any]) -> Any:
        return runner(prompt=self._prepare_input(payload))


__all__ = ["ClaudeAgentSDKAdapter"]
