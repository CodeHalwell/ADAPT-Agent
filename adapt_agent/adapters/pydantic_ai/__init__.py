"""Pydantic AI adapter for ADAPT-Agent.

`Pydantic AI <https://ai.pydantic.dev>`_ centres on an ``Agent`` whose
``run_sync(prompt)`` (sync) and ``run(prompt)`` (async) methods return an
``AgentRunResult`` carrying the final output on ``.output``.

This adapter wraps a Pydantic AI ``Agent`` (or anything exposing a callable
``run_sync`` / ``run``) and applies ADAPT-Agent's governance pipeline. Importing
this module never imports ``pydantic_ai``.

Example
-------
>>> from pydantic_ai import Agent                       # doctest: +SKIP
>>> agent = Agent("openai:gpt-4o", system_prompt="...")  # doctest: +SKIP
>>> from adapt_agent.adapters import PydanticAIAdapter
>>> from adapt_agent.security import Firewall
>>> adapter = PydanticAIAdapter(firewall=Firewall())
>>> guarded = adapter.wrap_agent(agent)                  # doctest: +SKIP
>>> guarded.execute({"messages": [{"role": "user", "content": "hi"}]})  # doctest: +SKIP
"""

from typing import Any

from adapt_agent.adapters._governed import GovernedAdapter, _extract_prompt


class PydanticAIAdapter(GovernedAdapter):
    """Adapter for integrating ADAPT-Agent with Pydantic AI agents.

    Wrap a Pydantic AI ``Agent`` (or any object exposing a callable
    ``run_sync`` / ``run``). The agent is run from a prompt string derived from
    the payload's latest user message. See
    :class:`~adapt_agent.adapters._governed.GovernedAdapter` for the full
    constructor signature.
    """

    framework_name = "Pydantic AI"
    run_method_names = ("run_sync", "run")
    async_run_method_names = ("run", "run_sync")
    operation = "pydantic_ai.run"

    def _prepare_input(self, payload: dict[str, Any]) -> Any:
        return _extract_prompt(payload)


__all__ = ["PydanticAIAdapter"]
