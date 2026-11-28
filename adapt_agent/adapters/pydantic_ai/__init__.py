"""Pydantic AI adapter for ADAPT-Agent.

`Pydantic AI <https://ai.pydantic.dev>`_ centres on an ``Agent`` whose
``run_sync(prompt)`` (sync) and ``run(prompt)`` (async) methods return an
``AgentRunResult`` carrying the final output on ``.output``.

This adapter wraps a Pydantic AI ``Agent`` (or anything exposing a callable
``run_sync`` / ``run``) and applies ADAPT-Agent's governance pipeline at the
boundary of the run. ``execute`` drives ``run_sync``; ``aexecute`` awaits
``run``. Importing this module never imports ``pydantic_ai``.

Run arguments
-------------
Keyword arguments given to ``execute`` / ``aexecute`` are forwarded to the run
method: ``deps=``, ``message_history=``, ``model_settings=``,
``usage_limits=``, ``deferred_tool_results=``, ``output_type=``,
``capabilities=`` and the rest. They are configuration, not content -- with
two exceptions. ``message_history`` *is* content: it reaches the model as the
conversation so far, so a poisoned history is as dangerous as a poisoned
prompt. ``deferred_tool_results`` is too: it supplies tool output produced
outside the run, and approvals that may replace a call's arguments. Both are
screened as input, like the payload, before the agent runs.

When a ``message_history`` is given and the payload carries no prompt (no
message, no ``prompt``/``input``/``query``/``text`` key) the run gets
``user_prompt=None`` -- how Pydantic AI continues a conversation, e.g. to
resume deferred tool calls -- rather than the payload's ``repr``.

The adapter sees the run's boundary only. To govern tool calls, model requests
and streams *inside* the run, give the agent the capability from
:func:`adapt_agent.integrations.pydantic_ai.governance_capability` -- through
the adapter too: ``guarded.execute(payload, capabilities=[governance])``.

Example
-------
>>> from pydantic_ai import Agent                       # doctest: +SKIP
>>> agent = Agent("openai:gpt-5.6-luna", deps_type=Deps)      # doctest: +SKIP
>>> from adapt_agent.adapters import PydanticAIAdapter
>>> from adapt_agent.security import Firewall
>>> adapter = PydanticAIAdapter(firewall=Firewall())
>>> guarded = adapter.wrap_agent(agent)                  # doctest: +SKIP
>>> guarded.execute({"messages": [{"role": "user", "content": "hi"}]}, deps=Deps())  # doctest: +SKIP
"""

from typing import Any

from adapt_agent.adapters._governed import GovernedAdapter, Runner, _extract_prompt
from adapt_agent.exceptions import SecurityBlockedError

#: Payload keys :func:`~adapt_agent.core.governance.extract_prompt` reads a prompt from.
_PROMPT_KEYS = ("prompt", "input", "query", "text")


class PydanticAIAdapter(GovernedAdapter):
    """Adapter for integrating ADAPT-Agent with Pydantic AI agents.

    Wrap a Pydantic AI ``Agent`` (or any object exposing a callable
    ``run_sync`` / ``run``). The agent is run from a prompt string derived from
    the payload's latest user message; extra ``execute`` keyword arguments are
    forwarded to the run (see the module docstring, including why
    ``message_history`` is screened). See
    :class:`~adapt_agent.adapters._governed.GovernedAdapter` for the full
    constructor signature.
    """

    framework_name = "Pydantic AI"
    run_method_names = ("run_sync", "run")
    async_run_method_names = ("run", "run_sync")
    operation = "pydantic_ai.run"

    def _prepare_input(self, payload: dict[str, Any]) -> Any:
        return _extract_prompt(payload)

    def _call_runner(self, runner: Runner, payload: dict[str, Any], **run_kwargs: Any) -> Any:
        history = run_kwargs.get("message_history")
        # Content, not configuration: both are what the model reads next. A
        # history is the conversation so far; deferred tool results are tool
        # output supplied from outside the run, and approvals that may carry
        # replacement arguments.
        for content in (history, run_kwargs.get("deferred_tool_results")):
            if content:
                threats = self._screen_input(content)
                if threats and self.block_on_violation:
                    raise SecurityBlockedError("Input blocked by security controls", threats)
        if history and not _has_prompt(payload):
            return runner(None, **run_kwargs)
        return runner(self._prepare_input(payload), **run_kwargs)


def _has_prompt(payload: Any) -> bool:
    """Whether ``payload`` carries a prompt of its own (vs. only continuing a history)."""
    if isinstance(payload, str):
        return bool(payload)
    if not isinstance(payload, dict):
        return payload is not None
    if payload.get("messages"):
        return True
    return any(isinstance(payload.get(key), str) for key in _PROMPT_KEYS)


__all__ = ["PydanticAIAdapter"]
