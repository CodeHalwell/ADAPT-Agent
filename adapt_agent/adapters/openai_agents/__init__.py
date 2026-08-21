"""OpenAI Agents SDK adapter for ADAPT-Agent.

The `OpenAI Agents SDK <https://openai.github.io/openai-agents-python>`_ runs an
``Agent`` through the ``Runner`` class: ``Runner.run_sync(agent, input)`` (sync)
or ``await Runner.run(agent, input)`` (async). The result is a ``RunResult``
whose final text is on ``.final_output``.

You can wrap an OpenAI ``Agent`` directly -- the adapter lazily imports
``agents.Runner`` at execution time and drives it for you -- or wrap any plain
callable ``run(input) -> result`` if you want full control over the run
configuration. Importing this module never imports ``agents``.

Example
-------
>>> from agents import Agent                       # doctest: +SKIP
>>> agent = Agent(name="Assistant", instructions="...")  # doctest: +SKIP
>>> from adapt_agent.adapters import OpenAIAgentsAdapter
>>> from adapt_agent.security import Firewall
>>> adapter = OpenAIAgentsAdapter(firewall=Firewall())
>>> guarded = adapter.wrap_agent(agent)            # doctest: +SKIP
>>> guarded.execute({"messages": [{"role": "user", "content": "hi"}]})  # doctest: +SKIP
"""

from typing import Any, Optional

from adapt_agent.adapters._governed import GovernedAdapter, Runner, _extract_prompt
from adapt_agent.exceptions import MissingDependencyError


class OpenAIAgentsAdapter(GovernedAdapter):
    """Adapter for integrating ADAPT-Agent with the OpenAI Agents SDK.

    Wrap an OpenAI ``Agent`` (driven via a lazily-imported ``Runner``) or any
    plain callable ``run(input) -> result``. See
    :class:`~adapt_agent.adapters._governed.GovernedAdapter` for the full
    constructor signature.
    """

    framework_name = "OpenAI Agents"
    run_method_names = ("run_sync", "run")
    async_run_method_names = ("run", "run_sync")
    operation = "openai_agents.run"

    def _resolve_runner(self, agent: Any) -> Runner | None:
        # An object exposing run_sync/run (e.g. a custom runner) or a plain
        # callable is used as-is.
        runner = super()._resolve_runner(agent)
        if runner is not None:
            return runner
        # Otherwise treat ``agent`` as an OpenAI Agent and drive it through the
        # SDK's Runner, imported lazily so importing this module stays cheap.
        return lambda prompt: self._run_with_sdk(agent, prompt)

    @staticmethod
    def _run_with_sdk(agent: Any, prompt: Any) -> Any:
        try:
            from agents import Runner
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise MissingDependencyError("openai-agents", "openai-agents") from exc
        return Runner.run_sync(agent, prompt)

    def _prepare_input(self, payload: dict[str, Any]) -> Any:
        return _extract_prompt(payload)


__all__ = ["OpenAIAgentsAdapter"]
