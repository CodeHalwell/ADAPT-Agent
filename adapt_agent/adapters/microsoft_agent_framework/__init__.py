"""Microsoft Agent Framework adapter for ADAPT-Agent.

`Microsoft Agent Framework <https://github.com/microsoft/agent-framework>`_ is
Microsoft's unified successor to Semantic Kernel and AutoGen. Its primary
runnable object is a ``ChatAgent`` (also exported as ``Agent``), whose
``run(prompt)`` coroutine returns an ``AgentRunResponse`` carrying the final
text on ``.text``.

This adapter wraps a ``ChatAgent`` (or anything exposing a callable ``run`` /
``run_sync``) and applies ADAPT-Agent's governance pipeline. ``run`` is async;
the wrapper resolves the coroutine transparently, so ``execute`` stays
synchronous. Importing this module never imports ``agent_framework``.

Example
-------
>>> from agent_framework.openai import OpenAIChatClient        # doctest: +SKIP
>>> agent = OpenAIChatClient().create_agent(instructions="...")  # doctest: +SKIP
>>> from adapt_agent.adapters import MicrosoftAgentFrameworkAdapter
>>> from adapt_agent.security import Firewall
>>> adapter = MicrosoftAgentFrameworkAdapter(firewall=Firewall())
>>> guarded = adapter.wrap_agent(agent)                          # doctest: +SKIP
>>> guarded.execute({"messages": [{"role": "user", "content": "hi"}]})  # doctest: +SKIP
"""

from typing import Any

from adapt_agent.adapters._governed import GovernedAdapter, _extract_prompt


class MicrosoftAgentFrameworkAdapter(GovernedAdapter):
    """Adapter for integrating ADAPT-Agent with Microsoft Agent Framework agents.

    Wrap a ``ChatAgent`` (or any object exposing a callable ``run`` / ``run_sync``).
    The agent is run from a prompt string derived from the payload's latest user
    message. See :class:`~adapt_agent.adapters._governed.GovernedAdapter` for the
    full constructor signature.
    """

    framework_name = "Microsoft Agent Framework"
    run_method_names = ("run", "run_sync")
    operation = "agent_framework.run"

    def _prepare_input(self, payload: dict[str, Any]) -> Any:
        return _extract_prompt(payload)


__all__ = ["MicrosoftAgentFrameworkAdapter"]
