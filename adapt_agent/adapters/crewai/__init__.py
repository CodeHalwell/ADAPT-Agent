"""CrewAI adapter for ADAPT-Agent.

.. warning::
   **Experimental / planned.** This adapter is a placeholder defining the
   intended interface. Its methods raise :class:`NotImplementedError`. For a
   fully implemented integration today, use
   :class:`~adapt_agent.adapters.LangGraphAdapter`. Track progress in the
   project's issue tracker.
"""

from typing import Any

from adapt_agent.adapters.base import BaseAdapter
from adapt_agent.core.types import Agent, AgentState

_PLANNED_MESSAGE = (
    "The CrewAI adapter is experimental and not yet implemented. "
    "Use adapt_agent.adapters.LangGraphAdapter for a supported integration, "
    "or follow https://github.com/CodeHalwell/ADAPT-Agent/issues for status."
)


class CrewAIAdapter(BaseAdapter):
    """Planned adapter for integrating with CrewAI agents.

    This class defines the target interface but does not yet provide a working
    implementation. All operations raise :class:`NotImplementedError`.
    """

    #: Marks this adapter as not production-ready.
    __experimental__ = True

    def wrap_agent(self, agent: Any) -> Agent:
        """Not implemented. See class docstring."""
        raise NotImplementedError(_PLANNED_MESSAGE)

    def extract_state(self, agent: Any) -> AgentState:
        """Not implemented. See class docstring."""
        raise NotImplementedError(_PLANNED_MESSAGE)

    def inject_middleware(self, agent: Any, middleware: Any) -> Any:
        """Not implemented. See class docstring."""
        raise NotImplementedError(_PLANNED_MESSAGE)


__all__ = ["CrewAIAdapter"]
