"""Framework adapters for ADAPT-Agent.

Adapters integrate ADAPT-Agent's security and governance primitives with
third-party LLM agent frameworks. Every adapter applies the same pipeline on
each ``execute`` call -- input screening, policy enforcement, middleware, traced
execution, output screening -- regardless of the underlying framework.

Support matrix:

==========================  ==========  =========================================
Framework                   Status      Class
==========================  ==========  =========================================
LangGraph                   Supported   :class:`LangGraphAdapter`
Microsoft Agent Framework   Supported   :class:`MicrosoftAgentFrameworkAdapter`
Google ADK                  Supported   :class:`GoogleADKAdapter`
Pydantic AI                 Supported   :class:`PydanticAIAdapter`
CrewAI                      Supported   :class:`CrewAIAdapter`
OpenAI Agents SDK           Supported   :class:`OpenAIAgentsAdapter`
Claude Agent SDK            Supported   :class:`ClaudeAgentSDKAdapter`
==========================  ==========  =========================================

Importing an adapter class never imports the underlying framework; the framework
is only imported lazily when you actually wrap and run an agent, so
``adapt_agent`` stays import-safe without optional dependencies installed.
"""

from adapt_agent.adapters._governed import GovernedAdapter
from adapt_agent.adapters.base import BaseAdapter
from adapt_agent.adapters.claude_agent import ClaudeAgentSDKAdapter
from adapt_agent.adapters.crewai import CrewAIAdapter
from adapt_agent.adapters.google_adk import GoogleADKAdapter
from adapt_agent.adapters.langgraph import LangGraphAdapter
from adapt_agent.adapters.microsoft_agent_framework import MicrosoftAgentFrameworkAdapter
from adapt_agent.adapters.openai_agents import OpenAIAgentsAdapter
from adapt_agent.adapters.pydantic_ai import PydanticAIAdapter

__all__ = [
    "BaseAdapter",
    "GovernedAdapter",
    "LangGraphAdapter",
    "MicrosoftAgentFrameworkAdapter",
    "GoogleADKAdapter",
    "PydanticAIAdapter",
    "CrewAIAdapter",
    "OpenAIAgentsAdapter",
    "ClaudeAgentSDKAdapter",
]
