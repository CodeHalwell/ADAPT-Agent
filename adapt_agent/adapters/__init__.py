"""Framework adapters for ADAPT-Agent.

Adapters integrate ADAPT-Agent's security and governance primitives with
third-party LLM agent frameworks.

Support matrix:

==================  ============  =====================================
Framework           Status        Class
==================  ============  =====================================
LangGraph           Supported     :class:`LangGraphAdapter`
Semantic Kernel     Experimental  :class:`SemanticKernelAdapter`
CrewAI              Experimental  :class:`CrewAIAdapter`
==================  ============  =====================================

Importing an adapter class never imports the underlying framework; the
framework is only imported lazily when you actually wrap an agent, so
``adapt_agent`` stays import-safe without optional dependencies installed.
"""

from adapt_agent.adapters.base import BaseAdapter
from adapt_agent.adapters.crewai import CrewAIAdapter
from adapt_agent.adapters.langgraph import LangGraphAdapter
from adapt_agent.adapters.semantic_kernel import SemanticKernelAdapter

__all__ = [
    "BaseAdapter",
    "LangGraphAdapter",
    "SemanticKernelAdapter",
    "CrewAIAdapter",
]
