"""Base adapter interface for framework integrations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from adapt_agent.core.types import Agent, AgentState


class BaseAdapter(ABC):
    """Base class for framework adapters.
    
    Adapters provide integration with different LLM agent frameworks,
    converting framework-specific APIs to ADAPT-Agent's unified interface.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the adapter.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def wrap_agent(self, agent: Any) -> Agent:
        """Wrap a framework-specific agent with ADAPT-Agent capabilities.
        
        Args:
            agent: Framework-specific agent instance
            
        Returns:
            Wrapped agent implementing the Agent protocol
        """
        pass
    
    @abstractmethod
    def extract_state(self, agent: Any) -> AgentState:
        """Extract the current state from a framework-specific agent.
        
        Args:
            agent: Framework-specific agent instance
            
        Returns:
            AgentState representing the current state
        """
        pass
    
    @abstractmethod
    def inject_middleware(self, agent: Any, middleware: Any) -> Any:
        """Inject ADAPT-Agent middleware into a framework-specific agent.
        
        Args:
            agent: Framework-specific agent instance
            middleware: Middleware to inject
            
        Returns:
            Modified agent with middleware
        """
        pass
    
    def validate_agent(self, agent: Any) -> bool:
        """Validate that an agent is compatible with this adapter.
        
        Args:
            agent: Agent to validate
            
        Returns:
            True if compatible, False otherwise
        """
        # Default implementation - should be overridden
        return True
    
    def get_framework_name(self) -> str:
        """Get the name of the framework this adapter supports.
        
        Returns:
            Framework name
        """
        return self.__class__.__name__.replace("Adapter", "")
