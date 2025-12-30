"""Middleware system for LLM agents."""

from typing import Any, Callable, Dict, List, Optional
from functools import wraps


MiddlewareFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


class Middleware:
    """Middleware system for intercepting and modifying agent interactions.
    
    Provides a composable middleware pipeline for pre-processing and
    post-processing agent inputs and outputs.
    """
    
    def __init__(self):
        """Initialize the Middleware system."""
        self._pre_middleware: List[MiddlewareFunc] = []
        self._post_middleware: List[MiddlewareFunc] = []
        self._middleware_metadata: Dict[str, Dict[str, Any]] = {}
    
    def add_pre_middleware(
        self,
        middleware: MiddlewareFunc,
        name: Optional[str] = None,
        priority: int = 0,
    ) -> None:
        """Add middleware to run before agent execution.
        
        Args:
            middleware: Middleware function
            name: Optional name for the middleware
            priority: Priority (higher runs first)
        """
        self._pre_middleware.append(middleware)
        
        # Sort by priority
        self._pre_middleware.sort(
            key=lambda m: self._middleware_metadata.get(m.__name__, {}).get("priority", 0),
            reverse=True,
        )
        
        if name:
            self._middleware_metadata[name] = {
                "type": "pre",
                "priority": priority,
                "function": middleware,
            }
    
    def add_post_middleware(
        self,
        middleware: MiddlewareFunc,
        name: Optional[str] = None,
        priority: int = 0,
    ) -> None:
        """Add middleware to run after agent execution.
        
        Args:
            middleware: Middleware function
            name: Optional name for the middleware
            priority: Priority (higher runs first)
        """
        self._post_middleware.append(middleware)
        
        # Sort by priority
        self._post_middleware.sort(
            key=lambda m: self._middleware_metadata.get(m.__name__, {}).get("priority", 0),
            reverse=True,
        )
        
        if name:
            self._middleware_metadata[name] = {
                "type": "post",
                "priority": priority,
                "function": middleware,
            }
    
    def remove_middleware(self, name: str) -> bool:
        """Remove a middleware by name.
        
        Args:
            name: Name of the middleware to remove
            
        Returns:
            True if removed, False if not found
        """
        if name not in self._middleware_metadata:
            return False
        
        metadata = self._middleware_metadata[name]
        middleware_func = metadata["function"]
        
        if metadata["type"] == "pre":
            self._pre_middleware.remove(middleware_func)
        else:
            self._post_middleware.remove(middleware_func)
        
        del self._middleware_metadata[name]
        return True
    
    def process_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through pre-middleware pipeline.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
        """
        result = data.copy()
        
        for middleware in self._pre_middleware:
            try:
                result = middleware(result)
            except Exception as e:
                # Log error and continue (or handle based on policy)
                print(f"Error in pre-middleware {middleware.__name__}: {e}")
        
        return result
    
    def process_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through post-middleware pipeline.
        
        Args:
            data: Output data to process
            
        Returns:
            Processed data
        """
        result = data.copy()
        
        for middleware in self._post_middleware:
            try:
                result = middleware(result)
            except Exception as e:
                # Log error and continue (or handle based on policy)
                print(f"Error in post-middleware {middleware.__name__}: {e}")
        
        return result
    
    def wrap_function(self, func: Callable) -> Callable:
        """Wrap a function with middleware processing.
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert args/kwargs to dict for middleware
            input_data = {"args": args, "kwargs": kwargs}
            
            # Pre-process
            processed_input = self.process_input(input_data)
            
            # Execute function
            result = func(
                *processed_input["args"],
                **processed_input["kwargs"],
            )
            
            # Post-process
            output_data = {"result": result}
            processed_output = self.process_output(output_data)
            
            return processed_output["result"]
        
        return wrapper
    
    def list_middleware(self) -> List[Dict[str, Any]]:
        """List all registered middleware.
        
        Returns:
            List of middleware metadata
        """
        return [
            {"name": name, **metadata}
            for name, metadata in self._middleware_metadata.items()
        ]
