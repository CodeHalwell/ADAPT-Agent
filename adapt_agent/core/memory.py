"""Memory systems for LLM agents."""

from typing import Any, Dict, List, Optional
from datetime import datetime


class MemorySystem:
    """Manages memory and context for LLM agents.
    
    Provides short-term and long-term memory storage, retrieval,
    and management for agent interactions.
    """
    
    def __init__(
        self,
        short_term_capacity: int = 100,
        long_term_capacity: int = 10000,
    ):
        """Initialize the MemorySystem.
        
        Args:
            short_term_capacity: Maximum items in short-term memory
            long_term_capacity: Maximum items in long-term memory
        """
        self.short_term_capacity = short_term_capacity
        self.long_term_capacity = long_term_capacity
        
        self._short_term_memory: List[Dict[str, Any]] = []
        self._long_term_memory: List[Dict[str, Any]] = []
        self._metadata: Dict[str, Any] = {}
    
    def store_short_term(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store an item in short-term memory.
        
        Args:
            key: Key for the memory item
            value: Value to store
            metadata: Optional metadata
        """
        memory_item = {
            "key": key,
            "value": value,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
            "access_count": 0,
        }
        
        self._short_term_memory.append(memory_item)
        
        # Maintain capacity
        if len(self._short_term_memory) > self.short_term_capacity:
            self._short_term_memory.pop(0)
    
    def store_long_term(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store an item in long-term memory.
        
        Args:
            key: Key for the memory item
            value: Value to store
            metadata: Optional metadata
        """
        memory_item = {
            "key": key,
            "value": value,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
            "access_count": 0,
        }
        
        self._long_term_memory.append(memory_item)
        
        # Maintain capacity
        if len(self._long_term_memory) > self.long_term_capacity:
            # Remove least accessed items
            self._long_term_memory.sort(key=lambda x: x["access_count"])
            self._long_term_memory.pop(0)
    
    def retrieve(
        self,
        key: str,
        from_long_term: bool = False,
    ) -> Optional[Any]:
        """Retrieve an item from memory.
        
        Args:
            key: Key of the item to retrieve
            from_long_term: Whether to search long-term memory
            
        Returns:
            Retrieved value or None if not found
        """
        memory = self._long_term_memory if from_long_term else self._short_term_memory
        
        for item in reversed(memory):
            if item["key"] == key:
                item["access_count"] += 1
                item["last_accessed"] = datetime.utcnow().isoformat()
                return item["value"]
        
        return None
    
    def search(
        self,
        query: str,
        from_long_term: bool = False,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search memory for items matching a query.
        
        Args:
            query: Search query
            from_long_term: Whether to search long-term memory
            limit: Maximum number of results
            
        Returns:
            List of matching memory items
        """
        memory = self._long_term_memory if from_long_term else self._short_term_memory
        
        results = []
        for item in reversed(memory):
            # Simple substring search (can be enhanced with semantic search)
            if query.lower() in str(item["value"]).lower():
                item["access_count"] += 1
                results.append(item)
                if len(results) >= limit:
                    break
        
        return results
    
    def consolidate(self) -> int:
        """Consolidate short-term memory into long-term memory.
        
        Moves frequently accessed short-term memories to long-term storage.
        
        Returns:
            Number of items consolidated
        """
        # Find frequently accessed items
        threshold = 3  # Access count threshold
        consolidated = 0
        
        for item in self._short_term_memory[:]:
            if item["access_count"] >= threshold:
                self._long_term_memory.append(item.copy())
                consolidated += 1
        
        return consolidated
    
    def clear_short_term(self) -> None:
        """Clear all short-term memory."""
        self._short_term_memory.clear()
    
    def clear_long_term(self) -> None:
        """Clear all long-term memory."""
        self._long_term_memory.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        return {
            "short_term_count": len(self._short_term_memory),
            "short_term_capacity": self.short_term_capacity,
            "long_term_count": len(self._long_term_memory),
            "long_term_capacity": self.long_term_capacity,
            "total_items": len(self._short_term_memory) + len(self._long_term_memory),
        }
