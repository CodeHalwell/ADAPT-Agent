"""Memory systems for LLM agents."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any


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

        # Short-term memory is a FIFO ring buffer: a deque with maxlen evicts
        # the oldest item in O(1) on append instead of the O(N) ``list.pop(0)``.
        self._short_term_memory: deque[dict[str, Any]] = deque(maxlen=short_term_capacity)
        self._long_term_memory: list[dict[str, Any]] = []
        self._metadata: dict[str, Any] = {}

    def store_short_term(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "access_count": 0,
        }

        # deque(maxlen=...) evicts the oldest item automatically on overflow.
        self._short_term_memory.append(memory_item)

    def store_long_term(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
        access_count: int = 0,
    ) -> None:
        """Store an item in long-term memory.

        Long-term storage is upsert-by-key: storing an existing key updates
        the stored value/metadata in place (and keeps the higher access count)
        rather than appending a duplicate entry. This prevents the same key
        accumulating across repeated consolidations.

        Args:
            key: Key for the memory item
            value: Value to store
            metadata: Optional metadata
            access_count: Initial access count to seed the item with. Used when
                consolidating a short-term item so the count that earned
                consolidation is carried over instead of reset to 0.
        """
        # Upsert: if the key already exists, update it in place.
        for existing in self._long_term_memory:
            if existing["key"] == key:
                existing["value"] = value
                if metadata is not None:
                    existing["metadata"] = metadata
                # Preserve the strongest signal of importance.
                existing["access_count"] = max(existing["access_count"], access_count)
                existing["timestamp"] = datetime.now(timezone.utc).isoformat()
                return

        memory_item = {
            "key": key,
            "value": value,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "access_count": access_count,
        }

        self._long_term_memory.append(memory_item)

        # Maintain capacity by evicting the least-accessed item.
        if len(self._long_term_memory) > self.long_term_capacity:
            min_idx = 0
            min_access = self._long_term_memory[0]["access_count"]
            for i in range(1, len(self._long_term_memory)):
                count = self._long_term_memory[i]["access_count"]
                if count < min_access:
                    min_access = count
                    min_idx = i
            self._long_term_memory.pop(min_idx)

    def retrieve(
        self,
        key: str,
        from_long_term: bool = False,
    ) -> Any | None:
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
                item["last_accessed"] = datetime.now(timezone.utc).isoformat()
                return item["value"]

        return None

    def search(
        self,
        query: str,
        from_long_term: bool = False,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
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
        # ⚡ Bolt: Cache query.lower() outside the loop to avoid redundant string allocations and conversions
        query_lower = query.lower()
        for item in reversed(memory):
            # Simple substring search (can be enhanced with semantic search)
            if query_lower in str(item["value"]).lower():
                item["access_count"] += 1
                results.append(item)
                if len(results) >= limit:
                    break

        return results

    def consolidate(self) -> int:
        """Consolidate short-term memory into long-term memory.

        Moves frequently accessed short-term memories to long-term storage,
        carrying over each item's access count so the signal that earned
        consolidation is preserved (and upsert-by-key avoids duplicates).

        Returns:
            Number of items consolidated
        """
        # Find frequently accessed items
        threshold = 3  # Access count threshold
        consolidated = 0
        new_short_term: deque[dict[str, Any]] = deque(maxlen=self.short_term_capacity)

        # ⚡ Bolt: Replace O(N²) list.remove() in a loop with O(N) list comprehension/rebuilding
        for item in self._short_term_memory:
            if item["access_count"] >= threshold:
                # SECURITY: Use store_long_term instead of append to enforce capacity limits (DoS prevention)
                self.store_long_term(
                    item["key"],
                    item["value"],
                    item.get("metadata"),
                    access_count=item["access_count"],
                )
                consolidated += 1
            else:
                new_short_term.append(item)

        self._short_term_memory = new_short_term

        return consolidated

    def clear_short_term(self) -> None:
        """Clear all short-term memory."""
        self._short_term_memory.clear()

    def clear_long_term(self) -> None:
        """Clear all long-term memory."""
        self._long_term_memory.clear()

    def get_stats(self) -> dict[str, Any]:
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
