"""Taint tracking for LLM agent data flow."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class TaintLevel(Enum):
    """Taint level enumeration."""

    UNTAINTED = "untainted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaintSource:
    """Represents a source of taint."""

    def __init__(
        self,
        source_id: str,
        source_type: str,
        level: TaintLevel,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Initialize a TaintSource.

        Args:
            source_id: Unique identifier for the source
            source_type: Type of taint source (e.g., 'user_input', 'external_api')
            level: Taint level
            metadata: Optional metadata
        """
        self.source_id = source_id
        self.source_type = source_type
        self.level = level
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()


class TaintTracker:
    """Tracks data taint throughout agent execution.

    Implements taint tracking to identify and monitor potentially
    unsafe or untrusted data as it flows through the agent system.
    """

    _LEVEL_PRIORITY = {
        TaintLevel.UNTAINTED: 0,
        TaintLevel.LOW: 1,
        TaintLevel.MEDIUM: 2,
        TaintLevel.HIGH: 3,
        TaintLevel.CRITICAL: 4,
    }

    def __init__(self, max_propagations: int = 1000, max_tracked_items: int = 1000):
        """Initialize the TaintTracker.

        Args:
            max_propagations: Maximum number of taint propagations to store in memory.
            max_tracked_items: Maximum number of sources and tainted items to track in memory.
        """
        self.max_propagations = max_propagations
        self.max_tracked_items = max_tracked_items
        self._taint_sources: dict[str, TaintSource] = {}
        self._tainted_data: dict[str, set[str]] = {}  # data_id -> set of source_ids
        self._taint_propagation: list[dict[str, Any]] = []

    def register_source(
        self,
        source_id: str,
        source_type: str,
        level: TaintLevel = TaintLevel.MEDIUM,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TaintSource:
        """Register a new taint source.

        Args:
            source_id: Unique identifier for the source
            source_type: Type of taint source
            level: Taint level
            metadata: Optional metadata

        Returns:
            Created TaintSource
        """
        source = TaintSource(source_id, source_type, level, metadata)
        self._taint_sources[source_id] = source

        # SECURITY: Prevent memory exhaustion from unbounded dictionary
        if len(self._taint_sources) > self.max_tracked_items:
            oldest_source = next(iter(self._taint_sources))
            del self._taint_sources[oldest_source]

        return source

    def mark_tainted(
        self,
        data_id: str,
        source_ids: list[str],
    ) -> None:
        """Mark data as tainted by specific sources.

        Args:
            data_id: Identifier for the data
            source_ids: List of taint source IDs
        """
        if data_id not in self._tainted_data:
            self._tainted_data[data_id] = set()

        self._tainted_data[data_id].update(source_ids)

        # SECURITY: Prevent memory exhaustion from unbounded dictionary
        if len(self._tainted_data) > self.max_tracked_items:
            oldest_data = next(iter(self._tainted_data))
            del self._tainted_data[oldest_data]

    def is_tainted(self, data_id: str) -> bool:
        """Check if data is tainted.

        Args:
            data_id: Identifier for the data

        Returns:
            True if data is tainted, False otherwise
        """
        return data_id in self._tainted_data and len(self._tainted_data[data_id]) > 0

    def get_taint_level(self, data_id: str) -> TaintLevel:
        """Get the highest taint level for data.

        Args:
            data_id: Identifier for the data

        Returns:
            Highest taint level affecting the data
        """
        if not self.is_tainted(data_id):
            return TaintLevel.UNTAINTED

        source_ids = self._tainted_data[data_id]

        max_level = TaintLevel.UNTAINTED
        max_priority = 0

        for sid in source_ids:
            if sid in self._taint_sources:
                lvl = self._taint_sources[sid].level
                if lvl is TaintLevel.CRITICAL:
                    return TaintLevel.CRITICAL
                pri = self._LEVEL_PRIORITY[lvl]
                if pri > max_priority:
                    max_level = lvl
                    max_priority = pri

        return max_level

    def get_taint_sources(self, data_id: str) -> list[TaintSource]:
        """Get all taint sources affecting data.

        Args:
            data_id: Identifier for the data

        Returns:
            List of TaintSource objects
        """
        if not self.is_tainted(data_id):
            return []

        source_ids = self._tainted_data[data_id]
        return [self._taint_sources[sid] for sid in source_ids if sid in self._taint_sources]

    def propagate_taint(
        self,
        from_data_id: str,
        to_data_id: str,
        operation: str = "unknown",
    ) -> None:
        """Propagate taint from one data to another.

        Args:
            from_data_id: Source data identifier
            to_data_id: Target data identifier
            operation: Description of the operation causing propagation
        """
        if not self.is_tainted(from_data_id):
            return

        # Copy taint sources to target
        source_ids = list(self._tainted_data[from_data_id])
        self.mark_tainted(to_data_id, source_ids)

        # Record propagation
        self._taint_propagation.append(
            {
                "from": from_data_id,
                "to": to_data_id,
                "operation": operation,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sources": source_ids,
            }
        )

        # SECURITY: Prevent unbounded memory growth
        if len(self._taint_propagation) > self.max_propagations:
            self._taint_propagation.pop(0)

    def sanitize(self, data_id: str) -> None:
        """Mark data as sanitized (remove taint).

        Args:
            data_id: Identifier for the data
        """
        if data_id in self._tainted_data:
            del self._tainted_data[data_id]

    def get_taint_flow(self, data_id: str) -> list[dict[str, Any]]:
        """Get the taint propagation flow for data.

        Args:
            data_id: Identifier for the data

        Returns:
            List of propagation records leading to this data
        """
        flow = []

        # Find all propagations that led to this data
        for prop in self._taint_propagation:
            if prop["to"] == data_id:
                flow.append(prop)

        return flow

    def get_stats(self) -> dict[str, Any]:
        """Get taint tracking statistics.

        Returns:
            Dictionary of statistics
        """
        # Count each distinct source once by its level. A source that taints
        # multiple data items must not be counted multiple times.
        taint_level_counts: dict[str, int] = {}
        seen_sources: set[str] = set()
        for sources in self._tainted_data.values():
            for source_id in sources:
                if source_id in seen_sources:
                    continue
                seen_sources.add(source_id)
                source = self._taint_sources.get(source_id)
                if source is not None:
                    level = source.level.value
                    taint_level_counts[level] = taint_level_counts.get(level, 0) + 1

        return {
            "total_sources": len(self._taint_sources),
            "tainted_data_count": len(self._tainted_data),
            "propagation_count": len(self._taint_propagation),
            "taint_level_distribution": taint_level_counts,
        }
