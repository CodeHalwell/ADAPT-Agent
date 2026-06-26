"""Taint tracking for LLM agent data flow."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any


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
        metadata: dict[str, Any] | None = None,
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

    Fail-closed design: if data is tainted by a ``source_id`` that is not (or no
    longer) present in ``_taint_sources`` -- for example because the source was
    evicted under memory pressure -- the missing source is treated as the
    configured ``unknown_source_level`` (default :data:`TaintLevel.CRITICAL`),
    NOT as untainted. Silently downgrading data referenced by an unknown source
    to UNTAINTED would be a security hole (an attacker could evict a CRITICAL
    source to launder its tainted data). For the same reason, eviction never
    removes a source that is still referenced by live tainted data.
    """

    _LEVEL_PRIORITY = {
        TaintLevel.UNTAINTED: 0,
        TaintLevel.LOW: 1,
        TaintLevel.MEDIUM: 2,
        TaintLevel.HIGH: 3,
        TaintLevel.CRITICAL: 4,
    }

    def __init__(
        self,
        max_propagations: int = 1000,
        max_tracked_items: int = 1000,
        unknown_source_level: TaintLevel = TaintLevel.CRITICAL,
    ):
        """Initialize the TaintTracker.

        Args:
            max_propagations: Maximum number of taint propagations to store in memory.
            max_tracked_items: Maximum number of sources and tainted items to track in memory.
            unknown_source_level: Taint level assigned to a referenced-but-missing
                source (fail-closed). Defaults to the maximum, CRITICAL.
        """
        self.max_propagations = max_propagations
        self.max_tracked_items = max_tracked_items
        self.unknown_source_level = unknown_source_level
        self._taint_sources: dict[str, TaintSource] = {}
        self._tainted_data: dict[str, set[str]] = {}  # data_id -> set of source_ids
        # deque(maxlen) bounds the ring buffer in O(1) without list.pop(0).
        self._taint_propagation: deque[dict[str, Any]] = deque(maxlen=max_propagations)

    def register_source(
        self,
        source_id: str,
        source_type: str,
        level: TaintLevel = TaintLevel.MEDIUM,
        metadata: dict[str, Any] | None = None,
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

        # SECURITY: Prevent memory exhaustion from an unbounded dictionary. Prefer
        # evicting an unreferenced source (invisible), but the cap must hold even
        # when every source is referenced -- otherwise the DoS bound is defeated.
        if len(self._taint_sources) > self.max_tracked_items:
            self._evict_source(protect=source_id)

        return source

    def _referenced_source_ids(self) -> set[str]:
        """Return the set of source IDs referenced by live tainted data."""
        referenced: set[str] = set()
        for sources in self._tainted_data.values():
            referenced.update(sources)
        return referenced

    def _evict_source(self, *, protect: str | None = None) -> None:
        """Evict one source to keep ``_taint_sources`` bounded (DoS cap).

        Eviction order, each step preserving the no-downgrade guarantee:

        1. The oldest source **not** referenced by live tainted data -- removing it
           is invisible to every data item.
        2. Otherwise (all sources referenced) the oldest referenced source whose
           level does not exceed ``unknown_source_level``. A referenced source that
           goes missing reads as ``unknown_source_level`` (fail-closed), so this can
           only *raise* a data item's effective taint, never lower it -- yet it
           still frees a slot so the cap holds.

        Only if no source can be evicted without lowering some item's taint (an
        unusual config where ``unknown_source_level`` sits below a live source's
        level) is the cap left temporarily exceeded, preferring the security
        guarantee over the memory bound.

        Args:
            protect: A source ID to never evict (e.g. the just-registered one).
        """
        referenced = self._referenced_source_ids()
        # 1. Prefer an unreferenced source -- eviction is fully invisible.
        for source_id in self._taint_sources:
            if source_id != protect and source_id not in referenced:
                del self._taint_sources[source_id]
                return
        # 2. Fall back to a referenced source we can drop without a downgrade,
        #    because the fail-closed unknown level is at least as high as its own.
        unknown_priority = self._LEVEL_PRIORITY[self.unknown_source_level]
        for source_id, source in self._taint_sources.items():
            if source_id == protect:
                continue
            if self._LEVEL_PRIORITY[source.level] <= unknown_priority:
                del self._taint_sources[source_id]
                return

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

        # SECURITY: Prevent memory exhaustion from unbounded dictionary.
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

        Fail-closed: any referenced source that is missing from
        ``_taint_sources`` is treated as ``unknown_source_level`` (default
        CRITICAL) rather than untainted.

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
            source = self._taint_sources.get(sid)
            lvl = source.level if source is not None else self.unknown_source_level
            if lvl is TaintLevel.CRITICAL:
                return TaintLevel.CRITICAL
            pri = self._LEVEL_PRIORITY[lvl]
            if pri > max_priority:
                max_level = lvl
                max_priority = pri

        return max_level

    def get_taint_sources(self, data_id: str) -> list[TaintSource]:
        """Get all taint sources affecting data.

        Fail-closed: for any referenced source missing from the registry, a
        synthetic :class:`TaintSource` at ``unknown_source_level`` is returned so
        callers never see tainted data as having zero sources.

        Args:
            data_id: Identifier for the data

        Returns:
            List of TaintSource objects
        """
        if not self.is_tainted(data_id):
            return []

        sources: list[TaintSource] = []
        for sid in self._tainted_data[data_id]:
            source = self._taint_sources.get(sid)
            if source is None:
                source = TaintSource(
                    source_id=sid,
                    source_type="unknown",
                    level=self.unknown_source_level,
                    metadata={"fail_closed": True},
                )
            sources.append(source)
        return sources

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

        # Record propagation (deque(maxlen) bounds this automatically).
        self._taint_propagation.append(
            {
                "from": from_data_id,
                "to": to_data_id,
                "operation": operation,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sources": source_ids,
            }
        )

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

        Fail-closed: distinct sources referenced by live tainted data but missing
        from the registry are counted under ``unknown_source_level`` rather than
        being dropped from the distribution.

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
                level = source.level if source is not None else self.unknown_source_level
                key = level.value
                taint_level_counts[key] = taint_level_counts.get(key, 0) + 1

        return {
            "total_sources": len(self._taint_sources),
            "tainted_data_count": len(self._tainted_data),
            "propagation_count": len(self._taint_propagation),
            "taint_level_distribution": taint_level_counts,
        }
