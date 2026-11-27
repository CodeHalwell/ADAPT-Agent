"""Taint tracking for LLM agent data flow."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict, deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


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

    The same promise holds for the *data* map. When it is full, the oldest item
    is evicted -- but not forgotten: it moves to a bounded ``floor`` map that
    remembers the highest level it had, so it keeps reading as tainted at that
    level. If even that map overflows, the tracker is *saturated*: from then on
    any identifier it has no record of reads as tainted at the highest level it
    ever had to forget. That is the deliberate trade-off -- a flood of junk
    identifiers can degrade a long-lived tracker to "everything unknown is
    tainted" (an availability cost; :meth:`reset` clears it), but it can no
    longer make real taint read clean. ``get_stats()["saturated"]`` reports it.

    Levels only ratchet up: a source cannot be re-registered at a lower level
    than it already had, and marking data with no
    sources is a no-op, so neither can be used to wash out taint or to push
    real entries out of the maps.
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
        # data_id -> lowest level the item may read as. Holds items evicted from
        # ``_tainted_data`` and items derived from them.
        self._data_floor: OrderedDict[str, TaintLevel] = OrderedDict()
        # Referenced sources that were evicted, with the fail-closed level they
        # read as; a later registration may not go below it.
        self._evicted_sources: OrderedDict[str, TaintLevel] = OrderedDict()
        # Set once ``_data_floor`` itself overflowed: the highest level lost.
        self._saturated_level: TaintLevel | None = None
        self._lock = threading.RLock()

    def _higher(self, a: TaintLevel, b: TaintLevel) -> TaintLevel:
        return a if self._LEVEL_PRIORITY[a] >= self._LEVEL_PRIORITY[b] else b

    def _remember_floor(self, data_id: str, level: TaintLevel) -> None:
        """Record that ``data_id`` may never read below ``level`` (bounded)."""
        previous = self._data_floor.get(data_id)
        if previous is not None:
            level = self._higher(level, previous)
        self._data_floor[data_id] = level
        if len(self._data_floor) > self.max_tracked_items:
            # Out of room to remember individuals: fall back to a blanket floor.
            _, dropped = self._data_floor.popitem(last=False)
            if self._saturated_level is None:
                logger.warning(
                    "TaintTracker is saturated: taint records were evicted faster than they "
                    "could be remembered, so identifiers it has no record of now read as "
                    "tainted (fail-closed). Call reset() once the session is over."
                )
                self._saturated_level = dropped
            else:
                self._saturated_level = self._higher(self._saturated_level, dropped)

    def reset(self) -> None:
        """Forget every source, taint record and saturation state."""
        with self._lock:
            self._taint_sources.clear()
            self._tainted_data.clear()
            self._data_floor.clear()
            self._evicted_sources.clear()
            self._taint_propagation.clear()
            self._saturated_level = None

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

        Raises:
            ValueError: if ``source_id`` is already registered at a higher level
                (or was evicted while still referenced, at a higher level than
                the one now given). Re-registering would
                silently downgrade every item the source already tainted;
                raise the level, or use a new ``source_id``.
        """
        with self._lock:
            existing = self._taint_sources.get(source_id)
            floor = existing.level if existing is not None else self._evicted_sources.get(source_id)
            if floor is not None and self._LEVEL_PRIORITY[level] < self._LEVEL_PRIORITY[floor]:
                raise ValueError(
                    f"Cannot re-register taint source {source_id!r} at level "
                    f"{level.value!r}: it is already {floor.value!r}, and lowering it would "
                    "downgrade data it has already tainted"
                )
            source = TaintSource(source_id, source_type, level, metadata)
            self._taint_sources[source_id] = source
            self._evicted_sources.pop(source_id, None)

            # SECURITY: Prevent memory exhaustion from an unbounded dictionary. Prefer
            # evicting an unreferenced source (invisible), but the cap must hold even
            # when every source is referenced -- otherwise the DoS bound is defeated.
            if len(self._taint_sources) > self.max_tracked_items:
                self._evict_source(protect=source_id)

            return source

    def _referenced_source_ids(self) -> set[str]:
        """Return the set of source IDs referenced by live tainted data."""
        referenced: set[str] = set()
        for sources in list(self._tainted_data.values()):
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
                # Remember the level it truly had, so that a later registration
                # under the same id cannot quietly lower it.
                self._evicted_sources[source_id] = source.level
                if len(self._evicted_sources) > self.max_tracked_items:
                    self._evicted_sources.popitem(last=False)
                return

    def mark_tainted(
        self,
        data_id: str,
        source_ids: list[str],
    ) -> None:
        """Mark data as tainted by specific sources.

        Marking with no sources is a no-op: it taints nothing, and recording it
        would only use up a slot that real taint could be evicted to make room for.

        Args:
            data_id: Identifier for the data
            source_ids: List of taint source IDs
        """
        source_ids = list(source_ids)
        if not source_ids:
            return
        with self._lock:
            if data_id not in self._tainted_data:
                self._tainted_data[data_id] = set()

            self._tainted_data[data_id].update(source_ids)

            # SECURITY: Prevent memory exhaustion from unbounded dictionary. The
            # evicted item is NOT forgotten (that would read as clean): its level
            # is kept in the bounded floor map.
            if len(self._tainted_data) > self.max_tracked_items:
                oldest_data = next(iter(self._tainted_data))
                self._remember_floor(oldest_data, self._derived_level(oldest_data))
                del self._tainted_data[oldest_data]

    def is_tainted(self, data_id: str) -> bool:
        """Check if data is tainted.

        Args:
            data_id: Identifier for the data

        Returns:
            True if data is tainted, False otherwise
        """
        with self._lock:
            if data_id in self._tainted_data and len(self._tainted_data[data_id]) > 0:
                return True
            return data_id in self._data_floor or self._saturated_level is not None

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
        with self._lock:
            if not self.is_tainted(data_id):
                return TaintLevel.UNTAINTED
            return self._derived_level(data_id)

    def _derived_level(self, data_id: str) -> TaintLevel:
        """Level from live sources, raised to any floor left by eviction."""
        max_level = TaintLevel.UNTAINTED
        for sid in self._tainted_data.get(data_id, ()):
            source = self._taint_sources.get(sid)
            lvl = source.level if source is not None else self.unknown_source_level
            max_level = self._higher(max_level, lvl)
        floor = self._data_floor.get(data_id)
        if floor is not None:
            max_level = self._higher(max_level, floor)
        elif data_id not in self._tainted_data and self._saturated_level is not None:
            max_level = self._higher(max_level, self._saturated_level)
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
        with self._lock:
            if not self.is_tainted(data_id):
                return []

            sources: list[TaintSource] = []
            for sid in self._tainted_data.get(data_id, ()):
                source = self._taint_sources.get(sid)
                if source is None:
                    source = TaintSource(
                        source_id=sid,
                        source_type="unknown",
                        level=self.unknown_source_level,
                        metadata={"fail_closed": True},
                    )
                sources.append(source)
            floor = self._data_floor.get(data_id)
            if floor is None and data_id not in self._tainted_data:
                floor = self._saturated_level
            if floor is not None:
                sources.append(
                    TaintSource(
                        source_id=f"evicted:{data_id}",
                        source_type="evicted",
                        level=floor,
                        metadata={"fail_closed": True},
                    )
                )
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
        with self._lock:
            if not self.is_tainted(from_data_id):
                return

            # Copy taint sources to target. Anything the source only carries as
            # an eviction floor travels as a floor on the target, so propagation
            # cannot be used to wash an evicted item clean.
            source_ids = list(self._tainted_data.get(from_data_id, ()))
            self.mark_tainted(to_data_id, source_ids)
            inherited = self._data_floor.get(from_data_id)
            if inherited is None and from_data_id not in self._tainted_data:
                inherited = self._saturated_level
            if inherited is not None:
                self._remember_floor(to_data_id, inherited)

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
        with self._lock:
            self._tainted_data.pop(data_id, None)
            self._data_floor.pop(data_id, None)

    def get_taint_flow(self, data_id: str) -> list[dict[str, Any]]:
        """Get the taint propagation flow for data.

        Args:
            data_id: Identifier for the data

        Returns:
            List of propagation records leading to this data
        """
        flow = []

        # Find all propagations that led to this data
        for prop in list(self._taint_propagation):
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
        for sources in list(self._tainted_data.values()):
            for source_id in list(sources):
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
            "evicted_data_count": len(self._data_floor),
            "saturated": self._saturated_level is not None,
        }
