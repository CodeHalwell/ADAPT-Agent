"""Taint tracking for LLM agent data flow."""

from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum


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
        metadata: Optional[Dict[str, Any]] = None,
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
        self.timestamp = datetime.utcnow().isoformat()


class TaintTracker:
    """Tracks data taint throughout agent execution.
    
    Implements taint tracking to identify and monitor potentially
    unsafe or untrusted data as it flows through the agent system.
    """
    
    def __init__(self):
        """Initialize the TaintTracker."""
        self._taint_sources: Dict[str, TaintSource] = {}
        self._tainted_data: Dict[str, Set[str]] = {}  # data_id -> set of source_ids
        self._taint_propagation: List[Dict[str, Any]] = []
    
    def register_source(
        self,
        source_id: str,
        source_type: str,
        level: TaintLevel = TaintLevel.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
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
        return source
    
    def mark_tainted(
        self,
        data_id: str,
        source_ids: List[str],
    ) -> None:
        """Mark data as tainted by specific sources.
        
        Args:
            data_id: Identifier for the data
            source_ids: List of taint source IDs
        """
        if data_id not in self._tainted_data:
            self._tainted_data[data_id] = set()
        
        self._tainted_data[data_id].update(source_ids)
    
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
        levels = [
            self._taint_sources[sid].level
            for sid in source_ids
            if sid in self._taint_sources
        ]
        
        if not levels:
            return TaintLevel.UNTAINTED
        
        # Return highest severity level
        level_order = [
            TaintLevel.UNTAINTED,
            TaintLevel.LOW,
            TaintLevel.MEDIUM,
            TaintLevel.HIGH,
            TaintLevel.CRITICAL,
        ]
        
        return max(levels, key=lambda l: level_order.index(l))
    
    def get_taint_sources(self, data_id: str) -> List[TaintSource]:
        """Get all taint sources affecting data.
        
        Args:
            data_id: Identifier for the data
            
        Returns:
            List of TaintSource objects
        """
        if not self.is_tainted(data_id):
            return []
        
        source_ids = self._tainted_data[data_id]
        return [
            self._taint_sources[sid]
            for sid in source_ids
            if sid in self._taint_sources
        ]
    
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
        self._taint_propagation.append({
            "from": from_data_id,
            "to": to_data_id,
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
            "sources": source_ids,
        })
    
    def sanitize(self, data_id: str) -> None:
        """Mark data as sanitized (remove taint).
        
        Args:
            data_id: Identifier for the data
        """
        if data_id in self._tainted_data:
            del self._tainted_data[data_id]
    
    def get_taint_flow(self, data_id: str) -> List[Dict[str, Any]]:
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get taint tracking statistics.
        
        Returns:
            Dictionary of statistics
        """
        taint_level_counts = {}
        for sources in self._tainted_data.values():
            for source_id in sources:
                if source_id in self._taint_sources:
                    level = self._taint_sources[source_id].level.value
                    taint_level_counts[level] = taint_level_counts.get(level, 0) + 1
        
        return {
            "total_sources": len(self._taint_sources),
            "tainted_data_count": len(self._tainted_data),
            "propagation_count": len(self._taint_propagation),
            "taint_level_distribution": taint_level_counts,
        }
