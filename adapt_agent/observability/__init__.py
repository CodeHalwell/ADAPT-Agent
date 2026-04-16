"""Observability and monitoring for LLM agents."""

from typing import Any, Dict, List, Optional
from datetime import datetime


class AgentObserver:
    """Provides observability and monitoring for LLM agents.

    Tracks agent execution, logs interactions, and provides
    debugging and monitoring capabilities.
    """

    def __init__(self):
        """Initialize the AgentObserver."""
        # Performance optimization: Store traces in a dictionary for O(1) lookup
        self._traces: Dict[str, Dict[str, Any]] = {}
        self._logs: List[Dict[str, Any]] = []
        self._metrics: Dict[str, List[float]] = {}

    def start_trace(
        self,
        trace_id: str,
        agent_id: str,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Start a new trace.

        Args:
            trace_id: Unique identifier for the trace
            agent_id: Agent identifier
            operation: Operation being traced
            metadata: Optional metadata

        Returns:
            Trace object
        """
        trace = {
            "trace_id": trace_id,
            "agent_id": agent_id,
            "operation": operation,
            "start_time": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "events": [],
            "status": "active",
        }
        self._traces[trace_id] = trace
        return trace

    def end_trace(
        self,
        trace_id: str,
        status: str = "completed",
        result: Optional[Any] = None,
    ) -> None:
        """End a trace.

        Args:
            trace_id: Trace identifier
            status: Final status
            result: Optional result data
        """
        if trace_id in self._traces:
            trace = self._traces[trace_id]
            trace["end_time"] = datetime.utcnow().isoformat()
            trace["status"] = status
            trace["result"] = result

    def log_event(
        self,
        trace_id: str,
        event_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an event within a trace.

        Args:
            trace_id: Trace identifier
            event_type: Type of event
            description: Event description
            metadata: Optional metadata
        """
        if trace_id in self._traces:
            event = {
                "event_type": event_type,
                "description": description,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {},
            }
            self._traces[trace_id]["events"].append(event)

    def log(
        self,
        level: str,
        message: str,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a message.

        Args:
            level: Log level (debug, info, warning, error)
            message: Log message
            agent_id: Optional agent identifier
            metadata: Optional metadata
        """
        log_entry = {
            "level": level,
            "message": message,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self._logs.append(log_entry)

    def record_metric(
        self,
        metric_name: str,
        value: float,
    ) -> None:
        """Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        if metric_name not in self._metrics:
            self._metrics[metric_name] = []
        self._metrics[metric_name].append(value)

    def get_traces(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get traces.

        Args:
            agent_id: Filter by agent ID
            status: Filter by status
            limit: Maximum number of traces to return

        Returns:
            List of traces
        """
        traces = list(self._traces.values())

        if agent_id:
            traces = [t for t in traces if t["agent_id"] == agent_id]

        if status:
            traces = [t for t in traces if t["status"] == status]

        if limit:
            traces = traces[-limit:]

        return traces

    def get_logs(
        self,
        level: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get logs.

        Args:
            level: Filter by log level
            agent_id: Filter by agent ID
            limit: Maximum number of logs to return

        Returns:
            List of log entries
        """
        logs = self._logs

        if level:
            logs = [log for log in logs if log["level"] == level]

        if agent_id:
            logs = [log for log in logs if log.get("agent_id") == agent_id]

        if limit:
            logs = logs[-limit:]

        return logs

    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary of statistics
        """
        if metric_name not in self._metrics or not self._metrics[metric_name]:
            return {}

        values = self._metrics[metric_name]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
        }


__all__ = ["AgentObserver"]
