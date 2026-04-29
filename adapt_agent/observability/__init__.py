"""Observability and monitoring for LLM agents."""

from datetime import datetime, timezone
from typing import Any, Optional


class AgentObserver:
    """Provides observability and monitoring for LLM agents.

    Tracks agent execution, logs interactions, and provides
    debugging and monitoring capabilities.
    """

    def __init__(self, max_logs: int = 1000):
        """Initialize the AgentObserver.

        Args:
            max_logs: Maximum number of logs to store in memory.
        """
        self.max_logs = max_logs
        # ⚡ Bolt: Using a dict instead of list for O(1) trace lookups
        self._traces: dict[str, dict[str, Any]] = {}
        self._logs: list[dict[str, Any]] = []
        self._metrics: dict[str, list[float]] = {}

    def start_trace(
        self,
        trace_id: str,
        agent_id: str,
        operation: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
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
            "start_time": datetime.now(timezone.utc).isoformat(),
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
        # ⚡ Bolt: Replaced O(N) loop with O(1) dictionary lookup
        if trace_id in self._traces:
            trace = self._traces[trace_id]
            trace["end_time"] = datetime.now(timezone.utc).isoformat()
            trace["status"] = status
            trace["result"] = result

    def log_event(
        self,
        trace_id: str,
        event_type: str,
        description: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log an event within a trace.

        Args:
            trace_id: Trace identifier
            event_type: Type of event
            description: Event description
            metadata: Optional metadata
        """
        event = {
            "event_type": event_type,
            "description": description,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        # ⚡ Bolt: Replaced O(N) loop with O(1) dictionary lookup
        if trace_id in self._traces:
            self._traces[trace_id]["events"].append(event)

    def log(
        self,
        level: str,
        message: str,
        agent_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        self._logs.append(log_entry)

        # SECURITY: Prevent unbounded memory growth
        if len(self._logs) > self.max_logs:
            self._logs.pop(0)

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
    ) -> list[dict[str, Any]]:
        """Get traces.

        Args:
            agent_id: Filter by agent ID
            status: Filter by status
            limit: Maximum number of traces to return

        Returns:
            List of traces
        """
        # ⚡ Bolt: Using reversed() with early-exit limits O(N) operations to O(limit)
        if limit is not None:
            results = []
            for t in reversed(self._traces.values()):
                if agent_id and t["agent_id"] != agent_id:
                    continue
                if status and t["status"] != status:
                    continue
                results.append(t)
                if len(results) >= limit:
                    break
            return list(reversed(results))

        traces = list(self._traces.values())
        if agent_id:
            traces = [t for t in traces if t["agent_id"] == agent_id]

        if status:
            traces = [t for t in traces if t["status"] == status]

        return traces

    def get_logs(
        self,
        level: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Get logs.

        Args:
            level: Filter by log level
            agent_id: Filter by agent ID
            limit: Maximum number of logs to return

        Returns:
            List of log entries
        """
        # ⚡ Bolt: Using reversed() with early-exit limits O(N) operations to O(limit)
        if limit is not None:
            results = []
            for log in reversed(self._logs):
                if level and log["level"] != level:
                    continue
                if agent_id and log.get("agent_id") != agent_id:
                    continue
                results.append(log)
                if len(results) >= limit:
                    break
            return list(reversed(results))

        logs = self._logs
        if level:
            logs = [log for log in logs if log["level"] == level]

        if agent_id:
            logs = [log for log in logs if log.get("agent_id") == agent_id]

        return logs

    def get_metric_stats(self, metric_name: str) -> dict[str, float]:
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
