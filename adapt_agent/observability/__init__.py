"""Observability and monitoring for LLM agents."""

from datetime import datetime, timezone
from typing import Any, Optional


class AgentObserver:
    """Provides observability and monitoring for LLM agents.

    Tracks agent execution, logs interactions, and provides
    debugging and monitoring capabilities.
    """

    def __init__(
        self,
        max_logs: int = 1000,
        max_traces: int = 1000,
        max_metrics: int = 1000,
        max_events_per_trace: int = 1000,
        max_metric_names: int = 1000,
    ):
        """Initialize the AgentObserver.

        Args:
            max_logs: Maximum number of logs to store in memory.
            max_traces: Maximum number of traces to store in memory.
            max_metrics: Maximum number of metrics to store in memory.
            max_events_per_trace: Maximum number of events to store per trace.
            max_metric_names: Maximum number of metric names to store in memory.
        """
        self.max_logs = max_logs
        self.max_traces = max_traces
        self.max_metrics = max_metrics
        self.max_events_per_trace = max_events_per_trace
        self.max_metric_names = max_metric_names
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

        # SECURITY: Prevent unbounded memory growth
        if len(self._traces) > self.max_traces:
            oldest_trace_id = next(iter(self._traces))
            del self._traces[oldest_trace_id]

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
            # SECURITY: Prevent log poisoning by escaping newlines and carriage returns
            "description": description[:10000].replace("\n", "\\n").replace("\r", "\\r"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        # ⚡ Bolt: Replaced O(N) loop with O(1) dictionary lookup
        if trace_id in self._traces:
            self._traces[trace_id]["events"].append(event)
            # SECURITY: Prevent unbounded memory growth
            if len(self._traces[trace_id]["events"]) > self.max_events_per_trace:
                self._traces[trace_id]["events"].pop(0)

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
            # SECURITY: Prevent log poisoning by escaping newlines and carriage returns
            "message": message[:10000].replace("\n", "\\n").replace("\r", "\\r"),
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
            # SECURITY: Prevent memory exhaustion from unbounded dictionary
            if len(self._metrics) > self.max_metric_names:
                oldest_metric = next(iter(self._metrics))
                del self._metrics[oldest_metric]

        self._metrics[metric_name].append(value)

        # SECURITY: Prevent unbounded memory growth
        if len(self._metrics[metric_name]) > self.max_metrics:
            self._metrics[metric_name].pop(0)

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
            results.reverse()
            return results

        # ⚡ Bolt: Use a single-pass list comprehension with combined conditions to avoid intermediate array allocations and O(N) passes
        if agent_id and status:
            return [
                t
                for t in self._traces.values()
                if t["agent_id"] == agent_id and t["status"] == status
            ]
        if agent_id:
            return [t for t in self._traces.values() if t["agent_id"] == agent_id]
        if status:
            return [t for t in self._traces.values() if t["status"] == status]

        return list(self._traces.values())

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
            results.reverse()
            return results

        # ⚡ Bolt: Use a single-pass list comprehension with combined conditions to avoid intermediate array allocations and O(N) passes
        if level and agent_id:
            return [
                log
                for log in self._logs
                if log["level"] == level and log.get("agent_id") == agent_id
            ]
        if level:
            return [log for log in self._logs if log["level"] == level]
        if agent_id:
            return [log for log in self._logs if log.get("agent_id") == agent_id]

        return list(self._logs)

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
