"""Optimization tools for LLM agents."""

from typing import Any, Optional


class AgentOptimizer:
    """Optimizes LLM agent performance and efficiency.

    Provides tools for analyzing and improving agent execution time,
    token usage, and overall performance.
    """

    def __init__(self, max_metrics: int = 1000, max_suggestions: int = 1000):
        """Initialize the AgentOptimizer.

        Args:
            max_metrics: Maximum number of performance metrics to store.
            max_suggestions: Maximum number of optimization suggestions to store.
        """
        self.max_metrics = max_metrics
        self.max_suggestions = max_suggestions
        self._metrics: list[dict[str, Any]] = []
        self._optimization_suggestions: list[dict[str, Any]] = []

    def analyze_performance(
        self,
        agent_id: str,
        execution_time: float,
        token_usage: Optional[int] = None,
        success: bool = True,
    ) -> dict[str, Any]:
        """Analyze agent performance metrics.

        Args:
            agent_id: Unique identifier for the agent
            execution_time: Execution time in seconds
            token_usage: Number of tokens used
            success: Whether execution was successful

        Returns:
            Analysis results
        """
        metric = {
            "agent_id": agent_id,
            "execution_time": execution_time,
            "token_usage": token_usage,
            "success": success,
        }
        self._metrics.append(metric)

        # SECURITY: Prevent unbounded memory growth
        if len(self._metrics) > self.max_metrics:
            self._metrics.pop(0)

        return self._compute_statistics(agent_id)

    def suggest_optimizations(self, agent_id: str) -> list[dict[str, Any]]:
        """Generate optimization suggestions for an agent.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # ⚡ Bolt: Single O(N) pass to collect metrics and avoid repeated array traversals
        count = 0
        total_time = 0.0
        total_tokens = 0
        has_tokens = False

        for m in self._metrics:
            if m["agent_id"] == agent_id:
                count += 1
                total_time += m["execution_time"]
                if "token_usage" in m and m["token_usage"] is not None:
                    has_tokens = True
                    total_tokens += m["token_usage"]

        if count == 0:
            return suggestions

        # Check for slow execution
        avg_time = total_time / count
        if avg_time > 5.0:  # threshold in seconds
            suggestions.append(
                {
                    "type": "performance",
                    "severity": "medium",
                    "suggestion": "Consider caching frequently accessed data or using faster models",
                    "metric": "execution_time",
                    "value": avg_time,
                }
            )

        # Check for high token usage
        if has_tokens:
            avg_tokens = total_tokens / count
            if avg_tokens > 1000:
                suggestions.append(
                    {
                        "type": "efficiency",
                        "severity": "low",
                        "suggestion": "High token usage detected. Consider prompt optimization",
                        "metric": "token_usage",
                        "value": avg_tokens,
                    }
                )

        self._optimization_suggestions.extend(suggestions)

        # SECURITY: Prevent unbounded memory growth
        excess = len(self._optimization_suggestions) - self.max_suggestions
        if excess > 0:
            self._optimization_suggestions = self._optimization_suggestions[excess:]

        return suggestions

    def _compute_statistics(self, agent_id: str) -> dict[str, Any]:
        """Compute statistics for an agent.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            Statistics dictionary
        """
        # ⚡ Bolt: Single O(N) pass to avoid repeated list traversals for calculation
        count = 0
        total_time = 0.0
        success_count = 0

        for m in self._metrics:
            if m["agent_id"] == agent_id:
                count += 1
                total_time += m["execution_time"]
                if m.get("success", False):
                    success_count += 1

        if count == 0:
            return {}

        return {
            "total_executions": count,
            "avg_execution_time": total_time / count,
            "success_rate": success_count / count,
        }


__all__ = ["AgentOptimizer"]
