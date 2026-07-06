"""Runtime performance metrics for LLM agents.

This module provides :class:`AgentOptimizer`, a lightweight collector of
execution-time / token-usage metrics that surfaces heuristic tuning suggestions.
It is complementary to the dataset-driven optimization engine in this package
(:class:`~adapt_agent.optimization.optimizers.Optimizer` and friends): use
``AgentOptimizer`` to watch a *running* agent, and the optimizers to *improve* one
against a golden dataset.
"""

from __future__ import annotations

from collections import deque
from typing import Any


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
        # deque(maxlen=...) is an O(1) ring buffer: it evicts the oldest entry
        # on overflow instead of the O(N) ``list.pop(0)`` shift used before.
        self._metrics: deque[dict[str, Any]] = deque(maxlen=max_metrics)
        # Latest suggestions stored per agent (replace, not append). This keeps
        # the buffer free of the duplicate suggestions that accumulated when the
        # same agent was analyzed repeatedly.
        self._suggestions_by_agent: dict[str, list[dict[str, Any]]] = {}
        self._suggestion_order: deque[str] = deque()
        self._total_suggestions = 0

    @property
    def _optimization_suggestions(self) -> list[dict[str, Any]]:
        """Flattened view of stored suggestions (one block per agent).

        Retained for backward compatibility with callers/tests that inspected
        the previously-public buffer.
        """
        flat: list[dict[str, Any]] = []
        for agent_id in self._suggestion_order:
            flat.extend(self._suggestions_by_agent.get(agent_id, []))
        return flat

    def analyze_performance(
        self,
        agent_id: str,
        execution_time: float,
        token_usage: int | None = None,
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
        # deque(maxlen=...) drops the oldest metric automatically on overflow.
        self._metrics.append(metric)

        return self._compute_statistics(agent_id)

    def suggest_optimizations(self, agent_id: str) -> list[dict[str, Any]]:
        """Generate optimization suggestions for an agent.

        Suggestions are stored per agent and replaced on each call (the latest
        analysis wins), so repeated calls do not accumulate duplicate entries.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            List of optimization suggestions
        """
        suggestions: list[dict[str, Any]] = []

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

        self._store_suggestions(agent_id, suggestions)

        return suggestions

    def _store_suggestions(self, agent_id: str, suggestions: list[dict[str, Any]]) -> None:
        """Store the latest suggestions for an agent (replace, not append).

        Args:
            agent_id: Unique identifier for the agent.
            suggestions: The suggestions produced for this agent.
        """
        if agent_id in self._suggestions_by_agent:
            self._total_suggestions -= len(self._suggestions_by_agent[agent_id])
            self._suggestion_order.remove(agent_id)

        # ⚡ Bolt: Maintain a running total for O(1) eviction condition
        self._suggestions_by_agent[agent_id] = list(suggestions)
        self._total_suggestions += len(self._suggestions_by_agent[agent_id])
        self._suggestion_order.append(agent_id)

        # SECURITY: Prevent unbounded memory growth. Evict whole agents (oldest
        # first) until the total stored suggestion count fits the budget.
        while (
            self._total_suggestions > self.max_suggestions
            and self._suggestion_order
        ):
            oldest = self._suggestion_order.popleft()
            evicted = self._suggestions_by_agent.pop(oldest, [])
            self._total_suggestions -= len(evicted)

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
