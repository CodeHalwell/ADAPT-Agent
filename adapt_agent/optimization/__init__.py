"""Optimization tools for LLM agents."""

from typing import Any, Dict, List, Optional


class AgentOptimizer:
    """Optimizes LLM agent performance and efficiency.
    
    Provides tools for analyzing and improving agent execution time,
    token usage, and overall performance.
    """
    
    def __init__(self):
        """Initialize the AgentOptimizer."""
        self._metrics: List[Dict[str, Any]] = []
        self._optimization_suggestions: List[Dict[str, Any]] = []
    
    def analyze_performance(
        self,
        agent_id: str,
        execution_time: float,
        token_usage: Optional[int] = None,
        success: bool = True,
    ) -> Dict[str, Any]:
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
        
        return self._compute_statistics(agent_id)
    
    def suggest_optimizations(self, agent_id: str) -> List[Dict[str, Any]]:
        """Generate optimization suggestions for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Analyze metrics for this agent
        agent_metrics = [m for m in self._metrics if m["agent_id"] == agent_id]
        
        if not agent_metrics:
            return suggestions
        
        # Check for slow execution
        avg_time = sum(m["execution_time"] for m in agent_metrics) / len(agent_metrics)
        if avg_time > 5.0:  # threshold in seconds
            suggestions.append({
                "type": "performance",
                "severity": "medium",
                "suggestion": "Consider caching frequently accessed data or using faster models",
                "metric": "execution_time",
                "value": avg_time,
            })
        
        # Check for high token usage
        if agent_metrics[0].get("token_usage"):
            avg_tokens = sum(m.get("token_usage", 0) for m in agent_metrics) / len(agent_metrics)
            if avg_tokens > 1000:
                suggestions.append({
                    "type": "efficiency",
                    "severity": "low",
                    "suggestion": "High token usage detected. Consider prompt optimization",
                    "metric": "token_usage",
                    "value": avg_tokens,
                })
        
        self._optimization_suggestions.extend(suggestions)
        return suggestions
    
    def _compute_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Compute statistics for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Statistics dictionary
        """
        agent_metrics = [m for m in self._metrics if m["agent_id"] == agent_id]
        
        if not agent_metrics:
            return {}
        
        return {
            "total_executions": len(agent_metrics),
            "avg_execution_time": sum(m["execution_time"] for m in agent_metrics) / len(agent_metrics),
            "success_rate": sum(1 for m in agent_metrics if m["success"]) / len(agent_metrics),
        }


__all__ = ["AgentOptimizer"]
