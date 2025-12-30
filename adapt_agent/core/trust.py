"""Trust management for LLM agents."""

from typing import Any, Dict, List, Optional
from datetime import datetime

from adapt_agent.core.types import AgentState, TrustScore


class TrustManager:
    """Manages trust scores and trust-based decision making for LLM agents.
    
    The TrustManager evaluates agent behavior, interactions, and outputs
    to assign and update trust scores dynamically.
    """
    
    def __init__(
        self,
        initial_trust: float = 0.5,
        min_trust: float = 0.0,
        max_trust: float = 1.0,
    ):
        """Initialize the TrustManager.
        
        Args:
            initial_trust: Initial trust score for new agents
            min_trust: Minimum allowed trust score
            max_trust: Maximum allowed trust score
        """
        self.initial_trust = initial_trust
        self.min_trust = min_trust
        self.max_trust = max_trust
        self._trust_scores: Dict[str, float] = {}
        self._trust_history: Dict[str, List[TrustScore]] = {}
    
    def get_trust_score(self, agent_id: str) -> float:
        """Get the current trust score for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Current trust score (between min_trust and max_trust)
        """
        return self._trust_scores.get(agent_id, self.initial_trust)
    
    def update_trust_score(
        self,
        agent_id: str,
        delta: float,
        reason: str = "",
        factors: Optional[Dict[str, float]] = None,
    ) -> float:
        """Update the trust score for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            delta: Change in trust score (positive or negative)
            reason: Reason for the trust score update
            factors: Dictionary of factors contributing to the update
            
        Returns:
            Updated trust score
        """
        current_score = self.get_trust_score(agent_id)
        new_score = max(self.min_trust, min(self.max_trust, current_score + delta))
        
        self._trust_scores[agent_id] = new_score
        
        # Record trust score history
        trust_record: TrustScore = {
            "score": new_score,
            "confidence": 1.0,
            "factors": factors or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if agent_id not in self._trust_history:
            self._trust_history[agent_id] = []
        self._trust_history[agent_id].append(trust_record)
        
        return new_score
    
    def evaluate_agent_state(self, agent_id: str, state: AgentState) -> TrustScore:
        """Evaluate an agent's state and calculate a trust score.
        
        Args:
            agent_id: Unique identifier for the agent
            state: Current agent state
            
        Returns:
            Trust score calculation with factors
        """
        factors: Dict[str, float] = {}
        
        # Example trust factors (can be extended)
        if "policy_violations" in state:
            violation_penalty = -0.1 * len(state["policy_violations"])
            factors["policy_compliance"] = violation_penalty
        
        if "trust_score" in state:
            factors["self_reported"] = state["trust_score"]
        
        # Calculate overall score
        current_score = self.get_trust_score(agent_id)
        factor_sum = sum(factors.values())
        
        trust_score: TrustScore = {
            "score": max(self.min_trust, min(self.max_trust, current_score + factor_sum)),
            "confidence": 0.8,
            "factors": factors,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        return trust_score
    
    def is_trusted(self, agent_id: str, threshold: float = 0.6) -> bool:
        """Check if an agent meets the trust threshold.
        
        Args:
            agent_id: Unique identifier for the agent
            threshold: Minimum trust score required
            
        Returns:
            True if agent is trusted, False otherwise
        """
        return self.get_trust_score(agent_id) >= threshold
    
    def get_trust_history(self, agent_id: str) -> List[TrustScore]:
        """Get the trust score history for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            List of historical trust scores
        """
        return self._trust_history.get(agent_id, [])
