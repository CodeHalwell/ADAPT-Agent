"""Trust management for LLM agents."""

from __future__ import annotations

from collections import OrderedDict, deque
from datetime import datetime, timezone

from adapt_agent.core.types import AgentState, TrustScore


class TrustManager:
    """Manages trust scores and trust-based decision making for LLM agents.

    The TrustManager evaluates agent behavior, interactions, and outputs
    to assign and update trust scores dynamically.

    Eviction is least-recently-used (LRU): when ``max_agents`` is exceeded the
    least-recently-accessed agent is dropped. This defends against a
    "trust-reset attack" where an adversary churns many throwaway agent IDs to
    push a specifically distrusted agent out of the cache and back to default
    trust. As an additional safeguard, an agent whose score is below
    ``distrust_floor`` is preferentially retained: a more-recently-used but
    higher-scored candidate is evicted instead, so a distrusted agent cannot be
    cheaply flushed.
    """

    def __init__(
        self,
        initial_trust: float = 0.5,
        min_trust: float = 0.0,
        max_trust: float = 1.0,
        max_history: int = 1000,
        max_agents: int = 1000,
        distrust_floor: float = 0.25,
    ):
        """Initialize the TrustManager.

        Args:
            initial_trust: Initial trust score for new agents
            min_trust: Minimum allowed trust score
            max_trust: Maximum allowed trust score
            max_history: Maximum number of trust history entries to store per agent
            max_agents: Maximum number of agents to track in memory
            distrust_floor: Scores strictly below this value are treated as
                "distrusted" and are preferentially retained during eviction so
                they cannot be flushed back to default trust by churning IDs.
        """
        self.initial_trust = initial_trust
        self.min_trust = min_trust
        self.max_trust = max_trust
        self.max_history = max_history
        self.max_agents = max_agents
        self.distrust_floor = distrust_floor
        # OrderedDict gives O(1) LRU ordering via move_to_end.
        self._trust_scores: OrderedDict[str, float] = OrderedDict()
        self._trust_history: dict[str, deque[TrustScore]] = {}

    def get_trust_score(self, agent_id: str) -> float:
        """Get the current trust score for an agent.

        Accessing an agent marks it as recently used for LRU eviction.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            Current trust score (between min_trust and max_trust)
        """
        if agent_id in self._trust_scores:
            self._trust_scores.move_to_end(agent_id)
            return self._trust_scores[agent_id]
        return self.initial_trust

    def update_trust_score(
        self,
        agent_id: str,
        delta: float,
        reason: str = "",
        factors: dict[str, float] | None = None,
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

        # Insert/update and mark as most-recently-used.
        self._trust_scores[agent_id] = new_score
        self._trust_scores.move_to_end(agent_id)

        # Record trust score history
        trust_record: TrustScore = {
            "score": new_score,
            "confidence": 1.0,
            "factors": factors or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # deque(maxlen) bounds history in O(1) without list.pop(0).
        history = self._trust_history.get(agent_id)
        if history is None:
            history = deque(maxlen=self.max_history)
            self._trust_history[agent_id] = history
        history.append(trust_record)

        # SECURITY: Prevent memory exhaustion from unbounded unique agents.
        if len(self._trust_scores) > self.max_agents:
            self._evict_one()

        return new_score

    def _evict_one(self) -> None:
        """Evict a single agent using LRU, protecting distrusted agents.

        Scans candidates from least- to most-recently-used and evicts the first
        one whose score is at or above ``distrust_floor``. Only if every tracked
        agent is below the floor does it fall back to evicting the strict LRU
        entry (so eviction always makes progress and stays bounded).
        """
        victim: str | None = None
        for agent_id, score in self._trust_scores.items():
            if score >= self.distrust_floor:
                victim = agent_id
                break

        if victim is None:
            # Every agent is distrusted; fall back to strict LRU to stay bounded.
            victim = next(iter(self._trust_scores))

        del self._trust_scores[victim]
        self._trust_history.pop(victim, None)

    def evaluate_agent_state(self, agent_id: str, state: AgentState) -> TrustScore:
        """Evaluate an agent's state and calculate a trust score.

        Args:
            agent_id: Unique identifier for the agent
            state: Current agent state

        Returns:
            Trust score calculation with factors
        """
        factors: dict[str, float] = {}

        # Example trust factors (can be extended)
        if "policy_violations" in state:
            violation_penalty = -0.1 * len(state["policy_violations"])
            factors["policy_compliance"] = violation_penalty

        # SECURITY: Do not incorporate unvalidated trust scores from the agent's state
        # to prevent privilege escalation where an agent self-assigns maximum trust.
        # if "trust_score" in state:
        #     factors["self_reported"] = state["trust_score"]

        # Calculate overall score
        current_score = self.get_trust_score(agent_id)
        factor_sum = sum(factors.values())

        trust_score: TrustScore = {
            "score": max(self.min_trust, min(self.max_trust, current_score + factor_sum)),
            "confidence": 0.8,
            "factors": factors,
            "timestamp": datetime.now(timezone.utc).isoformat(),
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

    def get_trust_history(self, agent_id: str) -> list[TrustScore]:
        """Get the trust score history for an agent.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            List of historical trust scores
        """
        history = self._trust_history.get(agent_id)
        if history is None:
            return []
        return list(history)
