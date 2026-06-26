"""Tests for the TrustManager."""

from adapt_agent.core import TrustManager


def test_get_trust_score_default():
    tm = TrustManager(initial_trust=0.5)
    assert tm.get_trust_score("unknown") == 0.5


def test_update_trust_score_clamps_to_max():
    tm = TrustManager(initial_trust=0.5, max_trust=1.0)
    score = tm.update_trust_score("a", 5.0)
    assert score == 1.0


def test_update_trust_score_clamps_to_min():
    tm = TrustManager(initial_trust=0.5, min_trust=0.0)
    score = tm.update_trust_score("a", -5.0)
    assert score == 0.0


def test_update_trust_score_records_history():
    tm = TrustManager()
    tm.update_trust_score("a", 0.1, reason="good", factors={"f": 0.1})
    tm.update_trust_score("a", 0.1)
    history = tm.get_trust_history("a")
    assert len(history) == 2
    assert history[0]["factors"] == {"f": 0.1}
    assert all("timestamp" in h for h in history)


def test_get_trust_history_empty_for_unknown():
    tm = TrustManager()
    assert tm.get_trust_history("nobody") == []


def test_is_trusted_threshold():
    tm = TrustManager(initial_trust=0.5)
    tm.update_trust_score("a", 0.2)  # 0.7
    assert tm.is_trusted("a", threshold=0.6) is True
    assert tm.is_trusted("a", threshold=0.8) is False
    # Exactly at threshold counts as trusted
    assert tm.is_trusted("a", threshold=0.7) is True


def test_evaluate_agent_state_applies_policy_violation_penalty():
    tm = TrustManager(initial_trust=0.5)
    state = {
        "messages": [],
        "context": {},
        "policy_violations": ["r1", "r2"],
    }
    result = tm.evaluate_agent_state("a", state)
    # penalty = -0.1 * 2 = -0.2
    assert result["factors"]["policy_compliance"] == -0.2
    assert abs(result["score"] - 0.3) < 1e-9
    assert result["confidence"] == 0.8


def test_evaluate_agent_state_ignores_self_reported_trust_score():
    """Privilege-escalation guard: self-reported trust_score is ignored."""
    tm = TrustManager(initial_trust=0.5)
    state = {
        "messages": [],
        "context": {},
        "trust_score": 1.0,  # attempt to self-assign max trust
    }
    result = tm.evaluate_agent_state("a", state)
    assert "self_reported" not in result["factors"]
    # Score is based on stored trust (0.5), not the injected value
    assert abs(result["score"] - 0.5) < 1e-9


def test_max_history_bounding():
    tm = TrustManager(max_history=3)
    for _ in range(5):
        tm.update_trust_score("a", 0.0)
    assert len(tm.get_trust_history("a")) == 3


def test_max_agents_bounding_evicts_oldest():
    tm = TrustManager(max_agents=2)
    tm.update_trust_score("agent1", 0.1)
    tm.update_trust_score("agent2", 0.1)
    tm.update_trust_score("agent3", 0.1)  # triggers eviction of LRU

    assert len(tm._trust_scores) == 2
    # Least-recently-used ("agent1") evicted -> back to default
    assert tm.get_trust_score("agent1") == tm.initial_trust
    assert "agent1" not in tm._trust_history
    assert "agent2" in tm._trust_scores
    assert "agent3" in tm._trust_scores


def test_lru_eviction_recently_used_survives():
    """Accessing an agent makes it most-recently-used, so it survives eviction."""
    tm = TrustManager(max_agents=2)
    tm.update_trust_score("a", 0.1)  # a: 0.6
    tm.update_trust_score("b", 0.1)  # b: 0.6
    # Touch "a" so it becomes most-recently-used; "b" is now the LRU.
    assert tm.get_trust_score("a") == 0.6
    tm.update_trust_score("c", 0.1)  # eviction triggered -> evicts "b"

    assert "a" in tm._trust_scores
    assert "c" in tm._trust_scores
    assert "b" not in tm._trust_scores


def test_lru_churn_does_not_reset_distrusted_agent():
    """Trust-reset attack: churning fake IDs must not flush a distrusted agent."""
    tm = TrustManager(initial_trust=0.5, max_agents=3, distrust_floor=0.25)
    # "bad" is firmly distrusted (below the floor).
    tm.update_trust_score("bad", -0.5)  # 0.0
    assert tm.get_trust_score("bad") == 0.0

    # Attacker churns many throwaway IDs trying to evict "bad".
    for i in range(50):
        tm.update_trust_score(f"fake{i}", 0.0)  # each at default 0.5

    # The distrusted agent is preferentially retained at its low score.
    assert "bad" in tm._trust_scores
    assert tm.get_trust_score("bad") == 0.0
    assert len(tm._trust_scores) <= 3


def test_distrust_floor_protects_low_score_over_lru():
    """A below-floor agent is retained even when it is the strict LRU entry."""
    tm = TrustManager(initial_trust=0.5, max_agents=2, distrust_floor=0.25)
    tm.update_trust_score("bad", -0.5)  # 0.0, becomes LRU after next inserts
    tm.update_trust_score("ok1", 0.1)  # 0.6
    tm.update_trust_score("ok2", 0.1)  # 0.6 -> triggers eviction

    # Strict LRU would evict "bad"; distrust floor protects it and evicts a
    # higher-scored candidate instead.
    assert "bad" in tm._trust_scores
    assert tm.get_trust_score("bad") == 0.0
    assert len(tm._trust_scores) == 2


def test_all_distrusted_falls_back_to_lru():
    """If every agent is below the floor, eviction still makes progress (LRU)."""
    tm = TrustManager(initial_trust=0.5, max_agents=2, distrust_floor=0.25)
    tm.update_trust_score("a", -0.5)  # 0.0
    tm.update_trust_score("b", -0.5)  # 0.0
    tm.update_trust_score("c", -0.5)  # 0.0 -> must evict someone

    assert len(tm._trust_scores) == 2
    # "a" was least-recently-used and gets evicted as the fallback.
    assert "a" not in tm._trust_scores


def test_history_uses_bounded_deque():
    from collections import deque

    tm = TrustManager(max_history=2)
    tm.update_trust_score("a", 0.0)
    assert isinstance(tm._trust_history["a"], deque)
    for _ in range(5):
        tm.update_trust_score("a", 0.0)
    assert len(tm.get_trust_history("a")) == 2
    assert isinstance(tm.get_trust_history("a"), list)
