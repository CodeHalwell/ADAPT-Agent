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
    tm.update_trust_score("agent3", 0.1)  # triggers eviction of oldest

    assert len(tm._trust_scores) == 2
    # Oldest ("agent1") evicted -> back to default
    assert tm.get_trust_score("agent1") == tm.initial_trust
    assert "agent1" not in tm._trust_history
    assert "agent2" in tm._trust_scores
    assert "agent3" in tm._trust_scores
