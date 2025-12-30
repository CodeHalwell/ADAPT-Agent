"""Tests for core functionality."""

import pytest
from adapt_agent.core import TrustManager, PolicyEnforcer, MemorySystem, Middleware


def test_trust_manager_initialization():
    """Test TrustManager initialization."""
    trust_manager = TrustManager()
    assert trust_manager.initial_trust == 0.5
    assert trust_manager.min_trust == 0.0
    assert trust_manager.max_trust == 1.0


def test_trust_manager_get_score():
    """Test getting trust score."""
    trust_manager = TrustManager()
    agent_id = "test_agent"
    score = trust_manager.get_trust_score(agent_id)
    assert score == 0.5  # Initial trust


def test_trust_manager_update_score():
    """Test updating trust score."""
    trust_manager = TrustManager()
    agent_id = "test_agent"
    
    # Increase trust
    new_score = trust_manager.update_trust_score(agent_id, 0.2)
    assert new_score == 0.7
    
    # Decrease trust
    new_score = trust_manager.update_trust_score(agent_id, -0.3)
    assert abs(new_score - 0.4) < 0.0001  # Use approximate comparison for floats


def test_policy_enforcer_add_rule():
    """Test adding a policy rule."""
    enforcer = PolicyEnforcer()
    enforcer.add_rule(
        name="test_rule",
        description="Test rule",
        condition="test",
        action="warn",
        severity="low"
    )
    
    rule = enforcer.get_rule("test_rule")
    assert rule is not None
    assert rule["name"] == "test_rule"


def test_memory_system_short_term():
    """Test short-term memory storage and retrieval."""
    memory = MemorySystem()
    memory.store_short_term("key1", "value1")
    
    value = memory.retrieve("key1")
    assert value == "value1"


def test_memory_system_long_term():
    """Test long-term memory storage and retrieval."""
    memory = MemorySystem()
    memory.store_long_term("key1", "value1")
    
    value = memory.retrieve("key1", from_long_term=True)
    assert value == "value1"


def test_middleware_initialization():
    """Test Middleware initialization."""
    middleware = Middleware()
    assert len(middleware.list_middleware()) == 0
