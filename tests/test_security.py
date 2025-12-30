"""Tests for security functionality."""

import pytest
from adapt_agent.security import Firewall, TaintTracker
from adapt_agent.security.taint_tracker import TaintLevel


def test_firewall_initialization():
    """Test Firewall initialization."""
    firewall = Firewall()
    stats = firewall.get_stats()
    assert stats["total_blocked"] == 0


def test_firewall_blocked_pattern():
    """Test blocking patterns."""
    firewall = Firewall()
    firewall.add_blocked_pattern(r"password")
    
    # Should block
    assert not firewall.check_input("My password is 123")
    
    # Should allow
    assert firewall.check_input("Hello world")


def test_firewall_sanitize():
    """Test content sanitization."""
    firewall = Firewall()
    firewall.add_blocked_pattern(r"\b\d{3}-\d{2}-\d{4}\b")  # SSN pattern
    
    content = "My SSN is 123-45-6789"
    sanitized = firewall.sanitize(content)
    assert "123-45-6789" not in sanitized
    assert "[REDACTED]" in sanitized


def test_taint_tracker_initialization():
    """Test TaintTracker initialization."""
    tracker = TaintTracker()
    assert not tracker.is_tainted("test_data")


def test_taint_tracker_mark_tainted():
    """Test marking data as tainted."""
    tracker = TaintTracker()
    tracker.register_source("source1", "user_input", TaintLevel.HIGH)
    tracker.mark_tainted("data1", ["source1"])
    
    assert tracker.is_tainted("data1")
    assert tracker.get_taint_level("data1") == TaintLevel.HIGH


def test_taint_tracker_propagation():
    """Test taint propagation."""
    tracker = TaintTracker()
    tracker.register_source("source1", "user_input", TaintLevel.MEDIUM)
    tracker.mark_tainted("data1", ["source1"])
    
    # Propagate taint
    tracker.propagate_taint("data1", "data2", "copy")
    
    assert tracker.is_tainted("data2")
    assert tracker.get_taint_level("data2") == TaintLevel.MEDIUM


def test_taint_tracker_sanitize():
    """Test data sanitization."""
    tracker = TaintTracker()
    tracker.register_source("source1", "user_input", TaintLevel.HIGH)
    tracker.mark_tainted("data1", ["source1"])
    
    assert tracker.is_tainted("data1")
    
    tracker.sanitize("data1")
    assert not tracker.is_tainted("data1")
