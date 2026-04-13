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

def test_taint_tracker_get_stats():
    """Test getting taint tracking statistics."""
    tracker = TaintTracker()

    # Check initial empty stats
    stats = tracker.get_stats()
    assert stats["total_sources"] == 0
    assert stats["tainted_data_count"] == 0
    assert stats["propagation_count"] == 0
    assert stats["taint_level_distribution"] == {}

    # Register some sources
    tracker.register_source("source1", "user_input", TaintLevel.HIGH)
    tracker.register_source("source2", "external_api", TaintLevel.LOW)
    tracker.register_source("source3", "database", TaintLevel.HIGH)

    # Mark data as tainted
    tracker.mark_tainted("data1", ["source1", "source2"])
    tracker.mark_tainted("data2", ["source3"])

    # Propagate taint
    tracker.propagate_taint("data1", "data3", "copy")

    # Check populated stats
    stats = tracker.get_stats()

    assert stats["total_sources"] == 3
    # data1, data2, data3
    assert stats["tainted_data_count"] == 3
    assert stats["propagation_count"] == 1

    # data1 is tainted by source1 (HIGH) and source2 (LOW)
    # data2 is tainted by source3 (HIGH)
    # data3 is tainted by source1 (HIGH) and source2 (LOW) (from data1)
    # total sources affecting data are:
    # data1: source1, source2
    # data2: source3
    # data3: source1, source2
    # So we have 3 sources contributing to high, 2 contributing to low
    assert stats["taint_level_distribution"][TaintLevel.HIGH.value] == 3
    assert stats["taint_level_distribution"][TaintLevel.LOW.value] == 2
