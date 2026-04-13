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
    """Test get_stats method of TaintTracker."""
    tracker = TaintTracker()

    # Initial stats
    stats = tracker.get_stats()
    assert stats["total_sources"] == 0
    assert stats["tainted_data_count"] == 0
    assert stats["propagation_count"] == 0
    assert stats["taint_level_distribution"] == {}

    # Register some sources
    tracker.register_source("source1", "user_input", TaintLevel.HIGH)
    tracker.register_source("source2", "external_api", TaintLevel.LOW)

    # Mark tainted
    tracker.mark_tainted("data1", ["source1"])
    tracker.mark_tainted("data2", ["source1", "source2"])

    # Propagate
    tracker.propagate_taint("data1", "data3", "copy")

    # Get updated stats
    stats = tracker.get_stats()
    assert stats["total_sources"] == 2
    assert stats["tainted_data_count"] == 3
    assert stats["propagation_count"] == 1
    assert stats["taint_level_distribution"][TaintLevel.HIGH.value] == 3
    assert stats["taint_level_distribution"][TaintLevel.LOW.value] == 1


def test_taint_tracker_get_taint_flow():
    """Test get_taint_flow method of TaintTracker."""
    tracker = TaintTracker()
    tracker.register_source("source1", "user_input", TaintLevel.MEDIUM)
    tracker.mark_tainted("data1", ["source1"])
    tracker.propagate_taint("data1", "data2", "copy")
    tracker.propagate_taint("data2", "data3", "transform")

    flow = tracker.get_taint_flow("data3")
    assert len(flow) == 1
    assert flow[0]["from"] == "data2"
    assert flow[0]["to"] == "data3"
    assert flow[0]["operation"] == "transform"
    assert "source1" in flow[0]["sources"]


def test_taint_tracker_untainted_data():
    """Test get_taint_level and get_taint_sources for untainted data."""
    tracker = TaintTracker()
    assert tracker.get_taint_level("clean_data") == TaintLevel.UNTAINTED
    assert tracker.get_taint_sources("clean_data") == []


def test_taint_tracker_propagate_untainted():
    """Test propagate_taint for untainted data."""
    tracker = TaintTracker()
    # Shouldn't error or do anything since from_data_id is untainted
    tracker.propagate_taint("clean_data", "target_data")
    assert not tracker.is_tainted("target_data")
    assert tracker.get_taint_flow("target_data") == []


def test_taint_tracker_get_taint_level_deleted_source():
    """Test get_taint_level when a source has been removed (or doesn't exist)."""
    tracker = TaintTracker()
    # Manually insert a taint referencing a non-existent source
    tracker._tainted_data["data1"] = {"non_existent_source"}
    # Should fallback to UNTAINTED
    assert tracker.get_taint_level("data1") == TaintLevel.UNTAINTED


def test_taint_tracker_get_taint_sources_deleted_source():
    """Test get_taint_sources when a source has been removed (or doesn't exist)."""
    tracker = TaintTracker()
    tracker._tainted_data["data1"] = {"non_existent_source"}
    # Should skip non-existent sources
    assert tracker.get_taint_sources("data1") == []


def test_taint_tracker_sanitize_non_existent():
    """Test sanitize for data that doesn't exist."""
    tracker = TaintTracker()
    tracker.sanitize("non_existent_data")
    assert not tracker.is_tainted("non_existent_data")
