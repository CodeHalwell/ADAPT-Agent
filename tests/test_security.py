"""Tests for security functionality."""

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


def test_firewall_custom_filter_fail_closed():
    """Test that firewall fails closed when a custom filter raises an exception."""
    firewall = Firewall()

    def buggy_filter(content: str) -> bool:
        raise ValueError("Something went wrong in the filter")

    firewall.add_custom_filter(buggy_filter)

    # Should block due to fail-closed design
    assert not firewall.check_input("Some valid input")

    stats = firewall.get_stats()
    assert stats["total_blocked"] == 1

    events = firewall.get_security_events()
    assert len(events) == 1
    assert events[0]["event_type"] == "blocked_input"
    assert events[0]["severity"] == "high"
    assert "custom filter error" in events[0]["description"].lower()


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

    # Each distinct source is counted once by its level, regardless of how many
    # data items it taints:
    #   source1 (HIGH) taints data1, data3
    #   source2 (LOW)  taints data1, data3
    #   source3 (HIGH) taints data2
    # So: HIGH -> {source1, source3} = 2, LOW -> {source2} = 1
    assert stats["taint_level_distribution"][TaintLevel.HIGH.value] == 2
    assert stats["taint_level_distribution"][TaintLevel.LOW.value] == 1
