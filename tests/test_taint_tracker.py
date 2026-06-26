"""Tests for the TaintTracker."""

from adapt_agent.security.taint_tracker import (
    TaintLevel,
    TaintSource,
    TaintTracker,
)


def test_register_source_and_mark_tainted():
    tracker = TaintTracker()

    source = tracker.register_source(
        source_id="s1",
        source_type="user_input",
        level=TaintLevel.HIGH,
    )
    assert isinstance(source, TaintSource)
    assert source.source_id == "s1"
    assert source.level is TaintLevel.HIGH

    assert tracker.is_tainted("d1") is False

    tracker.mark_tainted("d1", ["s1"])

    assert tracker.is_tainted("d1") is True
    assert tracker.get_taint_level("d1") is TaintLevel.HIGH

    sources = tracker.get_taint_sources("d1")
    assert len(sources) == 1
    assert sources[0] is source


def test_get_taint_level_unknown_data_is_untainted():
    tracker = TaintTracker()
    assert tracker.get_taint_level("nope") is TaintLevel.UNTAINTED
    assert tracker.is_tainted("nope") is False
    assert tracker.get_taint_sources("nope") == []


def test_get_taint_level_returns_highest_among_sources():
    tracker = TaintTracker()
    tracker.register_source("low", "user_input", TaintLevel.LOW)
    tracker.register_source("high", "external_api", TaintLevel.HIGH)
    tracker.register_source("med", "file", TaintLevel.MEDIUM)

    tracker.mark_tainted("d1", ["low", "med", "high"])

    assert tracker.get_taint_level("d1") is TaintLevel.HIGH


def test_get_taint_level_critical_short_circuits():
    tracker = TaintTracker()
    tracker.register_source("low", "user_input", TaintLevel.LOW)
    tracker.register_source("crit", "exploit", TaintLevel.CRITICAL)

    tracker.mark_tainted("d1", ["low", "crit"])

    assert tracker.get_taint_level("d1") is TaintLevel.CRITICAL


def test_propagate_taint_copies_sources_and_records():
    tracker = TaintTracker()
    tracker.register_source("s1", "user_input", TaintLevel.HIGH)
    tracker.mark_tainted("from", ["s1"])

    tracker.propagate_taint("from", "to", operation="concat")

    assert tracker.is_tainted("to") is True
    assert tracker.get_taint_level("to") is TaintLevel.HIGH

    flow = tracker.get_taint_flow("to")
    assert len(flow) == 1
    assert flow[0]["from"] == "from"
    assert flow[0]["to"] == "to"
    assert flow[0]["operation"] == "concat"
    assert "s1" in flow[0]["sources"]


def test_propagate_taint_from_untainted_is_noop():
    tracker = TaintTracker()

    tracker.propagate_taint("clean", "to")

    assert tracker.is_tainted("to") is False
    assert tracker.get_taint_flow("to") == []
    assert tracker.get_stats()["propagation_count"] == 0


def test_get_taint_flow_filters_by_to():
    tracker = TaintTracker()
    tracker.register_source("s1", "user_input", TaintLevel.MEDIUM)
    tracker.mark_tainted("a", ["s1"])

    tracker.propagate_taint("a", "b")
    tracker.propagate_taint("a", "c")

    flow_b = tracker.get_taint_flow("b")
    assert len(flow_b) == 1
    assert all(p["to"] == "b" for p in flow_b)

    flow_c = tracker.get_taint_flow("c")
    assert len(flow_c) == 1
    assert all(p["to"] == "c" for p in flow_c)


def test_sanitize_removes_taint():
    tracker = TaintTracker()
    tracker.register_source("s1", "user_input", TaintLevel.HIGH)
    tracker.mark_tainted("d1", ["s1"])
    assert tracker.is_tainted("d1") is True

    tracker.sanitize("d1")

    assert tracker.is_tainted("d1") is False
    assert tracker.get_taint_level("d1") is TaintLevel.UNTAINTED

    # Sanitizing unknown data does not raise.
    tracker.sanitize("never-existed")


def test_get_stats():
    tracker = TaintTracker()
    tracker.register_source("s1", "user_input", TaintLevel.HIGH)
    tracker.register_source("s2", "external_api", TaintLevel.LOW)

    tracker.mark_tainted("d1", ["s1"])
    tracker.mark_tainted("d2", ["s1", "s2"])
    tracker.propagate_taint("d1", "d3")

    stats = tracker.get_stats()

    assert stats["total_sources"] == 2
    # d1, d2, d3 are all tracked as tainted data.
    assert stats["tainted_data_count"] == 3
    assert stats["propagation_count"] == 1

    dist = stats["taint_level_distribution"]
    # Distinct sources are counted once regardless of how many data items they
    # taint: s1 (HIGH) and s2 (LOW) each appear once.
    assert dist["high"] == 1
    assert dist["low"] == 1


def test_max_tracked_items_bounds_sources():
    tracker = TaintTracker(max_tracked_items=3)

    for i in range(10):
        tracker.register_source(f"s{i}", "user_input", TaintLevel.LOW)

    assert len(tracker._taint_sources) <= 3


def test_evicted_source_is_fail_closed_to_max():
    """A source referenced by live tainted data but missing from the registry
    must be treated as the configured max taint level, not UNTAINTED."""
    tracker = TaintTracker()
    tracker.register_source("s1", "user_input", TaintLevel.CRITICAL)
    tracker.mark_tainted("d1", ["s1"])
    assert tracker.get_taint_level("d1") is TaintLevel.CRITICAL

    # Simulate the source being evicted/lost while data still references it.
    del tracker._taint_sources["s1"]

    # FAIL-CLOSED: still tainted, and at the configured maximum (CRITICAL).
    assert tracker.is_tainted("d1") is True
    assert tracker.get_taint_level("d1") is TaintLevel.CRITICAL

    sources = tracker.get_taint_sources("d1")
    assert len(sources) == 1
    assert sources[0].level is TaintLevel.CRITICAL
    assert sources[0].metadata.get("fail_closed") is True

    dist = tracker.get_stats()["taint_level_distribution"]
    assert dist.get("critical") == 1


def test_unknown_source_level_is_configurable():
    tracker = TaintTracker(unknown_source_level=TaintLevel.HIGH)
    tracker.mark_tainted("d1", ["ghost"])  # never registered
    assert tracker.get_taint_level("d1") is TaintLevel.HIGH


def test_eviction_skips_referenced_sources():
    """Sources still referenced by live tainted data are not evicted; an
    unreferenced one is dropped instead."""
    tracker = TaintTracker(max_tracked_items=2)
    tracker.register_source("ref1", "user_input", TaintLevel.HIGH)
    tracker.register_source("ref2", "user_input", TaintLevel.HIGH)
    # Both are referenced by live data.
    tracker.mark_tainted("d1", ["ref1", "ref2"])

    # Registering a third (unreferenced) source exceeds the cap. An unreferenced
    # source must be the victim, never a referenced one.
    tracker.register_source("unref", "user_input", TaintLevel.LOW)

    assert "ref1" in tracker._taint_sources
    assert "ref2" in tracker._taint_sources
    # The referenced data still resolves to its true level, not a downgrade.
    assert tracker.get_taint_level("d1") is TaintLevel.HIGH


def test_propagation_buffer_is_deque():
    from collections import deque

    tracker = TaintTracker(max_propagations=2)
    assert isinstance(tracker._taint_propagation, deque)
    tracker.register_source("s1", "user_input", TaintLevel.LOW)
    tracker.mark_tainted("src", ["s1"])
    for i in range(5):
        tracker.propagate_taint("src", f"dst{i}")
    assert len(tracker._taint_propagation) == 2


def test_max_tracked_items_bounds_tainted_data():
    tracker = TaintTracker(max_tracked_items=3)
    tracker.register_source("s1", "user_input", TaintLevel.LOW)

    for i in range(10):
        tracker.mark_tainted(f"d{i}", ["s1"])

    assert len(tracker._tainted_data) <= 3


def test_max_propagations_bounds_records():
    tracker = TaintTracker(max_propagations=3)
    tracker.register_source("s1", "user_input", TaintLevel.LOW)
    tracker.mark_tainted("src", ["s1"])

    for i in range(10):
        tracker.propagate_taint("src", f"dst{i}")

    assert len(tracker._taint_propagation) <= 3


def test_get_stats_counts_each_source_once():
    """A single source tainting many data items is counted once per level."""
    tracker = TaintTracker()
    tracker.register_source("s1", "user_input", TaintLevel.HIGH)

    for i in range(5):
        tracker.mark_tainted(f"d{i}", ["s1"])

    dist = tracker.get_stats()["taint_level_distribution"]
    assert dist["high"] == 1
