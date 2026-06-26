"""Tests for the MemorySystem."""

from adapt_agent.core import MemorySystem


def test_store_and_retrieve_short_term():
    mem = MemorySystem()
    mem.store_short_term("k", "v")
    assert mem.retrieve("k") == "v"
    assert mem.retrieve("missing") is None


def test_store_and_retrieve_long_term():
    mem = MemorySystem()
    mem.store_long_term("k", "v")
    assert mem.retrieve("k", from_long_term=True) == "v"
    # Not in short-term
    assert mem.retrieve("k") is None


def test_retrieve_increments_access_count():
    mem = MemorySystem()
    mem.store_short_term("k", "v")
    mem.retrieve("k")
    mem.retrieve("k")
    item = mem._short_term_memory[0]
    assert item["access_count"] == 2
    assert "last_accessed" in item


def test_search_substring_case_insensitive_with_limit():
    mem = MemorySystem()
    mem.store_short_term("a", "Hello World")
    mem.store_short_term("b", "hello there")
    mem.store_short_term("c", "goodbye")

    results = mem.search("HELLO")
    values = {r["value"] for r in results}
    assert values == {"Hello World", "hello there"}

    # Limit caps the number of results
    mem.store_short_term("d", "hello again")
    limited = mem.search("hello", limit=1)
    assert len(limited) == 1


def test_short_term_capacity_eviction_drops_oldest():
    mem = MemorySystem(short_term_capacity=2)
    mem.store_short_term("first", 1)
    mem.store_short_term("second", 2)
    mem.store_short_term("third", 3)

    assert len(mem._short_term_memory) == 2
    # Oldest ("first") dropped
    assert mem.retrieve("first") is None
    assert mem.retrieve("second") == 2
    assert mem.retrieve("third") == 3


def test_long_term_capacity_eviction_removes_least_accessed():
    mem = MemorySystem(long_term_capacity=2)
    mem.store_long_term("keep", "v1")
    mem.store_long_term("drop", "v2")

    # Give "keep" some access so "drop" is least accessed
    mem.retrieve("keep", from_long_term=True)

    # Adding a third triggers eviction of the least-accessed ("drop")
    mem.store_long_term("new", "v3")

    assert len(mem._long_term_memory) == 2
    assert mem.retrieve("drop", from_long_term=True) is None
    assert mem.retrieve("keep", from_long_term=True) == "v1"
    assert mem.retrieve("new", from_long_term=True) == "v3"


def test_consolidate_moves_frequently_accessed_items():
    mem = MemorySystem()
    mem.store_short_term("hot", "h")
    mem.store_short_term("cold", "c")

    # Access "hot" >= 3 times
    for _ in range(3):
        mem.retrieve("hot")

    moved = mem.consolidate()
    assert moved == 1
    # "hot" moved to long-term, "cold" stays in short-term
    assert mem.retrieve("hot", from_long_term=True) == "h"
    assert mem.retrieve("cold") == "c"
    assert mem.retrieve("hot") is None


def test_consolidate_keeps_below_threshold():
    mem = MemorySystem()
    mem.store_short_term("x", 1)
    mem.retrieve("x")  # access_count = 1 < 3
    moved = mem.consolidate()
    assert moved == 0
    assert len(mem._short_term_memory) == 1
    assert len(mem._long_term_memory) == 0


def test_long_term_upsert_dedupes_by_key():
    """Storing the same key in long-term updates in place, no duplicates."""
    mem = MemorySystem()
    mem.store_long_term("k", "v1")
    mem.store_long_term("k", "v2")

    assert len(mem._long_term_memory) == 1
    assert mem.retrieve("k", from_long_term=True) == "v2"


def test_consolidate_carries_access_count_and_upserts():
    """Consolidation carries the access count and dedupes across runs."""
    mem = MemorySystem()
    mem.store_short_term("hot", "h")

    # Access "hot" 4 times -> access_count = 4 (>= threshold 3).
    for _ in range(4):
        mem.retrieve("hot")

    assert mem.consolidate() == 1

    long_item = mem._long_term_memory[0]
    assert long_item["key"] == "hot"
    # The count that earned consolidation is carried over, not reset to 0.
    assert long_item["access_count"] == 4

    # Re-consolidating the same key does not create a duplicate entry.
    mem.store_short_term("hot", "h2")
    for _ in range(3):
        mem.retrieve("hot")
    assert mem.consolidate() == 1
    assert len(mem._long_term_memory) == 1
    assert mem.retrieve("hot", from_long_term=True) == "h2"


def test_clear_short_and_long_term():
    mem = MemorySystem()
    mem.store_short_term("a", 1)
    mem.store_long_term("b", 2)

    mem.clear_short_term()
    assert len(mem._short_term_memory) == 0
    assert len(mem._long_term_memory) == 1

    mem.clear_long_term()
    assert len(mem._long_term_memory) == 0


def test_get_stats_counts():
    mem = MemorySystem(short_term_capacity=5, long_term_capacity=50)
    mem.store_short_term("a", 1)
    mem.store_short_term("b", 2)
    mem.store_long_term("c", 3)

    stats = mem.get_stats()
    assert stats["short_term_count"] == 2
    assert stats["long_term_count"] == 1
    assert stats["short_term_capacity"] == 5
    assert stats["long_term_capacity"] == 50
    assert stats["total_items"] == 3
