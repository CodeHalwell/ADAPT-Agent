import pytest
from adapt_agent.core import MemorySystem

def test_long_term_memory_eviction():
    memory = MemorySystem(long_term_capacity=3)

    # Store items
    memory.store_long_term("k1", "v1")
    memory.store_long_term("k2", "v2")
    memory.store_long_term("k3", "v3")

    # Access items to change access_count
    memory.retrieve("k1", from_long_term=True)
    memory.retrieve("k1", from_long_term=True)
    memory.retrieve("k3", from_long_term=True)
    # k2 has 0 access count, k1 has 2, k3 has 1

    # Adding 4th item should evict k2 (least accessed)
    memory.store_long_term("k4", "v4")

    assert memory.retrieve("k2", from_long_term=True) is None
    assert memory.retrieve("k1", from_long_term=True) == "v1"
    assert memory.retrieve("k3", from_long_term=True) == "v3"
    assert memory.retrieve("k4", from_long_term=True) == "v4"
