## 2024-05-24 - Sorting for Eviction is an Anti-Pattern
**Learning:** In `adapt_agent/core/memory.py`, the long-term memory was sorting the entire list of memory items (`O(n log n)`) just to evict the least accessed item (`O(1)` removal after sorting). Sorting an entire structure just to find the minimum/maximum element is a common but expensive anti-pattern.
**Action:** Replace `list.sort(key=...)` followed by `.pop(0)` or `.pop(-1)` with an `O(n)` linear scan (a simple for loop finding the min/max) when only a single element needs to be evicted.
