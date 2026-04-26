## 2024-05-24 - Python lambda overhead in min()
**Learning:** In performance-critical paths such as memory eviction (`adapt_agent/core/memory.py`), a manual `for` loop to find a minimum element in a list of dictionaries is significantly faster (~1.5x) than using `min()` with a `lambda` key, and ~3x faster than `sort()`, due to Python function call overhead.
**Action:** When searching for min/max in large lists of dictionaries (like trace IDs or memory items), use a manual loop instead of built-ins with lambdas to optimize overhead.

## 2024-05-19 - Enum sorting optimization in critical paths
**Learning:** Python Enums in tight loops are relatively slow to compare. Sorting or finding the max of a list of enums using `list.index` or `lambda` creates overhead.
**Action:** When finding a maximum enum value by its priority in a performance-critical path, use a static class-level dictionary to map Enum members to integer priorities. Combine this with a manual loop for O(1) priority lookups and the ability to add early-exit conditions (short-circuiting on the highest priority).

## 2024-05-25 - Efficient recent item filtering
**Learning:** In `adapt_agent/core/policy.py`, filtering a large chronologically ordered list for a limited number of recent matches using standard list comprehensions (e.g., `[v for v in items if condition][-limit:]`) is highly inefficient because it processes all `N` elements.
**Action:** When searching for a limited number of recent items matching a condition, iterate backwards using `reversed(items)` and early-exit (`break`) once the limit is reached to achieve an `O(L)` operation where `L` is the number of inspected elements.
