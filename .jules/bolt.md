## 2024-05-24 - Python lambda overhead in min()
**Learning:** In performance-critical paths such as memory eviction (`adapt_agent/core/memory.py`), a manual `for` loop to find a minimum element in a list of dictionaries is significantly faster (~1.5x) than using `min()` with a `lambda` key, and ~3x faster than `sort()`, due to Python function call overhead.
**Action:** When searching for min/max in large lists of dictionaries (like trace IDs or memory items), use a manual loop instead of built-ins with lambdas to optimize overhead.
