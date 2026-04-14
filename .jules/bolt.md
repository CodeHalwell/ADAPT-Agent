## 2024-05-24 - Performance pattern in list eviction
**Learning:** In performance-critical paths such as memory eviction, a manual "for" loop to find a minimum element in a list of dictionaries is more efficient than using `min()` with a lambda key or `sort()`, as it minimizes Python function call overhead. This was found to be 40-50% faster in simple benchmarks.
**Action:** When needing to evict exactly one element based on a dynamic property from a list, prefer an inline O(N) loop rather than `sort()`+`pop()`.
