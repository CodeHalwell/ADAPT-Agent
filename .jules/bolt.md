## 2024-05-24 - Python lambda overhead in min()
**Learning:** In performance-critical paths such as memory eviction (`adapt_agent/core/memory.py`), a manual `for` loop to find a minimum element in a list of dictionaries is significantly faster (~1.5x) than using `min()` with a `lambda` key, and ~3x faster than `sort()`, due to Python function call overhead.
**Action:** When searching for min/max in large lists of dictionaries (like trace IDs or memory items), use a manual loop instead of built-ins with lambdas to optimize overhead.

## 2024-05-19 - Enum sorting optimization in critical paths
**Learning:** Python Enums in tight loops are relatively slow to compare. Sorting or finding the max of a list of enums using `list.index` or `lambda` creates overhead.
**Action:** When finding a maximum enum value by its priority in a performance-critical path, use a static class-level dictionary to map Enum members to integer priorities. Combine this with a manual loop for O(1) priority lookups and the ability to add early-exit conditions (short-circuiting on the highest priority).

## 2024-05-25 - Efficient recent item filtering
**Learning:** In `adapt_agent/core/policy.py`, filtering a large chronologically ordered list for a limited number of recent matches using standard list comprehensions (e.g., `[v for v in items if condition][-limit:]`) is highly inefficient because it processes all `N` elements.
**Action:** When searching for a limited number of recent items matching a condition, iterate backwards using `reversed(items)` and early-exit (`break`) once the limit is reached to achieve an `O(L)` operation where `L` is the number of inspected elements.
## 2024-05-18 - Early Exit Filtering
**Learning:** `get_evaluation_results` suffered from iterating over the whole history using list comprehensions when returning a limited set of results, adding overhead that scaled linearly with total items rather than requested limits.
**Action:** Replace `list_comprehensions` acting on full arrays with backwards iteration (`reversed()`) combined with an early exit (`break`) based on `limit`.

## 2024-05-26 - O(N*M) to O(N) aggregate metrics optimization
**Learning:** In `adapt_agent/evaluation/__init__.py`, calculating aggregate metrics previously involved multiple passes over all results (first finding unique metrics, then searching for each metric). This resulted in O(N*M) operations where N is results and M is metrics.
**Action:** Replace multiple-pass accumulation with a single O(N) pass using dictionaries to build sums and counts concurrently, then format the result dict.

## 2024-05-27 - Early Exit Filtering in Adversarial Defense
**Learning:** In `adapt_agent/adversarial/__init__.py`, `get_detected_attacks` filtered a large chronologically ordered list using standard list comprehensions, evaluating the entire list even when a limited subset was requested. This caused O(N) overhead when returning recent items.
**Action:** Replace `list_comprehensions` with backwards iteration (`reversed()`) combined with an early exit (`break`) based on `limit`.
## 2024-05-28 - Eliminating redundant list traversals
**Learning:** In `adapt_agent/optimization/__init__.py`, `_compute_statistics` and `suggest_optimizations` previously used list comprehensions followed by multiple generator expressions (`sum(...)`) to filter and compute metric totals (time, success, tokens). This resulted in iterating over the same list multiple times. A single-pass loop reduces overhead and improves performance.
**Action:** When computing multiple aggregates over a filtered list of dictionaries, use a single manual loop to calculate sums and counts concurrently, rather than stacking multiple list comprehensions or `sum()` generators.

## 2024-05-29 - Python AST parsing overhead in tight loops
**Learning:** In `adapt_agent/core/policy.py`, `PolicyEnforcer._evaluate_condition` is called frequently to check every message and state against multiple rules. Parsing the condition string (`ast.parse`) and creating operator mapping dictionaries on every evaluation is highly inefficient and creates significant overhead.
**Action:** When evaluating AST-based conditions or expressions repeatedly, cache the parsed AST using `@lru_cache` on a helper function, and extract static mapping dictionaries (like operator mappings) to class-level or module-level variables. This makes repeated evaluations of the same condition significantly faster (approx ~7x).
## 2024-05-05 - In-place reverse optimization for fetching limited recent items
**Learning:** When filtering large chronologically ordered lists for a limited number of recent matches, replacing `list(reversed(results))` with in-place `results.reverse()` provides a significant speedup by avoiding unnecessary iterator and list object allocations.
**Action:** Always prefer in-place `.reverse()` followed by returning the list over `list(reversed())` for returning chronologically filtered sub-lists.
