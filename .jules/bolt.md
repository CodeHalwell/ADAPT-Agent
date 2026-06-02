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
## 2024-05-30 - Single-pass list comprehensions for chained filters
**Learning:** In `adapt_agent/observability/__init__.py`, chaining multiple list comprehensions (e.g., `res = [x for x in data if condition1]; res = [x for x in res if condition2]`) causes the program to iterate over the dataset multiple times and allocate intermediate arrays in memory.
**Action:** When applying multiple filter conditions to a list or iterable, combine the conditions using `and` inside a single list comprehension or manual loop to avoid intermediate allocations and reduce the operation to a single O(N) pass.
## 2024-05-14 - Redundant string lowercasing in search loops
**Learning:** Redundantly calling `.lower()` on invariants inside a traversal loop over thousands of items causes measurable execution delay due to string allocation and iteration overhead.
**Action:** Extract loop-invariant string operations (like `query.lower()`) outside of tight iteration paths to speed up text-based searches, specifically in large internal datasets like `MemorySystem`.
## 2024-05-31 - Object instantiation overhead in tight iteration loops
**Learning:** In `adapt_agent/core/policy.py`, `PolicyEnforcer.check_message` and `check_state` iterate over all rules to evaluate conditions. Creating a new context dictionary (e.g., `{"message": message}`) inside this loop on every iteration introduces significant overhead due to repeated object creation and subsequent garbage collection.
**Action:** Always hoist invariant object creations (such as the context dictionary) outside of iteration loops to avoid redundant allocation overhead and improve execution speed (approx ~10% improvement in this case).
## 2024-06-01 - O(N) to O(1) membership checking optimization
**Learning:** In `adapt_agent/patches/__init__.py`, storing tracking data (like applied patches) in a `list` requires O(N) membership checks (`if item in list`). Using a `set` instead provides O(1) membership checks.
**Action:** When repeatedly checking if an item exists within a collection (e.g., membership checks or deduplication), use a `set` instead of a `list` to optimize time complexity, unless ordering or duplicate storage is strictly required.
## 2024-05-13 - O(N²) List Removal in Memory System Consolidation
**Learning:** Removing items from a list in Python using `.remove()` inside a loop that iterates over a copy of the list (`list[:]`) creates an O(N²) performance bottleneck. This occurs because `remove()` must scan the list to find the value and then shift all subsequent elements left, multiplying the loop iterations by the length of the list.
**Action:** Always prefer an O(N) list rebuilding strategy (e.g., using a list comprehension or creating a new list and appending items to keep) over using `.remove()` or `del` inside a loop for bulk removals.

## 2024-05-16 - Prevent redefining functions inside evaluation loops
**Learning:** Redefining inner functions (e.g., `def eval_node(node)`) recursively for every rule evaluation causes unnecessary object creation overhead in Python, acting as a major performance bottleneck for AST evaluation. Moving these inside instance methods and replacing `isinstance` with `type(node) is` speeds up evaluation by >30%.
**Action:** Always hoist functions out of tight evaluation loops into instance/class methods if they don't require external closures. Additionally, use `type(node) is X` over `isinstance` for slight speedup in hot paths where subtyping is not required.
## 2024-06-05 - Fast Path for Empty Middleware
**Learning:** `adapt_agent/core/middleware.py` wrapped all agent interactions, and even when no middleware was registered, it still performed `data.copy()` and multiple dictionary instantiations (for `args` and `kwargs`). This adds unnecessary overhead to hot paths where middleware isn't actively being used.
**Action:** When creating wrapper functions or processors that iterate over extensible lists (like middleware or plugins), add an early return (`if not self._pre_middleware`) to bypass dictionary allocation, dictionary copies, and wrapper overhead entirely.
## 2024-05-18 - Optimize AdversarialDefense String Formatting
**Learning:** In performance-critical paths (e.g., `AdversarialDefense.analyze_input`), redundant list instantiations and string transformations like `.lower()` can cause unnecessary recomputations and allocations in hot loops.
**Action:** Store indicator patterns as class-level tuples (e.g., `_INJECTION_INDICATORS`) to prevent redundant instantiations, and hoist repetitive string transformations to the parent caller, passing them down as arguments to sub-methods.
## 2024-06-06 - Pre-computing invariant text manipulations for parallel arrays
**Learning:** In `adapt_agent/adversarial/__init__.py`, `AdversarialDefense.detect_custom_pattern` evaluates input against patterns dynamically. Calling `.lower()` on each custom pattern inside the hot evaluation loop results in O(N) string allocation overhead per analysis run.
**Action:** Optimize custom pattern matching by pre-computing transformed strings (like `.lower()`) into a parallel list or tuple (e.g., `_attack_patterns_tuple`) during insertion (`add_attack_pattern`). Use `zip()` in the hot loop to iterate over the original and pre-computed values concurrently, avoiding redundant manipulation overhead while preserving the public API of the original list.
