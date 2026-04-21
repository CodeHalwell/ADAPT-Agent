## 2024-05-24 - AgentObserver Traces Bottleneck
**Learning:** `AgentObserver` used an O(N) linear search over a list of dictionaries to look up execution traces by `trace_id` for `end_trace` and `log_event`. This resulted in significant performance degradation (e.g. 2.6s for 10k end events).
**Action:** Changed `_traces` to a dictionary mapped by `trace_id` for O(1) lookups, and used `list(self._traces.values())` in `get_traces` to preserve chronological order for consumers. Always use dictionaries for id-based collections rather than iterating through lists.
