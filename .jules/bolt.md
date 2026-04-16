## 2024-04-16 - AgentObserver trace lookup bottleneck
**Learning:** Storing `_traces` as a List in `AgentObserver` causes O(n) lookups for `log_event` and `end_trace`. With 100k traces, this takes ~8 seconds for 1k events.
**Action:** Always use dictionaries mapping `trace_id` -> trace dict when constant O(1) lookups by ID are required, and convert back to list using `.values()` only when necessary for querying APIs like `get_traces()`.
