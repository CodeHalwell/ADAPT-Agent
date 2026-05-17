## 2024-05-17 - Middleware Fast Paths
**Learning:** In highly extensible systems, wrapper functions often introduce overhead (like dictionary allocations and loop setups) even when no extensions are registered.
**Action:** Always consider adding zero-extension fast paths (e.g., `if not extensions: return original_data`) to generic processing pipelines to avoid paying the abstraction cost when the abstraction isn't used.
