## 2026-04-21 - Fail-Closed Firewall Design
**Vulnerability:** The Firewall allowed custom filters to fail-open (returning True instead of False) if an exception occurred during `check_input`.
**Learning:** This could have allowed malicious actors to bypass the firewall entirely by purposefully providing input designed to crash poorly written custom filters. Security controls should inherently default to a fail-closed posture when errors occur to prevent accidental leakage or exploitation.
**Prevention:** Always implement fail-closed mechanisms for custom plugins or filters in security boundaries. When adding new security layers, verify that unhandled exceptions result in blocking the action (and logging the failure appropriately) rather than allowing the action.
## 2026-04-26 - Firewall Resource Exhaustion / Unbounded Growth
**Vulnerability:** The Firewall class tracked `_security_events` in an unbounded list and lacked a maximum length check on input payload length, opening the agent up to Memory Leak / Memory Exhaustion via thousands of generated security events, and DoS / ReDoS attacks on regular expressions.
**Learning:** Security controls processing arbitrary user input, particularly LLM inputs and adversarial attempts, can generate logs dynamically. If these logs are unbounded, they can crash the system. Processing arbitrarily long text fields with regex patterns runs the severe risk of taking O(2^N) or O(N^2) evaluation time (ReDoS).
**Prevention:** Always implement `max_events` limits (capping arrays/lists) for internal tracking arrays, and always implement explicit string length maximums for security scanners before running resource-intensive regex processing.
## 2026-05-02 - List Bounding for DoS Prevention
**Vulnerability:** Core tracking components (`PolicyEnforcer`, `TaintTracker`, `AgentObserver`) tracked events in unbounded lists (`_violations`, `_taint_propagation`, `_logs`), posing a Denial of Service (DoS) risk via memory exhaustion.
**Learning:** Unbounded tracking of events generated dynamically by arbitrary inputs can crash the system. Bounding list sizes is crucial for long-running agents.
**Prevention:** Always implement max length checks (`max_violations`, `max_propagations`, `max_logs`) for internal tracking arrays, and cap arrays/lists when appending.
## 2026-05-05 - AST Whitelisting vs Arbitrary eval()
**Vulnerability:** A feature designed to dynamically evaluate expressions (such as condition strings) used a placeholder that returned `False`. A naive implementation would have used Python's built-in `eval()`, which is a critical security risk (arbitrary code execution) if untrusted inputs can influence the evaluated string.
**Learning:** `eval()` should essentially never be used when processing inputs that may contain adversarial data or dynamically generated content without a strict, air-gapped sandbox.
**Prevention:** Always implement a safe expression evaluator using an Abstract Syntax Tree (AST) whitelisting approach (via `ast.parse` and carefully walking the nodes) when simple dynamic condition evaluations are required, explicitly forbidding function calls (`ast.Call`), attribute access (`ast.Attribute`), and unbounded operations (e.g. `ast.Pow`).
## 2024-05-18 - [Firewall Whitelist Bypass via regex search]
**Vulnerability:** The `Firewall` class's explicit whitelist mechanism (`_allowed_patterns`) used `pattern.search(content)`, which allowed arbitrary inputs to bypass security blocks as long as they contained an allowed substring anywhere within them (e.g., `allowed_str malicious_payload`).
**Learning:** Using regex `search` instead of `fullmatch` for access control or whitelisting creates a severe bypass vulnerability because it fails to enforce boundary conditions on the input.
**Prevention:** Always use `pattern.fullmatch()` or explicitly define boundary anchors (`^` and `$`) when evaluating inputs against a whitelist to ensure the *entire* input is allowed.
## 2026-05-09 - Dictionary Bounding for DoS Prevention
**Vulnerability:** Unbounded dictionaries in `TaintTracker` (e.g., `_tainted_data`, `_taint_sources`) and `TrustManager` (mapping dynamically generated `agent_id`s) present latent memory exhaustion (DoS) vulnerabilities if an attacker can continually generate unique IDs.
**Learning:** Even dictionaries that track keys rather than lists can lead to unbounded memory growth if the keys represent dynamic, externally influencable properties like unique IDs, agents, or tracking contexts.
**Prevention:** Always implement maximum length constraints (`max_tracked_items`, `max_agents`) for internally tracked dictionaries when elements are generated over time, and delete older entries using `next(iter(dict))` to prevent memory exhaustion DoS.
## 2024-05-24 - [Information Leakage in Event Metadata]
**Vulnerability:** Exception messages (`str(e)`) were being leaked into security event metadata which may be exposed or aggregated in systems accessible by lower-privilege users.
**Learning:** While full exception traces should be kept in internal application logs (`logger.error(..., exc_info=True)`) for debugging, they must never be included in system events, API responses, or metadata that could be accessed externally or by less privileged observers.
**Prevention:** Always separate internal diagnostic logging from event metadata. Sanitize errors in event metadata to generic messages like "An error occurred" while logging the full exception using the internal `logger`.
## 2026-05-07 - AgentObserver Nested List and Dictionary Bounding for DoS Prevention
**Vulnerability:** The AgentObserver class tracked metrics in a dictionary (`_metrics`) without bounding the number of unique metric names (keys), and tracked events inside traces (`_traces[trace_id]['events']`) in an unbounded list. This posed a Denial of Service (DoS) risk via memory exhaustion if an attacker dynamically generated infinite unique metric names or spammed events on an open trace.
**Learning:** Even deeply nested lists inside dictionaries, and dictionaries that track dynamically generated keys, represent latent memory exhaustion (DoS) vulnerabilities if left unbounded.
**Prevention:** Always implement maximum length constraints (`max_metric_names`, `max_events_per_trace`) for internally tracked dictionaries and nested lists. When bounded structures exceed their limits, securely evict older entries (using `.pop(0)` for lists and `del dict[next(iter(dict))]` for dictionaries).
## 2026-05-12 - Middleware Priority Inversion
**Vulnerability:** The `Middleware` class sorted its `_pre_middleware` and `_post_middleware` arrays prior to injecting the new middleware metadata into the tracking dictionary `_middleware_metadata`, and relied on `m.__name__` which could mismatch the provided string identifier `name`.
**Learning:** Priority sorting based on external tracking dictionaries must ensure the tracking dictionary is updated first. Further, sorting keys must accurately identify function objects instead of string names, which could be shadowed or overridden during dynamic registration. If misordered, security middlewares could run too late.
**Prevention:** Always register metadata state before executing side-effects that depend on it (like sorting), and use identity-based mapping (`func_priorities = {m["function"]: m["priority"] ...}`) rather than string lookups for robust sorting of callbacks or execution pipelines.
## 2026-05-18 - Privilege Escalation via Unvalidated Trust Input
**Vulnerability:** The `TrustManager.evaluate_agent_state` method incorporated an unvalidated `trust_score` directly from the agent's state (`state["trust_score"]`), allowing a malicious agent to self-assign maximum trust and bypass subsequent security checks based on trust levels.
**Learning:** Security evaluations must never blindly trust or incorporate inputs provided by the entity being evaluated, especially regarding its own trust, reputation, or authorization level.
**Prevention:** Always calculate trust, reputation, or authorization scores server-side based on objective metrics and behaviors (e.g., policy violations), and never rely on self-reported security attributes from untrusted entities.

## 2025-05-10 - [Memory Exhaustion DoS in MemorySystem]
**Vulnerability:** The `MemorySystem.consolidate` method in `adapt_agent/core/memory.py` directly appended items from short-term memory to `_long_term_memory`, bypassing the capacity limits enforced by `store_long_term`.
**Learning:** Hardcoded state updates or direct list append operations can bypass capacity bounding limits, making the system susceptible to a Denial of Service (DoS) memory exhaustion vulnerability.
**Prevention:** Always use established interface methods (like `store_long_term`) which internally handle bound limitations rather than directly mutating internal state arrays.
## 2026-05-13 - Broken Custom Attack Pattern Enforcement
**Vulnerability:** The `AdversarialDefense.analyze_input` method allowed adding custom attack patterns via `add_attack_pattern`, but failed to actually check the input text against these patterns, rendering custom defenses useless and allowing malicious inputs to bypass the intended blocklist.
**Learning:** Implementing configuration (adding patterns) without ensuring its integration into the main execution pathway (`analyze_input`) creates a false sense of security. Attackers could bypass rules administrators believed were enforced.
**Prevention:** Always verify that newly added security rules, patterns, or configurations are actively called and evaluated during the core request inspection flow. Write integration tests that explicitly trigger the custom rules.

## 2024-05-14 - Exposed sensitive data in firewall events
**Vulnerability:** Firewall records raw, unredacted content snippets (`content[:100]`) in its internal security events when blocking input, leaking the sensitive data it was supposed to block.
**Learning:** Security mechanisms that log or record events must also sanitize the data they store to avoid becoming the source of a leak themselves (logging blocked secrets).
**Prevention:** Always apply sanitization (e.g. `self.sanitize()`) before storing snippets or payloads in logs, metrics, or event trackers.
## 2026-05-19 - Unbounded Input Length DoS in AdversarialDefense
**Vulnerability:** The `AdversarialDefense.analyze_input` method processed potentially unbounded string inputs directly through multiple text search checks without enforcing a maximum length, leading to a Denial of Service (DoS) vulnerability via memory exhaustion and prolonged processing times.
**Learning:** Security components that analyze strings must limit the maximum length of the input they accept before applying rules or patterns, otherwise they can be forced to process excessively large strings, consuming resources and degrading availability.
**Prevention:** Always enforce a `max_content_length` check at the boundary of input analysis to immediately reject excessively large inputs before performing text processing or regex matching.
