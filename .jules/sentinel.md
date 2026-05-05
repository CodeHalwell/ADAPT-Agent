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
