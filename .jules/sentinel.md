## 2026-04-21 - Fail-Closed Firewall Design
**Vulnerability:** The Firewall allowed custom filters to fail-open (returning True instead of False) if an exception occurred during `check_input`.
**Learning:** This could have allowed malicious actors to bypass the firewall entirely by purposefully providing input designed to crash poorly written custom filters. Security controls should inherently default to a fail-closed posture when errors occur to prevent accidental leakage or exploitation.
**Prevention:** Always implement fail-closed mechanisms for custom plugins or filters in security boundaries. When adding new security layers, verify that unhandled exceptions result in blocking the action (and logging the failure appropriately) rather than allowing the action.
