## 2025-02-28 - [Fail-Open Custom Filters in Firewall]
**Vulnerability:** The `Firewall` class's custom filter evaluation wrapped filter execution in a `try...except` block but logged the error and returned `True` (fail-open) if an exception was raised.
**Learning:** Security controls that allow custom user logic (like filters) must always fail closed. Allowing an exception to bypass a block enables potential evasion by intentionally causing an error.
**Prevention:** Always default to a secure state (e.g., returning `False` or raising an error) when security-critical validation logic encounters an unexpected exception. Use logging frameworks correctly instead of `print` to ensure errors are captured in logs.
