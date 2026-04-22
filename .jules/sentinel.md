## 2025-02-28 - Information Leakage via `print` Statements
**Vulnerability:** Exception handlers throughout the codebase (`adapt_agent/core/middleware.py`, `adapt_agent/security/firewall.py`, `adapt_agent/patches/__init__.py`, `adapt_agent/evaluation/__init__.py`) were using `print()` to log errors, which could expose sensitive internal details (e.g., exception messages) to standard output.
**Learning:** Developers likely used `print()` for quick debugging and left them in production code instead of implementing a robust logging system. This violates the "fail securely" principle.
**Prevention:** Always use the standard Python `logging` module (`logger.error`) inside exception handlers and establish a global logging configuration to manage error visibility.
