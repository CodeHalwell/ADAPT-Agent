## 2024-05-24 - Avoid Information Leakage through Print Statements
**Vulnerability:** Core components (`firewall.py` and `middleware.py`) were using standard `print()` statements to output raw exception messages when catching errors.
**Learning:** This existed because `print()` was used for quick debugging or missing global logging configuration. Printing raw exceptions to stdout can inadvertently leak sensitive internal state, stack traces, or operational details to users or logs not meant for such verbosity.
**Prevention:** Avoid `print()` for error handling. Always use standard Python `logging` with appropriate severity levels (e.g., `logger.error()`) to safely handle errors without exposing potentially sensitive internal information.
