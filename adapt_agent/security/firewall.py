"""Firewall for LLM agent security.

.. warning::
    Regex patterns passed to :meth:`Firewall.add_blocked_pattern` and
    :meth:`Firewall.add_allowed_pattern` are evaluated with the standard library
    :mod:`re` engine, which is vulnerable to catastrophic backtracking (ReDoS).
    Patterns are therefore **trusted-author-only**: never compile patterns
    derived from untrusted/end-user input. :meth:`add_blocked_pattern` and
    :meth:`add_allowed_pattern` reject a few obviously catastrophic constructs as
    a best-effort guard, but this is not a substitute for author review.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from datetime import datetime, timezone
from re import Pattern
from typing import Any

from adapt_agent.core.types import AgentMessage, SecurityEvent

logger = logging.getLogger(__name__)

# Best-effort guard against the most common catastrophic-backtracking shapes,
# e.g. nested quantifiers like ``(a+)+``, ``(a*)*`` or ``(a+)*``. This is a
# heuristic, not a complete ReDoS analysis.
_CATASTROPHIC_RE = re.compile(r"\([^)(]*[+*]\)[+*]")


class Firewall:
    r"""Security firewall for LLM agents.

    Provides input/output filtering, pattern matching, and threat detection
    to protect against malicious inputs and prevent sensitive data leakage.

    Precedence (default, ``whitelist_mode=False``): blocked patterns, custom
    filters and the length cap are checked **first**, so a broad allowed pattern
    can never silently nullify the blocklist. Allowed patterns only short-circuit
    content that has already passed every block check.

    When ``whitelist_mode=True`` a full match against any allowed pattern
    short-circuits to *allowed* **before** block checks run (the historical
    behaviour). Use this only when the allowed patterns are themselves the
    security boundary.

    .. warning::
       Neither mode makes allowed patterns *restrictive*. They exempt content,
       they never reject it: content matching no allowed pattern is still
       allowed once the block checks pass. ``whitelist_mode`` changes only the
       precedence between allowed and blocked patterns, not this. To require
       that input match a shape, invert a custom filter instead::

           permitted = re.compile(r"^[\w\s.,?!-]+$")
           firewall.add_custom_filter(lambda c: not permitted.fullmatch(c))
    """

    def __init__(
        self,
        max_content_length: int | None = None,
        max_events: int = 1000,
        whitelist_mode: bool = False,
    ):
        """Initialize the Firewall.

        Args:
            max_content_length: Optional maximum allowed length for input/output
                content. If set, any content exceeding this length will be
                blocked (DoS protection).
            max_events: Maximum number of security events to store in memory.
            whitelist_mode: When False (default), blocked patterns always win
                over allowed patterns (block-first). When True, an allowed
                fullmatch short-circuits to allowed before block checks run
                (strict allowlist; legacy behaviour).
        """
        self._blocked_patterns: list[Pattern] = []
        self._allowed_patterns: list[Pattern] = []
        self._custom_filters: list[Callable[[str], bool]] = []
        self._security_events: list[SecurityEvent] = []
        self._blocked_count = 0
        self.max_content_length = max_content_length
        self.max_events = max_events
        self.whitelist_mode = whitelist_mode

    def add_blocked_pattern(self, pattern: str, flags: int = 0) -> None:
        """Add a regex pattern to block.

        Args:
            pattern: Regular expression pattern. Trusted-author-only (ReDoS risk;
                see module docstring).
            flags: Optional regex flags (e.g., re.IGNORECASE)

        Raises:
            ValueError: If the pattern matches an obviously catastrophic shape.
        """
        self._reject_catastrophic(pattern)
        compiled_pattern = re.compile(pattern, flags)
        self._blocked_patterns.append(compiled_pattern)

    def add_allowed_pattern(self, pattern: str, flags: int = 0) -> None:
        """Add a regex pattern to explicitly allow.

        Args:
            pattern: Regular expression pattern. Trusted-author-only (ReDoS risk;
                see module docstring).
            flags: Optional regex flags (e.g., re.IGNORECASE)

        Raises:
            ValueError: If the pattern matches an obviously catastrophic shape.
        """
        self._reject_catastrophic(pattern)
        compiled_pattern = re.compile(pattern, flags)
        self._allowed_patterns.append(compiled_pattern)

    @staticmethod
    def _reject_catastrophic(pattern: str) -> None:
        """Reject obviously catastrophic regex constructs (best-effort ReDoS guard)."""
        if _CATASTROPHIC_RE.search(pattern):
            raise ValueError(
                "Refusing to compile a regex with nested quantifiers "
                f"(catastrophic-backtracking / ReDoS risk): {pattern!r}"
            )

    def add_custom_filter(self, filter_func: Callable[[str], bool]) -> None:
        """Add a custom filter function.

        Args:
            filter_func: Function that returns True if content should be blocked
        """
        self._custom_filters.append(filter_func)

    @staticmethod
    def _truncate_escaped(text: str, max_length: int) -> str:
        """Safely truncate text to max_length after escaping to prevent partial escapes."""
        truncated = text[:max_length]
        escaped = truncated.replace("\n", "\\n").replace("\r", "\\r")
        if len(escaped) <= max_length:
            return escaped
        safe = escaped[:max_length]
        if safe.endswith("\\") and escaped[max_length] in ("n", "r"):
            safe = safe[:-1]
        return safe

    @staticmethod
    def _redact_snippet(content: str) -> str:
        """Return a non-sensitive snippet for storage in a security event.

        Stores only the first 12 characters plus a short sha256 prefix so that
        raw (possibly dangerous) content is never persisted verbatim while
        remaining useful for correlation/debugging.
        """
        digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:12]
        safe_snippet = Firewall._truncate_escaped(content, 12)
        return f"{safe_snippet}…(sha256:{digest})"

    def _check(self, content: str, event_type: str) -> bool:
        """Core block-first check shared by check_input/check_output.

        Args:
            content: Content to check.
            event_type: Event type label recorded on any security event
                (e.g. "blocked_input" or "blocked_output").

        Returns:
            True if content is allowed, False if blocked.
        """
        # In strict allowlist mode, a full allowed match short-circuits FIRST.
        if self.whitelist_mode:
            for pattern in self._allowed_patterns:
                if pattern.fullmatch(content):
                    return True

        # SECURITY: DoS protection by limiting content length (block-first).
        if self.max_content_length is not None and len(content) > self.max_content_length:
            self._record_security_event(
                event_type=event_type,
                severity="high",
                description=(
                    f"Content length {len(content)} exceeds maximum allowed "
                    f"{self.max_content_length}"
                ),
                metadata={
                    "content_snippet": self._redact_snippet(content),
                    "length": len(content),
                },
            )
            self._blocked_count += 1
            return False

        # Check blocked patterns (block-first: these always win over allowed
        # patterns in the default, non-whitelist mode).
        for pattern in self._blocked_patterns:
            if pattern.search(content):
                self._record_security_event(
                    event_type=event_type,
                    severity="high",
                    description=f"Content matched blocked pattern: {pattern.pattern}",
                    metadata={
                        # SECURITY: Sanitize complete content before truncation to prevent partial secret leakage
                        "content_snippet": self._truncate_escaped(self.sanitize(content), 100)
                    },
                )
                self._blocked_count += 1
                return False

        # Check custom filters.
        for filter_func in self._custom_filters:
            try:
                if filter_func(content):
                    self._record_security_event(
                        event_type=event_type,
                        severity="medium",
                        description="Content blocked by custom filter",
                        metadata={"content_snippet": self._redact_snippet(content)},
                    )
                    self._blocked_count += 1
                    return False
            except Exception as e:  # noqa: BLE001 - fail closed on any filter error
                # Log error and block on filter failure (fail-closed).
                logger.error("Error in custom filter: %s", e)
                self._record_security_event(
                    event_type=event_type,
                    severity="high",
                    description="Content blocked due to custom filter error",
                    metadata={
                        "content_snippet": self._redact_snippet(content),
                        "error": "An error occurred",
                    },
                )
                self._blocked_count += 1
                return False

        # In the default (block-first) mode, an allowed fullmatch may whitelist
        # content that has already passed every block check. It can never
        # override a block.
        if not self.whitelist_mode:
            for pattern in self._allowed_patterns:
                if pattern.fullmatch(content):
                    return True

        return True

    def check_input(self, content: str) -> bool:
        """Check if input content should be blocked.

        Block-first by default: blocked patterns, custom filters and the length
        cap are evaluated before any allowed-pattern whitelist (unless
        ``whitelist_mode=True``).

        Args:
            content: Input content to check

        Returns:
            True if content is allowed, False if blocked
        """
        return self._check(content, event_type="blocked_input")

    def check_output(self, content: str) -> bool:
        """Check if output content should be blocked.

        Applies the same block-first logic as :meth:`check_input`, but records
        any security event with the ``"blocked_output"`` event type.

        Args:
            content: Output content to check

        Returns:
            True if content is allowed, False if blocked
        """
        return self._check(content, event_type="blocked_output")

    def sanitize(self, content: str, replacement: str = "[REDACTED]") -> str:
        """Sanitize content by replacing blocked patterns.

        Args:
            content: Content to sanitize
            replacement: Replacement string for blocked patterns

        Returns:
            Sanitized content
        """
        sanitized = content

        for pattern in self._blocked_patterns:
            sanitized = pattern.sub(replacement, sanitized)

        return sanitized

    def check_message(self, message: AgentMessage) -> bool:
        """Check if a message should be allowed.

        Args:
            message: Message to check

        Returns:
            True if message is allowed, False if blocked
        """
        return self.check_input(message["content"])

    def get_security_events(
        self,
        severity: str | None = None,
        limit: int | None = None,
    ) -> list[SecurityEvent]:
        """Get recorded security events.

        Args:
            severity: Filter by severity level
            limit: Maximum number of events to return

        Returns:
            List of security events
        """
        events = self._security_events

        if severity:
            # ⚡ Bolt: Fast path for finding limited recent events with a specific severity
            if limit:
                results = []
                for e in reversed(events):
                    if e["severity"] == severity:
                        results.append(e)
                        if len(results) >= limit:
                            break
                results.reverse()
                return results
            return [e for e in events if e["severity"] == severity]

        if limit:
            return events[-limit:]

        return events

    def get_stats(self) -> dict[str, Any]:
        """Get firewall statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "total_blocked": self._blocked_count,
            "security_events": len(self._security_events),
            "blocked_patterns": len(self._blocked_patterns),
            "allowed_patterns": len(self._allowed_patterns),
            "custom_filters": len(self._custom_filters),
        }

    def _record_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        metadata: dict[str, Any],
    ) -> None:
        """Record a security event.

        Args:
            event_type: Type of security event
            severity: Severity level
            description: Event description
            metadata: Additional metadata
        """
        event: SecurityEvent = {
            "event_type": event_type,
            "severity": severity,
            "description": description,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
        }
        self._security_events.append(event)

        # SECURITY: Prevent unbounded memory growth
        if len(self._security_events) > self.max_events:
            self._security_events.pop(0)
