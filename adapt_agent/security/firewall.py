"""Firewall for LLM agent security."""

import logging
import re
from datetime import datetime, timezone
from re import Pattern
from typing import Any, Callable, Optional

from adapt_agent.core.types import AgentMessage, SecurityEvent

logger = logging.getLogger(__name__)


class Firewall:
    """Security firewall for LLM agents.

    Provides input/output filtering, pattern matching, and threat detection
    to protect against malicious inputs and prevent sensitive data leakage.
    """

    def __init__(self, max_content_length: Optional[int] = None, max_events: int = 1000):
        """Initialize the Firewall.

        Args:
            max_content_length: Optional maximum allowed length for input/output content. If set, any content exceeding this length will be blocked (DoS protection).
            max_events: Maximum number of security events to store in memory.
        """
        self._blocked_patterns: list[Pattern] = []
        self._allowed_patterns: list[Pattern] = []
        self._custom_filters: list[Callable[[str], bool]] = []
        self._security_events: list[SecurityEvent] = []
        self._blocked_count = 0
        self.max_content_length = max_content_length
        self.max_events = max_events

    def add_blocked_pattern(self, pattern: str, flags: int = 0) -> None:
        """Add a regex pattern to block.

        Args:
            pattern: Regular expression pattern
            flags: Optional regex flags (e.g., re.IGNORECASE)
        """
        compiled_pattern = re.compile(pattern, flags)
        self._blocked_patterns.append(compiled_pattern)

    def add_allowed_pattern(self, pattern: str, flags: int = 0) -> None:
        """Add a regex pattern to explicitly allow.

        Args:
            pattern: Regular expression pattern
            flags: Optional regex flags (e.g., re.IGNORECASE)
        """
        compiled_pattern = re.compile(pattern, flags)
        self._allowed_patterns.append(compiled_pattern)

    def add_custom_filter(self, filter_func: Callable[[str], bool]) -> None:
        """Add a custom filter function.

        Args:
            filter_func: Function that returns True if content should be blocked
        """
        self._custom_filters.append(filter_func)

    def check_input(self, content: str) -> bool:
        """Check if input content should be blocked.

        Args:
            content: Input content to check

        Returns:
            True if content is allowed, False if blocked
        """
        # SECURITY: DoS protection by limiting input length
        if self.max_content_length is not None and len(content) > self.max_content_length:
            # SECURITY: Prevent log poisoning by escaping newline and carriage return characters.
            safe_snippet = (
                self.sanitize(content[:256])[:100].replace("\n", "\\n").replace("\r", "\\r")
            )
            self._record_security_event(
                event_type="blocked_input",
                severity="high",
                description=f"Input length {len(content)} exceeds maximum allowed {self.max_content_length}",
                metadata={
                    "content_snippet": safe_snippet,
                    "length": len(content),
                },
            )
            self._blocked_count += 1
            return False

        # Check allowed patterns first (whitelist)
        for pattern in self._allowed_patterns:
            if pattern.fullmatch(content):
                return True

        # Check blocked patterns
        for pattern in self._blocked_patterns:
            if pattern.search(content):
                # SECURITY: Prevent log poisoning by escaping newline and carriage return characters.
                safe_snippet = (
                    self.sanitize(content[:256])[:100].replace("\n", "\\n").replace("\r", "\\r")
                )
                self._record_security_event(
                    event_type="blocked_input",
                    severity="high",
                    description=f"Input matched blocked pattern: {pattern.pattern}",
                    metadata={"content_snippet": safe_snippet},
                )
                self._blocked_count += 1
                return False

        # Check custom filters
        for filter_func in self._custom_filters:
            try:
                if filter_func(content):
                    # SECURITY: Prevent log poisoning by escaping newline and carriage return characters.
                    safe_snippet = (
                        self.sanitize(content[:256])[:100].replace("\n", "\\n").replace("\r", "\\r")
                    )
                    self._record_security_event(
                        event_type="blocked_input",
                        severity="medium",
                        description="Input blocked by custom filter",
                        metadata={"content_snippet": safe_snippet},
                    )
                    self._blocked_count += 1
                    return False
            except Exception as e:
                # Log error and block on filter failure (fail-closed)
                logger.error(f"Error in custom filter: {e}")
                # SECURITY: Prevent log poisoning by escaping newline and carriage return characters.
                safe_snippet = (
                    self.sanitize(content[:256])[:100].replace("\n", "\\n").replace("\r", "\\r")
                )
                self._record_security_event(
                    event_type="blocked_input",
                    severity="high",
                    description="Input blocked due to custom filter error",
                    metadata={
                        "content_snippet": safe_snippet,
                        "error": "An error occurred",
                    },
                )
                self._blocked_count += 1
                return False

        return True

    def check_output(self, content: str) -> bool:
        """Check if output content should be blocked.

        Args:
            content: Output content to check

        Returns:
            True if content is allowed, False if blocked
        """
        # Similar logic to check_input but for outputs
        return self.check_input(content)

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
        severity: Optional[str] = None,
        limit: Optional[int] = None,
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
