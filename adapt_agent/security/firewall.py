"""Firewall for LLM agent security."""

from typing import Any, Callable, Dict, List, Optional, Pattern
import re
from datetime import datetime

from adapt_agent.core.types import AgentMessage, SecurityEvent


class Firewall:
    """Security firewall for LLM agents.
    
    Provides input/output filtering, pattern matching, and threat detection
    to protect against malicious inputs and prevent sensitive data leakage.
    """
    
    def __init__(self):
        """Initialize the Firewall."""
        self._blocked_patterns: List[Pattern] = []
        self._allowed_patterns: List[Pattern] = []
        self._custom_filters: List[Callable[[str], bool]] = []
        self._security_events: List[SecurityEvent] = []
        self._blocked_count = 0
    
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
        # Check allowed patterns first (whitelist)
        for pattern in self._allowed_patterns:
            if pattern.search(content):
                return True
        
        # Check blocked patterns
        for pattern in self._blocked_patterns:
            if pattern.search(content):
                self._record_security_event(
                    event_type="blocked_input",
                    severity="high",
                    description=f"Input matched blocked pattern: {pattern.pattern}",
                    metadata={"content_snippet": content[:100]},
                )
                self._blocked_count += 1
                return False
        
        # Check custom filters
        for filter_func in self._custom_filters:
            try:
                if filter_func(content):
                    self._record_security_event(
                        event_type="blocked_input",
                        severity="medium",
                        description="Input blocked by custom filter",
                        metadata={"content_snippet": content[:100]},
                    )
                    self._blocked_count += 1
                    return False
            except Exception as e:
                # Log error but don't block on filter failure
                print(f"Error in custom filter: {e}")
        
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
    ) -> List[SecurityEvent]:
        """Get recorded security events.
        
        Args:
            severity: Filter by severity level
            limit: Maximum number of events to return
            
        Returns:
            List of security events
        """
        events = self._security_events
        
        if severity:
            events = [e for e in events if e["severity"] == severity]
        
        if limit:
            events = events[-limit:]
        
        return events
    
    def get_stats(self) -> Dict[str, Any]:
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
        metadata: Dict[str, Any],
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
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata,
        }
        self._security_events.append(event)
