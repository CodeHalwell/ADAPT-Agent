"""Adversarial defense for LLM agents."""

from datetime import datetime, timezone
from typing import Any, Optional


class AdversarialDefense:
    """Defends against adversarial attacks on LLM agents.

    Provides detection and mitigation strategies for common attack vectors
    including prompt injection, jailbreaking, and data poisoning.
    """

    _INJECTION_INDICATORS = (
        "ignore previous instructions",
        "disregard all",
        "new instructions:",
        "system:",
        "override",
    )

    _JAILBREAK_INDICATORS = (
        "pretend you are",
        "roleplay as",
        "act as if",
        "you are now",
    )

    def __init__(self, max_attacks: int = 1000, max_content_length: Optional[int] = None):
        """Initialize the AdversarialDefense.

        Args:
            max_attacks: Maximum number of detected attacks to store in memory.
            max_content_length: Optional maximum allowed length for input content.
        """
        self._attack_patterns: list[str] = []
        self._detected_attacks: list[dict[str, Any]] = []
        self._defense_strategies: dict[str, Any] = {}
        self.max_attacks = max_attacks
        self.max_content_length = max_content_length

    def detect_prompt_injection(self, prompt: str, prompt_lower: Optional[str] = None) -> bool:
        """Detect potential prompt injection attacks.

        Args:
            prompt: Input prompt to analyze
            prompt_lower: Optional pre-computed lower-cased prompt

        Returns:
            True if attack detected, False otherwise
        """
        prompt_lower = prompt_lower if prompt_lower is not None else prompt.lower()
        for indicator in self._INJECTION_INDICATORS:
            if indicator in prompt_lower:
                self._record_attack("prompt_injection", prompt, indicator)
                return True

        return False

    def detect_jailbreak(self, prompt: str, prompt_lower: Optional[str] = None) -> bool:
        """Detect jailbreak attempts.

        Args:
            prompt: Input prompt to analyze
            prompt_lower: Optional pre-computed lower-cased prompt

        Returns:
            True if jailbreak detected, False otherwise
        """
        prompt_lower = prompt_lower if prompt_lower is not None else prompt.lower()
        for indicator in self._JAILBREAK_INDICATORS:
            if indicator in prompt_lower:
                self._record_attack("jailbreak", prompt, indicator)
                return True

        return False

    def detect_custom_pattern(self, prompt: str, prompt_lower: Optional[str] = None) -> bool:
        """Detect custom attack patterns.

        Args:
            prompt: Input prompt to analyze
            prompt_lower: Optional pre-computed lower-cased prompt

        Returns:
            True if custom pattern detected, False otherwise
        """
        prompt_lower = prompt_lower if prompt_lower is not None else prompt.lower()
        for pattern in self._attack_patterns:
            if pattern.lower() in prompt_lower:
                self._record_attack("custom_pattern", prompt, pattern)
                return True

        return False

    def analyze_input(self, input_text: str) -> dict[str, Any]:
        """Analyze input for multiple attack vectors.

        Args:
            input_text: Input text to analyze

        Returns:
            Analysis results with detected threats
        """
        # SECURITY: DoS protection by limiting input length
        if self.max_content_length is not None and len(input_text) > self.max_content_length:
            self._record_attack(
                "content_too_long",
                input_text,
                f"Input length {len(input_text)} exceeds maximum allowed {self.max_content_length}",
            )
            return {
                "input": input_text[:100],  # Truncated for privacy
                "threats_detected": ["content_too_long"],
                "is_safe": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        threats = []
        input_lower = input_text.lower()

        if self.detect_prompt_injection(input_text, input_lower):
            threats.append("prompt_injection")

        if self.detect_jailbreak(input_text, input_lower):
            threats.append("jailbreak")

        if self.detect_custom_pattern(input_text, input_lower):
            threats.append("custom_pattern")

        return {
            "input": input_text[:100],  # Truncated for privacy
            "threats_detected": threats,
            "is_safe": len(threats) == 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def add_attack_pattern(self, pattern: str) -> None:
        """Add a custom attack pattern to detect.

        Args:
            pattern: Attack pattern string
        """
        self._attack_patterns.append(pattern)

    def get_detected_attacks(
        self,
        attack_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Get detected attacks.

        Args:
            attack_type: Filter by attack type
            limit: Maximum number of attacks to return

        Returns:
            List of detected attacks
        """
        if limit is None and not attack_type:
            return list(self._detected_attacks)

        results = []
        for attack in reversed(self._detected_attacks):
            if attack_type and attack["type"] != attack_type:
                continue

            results.append(attack)
            if limit and len(results) >= limit:
                break

        results.reverse()
        return results

    def _record_attack(
        self,
        attack_type: str,
        content: str,
        indicator: str,
    ) -> None:
        """Record a detected attack.

        Args:
            attack_type: Type of attack
            content: Attack content
            indicator: Indicator that triggered detection
        """
        attack = {
            "type": attack_type,
            "content": content[:100],  # Truncated
            "indicator": indicator,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._detected_attacks.append(attack)

        # SECURITY: Prevent unbounded memory growth
        if len(self._detected_attacks) > self.max_attacks:
            self._detected_attacks.pop(0)


__all__ = ["AdversarialDefense"]
