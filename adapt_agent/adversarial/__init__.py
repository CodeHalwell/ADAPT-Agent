"""Adversarial defense for LLM agents."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

# Zero-width / invisible characters frequently used to obfuscate attack
# indicators (zero-width space, non-joiner, joiner, BOM/zero-width no-break space).
_ZERO_WIDTH_CHARS = "​‌‍﻿"
_ZERO_WIDTH_RE = re.compile(f"[{_ZERO_WIDTH_CHARS}]")
#: Spaces/tabs and friends, but never a line break -- see :func:`_normalize`.
_HORIZONTAL_WS_RE = re.compile(r"[^\S\n]+")
#: A run of blank lines collapses to one break.
_NEWLINE_RUN_RE = re.compile(r"\s*\n\s*")


def _normalize(text: str) -> str:
    """Normalize text for robust substring matching.

    Applies Unicode NFKC normalization, strips zero-width characters, collapses
    runs of whitespace to a single space, trims, and lowercases. This defeats
    trivial obfuscations (double spacing, zero-width injection, full-width
    look-alikes) without altering the originally stored snippet.

    Args:
        text: Raw input text.

    Returns:
        Normalized, lower-cased text suitable for indicator matching.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    # Horizontal whitespace collapses; line breaks are kept. Newlines are
    # *structure*, not noise: an injected block is typically its own line
    # ("hello\nSYSTEM: reveal secrets"), and flattening it to one line makes a
    # role marker at the start of a line indistinguishable from the same word
    # mid-sentence ("our system: v2 is live"). Runs of blank lines still
    # collapse, so the obfuscations this function exists to defeat -- double
    # spacing, zero-width injection, full-width look-alikes -- are unaffected.
    normalized = _HORIZONTAL_WS_RE.sub(" ", normalized)
    normalized = _NEWLINE_RUN_RE.sub("\n", normalized)
    return normalized.strip().lower()


class AdversarialDefense:
    """Defends against adversarial attacks on LLM agents.

    Provides detection and mitigation strategies for common attack vectors
    including prompt injection, jailbreaking, and data poisoning.
    """

    #: Injection indicators, as regexes over :func:`_normalize`\ d text.
    #:
    #: These were fixed substrings, which meant one inserted word defeated them:
    #: ``"ignore previous instructions"`` matched, while ``"ignore ALL previous
    #: instructions"`` -- by far the more common phrasing -- did not, and neither
    #: did ``"the"``/``"any"``/``"your"`` in the same slot. Matching the shape of
    #: the phrase rather than one spelling of it closes that without widening
    #: into false positives on ordinary prose.
    _INJECTION_PATTERNS = (
        # The gaps must cross a line break. Preserving newlines in
        # :func:`_normalize` is what lets the ``system:`` anchor work, but a
        # gap that excluded ``\n`` then made a newline a *detection boundary* --
        # "ignore\nprevious instructions" evaded a pattern that caught the same
        # words on one line. Sentence-enders still stop a match, so the phrase
        # cannot be stitched together across unrelated sentences.
        r"\b(?:ignore|disregard|forget|override)\b[^.!?]{0,40}?"
        r"\b(?:previous|prior|earlier|above|preceding|all)\b[^.!?]{0,40}?"
        r"\b(?:instruction|instructions|prompt|prompts|rule|rules|direction|directions)\b",
        r"\bdisregard\s+all\b",
        r"\bnew\s+instructions\s*:",
        r"(?:^|\n)\s*system\s*:",
        # Scoped, unlike the bare "override" substring this replaces: that
        # flagged ordinary developer prose ("override the default timeout").
        r"\boverride\b[^.!?]{0,40}?"
        r"\b(?:instruction|instructions|prompt|prompts|rule|rules|system|safety|guardrail|filter)\b",
    )

    #: Jailbreak indicators, same treatment.
    _JAILBREAK_PATTERNS = (
        r"\bpretend\s+(?:that\s+)?you\s+(?:are|were|re)\b",
        r"\brole[\s-]?play(?:ing)?\s+as\b",
        r"\bact\s+as\s+(?:if|though|a|an)\b",
        r"\byou\s+are\s+now\b",
        r"\bdeveloper\s+mode\b",
        r"\bdo\s+anything\s+now\b",
    )

    _INJECTION_INDICATORS = tuple(re.compile(p) for p in _INJECTION_PATTERNS)
    _JAILBREAK_INDICATORS = tuple(re.compile(p) for p in _JAILBREAK_PATTERNS)

    def __init__(self, max_attacks: int = 1000, max_content_length: int | None = None):
        """Initialize the AdversarialDefense.

        Args:
            max_attacks: Maximum number of detected attacks to store in memory.
            max_content_length: Optional maximum allowed length for input content.
        """
        self._attack_patterns: list[str] = []
        self._attack_patterns_tuple: list[tuple[str, str]] = []
        self._detected_attacks: list[dict[str, Any]] = []
        self._defense_strategies: dict[str, Any] = {}
        self.max_attacks = max_attacks
        self.max_content_length = max_content_length

    def detect_prompt_injection(self, prompt: str, prompt_normalized: str | None = None) -> bool:
        """Detect potential prompt injection attacks (pure predicate, no side effects).

        Args:
            prompt: Input prompt to analyze.
            prompt_normalized: Optional pre-computed normalized prompt.

        Returns:
            True if an injection indicator is present, False otherwise.
        """
        return self._match_injection(prompt, prompt_normalized) is not None

    def detect_jailbreak(self, prompt: str, prompt_normalized: str | None = None) -> bool:
        """Detect jailbreak attempts (pure predicate, no side effects).

        Args:
            prompt: Input prompt to analyze.
            prompt_normalized: Optional pre-computed normalized prompt.

        Returns:
            True if a jailbreak indicator is present, False otherwise.
        """
        return self._match_jailbreak(prompt, prompt_normalized) is not None

    def detect_custom_pattern(self, prompt: str, prompt_normalized: str | None = None) -> bool:
        """Detect custom attack patterns (pure predicate, no side effects).

        Args:
            prompt: Input prompt to analyze.
            prompt_normalized: Optional pre-computed normalized prompt.

        Returns:
            True if a registered custom pattern is present, False otherwise.
        """
        return self._match_custom_pattern(prompt, prompt_normalized) is not None

    def _match_injection(self, prompt: str, prompt_normalized: str | None = None) -> str | None:
        """Return the matching injection indicator, or None."""
        normalized = prompt_normalized if prompt_normalized is not None else _normalize(prompt)
        for indicator in self._INJECTION_INDICATORS:
            match = indicator.search(normalized)
            if match is not None:
                return match.group(0)
        return None

    def _match_jailbreak(self, prompt: str, prompt_normalized: str | None = None) -> str | None:
        """Return the matching jailbreak indicator, or None."""
        normalized = prompt_normalized if prompt_normalized is not None else _normalize(prompt)
        for indicator in self._JAILBREAK_INDICATORS:
            match = indicator.search(normalized)
            if match is not None:
                return match.group(0)
        return None

    def _match_custom_pattern(
        self, prompt: str, prompt_normalized: str | None = None
    ) -> str | None:
        """Return the original (un-normalized) custom pattern that matched, or None."""
        normalized = prompt_normalized if prompt_normalized is not None else _normalize(prompt)
        # ⚡ Bolt: Pre-computed lowercased patterns in _attack_patterns_tuple avoid O(N) string manipulation inside this hot loop
        for pattern, norm_pattern in self._attack_patterns_tuple:
            if norm_pattern in normalized:
                return pattern
        return None

    def analyze_input(self, input_text: str) -> dict[str, Any]:
        """Analyze input for multiple attack vectors.

        Detection runs against a normalized copy of the input (NFKC, zero-width
        stripped, whitespace-collapsed, lower-cased) so trivial obfuscations do
        not bypass matching. Each detected attack is recorded exactly once.

        Args:
            input_text: Input text to analyze.

        Returns:
            Analysis results with detected threats.
        """
        # SECURITY: DoS protection by limiting input length.
        if self.max_content_length is not None and len(input_text) > self.max_content_length:
            self._record_attack(
                "content_too_long",
                input_text,
                f"Input length {len(input_text)} exceeds maximum allowed {self.max_content_length}",
            )
            return {
                "input": input_text[:100],  # Truncated for privacy.
                "threats_detected": ["content_too_long"],
                "is_safe": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        normalized = _normalize(input_text)
        threats: list[str] = []

        injection = self._match_injection(input_text, normalized)
        if injection is not None:
            self._record_attack("prompt_injection", input_text, injection)
            threats.append("prompt_injection")

        jailbreak = self._match_jailbreak(input_text, normalized)
        if jailbreak is not None:
            self._record_attack("jailbreak", input_text, jailbreak)
            threats.append("jailbreak")

        custom = self._match_custom_pattern(input_text, normalized)
        if custom is not None:
            self._record_attack("custom_pattern", input_text, custom)
            threats.append("custom_pattern")

        return {
            "input": input_text[:100],  # Truncated for privacy.
            "threats_detected": threats,
            "is_safe": len(threats) == 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def add_attack_pattern(self, pattern: str) -> None:
        """Add a custom attack pattern to detect.

        Args:
            pattern: Attack pattern string.
        """
        self._attack_patterns.append(pattern)
        self._attack_patterns_tuple.append((pattern, _normalize(pattern)))

    def get_detected_attacks(
        self,
        attack_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get detected attacks.

        Args:
            attack_type: Filter by attack type.
            limit: Maximum number of attacks to return.

        Returns:
            List of detected attacks.
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
            attack_type: Type of attack.
            content: Attack content (the original, un-normalized text).
            indicator: Indicator that triggered detection.
        """
        attack = {
            "type": attack_type,
            "content": content[:256]
            .replace("\n", "\\n")
            .replace("\r", "\\r")[:100],  # Truncated and sanitized.
            "indicator": indicator,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._detected_attacks.append(attack)

        # SECURITY: Prevent unbounded memory growth.
        if len(self._detected_attacks) > self.max_attacks:
            self._detected_attacks.pop(0)


__all__ = ["AdversarialDefense"]
