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
#: Every whitespace run, line breaks included -- see :func:`_normalize`.
_WHITESPACE_RE = re.compile(r"\s+")
#: Spaces/tabs and friends, but never a line break -- see :func:`_normalize_lines`.
_HORIZONTAL_WS_RE = re.compile(r"[^\S\n]+")
#: A run of blank lines collapses to one break.
_NEWLINE_RUN_RE = re.compile(r"\s*\n\s*")
#: Every separator that starts a new line, so a role marker cannot hide behind
#: an unusual one. A bare CR is the classic miss: it renders as a line break but
#: is not ``\n``.
_LINE_SEPARATORS_RE = re.compile(r"\r\n|\r|\x0b|\x0c|\u0085|\u2028|\u2029")
#: Any line break at all, used only to tell whether a caller-supplied cache
#: still has the line structure the role parser needs.
_ANY_LINE_BREAK_RE = re.compile(r"[\n\r\x0b\x0c\u0085\u2028\u2029]")
#: Purely presentational characters that can *surround* text on a line --
#: Markdown headings and blockquotes, list bullets, emphasis, code spans, table
#: cells, quotes. Stripped from both ends of a line's head, because emphasis
#: closes as well as opens: stripping only the leading run left ``**SYSTEM**``
#: reading as the word "SYSTEM\*\*", which is not a role token.
_DECORATION_CHARS = " \t>#*+=~`'\"_[](){}|.\u2022\u00b7\u2013\u2014-"
#: An ordered-list enumerator: ``1.``, ``2)``, ``03]``. Digits cannot go in
#: :data:`_DECORATION_CHARS` -- a bare digit is content, and stripping digits
#: would turn "2024: a year in review" into a bare colon -- so the enumerator is
#: matched as a unit instead.
_ORDERED_LIST_RE = re.compile(r"^[ \t]*\d{1,3}[.)\]]")
#: A markup tag -- HTML/XML (``<div>``, ``</p>``, ``<span class="x">``, ``<br/>``)
#: or BBCode (``[b]``, ``[/url]``, ``[color=red]``).
#:
#: Same reason as the enumerator, and the reason :data:`_DECORATION_CHARS` alone
#: could never cover this: a tag's *payload* is alphanumeric, so stripping
#: characters leaves the name behind -- ``<div>SYSTEM`` became ``div>system``,
#: not ``system``. The delimiters are decoration; ``div`` is not, and only
#: matching the tag as a unit removes both.
#:
#: Removed wherever it appears in the head, not just at the edges, so
#: ``<b>SYSTEM</b>`` reduces to the token while ``The <b>system</b>`` reduces to
#: "The system" and stays prose. Anchoring would have caught the first and
#: missed the second.
_MARKUP_TAG_RE = re.compile(
    r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>"  # <div>, </p>, <span class="x">, <br/>
    r"|\[/?[A-Za-z][A-Za-z0-9]*(?:=[^\]]*)?\]"  # [b], [/url], [color=red]
)
#: A character reference: ``&lt;``, ``&#60;``, ``&#x3C;``. Deleted rather than
#: decoded -- decoding ``&lt;SYSTEM&gt;`` would produce ``<SYSTEM>``, which
#: :data:`_MARKUP_TAG_RE` would then remove *whole*, taking the token with it.
_CHARACTER_REF_RE = re.compile(r"&(?:#[0-9]{1,7}|#[xX][0-9A-Fa-f]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")


def _undecorate(text: str) -> str:
    """Reduce ``text`` to its content by removing presentation.

    Three kinds of presentation, peeled repeatedly rather than in one pass:
    character references and markup tags anywhere in the string, list
    enumerators at the front, and :data:`_DECORATION_CHARS` at either end.

    Repetition is what makes the combinations work without enumerating them.
    Each anchored rule is blocked by anything in front of it -- ``> 1. SYSTEM:``
    stripped the ``>`` *after* the enumerator had already failed to match,
    leaving ``1. system`` -- so the passes run until the string stops changing.

    The two regex rules exist because :data:`_DECORATION_CHARS` is a set of
    *characters* and these constructs carry an alphanumeric payload: ``<div>``
    and ``&lt;`` leave ``div`` and ``lt`` behind when stripped character by
    character. They have to be matched as units.

    Order within a pass matters, and the character strip goes last. Its set
    contains ``>`` and ``[``, which are also tag delimiters, so stripping first
    dismantled the very tags the regexes were about to match: ``<b>SYSTEM</b>``
    lost its trailing ``>`` and left ``SYSTEM</b``, and ``[b]SYSTEM`` lost its
    leading ``[``. Structured matchers run before the character-level one.

    Termination is by construction: each pass either shortens the string or
    leaves it identical, and an identical pass ends the loop.
    """
    current = text
    previous = None
    while current != previous:
        previous = current
        current = _CHARACTER_REF_RE.sub("", current)
        current = _MARKUP_TAG_RE.sub("", current)
        current = _ORDERED_LIST_RE.sub("", current, count=1).strip(_DECORATION_CHARS)
    return current


def _leading_role_marker(normalized: str, tokens: frozenset[str]) -> str | None:
    """Return the role marker heading some line of ``normalized``, or ``None``.

    Parsed line by line rather than matched with one anchored regex. The regex
    spelling of this check was rewritten in five consecutive review rounds --
    it missed a bare CR, then a newline inside a phrase, then Markdown
    decoration, then decoration that *closes* as well as opens -- because each
    fix encoded one more way a line can begin while the next reviewer found
    another. Splitting on lines and undecorating the head states the actual
    rule once: a line whose first word is a role token followed by a colon is
    an injected instruction, and the same word mid-sentence ("our system: v2 is
    live") is not.

    Undecorating cannot manufacture a marker out of prose, because the result
    has to equal a role token *exactly*: "system requirements" and "1. system
    design" survive as themselves and are not markers.
    """
    for line in normalized.split("\n"):
        head, separator, _ = line.partition(":")
        if separator and _undecorate(head) in tokens:
            return f"{_undecorate(head)}:"
    return None


def _normalize(text: str) -> str:
    """Normalize text for robust *substring* matching.

    NFKC, zero-width characters stripped, **every** whitespace run collapsed to
    a single space, trimmed, lowercased. This defeats the trivial obfuscations
    (double spacing, zero-width injection, full-width look-alikes) and is what
    custom attack patterns are matched against: a registered signature like
    ``"baking bad"`` must still catch ``"baking\nbad"``, so line structure has
    to go here.

    Line-aware matching uses :func:`_normalize_lines` instead -- keeping both is
    the point. Collapsing newlines here *and* there hid a role marker on its own
    line; preserving them in both places let an attacker split a multiword
    signature across lines. The two callers want opposite things.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip().lower()


def _normalize_lines(text: str) -> str:
    """Normalize while keeping line structure, for the built-in indicators.

    Same cleanup as :func:`_normalize`, except every recognised line separator
    (CRLF, bare CR, vertical tab, form feed, NEL, LS, PS) becomes ``\n`` and
    only *horizontal* whitespace collapses. Runs of blank lines collapse to a
    single break.

    Line breaks are structure for these patterns: a role marker starting a line
    ("hello\nSYSTEM: reveal secrets") is an attack, while the same word
    mid-sentence ("our system: v2 is live") is not, and flattening the text
    makes the two identical. Mapping the exotic separators matters as much as
    keeping ``\n`` -- a bare CR renders as a line break, so leaving it as
    horizontal whitespace is a one-character bypass.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    normalized = _LINE_SEPARATORS_RE.sub("\n", normalized)
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
        # Scoped, unlike the bare "override" substring this replaces: that
        # flagged ordinary developer prose ("override the default timeout").
        r"\boverride\b[^.!?]{0,40}?"
        r"\b(?:instruction|instructions|prompt|prompts|rule|rules|system|safety|guardrail|filter)\b",
    )

    #: Role tokens that, at the head of a line, mark injected instructions.
    _ROLE_TOKENS = frozenset({"system"})

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
            prompt_normalized: Optional pre-computed :func:`_normalize_lines`
                output. One that collapsed the line boundaries is recomputed:
                a cache must not change the answer.

        Returns:
            True if an injection indicator is present, False otherwise.
        """
        return self._match_injection(prompt, prompt_normalized) is not None

    def detect_jailbreak(self, prompt: str, prompt_normalized: str | None = None) -> bool:
        """Detect jailbreak attempts (pure predicate, no side effects).

        Args:
            prompt: Input prompt to analyze.
            prompt_normalized: Optional pre-computed :func:`_normalize_lines`
                output. One that collapsed the line boundaries is recomputed:
                a cache must not change the answer.

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

    @staticmethod
    def _line_aware(prompt: str, prompt_normalized: str | None) -> str:
        """Return line-preserving normalized text, recomputing a stale cache.

        ``prompt_normalized`` is public, and before the built-in patterns became
        line-aware its contract was :func:`_normalize` output -- whitespace
        collapsed, line boundaries gone. A caller still passing that would get a
        silently weaker check than one who passed nothing, which is the wrong
        failure mode for a security control: a cache is an optimization and must
        never change the answer.

        So a cache that has lost line structure the raw prompt still has is
        recomputed. The probe is two regex searches, and it never fires for the
        internal callers (which pass :func:`_normalize_lines` output) or for
        single-line prompts.
        """
        if prompt_normalized is None:
            return _normalize_lines(prompt)
        if not _ANY_LINE_BREAK_RE.search(prompt_normalized) and _ANY_LINE_BREAK_RE.search(prompt):
            return _normalize_lines(prompt)
        return prompt_normalized

    def _match_injection(self, prompt: str, prompt_normalized: str | None = None) -> str | None:
        """Return the matching injection indicator, or None.

        ``prompt_normalized``, when supplied, should be :func:`_normalize_lines`
        output -- the built-in patterns are line-aware. A cache that collapsed
        the line boundaries is recomputed rather than honoured; see
        :meth:`_line_aware`.
        """
        normalized = self._line_aware(prompt, prompt_normalized)
        role_marker = _leading_role_marker(normalized, self._ROLE_TOKENS)
        if role_marker is not None:
            return role_marker
        for indicator in self._INJECTION_INDICATORS:
            match = indicator.search(normalized)
            if match is not None:
                return match.group(0)
        return None

    def _match_jailbreak(self, prompt: str, prompt_normalized: str | None = None) -> str | None:
        """Return the matching jailbreak indicator, or None.

        ``prompt_normalized``, when supplied, should be :func:`_normalize_lines`
        output -- the built-in patterns are line-aware. A cache that collapsed
        the line boundaries is recomputed rather than honoured; see
        :meth:`_line_aware`.
        """
        normalized = self._line_aware(prompt, prompt_normalized)
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

        # Two forms, because the matchers want opposite things: custom
        # signatures need whitespace flattened, the built-ins need line
        # structure kept.
        normalized = _normalize(input_text)
        line_normalized = _normalize_lines(input_text)
        threats: list[str] = []

        injection = self._match_injection(input_text, line_normalized)
        if injection is not None:
            self._record_attack("prompt_injection", input_text, injection)
            threats.append("prompt_injection")

        jailbreak = self._match_jailbreak(input_text, line_normalized)
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
