"""Firewall for LLM agent security.

.. warning::
    Regex patterns passed to :meth:`Firewall.add_blocked_pattern` and
    :meth:`Firewall.add_allowed_pattern` are **trusted-author-only**: never
    compile patterns derived from untrusted/end-user input. The standard
    library :mod:`re` engine backtracks, and a pattern with an ambiguous
    repetition can take seconds on a 27-character input -- the length cap does
    not bound that, because the cost is exponential in the length.

    Two guards, neither a substitute for author review. Registration rejects
    the ambiguous-repetition shapes that cause catastrophic backtracking (see
    :func:`_backtracking_risk`). And when the third-party ``regex`` module is
    importable, patterns are evaluated with it under a per-search time limit
    (:data:`PATTERN_TIMEOUT`); a search that runs out of time **blocks** the
    content rather than letting it through. Without ``regex`` there is no time
    limit, only the static check.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from re import Pattern
from typing import Any, NamedTuple, cast

from adapt_agent.core.types import AgentMessage, SecurityEvent
from adapt_agent.security._text import escape_for_log, normalize_for_matching

logger = logging.getLogger(__name__)

#: Upper bound on a stored security event's description, after escaping.
_MAX_DESCRIPTION_LENGTH = 10_000
#: How much of the content an event keeps in the clear, after redaction.
_SNIPPET_CHARS = 12
#: How much of an over-length input is scrubbed before its snippet is taken.
#: Scrubbing all of it would run every blocked pattern over exactly the input
#: the length cap exists to keep them away from.
_LENGTH_SCRUB_WINDOW = 1_000
#: Seconds one pattern may spend on one search, when the ``regex`` module is
#: available to enforce it.
PATTERN_TIMEOUT = 1.0


def _load_regex_engine() -> Any:
    """The third-party ``regex`` module, if it is installed and has timeouts."""
    try:
        import regex  # type: ignore[import-untyped, unused-ignore]
    except ImportError:
        return None
    try:
        regex.compile("a").search("a", timeout=PATTERN_TIMEOUT)
    except Exception:  # noqa: BLE001 - too old to take a timeout: do not use it
        return None
    return regex


_REGEX_ENGINE: Any = _load_regex_engine()

#: :mod:`re` flags and their :mod:`regex` equivalents. ``re.ASCII`` is the one
#: whose value differs; a flag outside this table keeps a pattern on :mod:`re`.
_FLAG_NAMES = ("IGNORECASE", "LOCALE", "MULTILINE", "DOTALL", "UNICODE", "VERBOSE", "ASCII")


class _PatternTimeout(Exception):
    """A pattern ran out of time on one input."""


class _CompiledPattern(NamedTuple):
    """A pattern as :mod:`re` compiled it, and as the time-limited engine did."""

    compiled: Pattern[str]
    limited: Any  # a ``regex`` pattern, or None

    def search(self, text: str) -> bool:
        if self.limited is not None:
            try:
                return self.limited.search(text, timeout=PATTERN_TIMEOUT) is not None
            except TimeoutError as exc:
                raise _PatternTimeout from exc
        return self.compiled.search(text) is not None

    def fullmatch(self, text: str) -> bool:
        if self.limited is not None:
            try:
                return self.limited.fullmatch(text, timeout=PATTERN_TIMEOUT) is not None
            except TimeoutError as exc:
                raise _PatternTimeout from exc
        return self.compiled.fullmatch(text) is not None

    def sub(self, replacement: str, text: str) -> str:
        if self.limited is not None:
            try:
                return cast(str, self.limited.sub(replacement, text, timeout=PATTERN_TIMEOUT))
            except TimeoutError as exc:
                raise _PatternTimeout from exc
        return self.compiled.sub(replacement, text)


def _compile(pattern: str, flags: int) -> _CompiledPattern:
    """Compile with :mod:`re` (the reference), and with ``regex`` when available.

    :mod:`re` always compiles first, so a pattern :mod:`re` rejects is rejected
    whatever is installed, and :attr:`Firewall._blocked_patterns` keeps holding
    :class:`re.Pattern` objects. The ``regex`` twin is only used to *evaluate*;
    if it cannot compile the pattern, or the flags have no equivalent there,
    evaluation stays on :mod:`re`.
    """
    compiled = re.compile(pattern, flags)
    limited = None
    if _REGEX_ENGINE is not None:
        translated = 0
        remaining = flags
        for name in _FLAG_NAMES:
            value = int(getattr(re, name))
            if remaining & value:
                translated |= int(getattr(_REGEX_ENGINE, name))
                remaining &= ~value
        if not remaining:
            try:
                limited = _REGEX_ENGINE.compile(pattern, translated)
            except Exception:  # noqa: BLE001 - stay on re for this pattern
                limited = None
    return _CompiledPattern(compiled, limited)


# -- the static backtracking check ---------------------------------------------

try:  # Python 3.11+
    from re import _parser as _sre_parser  # type: ignore[attr-defined, unused-ignore]
except ImportError:  # pragma: no cover - Python 3.10
    import sre_parse as _sre_parser  # type: ignore[no-redef, unused-ignore]

#: Characters a char-set is sampled on when two sets are tested for overlap:
#: all of Latin-1 and Latin Extended, plus a few separators, letters and
#: digits from elsewhere. A sample, so two sets that only share a character
#: outside it are called disjoint -- the check is best-effort.
_PROBE = tuple(chr(c) for c in range(0x250)) + tuple("\u2028\u3000Ж中٣०")
_ALL = frozenset(range(len(_PROBE)))


def _category(name: str, char: str) -> bool:
    if name.endswith("DIGIT"):
        hit = char.isdecimal()
    elif name.endswith("SPACE"):
        hit = char.isspace()
    elif name.endswith("WORD"):
        hit = char.isalnum() or char == "_"
    elif name.endswith("LINEBREAK"):
        hit = char == "\n"
    else:
        hit = True
    return not hit if "_NOT_" in name else hit


def _char_set(op: Any, av: Any, ignore_case: bool) -> frozenset[int]:
    """Indexes into :data:`_PROBE` of the characters one atom can match."""
    if op == _sre_parser.ANY:
        return _ALL
    if op in (_sre_parser.LITERAL, _sre_parser.NOT_LITERAL):
        chars = {chr(av)}
        if ignore_case:
            chars |= {chr(av).lower(), chr(av).upper()}
        hit = frozenset(i for i, c in enumerate(_PROBE) if c in chars)
        return _ALL - hit if op == _sre_parser.NOT_LITERAL else hit
    if op == _sre_parser.IN:
        negate = False
        hit_set: set[int] = set()
        for item_op, item_av in av:
            if item_op == _sre_parser.NEGATE:
                negate = True
            elif item_op == _sre_parser.LITERAL:
                hit_set |= _char_set(_sre_parser.LITERAL, item_av, ignore_case)
            elif item_op == _sre_parser.RANGE:
                lo, hi = item_av
                for i, c in enumerate(_PROBE):
                    variants = {c, c.lower(), c.upper()} if ignore_case else {c}
                    if any(lo <= ord(v) <= hi for v in variants if len(v) == 1):
                        hit_set.add(i)
            elif item_op == _sre_parser.CATEGORY:
                hit_set |= {i for i, c in enumerate(_PROBE) if _category(str(item_av), c)}
            else:
                return _ALL
        hit = frozenset(hit_set)
        return _ALL - hit if negate else hit
    return _ALL


class _Shape(NamedTuple):
    """What a piece of pattern can begin and end with, and if it can be empty."""

    first: frozenset[int]
    last: frozenset[int]
    nullable: bool
    unbounded: bool  # contains a repetition with no upper bound


def _shape(items: Any, ignore_case: bool) -> _Shape:
    """The :class:`_Shape` of a parsed sequence, and every risk inside it.

    Raises:
        ValueError: naming the first ambiguous repetition found.
    """
    first: set[int] = set()
    last: frozenset[int] = frozenset()
    nullable = True
    unbounded = False
    for op, av in items:
        shape = _item_shape(op, av, ignore_case)
        if nullable:
            first |= shape.first
        last = shape.last if not shape.nullable else last | shape.last
        nullable = nullable and shape.nullable
        unbounded = unbounded or shape.unbounded
    return _Shape(frozenset(first), last, nullable, unbounded)


def _item_shape(op: Any, av: Any, ignore_case: bool) -> _Shape:
    p = _sre_parser
    if op in (p.MAX_REPEAT, p.MIN_REPEAT):
        low, high, body = av
        inner = _shape(body, ignore_case)
        is_unbounded = high == p.MAXREPEAT
        if is_unbounded or high >= 8:
            # Ambiguous iteration: the body can end the way it begins (or be
            # empty), *and* contains a repetition of its own, so a run of input
            # splits between the iterations in exponentially many ways --
            # `(a+)+`, `((a+))+`, `(\w+\s?)+`, `(a+){2,}`, `(.*a){12}`.
            if inner.unbounded and (inner.nullable or inner.first & inner.last):
                raise ValueError("a repetition of a repetition that can overlap itself")
            # Ambiguous alternation: two branches that can start alike, under
            # a repetition. The parser factors a shared prefix out of the
            # branches, so `(a|aa)+` arrives as `(a(?:|a))+`: an alternation
            # with an empty branch, in a body that can end the way it begins.
            _check_branches(body, ignore_case)
            if inner.first & inner.last and _has_empty_branch(body):
                raise ValueError("an alternation whose branches can start alike, repeated")
        return _Shape(
            inner.first, inner.last, low == 0 or inner.nullable, is_unbounded or inner.unbounded
        )
    if op == p.SUBPATTERN:
        return _shape(av[-1], ignore_case)
    if op == p.BRANCH:
        shapes = [_shape(branch, ignore_case) for branch in av[1]]
        return _Shape(
            frozenset().union(*(s.first for s in shapes)),
            frozenset().union(*(s.last for s in shapes)),
            any(s.nullable for s in shapes),
            any(s.unbounded for s in shapes),
        )
    if op in (p.AT, p.ASSERT, p.ASSERT_NOT):
        if op != p.AT:
            _shape(av[1], ignore_case)  # a lookaround's body has risks of its own
        return _Shape(frozenset(), frozenset(), True, False)
    if op in (getattr(p, "ATOMIC_GROUP", None), getattr(p, "POSSESSIVE_REPEAT", None)):
        # Never backtracked into, so never ambiguous -- and opaque from outside.
        return _Shape(_ALL, _ALL, True, False)
    if op in (p.LITERAL, p.NOT_LITERAL, p.ANY, p.IN):
        chars = _char_set(op, av, ignore_case)
        return _Shape(chars, chars, False, False)
    # Back-references, conditionals: unknown, so assume they can be anything.
    return _Shape(_ALL, _ALL, True, False)


def _check_branches(body: Any, ignore_case: bool) -> None:
    """Raise when a top-level alternation in ``body`` has overlapping starts."""
    p = _sre_parser
    for op, av in body:
        if op == p.SUBPATTERN:
            _check_branches(av[-1], ignore_case)
        elif op == p.BRANCH:
            seen: frozenset[int] = frozenset()
            for branch in av[1]:
                first = _shape(branch, ignore_case).first
                if first & seen:
                    raise ValueError("an alternation whose branches can start alike, repeated")
                seen |= first


def _has_empty_branch(body: Any) -> bool:
    """Whether ``body`` holds an alternation with a branch that matches nothing."""
    p = _sre_parser
    for op, av in body:
        if op == p.SUBPATTERN and _has_empty_branch(av[-1]):
            return True
        if op == p.BRANCH and any(len(branch) == 0 for branch in av[1]):
            return True
    return False


def _backtracking_risk(pattern: str, flags: int = 0) -> str | None:
    """Why ``pattern`` risks catastrophic backtracking, or ``None``.

    Reads the parsed pattern rather than its text, so nesting, groups and
    spelling cannot hide a shape: the textual check this replaces caught
    ``(a+)+`` and missed ``((a+))+``, ``(a+){2,}``, ``(\\w+\\s?)+`` and
    ``(a+|b)+``, each of which takes seconds on 27 characters.

    The rule is ambiguity, not nesting. ``(?:\\s*,\\s*\\w+)*`` nests
    repetitions and is linear, because each iteration starts with a character
    the previous one cannot end with. A repetition is flagged only when its
    body contains a repetition *and* can end the way it begins -- or when it
    repeats an alternation whose branches can start alike. Possessive
    quantifiers and atomic groups never backtrack and are always accepted.

    Best-effort: character sets are compared on a sample of characters, and
    polynomial (rather than exponential) blow-ups such as ``\\d+\\d+$`` are
    not looked for.
    """
    try:
        parsed = _sre_parser.parse(pattern, flags)
    except re.error:
        return None  # re.compile reports it, with a better message
    try:
        _shape(parsed.data if hasattr(parsed, "data") else parsed, bool(flags & re.IGNORECASE))
    except ValueError as risk:
        return str(risk)
    except RecursionError:
        return "too deeply nested to analyse"
    return None


class Firewall:
    r"""Security firewall for LLM agents.

    Provides input/output filtering, pattern matching, and threat detection
    to protect against malicious inputs and prevent sensitive data leakage.

    Precedence (default, ``whitelist_mode=False``): blocked patterns, custom
    filters and the length cap are checked **first**, so a broad allowed pattern
    can never silently nullify the blocklist. Allowed patterns only short-circuit
    content that has already passed every block check.

    When ``whitelist_mode=True`` a full match against any allowed pattern
    short-circuits to *allowed* **before** the blocked patterns and custom
    filters run (the historical behaviour). Use this only when the allowed
    patterns are themselves the security boundary.

    The length cap is checked first in **both** modes: no allowed pattern can
    waive it.

    With ``normalize=True`` a blocked pattern is matched against the content
    **and** a normalized copy of it -- compatibility forms folded, invisible
    characters and combining marks removed, mixed-script look-alikes folded,
    whitespace runs collapsed (see
    :func:`~adapt_agent.security._text.normalize_for_matching`) -- and blocks if
    either matches. ``(?i)ignore previous instructions`` then also catches a
    double space, a no-break space, a zero-width space or a full-width letter,
    each of which gets past it otherwise. Matching both forms means a pattern
    never matches *less* than it did without normalization. Allowed patterns
    are only ever matched against the content as written: an exemption should
    be exact.

    It is opt-in. Turning it on can only block more, never less, but "more" is
    a behaviour change for an existing deployment's patterns -- ``"z z"`` starts
    blocking ``"z  z"`` -- so it is a decision for the deployment, not a silent
    upgrade. New deployments should turn it on.

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
        normalize: bool = False,
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
            normalize: When True, blocked patterns are also matched against a
                normalized copy of the content, so spacing, invisible
                characters and look-alike letters cannot dodge them. False
                (default) matches the content as written only.
        """
        self._blocked_patterns: list[Pattern] = []
        self._allowed_patterns: list[Pattern] = []
        self._blocked: list[_CompiledPattern] = []
        self._allowed: list[_CompiledPattern] = []
        self._custom_filters: list[Callable[[str], bool]] = []
        self._security_events: deque[SecurityEvent] = deque(maxlen=max_events)
        self._blocked_count = 0
        # Bounded by the number of blocked patterns: keyed by pattern source.
        self._pattern_hits: dict[str, int] = {}
        self.max_content_length = max_content_length
        self.max_events = max_events
        self.whitelist_mode = whitelist_mode
        self.normalize = normalize

    def add_blocked_pattern(self, pattern: str, flags: int = 0) -> None:
        """Add a regex pattern to block.

        Args:
            pattern: Regular expression pattern. Trusted-author-only (ReDoS risk;
                see module docstring).
            flags: Optional regex flags (e.g., re.IGNORECASE)

        Raises:
            ValueError: If the pattern has a catastrophic-backtracking shape.
        """
        self._reject_catastrophic(pattern, flags)
        compiled = _compile(pattern, flags)
        self._blocked.append(compiled)
        self._blocked_patterns.append(compiled.compiled)

    def add_allowed_pattern(self, pattern: str, flags: int = 0) -> None:
        """Add a regex pattern to explicitly allow.

        Args:
            pattern: Regular expression pattern. Trusted-author-only (ReDoS risk;
                see module docstring).
            flags: Optional regex flags (e.g., re.IGNORECASE)

        Raises:
            ValueError: If the pattern has a catastrophic-backtracking shape.
        """
        self._reject_catastrophic(pattern, flags)
        compiled = _compile(pattern, flags)
        self._allowed.append(compiled)
        self._allowed_patterns.append(compiled.compiled)

    @staticmethod
    def _reject_catastrophic(pattern: str, flags: int = 0) -> None:
        """Reject catastrophic-backtracking shapes (best-effort ReDoS guard)."""
        risk = _backtracking_risk(pattern, flags)
        if risk is not None:
            raise ValueError(
                f"Refusing to compile a regex with {risk} "
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
        """Escape every log-unsafe character, bounded to ``max_length`` after escaping.

        See :func:`~adapt_agent.security._text.escape_for_log`: every control
        character, line and paragraph separator and bidi override, never cut in
        the middle of an escape.
        """
        return escape_for_log(text, max_length)

    def _redact_snippet(self, content: str, scrub_window: int | None = None) -> str:
        """Return a non-sensitive snippet for storage in a security event.

        The same form on every path: the content with every blocked pattern
        redacted, cut to its first :data:`_SNIPPET_CHARS` characters and
        escaped, plus a short sha256 of the *original* content for
        correlation. Redaction runs over the whole content before the cut, so
        a secret straddling the cut is redacted rather than half-kept.

        ``scrub_window`` bounds how much is scrubbed, for content the length
        cap has already refused: scrubbing all of it would run every pattern
        over the input the cap exists to keep them from.
        """
        digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:12]
        visible = content if scrub_window is None else content[:scrub_window]
        safe_snippet = self._truncate_escaped(self.sanitize(visible), _SNIPPET_CHARS)
        return f"{safe_snippet}…(sha256:{digest})"

    def _decide(self, content: str) -> tuple[bool, str, str | None]:
        """Decide whether ``content`` is allowed, without recording anything.

        The single source of the firewall's decision order, shared by the
        enforcing path (:meth:`_check`) and the side-effect-free one
        (:meth:`dry_run`) so the two cannot drift apart.

        Returns:
            ``(allowed, reason, detail)``. ``reason`` is one of ``"length"``,
            ``"blocked_pattern"``, ``"pattern_timeout"``, ``"custom_filter"``,
            ``"custom_filter_error"`` (all blocked), ``"allowed_pattern"`` or
            ``"no_match"`` (allowed). ``detail`` is the matching pattern's
            source where there is one.
        """
        # SECURITY: DoS protection by limiting content length. Always first, in
        # both modes: it is what bounds the cost of every regex below, the
        # allowed patterns included. Checking it after the whitelist ran those
        # patterns over unbounded input and let `.*` waive the cap entirely.
        if self.max_content_length is not None and len(content) > self.max_content_length:
            return False, "length", None

        # In strict allowlist mode, a full allowed match short-circuits the
        # remaining checks (blocked patterns and custom filters).
        if self.whitelist_mode:
            allowed = self._allowed_match(content)
            if allowed is not None:
                return True, "allowed_pattern", allowed

        # Check blocked patterns (block-first: these always win over allowed
        # patterns in the default, non-whitelist mode). Against the content
        # and, when normalizing, its normalized form -- either one blocks.
        forms = [content]
        if self.normalize:
            normalized = normalize_for_matching(content)
            if normalized != content:
                forms.append(normalized)
        for pattern in self._blocked:
            try:
                if any(pattern.search(form) for form in forms):
                    return False, "blocked_pattern", pattern.compiled.pattern
            except _PatternTimeout:
                # SECURITY: a pattern that cannot reach a verdict blocks.
                logger.warning("Blocked pattern timed out: %r", pattern.compiled.pattern)
                return False, "pattern_timeout", pattern.compiled.pattern

        # Check custom filters.
        for filter_func in self._custom_filters:
            try:
                if filter_func(content):
                    return False, "custom_filter", None
            except Exception as e:  # noqa: BLE001 - fail closed on any filter error
                # Log error and block on filter failure (fail-closed).
                logger.error("Error in custom filter: %s", e)
                return False, "custom_filter_error", None

        # In the default (block-first) mode, an allowed fullmatch may whitelist
        # content that has already passed every block check. It can never
        # override a block.
        if not self.whitelist_mode:
            allowed = self._allowed_match(content)
            if allowed is not None:
                return True, "allowed_pattern", allowed

        return True, "no_match", None

    def _allowed_match(self, content: str) -> str | None:
        """The source of the first allowed pattern ``content`` fully matches.

        A pattern that runs out of time exempts nothing: not being exempt is
        the closed direction for an allowlist.
        """
        for pattern in self._allowed:
            try:
                if pattern.fullmatch(content):
                    return pattern.compiled.pattern
            except _PatternTimeout:
                logger.warning("Allowed pattern timed out: %r", pattern.compiled.pattern)
        return None

    def dry_run(self, content: str) -> dict[str, Any]:
        """Report what the firewall *would* decide, recording nothing.

        No security event is stored and no counter moves, so this is safe in a
        test or a startup check. Use it to show a pattern is not vacuous -- a
        blocked pattern you added should block at least one realistic example
        and let at least one other through::

            fw.dry_run("my card is 4111111111111111")
            # {'allowed': False, 'reason': 'blocked_pattern', 'pattern': '\\b\\d{16}\\b'}

        Custom filters *are* called (the decision depends on them), so a filter
        with side effects of its own will still have them.

        Returns:
            ``{"allowed": bool, "reason": str, "pattern": str | None}`` with
            ``reason`` as documented on :meth:`_decide`.
        """
        allowed, reason, detail = self._decide(content)
        return {"allowed": allowed, "reason": reason, "pattern": detail}

    def get_pattern_hits(self) -> dict[str, int]:
        """How many times each blocked pattern has blocked content.

        A blocked pattern still at ``0`` after real traffic has never matched
        anything: it may be mistyped, anchored wrongly, or shadowed by an
        earlier pattern that always matches first. The firewall cannot decide
        that for you -- "never matched" is also what a good pattern looks like
        on a quiet day -- but it can make it visible.
        """
        return {
            pattern.pattern: self._pattern_hits.get(pattern.pattern, 0)
            for pattern in self._blocked_patterns
        }

    def _check(self, content: str, event_type: str) -> bool:
        """Core block-first check shared by check_input/check_output.

        Args:
            content: Content to check.
            event_type: Event type label recorded on any security event
                (e.g. "blocked_input" or "blocked_output").

        Returns:
            True if content is allowed, False if blocked.
        """
        allowed, reason, detail = self._decide(content)
        if allowed:
            return True

        if reason == "length":
            self._record_security_event(
                event_type=event_type,
                severity="high",
                description=(
                    f"Content length {len(content)} exceeds maximum allowed "
                    f"{self.max_content_length}"
                ),
                metadata={
                    "content_snippet": self._redact_snippet(content, _LENGTH_SCRUB_WINDOW),
                    "length": len(content),
                },
            )
        elif reason == "blocked_pattern":
            self._pattern_hits[cast(str, detail)] = self._pattern_hits.get(cast(str, detail), 0) + 1
            self._record_security_event(
                event_type=event_type,
                severity="high",
                description=f"Content matched blocked pattern: {detail}",
                metadata={
                    # SECURITY: the same redacted form as every other path --
                    # the whole content scrubbed, then a short prefix and a
                    # hash. A 100-character window with only the match
                    # redacted kept whatever sat around it.
                    "content_snippet": self._redact_snippet(content)
                },
            )
        elif reason == "pattern_timeout":
            self._record_security_event(
                event_type=event_type,
                severity="high",
                description=f"Content blocked: blocked pattern timed out: {detail}",
                metadata={
                    # No scrubbing: running the pattern that just timed out
                    # over the same content again is the one thing not to do.
                    "content_snippet": self._redact_snippet(content, 0),
                },
            )
        elif reason == "custom_filter":
            self._record_security_event(
                event_type=event_type,
                severity="medium",
                description="Content blocked by custom filter",
                metadata={"content_snippet": self._redact_snippet(content)},
            )
        else:  # custom_filter_error
            self._record_security_event(
                event_type=event_type,
                severity="high",
                description="Content blocked due to custom filter error",
                metadata={
                    "content_snippet": self._redact_snippet(content),
                    # SECURITY: never the exception text -- events are read by
                    # people who should not see internals.
                    "error": "An error occurred",
                },
            )
        self._blocked_count += 1
        return False

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

        Matches are replaced in the content as written. A match that only
        exists in the normalized form (see ``normalize``) has no span in the
        original to replace, so it is not redacted here -- the check still
        blocks it. If a pattern runs out of time, the whole content is
        replaced: redaction that cannot finish must not return the original.

        Args:
            content: Content to sanitize
            replacement: Replacement string for blocked patterns

        Returns:
            Sanitized content
        """
        sanitized = content

        for pattern in self._blocked:
            try:
                sanitized = pattern.sub(replacement, sanitized)
            except _PatternTimeout:
                return replacement

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
        events = list(self._security_events)

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
            # Escaped *and* bounded. Slicing first and escaping afterwards
            # bounds the input, not the result: every newline doubles, so a
            # 10,000-character description could be stored at nearly 20,000.
            "description": self._truncate_escaped(description, _MAX_DESCRIPTION_LENGTH),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
        }
        # ⚡ Bolt: deque(maxlen) bounds the ring buffer in O(1) without list.pop(0).
        self._security_events.append(event)
