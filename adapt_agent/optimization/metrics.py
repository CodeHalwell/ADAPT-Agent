"""Scoring metrics for agent evaluation.

Every metric maps an agent ``output`` and an ``expected`` reference to a float in
``[0, 1]`` where higher is better. Plain ``Callable[[output, expected], float]``
functions work directly (so the existing
:class:`~adapt_agent.evaluation.AgentEvaluator` style keeps working), while the
:class:`Metric` wrapper adds a name and an opt-in to receive the full
:class:`~adapt_agent.optimization.dataset.Example` -- needed by reference-free,
context-aware metrics such as the LLM-as-judge
(:meth:`~adapt_agent.optimization.judge.LLMJudge.as_metric`).

All built-ins are pure-Python and dependency-free.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import re
import unicodedata
from collections.abc import Callable, Sequence
from typing import Any

from .extractors import extract_output_text
from .retry import note_declared_fallback

try:  # Python 3.11+ moved the regex parser; the old name warns there.
    import re._constants as _sre_constants  # type: ignore[import-not-found,unused-ignore]
    import re._parser as _sre_parse  # type: ignore[import-not-found,unused-ignore]
except ImportError:  # pragma: no cover - Python 3.10
    import sre_constants as _sre_constants  # type: ignore[no-redef,unused-ignore]
    import sre_parse as _sre_parse  # type: ignore[no-redef,unused-ignore]

#: A bare metric function. Either ``(output, expected)`` or, for example-aware
#: metrics wrapped in :class:`Metric`, ``(output, expected, example)``.
MetricFn = Callable[..., float]


class Metric:
    """A named scoring function with optional access to the example.

    Args:
        name: Stable identifier used as the key in evaluation reports.
        fn: The scoring callable returning a float (clamped to ``[0, 1]``).
        needs_example: When ``True`` the harness calls ``fn(output, expected,
            example)``; otherwise ``fn(output, expected)``.
        structural: Marks a metric that scores a *structured* output (a mapping)
            rather than text. :func:`~adapt_agent.optimization.evals.evaluate_agent`
            reads this to pick
            :func:`~adapt_agent.optimization.extractors.extract_output_payload`
            over ``extract_output_text``, so the structure a structural metric
            needs is not flattened away before it runs.
        on_error: Score to record when this metric fails *permanently* -- it
            raised, and the harness classified the error as a real failure
            rather than throttling. ``None`` means no opinion, and the harness
            records ``0.0``.

            This exists because a metric can have a documented fallback of its
            own: :meth:`~adapt_agent.optimization.judge.LLMJudge.as_metric`
            sets it from the judge's ``on_error``, so a judge configured with
            ``on_error=0.7`` scores a grading failure the same whether it is
            called directly or through a harness. A transient failure ignores
            this: that row is excluded from the aggregate, so no score it could
            carry would be read.
    """

    def __init__(
        self,
        name: str,
        fn: MetricFn,
        *,
        needs_example: bool = False,
        structural: bool = False,
        on_error: float | None = None,
    ):
        self.name = name
        self.fn = fn
        self.needs_example = needs_example
        self.structural = structural
        self.on_error = None if on_error is None else _clamp(on_error)
        #: The :class:`~adapt_agent.optimization.judge.LLMJudge` behind this
        #: metric, when it is model-graded (set by ``LLMJudge.as_metric``).
        self.judge: Any = None

    def __call__(self, output: Any, expected: Any, example: Any = None) -> float:
        try:
            if self.needs_example:
                raw = self.fn(output, expected, example)
            else:
                raw = self.fn(output, expected)
            # NaN compares false against both clamp bounds, so `max(0, min(1,
            # nan))` used to come back as 1.0 -- a metric that returned NaN on
            # every row gave the run a perfect score. A non-finite value is a
            # defect in the metric, so it fails the row loudly instead.
            _require_finite(self.name, raw)
        except Exception as exc:
            # The fallback belongs to the failure, so it travels with it. A
            # metric can dispatch to another -- `checks` routes each row to the
            # scorer that row declares -- and the harness only ever sees the
            # outermost one, whose `on_error` says nothing about the metric
            # that actually raised. Noting it here covers any depth of
            # wrapping, and `note_declared_fallback` keeps the innermost:
            # the metric nearest the failure is the one that answers for it.
            if self.on_error is not None:
                note_declared_fallback(exc, self.on_error)
            raise
        return _clamp(raw)

    def renamed(self, name: str) -> Metric:
        """This metric under a different reporting name, everything else intact.

        The mapping form of ``metrics`` renames each entry after its key, and
        both places that do it used to rebuild the metric by hand, listing the
        fields to carry. A list of fields beside a constructor that owns them
        goes stale the first time the constructor grows: ``structural`` was
        dropped that way once -- a renamed ``field_match`` scored a
        model-returning agent ``0.0`` -- and ``on_error`` was dropped the same
        way the round it was added, turning a judge's declared ``0.7`` fallback
        back into a hard zero.

        So the copy is whole rather than enumerated, and a field added later is
        carried without anyone remembering to. Copying also keeps a subclass a
        subclass, which rebuilding through ``Metric(...)`` did not.
        """
        clone = copy.copy(self)
        clone.name = name
        return clone

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Metric(name={self.name!r}, needs_example={self.needs_example})"


def coerce_metric(metric: Metric | MetricFn, *, default_name: str = "metric") -> Metric:
    """Normalise a metric or bare callable into a :class:`Metric`."""
    if isinstance(metric, Metric):
        return metric
    if callable(metric):
        name = getattr(metric, "__name__", default_name) or default_name
        return Metric(name, metric)
    raise TypeError(f"Expected a Metric or callable, got {type(metric)!r}")


# -- text normalisation -------------------------------------------------------

#: Punctuation peeled off the *start* of a token: opening quotes and brackets.
_LEADING_PUNCT = "\"'“‘«‹([{¿¡「『"
#: Punctuation peeled off the *end* of a token: sentence punctuation, closing
#: quotes and brackets. Deliberately an explicit list, not a Unicode category:
#: ``%`` is category ``Po`` and ``-`` is ``Pd``, and stripping either changes
#: what an answer *means* (``100%`` vs ``$100``, ``-5`` vs ``5``).
_TRAILING_PUNCT = ".,;:!?\"'”’»›)]}…。、，！？：；」』"

#: Characters of scripts written without spaces between words. Each becomes its
#: own token, so token metrics measure *something* on Japanese or Chinese text
#: instead of comparing one giant "word" per sentence.
_UNSEGMENTED_CHAR = re.compile("[぀-ヿ㐀-䶿一-鿿豈-﫿฀-๿຀-໿က-႟]")

_PUNCT_MODES = ("edge", "all", "none")


def _to_text(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _punct_mode(strip_punct: bool | str) -> str:
    if strip_punct is True:
        return "edge"
    if strip_punct is False:
        return "none"
    if strip_punct not in _PUNCT_MODES:
        raise ValueError(
            f"strip_punct must be a bool or one of {_PUNCT_MODES}, got {strip_punct!r}"
        )
    return str(strip_punct)


def _strip_edge_punct(token: str) -> str:
    return token.lstrip(_LEADING_PUNCT).rstrip(_TRAILING_PUNCT)


def normalize_text(
    text: Any,
    *,
    lower: bool = True,
    strip_punct: bool | str = True,
    unicode: bool = True,
) -> str:
    """Canonicalise text for robust comparison.

    Args:
        text: Value to normalise (non-strings are ``str()``-ed; ``None`` is ``""``).
        lower: Case-fold (``"STRASSE"`` and ``"straße"`` compare equal).
        strip_punct: How punctuation is handled. ``True`` / ``"edge"`` (the
            default) strips sentence punctuation and quotes/brackets from the
            *edges of each token* -- ``"Paris."`` equals ``"paris"`` -- but
            keeps symbols that carry meaning: ``-5`` is not ``5``, ``<=`` is
            not ``>=``, ``C++`` is not ``C``, ``$100`` is not ``100%``.
            ``"all"`` replaces every non-word character with a space (the
            pre-0.6 behaviour, which made all of those pairs equal); ``False`` /
            ``"none"`` keeps punctuation untouched.
        unicode: Apply NFKC normalisation first, so composed and decomposed
            ``café``, and full-width and ASCII letters, compare equal.
    """
    s = _to_text(text)
    if unicode:
        s = unicodedata.normalize("NFKC", s)
    if lower:
        s = s.casefold()
    mode = _punct_mode(strip_punct)
    if mode == "all":
        s = re.sub(r"[^\w\s]", " ", s)
        return re.sub(r"\s+", " ", s).strip()
    tokens = s.split()
    if mode == "edge":
        tokens = [t for t in (_strip_edge_punct(tok) for tok in tokens) if t]
    return " ".join(tokens)


def _tokens(text: Any, *, strip_punct: bool | str = True) -> list[str]:
    tokens: list[str] = []
    for token in normalize_text(text, strip_punct=strip_punct).split():
        if _UNSEGMENTED_CHAR.search(token):
            # Split unsegmented script runs into single characters; keep any
            # embedded Latin/number run whole.
            tokens.extend(
                piece
                for piece in re.findall(
                    f"{_UNSEGMENTED_CHAR.pattern}|[^{_UNSEGMENTED_CHAR.pattern[1:-1]}]+", token
                )
                if piece
            )
        else:
            tokens.append(token)
    return tokens


def _clamp(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        # NaN slips through `max`/`min` (every comparison with it is False),
        # which turned NaN into 1.0. A value with no meaning is no score.
        return 0.0
    return max(0.0, min(1.0, f))


class NonFiniteScoreError(ValueError):
    """A metric returned NaN or infinity, which is a defect in the metric."""


def _require_finite(name: str, raw: Any) -> None:
    if isinstance(raw, bool) or raw is None:
        return
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return
    if not math.isfinite(f):
        raise NonFiniteScoreError(f"metric {name!r} returned a non-finite score ({raw!r})")


# -- untrusted regular expressions --------------------------------------------

#: Longest pattern accepted from a dataset row. Patterns written in code
#: (``regex_match(pattern=...)``) are the author's and are not limited.
MAX_ROW_PATTERN_LENGTH = 512


def check_untrusted_pattern(pattern: str) -> None:
    """Refuse a regex from untrusted input that could backtrack catastrophically.

    Python's ``re`` has no timeout, so a dataset row carrying ``(a+)+$`` hangs
    the whole evaluation on a modest input. This is a static screen, not a
    proof of safety: it rejects patterns over :data:`MAX_ROW_PATTERN_LENGTH`
    and any *quantified* sub-pattern that itself contains a quantifier or an
    alternation -- ``(a+)+``, ``(a*)*``, ``(a|aa)+`` -- which are the shapes
    behind exponential backtracking. Bounded repeats such as ``(ab){2}`` and
    plain quantifiers such as ``\\d+`` pass.

    Raises:
        ValueError: If the pattern is too long, does not compile, or is
            rejected as potentially catastrophic.
    """
    if len(pattern) > MAX_ROW_PATTERN_LENGTH:
        raise ValueError(
            f"row-supplied regex is {len(pattern)} characters, over the "
            f"{MAX_ROW_PATTERN_LENGTH}-character limit"
        )
    try:
        parsed = _sre_parse.parse(pattern)
    except re.error as exc:
        raise ValueError(f"row-supplied regex does not compile: {exc}") from exc
    if _has_nested_repeat(parsed, inside_repeat=False):
        raise ValueError(
            f"row-supplied regex {pattern!r} nests a quantifier or alternation inside a "
            "quantified group, which can backtrack catastrophically; rewrite it "
            "without the nesting, or pass the pattern in code via regex_match(pattern=...)"
        )


def _has_nested_repeat(subpattern: Any, *, inside_repeat: bool) -> bool:
    repeat_ops = {_sre_constants.MAX_REPEAT, _sre_constants.MIN_REPEAT}
    possessive = getattr(_sre_constants, "POSSESSIVE_REPEAT", None)
    if possessive is not None:
        repeat_ops.add(possessive)
    for op, av in subpattern:
        if op in repeat_ops:
            _low, high, body = av
            unbounded = high is _sre_constants.MAXREPEAT or high > 1
            if unbounded and inside_repeat:
                return True
            if _has_nested_repeat(body, inside_repeat=inside_repeat or unbounded):
                return True
        elif op is _sre_constants.SUBPATTERN:
            if _has_nested_repeat(av[-1], inside_repeat=inside_repeat):
                return True
        elif op is _sre_constants.BRANCH:
            if inside_repeat:
                return True
            if any(_has_nested_repeat(b, inside_repeat=False) for b in av[1]):
                return True
        elif op in (_sre_constants.ASSERT, _sre_constants.ASSERT_NOT):
            if _has_nested_repeat(av[1], inside_repeat=inside_repeat):
                return True
        elif getattr(_sre_constants, "ATOMIC_GROUP", None) is op:
            if _has_nested_repeat(av, inside_repeat=inside_repeat):
                return True
    return False


# -- built-in metric factories ------------------------------------------------


def exact_match(*, normalize: bool = True, punctuation: bool | str = "edge") -> Metric:
    """1.0 iff output equals expected (optionally after normalisation).

    Normalisation (the default) applies NFKC, case-folding, whitespace
    collapsing and *edge* punctuation stripping -- see :func:`normalize_text`.
    ``punctuation="all"`` restores the older rule that erased every symbol
    (``"-5"`` equal to ``"5"``); ``normalize=False`` compares raw values.
    """
    mode = _punct_mode(punctuation)

    def _fn(output: Any, expected: Any) -> float:
        if normalize:
            same = normalize_text(output, strip_punct=mode) == normalize_text(
                expected, strip_punct=mode
            )
            return 1.0 if same else 0.0
        return 1.0 if output == expected else 0.0

    return Metric("exact_match", _fn)


def contains(*, normalize: bool = True, word_boundary: bool | None = None) -> Metric:
    """1.0 iff the expected reference appears in the output.

    With ``word_boundary=True`` (the default when normalising) the match must
    start and end on a
    word boundary, so ``"no"`` is not found inside ``"know"`` and ``"safe"`` is
    not found inside ``"unsafe"``. A boundary next to a character of an
    unsegmented script (Japanese, Chinese, Thai) is not required, since those
    scripts have no spaces to bound on. ``word_boundary=False`` is a plain
    substring test, and is the default with ``normalize=False``, which asks for
    the raw comparison.
    """
    bounded = normalize if word_boundary is None else word_boundary

    def _fn(output: Any, expected: Any) -> float:
        out = normalize_text(output) if normalize else _to_text(output)
        exp = normalize_text(expected) if normalize else _to_text(expected)
        if not exp:
            return 0.0
        if not bounded:
            return 1.0 if exp in out else 0.0
        before = "" if _UNSEGMENTED_CHAR.match(exp[0]) or not _is_word(exp[0]) else r"(?<!\w)"
        after = "" if _UNSEGMENTED_CHAR.match(exp[-1]) or not _is_word(exp[-1]) else r"(?!\w)"
        return 1.0 if re.search(before + re.escape(exp) + after, out) else 0.0

    return Metric("contains", _fn)


def _is_word(char: str) -> bool:
    return bool(re.match(r"\w", char))


def regex_match(pattern: str | None = None, *, flags: int = re.IGNORECASE) -> Metric:
    """1.0 iff the output matches a regex.

    With an explicit ``pattern`` the expected value is ignored. Without one, the
    per-example ``expected`` value is compiled as the pattern (handy for datasets
    that carry a different acceptance regex per row).

    A row-supplied pattern is untrusted input: it is screened by
    :func:`check_untrusted_pattern` (length cap, no nested quantifiers) before
    it is compiled. A row with no pattern -- ``expected`` is ``None`` or empty
    -- is an error for that row, not a pass: the empty pattern matches every
    output, so scoring it 1.0 would hide an unlabeled row as a success.
    """
    compiled = re.compile(pattern, flags) if pattern is not None else None
    cache: dict[str, re.Pattern[str]] = {}

    def _fn(output: Any, expected: Any) -> float:
        rx = compiled
        if rx is None:
            source = _to_text(expected)
            if not source:
                raise ValueError(
                    "regex_match has no pattern for this row: expected is empty and no "
                    "pattern= was given"
                )
            rx = cache.get(source)
            if rx is None:
                check_untrusted_pattern(source)
                rx = re.compile(source, flags)
                if len(cache) < 1024:
                    cache[source] = rx
        return 1.0 if rx.search(_to_text(output)) else 0.0

    return Metric("regex_match", _fn)


def token_f1() -> Metric:
    """Token-overlap F1 between output and expected (SQuAD-style).

    Text in scripts written without spaces (Japanese, Chinese, Thai) is
    tokenised per character, so a reordered Japanese sentence scores on its
    overlap instead of 0.0.
    """

    def _fn(output: Any, expected: Any) -> float:
        pred, gold = _tokens(output), _tokens(expected)
        if not pred and not gold:
            return 1.0
        if not pred or not gold:
            return 0.0
        gold_counts: dict[str, int] = {}
        for t in gold:
            gold_counts[t] = gold_counts.get(t, 0) + 1
        overlap = 0
        seen: dict[str, int] = {}
        for t in pred:
            seen[t] = seen.get(t, 0) + 1
            if seen[t] <= gold_counts.get(t, 0):
                overlap += 1
        if overlap == 0:
            return 0.0
        precision = overlap / len(pred)
        recall = overlap / len(gold)
        return 2 * precision * recall / (precision + recall)

    return Metric("token_f1", _fn)


def jaccard() -> Metric:
    """Jaccard similarity over the set of tokens."""

    def _fn(output: Any, expected: Any) -> float:
        a, b = set(_tokens(output)), set(_tokens(expected))
        if not a and not b:
            return 1.0
        union = a | b
        return len(a & b) / len(union) if union else 0.0

    return Metric("jaccard", _fn)


_NUMBER_PICKS = ("auto", "first", "last", "any")


def numeric_close(
    *, tolerance: float = 1e-6, relative: bool = False, which: str = "auto"
) -> Metric:
    """1.0 iff a number in the output is within tolerance of expected's.

    Numbers are read in the forms models write them: ``42``, ``-3.5``, ``.5``,
    ``1e3``, ``1,000`` (thousands separator) and ``3,14`` (decimal comma),
    including full-width and other Unicode digits.

    Args:
        tolerance: Absolute bound (or relative, with ``relative=True``).
        relative: Scale ``tolerance`` by ``|expected|``.
        which: Which number in the *output* is the answer. ``"auto"`` (the
            default) takes the number after the last answer marker (``answer``,
            ``result``, ``total``, ``=``) when there is one, else the last
            number -- so ``"Step 1: the answer is 42"`` reads 42, not 1.
            ``"first"`` and ``"last"`` do what they say (``"first"`` was the
            only behaviour before 0.6); ``"any"`` passes if any number matches
            (lenient -- an output listing many numbers can pass by accident).
    """
    if which not in _NUMBER_PICKS:
        raise ValueError(f"which must be one of {_NUMBER_PICKS}, got {which!r}")

    def _fn(output: Any, expected: Any) -> float:
        exp_num = _extract_number(expected)
        if exp_num is None or not math.isfinite(exp_num):
            return 0.0
        bound = tolerance * (abs(exp_num) if relative else 1.0)
        candidates = _output_numbers(output, which)
        return 1.0 if any(abs(n - exp_num) <= bound for n in candidates) else 0.0

    return Metric("numeric_close", _fn)


def json_subset(*, strict_types: bool = True) -> Metric:
    """Fraction of the expected mapping's top-level keys matched in the output.

    Matching is *recursive*: a nested expected mapping matches when it is a
    subset of the output's value at that key (``{"a": {"b": 1}}`` is matched by
    ``{"a": {"b": 1, "c": 2}}``); lists must have equal length with each element
    matching in turn. With ``strict_types=True`` (the default) ``True`` does
    not match ``1``. A JSON string output is parsed, including one in a
    ```` ```json ```` fence or embedded in surrounding prose.
    """

    def _fn(output: Any, expected: Any) -> float:
        out = _as_mapping(output)
        exp = _as_mapping(expected)
        if out is None or exp is None or not exp:
            return 0.0
        matched = sum(
            1 for k, v in exp.items() if k in out and _subset_match(v, out[k], strict_types)
        )
        return matched / len(exp)

    return Metric("json_subset", _fn, structural=True)


def _subset_match(expected: Any, actual: Any, strict_types: bool) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            k in actual and _subset_match(v, actual[k], strict_types) for k, v in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return False
        return all(_subset_match(e, a, strict_types) for e, a in zip(expected, actual, strict=True))
    if strict_types and (isinstance(expected, bool) != isinstance(actual, bool)):
        return False
    return bool(expected == actual)


def field_match(
    field: str,
    *,
    check: str = "exact_match",
    name: str | None = None,
    missing: float = 0.0,
    **options: Any,
) -> Metric:
    """Score **one field** of a structured output, reported under its own name.

    A single blended number hides which column moved. Scoring four fields
    separately turns a report's ``aggregate`` into the per-column table you
    actually reason about::

        {"lane": 0.94, "matter": 0.90, "action": 0.63, "pack": 0.0}

    That ``pack: 0.0`` is a column the agent never gets right; averaged into one
    score it would read as a mild dip. It also removes the main reason to reach
    for a judge on structured output -- a deterministic field check is both
    cheaper and more correct than asking an LLM whether two labels agree.

    Pair with
    :func:`~adapt_agent.optimization.extractors.extract_output_payload`, which
    keeps the structure that ``extract_output_text`` would flatten.

    Args:
        field: Key to compare on both sides.
        check: Name of the built-in metric used to compare the two values
            (``exact_match`` by default; ``numeric_close``, ``token_f1``, ...).
        name: Reporting name. Defaults to ``field``, which is what makes the
            aggregate read as a column table.
        missing: Score when either side lacks the field. ``0.0`` by default --
            a field the agent failed to emit is a failure, not a skip.
        **options: Forwarded to the inner check's factory, e.g.
            ``field_match("total", check="numeric_close", tolerance=0.01)``.

    Raises:
        ValueError: If ``check`` is unknown, or is ``field_match`` itself.
    """
    if check == "field_match":
        raise ValueError("field_match cannot nest inside itself; pick a scalar check")
    factory = BUILTIN_METRICS.get(check)
    if factory is None:
        raise ValueError(f"Unknown check {check!r}. Available: {sorted(BUILTIN_METRICS)}")
    inner = factory(**options)

    def _fn(output: Any, expected: Any) -> float:
        out, exp = _as_mapping(output), _as_mapping(expected)
        if out is None or exp is None or field not in exp or field not in out:
            return missing
        return inner(out[field], exp[field])

    return Metric(name or field, _fn, structural=True)


def field_metrics(fields: Sequence[str], **options: Any) -> list[Metric]:
    """Build one :func:`field_match` metric per field.

    ``metrics=field_metrics(["lane", "matter", "action", "pack"])`` replaces four
    hand-written closures, and the report comes back keyed by column.
    """
    return [field_match(field, **options) for field in fields]


def levenshtein_ratio() -> Metric:
    """Normalised edit-distance similarity in ``[0, 1]`` (1.0 == identical)."""

    def _fn(output: Any, expected: Any) -> float:
        a, b = normalize_text(output), normalize_text(expected)
        if not a and not b:
            return 1.0
        dist = _levenshtein(a, b)
        longest = max(len(a), len(b))
        return 1.0 - dist / longest if longest else 1.0

    return Metric("levenshtein_ratio", _fn)


def checks(
    *,
    default: Metric | MetricFn | str | None = "exact_match",
    judge: Any = None,
    aggregate: str = "min",
) -> Metric:
    """Per-example check dispatch: each dataset row declares how it is scored.

    Golden datasets often mix answer types -- one row wants an exact text match,
    the next a number within tolerance, another an LLM-judge grade. This metric
    reads the check specification from ``example.metadata["check"]`` (or
    ``"checks"``) and applies the matching scorer to that row:

    * a built-in name -- ``"exact_match"``, ``"numeric_close"``, ...
    * a parameterised form -- ``{"name": "numeric_close", "tolerance": 0.5}``
      (extra keys are passed to the metric factory)
    * ``"judge"`` / ``"llm_judge"`` -- routed to the supplied ``judge`` (its
      per-example ``criteria`` metadata is honoured as usual)
    * a list of any of the above -- combined per ``aggregate``

    Rows without a declaration fall back to ``default``.

    The dispatcher is marked **structural**, so
    :func:`~adapt_agent.optimization.evals.evaluate_agent` keeps the output's
    structure rather than flattening it to text: a row asking for
    ``field_match`` needs the fields, and which row asks is only known at call
    time. Rows asking for a text check still get text -- each one is flattened
    individually as it is scored. The one case this cannot rescue is ``checks``
    *mixed* with a plain text metric, where the all-or-nothing rule picks text
    extraction for the whole run; pass
    ``output_extractor=extract_output_payload`` if a row needs structure there,
    at the cost of the text metric, which then scores a mapping.

    Args:
        default: Check applied when a row declares none. A built-in name, a
            :class:`Metric`, or a bare callable. ``None`` makes an undeclared
            row an error (scored ``0.0`` by the harness).
        judge: An :class:`~adapt_agent.optimization.judge.LLMJudge` (anything
            with ``as_metric()``) backing rows that declare a judge check.
        aggregate: How multiple checks on one row combine: ``"min"`` (default;
            every check must pass for a perfect score) or ``"mean"``.
    """
    if aggregate not in ("min", "mean"):
        raise ValueError(f"aggregate must be 'min' or 'mean', got {aggregate!r}")
    judge_metric = _judge_check_metric(judge)
    resolved_default = None if default is None else _resolve_check(default, judge_metric, {})
    cache: dict[Any, Metric] = {}

    def _fn(output: Any, expected: Any, example: Any = None) -> float:
        spec: Any = None
        if example is not None:
            meta = getattr(example, "metadata", None) or {}
            spec = meta.get("check", meta.get("checks"))
        if spec is None:
            if resolved_default is None:
                raise ValueError("example declares no check and checks(default=None) was set")
            metric_list = [resolved_default]
        else:
            spec_list = list(spec) if isinstance(spec, (list, tuple)) else [spec]
            metric_list = [_resolve_check(item, judge_metric, cache) for item in spec_list]
        if not metric_list:
            raise ValueError("example declares an empty check list")
        # The dispatcher is marked structural, so the harness hands it the
        # *payload*. A row asking for a text check wants text, so flatten for
        # those and only those -- the row is the only thing that knows, and it
        # only knows at call time. Without this a structured output would score
        # 0.0 against a row's `exact_match`, which is the mirror of the bug
        # marking it structural fixes.
        scores = [
            m(output if m.structural else extract_output_text(output), expected, example)
            for m in metric_list
        ]
        return min(scores) if aggregate == "min" else sum(scores) / len(scores)

    return Metric("checks", _fn, needs_example=True, structural=True)


_JUDGE_CHECK_NAMES = ("judge", "llm_judge")


def _judge_check_metric(judge: Any) -> Metric | None:
    """Coerce the ``judge`` argument of :func:`checks` into a Metric (or None)."""
    if judge is None:
        return None
    as_metric = getattr(judge, "as_metric", None)
    if callable(as_metric):
        return coerce_metric(as_metric("judge"))
    return coerce_metric(judge, default_name="judge")


def _resolve_check(spec: Any, judge_metric: Metric | None, cache: dict[Any, Metric]) -> Metric:
    """Resolve one check specification into a :class:`Metric`."""
    if isinstance(spec, Metric):
        return spec
    if callable(spec):
        return coerce_metric(spec)
    if isinstance(spec, str):
        if spec in _JUDGE_CHECK_NAMES:
            if judge_metric is None:
                raise ValueError(
                    f"example declares check {spec!r} but checks() was built without a judge"
                )
            return judge_metric
        key: Any = spec
        params: dict[str, Any] = {}
    elif isinstance(spec, dict):
        params = dict(spec)
        name = params.pop("name", None) or params.pop("check", None)
        if not isinstance(name, str):
            raise ValueError(f"check mapping needs a 'name' string, got {spec!r}")
        if name in _JUDGE_CHECK_NAMES:
            if judge_metric is None:
                raise ValueError(
                    f"example declares check {name!r} but checks() was built without a judge"
                )
            return judge_metric
        spec = name
        try:
            key = (name, tuple(sorted(params.items())))
        except TypeError:  # unhashable parameter values: skip the cache
            key = None
    else:
        raise TypeError(f"Unsupported check specification: {spec!r}")

    if key is not None and key in cache:
        return cache[key]
    factory = metric_factory(spec)
    if factory is None:
        raise KeyError(f"Unknown check {spec!r}. Available: {available_metric_names()}")
    metric = factory(**params)
    if key is not None:
        cache[key] = metric
    return metric


# -- registry of built-in metric factories for config-driven use --------------
#
# Every factory is callable with no arguments (how :func:`get_metric` uses it);
# most also accept keyword options (how :func:`checks` builds parameterised
# per-row checks such as ``{"name": "numeric_close", "tolerance": 0.5}``).

BUILTIN_METRICS: dict[str, Callable[..., Metric]] = {
    "exact_match": exact_match,
    "contains": contains,
    "regex_match": regex_match,
    "token_f1": token_f1,
    "jaccard": jaccard,
    "numeric_close": numeric_close,
    "json_subset": json_subset,
    "levenshtein_ratio": levenshtein_ratio,
    "field_match": field_match,
    "checks": checks,
}


#: Metric families registered by other modules (trajectory metrics live in
#: :mod:`adapt_agent.optimization.trajectory`). Kept apart from
#: :data:`BUILTIN_METRICS`, whose keys are the text/structure built-ins, so a
#: lookup by name -- :func:`get_metric`, a row's ``"check"`` -- finds both.
EXTENSION_METRICS: dict[str, Callable[..., Metric]] = {}


def register_metric(name: str, factory: Callable[..., Metric]) -> None:
    """Make ``factory`` reachable by ``name`` from :func:`get_metric` and per-row checks."""
    if name in BUILTIN_METRICS:
        raise ValueError(f"{name!r} is a built-in metric name")
    EXTENSION_METRICS[name] = factory


def metric_factory(name: str) -> Callable[..., Metric] | None:
    """The factory registered under ``name`` (built-in or extension), else ``None``."""
    return BUILTIN_METRICS.get(name) or EXTENSION_METRICS.get(name)


def available_metric_names() -> list[str]:
    """Every metric name a dataset row, config or ``get_metric`` may use."""
    return sorted({*BUILTIN_METRICS, *EXTENSION_METRICS})


def get_metric(name: str) -> Metric:
    """Look up a built-in metric by name using default settings.

    Raises:
        KeyError: If ``name`` is not a built-in.
        TypeError: If the metric cannot be built without arguments --
            ``field_match`` needs a ``field``, so it is only reachable through a
            mapping spec such as ``{"name": "field_match", "field": "lane"}``.
    """
    factory = metric_factory(name)
    if factory is None:
        raise KeyError(f"Unknown built-in metric {name!r}. Available: {available_metric_names()}")
    try:
        return factory()
    except TypeError as exc:
        raise TypeError(
            f"Built-in metric {name!r} needs arguments and cannot be built by name alone. "
            f'Use a mapping spec, e.g. {{"name": "{name}", "field": "..."}}, '
            f"or call {name}(...) directly."
        ) from exc


# -- low-level helpers --------------------------------------------------------

#: A number as written in prose. The sign only counts when it is not glued to
#: a preceding word or digit, so the ``-01`` inside ``2024-01-05`` is not read
#: as minus one. Separators are captured raw and resolved by `_parse_number`.
_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+−]?(?:\d+(?:[.,]\d+)*|[.,]\d+)(?:[eE][-+]?\d+)?|\d+(?:[.,]\d+)*"
)

#: Words after which the next number is the answer (``which="auto"``).
_ANSWER_MARKER = re.compile(r"(?:answer|result|total|final|=|≈|is\s*:)", re.IGNORECASE)


def _parse_number(token: str) -> float | None:
    """Resolve one regex match into a float, deciding what each separator means.

    A comma followed by exactly three digits is a thousands separator
    (``1,000``); otherwise, with no dot present, it is a decimal comma
    (``3,14``). With both present, whichever comes last is the decimal mark
    (``1,000.5`` and ``1.000,5``).
    """
    t = token.replace("−", "-")
    match = re.match(r"^(.*?)([eE][-+]?\d+)?$", t)
    if match is None:  # pragma: no cover - the pattern always matches
        return None
    mantissa, exponent = match.group(1), match.group(2) or ""
    sign = ""
    if mantissa[:1] in "+-":
        sign, mantissa = mantissa[0], mantissa[1:]
    has_dot, has_comma = "." in mantissa, "," in mantissa
    if has_dot and has_comma:
        if mantissa.rfind(",") > mantissa.rfind("."):
            mantissa = mantissa.replace(".", "").replace(",", ".")
        else:
            mantissa = mantissa.replace(",", "")
    elif has_comma:
        groups = mantissa.split(",")
        if len(groups) > 1 and all(len(g) == 3 for g in groups[1:]) and groups[0]:
            mantissa = "".join(groups)
        elif len(groups) == 2:
            mantissa = groups[0] + "." + groups[1]
        else:
            return None
    elif mantissa.count(".") > 1:
        groups = mantissa.split(".")
        if all(len(g) == 3 for g in groups[1:]) and groups[0]:
            mantissa = "".join(groups)  # 1.000.000 (dot as thousands separator)
        else:
            return None
    try:
        return float(sign + mantissa + exponent)
    except ValueError:
        return None


def _numbers_in(text: str) -> list[tuple[int, float]]:
    found: list[tuple[int, float]] = []
    for match in _NUMBER_RE.finditer(text):
        value = _parse_number(match.group(0))
        if value is not None and math.isfinite(value):
            found.append((match.start(), value))
    return found


def _extract_number(value: Any) -> float | None:
    """The reference number: the value itself, or the first number in its text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    numbers = _numbers_in(unicodedata.normalize("NFKC", _to_text(value)))
    return numbers[0][1] if numbers else None


def _output_numbers(value: Any, which: str) -> list[float]:
    """The candidate answer number(s) in an output, per ``which``."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)] if math.isfinite(float(value)) else []
    text = unicodedata.normalize("NFKC", _to_text(value))
    numbers = _numbers_in(text)
    if not numbers:
        return []
    if which == "any":
        return [n for _, n in numbers]
    if which == "first":
        return [numbers[0][1]]
    if which == "last":
        return [numbers[-1][1]]
    markers = list(_ANSWER_MARKER.finditer(text))
    if markers:
        cut = markers[-1].end()
        after = [n for pos, n in numbers if pos >= cut]
        if after:
            return [after[0]]
    return [numbers[-1][1]]


def _as_mapping(value: Any) -> dict[str, Any] | None:
    """Coerce a structured output to a mapping, or ``None`` if it is not one.

    Accepts a dict, a JSON string, and a **model or dataclass**. The last matters
    for per-row checks: ``{"check": {"name": "field_match", ...}}`` is dispatched
    by the ``checks`` metric, which cannot be marked structural (the row decides
    at runtime), so automatic extraction leaves a Pydantic AI result as a model
    object. Without this the field would score a false 0.0.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return _mapping_from_text(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return {f.name: getattr(value, f.name, None) for f in dataclasses.fields(value)}
        except Exception:
            return None
    # Any object offering a dump. Deliberately more permissive than the
    # security-side `_structured_fields`, and matched to
    # `extractors._model_to_dict` so the two agree: there, an unconvertible
    # object would silently reach a metric as itself; here the worst case is a
    # surprising score, never a missed screen.
    for attr in ("model_dump", "dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                dumped = method()
            except Exception:
                continue
            if isinstance(dumped, dict):
                return dumped
    return None


def _mapping_from_text(text: str) -> dict[str, Any] | None:
    """Parse a JSON object out of model text: bare, fenced, or embedded in prose.

    Models wrap structured answers in a ```` ```json ```` fence or a sentence
    ("Result: {...}") far more often than they emit bare JSON, and scoring that
    0.0 measures the formatting, not the answer.
    """
    import json

    from .extractors import _parse_json

    parsed = _parse_json(text)
    if isinstance(parsed, dict):
        return parsed
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start : end + 1])
        except ValueError:
            return None
        return obj if isinstance(obj, dict) else None
    return None


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Two-row dynamic programming (O(min(len)) memory).
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


__all__ = [
    "Metric",
    "MetricFn",
    "coerce_metric",
    "normalize_text",
    "exact_match",
    "contains",
    "regex_match",
    "token_f1",
    "jaccard",
    "numeric_close",
    "json_subset",
    "levenshtein_ratio",
    "field_match",
    "field_metrics",
    "checks",
    "BUILTIN_METRICS",
    "EXTENSION_METRICS",
    "available_metric_names",
    "metric_factory",
    "register_metric",
    "NonFiniteScoreError",
    "check_untrusted_pattern",
    "MAX_ROW_PATTERN_LENGTH",
    "get_metric",
]
