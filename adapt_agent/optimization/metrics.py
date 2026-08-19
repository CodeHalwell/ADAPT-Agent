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

import re
from collections.abc import Callable
from typing import Any

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
    """

    def __init__(self, name: str, fn: MetricFn, *, needs_example: bool = False):
        self.name = name
        self.fn = fn
        self.needs_example = needs_example

    def __call__(self, output: Any, expected: Any, example: Any = None) -> float:
        if self.needs_example:
            raw = self.fn(output, expected, example)
        else:
            raw = self.fn(output, expected)
        return _clamp(raw)

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


def _to_text(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def normalize_text(text: Any, *, lower: bool = True, strip_punct: bool = True) -> str:
    """Canonicalise text for robust comparison (whitespace/case/punctuation)."""
    s = _to_text(text)
    if lower:
        s = s.lower()
    if strip_punct:
        s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(text: Any) -> list[str]:
    return normalize_text(text).split()


def _clamp(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


# -- built-in metric factories ------------------------------------------------


def exact_match(*, normalize: bool = True) -> Metric:
    """1.0 iff output equals expected (optionally after normalisation)."""

    def _fn(output: Any, expected: Any) -> float:
        if normalize:
            return 1.0 if normalize_text(output) == normalize_text(expected) else 0.0
        return 1.0 if output == expected else 0.0

    return Metric("exact_match", _fn)


def contains(*, normalize: bool = True) -> Metric:
    """1.0 iff the expected reference appears as a substring of the output."""

    def _fn(output: Any, expected: Any) -> float:
        out = normalize_text(output) if normalize else _to_text(output)
        exp = normalize_text(expected) if normalize else _to_text(expected)
        if not exp:
            return 0.0
        return 1.0 if exp in out else 0.0

    return Metric("contains", _fn)


def regex_match(pattern: str | None = None, *, flags: int = re.IGNORECASE) -> Metric:
    """1.0 iff the output matches a regex.

    With an explicit ``pattern`` the expected value is ignored. Without one, the
    per-example ``expected`` value is compiled as the pattern (handy for datasets
    that carry a different acceptance regex per row).
    """
    compiled = re.compile(pattern, flags) if pattern is not None else None

    def _fn(output: Any, expected: Any) -> float:
        rx = compiled if compiled is not None else re.compile(_to_text(expected), flags)
        return 1.0 if rx.search(_to_text(output)) else 0.0

    return Metric("regex_match", _fn)


def token_f1() -> Metric:
    """Token-overlap F1 between output and expected (SQuAD-style)."""

    def _fn(output: Any, expected: Any) -> float:
        pred, gold = _tokens(output), _tokens(expected)
        if not pred and not gold:
            return 1.0
        if not pred or not gold:
            return 0.0
        common: dict[str, int] = {}
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
        common.clear()
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


def numeric_close(*, tolerance: float = 1e-6, relative: bool = False) -> Metric:
    """1.0 iff the first number in output is within tolerance of expected's."""

    def _fn(output: Any, expected: Any) -> float:
        out_num, exp_num = _extract_number(output), _extract_number(expected)
        if out_num is None or exp_num is None:
            return 0.0
        diff = abs(out_num - exp_num)
        bound = tolerance * (abs(exp_num) if relative else 1.0)
        return 1.0 if diff <= bound else 0.0

    return Metric("numeric_close", _fn)


def json_subset() -> Metric:
    """1.0 iff every key/value in expected (a dict) is present in the output dict."""

    def _fn(output: Any, expected: Any) -> float:
        out = _as_mapping(output)
        exp = _as_mapping(expected)
        if out is None or exp is None or not exp:
            return 0.0
        matched = sum(1 for k, v in exp.items() if out.get(k) == v)
        return matched / len(exp)

    return Metric("json_subset", _fn)


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
        scores = [m(output, expected, example) for m in metric_list]
        if not scores:
            raise ValueError("example declares an empty check list")
        return min(scores) if aggregate == "min" else sum(scores) / len(scores)

    return Metric("checks", _fn, needs_example=True)


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
    factory = BUILTIN_METRICS.get(spec)
    if factory is None:
        raise KeyError(f"Unknown check {spec!r}. Available: {sorted(BUILTIN_METRICS)}")
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
    "checks": checks,
}


def get_metric(name: str) -> Metric:
    """Look up a built-in metric by name using default settings."""
    factory = BUILTIN_METRICS.get(name)
    if factory is None:
        raise KeyError(f"Unknown built-in metric {name!r}. Available: {sorted(BUILTIN_METRICS)}")
    return factory()


# -- low-level helpers --------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _NUMBER_RE.search(_to_text(value).replace(",", ""))
    return float(match.group(0)) if match else None


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import json

        try:
            obj = json.loads(value)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
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
    "checks",
    "BUILTIN_METRICS",
    "get_metric",
]
