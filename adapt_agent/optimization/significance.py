"""Paired significance tests for "is this configuration really better?".

An optimizer compares an *incumbent* configuration with a *challenger* on the
same dataset rows. Comparing two means throws away the pairing and treats every
wobble as signal: an agent that is right 70% of the time, scored twice on ten
rows, differs by 0.2 between runs about a third of the time with no change at
all. The decision has to be made on the **per-row deltas** -- challenger minus
incumbent on the same row -- and has to ask whether they are larger than chance
would produce.

:func:`compare_rows` does that with an exact **sign-flip (paired permutation)
test**: under "the change does nothing", each row's delta is as likely to be
positive as negative, so the observed sum is compared with the sum under every
assignment of signs. It needs no SciPy and makes no distributional assumption,
which matters for 0/1 metrics where a t-test is a poor fit. Exact enumeration
is used whenever the number of distinct sums stays small (always, for 0/1 or
coarsely graded metrics); a seeded Monte Carlo or a normal approximation takes
over for long runs of continuous deltas, so a result is still reproducible.

Small datasets are reported as what they are. With ``n`` rows the smallest
achievable one-sided p-value is ``0.5 ** n`` -- even when every row improves --
so at ``alpha=0.05`` four rows can *never* be significant.
:attr:`PairedComparison.decidable` says when the test could not have reached a
verdict, rather than letting a perfect-looking sweep on four rows read as proof.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from adapt_agent.optimization.evaluation import EvaluationReport

#: Above this many distinct partial sums the exact enumeration gives way to an
#: approximation. 0/1 metrics produce ``n + 1`` sums, so they are always exact.
_EXACT_SUM_LIMIT = 20_000
#: Changed rows from which the normal approximation to the sign-flip
#: distribution is used once exact enumeration is out of reach.
_NORMAL_APPROX_MIN_ROWS = 30
#: Monte Carlo draws when neither exact enumeration nor the normal
#: approximation is appropriate.
_MONTE_CARLO_DRAWS = 20_000
#: Bootstrap resamples for the reported interval, and the row count above which
#: a normal interval replaces the bootstrap (to keep long runs cheap).
_BOOTSTRAP_RESAMPLES = 2_000
_BOOTSTRAP_MAX_ROWS = 400
#: Deltas closer to zero than this are treated as ties (floating-point noise).
_TIE = 1e-12
#: Integer scaling used by the exact enumeration, so sums compare exactly.
_SCALE = 1_000_000


def min_rows_for_significance(alpha: float) -> int:
    """The fewest paired rows for which *any* outcome can reach ``alpha``.

    With ``n`` rows the strongest possible evidence -- every row improves -- has
    a one-sided sign-flip p-value of ``0.5 ** n``. Fewer rows than this cannot
    show significance whatever happens on them.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    return max(1, math.ceil(math.log(alpha) / math.log(0.5) - 1e-9))


@dataclass(frozen=True)
class PairedComparison:
    """The outcome of a paired comparison between two configurations.

    Args:
        delta: Mean per-row delta, challenger minus incumbent, over
            :attr:`n_rows`. Equal to the difference of the two means when both
            were scored on the same rows.
        p_value: One-sided p-value for "the challenger is better" under the
            sign-flip null hypothesis (``1.0`` when no row changed).
        n_rows: Rows compared: scored in both measurements and not excluded
            (e.g. a few-shot block's own demonstration rows).
        n_better: Rows where the challenger scored higher.
        n_worse: Rows where the challenger scored lower.
        alpha: Significance level the verdict uses.
        min_effect: Minimum mean delta required on top of significance.
        repeats: Measurements per configuration that were averaged per row.
        ci_low: Lower end of a two-sided ``1 - alpha`` interval for
            :attr:`delta`, when one was computed (bootstrap, or a normal
            interval on long runs). For reporting; the verdict uses the test.
        ci_high: Upper end of that interval.
    """

    delta: float
    p_value: float
    n_rows: int
    n_better: int
    n_worse: int
    alpha: float
    min_effect: float = 0.0
    repeats: int = 1
    ci_low: float | None = None
    ci_high: float | None = None

    @property
    def min_rows(self) -> int:
        """Rows needed before any outcome could be significant at :attr:`alpha`."""
        return min_rows_for_significance(self.alpha)

    @property
    def decidable(self) -> bool:
        """Whether this many rows could have shown significance at all."""
        return self.n_rows >= self.min_rows

    @property
    def significant(self) -> bool:
        """The gain is larger than chance would produce, and large enough to matter."""
        return self.decidable and self.p_value <= self.alpha and self.delta > self.min_effect

    def describe(self) -> str:
        """A one-line, human-readable account of the comparison."""
        interval = ""
        if self.ci_low is not None and self.ci_high is not None:
            interval = f", {1 - self.alpha:.0%} CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}]"
        text = (
            f"delta {self.delta:+.4f} over {self.n_rows} row(s) "
            f"({self.n_better} better, {self.n_worse} worse{interval}), "
            f"p={self.p_value:.4f} at alpha={self.alpha:g}"
        )
        if self.repeats > 1:
            text += f", {self.repeats} repeats per configuration"
        if not self.decidable:
            text += (
                f"; {self.n_rows} row(s) cannot reach significance "
                f"(even if every row improved, p={0.5 ** self.n_rows:.4f}; "
                f"at least {self.min_rows} needed)"
            )
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta": self.delta,
            "p_value": self.p_value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_rows": self.n_rows,
            "n_better": self.n_better,
            "n_worse": self.n_worse,
            "repeats": self.repeats,
            "alpha": self.alpha,
            "min_effect": self.min_effect,
            "decidable": self.decidable,
            "significant": self.significant,
        }


def row_scores(report: EvaluationReport | None) -> list[float | None] | None:
    """Per-row primary-metric scores, indexed by dataset row.

    A row whose primary score was lost to a transient provider failure is
    ``None``: it carries a placeholder zero, not a measurement. Returns ``None``
    when the report does not hold a record for every evaluated row (the harness
    stores at most ``max_results``, and hand-built reports may hold none), so a
    paired test is never run on a silently truncated sample.
    """
    if report is None:
        return None
    n_rows = report.n_evaluated
    if n_rows <= 0 or len(report.results) < n_rows:
        return None
    primary = report.primary_metric
    scores: list[float | None] = [None] * n_rows
    seen = 0
    for result in report.results:
        if not 0 <= result.index < n_rows:
            return None
        seen += 1
        if result.transient or primary in result.transient_metrics:
            continue
        scores[result.index] = float(result.scores.get(primary, 0.0))
    return scores if seen == n_rows else None


def mean_rows(measurements: Sequence[Sequence[float | None]]) -> list[float | None]:
    """Average repeated measurements of one configuration row by row.

    A row is ``None`` only when no repeat scored it.
    """
    if not measurements:
        return []
    width = len(measurements[0])
    out: list[float | None] = []
    for i in range(width):
        values = [m[i] for m in measurements if i < len(m) and m[i] is not None]
        out.append(sum(values) / len(values) if values else None)  # type: ignore[arg-type]
    return out


def rows_differ(first: Sequence[float | None], second: Sequence[float | None]) -> bool:
    """Whether two measurements of one configuration disagree on any scored row."""
    return any(
        a is not None and b is not None and abs(a - b) > _TIE
        for a, b in zip(first, second, strict=False)
    )


def compare_rows(
    incumbent: Sequence[float | None],
    challenger: Sequence[float | None],
    *,
    alpha: float = 0.05,
    min_effect: float = 0.0,
    exclude: Iterable[int] = (),
    repeats: int = 1,
    interval: bool = False,
    seed: int | None = 0,
) -> PairedComparison:
    """Paired comparison of two per-row measurements over the same rows.

    Only rows scored in both measurements, and not in ``exclude``, are compared
    -- so a row throttled in one measurement drops out of *both*, and the two
    sides are always means over the same rows.

    Args:
        incumbent: Per-row scores of the current best configuration.
        challenger: Per-row scores of the candidate.
        alpha: Significance level.
        min_effect: Mean delta that must be exceeded as well.
        exclude: Row indices to leave out (e.g. rows a candidate was built from).
        repeats: Recorded on the result; the scores are already averaged.
        interval: Also compute a two-sided ``1 - alpha`` interval for the mean
            delta (a seeded bootstrap; normal on long runs). Costs more than
            the test, so the search leaves it off and the final verdict asks.
        seed: Seed for any resampling, so a result is reproducible.
    """
    excluded = set(exclude)
    deltas = [
        c - i
        for index, (i, c) in enumerate(zip(incumbent, challenger, strict=False))
        if index not in excluded and i is not None and c is not None
    ]
    n = len(deltas)
    delta = sum(deltas) / n if n else 0.0
    ci_low = ci_high = None
    if interval and n:
        ci_low, ci_high = _interval(deltas, alpha, seed)
    return PairedComparison(
        delta=delta,
        p_value=sign_flip_p_value(deltas, seed=seed),
        n_rows=n,
        n_better=sum(1 for d in deltas if d > _TIE),
        n_worse=sum(1 for d in deltas if d < -_TIE),
        alpha=alpha,
        min_effect=min_effect,
        repeats=repeats,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def sign_flip_p_value(deltas: Sequence[float], *, seed: int | None = 0) -> float:
    """One-sided p-value that the mean of ``deltas`` exceeds zero.

    Under the null hypothesis each row's delta is symmetric about zero, so
    every assignment of signs to the observed magnitudes is equally likely. The
    p-value is the share of those assignments whose sum is at least the one
    observed. Ties (zero deltas) contribute nothing either way.
    """
    magnitudes = [abs(d) for d in deltas if abs(d) > _TIE]
    if not magnitudes:
        return 1.0
    signed = [round(d * _SCALE) for d in deltas if abs(d) > _TIE]
    scaled = [abs(s) for s in signed]
    observed = sum(signed)

    exact = _exact_tail(scaled, observed)
    if exact is not None:
        return exact
    if len(magnitudes) >= _NORMAL_APPROX_MIN_ROWS:
        spread = math.sqrt(sum(m * m for m in magnitudes))
        z = (observed / _SCALE) / spread if spread else 0.0
        return 1.0 - NormalDist().cdf(z)
    rng = random.Random(seed)
    hits = 0
    for _ in range(_MONTE_CARLO_DRAWS):
        total = sum(m if rng.random() < 0.5 else -m for m in scaled)
        if total >= observed:
            hits += 1
    return (hits + 1) / (_MONTE_CARLO_DRAWS + 1)


def _exact_tail(scaled: Sequence[int], observed: int) -> float | None:
    """Exact ``P(sum >= observed)`` over all sign assignments, or ``None``.

    Builds the distribution of signed sums one row at a time. Returns ``None``
    when the number of distinct sums would grow past :data:`_EXACT_SUM_LIMIT`
    (long runs of distinct continuous deltas), leaving the caller to
    approximate.
    """
    distribution: dict[int, int] = {0: 1}
    for magnitude in scaled:
        grown: dict[int, int] = {}
        for total, count in distribution.items():
            grown[total + magnitude] = grown.get(total + magnitude, 0) + count
            grown[total - magnitude] = grown.get(total - magnitude, 0) + count
        if len(grown) > _EXACT_SUM_LIMIT:
            return None
        distribution = grown
    tail = sum(count for total, count in distribution.items() if total >= observed)
    # Integer over integer: Python divides exactly, whatever the size. Converting
    # the denominator to a float first overflowed from 1,024 rows upwards.
    denominator: int = 2 ** len(scaled)
    return tail / denominator


def _interval(deltas: Sequence[float], alpha: float, seed: int | None) -> tuple[float, float]:
    """Two-sided ``1 - alpha`` interval for the mean delta."""
    n = len(deltas)
    mean = sum(deltas) / n
    if n == 1:
        return mean, mean
    if n > _BOOTSTRAP_MAX_ROWS:
        variance = sum((d - mean) ** 2 for d in deltas) / (n - 1)
        half = NormalDist().inv_cdf(1 - alpha / 2) * math.sqrt(variance / n)
        return mean - half, mean + half
    rng = random.Random(seed)
    means = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(_BOOTSTRAP_RESAMPLES)
    )
    low = means[int(math.floor((alpha / 2) * (_BOOTSTRAP_RESAMPLES - 1)))]
    high = means[int(math.ceil((1 - alpha / 2) * (_BOOTSTRAP_RESAMPLES - 1)))]
    return low, high


__all__ = [
    "PairedComparison",
    "compare_rows",
    "mean_rows",
    "min_rows_for_significance",
    "row_scores",
    "rows_differ",
    "sign_flip_p_value",
]
