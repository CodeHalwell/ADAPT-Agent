"""Optimizers: search strategies that improve an agent against a golden dataset.

Every optimizer shares the same contract:

    result = optimizer.optimize(agent, train_dataset, val_dataset=None)

``agent`` may be an :class:`~adapt_agent.optimization.target.OptimizableAgent` or
anything :func:`~adapt_agent.optimization.target.wrap` accepts (a callable, a
single framework object, ...). The optimizer measures a baseline with the
:class:`~adapt_agent.optimization.evaluation.EvaluationHarness`, searches the
parameter space for a better configuration, and -- crucially -- **applies the
best configuration to the live agent in place** before returning, so the caller's
agent is actually improved. An :class:`OptimizationResult` records the baseline,
the best score, and the full trial history.

Candidate generation is delegated to *proposers* (see
:mod:`adapt_agent.optimization.proposers`), so the same optimizer optimizes
prompts, few-shot blocks, models, hyperparameters, routing knobs and tool
allow-lists -- including LLM-judge-driven prompt rewrites.

Strategies provided:

* :class:`GridSearchOptimizer` -- exhaustive over discrete candidate sets.
* :class:`RandomSearchOptimizer` -- random whole-space sampling.
* :class:`CoordinateAscentOptimizer` -- greedy per-parameter improvement; the
  flagship for prompt / few-shot optimization.
* :class:`BootstrapFewShotOptimizer` -- coordinate ascent restricted to few-shot.
* :class:`EvolutionaryOptimizer` -- population-based mutation + selection.
* :class:`PipelineOptimizer` -- run several optimizers in sequence.

What "better" means
-------------------

A candidate replaces the incumbent only on a **paired statistical test** over
per-row deltas (see :mod:`adapt_agent.optimization.significance`), never on a
bare comparison of two means: an agent that is right 70% of the time differs
from itself by 0.2 on ten rows about a third of the time. A target whose
per-row results change between two runs of the same configuration is detected
up front and every comparison is then re-measured ``repeats`` times, incumbent
and challenger interleaved. The final configuration is applied only when it
shows a gain on held-out rows (``val_dataset``, or an automatic split of a
large enough golden set), and :attr:`OptimizationResult.improved` is ``True``
only when that gain is significant -- with the delta, p-value, interval, row
count and repeats recorded alongside it.
"""

from __future__ import annotations

import logging
import random
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapt_agent.exceptions import IncompleteEvaluationError
from adapt_agent.optimization.dataset import GoldenDataset
from adapt_agent.optimization.evaluation import EvaluationHarness, EvaluationReport
from adapt_agent.optimization.parameters import Parameter, ParameterKind
from adapt_agent.optimization.proposers import (
    ProposalContext,
    Proposer,
    default_proposers,
    proposers_for,
)
from adapt_agent.optimization.significance import (
    PairedComparison,
    compare_rows,
    mean_rows,
    row_scores,
    rows_differ,
)
from adapt_agent.optimization.target import OptimizableAgent, wrap

logger = logging.getLogger(__name__)


@dataclass
class Trial:
    """A single evaluated configuration in the optimization history."""

    config: dict[str, Any]
    score: float
    strategy: str
    accepted: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    #: False when transient provider failures cost this trial some rows, so its
    #: ``score`` is a mean over a subset and is not comparable with any other.
    complete: bool = True
    #: How many rows were dropped for transient failures.
    n_transient_errors: int = 0
    #: Mean per-row delta against the incumbent this trial was compared with
    #: (``None`` when no per-row comparison was possible).
    delta: float | None = None
    #: One-sided p-value of that paired comparison.
    p_value: float | None = None
    #: Whether the paired comparison was significant (``None``: not assessed).
    significant: bool | None = None
    #: Rows the paired comparison covered.
    n_rows: int = 0
    #: Measurements of this configuration averaged into :attr:`score`.
    repeats: int = 1


#: Reserved top-level name: the exported file's own metadata block. A tuned
#: parameter using it could be written but never reloaded, so it is described
#: instead -- see :meth:`OptimizationResult.to_config` and
#: :func:`load_tuned_config`.
_PROVENANCE_KEY = "_provenance"


@dataclass
class OptimizationResult:
    """The outcome of an optimization run.

    Args:
        best_config: The winning configuration (parameter name -> value).
        best_score: Primary-metric score of ``best_config``.
        baseline_score: Primary-metric score before optimization.
        baseline_config: Snapshot of the agent before optimization (for rollback).
        history: Every :class:`Trial` evaluated, in order.
        best_report: The :class:`EvaluationReport` for the winning configuration.
        validation_score: Score of ``best_config`` on a held-out set, if provided.
        validation_complete: Whether that score covered every held-out row.
        validation_baseline_score: Score of the *untuned* agent on the same
            held-out set, when it was measured (the validation gate needs it).
        validation_rejected: ``True`` when the search found a configuration
            that beat the baseline on the training data but showed no gain over
            the baseline on the held-out set, so it was **not** applied. In
            that case ``best_config`` is empty, the agent is left as it was,
            and the discarded configuration is in ``rejected_config``.
        rejected_config: The configuration the final verdict discarded (on
            held-out data, or as not significant on the training rows; see
            ``rejected_reason``).
        recommendations: Advisory, human-readable suggestions gathered during the
            run (e.g. new tools/skills the judge proposes). These are never applied
            automatically -- the optimizer only *selects* among existing tools.
    """

    best_config: dict[str, Any]
    best_score: float
    baseline_score: float
    baseline_config: dict[str, Any] = field(default_factory=dict)
    history: list[Trial] = field(default_factory=list)
    best_report: EvaluationReport | None = None
    validation_score: float | None = None
    recommendations: list[str] = field(default_factory=list)

    # -- fields added after 0.3.0 -----------------------------------------
    #
    # Appended, never inserted. This dataclass is public and positional
    # construction is a supported call shape, so a field placed in the
    # middle silently rebinds every argument after it: `recommendations`
    # was the eighth positional, and putting this flag in front of it
    # assigned the caller's list to a boolean and left the recommendations
    # empty -- lost on serialisation, with no error raised.
    #: False when transient provider failures cost the validation pass some
    #: rows, so :attr:`validation_score` is a mean over a subset and is not
    #: comparable with :attr:`best_score`. ``True`` when there was no
    #: validation set at all -- there is nothing partial about not running.
    validation_complete: bool = True
    #: Wall-clock seconds for the whole ``optimize`` call (baseline, search,
    #: and validation). ``0.0`` for a hand-built result never produced by a
    #: real run. Reported alongside the per-trial elapsed time in verbose
    #: logging (see :meth:`Optimizer._record`), so a long run's total cost is
    #: visible in the same place its progress was.
    duration_seconds: float = 0.0
    #: Held-out score of the untuned agent; ``None`` when not measured.
    validation_baseline_score: float | None = None
    #: The tuned config won on training data, lost on held-out data, and was
    #: discarded. See :class:`Optimizer`'s ``validation_gate``.
    validation_rejected: bool = False
    #: What the gate discarded -- kept for inspection, never applied.
    rejected_config: dict[str, Any] = field(default_factory=dict)
    #: Whether the gain of ``best_config`` over the untuned agent passed the
    #: paired significance test (see :mod:`adapt_agent.optimization.significance`).
    #: ``None`` only on a hand-built result that carries no such evidence; a
    #: run always sets it. When ``False`` the result is not an improvement,
    #: whatever the two means say.
    significant: bool | None = None
    #: Mean per-row delta behind that verdict (tuned minus untuned).
    delta: float | None = None
    #: One-sided p-value of the paired test.
    p_value: float | None = None
    #: Two-sided ``1 - alpha`` interval for :attr:`delta` (bootstrap).
    ci_low: float | None = None
    ci_high: float | None = None
    #: Rows the verdict was measured on.
    n_rows: int = 0
    #: Measurements per configuration averaged into each row of the verdict.
    repeats: int = 1
    #: Significance level used.
    alpha: float | None = None
    #: Which rows the verdict was measured on: ``"held_out"`` (the
    #: ``val_dataset``), ``"held_out_auto"`` (an automatic split of the golden
    #: set), or ``"training"`` (the rows the search selected on, which flatters
    #: the winner). Empty when nothing was compared.
    significance_basis: str = ""
    #: The target's per-row results changed between two runs of the same
    #: configuration, so every comparison was re-measured ``repeats`` times.
    nondeterministic: bool = False
    #: Parameters in this optimizer's search space that no trial ever varied
    #: (the budget ran out, or no candidate was produced). A warning names
    #: them too; they were left exactly as they were.
    untuned_parameters: list[str] = field(default_factory=list)
    #: Why ``rejected_config`` was discarded, in one line ("" when nothing was).
    rejected_reason: str = ""

    @property
    def improved(self) -> bool:
        """``True`` only for a significant gain over the untuned agent.

        A hand-built result with no significance evidence (``significant is
        None``) falls back to comparing the two scores.
        """
        if self.significant is None:
            return self.best_score > self.baseline_score
        return self.significant and self.best_score > self.baseline_score

    @property
    def improvement(self) -> float:
        return self.best_score - self.baseline_score

    @property
    def n_evals(self) -> int:
        return len(self.history)

    def to_config(
        self,
        path: str | Path | None = None,
        *,
        header: bool = True,
    ) -> dict[str, Any]:
        """Export the winning configuration as a reviewable, nested mapping.

        :attr:`best_config` is applied *in place*, to live objects, and dies with
        the process. For any project whose prompts are version-controlled that is
        the wrong end state -- and an LLM-rewritten prompt reaching production
        without a human reading the diff is worse than a lost tuning run.

        This turns the result into an artifact, so the loop becomes
        **optimize -> diff -> review -> commit** and the application keeps
        loading its prompts from the YAML it always did::

            result.to_config("specialists/.config/tuned.yaml")

        Parameter names follow the ``"<component>.<knob>"`` convention, so the
        flat config nests by component. A name with no prefix stays at the top
        level as a scalar, so it round-trips back under its own name::

            researcher:
              system_prompt: |
                You are a careful researcher...
              temperature: 0.2
            writer:
              model: gpt-5.6-luna
            temperature: 0.2        # a bare name, unchanged

        Round-trips through :func:`load_tuned_config`, which flattens it back
        for :meth:`OptimizableAgent.apply`.

        The format follows the extension: ``.json`` writes JSON, anything else
        writes YAML. Provenance -- baseline, best and validation scores, and the
        number of evaluations -- travels with the config, so a diff shows *what
        it bought* rather than only what changed. In YAML that is a comment
        header; JSON has no comments, so it goes in a ``"_provenance"`` key
        which :func:`load_tuned_config` ignores.

        **Live objects are described, not exported.** A ``TOOL``/``SKILL``
        parameter holds callables or SDK objects, which no YAML or JSON encoder
        can represent. Writing a stand-in string would be worse than useless:
        :meth:`OptimizableAgent.apply` would later set that string over the real
        tool list and break the agent. So such parameters are listed -- by
        ``module:qualname``, which is what a reviewer wants to see anyway -- in a
        comment header (YAML) or under ``_provenance.unexportable`` (JSON), and
        left out of the config body, which therefore stays safely reloadable.
        That listing appears even when ``header=False``: silently dropping a
        tuned parameter is worse than two lines of comment.

        Args:
            path: Where to write. ``None`` returns the mapping without writing.
                The parent directory is created if needed.
            header: Record provenance in the file. Comments do not affect
                parsing or the round-trip.

        Returns:
            The nested ``{component: {knob: value}}`` mapping (never including
            the provenance block or unexportable parameters, so the return value
            is exactly the config that will apply cleanly).
        """
        nested: dict[str, Any] = {}
        unexportable: dict[str, str] = {}
        # A bare name that is also a component prefix -- `{"agent": 1,
        # "agent.temperature": 0.2}` -- cannot share the top level with its own
        # namespace. Written first it is indexed into as a mapping (`TypeError`);
        # written second it overwrites the component's knobs. Both outcomes
        # depend only on dict ordering, so the collision is resolved up front:
        # the qualified knobs are the ones that can round-trip, and the bare
        # name is described instead.
        namespaces = {name.partition(".")[0] for name in self.best_config if "." in name}
        for name, value in self.best_config.items():
            component, _, knob = name.partition(".")
            if component == _PROVENANCE_KEY:
                # The file's own metadata block owns this name, and
                # `load_tuned_config` skips it on the way back in. Writing a
                # tuned parameter there would export a value that never
                # reloads -- and in JSON would be overwritten by the real
                # provenance. Described, so the loss is visible rather than
                # silent.
                unexportable[name] = _describe(value)
            elif not knob and name in namespaces:
                unexportable[name] = _describe(value)
            elif not knob and isinstance(value, dict):
                # A bare name written as a top-level mapping is indistinguishable
                # from a component section on reload, so `load_tuned_config`
                # would return it as "<name>.<key>" and `apply()` would skip it.
                # Ambiguous beats wrong: describe it rather than export a value
                # that cannot come back as itself.
                unexportable[name] = _describe(value)
            elif not _is_exportable(value):
                # A TOOL/SKILL parameter holds live callables or SDK objects. They
                # cannot be written to YAML/JSON, and a string standing in for one
                # would corrupt the agent if `apply()` later wrote it back over the
                # real tool list. So they are described, not exported -- the file
                # records which were selected, for review, and stays safely
                # reloadable.
                unexportable[name] = _describe(value)
            elif knob:
                nested.setdefault(component, {})[knob] = value
            else:
                # A parameter with no "<component>." prefix stays at the top
                # level as a scalar. Filing it under a synthetic component would
                # rename it on the way out, and `load_tuned_config` could not
                # recover the original -- `apply()` would then silently skip it,
                # breaking the round trip this method exists to provide.
                nested[name] = value

        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.suffix.lower() == ".json":
                import json

                payload = dict(nested)
                provenance = self._provenance() if header else {}
                if unexportable:
                    provenance["unexportable"] = unexportable
                if provenance:
                    payload[_PROVENANCE_KEY] = provenance
                text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            else:
                import yaml

                text = yaml.safe_dump(nested, sort_keys=True, allow_unicode=True, width=100)
                if unexportable:
                    # Reported even when `header=False`: silently dropping a
                    # tuned parameter is worse than a couple of comment lines.
                    text = (
                        "# Not exported (live objects, descriptive only):\n"
                        + "".join(f"#   {k}: {v}\n" for k, v in sorted(unexportable.items()))
                    ) + text
                if header:
                    provenance = self._provenance()
                    text = (
                        "# Tuned by adapt-agent.\n"
                        f"# baseline={provenance['baseline_score']:.4f} "
                        f"best={provenance['best_score']:.4f} "
                        f"improvement={provenance['improvement']:+.4f} "
                        f"validation={provenance['validation_score']}"
                        f"{'' if provenance['validation_complete'] else ' (PARTIAL)'} "
                        f"over {provenance['n_evals']} evals in "
                        f"{provenance['duration_seconds']:.1f}s.\n"
                        f"# significant={provenance['significant']} "
                        f"p={_fmt_p(provenance['p_value'])} n_rows={provenance['n_rows']} "
                        f"repeats={provenance['repeats']} "
                        f"basis={provenance['significance_basis'] or 'none'}.\n"
                        "# Review this diff before committing: prompts here were "
                        "machine-written.\n"
                    ) + text
            destination.write_text(text, encoding="utf-8")
        return nested

    def _provenance(self) -> dict[str, Any]:
        """What the tuning run bought, recorded alongside the config."""
        return {
            "baseline_score": self.baseline_score,
            "best_score": self.best_score,
            "improvement": self.improvement,
            "validation_score": self.validation_score,
            # Travels with the score it qualifies. A partial validation is the
            # same number as a whole one until something says otherwise, and a
            # persisted result is read long after the run's logs are gone.
            "validation_complete": self.validation_complete,
            "validation_baseline_score": self.validation_baseline_score,
            "validation_rejected": self.validation_rejected,
            "n_evals": self.n_evals,
            "duration_seconds": self.duration_seconds,
            **self._significance_fields(),
        }

    def _significance_fields(self) -> dict[str, Any]:
        return {
            "significant": self.significant,
            "delta": self.delta,
            "p_value": self.p_value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_rows": self.n_rows,
            "repeats": self.repeats,
            "alpha": self.alpha,
            "significance_basis": self.significance_basis,
            "nondeterministic": self.nondeterministic,
            "untuned_parameters": list(self.untuned_parameters),
            "rejected_reason": self.rejected_reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "improved": self.improved,
            "baseline_score": self.baseline_score,
            "best_score": self.best_score,
            "improvement": self.improvement,
            "validation_score": self.validation_score,
            "validation_complete": self.validation_complete,
            "validation_baseline_score": self.validation_baseline_score,
            "validation_rejected": self.validation_rejected,
            "n_evals": self.n_evals,
            "duration_seconds": self.duration_seconds,
            "best_config": self.best_config,
            "recommendations": list(self.recommendations),
            **self._significance_fields(),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        verdict = ""
        if self.significant is not None:
            verdict = (
                f", significant={self.significant} (p={_fmt_p(self.p_value)}, n={self.n_rows})"
            )
        return (
            f"OptimizationResult(baseline={self.baseline_score:.3f}, "
            f"best={self.best_score:.3f}, improvement={self.improvement:+.3f}, "
            f"evals={self.n_evals}, duration={self.duration_seconds:.1f}s{verdict})"
        )


@dataclass
class _ValidationGate:
    """The final verdict on the search's winner, and what was done about it."""

    score: float | None = None
    complete: bool = True
    baseline_score: float | None = None
    #: The *held-out* check discarded the winner (``validation_rejected``).
    rejected: bool = False
    rejected_config: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: The paired comparison the verdict rests on, when one could be made.
    comparison: PairedComparison | None = None
    #: ``"held_out"``, ``"held_out_auto"``, ``"training"`` or ``""``.
    basis: str = ""
    #: The winner was discarded, by the held-out check *or* the training one.
    discarded: bool = False
    reason: str = ""
    #: A verdict was reached (False only for a nested pipeline stage, whose
    #: pipeline decides for it).
    assessed: bool = True


#: Default ``repeats`` once a target is found to be non-deterministic: enough
#: to halve the per-row variance's effect on a comparison, and each one costs a
#: pass over the dataset for both incumbent and challenger.
DEFAULT_NONDETERMINISTIC_REPEATS = 3

#: Sentinel for "attribute was absent" when a pipeline borrows a stage.
_MISSING = object()


class Optimizer:
    """Base class with the shared optimize loop and evaluation plumbing.

    Args:
        harness: The :class:`EvaluationHarness` used to score configurations.
        max_evals: Hard cap on candidate evaluations (budget guard): the number
            of trials, each a comparison of one candidate with the incumbent.
            Must be at least 1. With ``repeats > 1`` a trial costs
            ``2 * repeats`` passes over the dataset.
        seed: RNG seed for reproducible search. The RNG is re-seeded at the
            start of every ``optimize`` call, so one instance reused for
            several runs proposes the same trials each time.
        min_improvement: Minimum mean per-row gain (the *effect size*) a
            candidate must show over the current best, on top of passing the
            significance test. Defaults to ``1e-3``: small enough to keep real
            gains, but large enough to avoid *chasing judge noise*.
        alpha: Significance level of the paired test every acceptance rests on
            (default ``0.05``). A candidate replaces the incumbent only if its
            per-row deltas are unlikely under "the change does nothing"; see
            :mod:`adapt_agent.optimization.significance`.
        repeats: Measurements of each configuration per comparison, with the
            incumbent and the challenger re-measured alternately on the same
            rows. ``None`` (default) means 1 for a deterministic target and
            :data:`DEFAULT_NONDETERMINISTIC_REPEATS` once the baseline is found
            to be non-deterministic. Any value above 1 makes every measurement
            a real re-run (the evaluation cache is bypassed).
        detect_nondeterminism: Re-run the baseline once and compare per-row
            results (default ``True``; one extra pass). A target whose rows
            change between identical runs is measured with ``repeats`` and a
            warning, rather than having its noise read as signal. Skipped when
            ``repeats`` is given explicitly.
        holdout_fraction: When no ``val_dataset`` is given, hold out this
            fraction of the golden set for the final check (default ``0.25``)
            -- but only when both parts keep at least ``min_holdout_rows`` rows;
            a smaller set is searched whole and the result says so. ``0``
            disables the automatic split.
        min_holdout_rows: Rows each side of an automatic split must keep
            (default 10).
        allow_small_sample: A comparison over fewer rows than any outcome
            could make significant (five at ``alpha=0.05``) cannot reach a
            verdict. ``True`` (default) decides such comparisons on the point
            estimate -- the agent can still be tuned on a toy set -- but the
            result reports ``significant=False`` and ``improved=False`` and says
            why. ``False`` accepts nothing that has not been shown significant.
        judge: Optional LLM judge made available to proposers.
        proposer_judge: A *separate* judge for the proposing side -- critiquing
            failures and rewriting prompts -- leaving ``judge`` (or whatever
            judge backs the harness's metric) to do the scoring. When one model
            both writes a candidate and marks it, the mark is not independent:
            a model tends to prefer text in its own style, and a rewrite can
            learn what the marker rewards rather than what the task needs.
            Pass a judge built on a different model family here to break that
            loop. Defaults to ``judge``. When the proposing judge turns out to
            be the scoring judge, the run says so in
            ``OptimizationResult.recommendations`` -- and the validation gate
            cannot help, because the same judge scores the held-out set too.
        validation_gate: When ``True`` (default) and there is a held-out set
            (``val_dataset``, or the automatic split), the winning
            configuration is applied only if it shows a **gain** over the
            untuned agent on those rows: significant under the paired test, or
            -- on a held-out set too small for any outcome to be significant,
            with ``allow_small_sample`` -- a positive point estimate. Not
            scoring worse is not enough: a no-op that won the training search
            by luck does not regress either. The search only ever sees the
            training data, and is rewarded for whatever raises that one number
            -- including a prompt that has memorised the training rows. The
            held-out set is the one thing it cannot have fitted, so it gets the
            final say. A rejected run leaves the agent exactly as it was and
            reports ``validation_rejected=True``. Rows lost to throttling in
            either held-out pass drop out of *both* sides of the comparison.
            ``False`` reports the held-out score without acting on it; the
            training verdict then decides.
        validation_tolerance: No longer used, and refused loudly (a
            ``FutureWarning``) when given: the gate used to accept anything
            within this distance of the baseline, which let a no-op through.
            The held-out requirement is now a gain, sized by ``min_improvement``
            and ``alpha``.
        cache_evaluations: Reuse the report of an already-measured
            configuration instead of re-running the agent over the dataset
            (default ``True``). The cache is keyed on this optimizer's
            ``harness`` *and* the *live parameter state* at evaluation time --
            the harness because a report is only reusable by the evaluator
            that would have produced it, and the live state (rather than the
            candidate diff) because that is what lets a stage baseline hit a
            previous stage's winning trial reached through a different
            config. It lives for one ``optimize`` call -- or one pipeline
            run, where :class:`PipelineOptimizer` shares one cache across
            stages that carry the same harness, so each such stage's baseline
            stops costing a full-dataset pass. Only complete reports are
            reused, so the transient-failure machinery (baseline re-runs,
            incomplete-trial exclusion) is unaffected. Set ``False`` to
            re-measure every configuration. (A non-deterministic target does
            not need this: its repeated measurements always bypass the cache;
            see ``repeats``.) A pipeline's own ``False`` also
            suppresses caching in every stage for that run, even one left at
            its own default ``True``; a stage's own ``False`` inside a
            caching pipeline is still respected.
        suggest_tools: When True *and* a ``judge`` is set, after the search the
            optimizer asks the judge to propose NEW tools/skills for components
            that own a TOOL/SKILL parameter and records them as advisory
            ``recommendations`` (never applied automatically). Off by default;
            :func:`make_default_optimizer` turns it on when a judge is present.
        verbose: Emit per-trial logging at INFO level: each trial logs its
            position out of the run's evaluation budget (``max_evals``) and
            the wall-clock time elapsed since the run started (e.g.
            ``trial 5/60 score=0.7100 ACCEPT (142.3s elapsed)``), so a long
            run distinguishes "still working" from "hung" without needing an
            external progress indicator. The denominator is each optimizer's
            own budget: a :class:`PipelineOptimizer` stage logs against its
            *own* (budget-clamped) ``max_evals``, and the pipeline itself logs
            against the shared overall budget.
    """

    strategy_name = "optimizer"

    def __init__(
        self,
        harness: EvaluationHarness,
        *,
        max_evals: int = 50,
        seed: int | None = 0,
        min_improvement: float = 1e-3,
        judge: Any = None,
        suggest_tools: bool = False,
        cache_evaluations: bool = True,
        verbose: bool = False,
        validation_gate: bool = True,
        validation_tolerance: float | None = None,
        proposer_judge: Any = None,
        alpha: float = 0.05,
        repeats: int | None = None,
        detect_nondeterminism: bool = True,
        holdout_fraction: float = 0.25,
        min_holdout_rows: int = 10,
        allow_small_sample: bool = True,
    ):
        _check_budget("max_evals", max_evals)
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if repeats is not None:
            _check_budget("repeats", repeats)
        if not 0.0 <= holdout_fraction < 1.0:
            raise ValueError(f"holdout_fraction must be in [0, 1), got {holdout_fraction!r}")
        _check_budget("min_holdout_rows", min_holdout_rows)
        if min_improvement < 0:
            raise ValueError(f"min_improvement must be >= 0, got {min_improvement!r}")
        if validation_tolerance is not None:
            # Accepting it silently would promise a behaviour that no longer
            # exists: the gate now requires a held-out gain.
            warnings.warn(
                "validation_tolerance is no longer used: the validation gate now requires a "
                "held-out gain (see min_improvement and alpha), not merely no regression.",
                FutureWarning,
                stacklevel=2,
            )
        self.harness = harness
        self.max_evals = max_evals
        self.seed = seed
        self.min_improvement = min_improvement
        self.validation_gate = validation_gate
        self.validation_tolerance = validation_tolerance
        self.alpha = alpha
        self.repeats = repeats
        self.detect_nondeterminism = detect_nondeterminism
        self.holdout_fraction = holdout_fraction
        self.min_holdout_rows = min_holdout_rows
        self.allow_small_sample = allow_small_sample
        # Per-run measurement mode, set by ``optimize`` (or a pipeline that
        # owns this stage): whether the target is non-deterministic and how
        # many measurements each comparison averages.
        self._nondeterministic = False
        self._repeats_eff = repeats if repeats is not None else 1
        # Few-shot candidates' source rows: (parameter, repr(value)) -> row
        # indices. A demonstration block is never scored on the rows it was
        # copied from. Shared across a pipeline's stages.
        self._demo_rows: dict[tuple[str, str], frozenset[int]] = {}
        # True while a PipelineOptimizer runs this optimizer as a stage: the
        # pipeline owns the split, the non-determinism check and the verdict.
        self._nested = False
        self.judge = judge
        self.proposer_judge = proposer_judge
        self.suggest_tools = suggest_tools
        self.cache_evaluations = cache_evaluations
        self.verbose = verbose
        self._rng = random.Random(seed)
        # Set per-run by the search strategy before evaluating candidates; the
        # baseline values every candidate config is composed on top of.
        self._baseline_snapshot: dict[str, Any] = {}
        # Shared advisory-suggestion sink, replaced per ``optimize`` call.
        self._recommendations: list[str] = []
        # Log the shared-judge warning once per optimizer, not once per run.
        self._shared_judge_warned = False
        # Evaluation-report cache: owned per ``optimize`` call, or injected by
        # a PipelineOptimizer so its stages share hits. ``None`` = no cache.
        self._eval_cache: dict[str, EvaluationReport] | None = None
        # Wall-clock start of the current ``optimize`` call, for verbose
        # per-trial elapsed-time logging and the result's duration_seconds.
        # Reset at the top of every call, since one optimizer instance is
        # commonly reused across several runs.
        self._run_started_at: float = 0.0

    # -- public API ------------------------------------------------------------

    def optimize(
        self,
        agent: Any,
        dataset: GoldenDataset,
        *,
        val_dataset: GoldenDataset | None = None,
        runner: Any = None,
        components: dict[str, Any] | None = None,
        parameters: list[Parameter] | None = None,
    ) -> OptimizationResult:
        """Optimize ``agent`` against ``dataset`` and return the result.

        The best configuration is applied to the live agent before returning --
        unless the final verdict discards it: no gain on the held-out set
        (``val_dataset``, or an automatic split of ``dataset``; see
        ``validation_gate``), or a gain that is not significant where enough
        rows exist to tell. A discarded run leaves the agent exactly as it was.
        Extra kwargs (``runner``/``components``/``parameters``) are forwarded to
        :func:`~adapt_agent.optimization.target.wrap` when ``agent`` is not
        already an :class:`OptimizableAgent`.

        Raises:
            ValueError: If ``dataset`` is empty, or ``val_dataset`` is given but
                empty (an empty held-out set would silently disable the gate).
        """
        target = wrap(agent, runner=runner, components=components, parameters=parameters)
        if not dataset:
            raise ValueError("Cannot optimize against an empty dataset")
        nested = getattr(self, "_nested", False)
        # Re-seeded per call: the same instance and seed must propose the same
        # trials on its second run as on its first.
        self._rng = random.Random(self.seed)
        if nested:
            train, holdout, auto = dataset, val_dataset, False
        else:
            self._demo_rows = {}
            train, holdout, auto = self._resolve_holdout(dataset, val_dataset)

        # Own a fresh evaluation cache for this run -- unless one was injected
        # (a PipelineOptimizer sharing hits across its stages), which outlives
        # this call and is left alone.
        owns_cache = self.cache_evaluations and self._eval_cache is None
        if owns_cache:
            self._eval_cache = {}
        self._run_started_at = time.perf_counter()
        try:
            baseline_snapshot = target.snapshot()
            baseline_report = self._evaluate_baseline(target, train)
            baseline_rows = row_scores(baseline_report)
            if not nested:
                baseline_rows = self._detect_nondeterminism(
                    target, train, baseline_report, baseline_rows
                )
            baseline_score = baseline_report.score
            if self.verbose:
                logger.info(
                    "[%s] baseline score=%.4f (search budget: up to %d evals)",
                    self.strategy_name,
                    baseline_score,
                    self.max_evals,
                )

            # One shared sink threaded through every ProposalContext and read back
            # after the search (proposers may append advisory suggestions to it).
            self._recommendations = []
            self._recommendations.extend(self._shared_judge_notes())

            state = _SearchState(
                best_config={},
                best_score=baseline_score,
                best_report=baseline_report,
                baseline_snapshot=baseline_snapshot,
                history=[],
                best_rows=baseline_rows,
            )
            self._search(target, train, state)

            # Apply the winning configuration permanently to the live agent.
            target.restore(baseline_snapshot)
            if state.best_config:
                target.apply(state.best_config)

            if nested:
                # The owning pipeline judges the cumulative result once.
                gate = _ValidationGate(assessed=False)
            else:
                gate = self._final_verdict(
                    target, train, holdout, auto, baseline_snapshot, baseline_rows, state
                )
            if gate.discarded:
                state.best_config = {}
                state.best_score = baseline_score
                state.best_report = baseline_report

            untuned = self._untuned(self._tunable_names(target), state.history, baseline_snapshot)
            recommendations = list(self._recommendations)
            recommendations.extend(gate.notes)
            if untuned and not nested:
                recommendations.append(self._warn_untuned(untuned))
            recommendations.extend(self._suggest_tools(target, state))

            result = self._build_result(
                state, baseline_score, baseline_snapshot, gate, recommendations, untuned
            )
            if self.verbose:
                logger.info("[%s] %r", self.strategy_name, result)
            return result
        finally:
            if owns_cache:
                self._eval_cache = None

    def _build_result(
        self,
        state: _SearchState,
        baseline_score: float,
        baseline_snapshot: dict[str, Any],
        gate: _ValidationGate,
        recommendations: list[str],
        untuned: list[str],
    ) -> OptimizationResult:
        """Assemble the result, carrying the verdict's evidence alongside the scores."""
        comparison = gate.comparison
        significant: bool | None = None
        if gate.assessed:
            significant = bool(comparison is not None and comparison.significant)
            if gate.discarded:
                significant = False
        return OptimizationResult(
            best_config=state.best_config,
            best_score=state.best_score,
            baseline_score=baseline_score,
            baseline_config=baseline_snapshot,
            history=state.history,
            best_report=state.best_report,
            validation_score=gate.score,
            validation_complete=gate.complete,
            recommendations=recommendations,
            duration_seconds=time.perf_counter() - self._run_started_at,
            validation_baseline_score=gate.baseline_score,
            validation_rejected=gate.rejected,
            rejected_config=gate.rejected_config,
            significant=significant,
            delta=comparison.delta if comparison is not None else None,
            p_value=comparison.p_value if comparison is not None else None,
            ci_low=comparison.ci_low if comparison is not None else None,
            ci_high=comparison.ci_high if comparison is not None else None,
            n_rows=comparison.n_rows if comparison is not None else 0,
            repeats=self._repeats(),
            alpha=self.alpha,
            significance_basis=gate.basis,
            nondeterministic=self._nondeterministic,
            untuned_parameters=list(untuned),
            rejected_reason=gate.reason,
        )

    # -- measurement mode --------------------------------------------------------

    def _resolve_holdout(
        self, dataset: GoldenDataset, val_dataset: GoldenDataset | None
    ) -> tuple[GoldenDataset, GoldenDataset | None, bool]:
        """Return ``(train, holdout, automatic)`` for this run.

        An explicit but empty ``val_dataset`` is an error rather than "no
        validation": treating it as absent would switch the gate off without a
        word, and the caller plainly meant to have one.
        """
        if val_dataset is not None:
            if len(val_dataset) == 0:
                raise ValueError(
                    "val_dataset is empty, so the validation gate could not check anything. "
                    "Pass val_dataset=None to optimize without one."
                )
            return dataset, val_dataset, False
        fraction = self.holdout_fraction
        if fraction <= 0:
            return dataset, None, False
        n_holdout = int(round(len(dataset) * fraction))
        floor = self.min_holdout_rows
        if n_holdout < floor or len(dataset) - n_holdout < floor:
            return dataset, None, False
        train, holdout = dataset.split(1.0 - fraction, seed=self._stat_seed)
        logger.info(
            "[%s] no val_dataset given: holding out %d of %d rows for the final check",
            self.strategy_name,
            len(holdout),
            len(dataset),
        )
        return train, holdout, True

    def _detect_nondeterminism(
        self,
        target: OptimizableAgent,
        dataset: GoldenDataset,
        baseline_report: EvaluationReport,
        baseline_rows: list[float | None] | None,
    ) -> list[float | None] | None:
        """Re-run the baseline once; if rows change, measure everything repeatedly.

        One measurement of a non-deterministic target is a sample, not a
        score: comparing one sample per configuration is how a knob with no
        effect at all came out "better" most of the time. Returns the baseline
        rows to seed the search with (averaged over both runs when they
        differed).
        """
        self._nondeterministic = False
        if self.repeats is not None:
            self._repeats_eff = self.repeats
            return baseline_rows
        self._repeats_eff = 1
        if not self.detect_nondeterminism or baseline_rows is None:
            return baseline_rows
        again = self.harness.evaluate(target, dataset)
        again_rows = row_scores(again)
        if again_rows is None or not rows_differ(baseline_rows, again_rows):
            return baseline_rows
        changed = sum(
            1
            for a, b in zip(baseline_rows, again_rows, strict=False)
            if a is not None and b is not None and a != b
        )
        self._nondeterministic = True
        self._repeats_eff = DEFAULT_NONDETERMINISTIC_REPEATS
        logger.warning(
            "[%s] the target is non-deterministic: %d of %d rows scored differently on a "
            "second run of the unchanged baseline. Every comparison is now measured %d "
            "times per configuration (incumbent and challenger alternately); set repeats= "
            "to choose the count.",
            self.strategy_name,
            changed,
            len(baseline_rows),
            self._repeats_eff,
        )
        return mean_rows([baseline_rows, again_rows])

    def _repeats(self) -> int:
        return max(1, int(getattr(self, "_repeats_eff", 1) or 1))

    def _fresh_measurements(self) -> bool:
        """Whether each measurement must be a real re-run (never a cache hit)."""
        return bool(getattr(self, "_nondeterministic", False)) or self._repeats() > 1

    @property
    def _stat_seed(self) -> int:
        """Seed for resampling in the significance test and the automatic split.

        Fixed, and separate from the search RNG, so testing a candidate never
        perturbs which candidates are proposed next.
        """
        seed = getattr(self, "seed", 0)
        return seed if isinstance(seed, int) else 0

    def _passes(self, comparison: PairedComparison) -> bool:
        """The acceptance rule: a large enough gain that chance does not explain.

        Below the row count at which any outcome could be significant the test
        cannot decide; ``allow_small_sample`` then falls back to the point
        estimate (and the result reports the gain as unproven).
        """
        if comparison.delta <= getattr(self, "min_improvement", 0.0):
            return False
        if comparison.decidable:
            return comparison.significant
        return bool(getattr(self, "allow_small_sample", True))

    def _compare(
        self,
        incumbent: Sequence[float | None] | None,
        challenger: Sequence[float | None] | None,
        *,
        exclude: frozenset[int] = frozenset(),
        repeats: int = 1,
        interval: bool = False,
    ) -> PairedComparison | None:
        if incumbent is None or challenger is None or len(incumbent) != len(challenger):
            return None
        return compare_rows(
            incumbent,
            challenger,
            alpha=getattr(self, "alpha", 0.05),
            min_effect=getattr(self, "min_improvement", 0.0),
            exclude=exclude,
            repeats=repeats,
            interval=interval,
            seed=self._stat_seed,
        )

    def _excluded_rows(self, *states: dict[str, Any]) -> frozenset[int]:
        """Rows any few-shot block in these live states was built from.

        A demonstration copied from a row answers that row; scoring the block
        there measures the copy, not the agent. Rows excluded for either side
        are left out of both, so the comparison stays paired.
        """
        registry: dict[tuple[str, str], frozenset[int]] = getattr(self, "_demo_rows", {})
        if not registry:
            return frozenset()
        rows: set[int] = set()
        for state in states:
            for name, value in state.items():
                found = registry.get((name, _value_key(value)))
                if found:
                    rows.update(found)
        return frozenset(rows)

    def _register_demo_rows(self, param: Parameter, ctx: ProposalContext) -> None:
        """Record which rows each proposed few-shot block was copied from."""
        demo_rows = getattr(ctx, "demo_rows", None)
        if not demo_rows:
            return
        registry = getattr(self, "_demo_rows", None)
        if registry is None:
            registry = self._demo_rows = {}
        for value, rows in demo_rows.items():
            registry[(param.name, _value_key(value))] = frozenset(rows)

    def _measure_config(
        self,
        target: OptimizableAgent,
        snapshot: dict[str, Any],
        config: dict[str, Any],
        dataset: GoldenDataset,
    ) -> EvaluationReport:
        """A fresh, uncached pass of ``snapshot`` + ``config`` (re-run once if throttled)."""
        target.restore(snapshot)
        if config:
            target.apply(config)
        return self._evaluate_validation(target, dataset)

    def _paired_passes(
        self,
        target: OptimizableAgent,
        snapshot: dict[str, Any],
        config: dict[str, Any],
        dataset: GoldenDataset,
    ) -> tuple[list[EvaluationReport], list[EvaluationReport]]:
        """Measure ``config`` and the untuned agent alternately, ``repeats`` times each.

        Alternating -- tuned, untuned, tuned, ... -- rather than all of one and
        then all of the other keeps a drifting provider (load, a model update
        mid-run) from landing on one side of the comparison. Leaves the target
        at the untuned snapshot.
        """
        tuned: list[EvaluationReport] = []
        untuned: list[EvaluationReport] = []
        for _ in range(self._repeats()):
            tuned.append(self._measure_config(target, snapshot, config, dataset))
            untuned.append(self._measure_config(target, snapshot, {}, dataset))
        target.restore(snapshot)
        return tuned, untuned

    @staticmethod
    def _rows_of(reports: Sequence[EvaluationReport]) -> list[float | None] | None:
        rows = [row_scores(r) for r in reports]
        present = [r for r in rows if r is not None]
        if not rows or len(present) != len(rows):
            return None
        return mean_rows(present)

    # -- final verdict -----------------------------------------------------------

    def _final_verdict(
        self,
        target: OptimizableAgent,
        train: GoldenDataset,
        holdout: GoldenDataset | None,
        automatic: bool,
        baseline_snapshot: dict[str, Any],
        baseline_rows: list[float | None] | None,
        state: _SearchState,
    ) -> _ValidationGate:
        """Decide whether the search's winner is applied, and how sure that is.

        On entry the live ``target`` carries ``state.best_config``; on exit it
        carries it if the verdict passed and the baseline if it did not.

        With a held-out set (and the gate on) the winner needs a held-out
        **gain**. Otherwise it is judged on the training rows -- re-measured
        afresh for a non-deterministic target, since the search's own
        measurements of the winner are the lucky draws that made it win.
        """
        basis = "held_out_auto" if automatic else "held_out"
        if holdout is not None and not state.best_config:
            # Nothing was tuned: the agent *is* the baseline, measured once.
            tuned = self._evaluate_validation(target, holdout)
            return _ValidationGate(
                score=tuned.score,
                complete=tuned.is_complete,
                baseline_score=tuned.score,
                basis=basis,
            )
        if not state.best_config:
            return _ValidationGate(basis="")
        if holdout is not None and self.validation_gate:
            return self._held_out_verdict(
                target, holdout, baseline_snapshot, state.best_config, basis
            )
        gate = self._training_verdict(target, train, baseline_snapshot, baseline_rows, state)
        if holdout is not None:
            # The gate is off: the held-out score is reported, not acted on.
            # Measured on whatever the training verdict left applied.
            tuned = self._evaluate_validation(target, holdout)
            gate.score = tuned.score
            gate.complete = tuned.is_complete
        return gate

    def _held_out_verdict(
        self,
        target: OptimizableAgent,
        holdout: GoldenDataset,
        baseline_snapshot: dict[str, Any],
        best_config: dict[str, Any],
        basis: str,
    ) -> _ValidationGate:
        """Apply the winner only if it gains on rows the search never saw.

        Both sides are compared on the rows *both* passes scored, so a row
        throttled in one pass drops out of the other too and a partial pass is
        still a like-for-like comparison. If no row survives, there is no
        evidence of a gain and the winner is not applied.
        """
        tuned_reports, base_reports = self._paired_passes(
            target, baseline_snapshot, best_config, holdout
        )
        tuned_score = _mean_score(tuned_reports)
        base_score = _mean_score(base_reports)
        base_complete = all(r.is_complete for r in base_reports)
        complete = base_complete and all(r.is_complete for r in tuned_reports)
        comparison = self._compare(
            self._rows_of(base_reports),
            self._rows_of(tuned_reports),
            repeats=self._repeats(),
            interval=True,
        )
        if comparison is None:
            # The harness kept fewer records than rows: no per-row evidence.
            # Only a complete pair of means is comparable at all.
            held_out_gain = complete and tuned_score - base_score > self.min_improvement
        else:
            held_out_gain = self._passes(comparison)
        if not held_out_gain:
            # Leave the baseline applied: the agent is exactly as it was given.
            evidence = comparison.describe() if comparison is not None else "no per-row results"
            note = (
                f"Validation gate: the tuned configuration scored {tuned_score:.4f} on the "
                f"held-out set against {base_score:.4f} for the untuned agent ({evidence}). "
                "A held-out gain is required, so it was NOT applied. The gain on the "
                "training data did not generalise; add rows or review "
                "OptimizationResult.rejected_config by hand."
            )
            logger.warning("[%s] %s", self.strategy_name, note)
            return _ValidationGate(
                score=base_score,
                complete=base_complete,
                baseline_score=base_score if base_complete else None,
                rejected=True,
                rejected_config=dict(best_config),
                notes=[note],
                comparison=comparison,
                basis=basis,
                discarded=True,
                reason="no gain on the held-out set",
            )
        target.restore(baseline_snapshot)
        target.apply(best_config)
        notes = self._verdict_notes(comparison, basis)
        return _ValidationGate(
            score=tuned_score,
            complete=complete,
            baseline_score=base_score if base_complete else None,
            comparison=comparison,
            basis=basis,
            notes=notes,
        )

    def _training_verdict(
        self,
        target: OptimizableAgent,
        train: GoldenDataset,
        baseline_snapshot: dict[str, Any],
        baseline_rows: list[float | None] | None,
        state: _SearchState,
    ) -> _ValidationGate:
        """Judge the winner on the training rows, when there is no held-out set."""
        exclude = self._excluded_rows(_compose(baseline_snapshot, state.best_config))
        if self._fresh_measurements():
            tuned_reports, base_reports = self._paired_passes(
                target, baseline_snapshot, state.best_config, train
            )
            incumbent = self._rows_of(base_reports)
            challenger = self._rows_of(tuned_reports)
        else:
            incumbent = baseline_rows
            challenger = state.best_rows
            if challenger is None:
                challenger = row_scores(state.best_report)
        comparison = self._compare(
            incumbent, challenger, exclude=exclude, repeats=self._repeats(), interval=True
        )
        if comparison is not None and not self._passes(comparison):
            target.restore(baseline_snapshot)
            note = (
                f"Significance: the best configuration found is not a significant gain over "
                f"the untuned agent ({comparison.describe()}), so it was NOT applied. Add "
                "rows, or raise repeats for a noisy target."
            )
            logger.warning("[%s] %s", self.strategy_name, note)
            return _ValidationGate(
                rejected_config=dict(state.best_config),
                notes=[note],
                comparison=comparison,
                basis="training",
                discarded=True,
                reason="gain not significant",
            )
        target.restore(baseline_snapshot)
        target.apply(state.best_config)
        notes = self._verdict_notes(comparison, "training")
        if comparison is None:
            notes.append(
                "Significance could not be assessed: the harness stored fewer per-row "
                "results than rows were run (raise EvaluationHarness max_results)."
            )
        return _ValidationGate(comparison=comparison, basis="training", notes=notes)

    def _verdict_notes(self, comparison: PairedComparison | None, basis: str) -> list[str]:
        """What an applied-but-unproven or selection-biased verdict needs to say."""
        if comparison is None:
            return []
        notes: list[str] = []
        if not comparison.decidable:
            notes.append(
                f"Small sample: {comparison.describe()}. The configuration was applied on "
                "its point estimate, but the gain is unproven (significant=False, "
                "improved=False) until it holds on more rows."
            )
        elif basis == "training" and comparison.significant:
            split_hint = ""
            if self.holdout_fraction > 0:
                split_hint = (
                    f" (or give the golden set at least {self._rows_for_auto_holdout()} rows "
                    "for an automatic hold-out)"
                )
            notes.append(
                "Significance was measured on the training rows the search selected on, "
                f"which flatters the winner. Pass a val_dataset{split_hint} for an unbiased "
                "check."
            )
        return notes

    def _rows_for_auto_holdout(self) -> int:
        """The smallest golden set :meth:`_resolve_holdout` would split."""
        n = self.min_holdout_rows
        while True:
            held = int(round(n * self.holdout_fraction))
            if held >= self.min_holdout_rows and n - held >= self.min_holdout_rows:
                return n
            n += 1

    def _tunable_names(self, target: OptimizableAgent) -> list[str]:
        """Parameters this strategy could vary: those with more than one value."""
        names: list[str] = []
        for param in target.search_space.optimizable():
            try:
                options = param.enumerate_candidates()
            except Exception:
                continue
            if len(options) > 1:
                names.append(param.name)
        return names

    @staticmethod
    def _untuned(names: list[str], history: list[Trial], snapshot: dict[str, Any]) -> list[str]:
        """Which of ``names`` no trial ever moved away from its starting value."""
        return [
            name
            for name in names
            if not any(
                name in t.config and not _same(t.config[name], snapshot.get(name)) for t in history
            )
        ]

    def _warn_untuned(self, untuned: list[str]) -> str:
        note = (
            f"Not tuned: {', '.join(untuned)} received zero evaluations (the budget ran out "
            "or no candidate was proposed) and were left as they were. Raise max_evals or "
            "narrow the search space."
        )
        logger.warning("[%s] %s", self.strategy_name, note)
        return note

    # -- strategy hook ---------------------------------------------------------

    def _search(
        self, target: OptimizableAgent, dataset: GoldenDataset, state: _SearchState
    ) -> None:
        """Populate ``state`` with trials. Overridden by concrete optimizers."""
        raise NotImplementedError

    # -- shared helpers --------------------------------------------------------

    def _eval_config(
        self,
        target: OptimizableAgent,
        config: dict[str, Any],
        dataset: GoldenDataset,
        *,
        fresh: bool = False,
    ) -> EvaluationReport:
        """Evaluate a *full* config relative to the baseline snapshot.

        Restores the baseline first so configs compose predictably regardless of
        whatever the previous trial left applied. When evaluation caching is on
        and this exact live state has already been measured against this
        dataset, the recorded report is returned without re-running the agent
        -- the restore/apply still happens, so live-state side effects (what
        proposers read off the components) are identical either way.

        ``fresh=True`` always re-runs and never stores: a repeated measurement
        of a non-deterministic target is only a measurement if it is a new
        draw, and a cached one would make every "repeat" the same sample.
        """
        target.restore(self._current_baseline)
        if config:
            target.apply(config)
        if fresh:
            return self.harness.evaluate(target, dataset)
        cached = self._cached_report(target, dataset)
        if cached is not None:
            return cached
        report = self.harness.evaluate(target, dataset)
        self._store_report(target, dataset, report)
        return report

    def _trial(
        self,
        target: OptimizableAgent,
        state: _SearchState,
        config: dict[str, Any],
        dataset: GoldenDataset,
    ) -> tuple[bool, float]:
        """Measure ``config`` against the incumbent, record the trial.

        Deterministic target: one (cacheable) measurement of the challenger,
        compared row by row with the incumbent's recorded rows. Otherwise both
        are re-measured ``repeats`` times, alternately, on the same rows -- a
        fresh incumbent each time, so one lucky-low incumbent measurement
        cannot make every later challenger look good.

        Returns ``(accepted, rank_score)``; the rank score is the challenger's
        mean over the rows it was compared on (its own demonstration rows
        left out), for strategies that rank a population.
        """
        exclude = self._excluded_rows(
            _compose(self._current_baseline, config),
            _compose(self._current_baseline, state.best_config),
        )
        repeats = self._repeats()
        if self._fresh_measurements():
            incumbent_reports: list[EvaluationReport] = []
            reports: list[EvaluationReport] = []
            for _ in range(repeats):
                incumbent_reports.append(
                    self._eval_config(target, state.best_config, dataset, fresh=True)
                )
                reports.append(self._eval_config(target, config, dataset, fresh=True))
            incumbent = self._rows_of(incumbent_reports)
            challenger = self._rows_of(reports)
        else:
            reports = [self._eval_config(target, config, dataset)]
            incumbent = getattr(state, "best_rows", None)
            if incumbent is None:
                incumbent = row_scores(state.best_report)
            challenger = row_scores(reports[0])
        comparison = self._compare(incumbent, challenger, exclude=exclude, repeats=repeats)
        accepted = self._record(
            state, config, reports[-1], comparison=comparison, rows=challenger, reports=reports
        )
        rank = _mean_score(reports)
        if challenger is not None:
            kept = [v for i, v in enumerate(challenger) if v is not None and i not in exclude]
            if kept:
                rank = sum(kept) / len(kept)
        return accepted, rank

    # -- evaluation cache ------------------------------------------------------

    def _cache_key(self, target: OptimizableAgent, dataset: GoldenDataset) -> str | None:
        """Key the harness and the *live* parameter state against a dataset.

        The key renders every parameter's current value with ``repr``: equal
        primitives (prompts, models, temperatures) collide -- a hit -- while
        live objects (tool lists holding callables) render by identity, so two
        distinct-but-equal objects miss rather than falsely hit, which only
        costs a re-run. Keyed on the live snapshot rather than the candidate
        diff so a stage baseline and the previous stage's winning trial -- the
        same state reached through different configs -- share a key.

        The harness is part of the key because a report is only reusable by an
        evaluator that would have produced it: a pipeline may be assembled
        from stages carrying *different* harnesses (other metrics, another
        judge), and without the harness in the key such a stage would rank its
        candidates against scores measured by a different evaluator. Stages
        sharing one harness instance -- how :func:`make_default_optimizer`
        builds pipelines -- still share hits. Returns ``None`` (no caching)
        when the state cannot be rendered.
        """
        try:
            state = target.snapshot()
            rendered = ",".join(f"{name}={state[name]!r}" for name in sorted(state))
        except Exception:
            return None
        return f"{id(self.harness)}|{id(dataset)}|{rendered}"

    def _cached_report(
        self, target: OptimizableAgent, dataset: GoldenDataset
    ) -> EvaluationReport | None:
        """Return the recorded report for the current live state, if any.

        Attribute access is defensive (``getattr`` with a disabled default)
        because ``_evaluate_baseline`` is a documented extension point also
        exercised on minimal subclasses that skip ``Optimizer.__init__``.
        """
        cache: dict[str, EvaluationReport] | None = getattr(self, "_eval_cache", None)
        if cache is None or not getattr(self, "cache_evaluations", False):
            return None
        key = self._cache_key(target, dataset)
        if key is None:
            return None
        report = cache.get(key)
        if report is not None and self.verbose:
            logger.info("[%s] evaluation cache hit", self.strategy_name)
        return report

    def _store_report(
        self, target: OptimizableAgent, dataset: GoldenDataset, report: EvaluationReport
    ) -> None:
        """Record a report for the current live state.

        Incomplete reports (rows lost to transient provider failures) are never
        stored: their scores are means over a subset, and freezing one would
        also defeat the baseline's deliberate re-run-on-incomplete machinery.
        """
        cache: dict[str, EvaluationReport] | None = getattr(self, "_eval_cache", None)
        if cache is None or not getattr(self, "cache_evaluations", False):
            return
        if not report.is_complete:
            return
        key = self._cache_key(target, dataset)
        if key is not None:
            cache[key] = report

    def _evaluate_baseline(
        self, target: OptimizableAgent, dataset: GoldenDataset
    ) -> EvaluationReport:
        """Evaluate the starting configuration, re-running an incomplete result.

        The baseline never passes through :meth:`_record`, so the completeness
        invariant enforced there does not reach it -- and the baseline is the
        one score every candidate is measured against. A baseline throttled on a
        hard row scores over its survivors, sets ``best_score`` too high, and no
        fully-evaluated candidate can beat it: the search then returns the
        starting config as the winner, which looks like "nothing improved on
        your prompt" rather than like a broken run.

        A baseline is re-run once rather than rejected outright, because there is
        nothing to fall back to. If it is *still* incomplete this raises: a
        logged warning is too easy to lose in a long run's output, and the
        failure mode it precedes is a wrong answer wearing the costume of a
        valid one.

        With evaluation caching on, a baseline whose exact state was already
        measured in this run reuses that report -- which is how a pipeline
        stage's baseline (the previous stage's winner) stops costing a
        full-dataset pass. Only complete reports are ever cached, so the
        re-run-on-incomplete path below always operates on fresh evaluations.
        """
        cached = self._cached_report(target, dataset)
        if cached is not None:
            return cached
        report = self.harness.evaluate(target, dataset)
        if report.is_complete:
            self._store_report(target, dataset, report)
            return report
        logger.warning(
            "[%s] baseline scored %.4f over %d of %d rows (%d transient failures); re-running",
            self.strategy_name,
            report.score,
            report.n_scored,
            report.n_evaluated,
            report.n_transient_errors,
        )
        retried = self.harness.evaluate(target, dataset)
        if not retried.is_complete:
            raise IncompleteEvaluationError(
                f"{self.strategy_name}: the baseline evaluation still could not score "
                f"{retried.n_transient_errors} of {retried.n_evaluated} rows after a re-run "
                f"(transient provider failures). Every candidate is compared against the "
                f"baseline, so continuing would silently return the starting configuration "
                f"as the winner. Lower the concurrency, widen the retry policy, or wait for "
                f"the provider to recover."
            )
        self._store_report(target, dataset, retried)
        return retried

    @property
    def _rewriting_judge(self) -> Any:
        """The judge that critiques and rewrites: ``proposer_judge``, else ``judge``."""
        return self.proposer_judge if self.proposer_judge is not None else self.judge

    def _shared_judge_notes(self) -> list[str]:
        """Say so when the judge that writes candidates also marks them.

        "Same judge" means the same object *or* the same underlying completion
        callable: two ``LLMJudge`` instances over one model are one marker with
        two rubrics, not an independent second opinion.
        """
        rewriting = self._rewriting_judge
        if rewriting is None:
            return []
        scoring = [
            j
            for j in (getattr(m, "judge", None) for m in getattr(self.harness, "metrics", []))
            if j is not None
        ]

        def same(a: Any, b: Any) -> bool:
            if a is b:
                return True
            ca, cb = getattr(a, "complete", None), getattr(b, "complete", None)
            return ca is not None and ca is cb

        if not any(same(rewriting, j) for j in scoring):
            return []
        note = (
            "Independence: the judge that rewrites prompts is also the judge that scores "
            "them, so reported gains may partly reflect the judge preferring its own "
            "writing. The held-out validation gate does not correct for this (the same "
            "judge scores it). Pass proposer_judge=<a judge on a different model family>, "
            "or score with a deterministic metric."
        )
        if not self._shared_judge_warned:
            self._shared_judge_warned = True
            logger.warning("[%s] %s", self.strategy_name, note)
        return [note]

    def _evaluate_validation(
        self, target: OptimizableAgent, dataset: GoldenDataset
    ) -> EvaluationReport:
        """Evaluate the held-out set, re-running once if throttling cost it rows.

        Validation does not steer the search, so unlike the baseline this never
        aborts -- a partial held-out score is still worth reporting. It must not
        be reported *as if* it were whole, though: it is the number a user reads
        to decide whether the tuned config generalises, and a mean over the rows
        that happened to survive throttling silently answers a different
        question. The completeness travels with it on
        :attr:`OptimizationResult.validation_complete`.
        """
        report = self.harness.evaluate(target, dataset)
        if report.is_complete:
            return report
        logger.warning(
            "[%s] validation scored %.4f over %d of %d rows (%d transient failures); re-running",
            self.strategy_name,
            report.score,
            report.n_scored,
            report.n_evaluated,
            report.n_transient_errors,
        )
        retried = self.harness.evaluate(target, dataset)
        if not retried.is_complete:
            logger.error(
                "[%s] validation is still incomplete after a re-run (%d of %d rows lost to "
                "transient failures). validation_score covers only the rows that succeeded; "
                "OptimizationResult.validation_complete is False.",
                self.strategy_name,
                retried.n_transient_errors,
                retried.n_evaluated,
            )
        return retried

    def _record(
        self,
        state: _SearchState,
        config: dict[str, Any],
        report: EvaluationReport,
        *,
        comparison: PairedComparison | None = None,
        rows: list[float | None] | None = None,
        reports: Sequence[EvaluationReport] | None = None,
    ) -> bool:
        """Record a trial and update the best-so-far. Returns True if accepted.

        Acceptance is a statistical decision on the per-row deltas between the
        challenger and the incumbent (:meth:`_passes`), never a comparison of
        two means. ``comparison`` is computed here from the two reports when
        the caller has not already made it; only when neither side has per-row
        results (a report built by hand) does the mean comparison remain.

        An *incomplete* trial can never be accepted, however well it scored.
        Transient failures drop rows from the score, so an incomplete trial's
        mean is taken over a different subset than its rivals' -- a candidate
        that answers one easy row and gets throttled on a hard one scores 1.0
        and would otherwise beat a fully-evaluated 0.9. Excluding throttled rows
        stops throttling from *penalising* a candidate; refusing to crown a
        partial trial stops it from *rewarding* one. Both are needed, or the
        search still selects for luck, just in the opposite direction.
        """
        measured = list(reports) if reports else [report]
        complete = all(r.is_complete for r in measured)
        score = _mean_score(measured)
        if comparison is None:
            incumbent = getattr(state, "best_rows", None)
            if incumbent is None:
                incumbent = row_scores(state.best_report)
            if rows is None:
                rows = row_scores(report)
            comparison = self._compare(incumbent, rows)
        if comparison is None:
            gain = score > state.best_score + self.min_improvement
        else:
            gain = self._passes(comparison)
        accepted = complete and gain
        trial = Trial(
            config=dict(config),
            score=score,
            strategy=self.strategy_name,
            accepted=accepted,
            metrics=_mean_aggregate(measured),
            complete=complete,
            n_transient_errors=sum(r.n_transient_errors for r in measured),
            delta=comparison.delta if comparison is not None else None,
            p_value=comparison.p_value if comparison is not None else None,
            significant=comparison.significant if comparison is not None else None,
            n_rows=comparison.n_rows if comparison is not None else 0,
            repeats=len(measured),
        )
        if not complete:
            logger.warning(
                "[%s] trial #%d scored %.4f over %d of %d rows (%d transient failures); "
                "not eligible to win -- its score is not comparable with a complete trial's",
                self.strategy_name,
                len(state.history) + 1,
                report.score,
                report.n_scored,
                report.n_evaluated,
                report.n_transient_errors,
            )
        state.history.append(trial)
        if accepted:
            state.best_config = dict(config)
            state.best_score = score
            state.best_report = report
            state.best_rows = rows if rows is not None else row_scores(report)
        if self.verbose:
            # Position out of the run's budget plus wall-clock elapsed time --
            # not just the score -- so a long run distinguishes "still
            # working" from "hung" without an external progress indicator.
            logger.info(
                "[%s] trial %d/%d score=%.4f %s (%.1fs elapsed)",
                self.strategy_name,
                len(state.history),
                self.max_evals,
                score,
                "ACCEPT" if accepted else "",
                time.perf_counter() - self._run_started_at,
            )
        return accepted

    @property
    def _current_baseline(self) -> dict[str, Any]:
        return self._baseline_snapshot

    def _proposal_context(
        self,
        target: OptimizableAgent,
        param: Parameter,
        dataset: GoldenDataset,
        report: EvaluationReport | None,
        n: int,
    ) -> ProposalContext:
        return ProposalContext(
            parameter=param,
            agent=target,
            dataset=dataset,
            report=report,
            judge=self._rewriting_judge,
            rng=self._rng,
            n=n,
            recommendations=self._recommendations,
        )

    # -- tool/skill suggestion -------------------------------------------------

    def _suggest_tools(self, target: OptimizableAgent, state: _SearchState) -> list[str]:
        """Ask the judge to propose new tools/skills for TOOL/SKILL components.

        Advisory only: the returned strings describe tools/skills the judge thinks
        would help, drawn from the best configuration's remaining failures. Nothing
        here is applied to the live agent -- the optimizer only *selects* among
        existing tools (via ablation proposers).
        """
        if not self.judge or not self.suggest_tools:
            return []
        suggest = getattr(self.judge, "suggest_tools", None)
        if not callable(suggest):
            return []

        params = [
            p
            for p in target.search_space.optimizable()
            if p.kind in (ParameterKind.TOOL, ParameterKind.SKILL)
        ]
        if not params:
            return []

        failures = self._failure_records(state)
        out: list[str] = []
        for param in params:
            component = param.component or param.name
            current = param.read()
            current_tools = list(current) if isinstance(current, (list, tuple)) else []
            try:
                items = suggest(component, failures, current_tools)
            except Exception as exc:  # a flaky judge must not abort the run
                logger.warning("suggest_tools failed for %s: %s", component, exc)
                continue
            for item in items or []:
                out.append(self._format_tool_suggestion(component, item))
        return out

    @staticmethod
    def _failure_records(state: _SearchState) -> list[dict[str, Any]]:
        """Convert the best report's failing ExampleResults into judge-ready dicts."""
        report = state.best_report
        if report is None:
            return []
        records: list[dict[str, Any]] = []
        for r in report.failures():
            records.append(
                {
                    "input": r.inputs,
                    "output": r.output if r.error is None else f"<error: {r.error}>",
                    "expected": r.expected,
                }
            )
        return records

    @staticmethod
    def _format_tool_suggestion(component: str, item: Any) -> str:
        """Render a judge tool suggestion as a single human-readable line."""
        if isinstance(item, dict):
            name = item.get("name", "<unnamed>")
            description = item.get("description", "")
            rationale = item.get("rationale", "")
            parts = [f"[{component}] tool '{name}'"]
            if description:
                parts.append(str(description))
            if rationale:
                parts.append(f"(why: {rationale})")
            return ": ".join(parts[:2]) + (f" {parts[2]}" if len(parts) > 2 else "")
        return f"[{component}] {item}"


@dataclass
class _SearchState:
    best_config: dict[str, Any]
    best_score: float
    best_report: EvaluationReport | None
    baseline_snapshot: dict[str, Any]
    history: list[Trial]
    #: Per-row scores of the current best (averaged over repeats when there
    #: were any); ``None`` when unknown, and then derived from ``best_report``.
    best_rows: list[float | None] | None = None


# A tiny helper so strategies can set the baseline snapshot on the optimizer.
def _bind_baseline(opt: Optimizer, snapshot: dict[str, Any]) -> None:
    opt._baseline_snapshot = snapshot


class GridSearchOptimizer(Optimizer):
    """Evaluate the Cartesian product of discrete candidate sets.

    When the full product fits the budget (``min(max_configs, max_evals)``) it
    is searched exhaustively. When it does not, truncating the product would
    try only the first few values of the first knob and never move the others
    -- so the search falls back to sweeping each knob in turn, holding the
    rest at the best found so far, with the budget shared fairly between
    knobs. Any knob that still received no evaluation is named in a warning
    and on :attr:`OptimizationResult.untuned_parameters`.
    """

    strategy_name = "grid"

    def __init__(
        self,
        harness: EvaluationHarness,
        *,
        numeric_points: int = 5,
        max_configs: int = 64,
        **kw: Any,
    ):
        super().__init__(harness, **kw)
        _check_budget("numeric_points", numeric_points)
        _check_budget("max_configs", max_configs)
        self.numeric_points = numeric_points
        self.max_configs = max_configs

    def _knobs(self, target: OptimizableAgent) -> list[tuple[str, list[Any]]]:
        knobs: list[tuple[str, list[Any]]] = []
        for param in target.search_space.optimizable():
            options = param.enumerate_candidates(numeric_points=self.numeric_points)
            if len(options) > 1:
                knobs.append((param.name, options))
        return knobs

    def _tunable_names(self, target: OptimizableAgent) -> list[str]:
        return [name for name, _ in self._knobs(target)]

    def _search(
        self, target: OptimizableAgent, dataset: GoldenDataset, state: _SearchState
    ) -> None:
        _bind_baseline(self, state.baseline_snapshot)
        knobs = self._knobs(target)
        if not knobs:
            return
        size = 1
        for _, options in knobs:
            size *= len(options)
        if size <= min(self.max_configs, self.max_evals):
            configs = target.search_space.grid(
                numeric_points=self.numeric_points, max_configs=self.max_configs
            )
            for config in configs:
                if len(state.history) >= self.max_evals:
                    return
                if not config:
                    continue
                self._trial(target, state, config, dataset)
            return
        logger.info(
            "[%s] the full grid has %d configurations but the budget allows %d; sweeping "
            "each of the %d knobs in turn instead, so none is skipped",
            self.strategy_name,
            size,
            min(self.max_configs, self.max_evals),
            len(knobs),
        )
        self._sweep(target, dataset, state, knobs)

    def _sweep(
        self,
        target: OptimizableAgent,
        dataset: GoldenDataset,
        state: _SearchState,
        knobs: list[tuple[str, list[Any]]],
    ) -> None:
        """Coordinate sweep over ``knobs`` within the budget.

        The first pass gives every knob a fair share of what is left (values
        spread evenly across its range when the share is smaller than its
        candidate list); later passes spend any remainder on untried values.
        """
        pending = {
            name: [o for o in options if not _same(o, state.baseline_snapshot.get(name))]
            for name, options in knobs
        }
        order = [name for name, _ in knobs]
        first_pass = True
        while any(pending.values()):
            for position, name in enumerate(order):
                queue = pending[name]
                remaining = self.max_evals - len(state.history)
                if remaining <= 0:
                    return
                if not queue:
                    continue
                if first_pass:
                    knobs_left = sum(1 for n in order[position:] if pending[n])
                    chosen = _spread(len(queue), max(1, remaining // knobs_left))
                else:
                    chosen = list(range(len(queue)))
                for index in chosen:
                    if len(state.history) >= self.max_evals:
                        return
                    config = dict(state.best_config)
                    config[name] = queue[index]
                    self._trial(target, state, config, dataset)
                pending[name] = [v for i, v in enumerate(queue) if i not in set(chosen)]
            first_pass = False


class RandomSearchOptimizer(Optimizer):
    """Sample random whole-space configurations and keep the best."""

    strategy_name = "random"

    def _tunable_names(self, target: OptimizableAgent) -> list[str]:
        return [
            p.name
            for p in target.search_space.optimizable()
            if (p.candidates and len(p.candidates) > 1) or p.bounds is not None
        ]

    def _search(
        self, target: OptimizableAgent, dataset: GoldenDataset, state: _SearchState
    ) -> None:
        _bind_baseline(self, state.baseline_snapshot)
        if not target.search_space.optimizable():
            return
        seen: set[str] = set()
        for _ in range(self.max_evals):
            config = target.search_space.sample_config(self._rng)
            key = repr(sorted(config.items()))
            if key in seen:
                continue
            seen.add(key)
            self._trial(target, state, config, dataset)


class CoordinateAscentOptimizer(Optimizer):
    """Greedy per-parameter improvement driven by proposers.

    For each optimizable parameter, ask the supporting proposers for candidate
    values, evaluate each (holding the current best for all other parameters),
    and keep any significant improvement. Repeat for ``rounds`` passes or until
    a pass yields no improvement. This is the flagship strategy for prompt and
    few-shot optimization and is where the LLM judge improves instructions.

    A few-shot block is scored only on rows it was not built from: its
    demonstrations are copies of dataset rows, and a block scored on the rows
    it copied measures the copy.

    Args:
        proposers: Candidate generators. Defaults to
            :func:`~adapt_agent.optimization.proposers.default_proposers` (which
            includes the LLM proposer when a ``judge`` is supplied).
        rounds: Maximum number of full passes over the parameters.
        candidates_per_param: Proposal count hint per parameter per round.
        kinds: Restrict optimization to these parameter kinds (default: all).
    """

    strategy_name = "coordinate_ascent"

    def __init__(
        self,
        harness: EvaluationHarness,
        *,
        proposers: list[Proposer] | None = None,
        rounds: int = 2,
        candidates_per_param: int = 4,
        kinds: tuple[ParameterKind, ...] | None = None,
        **kw: Any,
    ):
        super().__init__(harness, **kw)
        self.proposers = (
            proposers if proposers is not None else default_proposers(self._rewriting_judge)
        )
        self.rounds = rounds
        self.candidates_per_param = candidates_per_param
        self.kinds = kinds

    def _params(self, target: OptimizableAgent) -> list[Parameter]:
        return [
            p
            for p in target.search_space.optimizable()
            if self.kinds is None or p.kind in self.kinds
        ]

    def _tunable_names(self, target: OptimizableAgent) -> list[str]:
        return [p.name for p in self._params(target) if proposers_for(p, self.proposers)]

    def _search(
        self, target: OptimizableAgent, dataset: GoldenDataset, state: _SearchState
    ) -> None:
        _bind_baseline(self, state.baseline_snapshot)
        params = self._params(target)
        if not params:
            return

        for round_idx in range(self.rounds):
            improved_this_round = False
            for param in params:
                if len(state.history) >= self.max_evals:
                    return
                supporting = proposers_for(param, self.proposers)
                if not supporting:
                    continue
                # Build candidates from every supporting proposer, using the
                # current best report so failure-driven proposers see live errors.
                ctx = self._proposal_context(
                    target, param, dataset, state.best_report, self.candidates_per_param
                )
                candidates: list[Any] = []
                for proposer in supporting:
                    try:
                        candidates.extend(proposer.propose(ctx))
                    except Exception as exc:  # a flaky proposer must not abort the run
                        logger.warning(
                            "Proposer %s failed for %s: %s", proposer.name, param.name, exc
                        )
                self._register_demo_rows(param, ctx)
                for value in _dedup(candidates):
                    if len(state.history) >= self.max_evals:
                        return
                    # Compose on top of the current best configuration.
                    config = dict(state.best_config)
                    config[param.name] = value
                    accepted, _ = self._trial(target, state, config, dataset)
                    if accepted:
                        improved_this_round = True
            if not improved_this_round:
                if self.verbose:
                    logger.info("[%s] converged after round %d", self.strategy_name, round_idx + 1)
                break


class BootstrapFewShotOptimizer(CoordinateAscentOptimizer):
    """Coordinate ascent restricted to few-shot parameters (DSPy-style bootstrap).

    Each proposed block is scored on the rows it did not copy its
    demonstrations from, so a block can only win by helping on unseen rows.
    """

    strategy_name = "bootstrap_few_shot"

    def __init__(self, harness: EvaluationHarness, **kw: Any):
        kw.setdefault("kinds", (ParameterKind.FEW_SHOT,))
        super().__init__(harness, **kw)


class EvolutionaryOptimizer(Optimizer):
    """Population-based search: mutate configs with proposers, select the best.

    Survivors carry their measured score into the next generation instead of
    being re-evaluated, and a configuration already evaluated in this run is
    never evaluated again, so the whole budget goes on new candidates.

    Args:
        proposers: Mutation operators (default proposer set).
        population: Number of configs carried between generations.
        generations: Number of evolve/select cycles.
        mutations_per_parent: Mutations attempted per surviving parent.
    """

    strategy_name = "evolutionary"

    def __init__(
        self,
        harness: EvaluationHarness,
        *,
        proposers: list[Proposer] | None = None,
        population: int = 4,
        generations: int = 3,
        mutations_per_parent: int = 2,
        **kw: Any,
    ):
        super().__init__(harness, **kw)
        self.proposers = (
            proposers if proposers is not None else default_proposers(self._rewriting_judge)
        )
        self.population = population
        self.generations = generations
        self.mutations_per_parent = mutations_per_parent

    def _tunable_names(self, target: OptimizableAgent) -> list[str]:
        return [
            p.name
            for p in target.search_space.optimizable()
            if proposers_for(p, self.proposers) or p.candidates or p.bounds is not None
        ]

    def _search(
        self, target: OptimizableAgent, dataset: GoldenDataset, state: _SearchState
    ) -> None:
        _bind_baseline(self, state.baseline_snapshot)
        params = target.search_space.optimizable()
        if not params:
            return
        # The baseline joins the population with the score it already has:
        # re-recording it would spend a budgeted evaluation to learn nothing.
        seen: set[str] = {_config_key({})}
        population: list[dict[str, Any]] = []
        for _ in range(self.population - 1):
            population.append(target.search_space.sample_config(self._rng))

        scored: list[tuple[dict[str, Any], float]] = [({}, state.best_score)]
        scored.extend(self._score_population(target, dataset, state, population, seen=seen))
        scored.sort(key=lambda cs: cs[1], reverse=True)
        for _ in range(self.generations):
            if len(state.history) >= self.max_evals:
                break
            survivors = scored[: max(1, self.population // 2)]
            offspring: list[dict[str, Any]] = []
            for parent, _ in survivors:
                for _ in range(self.mutations_per_parent):
                    child = self._mutate(target, dataset, state, parent)
                    if child is not None:
                        offspring.append(child)
            if not offspring:
                break
            # Survivors keep their scores; only children not yet measured in
            # this run spend budget.
            scored = list(survivors)
            scored.extend(self._score_population(target, dataset, state, offspring, seen=seen))
            scored.sort(key=lambda cs: cs[1], reverse=True)

    def _mutate(
        self,
        target: OptimizableAgent,
        dataset: GoldenDataset,
        state: _SearchState,
        parent: dict[str, Any],
    ) -> dict[str, Any] | None:
        params = target.search_space.optimizable()
        param = self._rng.choice(params)
        supporting = proposers_for(param, self.proposers)
        if not supporting:
            return None
        proposer = self._rng.choice(supporting)
        ctx = self._proposal_context(target, param, dataset, state.best_report, 2)
        try:
            candidates = proposer.propose(ctx)
        except Exception:
            return None
        self._register_demo_rows(param, ctx)
        if not candidates:
            return None
        child = dict(parent)
        child[param.name] = self._rng.choice(candidates)
        return child

    def _score_population(
        self,
        target: OptimizableAgent,
        dataset: GoldenDataset,
        state: _SearchState,
        population: list[dict[str, Any]],
        *,
        seen: set[str] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Evaluate the not-yet-seen members of ``population``, best first.

        ``seen`` is shared across generations by :meth:`_search`, so a child
        identical to an earlier configuration is skipped rather than paid for.
        """
        scored: list[tuple[dict[str, Any], float]] = []
        seen = set() if seen is None else seen
        for config in population:
            if len(state.history) >= self.max_evals:
                break
            key = _config_key(config)
            if key in seen:
                continue
            seen.add(key)
            _, rank = self._trial(target, state, config, dataset)
            if not state.history[-1].complete:
                # `_record` only guards the global best. Survivors and parents
                # are chosen from *this* list, so an incomplete candidate left
                # in it would breed from a score measured over an easier subset
                # -- throttling would steer the search even while being barred
                # from winning it.
                continue
            scored.append((config, rank))
        scored.sort(key=lambda cs: cs[1], reverse=True)
        return scored


#: Stage attributes a pipeline overrides for the duration of one stage.
_STAGE_ATTRS = (
    "max_evals",
    "_eval_cache",
    "cache_evaluations",
    "_nested",
    "_nondeterministic",
    "_repeats_eff",
    "_demo_rows",
)


class PipelineOptimizer(Optimizer):
    """Run several optimizers in sequence, threading the best config forward.

    Each stage starts from the previous stage's best configuration (applied to
    the live agent), so you can, e.g., first bootstrap few-shot examples, then
    rewrite prompts with the LLM judge, then tune temperature.

    The pipeline owns the run-level decisions: the automatic hold-out split,
    the non-determinism check (stages inherit the verdict and the repeat
    count), and the one final verdict on the cumulative configuration. Budget
    a stage leaves unused -- a few-shot stage on an agent with no few-shot
    knob uses none of it -- is carried forward to the stages after it, always
    within the pipeline's own ``max_evals``.
    """

    strategy_name = "pipeline"

    def __init__(self, harness: EvaluationHarness, stages: list[Optimizer], **kw: Any):
        super().__init__(harness, **kw)
        if not stages:
            raise ValueError("PipelineOptimizer requires at least one stage")
        self.stages = stages

    def _tunable_names(self, target: OptimizableAgent) -> list[str]:
        names: list[str] = []
        for stage in self.stages:
            tunable = getattr(stage, "_tunable_names", None)
            if not callable(tunable):
                continue
            try:
                found = tunable(target)
            except Exception:
                continue
            names.extend(n for n in found if n not in names)
        return names

    def optimize(
        self,
        agent: Any,
        dataset: GoldenDataset,
        *,
        val_dataset: GoldenDataset | None = None,
        runner: Any = None,
        components: dict[str, Any] | None = None,
        parameters: list[Parameter] | None = None,
    ) -> OptimizationResult:
        target = wrap(agent, runner=runner, components=components, parameters=parameters)
        if not dataset:
            raise ValueError("Cannot optimize against an empty dataset")
        self._rng = random.Random(self.seed)
        self._demo_rows = {}
        train, holdout, auto = self._resolve_holdout(dataset, val_dataset)

        # One evaluation cache for the whole pipeline. Injected into every
        # stage, so a stage's baseline -- the previous stage's winner, already
        # measured -- comes back as a cache hit instead of a full-dataset pass
        # (the default four-stage pipeline otherwise re-runs the baseline five
        # times). Stages that were built with ``cache_evaluations=False`` see
        # the injected dict but never read or write it.
        owns_cache = self.cache_evaluations and self._eval_cache is None
        if owns_cache:
            self._eval_cache = {}
        shared_cache = self._eval_cache
        # This method overrides Optimizer.optimize() entirely rather than
        # calling super(), so the base class's verbose start/trial/finish
        # logging never runs for the *pipeline* itself (each stage still logs
        # its own trials against its own budget, via its own optimize() call).
        # Without a line marking stage transitions, a long default pipeline
        # -- bootstrap, then prompts, then models, then tools -- reads as one
        # continuous, unexplained pause between each stage's own output.
        self._run_started_at = time.perf_counter()
        try:
            baseline_snapshot = target.snapshot()
            baseline_report = self._evaluate_baseline(target, train)
            baseline_rows = self._detect_nondeterminism(
                target, train, baseline_report, row_scores(baseline_report)
            )
            baseline_score = baseline_report.score
            if self.verbose:
                logger.info(
                    "[%s] baseline score=%.4f (%d stage(s), search budget: up to %d evals)",
                    self.strategy_name,
                    baseline_score,
                    len(self.stages),
                    self.max_evals,
                )

            combined_history: list[Trial] = []
            combined_recommendations: list[str] = []
            best_config: dict[str, Any] = {}
            best_score = baseline_score
            best_report: EvaluationReport | None = None
            carried = 0

            for stage_index, stage in enumerate(self.stages, start=1):
                # Enforce a shared budget so the pipeline never exceeds the documented
                # max_evals hard cap, regardless of each stage's own max_evals.
                remaining = self.max_evals - len(combined_history)
                if remaining <= 0:
                    break
                if self.verbose:
                    logger.info(
                        "[%s] stage %d/%d (%s) starting: %d/%d evals used so far "
                        "(%.1fs elapsed)",
                        self.strategy_name,
                        stage_index,
                        len(self.stages),
                        stage.strategy_name,
                        len(combined_history),
                        self.max_evals,
                        time.perf_counter() - self._run_started_at,
                    )
                # ``getattr`` because a stage may be a minimal Optimizer
                # subclass that skipped ``__init__`` -- the same pattern the
                # cache helpers themselves tolerate.
                saved = {name: getattr(stage, name, _MISSING) for name in _STAGE_ATTRS}
                allowance = min(stage.max_evals + carried, remaining)
                stage.max_evals = allowance
                stage._eval_cache = shared_cache
                # A pipeline-level opt-out must reach the stages: with the
                # shared cache merely absent, a stage's own default flag would
                # quietly build a private per-stage cache and repeated
                # configurations would still be reused -- exactly what
                # ``cache_evaluations=False`` promises to prevent. A stage's
                # own opt-out inside a caching pipeline is still respected.
                own_flag = saved["cache_evaluations"]
                stage.cache_evaluations = (
                    own_flag is not _MISSING and bool(own_flag) and self.cache_evaluations
                )
                stage._nested = True
                stage._nondeterministic = self._nondeterministic
                stage._repeats_eff = self._repeats()
                stage._demo_rows = self._demo_rows
                try:
                    stage_result = stage.optimize(target, train)  # applies stage best in place
                finally:
                    for name, value in saved.items():
                        if value is _MISSING:
                            if name in vars(stage):
                                delattr(stage, name)
                        else:
                            setattr(stage, name, value)
                combined_history.extend(stage_result.history)
                combined_recommendations.extend(stage_result.recommendations)
                carried = max(0, allowance - len(stage_result.history))
                # The live target now carries this stage's best; accumulate the
                # diff. A stage only reports a config it accepted on the
                # significance rule, relative to the cumulative state it began
                # from.
                if stage_result.best_config:
                    best_score = stage_result.best_score
                    best_config.update(stage_result.best_config)
                    best_report = stage_result.best_report or best_report

            # Live agent already has the cumulative best applied by the final stage
            # that improved; ensure consistency by re-applying the accumulated config.
            target.restore(baseline_snapshot)
            if best_config:
                target.apply(best_config)

            pipeline_state = _SearchState(
                best_config=best_config,
                best_score=best_score,
                best_report=best_report,
                baseline_snapshot=baseline_snapshot,
                history=combined_history,
                best_rows=row_scores(best_report) if best_config else baseline_rows,
            )
            gate = self._final_verdict(
                target, train, holdout, auto, baseline_snapshot, baseline_rows, pipeline_state
            )
            if gate.discarded:
                pipeline_state.best_config = {}
                pipeline_state.best_score = baseline_score
                pipeline_state.best_report = baseline_report
            combined_recommendations.extend(gate.notes)
            untuned = self._untuned(
                self._tunable_names(target), combined_history, baseline_snapshot
            )
            if untuned:
                combined_recommendations.append(self._warn_untuned(untuned))

            # Optionally run the pipeline-level tool suggestion on the cumulative best,
            # then dedupe so a repeated suggestion from several stages appears once.
            combined_recommendations.extend(self._suggest_tools(target, pipeline_state))
            result = self._build_result(
                pipeline_state,
                baseline_score,
                baseline_snapshot,
                gate,
                _dedup(combined_recommendations),
                untuned,
            )
            if self.verbose:
                logger.info("[%s] %r", self.strategy_name, result)
            return result
        finally:
            if owns_cache:
                self._eval_cache = None

    def _search(
        self, target: OptimizableAgent, dataset: GoldenDataset, state: _SearchState
    ) -> None:  # pragma: no cover
        raise NotImplementedError("PipelineOptimizer overrides optimize() directly")


def make_default_optimizer(
    harness: EvaluationHarness,
    *,
    judge: Any = None,
    max_evals: int = 60,
    seed: int | None = 0,
    verbose: bool = False,
    min_improvement: float | None = None,
    suggest_tools: bool | None = None,
    proposer_judge: Any = None,
    alpha: float | None = None,
    repeats: int | None = None,
    holdout_fraction: float | None = None,
    allow_small_sample: bool | None = None,
) -> PipelineOptimizer:
    """Build a sensible "do all the optimizations" pipeline.

    Stages: bootstrap few-shot examples, then coordinate-ascent over prompts
    (LLM-judge-driven when a ``judge`` is supplied), then grid over models /
    hyperparameters / routing, then coordinate-ascent over tools/skills
    (drop-one ablation, plus judge-driven new-tool *suggestions* when a judge is
    supplied). Budget is split across the four stages, and whatever a stage
    leaves unused (an agent with no few-shot knob, say) goes to the stages
    after it. The grid stage sweeps knob by knob when the full product does
    not fit its share, so no model, hyperparameter or routing knob is silently
    left untried.

    ``suggest_tools`` controls whether the pipeline records advisory new-tool/skill
    recommendations: ``None`` (default) auto-enables it whenever a ``judge`` is
    supplied; pass ``True``/``False`` to force it on or off.

    ``proposer_judge`` is a separate judge used for critique and prompt
    rewriting, so the model that writes a candidate is not the one that scores
    it (see :class:`Optimizer`).

    ``alpha``, ``repeats``, ``holdout_fraction`` and ``allow_small_sample`` are
    passed to every stage and the pipeline (see :class:`Optimizer`); ``None``
    keeps the library default.
    """
    _check_budget("max_evals", max_evals)
    per_stage = max(5, max_evals // 4)
    common: dict[str, Any] = {"seed": seed, "judge": judge, "verbose": verbose}
    if proposer_judge is not None:
        # A separate judge for critique + rewriting, so the model that writes a
        # candidate prompt is not the one that marks it.
        common["proposer_judge"] = proposer_judge
    if min_improvement is not None:
        common["min_improvement"] = min_improvement
    for name, value in (
        ("alpha", alpha),
        ("repeats", repeats),
        ("holdout_fraction", holdout_fraction),
        ("allow_small_sample", allow_small_sample),
    ):
        if value is not None:
            common[name] = value
    suggest = (judge is not None) if suggest_tools is None else suggest_tools
    stages: list[Optimizer] = [
        BootstrapFewShotOptimizer(harness, max_evals=per_stage, **common),
        CoordinateAscentOptimizer(
            harness,
            max_evals=per_stage,
            kinds=(ParameterKind.PROMPT,),
            **common,
        ),
        # `max_configs` is the overall budget: the stage's real limit is the
        # allowance the pipeline hands it (its share plus what earlier stages
        # left), and past that the grid sweeps knob by knob rather than
        # truncating.
        GridSearchOptimizer(harness, max_evals=per_stage, max_configs=max_evals, **common),
        CoordinateAscentOptimizer(
            harness,
            max_evals=per_stage,
            kinds=(ParameterKind.TOOL, ParameterKind.SKILL),
            suggest_tools=suggest,
            **common,
        ),
    ]
    return PipelineOptimizer(harness, stages, max_evals=max_evals, suggest_tools=suggest, **common)


def _check_budget(name: str, value: Any) -> None:
    """Refuse a budget that is not a positive integer, before any work starts.

    ``max_evals=-1`` used to run as if unbounded for a grid, and ``0`` as "do
    nothing and report the baseline as the result" -- neither is what a caller
    who typed it could have wanted, and both looked like a completed run.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _compose(snapshot: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """The live state ``config`` produces on top of ``snapshot``."""
    if not config:
        return snapshot
    return {**snapshot, **config}


def _same(a: Any, b: Any) -> bool:
    """Equality that never raises (live objects may compare badly)."""
    try:
        return bool(a == b)
    except Exception:
        return a is b


def _value_key(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}@{id(value)}>"


def _config_key(config: dict[str, Any]) -> str:
    return repr(sorted(config.items(), key=lambda kv: kv[0]))


def _spread(n: int, k: int) -> list[int]:
    """``k`` indices spread evenly over ``range(n)`` (all of them when ``k >= n``)."""
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    picked = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return picked


def _fmt_p(p_value: float | None) -> str:
    return "n/a" if p_value is None else f"{p_value:.4g}"


def _mean_score(reports: Sequence[EvaluationReport]) -> float:
    return sum(r.score for r in reports) / len(reports) if reports else 0.0


def _mean_aggregate(reports: Sequence[EvaluationReport]) -> dict[str, float]:
    if len(reports) == 1:
        return dict(reports[0].aggregate)
    names = {name for r in reports for name in r.aggregate}
    return {
        name: sum(r.aggregate.get(name, 0.0) for r in reports) / len(reports)
        for name in sorted(names)
    }


def _dedup(values: list[Any]) -> list[Any]:
    """Order-preserving de-duplication tolerant of unhashable values."""
    out: list[Any] = []
    seen_hashable: set[Any] = set()
    for v in values:
        try:
            if v in seen_hashable:
                continue
            seen_hashable.add(v)
        except TypeError:
            if v in out:
                continue
        out.append(v)
    return out


def _is_exportable(value: Any) -> bool:
    """Can ``value`` survive a YAML/JSON round trip **as itself**?

    Note the tuple exclusion. Both encoders write a tuple as a sequence and read
    it back as a ``list``, so exporting one would change the winning value's type
    on reload -- and a setter or framework field expecting a tuple would then
    receive a list. The invariant here is that everything in the config body
    applies cleanly, so a tuple is described rather than exported.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, tuple):
        return False
    if isinstance(value, list):
        return all(_is_exportable(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_exportable(v) for k, v in value.items())
    return False


def _describe(value: Any) -> str:
    """A stable, readable identifier for a live object, for the review diff."""
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_describe(v) for v in value) + "]"
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if name:
        return f"{module}:{name}" if module else str(name)
    return f"<{type(value).__name__}>"


def load_tuned_config(path: str | Path) -> dict[str, Any]:
    """Load a file written by :meth:`OptimizationResult.to_config` as a flat config.

    Flattens ``{component: {knob: value}}`` back to the ``"<component>.<knob>"``
    keys :meth:`OptimizableAgent.apply` expects, closing the loop::

        target.apply(load_tuned_config("specialists/.config/tuned.yaml"))

    A scalar at the top level is taken as an already-flat entry, so a hand-edited
    file mixing both shapes still loads. JSON and YAML both work -- YAML is a
    superset, so one parser reads either -- and the ``_provenance`` block written
    into a JSON export is skipped rather than applied as a component.

    Raises:
        ValueError: If the file does not contain a mapping.
    """
    import yaml

    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} does not contain a mapping of components to parameters")
    flat: dict[str, Any] = {}
    for component, values in loaded.items():
        if component == _PROVENANCE_KEY:
            continue  # metadata about the run, not a tunable component
        if isinstance(values, dict):
            for knob, value in values.items():
                flat[f"{component}.{knob}"] = value
        else:
            flat[component] = values
    return flat


__all__ = [
    "Trial",
    "OptimizationResult",
    "Optimizer",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "CoordinateAscentOptimizer",
    "BootstrapFewShotOptimizer",
    "EvolutionaryOptimizer",
    "PipelineOptimizer",
    "make_default_optimizer",
    "load_tuned_config",
]
