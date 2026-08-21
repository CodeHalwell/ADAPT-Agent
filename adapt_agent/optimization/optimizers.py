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
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapt_agent.optimization.dataset import GoldenDataset
from adapt_agent.optimization.evaluation import EvaluationHarness, EvaluationReport
from adapt_agent.optimization.parameters import Parameter, ParameterKind
from adapt_agent.optimization.proposers import (
    ProposalContext,
    Proposer,
    default_proposers,
    proposers_for,
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

    @property
    def improved(self) -> bool:
        return self.best_score > self.baseline_score

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
              model: gpt-4o
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
                        f"validation={provenance['validation_score']} "
                        f"over {provenance['n_evals']} evals.\n"
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
            "n_evals": self.n_evals,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "improved": self.improved,
            "baseline_score": self.baseline_score,
            "best_score": self.best_score,
            "improvement": self.improvement,
            "validation_score": self.validation_score,
            "n_evals": self.n_evals,
            "best_config": self.best_config,
            "recommendations": list(self.recommendations),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"OptimizationResult(baseline={self.baseline_score:.3f}, "
            f"best={self.best_score:.3f}, improvement={self.improvement:+.3f}, "
            f"evals={self.n_evals})"
        )


class Optimizer:
    """Base class with the shared optimize loop and evaluation plumbing.

    Args:
        harness: The :class:`EvaluationHarness` used to score configurations.
        max_evals: Hard cap on candidate evaluations (budget guard).
        seed: RNG seed for reproducible search.
        min_improvement: Minimum primary-metric gain required to accept a
            candidate over the current best. Defaults to ``1e-3``: small enough to
            keep real gains, but large enough to avoid *chasing judge noise* -- an
            LLM judge's score wobbles by far more than ``1e-9`` between runs, so a
            near-zero threshold would "accept" candidates that only won by jitter.
        judge: Optional LLM judge made available to proposers.
        suggest_tools: When True *and* a ``judge`` is set, after the search the
            optimizer asks the judge to propose NEW tools/skills for components
            that own a TOOL/SKILL parameter and records them as advisory
            ``recommendations`` (never applied automatically). Off by default;
            :func:`make_default_optimizer` turns it on when a judge is present.
        verbose: Emit per-trial logging at INFO level.
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
        verbose: bool = False,
    ):
        self.harness = harness
        self.max_evals = max_evals
        self.seed = seed
        self.min_improvement = min_improvement
        self.judge = judge
        self.suggest_tools = suggest_tools
        self.verbose = verbose
        self._rng = random.Random(seed)
        # Set per-run by the search strategy before evaluating candidates; the
        # baseline values every candidate config is composed on top of.
        self._baseline_snapshot: dict[str, Any] = {}
        # Shared advisory-suggestion sink, replaced per ``optimize`` call.
        self._recommendations: list[str] = []

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

        The best configuration is applied to the live agent before returning.
        Extra kwargs (``runner``/``components``/``parameters``) are forwarded to
        :func:`~adapt_agent.optimization.target.wrap` when ``agent`` is not
        already an :class:`OptimizableAgent`.
        """
        target = wrap(agent, runner=runner, components=components, parameters=parameters)
        if not dataset:
            raise ValueError("Cannot optimize against an empty dataset")

        baseline_snapshot = target.snapshot()
        baseline_report = self.harness.evaluate(target, dataset)
        baseline_score = baseline_report.score
        if self.verbose:
            logger.info("[%s] baseline score=%.4f", self.strategy_name, baseline_score)

        # One shared sink threaded through every ProposalContext and read back
        # after the search (proposers may append advisory suggestions to it).
        self._recommendations = []

        state = _SearchState(
            best_config={},
            best_score=baseline_score,
            best_report=baseline_report,
            baseline_snapshot=baseline_snapshot,
            history=[],
        )
        self._search(target, dataset, state)

        # Apply the winning configuration permanently to the live agent.
        target.restore(baseline_snapshot)
        if state.best_config:
            target.apply(state.best_config)

        validation_score: float | None = None
        if val_dataset:
            validation_score = self.harness.evaluate(target, val_dataset).score

        recommendations = list(self._recommendations)
        recommendations.extend(self._suggest_tools(target, state))

        result = OptimizationResult(
            best_config=state.best_config,
            best_score=state.best_score,
            baseline_score=baseline_score,
            baseline_config=baseline_snapshot,
            history=state.history,
            best_report=state.best_report,
            validation_score=validation_score,
            recommendations=recommendations,
        )
        if self.verbose:
            logger.info("[%s] %r", self.strategy_name, result)
        return result

    # -- strategy hook ---------------------------------------------------------

    def _search(
        self, target: OptimizableAgent, dataset: GoldenDataset, state: _SearchState
    ) -> None:
        """Populate ``state`` with trials. Overridden by concrete optimizers."""
        raise NotImplementedError

    # -- shared helpers --------------------------------------------------------

    def _eval_config(
        self, target: OptimizableAgent, config: dict[str, Any], dataset: GoldenDataset
    ) -> EvaluationReport:
        """Evaluate a *full* config relative to the baseline snapshot.

        Restores the baseline first so configs compose predictably regardless of
        whatever the previous trial left applied.
        """
        target.restore(self._current_baseline)
        if config:
            target.apply(config)
        report = self.harness.evaluate(target, dataset)
        return report

    def _record(
        self,
        state: _SearchState,
        config: dict[str, Any],
        report: EvaluationReport,
    ) -> bool:
        """Record a trial and update the best-so-far. Returns True if accepted.

        An *incomplete* trial can never be accepted, however well it scored.
        Transient failures drop rows from the score, so an incomplete trial's
        mean is taken over a different subset than its rivals' -- a candidate
        that answers one easy row and gets throttled on a hard one scores 1.0
        and would otherwise beat a fully-evaluated 0.9. Excluding throttled rows
        stops throttling from *penalising* a candidate; refusing to crown a
        partial trial stops it from *rewarding* one. Both are needed, or the
        search still selects for luck, just in the opposite direction.
        """
        complete = report.is_complete
        accepted = complete and report.score > state.best_score + self.min_improvement
        trial = Trial(
            config=dict(config),
            score=report.score,
            strategy=self.strategy_name,
            accepted=accepted,
            metrics=dict(report.aggregate),
            complete=complete,
            n_transient_errors=report.n_transient_errors,
        )
        if not complete:
            logger.warning(
                "[%s] trial #%d scored %.4f over %d of %d rows (%d transient failures); "
                "not eligible to win -- its score is not comparable with a complete trial's",
                self.strategy_name,
                len(state.history) + 1,
                report.score,
                report.n - report.n_transient_errors,
                report.n,
                report.n_transient_errors,
            )
        state.history.append(trial)
        if accepted:
            state.best_config = dict(config)
            state.best_score = report.score
            state.best_report = report
        if self.verbose:
            logger.info(
                "[%s] trial #%d score=%.4f %s",
                self.strategy_name,
                len(state.history),
                report.score,
                "ACCEPT" if accepted else "",
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
            judge=self.judge,
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


# A tiny helper so strategies can set the baseline snapshot on the optimizer.
def _bind_baseline(opt: Optimizer, snapshot: dict[str, Any]) -> None:
    opt._baseline_snapshot = snapshot


class GridSearchOptimizer(Optimizer):
    """Exhaustively evaluate the Cartesian product of discrete candidate sets."""

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
        self.numeric_points = numeric_points
        self.max_configs = max_configs

    def _search(
        self, target: OptimizableAgent, dataset: GoldenDataset, state: _SearchState
    ) -> None:
        _bind_baseline(self, state.baseline_snapshot)
        configs = target.search_space.grid(
            numeric_points=self.numeric_points, max_configs=self.max_configs
        )
        for config in configs[: self.max_evals]:
            if not config:
                continue
            report = self._eval_config(target, config, dataset)
            self._record(state, config, report)


class RandomSearchOptimizer(Optimizer):
    """Sample random whole-space configurations and keep the best."""

    strategy_name = "random"

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
            report = self._eval_config(target, config, dataset)
            self._record(state, config, report)


class CoordinateAscentOptimizer(Optimizer):
    """Greedy per-parameter improvement driven by proposers.

    For each optimizable parameter, ask the supporting proposers for candidate
    values, evaluate each (holding the current best for all other parameters),
    and keep any improvement. Repeat for ``rounds`` passes or until a pass yields
    no improvement. This is the flagship strategy for prompt and few-shot
    optimization and is where the LLM judge improves instructions.

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
        self.proposers = proposers if proposers is not None else default_proposers(self.judge)
        self.rounds = rounds
        self.candidates_per_param = candidates_per_param
        self.kinds = kinds

    def _search(
        self, target: OptimizableAgent, dataset: GoldenDataset, state: _SearchState
    ) -> None:
        _bind_baseline(self, state.baseline_snapshot)
        params = [
            p
            for p in target.search_space.optimizable()
            if self.kinds is None or p.kind in self.kinds
        ]
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
                for value in _dedup(candidates):
                    if len(state.history) >= self.max_evals:
                        return
                    # Compose on top of the current best configuration.
                    config = dict(state.best_config)
                    config[param.name] = value
                    report = self._eval_config(target, config, dataset)
                    if self._record(state, config, report):
                        improved_this_round = True
            if not improved_this_round:
                if self.verbose:
                    logger.info("[%s] converged after round %d", self.strategy_name, round_idx + 1)
                break


class BootstrapFewShotOptimizer(CoordinateAscentOptimizer):
    """Coordinate ascent restricted to few-shot parameters (DSPy-style bootstrap)."""

    strategy_name = "bootstrap_few_shot"

    def __init__(self, harness: EvaluationHarness, **kw: Any):
        kw.setdefault("kinds", (ParameterKind.FEW_SHOT,))
        super().__init__(harness, **kw)


class EvolutionaryOptimizer(Optimizer):
    """Population-based search: mutate configs with proposers, select the best.

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
        self.proposers = proposers if proposers is not None else default_proposers(self.judge)
        self.population = population
        self.generations = generations
        self.mutations_per_parent = mutations_per_parent

    def _search(
        self, target: OptimizableAgent, dataset: GoldenDataset, state: _SearchState
    ) -> None:
        _bind_baseline(self, state.baseline_snapshot)
        params = target.search_space.optimizable()
        if not params:
            return
        # Seed population with the baseline (empty diff) plus random samples.
        population: list[dict[str, Any]] = [{}]
        for _ in range(self.population - 1):
            population.append(target.search_space.sample_config(self._rng))

        scored = self._score_population(target, dataset, state, population)
        for _ in range(self.generations):
            if len(state.history) >= self.max_evals:
                break
            survivors = [cfg for cfg, _ in scored[: max(1, self.population // 2)]]
            offspring: list[dict[str, Any]] = []
            for parent in survivors:
                for _ in range(self.mutations_per_parent):
                    child = self._mutate(target, dataset, state, parent)
                    if child is not None:
                        offspring.append(child)
            if not offspring:
                break
            scored = self._score_population(target, dataset, state, survivors + offspring)

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
    ) -> list[tuple[dict[str, Any], float]]:
        scored: list[tuple[dict[str, Any], float]] = []
        seen: set[str] = set()
        for config in population:
            if len(state.history) >= self.max_evals:
                break
            key = repr(sorted(config.items()))
            if key in seen:
                continue
            seen.add(key)
            report = self._eval_config(target, config, dataset)
            self._record(state, config, report)
            scored.append((config, report.score))
        scored.sort(key=lambda cs: cs[1], reverse=True)
        return scored


class PipelineOptimizer(Optimizer):
    """Run several optimizers in sequence, threading the best config forward.

    Each stage starts from the previous stage's best configuration (applied to
    the live agent), so you can, e.g., first bootstrap few-shot examples, then
    rewrite prompts with the LLM judge, then tune temperature.
    """

    strategy_name = "pipeline"

    def __init__(self, harness: EvaluationHarness, stages: list[Optimizer], **kw: Any):
        super().__init__(harness, **kw)
        if not stages:
            raise ValueError("PipelineOptimizer requires at least one stage")
        self.stages = stages

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
        baseline_snapshot = target.snapshot()
        baseline_score = self.harness.evaluate(target, dataset).score

        combined_history: list[Trial] = []
        combined_recommendations: list[str] = []
        best_config: dict[str, Any] = {}
        best_score = baseline_score
        best_report: EvaluationReport | None = None

        for stage in self.stages:
            # Enforce a shared budget so the pipeline never exceeds the documented
            # max_evals hard cap, regardless of each stage's own max_evals.
            remaining = self.max_evals - len(combined_history)
            if remaining <= 0:
                break
            original_max = stage.max_evals
            stage.max_evals = min(original_max, remaining)
            try:
                stage_result = stage.optimize(target, dataset)  # applies stage best in place
            finally:
                stage.max_evals = original_max
            combined_history.extend(stage_result.history)
            combined_recommendations.extend(stage_result.recommendations)
            # The live target now carries this stage's best; accumulate the diff.
            if stage_result.best_score >= best_score:
                best_score = stage_result.best_score
                best_config.update(stage_result.best_config)
                best_report = stage_result.best_report or best_report

        # Live agent already has the cumulative best applied by the final stage
        # that improved; ensure consistency by re-applying the accumulated config.
        target.restore(baseline_snapshot)
        if best_config:
            target.apply(best_config)

        validation_score = self.harness.evaluate(target, val_dataset).score if val_dataset else None

        # Optionally run the pipeline-level tool suggestion on the cumulative best,
        # then dedupe so a repeated suggestion from several stages appears once.
        pipeline_state = _SearchState(
            best_config=best_config,
            best_score=best_score,
            best_report=best_report,
            baseline_snapshot=baseline_snapshot,
            history=combined_history,
        )
        combined_recommendations.extend(self._suggest_tools(target, pipeline_state))
        return OptimizationResult(
            best_config=best_config,
            best_score=best_score,
            baseline_score=baseline_score,
            baseline_config=baseline_snapshot,
            history=combined_history,
            best_report=best_report,
            validation_score=validation_score,
            recommendations=_dedup(combined_recommendations),
        )

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
) -> PipelineOptimizer:
    """Build a sensible "do all the optimizations" pipeline.

    Stages: bootstrap few-shot examples, then coordinate-ascent over prompts
    (LLM-judge-driven when a ``judge`` is supplied), then grid over models /
    hyperparameters / routing, then coordinate-ascent over tools/skills
    (drop-one ablation, plus judge-driven new-tool *suggestions* when a judge is
    supplied). Budget is split across the four stages.

    ``suggest_tools`` controls whether the pipeline records advisory new-tool/skill
    recommendations: ``None`` (default) auto-enables it whenever a ``judge`` is
    supplied; pass ``True``/``False`` to force it on or off.
    """
    per_stage = max(5, max_evals // 4)
    common: dict[str, Any] = {"seed": seed, "judge": judge, "verbose": verbose}
    if min_improvement is not None:
        common["min_improvement"] = min_improvement
    suggest = (judge is not None) if suggest_tools is None else suggest_tools
    stages: list[Optimizer] = [
        BootstrapFewShotOptimizer(harness, max_evals=per_stage, **common),
        CoordinateAscentOptimizer(
            harness,
            max_evals=per_stage,
            kinds=(ParameterKind.PROMPT,),
            **common,
        ),
        GridSearchOptimizer(harness, max_evals=per_stage, max_configs=per_stage, **common),
        CoordinateAscentOptimizer(
            harness,
            max_evals=per_stage,
            kinds=(ParameterKind.TOOL, ParameterKind.SKILL),
            suggest_tools=suggest,
            **common,
        ),
    ]
    return PipelineOptimizer(harness, stages, max_evals=max_evals, suggest_tools=suggest, **common)


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
