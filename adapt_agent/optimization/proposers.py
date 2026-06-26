"""Candidate proposers: how new parameter values are generated.

An optimizer never invents values itself; it asks *proposers* for candidate
values for a given :class:`~adapt_agent.optimization.parameters.Parameter`, then
evaluates each one. This keeps the optimizer's control flow (search, accept,
loop) separate from the strategy for generating ideas, and lets the
LLM-as-judge participate in *improvement* (not just scoring) by rewriting prompts
from observed failures.

Proposers fall into two camps:

* **Deterministic / offline** -- :class:`CandidateProposer`,
  :class:`NumericProposer`, :class:`PromptMutationProposer`,
  :class:`FewShotProposer`. No network, fully reproducible under a seed.
* **LLM-driven** -- :class:`LLMProposer` uses an
  :class:`~adapt_agent.optimization.judge.LLMJudge` to critique failures and
  rewrite an instruction. The judge wraps a user-supplied completion function,
  so even this path adds no hard dependency.

All proposers implement :meth:`Proposer.supports` and :meth:`Proposer.propose`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from adapt_agent.optimization.dataset import GoldenDataset
from adapt_agent.optimization.parameters import Parameter, ParameterKind

if TYPE_CHECKING:  # avoid import cycles at runtime
    from adapt_agent.optimization.evaluation import EvaluationReport
    from adapt_agent.optimization.judge import LLMJudge
    from adapt_agent.optimization.target import OptimizableAgent


@dataclass
class ProposalContext:
    """Everything a proposer may need to generate candidates.

    Args:
        parameter: The parameter to propose new values for.
        agent: The optimization target (for component lookups).
        dataset: The dataset being optimized against (train split).
        report: The most recent evaluation report (baseline / current best),
            the source of failures used to drive improvement. May be ``None``.
        judge: An optional LLM judge for critique-driven proposals.
        rng: Seeded RNG for reproducible sampling/selection.
        n: Desired number of candidates (a hint, not a hard guarantee).
    """

    parameter: Parameter
    agent: OptimizableAgent
    dataset: GoldenDataset
    report: EvaluationReport | None = None
    judge: LLMJudge | None = None
    rng: random.Random = field(default_factory=random.Random)
    n: int = 4


class Proposer:
    """Base class for candidate proposers."""

    name: str = "proposer"

    def supports(self, parameter: Parameter) -> bool:
        """Return ``True`` if this proposer can handle ``parameter``."""
        raise NotImplementedError

    def propose(self, ctx: ProposalContext) -> list[Any]:
        """Return candidate values for ``ctx.parameter`` (excluding the current)."""
        raise NotImplementedError


class CandidateProposer(Proposer):
    """Yields a parameter's explicit candidates (or a numeric grid)."""

    name = "candidates"

    def supports(self, parameter: Parameter) -> bool:
        return bool(parameter.candidates) or parameter.bounds is not None

    def propose(self, ctx: ProposalContext) -> list[Any]:
        current = ctx.parameter.read()
        options = ctx.parameter.enumerate_candidates()
        return [o for o in options if o != current]


class NumericProposer(Proposer):
    """Samples numeric values within bounds, plus local perturbations of current."""

    name = "numeric"

    def supports(self, parameter: Parameter) -> bool:
        return parameter.bounds is not None

    def propose(self, ctx: ProposalContext) -> list[Any]:
        param = ctx.parameter
        assert param.bounds is not None
        low, high = param.bounds
        current = param.read()
        out: list[Any] = []
        # Local perturbations around the current value (coordinate-ascent style).
        if isinstance(current, (int, float)):
            span = (high - low) or 1.0
            for delta in (-0.2, -0.1, 0.1, 0.2):
                cand = param._coerce_numeric(min(high, max(low, current + delta * span)))
                if cand != current and cand not in out:
                    out.append(cand)
        # Random exploration to escape local optima.
        while len(out) < ctx.n:
            cand = param.sample(ctx.rng)
            if cand != current and cand not in out:
                out.append(cand)
            else:
                break
        return out[: ctx.n]


#: Generic, task-agnostic instruction reinforcements appended to a base prompt.
_PROMPT_DIRECTIVES = (
    "Think step by step and reason carefully before giving your final answer.",
    "Be concise and precise. Do not include irrelevant information.",
    "Ground every claim in the provided context; if you are unsure, say so "
    "rather than guessing.",
    "Follow the requested output format exactly.",
    "Double-check your answer for correctness before responding.",
)


class PromptMutationProposer(Proposer):
    """Deterministic prompt variants by appending generic best-practice directives.

    A dependency-free baseline for prompt optimization: it never needs an LLM and
    is fully reproducible, yet often yields measurable gains. Pair it with
    :class:`LLMProposer` for failure-targeted rewrites.
    """

    name = "prompt_mutation"

    def __init__(self, directives: tuple[str, ...] = _PROMPT_DIRECTIVES):
        self.directives = directives

    def supports(self, parameter: Parameter) -> bool:
        return parameter.kind is ParameterKind.PROMPT

    def propose(self, ctx: ProposalContext) -> list[Any]:
        base = ctx.parameter.read()
        if not isinstance(base, str):
            return []
        base = base.rstrip()
        out: list[Any] = []
        for directive in self.directives:
            candidate = f"{base}\n\n{directive}" if base else directive
            if candidate != ctx.parameter.read():
                out.append(candidate)
        return out[: ctx.n]


class FewShotProposer(Proposer):
    """Bootstraps in-context example blocks from currently-correct dataset rows.

    For a :data:`~adapt_agent.optimization.parameters.ParameterKind.FEW_SHOT`
    parameter, it selects up to ``shots`` examples the agent already answers well
    (per the latest report) and formats them into a demonstration block. Several
    blocks are proposed using different example subsets.
    """

    name = "few_shot"

    def __init__(self, shots: int = 3, *, variants: int = 3):
        self.shots = shots
        self.variants = variants

    def supports(self, parameter: Parameter) -> bool:
        return parameter.kind is ParameterKind.FEW_SHOT

    def propose(self, ctx: ProposalContext) -> list[Any]:
        pool = self._labeled_pool(ctx)
        if not pool:
            return []
        out: list[Any] = []
        for _ in range(self.variants):
            if len(pool) <= self.shots:
                chosen = list(pool)
            else:
                chosen = ctx.rng.sample(pool, self.shots)
            block = self._format(chosen)
            if block and block not in out:
                out.append(block)
        return out

    def _labeled_pool(self, ctx: ProposalContext) -> list[tuple[Any, Any]]:
        # Prefer examples the agent currently gets right (high-quality demos).
        correct_inputs: set[int] = set()
        if ctx.report is not None:
            primary = ctx.report.primary_metric
            for r in ctx.report.results:
                if r.error is None and r.scores.get(primary, 0.0) >= 1.0:
                    correct_inputs.add(r.index)
        pool: list[tuple[Any, Any]] = []
        for i, ex in enumerate(ctx.dataset):
            if ex.expected is None:
                continue
            if correct_inputs and i not in correct_inputs:
                continue
            pool.append((ex.inputs, ex.expected))
        # Fall back to all labeled rows if filtering left nothing.
        if not pool:
            pool = [(ex.inputs, ex.expected) for ex in ctx.dataset if ex.expected is not None]
        return pool

    @staticmethod
    def _format(pairs: list[tuple[Any, Any]]) -> str:
        lines = ["Here are some examples:"]
        for inp, exp in pairs:
            lines.append(f"\nInput: {inp}\nOutput: {exp}")
        return "\n".join(lines)


class LLMProposer(Proposer):
    """Rewrites prompts from judged failures using an LLM judge.

    For each proposal it collects the examples the current configuration still
    fails, asks the judge to critique a sample of them, then asks the judge to
    rewrite the instruction so those failures are avoided. This is the
    "LLM-as-judge improves the prompt" loop.

    Args:
        judge: The judge used for critique + rewrite. If ``None`` at propose
            time, falls back to the judge on the :class:`ProposalContext`.
        rewrites: How many rewritten variants to request.
        critique_samples: How many failures to critique per rewrite (bounded).
    """

    name = "llm"

    def __init__(
        self,
        judge: LLMJudge | None = None,
        *,
        rewrites: int = 2,
        critique_samples: int = 3,
    ):
        self.judge = judge
        self.rewrites = rewrites
        self.critique_samples = critique_samples

    def supports(self, parameter: Parameter) -> bool:
        return parameter.kind is ParameterKind.PROMPT

    def propose(self, ctx: ProposalContext) -> list[Any]:
        judge = self.judge or ctx.judge
        current = ctx.parameter.read()
        if judge is None or not isinstance(current, str) or not current:
            return []
        failures = self._collect_failures(ctx, judge)
        if not failures:
            return []
        criteria = self._criteria(ctx)
        out: list[Any] = []
        for _ in range(self.rewrites):
            rewritten = judge.improve_prompt(current, failures, criteria=criteria)
            if rewritten and rewritten not in out and rewritten != current:
                out.append(rewritten)
        return out

    def _collect_failures(self, ctx: ProposalContext, judge: LLMJudge) -> list[dict[str, Any]]:
        if ctx.report is None:
            return []
        failing = ctx.report.failures()
        if not failing:
            return []
        # Sample (seeded) across the whole failure pool rather than always taking
        # the first few, so later-failing examples are not systematically ignored.
        sample = ctx.rng.sample(failing, min(len(failing), self.critique_samples))
        records: list[dict[str, Any]] = []
        for r in sample:
            critique = ""
            if r.error is None:
                critique = judge.critique(r.inputs, r.output, r.expected)
            records.append(
                {
                    "input": r.inputs,
                    "output": r.output if r.error is None else f"<error: {r.error}>",
                    "expected": r.expected,
                    "critique": critique,
                }
            )
        return records

    @staticmethod
    def _criteria(ctx: ProposalContext) -> str | None:
        # Surface a shared criteria hint if every example agrees on one.
        criteria = {
            ex.metadata.get("criteria")
            for ex in ctx.dataset
            if isinstance(ex.metadata, dict) and ex.metadata.get("criteria")
        }
        return next(iter(criteria)) if len(criteria) == 1 else None


def default_proposers(judge: LLMJudge | None = None) -> list[Proposer]:
    """A sensible default proposer set covering every parameter kind.

    Includes the LLM proposer first (so prompt rewrites are tried ahead of
    generic mutations) only when a ``judge`` is available.
    """
    proposers: list[Proposer] = []
    if judge is not None:
        proposers.append(LLMProposer(judge))
    proposers.extend(
        [
            PromptMutationProposer(),
            FewShotProposer(),
            CandidateProposer(),
            NumericProposer(),
        ]
    )
    return proposers


def proposers_for(parameter: Parameter, proposers: list[Proposer]) -> list[Proposer]:
    """Filter ``proposers`` to those that support ``parameter``."""
    return [p for p in proposers if p.supports(parameter)]


__all__ = [
    "ProposalContext",
    "Proposer",
    "CandidateProposer",
    "NumericProposer",
    "PromptMutationProposer",
    "FewShotProposer",
    "LLMProposer",
    "default_proposers",
    "proposers_for",
]
