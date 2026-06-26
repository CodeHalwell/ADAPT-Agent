"""Tests for adapt_agent.optimization.proposers.

Offline and deterministic: any judge-driven path uses a plain Python function
(no network) wrapped in :class:`LLMJudge`.
"""

import random

import pytest

from adapt_agent.optimization.dataset import Example, GoldenDataset
from adapt_agent.optimization.evaluation import EvaluationReport, ExampleResult
from adapt_agent.optimization.judge import LLMJudge
from adapt_agent.optimization.parameters import Parameter, ParameterKind
from adapt_agent.optimization.proposers import (
    CandidateProposer,
    FewShotProposer,
    LLMProposer,
    LLMToolProposer,
    NumericProposer,
    PromptMutationProposer,
    ProposalContext,
    Proposer,
    ToolAblationProposer,
    default_proposers,
    proposers_for,
)
from adapt_agent.optimization.providers import CallableProvider
from adapt_agent.optimization.target import OptimizableAgent

# -- shared fixtures / helpers ------------------------------------------------


class Box:
    """A live, settable string/value holder used to back parameters."""

    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, v):
        self.value = v


def make_agent():
    """A trivial OptimizableAgent (the proposers only read its search space)."""
    return OptimizableAgent.from_callable(lambda x: x, name="toy")


def prompt_param(value="Answer the question.", candidates=None):
    box = Box(value)
    return (
        Parameter(
            name="prompt",
            kind=ParameterKind.PROMPT,
            candidates=candidates,
            getter=box.get,
            setter=box.set,
        ),
        box,
    )


def numeric_param(value=0.5, bounds=(0.0, 1.0), step=None, candidates=None):
    box = Box(value)
    return (
        Parameter(
            name="temp",
            kind=ParameterKind.HYPERPARAM,
            bounds=bounds,
            step=step,
            candidates=candidates,
            getter=box.get,
            setter=box.set,
        ),
        box,
    )


def few_shot_param(value=""):
    box = Box(value)
    return (
        Parameter(
            name="few_shot",
            kind=ParameterKind.FEW_SHOT,
            getter=box.get,
            setter=box.set,
        ),
        box,
    )


def ctx_for(param, *, dataset=None, report=None, judge=None, seed=0, n=4):
    return ProposalContext(
        parameter=param,
        agent=make_agent(),
        dataset=dataset if dataset is not None else GoldenDataset(),
        report=report,
        judge=judge,
        rng=random.Random(seed),
        n=n,
    )


def make_report(results, primary="exact_match"):
    return EvaluationReport(
        aggregate={primary: 0.0},
        primary_metric=primary,
        results=results,
    )


def result(
    index, *, score=0.0, error=None, inputs="i", output="o", expected="e", primary="exact_match"
):
    return ExampleResult(
        index=index,
        inputs=inputs,
        output=output,
        expected=expected,
        scores={primary: score},
        latency=0.0,
        error=error,
    )


# -- Proposer base ------------------------------------------------------------


def test_base_proposer_is_abstract():
    p = Proposer()
    param, _ = prompt_param()
    with pytest.raises(NotImplementedError):
        p.supports(param)
    with pytest.raises(NotImplementedError):
        p.propose(ctx_for(param))


# -- CandidateProposer --------------------------------------------------------


def test_candidate_supports_gating():
    cp = CandidateProposer()
    with_cands, _ = prompt_param(candidates=["a", "b"])
    with_bounds, _ = numeric_param()
    bare = Parameter(name="m", kind=ParameterKind.MODEL)
    assert cp.supports(with_cands) is True
    assert cp.supports(with_bounds) is True
    assert cp.supports(bare) is False


def test_candidate_excludes_current_value():
    param, box = prompt_param(value="b", candidates=["a", "b", "c"])
    box.set("b")
    out = CandidateProposer().propose(ctx_for(param))
    assert out == ["a", "c"]
    assert "b" not in out


def test_candidate_returns_numeric_grid_when_no_candidates():
    # bounds only -> enumerate_candidates produces a grid, minus current.
    # Use non-integral bounds so values stay floats (integral bounds collapse
    # to ints via Parameter._coerce_numeric).
    param, box = numeric_param(value=0.1, bounds=(0.1, 0.9))
    box.set(0.1)
    out = CandidateProposer().propose(ctx_for(param))
    # grid is 5 points 0.1,0.3,0.5,0.7,0.9; current 0.1 excluded.
    assert 0.1 not in out
    assert 0.3 in out and 0.9 in out


# -- NumericProposer ----------------------------------------------------------


def test_numeric_supports_only_with_bounds():
    np_ = NumericProposer()
    with_bounds, _ = numeric_param()
    without = Parameter(name="p", kind=ParameterKind.HYPERPARAM)
    assert np_.supports(with_bounds) is True
    assert np_.supports(without) is False


def test_numeric_local_perturbations_present():
    # Non-integral bounds so perturbations stay floats; span = 0.9 - 0.1 = 0.8.
    param, box = numeric_param(value=0.5, bounds=(0.1, 0.9))
    box.set(0.5)
    out = NumericProposer().propose(ctx_for(param, n=4))
    # deltas -0.2,-0.1,0.1,0.2 * span(0.8) => +/-0.16, +/-0.08 around 0.5
    assert 0.34 in out and 0.42 in out and 0.58 in out and 0.66 in out
    assert 0.5 not in out


def test_numeric_respects_bounds_clamping():
    param, box = numeric_param(value=0.95, bounds=(0.0, 1.0))
    box.set(0.95)
    out = NumericProposer().propose(ctx_for(param, n=6))
    # all candidates must lie within [0, 1]
    assert all(0.0 <= v <= 1.0 for v in out)


def test_numeric_deterministic_with_seed():
    param, box = numeric_param(value=0.5, bounds=(0.0, 1.0))
    box.set(0.5)
    a = NumericProposer().propose(ctx_for(param, seed=123, n=8))
    box.set(0.5)
    b = NumericProposer().propose(ctx_for(param, seed=123, n=8))
    assert a == b


def test_numeric_integer_bounds_yield_ints():
    box = Box(5)
    param = Parameter(
        name="k",
        kind=ParameterKind.HYPERPARAM,
        bounds=(0, 10),
        getter=box.get,
        setter=box.set,
    )
    out = NumericProposer().propose(ctx_for(param, n=6))
    assert all(isinstance(v, int) for v in out)


def test_numeric_caps_at_n():
    param, box = numeric_param(value=0.5, bounds=(0.0, 1.0))
    box.set(0.5)
    out = NumericProposer().propose(ctx_for(param, n=2))
    assert len(out) <= 2


# -- PromptMutationProposer ---------------------------------------------------


def test_prompt_mutation_supports_only_prompt():
    pm = PromptMutationProposer()
    p, _ = prompt_param()
    fs, _ = few_shot_param()
    assert pm.supports(p) is True
    assert pm.supports(fs) is False


def test_prompt_mutation_appends_directives():
    param, box = prompt_param(value="Base instruction.")
    box.set("Base instruction.")
    out = PromptMutationProposer().propose(ctx_for(param, n=5))
    assert len(out) == 5
    assert all(v.startswith("Base instruction.\n\n") for v in out)
    # each candidate is base + one of the known directives
    from adapt_agent.optimization.proposers import _PROMPT_DIRECTIVES

    for v, d in zip(out, _PROMPT_DIRECTIVES, strict=False):
        assert v.endswith(d)


def test_prompt_mutation_empty_base_uses_directive_only():
    param, box = prompt_param(value="")
    box.set("")
    out = PromptMutationProposer().propose(ctx_for(param, n=5))
    from adapt_agent.optimization.proposers import _PROMPT_DIRECTIVES

    assert out[0] == _PROMPT_DIRECTIVES[0]


def test_prompt_mutation_non_string_base_returns_empty():
    box = Box(123)  # non-string current value
    param = Parameter(
        name="prompt",
        kind=ParameterKind.PROMPT,
        getter=box.get,
        setter=box.set,
    )
    assert PromptMutationProposer().propose(ctx_for(param)) == []


def test_prompt_mutation_caps_at_n():
    param, box = prompt_param(value="X")
    box.set("X")
    out = PromptMutationProposer().propose(ctx_for(param, n=2))
    assert len(out) == 2


def test_prompt_mutation_custom_directives():
    param, box = prompt_param(value="X")
    box.set("X")
    pm = PromptMutationProposer(directives=("D1", "D2"))
    out = pm.propose(ctx_for(param, n=5))
    assert out == ["X\n\nD1", "X\n\nD2"]


# -- FewShotProposer ----------------------------------------------------------


def labeled_dataset():
    return GoldenDataset(
        [
            Example(inputs="q0", expected="a0"),
            Example(inputs="q1", expected="a1"),
            Example(inputs="q2", expected="a2"),
            Example(inputs="q3", expected="a3"),
        ]
    )


def test_few_shot_supports_only_few_shot():
    fp = FewShotProposer()
    fs, _ = few_shot_param()
    p, _ = prompt_param()
    assert fp.supports(fs) is True
    assert fp.supports(p) is False


def test_few_shot_empty_when_no_labeled_data():
    fp = FewShotProposer()
    param, _ = few_shot_param()
    ds = GoldenDataset([Example(inputs="q", expected=None)])
    assert fp.propose(ctx_for(param, dataset=ds)) == []


def test_few_shot_builds_blocks_from_labeled_pool():
    fp = FewShotProposer(shots=2, variants=3)
    param, _ = few_shot_param()
    out = fp.propose(ctx_for(param, dataset=labeled_dataset(), seed=1))
    assert out  # non-empty
    assert all(v.startswith("Here are some examples:") for v in out)
    # each block references inputs/outputs
    assert all("Input:" in v and "Output:" in v for v in out)


def test_few_shot_uses_all_when_pool_small():
    # pool size == shots boundary: chosen = list(pool)
    fp = FewShotProposer(shots=4, variants=2)
    param, _ = few_shot_param()
    out = fp.propose(ctx_for(param, dataset=labeled_dataset(), seed=0))
    # all four examples appear; variants dedupe to a single identical block
    assert len(out) == 1
    for ans in ("a0", "a1", "a2", "a3"):
        assert ans in out[0]


def test_few_shot_prefers_report_correct_examples():
    fp = FewShotProposer(shots=2, variants=1)
    param, _ = few_shot_param()
    # Mark only indices 1 and 3 as correct (score >= 1.0, no error).
    report = make_report(
        [
            result(0, score=0.0),
            result(1, score=1.0),
            result(2, score=0.0),
            result(3, score=1.0),
        ]
    )
    out = fp.propose(ctx_for(param, dataset=labeled_dataset(), report=report, seed=0))
    assert len(out) == 1
    block = out[0]
    # Only the correct rows' expected answers should appear.
    assert "a1" in block and "a3" in block
    assert "a0" not in block and "a2" not in block


def test_few_shot_falls_back_when_report_has_no_correct():
    fp = FewShotProposer(shots=4, variants=1)
    param, _ = few_shot_param()
    # No example is correct -> correct_inputs empty -> uses all labeled rows.
    report = make_report([result(i, score=0.0) for i in range(4)])
    out = fp.propose(ctx_for(param, dataset=labeled_dataset(), report=report, seed=0))
    assert len(out) == 1
    for ans in ("a0", "a1", "a2", "a3"):
        assert ans in out[0]


def test_few_shot_errored_results_not_counted_correct():
    fp = FewShotProposer(shots=4, variants=1)
    param, _ = few_shot_param()
    # An errored result with score 1.0 must NOT count as correct.
    report = make_report(
        [
            result(0, score=1.0, error="boom"),
            result(1, score=0.0),
            result(2, score=0.0),
            result(3, score=0.0),
        ]
    )
    out = fp.propose(ctx_for(param, dataset=labeled_dataset(), report=report, seed=0))
    # correct_inputs is empty (the 1.0 was errored) -> fall back to all labeled.
    assert len(out) == 1
    for ans in ("a0", "a1", "a2", "a3"):
        assert ans in out[0]


# -- LLMProposer (fake judge) -------------------------------------------------


def fake_complete_factory(rewrite="REWRITTEN INSTRUCTION"):
    """Deterministic completion fn: returns a rewrite for improve prompts,
    a critique for critique prompts, and a generic answer otherwise.

    The judge now passes its instructions via the ``system`` argument (and the
    untrusted input/output in the user ``prompt``), so we inspect both.
    """

    def _complete(prompt: str, *, system: str | None = None, **_kw) -> str:
        haystack = f"{system or ''}\n{prompt}"
        if "Rewrite the instruction" in haystack:
            return rewrite
        if "Explain concisely" in haystack:
            return "The answer was wrong; be more precise."
        return "ok"

    return _complete


def make_judge(rewrite="REWRITTEN INSTRUCTION"):
    return LLMJudge(fake_complete_factory(rewrite))


def report_with_failures():
    # one failing (score 0) and one passing (score 1) example
    return make_report(
        [
            result(0, score=0.0, inputs="q0", output="wrong", expected="right"),
            result(1, score=1.0, inputs="q1", output="right", expected="right"),
        ]
    )


def test_llm_supports_only_prompt():
    lp = LLMProposer()
    p, _ = prompt_param()
    fs, _ = few_shot_param()
    assert lp.supports(p) is True
    assert lp.supports(fs) is False


def test_llm_returns_empty_without_judge():
    lp = LLMProposer(judge=None)
    param, box = prompt_param(value="Do the task.")
    box.set("Do the task.")
    out = lp.propose(ctx_for(param, report=report_with_failures(), judge=None))
    assert out == []


def test_llm_returns_empty_for_non_string_current():
    box = Box(42)
    param = Parameter(
        name="prompt",
        kind=ParameterKind.PROMPT,
        getter=box.get,
        setter=box.set,
    )
    out = LLMProposer(judge=make_judge()).propose(ctx_for(param, report=report_with_failures()))
    assert out == []


def test_llm_returns_empty_when_no_report():
    param, box = prompt_param(value="Do the task.")
    box.set("Do the task.")
    out = LLMProposer(judge=make_judge()).propose(ctx_for(param, report=None))
    assert out == []


def test_llm_returns_empty_when_no_failures():
    param, box = prompt_param(value="Do the task.")
    box.set("Do the task.")
    all_pass = make_report([result(0, score=1.0), result(1, score=1.0)])
    out = LLMProposer(judge=make_judge()).propose(ctx_for(param, report=all_pass))
    assert out == []


def test_llm_returns_rewrites_from_failures():
    param, box = prompt_param(value="Do the task.")
    box.set("Do the task.")
    out = LLMProposer(judge=make_judge("BETTER PROMPT"), rewrites=2).propose(
        ctx_for(param, report=report_with_failures())
    )
    # deterministic judge returns the same rewrite -> deduped to one
    assert out == ["BETTER PROMPT"]


def test_llm_uses_context_judge_as_fallback():
    param, box = prompt_param(value="Do the task.")
    box.set("Do the task.")
    # judge=None on the proposer, supplied via ctx instead
    out = LLMProposer(judge=None).propose(
        ctx_for(param, report=report_with_failures(), judge=make_judge("CTX"))
    )
    assert out == ["CTX"]


def test_llm_handles_errored_failure_records():
    param, box = prompt_param(value="Do the task.")
    box.set("Do the task.")
    rep = make_report(
        [
            result(0, score=0.0, error="kaboom", inputs="q0", output=None, expected="r"),
        ]
    )
    # errored failures skip the critique call but still produce a rewrite
    out = LLMProposer(judge=make_judge("FIX")).propose(ctx_for(param, report=rep))
    assert out == ["FIX"]


def test_llm_criteria_surfaced_when_shared():
    # All examples agree on one criteria -> _criteria returns it (smoke path).
    param, box = prompt_param(value="Do the task.")
    box.set("Do the task.")
    ds = GoldenDataset(
        [
            Example(inputs="q0", expected="r", metadata={"criteria": "be terse"}),
            Example(inputs="q1", expected="r", metadata={"criteria": "be terse"}),
        ]
    )
    out = LLMProposer(judge=make_judge("WITH_CRIT")).propose(
        ctx_for(param, dataset=ds, report=report_with_failures())
    )
    assert out == ["WITH_CRIT"]


# -- default_proposers / proposers_for ----------------------------------------


def test_default_proposers_without_judge_excludes_llm():
    proposers = default_proposers(judge=None)
    names = [p.name for p in proposers]
    assert "llm" not in names
    assert "llm_tool" not in names
    assert names == [
        "prompt_mutation",
        "few_shot",
        "candidates",
        "numeric",
        "tool_ablation",
    ]


def test_default_proposers_with_judge_includes_llm_first():
    proposers = default_proposers(judge=make_judge())
    names = [p.name for p in proposers]
    assert names[0] == "llm"
    assert set(names) == {
        "llm",
        "prompt_mutation",
        "few_shot",
        "candidates",
        "numeric",
        "tool_ablation",
        "llm_tool",
    }


def test_proposers_for_filters_by_kind_prompt():
    proposers = default_proposers(judge=make_judge())
    p, _ = prompt_param(candidates=["a", "b"])
    supporting = {x.name for x in proposers_for(p, proposers)}
    # prompt + candidates -> llm, prompt_mutation, candidates
    assert supporting == {"llm", "prompt_mutation", "candidates"}


def test_proposers_for_filters_numeric():
    proposers = default_proposers(judge=None)
    num, _ = numeric_param()
    supporting = {x.name for x in proposers_for(num, proposers)}
    assert supporting == {"candidates", "numeric"}


def test_proposers_for_filters_few_shot():
    proposers = default_proposers(judge=None)
    fs, _ = few_shot_param()
    supporting = {x.name for x in proposers_for(fs, proposers)}
    assert supporting == {"few_shot"}


# -- ToolAblationProposer -----------------------------------------------------


def tool_param(value, candidates, *, kind=ParameterKind.TOOL):
    box = Box(value)
    return (
        Parameter(
            name="tools",
            kind=kind,
            candidates=candidates,
            getter=box.get,
            setter=box.set,
        ),
        box,
    )


def test_tool_ablation_supports_tool_and_skill():
    tp = ToolAblationProposer()
    tool, _ = tool_param(["a", "b"], [["a", "b"], ["b"], ["a"]])
    skill, _ = tool_param(["a"], [["a"]], kind=ParameterKind.SKILL)
    prompt, _ = prompt_param()
    assert tp.supports(tool) is True
    assert tp.supports(skill) is True
    assert tp.supports(prompt) is False


def test_tool_ablation_proposes_drop_one_subsets_excluding_current():
    # Full set is the current value; candidates include drop-one subsets.
    full = ["a", "b", "c"]
    candidates = [["a", "b", "c"], ["b", "c"], ["a", "c"], ["a", "b"]]
    param, box = tool_param(full, candidates)
    box.set(full)
    out = ToolAblationProposer().propose(ctx_for(param))
    # current full set excluded; the three drop-one subsets remain.
    assert out == [["b", "c"], ["a", "c"], ["a", "b"]]
    assert full not in out


def test_tool_ablation_skill_kind_works():
    full = ["search", "calc"]
    candidates = [["search", "calc"], ["calc"], ["search"]]
    param, box = tool_param(full, candidates, kind=ParameterKind.SKILL)
    box.set(full)
    out = ToolAblationProposer().propose(ctx_for(param))
    assert out == [["calc"], ["search"]]


# -- LLMToolProposer ----------------------------------------------------------


class StubToolJudge(LLMJudge):
    """A CallableProvider-backed judge whose suggest_tools returns a fixed list."""

    def __init__(self, suggestions):
        super().__init__(CallableProvider(lambda prompt, **kw: "ok"))
        self._suggestions = suggestions
        self.calls = []

    def suggest_tools(self, component, failures, current_tools, *, n=3):
        self.calls.append((component, failures, current_tools, n))
        return list(self._suggestions)


def tool_failures_report():
    return make_report(
        [
            result(0, score=0.0, inputs="q0", output="wrong", expected="right"),
            result(1, score=1.0, inputs="q1", output="right", expected="right"),
        ]
    )


def test_llm_tool_supports_tool_and_skill():
    lp = LLMToolProposer()
    tool, _ = tool_param(["a"], [["a"]])
    skill, _ = tool_param(["a"], [["a"]], kind=ParameterKind.SKILL)
    prompt, _ = prompt_param()
    assert lp.supports(tool) is True
    assert lp.supports(skill) is True
    assert lp.supports(prompt) is False


def test_llm_tool_returns_empty_without_judge():
    param, _ = tool_param(["a"], [["a"]])
    ctx = ctx_for(param, report=tool_failures_report(), judge=None)
    assert LLMToolProposer(judge=None).propose(ctx) == []
    assert ctx.recommendations == []


def test_llm_tool_writes_recommendations_and_returns_no_values():
    suggestions = [
        {"name": "web_search", "description": "search the web", "rationale": "needs facts"},
        {"name": "calculator", "description": "do math"},
    ]
    judge = StubToolJudge(suggestions)
    box = Box(["existing"])
    param = Parameter(
        name="tools",
        kind=ParameterKind.TOOL,
        component="retriever",
        candidates=[["existing"]],
        getter=box.get,
        setter=box.set,
    )
    ctx = ctx_for(param, report=tool_failures_report(), judge=judge)
    out = LLMToolProposer().propose(ctx)
    # Advisory only: never returns candidate values.
    assert out == []
    # Suggestions recorded as human-readable strings into ctx.recommendations.
    assert ctx.recommendations == [
        "[retriever] consider tool: web_search -- search the web (needs facts)",
        "[retriever] consider tool: calculator -- do math",
    ]
    # Judge invoked with component, failure dicts, and current tool names.
    component, failures, current_tools, n = judge.calls[0]
    assert component == "retriever"
    assert current_tools == ["existing"]
    assert all(set(f) >= {"input", "output", "expected", "critique"} for f in failures)
    # Only the failing example is passed (the passing one is excluded).
    assert [f["input"] for f in failures] == ["q0"]


def test_llm_tool_uses_context_judge_as_fallback():
    judge = StubToolJudge([{"name": "t", "description": "d", "rationale": "r"}])
    param, box = tool_param(["x"], [["x"]])
    box.set(["x"])
    ctx = ctx_for(param, report=tool_failures_report(), judge=judge)
    out = LLMToolProposer(judge=None).propose(ctx)
    assert out == []
    assert ctx.recommendations == ["consider tool: t -- d (r)"]


def test_llm_tool_dedupes_recommendations():
    judge = StubToolJudge([{"name": "t", "description": "d"}])
    param, box = tool_param(["x"], [["x"]])
    box.set(["x"])
    ctx = ctx_for(param, report=tool_failures_report(), judge=judge)
    LLMToolProposer().propose(ctx)
    LLMToolProposer().propose(ctx)
    assert ctx.recommendations == ["consider tool: t -- d"]


# -- default_proposers tool/skill coverage ------------------------------------


def test_default_proposers_include_tool_ablation_always():
    names = [p.name for p in default_proposers(judge=None)]
    assert "tool_ablation" in names
    assert "llm_tool" not in names


def test_default_proposers_include_llm_tool_with_judge():
    names = [p.name for p in default_proposers(judge=make_judge())]
    assert "tool_ablation" in names
    assert "llm_tool" in names


def test_proposers_for_filters_tool_kind():
    proposers = default_proposers(judge=make_judge())
    tool, _ = tool_param(["a", "b"], [["a", "b"], ["b"], ["a"]])
    supporting = {x.name for x in proposers_for(tool, proposers)}
    # CandidateProposer also supports a TOOL param that carries explicit
    # candidates (it gates on candidates/bounds, not kind).
    assert supporting == {"tool_ablation", "llm_tool", "candidates"}
