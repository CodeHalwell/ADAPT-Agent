"""Tests for adapt_agent.optimization.optimizers.

A controllable toy agent backs every test: a dict of live state plus a runner
closure whose output quality depends on that state, wrapped via
``OptimizableAgent.from_callable`` with explicit Parameters. Scoring is exact and
deterministic (no network); judge-driven paths use a plain completion function.
"""

import pytest

from adapt_agent.optimization.dataset import Example, GoldenDataset
from adapt_agent.optimization.evaluation import EvaluationHarness
from adapt_agent.optimization.judge import LLMJudge
from adapt_agent.optimization.metrics import exact_match
from adapt_agent.optimization.optimizers import (
    BootstrapFewShotOptimizer,
    CoordinateAscentOptimizer,
    EvolutionaryOptimizer,
    GridSearchOptimizer,
    OptimizationResult,
    Optimizer,
    PipelineOptimizer,
    RandomSearchOptimizer,
    Trial,
    _dedup,
    make_default_optimizer,
)
from adapt_agent.optimization.parameters import Parameter, ParameterKind
from adapt_agent.optimization.proposers import Proposer
from adapt_agent.optimization.target import OptimizableAgent

# -- Toy agent ----------------------------------------------------------------

#: The single prompt that makes the agent answer correctly.
GOOD_PROMPT = "GOOD"
#: The temperature that makes the agent answer correctly.
GOOD_TEMP = 1.0

QUESTIONS = ["q0", "q1", "q2", "q3"]
ANSWERS = {"q0": "a0", "q1": "a1", "q2": "a2", "q3": "a3"}


def build_toy(*, prompt="BAD", temp=0.0, few_shot="", prompt_candidates=None, gate="prompt"):
    """Return (state, agent).

    The runner answers correctly only when the gating condition is met:

    * gate="prompt": prompt must equal GOOD_PROMPT (or contain it).
    * gate="temp":   temp must equal GOOD_TEMP.
    * gate="few_shot": few_shot block must mention the question's answer.

    Otherwise it returns a wrong answer, so the baseline scores 0 and the
    optimizer must discover the right configuration to score 1.
    """
    state = {"prompt": prompt, "temp": temp, "few_shot": few_shot}

    def runner(question):
        if gate == "prompt":
            ok = GOOD_PROMPT in str(state["prompt"])
        elif gate == "temp":
            ok = state["temp"] == GOOD_TEMP
        elif gate == "few_shot":
            ok = ANSWERS.get(question, "") in str(state["few_shot"])
        else:  # pragma: no cover - defensive
            ok = False
        return ANSWERS[question] if ok else "WRONG"

    params = [
        Parameter(
            name="prompt",
            kind=ParameterKind.PROMPT,
            candidates=prompt_candidates,
            getter=lambda: state["prompt"],
            setter=lambda v: state.__setitem__("prompt", v),
        ),
        Parameter(
            name="temp",
            kind=ParameterKind.HYPERPARAM,
            bounds=(0.0, 1.0),
            candidates=[0.0, 0.5, 1.0],
            getter=lambda: state["temp"],
            setter=lambda v: state.__setitem__("temp", v),
        ),
        Parameter(
            name="few_shot",
            kind=ParameterKind.FEW_SHOT,
            getter=lambda: state["few_shot"],
            setter=lambda v: state.__setitem__("few_shot", v),
        ),
    ]
    agent = OptimizableAgent.from_callable(runner, parameters=params, name="toy")
    return state, agent


def dataset():
    return GoldenDataset([Example(inputs=q, expected=ANSWERS[q]) for q in QUESTIONS])


def small_dataset():
    """Three examples so a default 3-shot block can cover every answer."""
    qs = QUESTIONS[:3]
    return GoldenDataset([Example(inputs=q, expected=ANSWERS[q]) for q in qs])


def harness():
    return EvaluationHarness(exact_match())


# -- OptimizationResult -------------------------------------------------------


def test_result_properties():
    res = OptimizationResult(
        best_config={"prompt": GOOD_PROMPT},
        best_score=1.0,
        baseline_score=0.25,
        history=[
            Trial(config={}, score=0.25, strategy="x"),
            Trial(config={"prompt": GOOD_PROMPT}, score=1.0, strategy="x", accepted=True),
        ],
    )
    assert res.improved is True
    assert res.improvement == pytest.approx(0.75)
    assert res.n_evals == 2
    d = res.to_dict()
    assert d["improved"] is True
    assert d["baseline_score"] == 0.25
    assert d["best_score"] == 1.0
    assert d["improvement"] == pytest.approx(0.75)
    assert d["n_evals"] == 2
    assert d["best_config"] == {"prompt": GOOD_PROMPT}
    assert "baseline=0.250" in repr(res) and "best=1.000" in repr(res)


def test_result_not_improved_when_equal():
    res = OptimizationResult(best_config={}, best_score=0.5, baseline_score=0.5)
    assert res.improved is False
    assert res.improvement == 0.0
    assert res.n_evals == 0


# -- base optimizer error paths -----------------------------------------------


def test_optimize_empty_dataset_raises():
    _, agent = build_toy()
    opt = GridSearchOptimizer(harness())
    with pytest.raises(ValueError):
        opt.optimize(agent, GoldenDataset())


def test_base_search_not_implemented():
    opt = Optimizer(harness())
    with pytest.raises(NotImplementedError):
        opt._search(None, dataset(), None)  # type: ignore[arg-type]


# -- GridSearchOptimizer ------------------------------------------------------


def test_grid_finds_best_prompt_and_applies_in_place():
    state, agent = build_toy(
        prompt="BAD",
        gate="prompt",
        prompt_candidates=["BAD", "ALSO_BAD", GOOD_PROMPT],
    )
    opt = GridSearchOptimizer(harness(), seed=0)
    res = opt.optimize(agent, dataset())
    assert res.baseline_score == 0.0
    assert res.best_score == 1.0
    assert res.improved
    # Winning config applied to the LIVE agent state.
    assert state["prompt"] == GOOD_PROMPT
    assert agent.run("q0") == "a0"


def test_grid_respects_max_evals_cap():
    state, agent = build_toy(
        prompt="BAD",
        gate="prompt",
        prompt_candidates=["BAD", "B1", "B2", GOOD_PROMPT],
    )
    # temp has 3 candidates, prompt has 4 -> grid is large; cap at 2 evals.
    opt = GridSearchOptimizer(harness(), seed=0, max_evals=2)
    res = opt.optimize(agent, dataset())
    assert res.n_evals <= 2


def test_grid_validation_score_computed():
    _, agent = build_toy(prompt="BAD", gate="prompt", prompt_candidates=["BAD", GOOD_PROMPT])
    val = GoldenDataset([Example(inputs="q0", expected="a0")])
    res = GridSearchOptimizer(harness(), seed=0).optimize(agent, dataset(), val_dataset=val)
    assert res.validation_score == 1.0


# -- RandomSearchOptimizer ----------------------------------------------------


def test_random_search_finds_best_and_is_seeded():
    state1, agent1 = build_toy(
        prompt="BAD", gate="prompt", prompt_candidates=["BAD", "B1", GOOD_PROMPT]
    )
    res1 = RandomSearchOptimizer(harness(), seed=42, max_evals=20).optimize(agent1, dataset())
    state2, agent2 = build_toy(
        prompt="BAD", gate="prompt", prompt_candidates=["BAD", "B1", GOOD_PROMPT]
    )
    res2 = RandomSearchOptimizer(harness(), seed=42, max_evals=20).optimize(agent2, dataset())
    # Determinism under a fixed seed.
    assert [t.config for t in res1.history] == [t.config for t in res2.history]
    assert res1.best_score == 1.0
    assert state1["prompt"] == GOOD_PROMPT


def test_random_search_no_optimizable_params_returns_baseline():
    # An agent with no settable params -> search short-circuits.
    agent = OptimizableAgent.from_callable(lambda q: "WRONG", name="ro")
    res = RandomSearchOptimizer(harness(), seed=0).optimize(agent, dataset())
    assert res.n_evals == 0
    assert res.best_config == {}
    assert res.best_score == res.baseline_score


# -- CoordinateAscentOptimizer ------------------------------------------------


def test_coordinate_ascent_improves_via_candidates():
    state, agent = build_toy(prompt="BAD", gate="prompt", prompt_candidates=["BAD", GOOD_PROMPT])
    res = CoordinateAscentOptimizer(harness(), seed=0).optimize(agent, dataset())
    assert res.improved
    assert res.best_score == 1.0
    assert state["prompt"] == GOOD_PROMPT


def fake_complete(prompt: str) -> str:
    """Deterministic judge completion: rewrites instructions to GOOD_PROMPT.

    The judge now places its task instructions in the provider ``system=`` argument
    and passes only the data (current instruction + failures) in the user prompt, so
    we trigger on the prompt's structural marker instead of an instruction phrase.
    The rewrite prompt embeds the CURRENT INSTRUCTION block; the critique prompt does
    not, so this returns GOOD_PROMPT only for rewrites.
    """
    if "CURRENT INSTRUCTION" in prompt or "Rewrite the instruction" in prompt:
        return GOOD_PROMPT
    return "It produced the wrong answer."


def test_coordinate_ascent_improves_via_llm_judge():
    # No prompt candidates: improvement must come from the LLM proposer
    # rewriting the (string) prompt into GOOD_PROMPT.
    state, agent = build_toy(prompt="initial instruction", gate="prompt", prompt_candidates=None)
    judge = LLMJudge(fake_complete)
    res = CoordinateAscentOptimizer(harness(), seed=0, judge=judge).optimize(agent, dataset())
    assert res.improved
    assert res.best_score == 1.0
    assert GOOD_PROMPT in str(state["prompt"])


def test_coordinate_ascent_min_improvement_gating():
    # Set min_improvement high enough that the full +1.0 jump is below it,
    # so nothing is accepted and the agent stays at baseline.
    state, agent = build_toy(prompt="BAD", gate="prompt", prompt_candidates=["BAD", GOOD_PROMPT])
    res = CoordinateAscentOptimizer(harness(), seed=0, min_improvement=5.0).optimize(
        agent, dataset()
    )
    assert not res.improved
    assert res.best_config == {}
    # Restored to baseline because no config was accepted.
    assert state["prompt"] == "BAD"


def test_coordinate_ascent_no_params_for_kind_returns_early():
    # Restrict to ROUTING kind which the toy agent has none of.
    _, agent = build_toy()
    res = CoordinateAscentOptimizer(harness(), seed=0, kinds=(ParameterKind.ROUTING,)).optimize(
        agent, dataset()
    )
    assert res.n_evals == 0
    assert res.best_config == {}


def test_coordinate_ascent_respects_max_evals():
    state, agent = build_toy(
        prompt="BAD", gate="prompt", prompt_candidates=["B0", "B1", "B2", "B3", GOOD_PROMPT]
    )
    res = CoordinateAscentOptimizer(harness(), seed=0, max_evals=1, rounds=3).optimize(
        agent, dataset()
    )
    assert res.n_evals <= 1


def test_coordinate_ascent_flaky_proposer_does_not_abort():
    class Boom(Proposer):
        name = "boom"

        def supports(self, parameter):
            return parameter.kind is ParameterKind.PROMPT

        def propose(self, ctx):
            raise RuntimeError("kaboom")

    from adapt_agent.optimization.proposers import CandidateProposer

    state, agent = build_toy(prompt="BAD", gate="prompt", prompt_candidates=["BAD", GOOD_PROMPT])
    opt = CoordinateAscentOptimizer(harness(), seed=0, proposers=[Boom(), CandidateProposer()])
    res = opt.optimize(agent, dataset())
    # The good CandidateProposer still drives improvement despite Boom raising.
    assert res.best_score == 1.0
    assert state["prompt"] == GOOD_PROMPT


def test_coordinate_ascent_no_supporting_proposer_skips_param():
    # A MODEL param with no candidates/bounds: no proposer supports it, so the
    # `if not supporting: continue` branch is exercised.
    state = {"m": "x"}
    param = Parameter(
        name="m",
        kind=ParameterKind.MODEL,
        getter=lambda: state["m"],
        setter=lambda v: state.__setitem__("m", v),
    )
    agent = OptimizableAgent.from_callable(lambda q: "WRONG", parameters=[param])
    res = CoordinateAscentOptimizer(harness(), seed=0).optimize(agent, dataset())
    assert res.n_evals == 0
    assert res.best_config == {}


def test_coordinate_ascent_verbose_logs(caplog):
    import logging

    state, agent = build_toy(prompt="BAD", gate="prompt", prompt_candidates=["BAD", GOOD_PROMPT])
    with caplog.at_level(logging.INFO, logger="adapt_agent.optimization.optimizers"):
        res = CoordinateAscentOptimizer(harness(), seed=0, verbose=True).optimize(agent, dataset())
    assert res.best_score == 1.0
    assert any("baseline" in r.message for r in caplog.records)


# -- BootstrapFewShotOptimizer ------------------------------------------------


def test_bootstrap_few_shot_restricted_to_few_shot_kind():
    opt = BootstrapFewShotOptimizer(harness(), seed=0)
    assert opt.kinds == (ParameterKind.FEW_SHOT,)


def test_bootstrap_few_shot_improves_few_shot_block():
    # The agent answers correctly only when the few_shot block contains the
    # answer. FewShotProposer falls back to all labeled rows (baseline 0
    # correct), building a block containing every answer -> score 1.
    state, agent = build_toy(prompt="BAD", few_shot="", gate="few_shot")
    ds = small_dataset()
    res = BootstrapFewShotOptimizer(harness(), seed=0).optimize(agent, ds)
    assert res.improved
    assert res.best_score == 1.0
    # Only the few_shot parameter was touched.
    assert set(res.best_config) == {"few_shot"}
    for q in ("q0", "q1", "q2"):
        assert ANSWERS[q] in str(state["few_shot"])


# -- EvolutionaryOptimizer ----------------------------------------------------


def test_evolutionary_improves_over_generations():
    state, agent = build_toy(
        prompt="BAD", gate="prompt", prompt_candidates=["BAD", "B1", GOOD_PROMPT]
    )
    res = EvolutionaryOptimizer(harness(), seed=0, population=4, generations=3).optimize(
        agent, dataset()
    )
    assert res.best_score == 1.0
    assert state["prompt"] == GOOD_PROMPT


def test_evolutionary_no_params_returns_baseline():
    agent = OptimizableAgent.from_callable(lambda q: "WRONG", name="ro")
    res = EvolutionaryOptimizer(harness(), seed=0).optimize(agent, dataset())
    assert res.n_evals == 0
    assert res.best_config == {}


def test_evolutionary_mutate_no_supporting_proposer():
    # All optimizable params are MODEL kind with no candidates -> no proposer
    # supports them, so _mutate returns None and offspring is empty.
    state = {"m": "x"}
    param = Parameter(
        name="m",
        kind=ParameterKind.MODEL,
        getter=lambda: state["m"],
        setter=lambda v: state.__setitem__("m", v),
    )
    agent = OptimizableAgent.from_callable(lambda q: "WRONG", parameters=[param])
    res = EvolutionaryOptimizer(harness(), seed=0, population=4, generations=3).optimize(
        agent, dataset()
    )
    # No improvement possible; best stays at baseline.
    assert not res.improved


def test_evolutionary_respects_max_evals():
    state, agent = build_toy(
        prompt="BAD", gate="prompt", prompt_candidates=["BAD", "B1", GOOD_PROMPT]
    )
    res = EvolutionaryOptimizer(
        harness(), seed=0, max_evals=2, population=4, generations=5
    ).optimize(agent, dataset())
    assert res.n_evals <= 2


# -- PipelineOptimizer --------------------------------------------------------


def test_pipeline_requires_stages():
    with pytest.raises(ValueError):
        PipelineOptimizer(harness(), [])


def test_pipeline_threads_best_across_stages():
    # Stage 1 fixes few_shot; gate on few_shot. Combine two coordinate stages.
    state, agent = build_toy(prompt="BAD", few_shot="", gate="few_shot")
    stages = [
        BootstrapFewShotOptimizer(harness(), seed=0),
        CoordinateAscentOptimizer(harness(), seed=0, kinds=(ParameterKind.PROMPT,)),
    ]
    res = PipelineOptimizer(harness(), stages, seed=0).optimize(agent, small_dataset())
    assert res.best_score == 1.0
    assert "few_shot" in res.best_config
    # Live agent carries the accumulated best.
    assert agent.run("q0") == "a0"


def test_pipeline_enforces_shared_max_evals_budget():
    # Two prompt stages that could each run many trials; the pipeline-wide
    # max_evals must cap the TOTAL number of trials across stages (hard cap).
    state, agent = build_toy(prompt="BAD", gate="prompt", prompt_candidates=["BAD", GOOD_PROMPT])
    stages = [
        CoordinateAscentOptimizer(harness(), seed=0, kinds=(ParameterKind.PROMPT,), max_evals=50),
        CoordinateAscentOptimizer(harness(), seed=0, kinds=(ParameterKind.PROMPT,), max_evals=50),
    ]
    res = PipelineOptimizer(harness(), stages, seed=0, max_evals=3).optimize(agent, small_dataset())
    assert len(res.history) <= 3


def test_pipeline_search_not_implemented():
    opt = PipelineOptimizer(harness(), [GridSearchOptimizer(harness())])
    with pytest.raises(NotImplementedError):
        opt._search(None, dataset(), None)  # type: ignore[arg-type]


def test_pipeline_validation_score():
    state, agent = build_toy(prompt="BAD", gate="prompt", prompt_candidates=["BAD", GOOD_PROMPT])
    val = GoldenDataset([Example(inputs="q1", expected="a1")])
    stages = [CoordinateAscentOptimizer(harness(), seed=0, kinds=(ParameterKind.PROMPT,))]
    res = PipelineOptimizer(harness(), stages, seed=0).optimize(agent, dataset(), val_dataset=val)
    assert res.validation_score == 1.0


# -- make_default_optimizer ---------------------------------------------------


def test_make_default_optimizer_returns_pipeline_and_runs():
    state, agent = build_toy(prompt="BAD", gate="prompt", prompt_candidates=["BAD", GOOD_PROMPT])
    opt = make_default_optimizer(harness(), max_evals=30)
    assert isinstance(opt, PipelineOptimizer)
    assert len(opt.stages) == 4
    res = opt.optimize(agent, dataset())
    assert res.best_score == 1.0
    assert state["prompt"] == GOOD_PROMPT


def test_make_default_optimizer_with_judge_drives_llm_rewrite():
    # No prompt candidates -> the coordinate-ascent prompt stage must use the
    # judge's LLM proposer to rewrite into GOOD_PROMPT.
    state, agent = build_toy(prompt="initial", gate="prompt", prompt_candidates=None)
    judge = LLMJudge(fake_complete)
    opt = make_default_optimizer(harness(), judge=judge, max_evals=30)
    res = opt.optimize(agent, dataset())
    assert res.best_score == 1.0
    assert GOOD_PROMPT in str(state["prompt"])


# -- min_improvement default --------------------------------------------------


def test_min_improvement_default_is_1e_3():
    # Documented default: avoid chasing judge noise (was 1e-9).
    assert Optimizer(harness()).min_improvement == pytest.approx(1e-3)
    assert CoordinateAscentOptimizer(harness()).min_improvement == pytest.approx(1e-3)


# -- tool/skill recommendations -----------------------------------------------


class _FakeToolJudge:
    """A judge stub: suggest_tools returns fixed items regardless of input."""

    def __init__(self, items):
        self._items = items
        self.calls = []

    def suggest_tools(self, component, failures, current_tools, *, n=3):
        self.calls.append((component, list(failures), list(current_tools)))
        return list(self._items)


def build_tool_agent(*, tools=("search",)):
    """Toy agent with a TOOL parameter backed by simple in-memory state.

    The runner always answers WRONG so every example fails (giving the judge a
    non-empty failure pool), and the tool param is optimizable but inert.
    """
    state = {"tools": list(tools)}

    def runner(question):
        return "WRONG"

    param = Parameter(
        name="tools",
        kind=ParameterKind.TOOL,
        component="agent",
        candidates=[["search"], ["search", "calc"]],
        getter=lambda: state["tools"],
        setter=lambda v: state.__setitem__("tools", list(v)),
    )
    agent = OptimizableAgent.from_callable(runner, parameters=[param], name="tool_toy")
    return state, agent


def test_suggest_tools_populates_recommendations():
    state, agent = build_tool_agent()
    judge = _FakeToolJudge(
        [
            {"name": "calculator", "description": "do math", "rationale": "math fails"},
            {"name": "wiki", "description": "lookup facts"},
        ]
    )
    opt = CoordinateAscentOptimizer(
        harness(),
        seed=0,
        judge=judge,
        suggest_tools=True,
        kinds=(ParameterKind.TOOL,),
    )
    res = opt.optimize(agent, dataset())
    assert res.recommendations  # non-empty
    joined = " ".join(res.recommendations)
    assert "calculator" in joined
    assert "wiki" in joined
    assert all(r.startswith("[agent]") for r in res.recommendations)
    # to_dict carries them.
    assert res.to_dict()["recommendations"] == res.recommendations
    # The judge saw the component name, failure records and current tools. The
    # optimizer's post-search call (the last one) passes input/output/expected
    # records derived from state.best_report.failures().
    assert judge.calls
    component, failures, current = judge.calls[-1]
    assert component == "agent"
    assert current == ["search"]
    assert failures and set(failures[0]) == {"input", "output", "expected"}


def test_suggest_tools_off_by_default():
    # Isolate the optimizer's post-search suggestion from the proposer set: with
    # only a CandidateProposer (no llm_tool proposer) and suggest_tools left at its
    # default (False), the optimizer never calls the judge for tool ideas.
    from adapt_agent.optimization.proposers import CandidateProposer

    state, agent = build_tool_agent()
    judge = _FakeToolJudge([{"name": "x", "description": "y"}])
    opt = CoordinateAscentOptimizer(
        harness(),
        seed=0,
        judge=judge,
        kinds=(ParameterKind.TOOL,),
        proposers=[CandidateProposer()],
    )
    assert opt.suggest_tools is False  # default
    res = opt.optimize(agent, dataset())
    assert res.recommendations == []
    assert judge.calls == []


def test_make_default_optimizer_has_four_stages_and_tool_stage():
    opt = make_default_optimizer(harness(), max_evals=40)
    assert isinstance(opt, PipelineOptimizer)
    assert len(opt.stages) == 4
    tool_stage = opt.stages[-1]
    assert tool_stage.kinds == (ParameterKind.TOOL, ParameterKind.SKILL)


def test_make_default_optimizer_enables_suggest_tools_with_judge():
    judge = _FakeToolJudge([])
    opt = make_default_optimizer(harness(), judge=judge, max_evals=40)
    assert opt.suggest_tools is True
    assert opt.stages[-1].suggest_tools is True
    # Without a judge, suggest_tools stays off.
    plain = make_default_optimizer(harness(), max_evals=40)
    assert plain.suggest_tools is False
    assert plain.stages[-1].suggest_tools is False


def test_pipeline_merges_child_recommendations():
    state, agent = build_tool_agent()
    judge = _FakeToolJudge([{"name": "calculator", "description": "do math"}])
    stages = [
        CoordinateAscentOptimizer(
            harness(),
            seed=0,
            judge=judge,
            suggest_tools=True,
            kinds=(ParameterKind.TOOL,),
        )
    ]
    res = PipelineOptimizer(harness(), stages, seed=0).optimize(agent, dataset())
    assert any("calculator" in r for r in res.recommendations)


# -- _dedup -------------------------------------------------------------------


def test_dedup_hashable_order_preserving():
    assert _dedup([3, 1, 3, 2, 1]) == [3, 1, 2]


def test_dedup_handles_unhashable_lists():
    out = _dedup([[1, 2], [1, 2], [3], "x", "x", [3]])
    assert out == [[1, 2], [3], "x"]


def test_dedup_mixed_hashable_unhashable():
    out = _dedup([1, [1], 1, [1], "a"])
    assert out == [1, [1], "a"]


# -- evaluation cache ---------------------------------------------------------
#
# Re-measuring a configuration already scored in the same run is pure waste for
# a deterministic agent, and the default pipeline paid it five times over in
# stage baselines alone. These tests count actual agent invocations.


def _counting_agent(prompt_candidates):
    """A toy agent whose runner counts invocations."""
    state = {"prompt": "BAD"}
    counter = {"runs": 0}

    def runner(question):
        counter["runs"] += 1
        return ANSWERS[question] if GOOD_PROMPT in str(state["prompt"]) else "WRONG"

    params = [
        Parameter(
            name="prompt",
            kind=ParameterKind.PROMPT,
            candidates=prompt_candidates,
            getter=lambda: state["prompt"],
            setter=lambda v: state.__setitem__("prompt", v),
        )
    ]
    agent = OptimizableAgent.from_callable(runner, parameters=params, name="counting-toy")
    return counter, agent


def _run_two_stage_pipeline(*, cache: bool, stage_cache: bool | None = None):
    """Run a two-stage grid pipeline; return (agent_runs, result)."""
    counter, agent = _counting_agent(["BAD", GOOD_PROMPT])
    per_stage = cache if stage_cache is None else stage_cache
    stages = [
        GridSearchOptimizer(harness(), max_evals=8, cache_evaluations=per_stage),
        GridSearchOptimizer(harness(), max_evals=8, cache_evaluations=per_stage),
    ]
    pipe = PipelineOptimizer(harness(), stages, max_evals=16, cache_evaluations=cache)
    result = pipe.optimize(agent, dataset())
    return counter["runs"], result


def test_pipeline_cache_skips_redundant_full_dataset_passes():
    runs_cached, result_cached = _run_two_stage_pipeline(cache=True)
    runs_uncached, result_uncached = _run_two_stage_pipeline(cache=False)

    # Identical outcome: same winner, same score, same trial history scores.
    assert result_cached.best_score == result_uncached.best_score == 1.0
    assert result_cached.best_config == result_uncached.best_config
    assert [t.score for t in result_cached.history] == [t.score for t in result_uncached.history]

    # Uncached: baseline runs for the pipeline and both stages, plus both grid
    # candidates per stage = 7 full-dataset passes. Cached: only the pipeline
    # baseline and the single novel configuration are ever run.
    n = len(dataset())
    assert runs_uncached == 7 * n
    assert runs_cached == 2 * n


def test_stage_level_cache_opt_out_is_respected():
    # Stages built with cache_evaluations=False ignore the injected pipeline
    # cache entirely (they re-measure), even though the pipeline itself caches.
    runs, result = _run_two_stage_pipeline(cache=True, stage_cache=False)
    assert result.best_score == 1.0
    assert runs == 7 * len(dataset())


def test_eval_config_reuses_the_baseline_report_for_identical_state():
    from adapt_agent.optimization.optimizers import _bind_baseline

    counter, agent = _counting_agent(["BAD", GOOD_PROMPT])
    data = dataset()
    opt = GridSearchOptimizer(harness())
    opt._eval_cache = {}

    baseline = opt._evaluate_baseline(agent, data)
    _bind_baseline(opt, agent.snapshot())
    # An empty config restores the exact baseline state -> served from cache.
    assert opt._eval_config(agent, {}, data) is baseline
    assert counter["runs"] == len(data)


def test_incomplete_reports_are_never_cached():
    from adapt_agent.optimization.evaluation import EvaluationReport
    from adapt_agent.optimization.optimizers import _bind_baseline

    class StubHarness:
        """Returns scripted reports; counts evaluate() calls."""

        def __init__(self, reports):
            self.reports = list(reports)
            self.calls = 0

        def evaluate(self, target, data):
            self.calls += 1
            return self.reports.pop(0)

    def report(*, transient: int) -> EvaluationReport:
        return EvaluationReport(
            aggregate={"exact_match": 1.0},
            primary_metric="exact_match",
            n_transient_errors=transient,
            n_evaluated=4,
        )

    stub = StubHarness(
        [report(transient=1), report(transient=1), report(transient=0), report(transient=0)]
    )
    _, agent = _counting_agent(["BAD", GOOD_PROMPT])
    data = dataset()
    opt = GridSearchOptimizer(stub)
    opt._eval_cache = {}
    _bind_baseline(opt, agent.snapshot())

    config = {"prompt": GOOD_PROMPT}
    # Incomplete reports pass through uncached: the same config re-evaluates.
    assert opt._eval_config(agent, config, data).is_complete is False
    assert opt._eval_config(agent, config, data).is_complete is False
    assert stub.calls == 2
    # A complete report is cached: the fourth request never reaches the harness.
    assert opt._eval_config(agent, config, data).is_complete is True
    assert opt._eval_config(agent, config, data).is_complete is True
    assert stub.calls == 3


def test_cache_is_cleared_between_optimize_calls():
    # A fresh optimize() must own a fresh cache: the attribute is reset after
    # each run, so nothing leaks across runs (or across datasets whose id()
    # might be recycled once the first dataset is garbage-collected).
    counter, agent = _counting_agent(["BAD", GOOD_PROMPT])
    opt = GridSearchOptimizer(harness(), max_evals=8)
    opt.optimize(agent, dataset())
    assert opt._eval_cache is None
    first = counter["runs"]
    opt.optimize(agent, dataset())
    assert counter["runs"] > first
