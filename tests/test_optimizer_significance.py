"""An optimizer "improvement" has to be more than noise.

Every optimizer used to accept a candidate whose mean beat the incumbent's by
``1e-3``, on one measurement per configuration, cached. A knob with no effect
at all on an agent right 70% of the time then "improved" the agent in most
runs. These tests pin the replacement: a paired test over per-row deltas,
repeated measurement of non-deterministic targets, a held-out *gain*
requirement, honest small-sample reporting -- and the budget, few-shot and
reproducibility fixes that make the search spend its evaluations on real
comparisons.

Everything is offline and deterministic: "noisy" agents draw from a seeded RNG.
"""

from __future__ import annotations

import logging
import random

import pytest

from adapt_agent.optimization.dataset import Example, GoldenDataset
from adapt_agent.optimization.evaluation import EvaluationHarness, EvaluationReport, ExampleResult
from adapt_agent.optimization.judge import LLMJudge
from adapt_agent.optimization.metrics import Metric, exact_match
from adapt_agent.optimization.optimizers import (
    DEFAULT_NONDETERMINISTIC_REPEATS,
    BootstrapFewShotOptimizer,
    CoordinateAscentOptimizer,
    EvolutionaryOptimizer,
    GridSearchOptimizer,
    OptimizationResult,
    PipelineOptimizer,
    RandomSearchOptimizer,
    make_default_optimizer,
)
from adapt_agent.optimization.parameters import Parameter, ParameterKind
from adapt_agent.optimization.proposers import LLMProposer, ProposalContext
from adapt_agent.optimization.significance import (
    compare_rows,
    min_rows_for_significance,
    row_scores,
    sign_flip_p_value,
)
from adapt_agent.optimization.target import OptimizableAgent

# -- helpers ------------------------------------------------------------------


def knob(state: dict, name: str, kind=ParameterKind.HYPERPARAM, **space) -> Parameter:
    return Parameter(
        name,
        kind,
        getter=lambda: state[name],
        setter=lambda v: state.__setitem__(name, v),
        **space,
    )


def rows(n: int, prefix: str = "q", expected: str = "yes") -> GoldenDataset:
    return GoldenDataset([Example(inputs=f"{prefix}{i}", expected=expected) for i in range(n)])


def harness() -> EvaluationHarness:
    return EvaluationHarness(exact_match())


def noisy_no_effect_agent(seed: int, p_correct: float = 0.7):
    """A knob with no causal effect on an agent that is right ``p_correct`` of the time."""
    rng = random.Random(1000 + seed)
    state = {"a.temperature": 0.0}
    candidates = [round(0.1 * i, 1) for i in range(11)]
    agent = OptimizableAgent.from_callable(
        lambda x: "yes" if rng.random() < p_correct else "no",
        parameters=[knob(state, "a.temperature", candidates=candidates)],
    )
    return state, agent


# -- the paired test itself ---------------------------------------------------


def test_the_smallest_achievable_p_value_is_half_to_the_power_of_n():
    assert sign_flip_p_value([1.0] * 4) == pytest.approx(1 / 16)
    assert sign_flip_p_value([1.0] * 5) == pytest.approx(1 / 32)
    assert min_rows_for_significance(0.05) == 5
    assert min_rows_for_significance(0.0625) == 4


def test_ties_carry_no_evidence_and_no_change_is_p_one():
    assert sign_flip_p_value([0.0, 0.0, 0.0]) == 1.0
    # Zeros neither help nor hurt: same p as the two non-zero rows alone.
    assert sign_flip_p_value([1.0, 1.0, 0.0, 0.0]) == pytest.approx(sign_flip_p_value([1, 1]))
    # A mixed result is weak evidence.
    assert sign_flip_p_value([1.0, 1.0, -1.0]) == pytest.approx(0.5)


def test_large_continuous_runs_use_a_reproducible_approximation():
    rng = random.Random(3)
    deltas = [rng.uniform(-0.3, 0.6) for _ in range(60)]
    first = sign_flip_p_value(deltas, seed=1)
    assert first == sign_flip_p_value(deltas, seed=1)
    assert 0.0 <= first < 0.01  # a clearly positive shift
    medium = [rng.uniform(-1, 1) for _ in range(22)]  # Monte Carlo branch
    assert sign_flip_p_value(medium, seed=5) == sign_flip_p_value(medium, seed=5)


def test_compare_rows_pairs_only_rows_scored_on_both_sides_and_not_excluded():
    incumbent = [0.0, 0.0, None, 1.0, 0.0, 0.0]
    challenger = [1.0, 1.0, 1.0, None, 1.0, 1.0]
    result = compare_rows(incumbent, challenger, exclude={5}, interval=True)
    assert result.n_rows == 3  # rows 0, 1, 4
    assert result.delta == 1.0 and result.n_better == 3 and result.n_worse == 0
    assert result.decidable is False and result.significant is False
    assert result.ci_low == result.ci_high == 1.0
    assert "cannot reach significance" in result.describe()


def test_a_significant_result_needs_enough_rows_a_small_p_and_a_big_enough_effect():
    sweep = compare_rows([0.0] * 6, [1.0] * 6)
    assert sweep.significant is True
    tiny = compare_rows([0.0] * 6, [0.0001] * 6, min_effect=0.001)
    assert tiny.p_value < 0.05 and tiny.significant is False


def test_row_scores_refuses_a_truncated_report_and_blanks_throttled_rows():
    def result(index, score, **kw):
        return ExampleResult(index, f"q{index}", "o", "e", {"m": score}, 0.0, **kw)

    full = EvaluationReport({"m": 0.5}, "m", [result(0, 1.0), result(1, 0.0, transient=True)])
    assert row_scores(full) == [1.0, None]
    truncated = EvaluationReport({"m": 1.0}, "m", [result(0, 1.0)], n_evaluated=3)
    assert row_scores(truncated) is None
    assert row_scores(EvaluationReport({"m": 1.0}, "m")) is None


# -- finding 1: noise is not an improvement ------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda s: GridSearchOptimizer(harness(), seed=s),
        lambda s: CoordinateAscentOptimizer(harness(), seed=s),
        lambda s: RandomSearchOptimizer(harness(), seed=s, max_evals=20),
    ],
    ids=["grid", "coordinate_ascent", "random"],
)
def test_a_knob_with_no_effect_is_almost_never_reported_as_an_improvement(build):
    runs = 20
    false_accepts = 0
    applied = 0
    for seed in range(runs):
        _, agent = noisy_no_effect_agent(seed)
        result = build(seed).optimize(agent, rows(10))
        false_accepts += result.improved
        applied += bool(result.best_config)
        assert result.nondeterministic is True
        assert result.repeats == DEFAULT_NONDETERMINISTIC_REPEATS
    # Before: 31/40 grid runs claimed a gain. The rate must now sit at or
    # under alpha (0.05) within noise: at most 2 of 20.
    assert false_accepts <= 2
    assert applied <= 2


def test_a_no_op_config_is_not_applied_through_the_validation_gate():
    applied = 0
    for seed in range(20):
        _, agent = noisy_no_effect_agent(seed)
        result = GridSearchOptimizer(harness(), seed=seed).optimize(
            agent, rows(10), val_dataset=rows(10, prefix="v")
        )
        applied += bool(result.best_config)
    # Before: 16/40 -- the gate only rejected regressions, and a no-op does
    # not regress.
    assert applied <= 2


def test_a_real_effect_on_a_noisy_agent_is_still_found():
    rng = random.Random(11)
    state = {"a.mode": "plain"}
    agent = OptimizableAgent.from_callable(
        lambda x: "yes" if rng.random() < (0.95 if state["a.mode"] == "careful" else 0.4) else "no",
        parameters=[knob(state, "a.mode", candidates=["plain", "careful"])],
    )
    result = GridSearchOptimizer(harness(), seed=0).optimize(agent, rows(30))
    assert result.nondeterministic is True
    assert result.significant is True and result.improved is True
    assert result.best_config == {"a.mode": "careful"}
    assert result.p_value is not None and result.p_value <= 0.05
    assert result.ci_low is not None and result.ci_low > 0


def test_a_single_row_flip_is_applied_but_not_called_an_improvement():
    state = {"a.t": 0}
    agent = OptimizableAgent.from_callable(
        lambda x: str(state["a.t"]), parameters=[knob(state, "a.t", candidates=[0, 1, 2])]
    )
    result = GridSearchOptimizer(harness()).optimize(agent, rows(1, expected="2"))
    assert result.best_config == {"a.t": 2}  # point estimate, small-sample mode
    assert result.best_score > result.baseline_score
    assert result.improved is False and result.significant is False
    assert result.n_rows == 1 and result.p_value == 0.5
    assert any(r.startswith("Small sample:") for r in result.recommendations)


def test_small_samples_can_be_refused_outright():
    state = {"a.t": 0}
    agent = OptimizableAgent.from_callable(
        lambda x: str(state["a.t"]), parameters=[knob(state, "a.t", candidates=[0, 1, 2])]
    )
    result = GridSearchOptimizer(harness(), allow_small_sample=False).optimize(
        agent, rows(4, expected="2")
    )
    assert result.best_config == {} and state["a.t"] == 0
    assert all(not t.accepted for t in result.history)


def test_the_result_carries_the_evidence():
    state = {"a.t": 0}
    agent = OptimizableAgent.from_callable(
        lambda x: str(state["a.t"]), parameters=[knob(state, "a.t", candidates=[0, 2])]
    )
    result = GridSearchOptimizer(harness()).optimize(agent, rows(8, expected="2"))
    assert result.improved and result.significant
    assert result.delta == 1.0 and result.n_rows == 8 and result.repeats == 1
    assert result.p_value == pytest.approx(1 / 256)
    assert result.alpha == 0.05 and result.significance_basis == "training"
    as_dict = result.to_dict()
    for key in ("significant", "delta", "p_value", "ci_low", "ci_high", "n_rows", "repeats"):
        assert key in as_dict
    accepted = [t for t in result.history if t.accepted]
    assert accepted and accepted[0].significant is True and accepted[0].n_rows == 8
    assert "significant=True" in repr(result)


def test_improved_is_false_whenever_the_gain_was_not_significant():
    result = OptimizationResult(
        best_config={"a": 1}, best_score=1.0, baseline_score=0.0, significant=False
    )
    assert result.improved is False
    # A hand-built result with no evidence keeps the arithmetic meaning.
    assert OptimizationResult(best_config={}, best_score=1.0, baseline_score=0.0).improved


def test_the_export_header_records_the_verdict(tmp_path):
    state = {"a.t": 0}
    agent = OptimizableAgent.from_callable(
        lambda x: str(state["a.t"]), parameters=[knob(state, "a.t", candidates=[0, 2])]
    )
    result = GridSearchOptimizer(harness()).optimize(agent, rows(8, expected="2"))
    path = tmp_path / "tuned.yaml"
    result.to_config(path)
    assert "# significant=True p=0.003906 n_rows=8 repeats=1 basis=training." in path.read_text()


# -- repeats and non-determinism detection ------------------------------------


def _counting_noisy_agent(p_by_mode: dict[str, float], seed: int = 0):
    rng = random.Random(seed)
    state = {"a.mode": "x"}
    calls = {"n": 0}

    def run(x):
        calls["n"] += 1
        return "yes" if rng.random() < p_by_mode[state["a.mode"]] else "no"

    agent = OptimizableAgent.from_callable(
        run, parameters=[knob(state, "a.mode", candidates=list(p_by_mode))]
    )
    return calls, agent


def test_a_deterministic_target_is_measured_once_per_configuration():
    state = {"a.t": 0}
    agent = OptimizableAgent.from_callable(
        lambda x: str(state["a.t"]), parameters=[knob(state, "a.t", candidates=[0, 1])]
    )
    result = GridSearchOptimizer(harness()).optimize(agent, rows(6, expected="1"))
    assert result.nondeterministic is False and result.repeats == 1


def test_repeats_are_real_re_runs_interleaved_with_the_incumbent():
    calls, agent = _counting_noisy_agent({"x": 0.5, "y": 0.5})
    n = 6
    GridSearchOptimizer(harness(), repeats=2).optimize(agent, rows(n))
    # baseline (1 pass, no detection when repeats is explicit) + two trials
    # ({x}: the baseline state, {y}) x 2 repeats x (incumbent + challenger)
    # + the final verdict: 2 repeats x (tuned + untuned) when something won.
    trial_passes = 2 * 2 * 2
    assert calls["n"] in {n * (1 + trial_passes), n * (1 + trial_passes + 4)}


def test_detection_re_runs_the_baseline_once_and_raises_repeats(caplog):
    calls, agent = _counting_noisy_agent({"x": 0.5})
    with caplog.at_level(logging.WARNING, logger="adapt_agent.optimization.optimizers"):
        result = GridSearchOptimizer(harness()).optimize(agent, rows(10))
    assert result.nondeterministic is True
    assert result.repeats == DEFAULT_NONDETERMINISTIC_REPEATS
    assert calls["n"] == 2 * 10  # one knob value only: baseline + its re-run
    assert any("non-deterministic" in r.message for r in caplog.records)


def test_detection_can_be_switched_off():
    calls, agent = _counting_noisy_agent({"x": 0.5})
    result = GridSearchOptimizer(harness(), detect_nondeterminism=False).optimize(agent, rows(10))
    assert result.nondeterministic is False and calls["n"] == 10


# -- the held-out requirement -------------------------------------------------


def _prompt_agent(right_on):
    """Prompt GOOD is right everywhere; BASE only on rows in ``right_on``."""
    state = {"a.prompt": "BASE"}
    agent = OptimizableAgent.from_callable(
        lambda x: "yes" if state["a.prompt"] == "GOOD" or x in right_on else "no",
        parameters=[knob(state, "a.prompt", ParameterKind.PROMPT, candidates=["BASE", "GOOD"])],
    )
    return state, agent


def test_a_config_that_only_ties_on_held_out_data_is_not_applied():
    # Wins every training row, but the untuned agent is already perfect on the
    # held-out rows: there is no held-out gain, only an absence of regression.
    held_out = rows(6, prefix="v")
    state, agent = _prompt_agent(right_on={f"v{i}" for i in range(6)})
    result = GridSearchOptimizer(harness()).optimize(agent, rows(8), val_dataset=held_out)
    assert state["a.prompt"] == "BASE"
    assert result.validation_rejected is True
    assert result.rejected_config == {"a.prompt": "GOOD"}
    assert result.rejected_reason == "no gain on the held-out set"
    assert result.improved is False and result.significance_basis == "held_out"


def test_a_held_out_gain_is_applied_and_judged_on_the_held_out_rows():
    state, agent = _prompt_agent(right_on=set())
    result = GridSearchOptimizer(harness()).optimize(agent, rows(8), val_dataset=rows(6, "v"))
    assert state["a.prompt"] == "GOOD"
    assert result.significance_basis == "held_out" and result.n_rows == 6
    assert result.significant is True and result.improved is True


def test_a_large_golden_set_is_split_automatically():
    state, agent = _prompt_agent(right_on=set())
    result = GridSearchOptimizer(harness()).optimize(agent, rows(40))
    assert result.significance_basis == "held_out_auto"
    assert result.validation_score == 1.0 and result.validation_baseline_score == 0.0
    assert result.n_rows == 10
    assert state["a.prompt"] == "GOOD"


@pytest.mark.parametrize("n_rows, kw", [(37, {}), (40, {"holdout_fraction": 0.0})])
def test_no_automatic_split_when_too_small_or_disabled(n_rows, kw):
    _, agent = _prompt_agent(right_on=set())
    result = GridSearchOptimizer(harness(), **kw).optimize(agent, rows(n_rows))
    assert result.significance_basis == "training"
    assert result.validation_score is None
    assert any("training rows the search selected on" in r for r in result.recommendations)


def test_an_empty_validation_set_is_refused_rather_than_silently_skipping_the_gate():
    _, agent = _prompt_agent(right_on=set())
    with pytest.raises(ValueError, match="val_dataset is empty"):
        GridSearchOptimizer(harness()).optimize(agent, rows(8), val_dataset=GoldenDataset([]))


def test_a_held_out_pass_with_no_surviving_row_is_no_evidence_of_a_gain():
    from adapt_agent.optimization.retry import RetryPolicy

    state = {"a.prompt": "BASE"}

    def run(x):
        if x.startswith("v"):
            raise RuntimeError("429 Too Many Requests")
        return "yes" if state["a.prompt"] == "GOOD" else "no"

    agent = OptimizableAgent.from_callable(
        run, parameters=[knob(state, "a.prompt", ParameterKind.PROMPT, candidates=["BASE", "GOOD"])]
    )
    h = EvaluationHarness(exact_match(), retry=RetryPolicy(attempts=1))
    result = GridSearchOptimizer(h).optimize(agent, rows(8), val_dataset=rows(3, "v"))
    assert result.validation_rejected is True and state["a.prompt"] == "BASE"
    assert result.validation_complete is False


# -- finding 2: no knob is silently skipped -----------------------------------


def test_the_default_pipeline_tunes_temperature_beside_a_model_pool():
    state = {"a.model": "m0", "a.temperature": 0.0}
    agent = OptimizableAgent.from_callable(
        lambda x: "ok" if state["a.temperature"] == 1.0 else "bad",
        parameters=[
            knob(state, "a.model", ParameterKind.MODEL, candidates=["m0", "m1", "m2", "m3"]),
            knob(state, "a.temperature", bounds=(0.0, 1.0)),
        ],
    )
    result = make_default_optimizer(harness(), max_evals=60).optimize(agent, rows(6, expected="ok"))
    varied = {k for t in result.history for k in t.config}
    assert varied == {"a.model", "a.temperature"}
    assert result.best_score == 1.0 and state["a.temperature"] == 1.0
    assert result.untuned_parameters == []


def test_a_sixteen_candidate_knob_is_not_skipped():
    state = {"a.model": "m0"}
    agent = OptimizableAgent.from_callable(
        lambda x: "ok" if state["a.model"] == "m7" else "bad",
        parameters=[
            knob(state, "a.model", ParameterKind.MODEL, candidates=[f"m{i}" for i in range(16)])
        ],
    )
    result = make_default_optimizer(harness(), max_evals=60).optimize(agent, rows(6, expected="ok"))
    assert state["a.model"] == "m7" and result.n_evals >= 8


def test_a_truncated_grid_sweeps_every_knob_instead_of_the_first():
    state = {"a.x": 0, "a.y": 0}
    agent = OptimizableAgent.from_callable(
        lambda x: "ok" if state["a.x"] == 4 and state["a.y"] == 3 else "bad",
        parameters=[
            knob(state, "a.x", candidates=[0, 1, 2, 3, 4]),
            knob(state, "a.y", candidates=[0, 1, 2, 3, 4]),
        ],
    )
    # Fixing only one knob gains nothing here, so no single change wins:
    # the point is that both knobs were *tried*.
    result = GridSearchOptimizer(harness(), max_evals=10).optimize(agent, rows(6, expected="ok"))
    tried_x = {t.config["a.x"] for t in result.history if "a.x" in t.config}
    tried_y = {t.config["a.y"] for t in result.history if "a.y" in t.config}
    assert tried_x == {1, 2, 3, 4} and tried_y == {1, 2, 3, 4}
    assert result.n_evals == 8 and result.untuned_parameters == []


def test_a_knob_left_untried_is_named_in_a_warning_and_on_the_result(caplog):
    state = {"a.x": 0, "a.y": 0, "a.z": 0}
    agent = OptimizableAgent.from_callable(
        lambda x: "bad",
        parameters=[knob(state, name, candidates=[0, 1, 2]) for name in ("a.x", "a.y", "a.z")],
    )
    with caplog.at_level(logging.WARNING, logger="adapt_agent.optimization.optimizers"):
        result = GridSearchOptimizer(harness(), max_evals=2).optimize(agent, rows(3))
    assert result.untuned_parameters == ["a.z"]
    assert any("a.z received zero evaluations" in r.message for r in caplog.records)
    assert any("Not tuned: a.z" in r for r in result.recommendations)


def test_budget_a_stage_leaves_unused_goes_to_the_stages_after_it():
    state = {"a.m": "m0"}
    agent = OptimizableAgent.from_callable(
        lambda x: "ok" if state["a.m"] == "m9" else "bad",
        parameters=[
            knob(state, "a.m", ParameterKind.MODEL, candidates=[f"m{i}" for i in range(10)])
        ],
    )
    stages = [
        BootstrapFewShotOptimizer(harness(), max_evals=5),  # nothing to do: no few-shot knob
        GridSearchOptimizer(harness(), max_evals=5, max_configs=64),
    ]
    result = PipelineOptimizer(harness(), stages, max_evals=10).optimize(
        agent, rows(6, expected="ok")
    )
    assert result.n_evals == 10  # the grid got the few-shot stage's share too
    assert state["a.m"] == "m9"
    assert [s.max_evals for s in stages] == [5, 5], "stage budgets are restored afterwards"


# -- finding 3: few-shot demonstrations are not scored on their own rows -------


def test_bootstrap_cannot_win_by_copying_answers_into_the_prompt():
    state = {"a.few_shot": ""}

    def lookup(x):  # can only answer what is literally in its demonstration block
        key = f"Input: {x}\nOutput: "
        block = state["a.few_shot"]
        return block.split(key)[1].split("\n")[0] if key in block else "dunno"

    agent = OptimizableAgent.from_callable(
        lookup, parameters=[knob(state, "a.few_shot", ParameterKind.FEW_SHOT)]
    )
    train = GoldenDataset([Example(inputs=f"q{i}", expected=f"a{i}") for i in range(10)])
    val = GoldenDataset([Example(inputs=f"v{i}", expected=f"b{i}") for i in range(10)])
    result = BootstrapFewShotOptimizer(harness()).optimize(agent, train, val_dataset=val)
    # Before: train 0.0 -> 0.3, validation 0.0 -> 0.0, applied, improved=True.
    assert result.best_config == {} and state["a.few_shot"] == ""
    assert result.improved is False
    assert all(t.delta == 0.0 for t in result.history)


# -- finding 4: budget, reproducibility, proposer calls, edge cases -----------


def test_evolutionary_search_never_pays_twice_for_one_configuration():
    state = {"a.t": 0}
    agent = OptimizableAgent.from_callable(
        lambda x: str(state["a.t"]), parameters=[knob(state, "a.t", candidates=list(range(10)))]
    )
    result = EvolutionaryOptimizer(harness(), max_evals=12).optimize(agent, rows(4, expected="5"))
    keys = [repr(sorted(t.config.items())) for t in result.history]
    assert len(keys) == len(set(keys)), "a survivor or duplicate child consumed budget"
    assert {} not in [t.config for t in result.history], "the baseline was re-recorded"


def test_one_instance_reproduces_its_trials_on_every_run():
    def run_once(opt):
        state = {"a.t": 0.0}
        agent = OptimizableAgent.from_callable(
            lambda x: "y" if 0.4 < state["a.t"] < 0.6 else "n",
            parameters=[knob(state, "a.t", bounds=(0.0, 1.0))],
        )
        return [t.config for t in opt.optimize(agent, rows(1, expected="y")).history]

    opt = RandomSearchOptimizer(harness(), seed=0, max_evals=6)
    first = run_once(opt)
    assert run_once(opt) == first
    assert run_once(RandomSearchOptimizer(harness(), seed=0, max_evals=6)) == first


def _judge_counting_rewrites():
    calls: list[str] = []

    def complete(prompt, *, system=None, **_kw):
        if "Rewrite the instruction" in f"{system}\n{prompt}":
            calls.append(prompt)
            return "BETTER"
        return "critique"

    return calls, LLMJudge(complete)


def _report(n_failing: int) -> EvaluationReport:
    results = [
        ExampleResult(i, f"q{i}", "wrong", "right", {"exact_match": 0.0}, 0.0)
        for i in range(n_failing)
    ]
    return EvaluationReport({"exact_match": 0.0}, "exact_match", results)


def _prompt_ctx(report, judge):
    state = {"p": "Do the task."}
    param = knob(state, "p", ParameterKind.PROMPT)
    agent = OptimizableAgent.from_callable(lambda x: x, parameters=[param])
    return ProposalContext(
        parameter=param, agent=agent, dataset=rows(1), report=report, judge=judge,
        rng=random.Random(0),
    )  # fmt: skip


def test_the_llm_proposer_does_not_send_the_same_rewrite_request_twice():
    calls, judge = _judge_counting_rewrites()
    out = LLMProposer(judge, rewrites=2).propose(_prompt_ctx(_report(1), judge))
    assert out == ["BETTER"]
    assert len(calls) == 1


def test_the_llm_proposer_varies_the_failures_behind_each_rewrite():
    calls, judge = _judge_counting_rewrites()
    LLMProposer(judge, rewrites=2, critique_samples=2).propose(_prompt_ctx(_report(6), judge))
    assert len(calls) == 2 and calls[0] != calls[1]


@pytest.mark.parametrize("bad", [0, -1, True, 2.5])
def test_budgets_are_validated_up_front(bad):
    with pytest.raises(ValueError, match="max_evals"):
        GridSearchOptimizer(harness(), max_evals=bad)
    with pytest.raises(ValueError, match="max_evals"):
        make_default_optimizer(harness(), max_evals=bad)


@pytest.mark.parametrize(
    "kw", [{"repeats": 0}, {"alpha": 0.0}, {"alpha": 1.0}, {"holdout_fraction": 1.0}]
)
def test_other_measurement_options_are_validated(kw):
    with pytest.raises(ValueError):
        GridSearchOptimizer(harness(), **kw)


def test_a_budget_of_one_runs_exactly_one_trial():
    state = {"a.t": 0}
    agent = OptimizableAgent.from_callable(
        lambda x: str(state["a.t"]), parameters=[knob(state, "a.t", candidates=[0, 1, 2])]
    )
    assert GridSearchOptimizer(harness(), max_evals=1).optimize(agent, rows(3)).n_evals == 1


def test_all_equal_scores_change_nothing():
    state = {"a.t": 0}
    agent = OptimizableAgent.from_callable(
        lambda x: "nope", parameters=[knob(state, "a.t", candidates=[0, 1, 2])]
    )
    result = GridSearchOptimizer(harness()).optimize(agent, rows(6))
    assert result.best_config == {} and result.improved is False
    assert all(t.delta == 0.0 and not t.accepted for t in result.history)


def test_a_metric_that_always_raises_changes_nothing():
    state = {"a.t": 0}
    agent = OptimizableAgent.from_callable(
        lambda x: str(state["a.t"]), parameters=[knob(state, "a.t", candidates=[0, 1, 2])]
    )
    boom = Metric("boom", lambda output, expected: 1 / 0)
    result = GridSearchOptimizer(EvaluationHarness([boom])).optimize(agent, rows(6, expected="2"))
    assert result.best_config == {} and result.improved is False


def test_a_candidate_that_crashes_on_some_rows_is_not_crowned():
    state = {"a.t": 0}

    def run(x):
        if state["a.t"] == 2 and x in ("q1", "q2", "q3"):
            raise ValueError("candidate bug")  # permanent: counts against it
        return "ok" if state["a.t"] == 2 else "bad"

    agent = OptimizableAgent.from_callable(run, parameters=[knob(state, "a.t", candidates=[0, 2])])
    result = GridSearchOptimizer(harness()).optimize(agent, rows(6, expected="ok"))
    # Right on 3 rows, crashed on 3: a gain on half the rows is not significant.
    assert result.best_config == {} and state["a.t"] == 0
