"""Example 13: Every optimizer strategy, on one agent.

Example 06 tunes a single prompt with one strategy. A real agent has several
kinds of knob -- prompt, model, temperature, few-shot examples -- and no single
search strategy suits them all. This example puts five kinds of knob on one
ticket-triage agent and runs each strategy against it.

1.  Declaring the search space: ``Parameter`` kinds, candidates and bounds.
2.  ``GridSearchOptimizer``      -- exhaustive over small discrete spaces.
3.  ``RandomSearchOptimizer``    -- a fixed budget over a larger space.
4.  ``CoordinateAscentOptimizer``-- one knob at a time, keeping improvements.
5.  ``BootstrapFewShotOptimizer``-- build demonstrations from rows it gets right.
6.  ``EvolutionaryOptimizer``    -- mutate and select a population.
7.  ``make_default_optimizer``   -- a staged pipeline that does all of it.
8.  The validation gate refusing a configuration that only memorised training,
    and a noisy agent whose lucky draws are not mistaken for an improvement.
9.  ``to_config`` / ``load_tuned_config`` -- review the result, then reload it.
10. ``AgentOptimizer`` -- runtime performance hints (a different thing entirely).

Every optimizer accepts a change only when its per-row gain over the current
best is statistically significant (a paired test), so the dataset has to be
large enough for a real gain to show -- see ``TRAIN_TICKETS``.

Everything is deterministic and offline: no API key, no network.

    python examples/13_optimizer_strategies.py
"""

import logging
import random
import tempfile
from pathlib import Path

from adapt_agent import AgentOptimizer
from adapt_agent.optimization import (
    BootstrapFewShotOptimizer,
    CoordinateAscentOptimizer,
    EvaluationHarness,
    EvolutionaryOptimizer,
    GoldenDataset,
    GridSearchOptimizer,
    OptimizableAgent,
    Parameter,
    ParameterKind,
    RandomSearchOptimizer,
    exact_match,
    load_tuned_config,
    make_default_optimizer,
)

logging.getLogger("adapt_agent").setLevel(logging.ERROR)

#: Eighteen training tickets and nine held out, three labels each. The size is
#: not decoration: every optimizer accepts a change only when its per-row gain
#: is statistically significant, and a knob that fixes one label fixes a third
#: of the rows -- two rows out of six is a coin-flip-sized gain (p=0.25), six
#: out of eighteen is not (p=0.016).
TRAIN_TICKETS = {
    "I was charged twice this month": "billing",
    "Refund has not arrived": "billing",
    "Where is my parcel?": "shipping",
    "The tracking link is dead": "shipping",
    "App crashes when I log in": "technical",
    "Password reset email never comes": "technical",
    "My card was declined at checkout": "billing",
    "I was billed after cancelling": "billing",
    "The coupon was not applied to my order": "billing",
    "Billing address cannot be updated": "billing",
    "The parcel arrived damaged": "shipping",
    "Delivery is a week late": "shipping",
    "Wrong item in the box": "shipping",
    "Can I change the delivery address?": "shipping",
    "The page is blank after the update": "technical",
    "Notifications stopped working": "technical",
    "Export to CSV throws an error": "technical",
    "The app will not install on my phone": "technical",
}
HELD_OUT_TICKETS = {
    "Invoice shows the wrong VAT": "billing",
    "I need a receipt for my last payment": "billing",
    "Why did the price go up?": "billing",
    "Courier left it with a neighbour": "shipping",
    "The parcel says delivered but it is not here": "shipping",
    "Do you ship to Norway?": "shipping",
    "Two-factor code is rejected": "technical",
    "Search returns nothing": "technical",
    "Dark mode resets every time": "technical",
}
TICKETS = {**TRAIN_TICKETS, **HELD_OUT_TICKETS}
TRAIN = list(TRAIN_TICKETS)
HELD_OUT = list(HELD_OUT_TICKETS)

BASELINE = {
    "prompt": "Help the customer.",
    "model": "small",
    "temperature": 1.0,
    "few_shot": "",
}


HONEST_PROMPTS = ["Help the customer.", "Classify the ticket. Reply with a single label."]


def build_agent(
    *, memorised_prompt: bool = False, start: dict | None = None
) -> tuple[dict, OptimizableAgent]:
    """A triage agent whose behaviour depends on four live knobs.

    The "model" is a deterministic stand-in with realistic failure modes:

    * without a prompt asking for a single label it rambles;
    * above temperature 0.5 it mislabels shipping tickets;
    * the small model cannot recognise technical tickets unless it is shown
      some worked examples first;
    * one prompt has simply memorised the training tickets.
    """
    state = {**BASELINE, **(start or {})}

    def run(ticket: str) -> str:
        label = TICKETS[ticket]
        prompt = state["prompt"]
        if prompt == "MEMORISED ANSWERS":
            return label if ticket in TRAIN else "billing"
        if "single label" not in prompt.lower():
            return f"This looks like a {label} problem to me."
        if state["temperature"] > 0.5 and label == "shipping":
            return "billing"
        if label == "technical" and state["model"] == "small" and not state["few_shot"]:
            return "billing"
        return label

    def knob(name: str, kind: ParameterKind, **space) -> Parameter:
        return Parameter(
            name=f"triage.{name}",
            kind=kind,
            getter=lambda: state[name],
            setter=lambda value: state.__setitem__(name, value),
            component="triage",
            **space,
        )

    parameters = [
        knob(
            "prompt",
            ParameterKind.PROMPT,
            # Section 8 adds a candidate that has memorised the training rows.
            candidates=HONEST_PROMPTS + (["MEMORISED ANSWERS"] if memorised_prompt else []),
        ),
        knob("model", ParameterKind.MODEL, candidates=["small", "large"]),
        knob("temperature", ParameterKind.HYPERPARAM, bounds=(0.0, 1.0), step=0.25),
        knob("few_shot", ParameterKind.FEW_SHOT),
    ]
    return state, OptimizableAgent.from_callable(run, parameters=parameters, name="triage")


def dataset(tickets: list[str]) -> GoldenDataset:
    return GoldenDataset.from_list([{"input": t, "expected": TICKETS[t]} for t in tickets])


def harness() -> EvaluationHarness:
    return EvaluationHarness(exact_match())


def show(label: str, result, state: dict) -> None:
    changed = {k.split(".")[1]: v for k, v in result.best_config.items()}
    if not changed.get("few_shot"):
        changed.pop("few_shot", None)  # an empty block is no change worth printing
    else:
        changed["few_shot"] = f"<{len(str(changed['few_shot']).splitlines())}-line block>"
    if "temperature" in changed:
        changed["temperature"] = round(changed["temperature"], 2)
    print(
        f"  {label:22} {result.baseline_score:.2f} -> {result.best_score:.2f} "
        f"in {result.n_evals:2d} evals  changed={changed}"
    )


def search_space() -> None:
    print("== 1. The search space ==")
    _, agent = build_agent()
    for p in agent.parameters:
        space = p.candidates or p.bounds or "bootstrapped from the dataset"
        print(f"  {p.name:19} {p.kind.value:10} {space}")
    print(f"  baseline score: {harness().evaluate(agent, dataset(TRAIN)).score:.2f}")


def strategies() -> None:
    print("\n== 2-7. One agent, six strategies (each starts from the same baseline) ==")
    train = dataset(TRAIN)
    runs = {
        "grid": lambda: GridSearchOptimizer(harness(), seed=0, max_evals=40),
        "random": lambda: RandomSearchOptimizer(harness(), seed=3, max_evals=12),
        "coordinate ascent": lambda: CoordinateAscentOptimizer(harness(), seed=0, max_evals=30),
        "bootstrap few-shot": lambda: BootstrapFewShotOptimizer(harness(), seed=0, max_evals=10),
        "evolutionary": lambda: EvolutionaryOptimizer(
            harness(), seed=0, max_evals=40, population=4, generations=4
        ),
        "default pipeline": lambda: make_default_optimizer(harness(), seed=0, max_evals=120),
    }
    for label, build in runs.items():
        state, agent = build_agent()
        result = build().optimize(agent, train)
        show(label, result, state)
    print(
        "  Few-shot alone gets nowhere from here: it builds demonstrations out of rows the\n"
        "  agent already answers correctly, and this baseline answers none."
    )

    # Where few-shot earns its place: the prompt and temperature are already
    # right and you cannot afford the large model. Demonstrations close the gap.
    state, agent = build_agent(start={"prompt": HONEST_PROMPTS[1], "temperature": 0.0})
    next(p for p in agent.parameters if p.name == "triage.model").mutable = False
    result = BootstrapFewShotOptimizer(harness(), seed=0, max_evals=10).optimize(agent, train)
    show("few-shot, small model", result, state)


def validation_gate() -> None:
    print("\n== 8. The validation gate ==")
    train, held_out = dataset(TRAIN), dataset(HELD_OUT)

    # Start from a decent agent: right prompt and temperature, but the small
    # model, so it misses technical tickets (4/6 on training, 2/3 held out).
    # Only the prompt is searched, and one candidate has memorised the six
    # training tickets: 6/6 there, and wrong on anything it has not seen.
    decent = {"prompt": HONEST_PROMPTS[1], "temperature": 0.0}

    def prompt_only() -> CoordinateAscentOptimizer:
        return CoordinateAscentOptimizer(harness(), seed=0, kinds=(ParameterKind.PROMPT,))

    state, agent = build_agent(memorised_prompt=True, start=decent)
    result = prompt_only().optimize(agent, train)
    print(
        f"  no held-out data : train {result.baseline_score:.2f} -> {result.best_score:.2f}, "
        f"applied {state['prompt']!r}"
    )

    state, agent = build_agent(memorised_prompt=True, start=decent)
    result = prompt_only().optimize(agent, train, val_dataset=held_out)
    print(
        f"  with held-out    : rejected={result.validation_rejected}; held-out score was "
        f"{result.validation_baseline_score:.2f}, kept"
    )
    print(f"  prompt left as   : {state['prompt']!r}")
    print(f"  what was refused : {result.rejected_config}")


def noise_is_not_an_improvement() -> None:
    print("\n== 8b. A knob that does nothing, on an agent that is right 70% of the time ==")
    rng = random.Random(7)
    state = {"temperature": 0.0}
    agent = OptimizableAgent.from_callable(
        lambda ticket: TICKETS[ticket] if rng.random() < 0.7 else "billing",
        parameters=[
            Parameter(
                name="triage.temperature",
                kind=ParameterKind.HYPERPARAM,
                candidates=[0.0, 0.25, 0.5, 0.75, 1.0],
                getter=lambda: state["temperature"],
                setter=lambda value: state.__setitem__("temperature", value),
            )
        ],
    )
    # The optimizer re-runs the baseline, sees rows change, and measures every
    # comparison three times. Whatever the scores wobble to, nothing is applied.
    result = GridSearchOptimizer(harness(), seed=0).optimize(agent, dataset(TRAIN))
    print(f"  non-deterministic target detected : {result.nondeterministic}")
    print(f"  measurements per comparison       : {result.repeats}")
    print(
        f"  improved={result.improved}  significant={result.significant}  applied={result.best_config}"
    )


def export_and_reload() -> None:
    print("\n== 9. Export for review, reload elsewhere ==")
    state, agent = build_agent()
    result = GridSearchOptimizer(harness(), seed=0, max_evals=40).optimize(
        agent, dataset(TRAIN), val_dataset=dataset(HELD_OUT)
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tuned.yaml"
        result.to_config(path)  # the file you diff, review and commit
        print("  " + path.read_text().strip().replace("\n", "\n  "))

        # A fresh process later: build the agent, apply the reviewed file.
        fresh_state, fresh_agent = build_agent()
        fresh_agent.apply(load_tuned_config(path))
        score = harness().evaluate(fresh_agent, dataset(HELD_OUT)).score
        print(f"  reloaded into a fresh agent -> held-out score {score:.2f}")

    accepted = [t for t in result.history if t.accepted]
    print(f"  history: {len(result.history)} trials, {len(accepted)} accepted")


def runtime_performance() -> None:
    print("\n== 10. AgentOptimizer: runtime hints, not configuration search ==")
    perf = AgentOptimizer()
    for seconds, tokens, ok in [(0.8, 900, True), (6.5, 7200, True), (7.1, 8100, False)]:
        perf.analyze_performance("triage", execution_time=seconds, token_usage=tokens, success=ok)
    for hint in perf.suggest_optimizations("triage"):
        print(f"  {hint.get('type')}: {hint.get('suggestion')}")


def main() -> None:
    search_space()
    strategies()
    validation_gate()
    noise_is_not_an_improvement()
    export_and_reload()
    runtime_performance()


if __name__ == "__main__":
    main()
