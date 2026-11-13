"""Example 06: Optimize a multi-agent system against a golden dataset.

This shows the full ADAPT-Agent optimization loop end to end:

1.  A **golden dataset** of inputs and expected outputs.
2.  An **OptimizableAgent** that wraps your agent code -- here a tiny
    "orchestrator + two specialists" system whose behaviour depends on a tunable
    prompt. Real systems plug in a LangGraph graph, a CrewAI ``Crew``, a Pydantic
    AI ``Agent``, etc.; the optimizer treats them all the same.
3.  **Two LLM judges** -- one scores answers, a different one critiques
    failures and rewrites the prompt -- so the model that writes a candidate is
    not the one that marks it. Judges are provider-agnostic; here both are
    deterministic offline stubs so the example runs with **no API key and no
    network**. Swap in ``LLMJudge("anthropic")`` / ``LLMJudge("openai")`` (two
    different providers) for the real thing.
4.  A **CoordinateAscentOptimizer** that searches for a better prompt against
    the training rows, and applies the winner to the live agent only if it does
    not score worse than the untuned agent on **held-out** rows.

Run it with:

    python examples/06_optimize_with_golden_dataset.py
"""

from adapt_agent.optimization import (
    CoordinateAscentOptimizer,
    EvaluationHarness,
    GoldenDataset,
    LLMJudge,
    OptimizableAgent,
    Parameter,
    ParameterKind,
    exact_match,
)

# --- A tiny multi-component agent "spread across the codebase" -------------- #
# In a real project these live in src/agents/*.py, a FastAPI backend, etc. The
# optimizer only needs (a) a runner that drives the whole system and (b) the
# live objects whose knobs it may tune. Here the shared "prompt" is the knob.
SHARED = {"capital_prompt": "Answer the question."}

_CAPITALS = {
    "France": "Paris",
    "Japan": "Tokyo",
    "Italy": "Rome",
    "Egypt": "Cairo",
    "Spain": "Madrid",
    "Kenya": "Nairobi",
}


def specialist_lookup(country: str) -> str:
    """A specialist sub-agent. It only answers cleanly under a precise prompt."""
    if "ONLY" in SHARED["capital_prompt"] or "city name" in SHARED["capital_prompt"].lower():
        return _CAPITALS.get(country, "unknown")
    return f"Well, the capital of {country} is probably a large city somewhere."


def orchestrator(question: str) -> str:
    """The system entrypoint: routes to the specialist (could be many of them)."""
    country = question.replace("What is the capital of", "").strip(" ?")
    return specialist_lookup(country)


def _response_of(prompt: str) -> str:
    """The text inside the judge prompt's ``<response>`` fence."""
    return prompt.split("<response>")[-1].split("</response>")[0].strip()


def scoring_judge_stub(prompt: str, system: str | None = None) -> str:
    """Offline stand-in for the judge that *scores* answers (no network).

    Grades higher when the response is a single clean word.
    Real usage: ``LLMJudge("anthropic")`` or any provider.
    """
    response = _response_of(prompt)
    score = 9 if response and " " not in response else 2
    return f'{{"score": {score}, "pass": {str(score >= 6).lower()}, "reasoning": "auto"}}'


def writing_judge_stub(prompt: str, system: str | None = None) -> str:
    """Offline stand-in for the judge that *critiques and rewrites* prompts.

    A different model from the scoring judge in real use -- see step 3 below.
    """
    if "CURRENT INSTRUCTION" in prompt:
        return "Answer with ONLY the capital city name, nothing else."
    return "The answer hedges instead of naming the city."


def main() -> None:
    # 1. Golden dataset, with some rows held back. The optimizer only ever
    #    searches against `train`; `holdout` is what tells you whether the
    #    result generalises rather than having memorised the training rows.
    data = GoldenDataset.from_list(
        [
            {"input": "What is the capital of France?", "expected": "Paris"},
            {"input": "What is the capital of Japan?", "expected": "Tokyo"},
            {"input": "What is the capital of Italy?", "expected": "Rome"},
            {"input": "What is the capital of Egypt?", "expected": "Cairo"},
            {"input": "What is the capital of Spain?", "expected": "Madrid"},
            {"input": "What is the capital of Kenya?", "expected": "Nairobi"},
        ]
    )
    train, holdout = data.split(0.67, seed=0)

    # 2. Wrap the agent. The tunable prompt is declared as a Parameter bound to
    #    the live SHARED dict, so mutating it changes what the specialist does.
    prompt_param = Parameter(
        name="capital.prompt",
        kind=ParameterKind.PROMPT,
        value=SHARED["capital_prompt"],
        getter=lambda: SHARED["capital_prompt"],
        setter=lambda v: SHARED.__setitem__("capital_prompt", v),
        component="capital",
    )
    agent = OptimizableAgent.from_callable(
        orchestrator, parameters=[prompt_param], name="capital-system"
    )
    print("Tunable parameters:", [p.name for p in agent.parameters])

    # 3. Two judges, deliberately. One *scores* answers; a different one
    #    *critiques failures and rewrites the prompt*. If a single model does
    #    both, it is marking its own homework -- it tends to prefer text in its
    #    own style -- and the reported gain is not independent. (Pass only
    #    `judge=` and the run says so in `result.recommendations`.)
    scorer = LLMJudge(scoring_judge_stub)
    writer = LLMJudge(writing_judge_stub)
    harness = EvaluationHarness(
        metrics=[exact_match(), scorer.as_metric("judge")],
        primary_metric="exact_match",
    )

    print("\nBaseline:", harness.evaluate(agent, train))
    print("Baseline answer for France:", orchestrator("What is the capital of France?"))

    # 4. Optimize. The writing judge proposes prompt rewrites from failures;
    #    the winner is applied only if it does not score worse than the untuned
    #    agent on the held-out rows (the validation gate).
    optimizer = CoordinateAscentOptimizer(
        harness, judge=scorer, proposer_judge=writer, seed=0, verbose=False
    )
    result = optimizer.optimize(agent, train, val_dataset=holdout)

    print("\nResult:", result)
    print("Best config:", result.best_config)
    print(
        f"Held-out score: {result.validation_baseline_score} -> {result.validation_score}"
        f" (rejected by the gate: {result.validation_rejected})"
    )
    print("Final prompt applied in place:", SHARED["capital_prompt"])
    print("Answer for France now:", orchestrator("What is the capital of France?"))


if __name__ == "__main__":
    main()
