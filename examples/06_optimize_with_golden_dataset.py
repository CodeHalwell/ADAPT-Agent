"""Example 06: Optimize a multi-agent system against a golden dataset.

This shows the full ADAPT-Agent optimization loop end to end:

1.  A **golden dataset** of inputs and expected outputs.
2.  An **OptimizableAgent** that wraps your agent code -- here a tiny
    "orchestrator + two specialists" system whose behaviour depends on a tunable
    prompt. Real systems plug in a LangGraph graph, a CrewAI ``Crew``, a Pydantic
    AI ``Agent``, etc.; the optimizer treats them all the same.
3.  An **LLM-as-judge** used both as a scoring metric *and* to rewrite prompts
    from observed failures. The judge is provider-agnostic; here we back it with
    a deterministic offline stub so the example runs with **no API key and no
    network**. Swap in ``ClaudeJudge(model=...)`` / ``OpenAIJudge(...)`` /
    ``GeminiJudge(...)`` for the real thing.
4.  A **CoordinateAscentOptimizer** that searches for a better prompt and applies
    the winning configuration back onto the live agent.

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

_CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Egypt": "Cairo"}


def specialist_lookup(country: str) -> str:
    """A specialist sub-agent. It only answers cleanly under a precise prompt."""
    if "ONLY" in SHARED["capital_prompt"] or "city name" in SHARED["capital_prompt"].lower():
        return _CAPITALS.get(country, "unknown")
    return f"Well, the capital of {country} is probably a large city somewhere."


def orchestrator(question: str) -> str:
    """The system entrypoint: routes to the specialist (could be many of them)."""
    country = question.replace("What is the capital of", "").strip(" ?")
    return specialist_lookup(country)


def deterministic_judge_stub(prompt: str) -> str:
    """Offline stand-in for a real LLM judge (no network).

    * For *grading* prompts it returns a JSON score.
    * For *prompt-rewrite* prompts it returns an improved instruction.
    Real usage: ``LLMJudge(ClaudeJudge(...))`` or any provider.
    """
    if "Rewrite the instruction" in prompt:
        return "Answer with ONLY the capital city name, nothing else."
    # Grade higher when the response is a single clean word.
    response = prompt.split("RESPONSE:")[-1].split("REFERENCE")[0].strip()
    score = 9 if response and " " not in response else 2
    return f'{{"score": {score}, "pass": {str(score >= 6).lower()}, "reasoning": "auto"}}'


def main() -> None:
    # 1. Golden dataset.
    data = GoldenDataset.from_list(
        [
            {"input": "What is the capital of France?", "expected": "Paris"},
            {"input": "What is the capital of Japan?", "expected": "Tokyo"},
            {"input": "What is the capital of Italy?", "expected": "Rome"},
            {"input": "What is the capital of Egypt?", "expected": "Cairo"},
        ]
    )

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

    # 3. Judge: used as a metric AND to rewrite prompts. Provider-agnostic.
    judge = LLMJudge(deterministic_judge_stub)
    harness = EvaluationHarness(
        metrics=[exact_match(), judge.as_metric("judge")],
        primary_metric="exact_match",
    )

    print("\nBaseline:", harness.evaluate(agent, data))
    print("Baseline answer for France:", orchestrator("What is the capital of France?"))

    # 4. Optimize. The LLM judge proposes prompt rewrites from failures.
    optimizer = CoordinateAscentOptimizer(harness, judge=judge, seed=0, verbose=False)
    result = optimizer.optimize(agent, data)

    print("\nResult:", result)
    print("Best config:", result.best_config)
    print("Final prompt applied in place:", SHARED["capital_prompt"])
    print("Answer for France now:", orchestrator("What is the capital of France?"))


if __name__ == "__main__":
    main()
