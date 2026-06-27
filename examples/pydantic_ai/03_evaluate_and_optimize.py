"""Example 03: Evaluate and optimize a single Pydantic AI agent (offline).

This shows the ADAPT-Agent *training* half end to end on ONE Pydantic AI
``Agent``:

1.  Wrap the live ``Agent`` in an ``OptimizableAgent`` -- ADAPT-Agent introspects
    it and discovers tunable knobs automatically (its system prompt, model, and
    temperature) with live getters/setters.
2.  Build a ``GoldenDataset`` of input/expected pairs.
3.  Score it with an ``EvaluationHarness`` combining a deterministic metric
    (``exact_match``) and an offline ``LLMJudge`` used as a quality metric.
4.  Run a ``CoordinateAscentOptimizer`` that proposes better system prompts (the
    judge rewrites the prompt from observed failures) and applies the winning
    configuration back onto the live agent in place.

To make the improvement real *and* offline, the agent's model is a Pydantic AI
``FunctionModel`` whose reply quality depends on its system prompt: only once the
prompt instructs it to answer with the bare city name does it return clean
single-word answers. The optimizer discovers that rewrite and the score climbs.
No API key or network is required.

Run it with:

    python examples/pydantic_ai/03_evaluate_and_optimize.py
"""

from __future__ import annotations

import logging

# --- Friendly skip if Pydantic AI is not installed ------------------------- #
try:
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
except ImportError:
    raise SystemExit(
        "This example needs Pydantic AI: pip install 'adapt-agent[pydantic-ai]'\n"
        "(or: pip install pydantic-ai)"
    ) from None

from adapt_agent.optimization import (
    CoordinateAscentOptimizer,
    EvaluationHarness,
    GoldenDataset,
    LLMJudge,
    OptimizableAgent,
    exact_match,
)
from adapt_agent.optimization.introspection import detect, introspect

_CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Egypt": "Cairo"}


def _system_prompt_of(messages: list) -> str:
    """Pull the current system prompt text out of the message history."""
    for message in messages:
        for part in getattr(message, "parts", []):
            if type(part).__name__ == "SystemPromptPart":
                return getattr(part, "content", "") or ""
    return ""


def _latest_user_text(messages: list) -> str:
    """Pull the most recent user prompt text out of the message history."""
    text = ""
    for message in messages:
        for part in getattr(message, "parts", []):
            if type(part).__name__ == "UserPromptPart":
                content = getattr(part, "content", "")
                text = content if isinstance(content, str) else str(content)
    return text


def _offline_model_fn(messages: list, info: AgentInfo) -> ModelResponse:
    """An offline model whose answer quality depends on the system prompt.

    With a vague prompt it rambles; once the prompt says to answer with ONLY the
    city name it returns the bare capital. This gives the optimizer a real signal
    to chase -- entirely offline.
    """
    system = _system_prompt_of(messages)
    question = _latest_user_text(messages)
    country = question.replace("What is the capital of", "").strip(" ?")
    crisp = "ONLY" in system or "city name" in system.lower()
    if crisp:
        answer = _CAPITALS.get(country, "unknown")
    else:
        answer = f"Well, the capital of {country} is probably a big city somewhere."
    return ModelResponse(parts=[TextPart(answer)])


def build_agent() -> Agent:
    """A single Pydantic AI Agent starting from a deliberately vague prompt."""
    return Agent(_make_model(), system_prompt="Answer the geography question.")


def _make_model() -> FunctionModel:
    return FunctionModel(_offline_model_fn)


def deterministic_judge_stub(prompt: str) -> str:
    """Offline stand-in for a real LLM judge (no network).

    The harness drives the judge two ways and we branch on the user-prompt shape:

    * A *prompt-rewrite* request carries ``CURRENT INSTRUCTION`` and ``FAILURES``;
      we return a sharper instruction (the rewrite the optimizer then tries).
    * A *grading* request wraps the agent's answer in ``<response>...</response>``;
      we score high when that answer is a single clean word (a crisp capital).

    Real usage: ``LLMJudge(ClaudeJudge(model="claude-opus-4-8"))`` or any provider.
    """
    if "CURRENT INSTRUCTION" in prompt or "FAILURES" in prompt:
        return "Answer with ONLY the capital city name, nothing else."
    response = ""
    if "<response>" in prompt and "</response>" in prompt:
        response = prompt.split("<response>")[1].split("</response>")[0].strip()
    score = 9 if response and " " not in response else 2
    return f'{{"score": {score}, "pass": {str(score >= 6).lower()}, "reasoning": "auto"}}'


def main() -> None:
    # The offline FunctionModel exposes a read-only ``model_name``, so the
    # discovered "agent.model" knob has no working setter. ADAPT-Agent handles
    # this gracefully -- it logs a warning and marks the knob non-optimizable
    # rather than crashing. We quiet that one warning so this teaching example's
    # output stays focused on the prompt optimization.
    logging.getLogger("adapt_agent.optimization.parameters").setLevel(logging.ERROR)

    agent = build_agent()

    # 1. ADAPT-Agent recognizes the framework and discovers tunable knobs.
    print("Detected framework:", detect(agent))
    print("Discovered knobs:")
    for param in introspect(agent):
        print(f"  - {param.name:22} kind={param.kind.value:10} value={param.value!r}")

    # 2. A runner the harness can call with a plain input string. Pydantic AI's
    #    run_sync returns an AgentRunResult; we unwrap .output to a clean string.
    def run(question: str) -> str:
        return agent.run_sync(question).output

    target = OptimizableAgent.from_agent(agent, runner=run, name="capital-agent")

    # 3. Golden dataset of question -> expected bare-city-name.
    data = GoldenDataset.from_list(
        [
            {"input": "What is the capital of France?", "expected": "Paris"},
            {"input": "What is the capital of Japan?", "expected": "Tokyo"},
            {"input": "What is the capital of Italy?", "expected": "Rome"},
            {"input": "What is the capital of Egypt?", "expected": "Cairo"},
        ]
    )

    # 4. Judge + harness. The judge is both a scoring metric AND a prompt rewriter.
    judge = LLMJudge(deterministic_judge_stub)
    harness = EvaluationHarness(
        metrics=[exact_match(), judge.as_metric("quality")],
        primary_metric="quality",
    )

    print("\nBaseline:", harness.evaluate(target, data))
    print("Baseline answer:", run("What is the capital of France?"))

    # 5. Optimize. CoordinateAscent tunes one knob at a time; the judge proposes
    #    prompt rewrites from failures and the winner is applied in place.
    optimizer = CoordinateAscentOptimizer(harness, judge=judge, seed=0, verbose=False)
    result = optimizer.optimize(target, data)

    print("\nResult:", result)
    print("Improvement:", round(result.improvement, 4))
    print("Best config:", result.best_config)
    print("Final system prompt (applied in place):", agent._system_prompts)
    print("Answer now:", run("What is the capital of France?"))


if __name__ == "__main__":
    main()
