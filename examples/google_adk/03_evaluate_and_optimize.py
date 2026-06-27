"""Example 03 (Google ADK): evaluate and optimize a single ADK agent (offline).

The second half of ADAPT-Agent is **offline optimization** ("training"): turn an
agent into a tunable search space, score it over a golden dataset, and search for
a better configuration - then apply the winner back onto the live agent.

This example optimizes a *single* Google ADK ``LlmAgent``. It runs with no API
key and no network:

* We define an ADK-agent-shaped object (``name``, ``instruction``,
  ``generate_content_config``, ``tools``, ``sub_agents``) so ADAPT-Agent's
  introspection recognises it as ``google_adk`` and discovers its tunable knobs -
  *without* importing ``google.adk`` (everything is duck-typed). Swap in a real
  ``LlmAgent`` and the exact same code applies.
* A deterministic runner reads the agent's live ``instruction`` and answers
  accordingly, so changing the prompt actually changes behaviour.
* An ``LLMJudge`` backed by a deterministic stub scores answers and rewrites the
  prompt from observed failures.

What gets introspected for a Google ADK agent (see the doc page for the full
table): ``instruction`` / ``global_instruction`` (PROMPT, only when plain
strings), ``model`` (MODEL), ``temperature`` / ``top_p`` / ``max_output_tokens``
from ``generate_content_config`` (HYPERPARAM), ``tools`` (TOOL, drop-one
ablation when 2+), and ``sub_agents`` (ROUTING). Nested ``sub_agents`` recurse.

Run it with:

    python examples/google_adk/03_evaluate_and_optimize.py
"""

from __future__ import annotations

from types import SimpleNamespace

from adapt_agent.optimization import (
    CoordinateAscentOptimizer,
    EvaluationHarness,
    GoldenDataset,
    LLMJudge,
    OptimizableAgent,
    exact_match,
)
from adapt_agent.optimization.introspection import detect, introspect


# --- A Google-ADK-shaped agent (no google.adk import required) ------------- #
# ADAPT-Agent's introspector recognises an ADK LlmAgent by duck typing: it must
# have `sub_agents` and `instruction`, and must NOT carry foreign-framework
# attributes (`handoffs`, `kickoff`, `allowed_tools`). A real LlmAgent matches
# this shape exactly, so this stand-in is faithful while staying offline.
def build_agent() -> SimpleNamespace:
    return SimpleNamespace(
        name="capital_agent",
        model="gemini-flash-latest",
        instruction="Answer the question.",  # the tunable PROMPT knob
        global_instruction=None,
        # generate_content_config carries temperature / top_p / max_output_tokens.
        generate_content_config=SimpleNamespace(temperature=0.7, top_p=1.0, max_output_tokens=256),
        tools=[],
        sub_agents=[],
    )


_CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Egypt": "Cairo"}


def make_runner(agent: SimpleNamespace):
    """Return a runner that drives the agent; behaviour depends on its prompt.

    In a real project this would build an ADK ``Runner`` and call ``run`` (see
    example 01). Here it inspects the agent's live ``instruction`` so the
    optimizer's prompt edits visibly change the output.
    """

    def run(question: str) -> str:
        country = question.replace("What is the capital of", "").strip(" ?")
        instruction = agent.instruction.lower()
        if "only" in instruction or "city name" in instruction:
            return _CAPITALS.get(country, "unknown")
        return f"The capital of {country} is a major city."

    return run


def deterministic_judge_stub(prompt: str, system: str | None = None) -> str:
    """Offline stand-in for a real LLM judge (no network).

    The judge sends grading rubrics and prompt-rewrite requests in the ``system``
    message and the data in ``prompt``; we accept both so the offline stub can
    tell them apart. A real judge is provider-backed - swap for
    ``ClaudeJudge(model="claude-opus-4-8")`` / ``OpenAIJudge(...)``.

    * For prompt-rewrite requests it returns a sharper instruction.
    * For grading it returns a JSON score, rewarding single-word answers.
    """
    rubric = system or ""
    if "prompt engineer" in rubric or "CURRENT INSTRUCTION" in prompt:
        return "Answer with ONLY the capital city name, nothing else."
    # Grading: the model's answer arrives inside <response>...</response> fences.
    import re

    match = re.search(r"<response>(.*?)</response>", prompt, re.DOTALL)
    response = (match.group(1) if match else prompt).strip()
    score = 9 if response and " " not in response else 2
    return f'{{"score": {score}, "pass": {str(score >= 6).lower()}, "reasoning": "auto"}}'


def main() -> None:
    agent = build_agent()

    # 1. Confirm introspection recognises the framework and lists its knobs.
    print("Detected framework:", detect(agent))
    print("Discovered tunable parameters:")
    for p in introspect(agent):
        print(f"  - {p.name:35} kind={p.kind.value}")

    # 2. Golden dataset of inputs + expected outputs.
    data = GoldenDataset.from_list(
        [
            {"input": "What is the capital of France?", "expected": "Paris"},
            {"input": "What is the capital of Japan?", "expected": "Tokyo"},
            {"input": "What is the capital of Italy?", "expected": "Rome"},
            {"input": "What is the capital of Egypt?", "expected": "Cairo"},
        ]
    )

    # 3. Wrap the agent as an OptimizableAgent. `from_agent` introspects the live
    #    object for knobs; we pass an explicit `runner` because an ADK agent is
    #    driven through a Runner, not by calling the object directly.
    target = OptimizableAgent.from_agent(agent, runner=make_runner(agent), name="capital-agent")

    # 4. Judge as a scoring metric AND a prompt rewriter; pair with exact_match.
    judge = LLMJudge(deterministic_judge_stub)
    harness = EvaluationHarness(
        metrics=[exact_match(), judge.as_metric("quality")],
        primary_metric="exact_match",
    )

    print("\nBaseline:", harness.evaluate(target, data))
    print("Baseline answer (France):", make_runner(agent)("What is the capital of France?"))

    # 5. Optimize. CoordinateAscent tunes one parameter at a time; the judge
    #    proposes prompt rewrites from failures, applied in place on success.
    optimizer = CoordinateAscentOptimizer(harness, judge=judge, seed=0, verbose=False)
    result = optimizer.optimize(target, data)

    print("\nimprovement:", round(result.improvement, 3))
    print("best_config:", result.best_config)
    print("instruction now:", repr(agent.instruction))
    print("Answer (France) now:", make_runner(agent)("What is the capital of France?"))


if __name__ == "__main__":
    main()
