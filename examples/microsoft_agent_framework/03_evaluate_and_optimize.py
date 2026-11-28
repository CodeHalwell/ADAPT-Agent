"""Example 03: Evaluate and optimize a single Microsoft Agent Framework agent.

This shows the ADAPT-Agent *training* (offline optimization) loop for one
``Agent`` (``ChatAgent`` before 1.0):

1. A :class:`GoldenDataset` of inputs and expected answers.
2. An :class:`OptimizableAgent` wrapping the agent. ADAPT-Agent *introspects* a
   Microsoft ``Agent`` and discovers its tunable knobs automatically: the
   instructions (PROMPT), sampling options such as ``temperature`` (HYPERPARAM)
   and ``tools`` (TOOL). No manual wiring needed.
3. An :class:`EvaluationHarness` scoring outputs with a metric plus an
   :class:`LLMJudge`. The judge is provider-agnostic; here it is backed by a
   deterministic offline stub so the example runs with **no API key, no network**.
4. A :class:`CoordinateAscentOptimizer` that searches for a better config and
   applies the winner back onto the live agent in place.

The agent is a real ``agent_framework.Agent``; only the model is an offline
function (see ``_offline.py``), so ``introspect(agent)`` reports the parameters
of a genuine MAF agent. Swap in ``OpenAIChatClient(...)`` and nothing else
changes.

Run it with:

    python examples/microsoft_agent_framework/03_evaluate_and_optimize.py
"""

from __future__ import annotations

from typing import Any

from _offline import last_user_text, offline_client, require_framework, run_sync

require_framework()

from agent_framework import Agent, tool  # noqa: E402

from adapt_agent.optimization import (  # noqa: E402
    CoordinateAscentOptimizer,
    EvaluationHarness,
    GoldenDataset,
    LLMJudge,
    OptimizableAgent,
    exact_match,
)
from adapt_agent.optimization.introspection import detect, introspect  # noqa: E402

CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Egypt": "Cairo"}


def capital_model(messages: list[Any], options: dict[str, Any]) -> str:
    """The offline 'model'. Its behaviour depends on the agent's instructions
    (``options["instructions"]``, exactly what a real chat client receives), so
    optimizing the prompt really changes the score -- the point of the demo."""
    instructions = str(options.get("instructions") or "")
    country = last_user_text(messages).replace("What is the capital of", "").strip(" ?")
    if "ONLY" in instructions or "city name" in instructions.lower():
        return CAPITALS.get(country, "unknown")
    return f"Well, {country}'s capital is some large city, probably."


@tool(approval_mode="never_require")
def search(query: str) -> str:
    """Search the web."""
    return "no results"


@tool(approval_mode="never_require")
def calculator(expression: str) -> str:
    """Evaluate arithmetic."""
    return "0"


def build_agent() -> Agent:
    return Agent(
        offline_client(capital_model),
        "Answer the question.",
        name="capital_expert",
        tools=[search, calculator],
        default_options={"temperature": 0.7},
    )


def _names(candidates: Any) -> Any:
    """Tool-list candidates print as tool names, not object reprs."""
    if not isinstance(candidates, list):
        return candidates
    return [[getattr(t, "name", t) for t in c] if isinstance(c, list) else c for c in candidates]


def run_agent(agent: Agent, question: str) -> str:
    """Drive the async agent synchronously and return its text (the runner)."""
    return str(run_sync(agent.run(question)).text)


def _fenced(prompt: str, label: str) -> str:
    """Pull the text out of a ``<label>...</label>`` fence the judge uses."""
    start = prompt.find(f"<{label}>")
    end = prompt.find(f"</{label}>")
    if start == -1 or end == -1:
        return ""
    return prompt[start + len(label) + 2 : end].strip()


def deterministic_judge_stub(prompt: str, system: str | None = None) -> str:
    """Offline stand-in for a real LLM judge (no network).

    The real judge sends two kinds of prompts; we recognize both:

    * *grading* prompts wrap the answer in a ``<response>...</response>`` fence;
      we return JSON ``{"score": ...}`` -- higher for a single clean word.
    * *prompt-rewrite* prompts contain ``CURRENT INSTRUCTION``; we return a
      better instruction that makes the toy agent answer cleanly.

    Real usage: ``LLMJudge(ClaudeJudge(...))`` / ``OpenAIJudge(...)`` / any
    provider, optionally ``adversarial=True``.
    """
    if "CURRENT INSTRUCTION" in prompt:
        return "Answer with ONLY the capital city name, nothing else."
    response = _fenced(prompt, "response")
    score = 9 if response and " " not in response and response != "unknown" else 2
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

    # 2. The live agent + the OptimizableAgent wrapper. The runner closes over
    #    the live ``agent`` so applying a candidate config changes the next run.
    agent = build_agent()
    target = OptimizableAgent.from_agent(
        agent,
        runner=lambda q: run_agent(agent, q),
        component_name="capital_expert",
        name="capital-agent",
    )

    # Show what the framework introspection discovered automatically.
    print("detect(agent) =", detect(agent))
    print("Introspected parameters:")
    for p in introspect(agent):
        print(f"  - {p.name:<32} kind={p.kind.value:<10} candidates={_names(p.candidates)}")

    # 3. Judge + harness. exact_match is the headline metric; the judge adds a
    #    quality score and also drives prompt rewrites during optimization.
    judge = LLMJudge(deterministic_judge_stub)
    harness = EvaluationHarness(
        metrics=[exact_match(), judge.as_metric("quality")],
        primary_metric="exact_match",
    )

    baseline = harness.evaluate(target, data)
    print(f"\nBaseline: {baseline}")
    print("Baseline answer (France):", run_agent(agent, "What is the capital of France?"))

    # 4. Optimize. CoordinateAscent greedily improves one parameter at a time;
    #    the judge proposes prompt rewrites from observed failures.
    optimizer = CoordinateAscentOptimizer(harness, judge=judge, seed=0)
    result = optimizer.optimize(target, data)

    print(
        f"\nbaseline={result.baseline_score:.3f}  best={result.best_score:.3f}  "
        f"improvement={result.improvement:+.3f}"
    )
    print("Best config:", result.best_config)
    print("Instructions applied in place:", repr(agent.default_options.get("instructions")))
    print("Answer (France) now:", run_agent(agent, "What is the capital of France?"))


if __name__ == "__main__":
    main()
