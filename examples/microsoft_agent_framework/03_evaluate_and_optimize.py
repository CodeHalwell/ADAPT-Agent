"""Example 03: Evaluate and optimize a single Microsoft Agent Framework agent.

This shows the ADAPT-Agent *training* (offline optimization) loop for one
``ChatAgent``:

1. A :class:`GoldenDataset` of inputs and expected answers.
2. An :class:`OptimizableAgent` wrapping the agent. ADAPT-Agent *introspects* a
   Microsoft ``ChatAgent`` and discovers its tunable knobs automatically:
   ``instructions`` (PROMPT), the model on ``chat_client`` (MODEL),
   ``temperature`` / ``top_p`` / ``max_tokens`` (HYPERPARAM), ``tools`` (TOOL),
   and ``skills`` (SKILL). No manual wiring needed.
3. An :class:`EvaluationHarness` scoring outputs with a metric plus an
   :class:`LLMJudge`. The judge is provider-agnostic; here it is backed by a
   deterministic offline stub so the example runs with **no API key, no network**.
4. A :class:`CoordinateAscentOptimizer` that searches for a better config and
   applies the winner back onto the live agent in place.

We use an offline ``OfflineChatAgent`` that exposes exactly the attributes the
introspector keys off (``instructions``, ``chat_client``, an async ``run``), so
``introspect(agent)`` reports real, framework-derived parameters. Swap in a real
``OpenAIChatClient().create_agent(...)`` and the same introspection applies.

Run it with:

    python examples/microsoft_agent_framework/03_evaluate_and_optimize.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from adapt_agent.optimization import (
    CoordinateAscentOptimizer,
    EvaluationHarness,
    GoldenDataset,
    LLMJudge,
    OptimizableAgent,
    exact_match,
)
from adapt_agent.optimization.introspection import detect, introspect


# --- An offline ChatAgent with the real ChatAgent attribute surface --------- #
@dataclass
class AgentRunResponse:
    text: str


@dataclass
class FakeChatClient:
    """Stands in for an OpenAIChatClient: holds the model + sampling settings.

    The introspector reads ``model_id`` (MODEL) and ``temperature`` / ``top_p`` /
    ``max_tokens`` (HYPERPARAM) off this object, mirroring a real chat client.
    """

    model_id: str = "gpt-4o-mini"
    temperature: float = 0.7


@dataclass
class OfflineChatAgent:
    """Offline stand-in for a Microsoft ``ChatAgent`` (see example 01).

    Carries the attributes the introspector recognizes: ``instructions``,
    ``chat_client``, ``tools``, and a callable async ``run``. Its (toy) behaviour
    depends on the instructions, so optimizing the prompt actually changes the
    score -- which is the whole point of the demo.
    """

    instructions: str
    chat_client: FakeChatClient = field(default_factory=FakeChatClient)
    name: str = "capital_expert"
    tools: list[str] = field(default_factory=lambda: ["search", "calculator"])

    async def run(self, prompt: str) -> AgentRunResponse:
        country = prompt.replace("What is the capital of", "").strip(" ?")
        # The agent only answers cleanly when the instructions demand a bare city
        # name; otherwise it rambles (and scores poorly).
        wants_bare = "ONLY" in self.instructions or "city name" in self.instructions.lower()
        capitals = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Egypt": "Cairo"}
        if wants_bare:
            return AgentRunResponse(text=capitals.get(country, "unknown"))
        return AgentRunResponse(text=f"Well, {country}'s capital is some large city, probably.")


def run_agent(agent: OfflineChatAgent, question: str) -> str:
    """Drive the async agent synchronously and return its text (the runner)."""
    response = asyncio.run(agent.run(question))
    return response.text


def _fenced(prompt: str, label: str) -> str:
    """Pull the text out of a ``<label>...</label>`` fence the judge uses."""
    start = prompt.find(f"<{label}>")
    end = prompt.find(f"</{label}>")
    if start == -1 or end == -1:
        return ""
    return prompt[start + len(label) + 2 : end].strip()


def deterministic_judge_stub(prompt: str) -> str:
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
    agent = OfflineChatAgent(instructions="Answer the question.")
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
        print(f"  - {p.name:<32} kind={p.kind.value:<10} candidates={p.candidates}")

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
    print("Instructions applied in place:", repr(agent.instructions))
    print("Answer (France) now:", run_agent(agent, "What is the capital of France?"))


if __name__ == "__main__":
    main()
