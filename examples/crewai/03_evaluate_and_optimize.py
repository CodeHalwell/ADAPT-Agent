"""Example 03 (CrewAI): evaluate and optimize one crew (offline, no API key).

This is ADAPT-Agent's *training* half applied to a real CrewAI ``Crew``:

1. Build a single-agent crew whose answer quality depends on its agent's
   ``role``/``goal``/``backstory`` prompts and ``max_iter`` -- exactly the knobs
   ADAPT-Agent's CrewAI introspector discovers.
2. Wrap the crew in an ``OptimizableAgent`` via ``from_components``, providing a
   ``runner`` that drives ``crew.kickoff(...)`` and returns the final text. The
   introspector walks the live crew and exposes its tunable parameters; mutating
   one rewrites the live agent so the next ``kickoff`` behaves differently.
3. Score it with an ``EvaluationHarness`` combining a deterministic metric
   (``exact_match``) and an offline ``LLMJudge`` used as a quality metric.
4. Run a ``CoordinateAscentOptimizer`` and print baseline -> best plus the knobs
   that ``introspect()`` discovered on the crew.

Everything runs offline: a deterministic local LLM for the crew AND a
deterministic stub for the judge -- no API key, no network.

Run it with:

    python examples/crewai/03_evaluate_and_optimize.py
"""

from __future__ import annotations

from typing import Any

try:
    from crewai import LLM, Agent, Crew, Process, Task
except ImportError:
    raise SystemExit(
        "This example needs CrewAI: pip install 'adapt-agent[crewai]'\n" "(or: pip install crewai)"
    )

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


class PromptSensitiveLLM(LLM):
    """A deterministic LLM whose output quality depends on the agent's prompt.

    This stands in for a real model so the optimizer has a measurable signal: it
    answers cleanly (just the city name) ONLY when the agent's goal has been
    tightened to demand a terse answer. The optimizer's job is to discover that
    rewriting the prompt improves the score.
    """

    #: Set by ``build_crew`` so ``call`` can read the live agent goal.
    agent_goal: str = ""

    def __init__(self) -> None:
        super().__init__(model="offline/prompt-sensitive")

    def call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
        text = _stringify(messages)
        country = next((c for c in _CAPITALS if c in text), None)
        city = _CAPITALS.get(country or "", "unknown")
        goal = self.agent_goal.lower()
        terse = "only" in goal or "just the city" in goal
        if terse:
            return city
        return f"Well, the capital is most likely {city}, a major city."


def _stringify(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        return " ".join(_stringify(m) for m in messages)
    if isinstance(messages, dict):
        return " ".join(str(v) for v in messages.values())
    return str(messages)


def build_crew() -> tuple[Crew, PromptSensitiveLLM, Agent]:
    llm = PromptSensitiveLLM()
    geographer = Agent(
        role="Geographer",
        # Deliberately vague goal -- the optimizer should tighten it.
        goal="Answer geography questions.",
        backstory="A cartographer.",
        llm=llm,
        max_iter=5,
        verbose=False,
    )
    # Keep the LLM stub in sync with the (mutable) agent goal so prompt rewrites
    # actually change the model's behaviour.
    llm.agent_goal = geographer.goal
    task = Task(
        description="Name the capital of: {question}",
        expected_output="The capital city.",
        agent=geographer,
    )
    crew = Crew(agents=[geographer], tasks=[task], process=Process.sequential, verbose=False)
    return crew, llm, geographer


def _extract_fence(text: str, label: str) -> str:
    start, end = text.find(f"<{label}>"), text.find(f"</{label}>")
    return "" if start == -1 or end == -1 else text[start + len(label) + 2 : end].strip()


def deterministic_judge_stub(prompt: str) -> str:
    """Offline judge matching LLMJudge's prompt format (no network, no key).

    LLMJudge's prompt-rewrite request puts ``CURRENT INSTRUCTION:`` in the user
    prompt, and its grading request wraps the answer in a ``<response>`` fence, so
    we branch on those (not the old ``Rewrite``/``RESPONSE:`` strings, which now
    live in the system prompt the provider passes separately).
    """
    if "CURRENT INSTRUCTION:" in prompt:
        return "Answer with ONLY the capital city name, nothing else."
    response = _extract_fence(prompt, "response")
    score = 9 if response and " " not in response else 2
    return f'{{"score": {score}, "pass": {str(score >= 6).lower()}, "reasoning": "auto"}}'


def main() -> None:
    crew, llm, agent = build_crew()

    # The runner drives the live crew and returns plain text for scoring. It
    # closes over the live ``crew``/``llm``, so applying a candidate config (a
    # rewritten goal) changes what this runner produces on the next call.
    def runner(question: str) -> str:
        llm.agent_goal = agent.goal  # re-sync after any in-place prompt rewrite
        out = crew.kickoff(inputs={"question": question})
        return getattr(out, "raw", str(out))

    # Detect + introspect: show exactly what ADAPT-Agent discovers on this crew.
    print("Detected framework:", detect(crew))
    print("Discovered knobs:")
    for p in introspect(crew):
        print(f"  - {p.name:<28} kind={p.kind.value}")

    target = OptimizableAgent.from_components(
        components={"crew": crew},
        runner=runner,
        name="crewai-geographer",
    )

    data = GoldenDataset.from_list(
        [
            {"input": "France", "expected": "Paris"},
            {"input": "Japan", "expected": "Tokyo"},
            {"input": "Italy", "expected": "Rome"},
            {"input": "Egypt", "expected": "Cairo"},
        ]
    )

    judge = LLMJudge(deterministic_judge_stub)
    harness = EvaluationHarness(
        metrics=[exact_match(), judge.as_metric("quality")],
        primary_metric="quality",
    )

    print("\nBaseline:", harness.evaluate(target, data))
    print("Baseline answer for France:", runner("France"))

    optimizer = CoordinateAscentOptimizer(harness, judge=judge, seed=0, verbose=False)
    result = optimizer.optimize(target, data)

    print("\nResult:", result)
    print("Improvement:", result.improvement)
    print("Best config:", result.best_config)
    print("Final agent goal applied in place:", repr(agent.goal))
    print("Answer for France now:", runner("France"))


if __name__ == "__main__":
    main()
