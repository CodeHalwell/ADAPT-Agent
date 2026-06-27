"""Example 03 (Claude Agent SDK): evaluate and optimize a single agent.

ADAPT-Agent's *training* half turns any agent into a tunable search space, scores
it over a golden dataset, and searches for a better configuration -- entirely
**offline** when you back the judge with a deterministic stub (no API key, no
network).

For the Claude Agent SDK the unit of configuration is a ``ClaudeAgentOptions``
object: its ``system_prompt``, ``model``, ``allowed_tools`` / ``disallowed_tools``,
``max_turns`` and ``permission_mode``. ADAPT-Agent's introspector reads exactly
those fields off the options object and exposes them as tunable
:class:`Parameter` objects -- *without importing the SDK* (it duck-types with
``getattr``/``hasattr``).

To stay runnable without the SDK we use a tiny ``FakeOptions`` that carries the
same attributes the real ``ClaudeAgentOptions`` does, plus a deterministic runner
that "answers" based on the current ``system_prompt`` and ``model``. The whole
optimization machinery -- ``GoldenDataset``, ``EvaluationHarness``,
``LLMJudge``, ``CoordinateAscentOptimizer`` -- treats it identically to a real
options object.

Run it with:

    python examples/claude_agent/03_evaluate_and_optimize.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    import claude_agent_sdk  # noqa: F401
except ImportError:
    claude_agent_sdk = None  # type: ignore[assignment]

from adapt_agent.optimization import (
    CoordinateAscentOptimizer,
    EvaluationHarness,
    GoldenDataset,
    LLMJudge,
    OptimizableAgent,
    exact_match,
    token_f1,
)
from adapt_agent.optimization.introspection import detect, introspect

# Model ids the optimizer may choose between. Current Claude ids; the cheaper
# Haiku is the "wrong" choice for this toy task so the search has a real signal.
_OPUS = "claude-opus-4-8"
_HAIKU = "claude-haiku-4-5"

_CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Egypt": "Cairo"}


@dataclass
class FakeOptions:
    """Stands in for ``claude_agent_sdk.ClaudeAgentOptions``.

    It exposes the same attributes the SDK's options object does, so the
    ``claude_agent`` introspector recognizes it and surfaces each as a tunable
    knob. The introspector explicitly rejects objects carrying *other*
    frameworks' attributes (``handoffs``/``instructions``/...), so we keep this
    shape clean.
    """

    system_prompt: str = "Answer the question."
    model: str = _HAIKU
    allowed_tools: list[str] = field(default_factory=lambda: ["lookup", "calculator"])
    disallowed_tools: list[str] = field(default_factory=list)
    max_turns: int = 1
    permission_mode: str = "default"


# A single live options object -- the optimizer rewrites its fields in place.
OPTIONS = FakeOptions()


def run_agent(question: str) -> str:
    """Deterministic stand-in for ``query(prompt, options=OPTIONS)``.

    The "quality" of the answer depends on the live OPTIONS: a precise system
    prompt and the stronger model yield a clean one-word capital; otherwise the
    answer is hedged and wrong-shaped. This gives the optimizer a real gradient
    to climb -- in a real setup the SDK's actual model does this for you.
    """
    country = question.replace("What is the capital of", "").strip(" ?")
    precise = "ONLY" in OPTIONS.system_prompt or "city name" in OPTIONS.system_prompt.lower()
    if precise:
        # A sharp instruction yields the clean one-word capital.
        return _CAPITALS.get(country, "unknown")
    # A vague instruction yields a hedged, wrong-shaped answer.
    return f"The capital of {country} is a major city."


def deterministic_judge_stub(prompt: str, system: str | None = None) -> str:
    """Offline LLM-judge stand-in (no network, no API key).

    ``LLMJudge`` calls this with the rendered ``prompt`` (user turn) and a
    ``system`` instruction. We branch on the system prompt:

    * a *prompt-rewrite* request (system mentions "Rewrite the instruction")
      returns a sharper instruction;
    * otherwise it is a *grading* request -- we pull the agent's answer out of
      the ``<response>...</response>`` fence and score a clean single-word
      capital highly.

    Swap in ``ClaudeJudge(model="claude-opus-4-8")`` for the real thing.
    """
    if system and "Rewrite the instruction" in system:
        return "Answer with ONLY the capital city name, nothing else."
    # Grading: extract the fenced response and reward a clean one-word answer.
    answer = ""
    if "<response>" in prompt and "</response>" in prompt:
        answer = prompt.split("<response>", 1)[1].split("</response>", 1)[0].strip()
    score = 9 if answer and " " not in answer else 2
    return f'{{"score": {score}, "pass": {str(score >= 6).lower()}, "reasoning": "auto"}}'


def main() -> None:
    # The introspector recognizes our options object as a Claude Agent SDK one.
    print("Detected framework:", detect(OPTIONS))
    discovered = introspect(OPTIONS)
    print("Discovered tunable knobs:")
    for p in discovered:
        print(f"  - {p.name:28} kind={p.kind.value}")

    # 1. Golden dataset of question -> expected capital.
    data = GoldenDataset.from_list(
        [
            {"input": "What is the capital of France?", "expected": "Paris"},
            {"input": "What is the capital of Japan?", "expected": "Tokyo"},
            {"input": "What is the capital of Italy?", "expected": "Rome"},
            {"input": "What is the capital of Egypt?", "expected": "Cairo"},
        ]
    )

    # 2. Wrap the single agent. `from_components` registers OPTIONS as the live
    #    component to introspect, and `run_agent` is the runner that consults it.
    agent = OptimizableAgent.from_components(
        components={"agent": OPTIONS},
        runner=run_agent,
        name="claude-capital-agent",
    )
    print("\nParameters in the search space:", [p.name for p in agent.parameters])

    # 3. Score with two hard metrics plus an offline LLM judge.
    judge = LLMJudge(deterministic_judge_stub)
    harness = EvaluationHarness(
        metrics=[exact_match(), token_f1(), judge.as_metric("quality")],
        primary_metric="exact_match",
    )

    baseline = harness.evaluate(agent, data)
    print("\nBaseline scores:", baseline)
    print("Baseline answer for France:", run_agent("What is the capital of France?"))

    # 4. Optimize. CoordinateAscent tunes one knob at a time; the judge rewrites
    #    the system prompt from observed failures, and the model knob is searched
    #    over its discovered candidates.
    optimizer = CoordinateAscentOptimizer(harness, judge=judge, seed=0, verbose=False)
    result = optimizer.optimize(agent, data)

    best = harness.evaluate(agent, data)
    print("\nBest scores:", best)
    print("Improvement (primary):", result.improvement)
    print("Best config:", result.best_config)
    print("\nApplied in place -> system_prompt:", repr(OPTIONS.system_prompt))
    print("Applied in place -> model:", OPTIONS.model)
    print("Answer for France now:", run_agent("What is the capital of France?"))


if __name__ == "__main__":
    main()
