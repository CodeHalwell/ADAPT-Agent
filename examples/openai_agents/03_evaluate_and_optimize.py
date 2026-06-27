"""OpenAI Agents SDK example 03: evaluate and optimize a single agent (offline).

The third rung leaves the runtime guard behind and enters the *training* half of
ADAPT-Agent. We:

1. Build a single OpenAI ``Agent`` and discover its tunable knobs with
   ``introspect()`` (instructions -> PROMPT, model -> MODEL, model_settings ->
   HYPERPARAM, tools -> TOOL).
2. Wrap it as an ``OptimizableAgent`` -- the bridge between your agent and the
   optimizers.
3. Score it over a ``GoldenDataset`` with an ``EvaluationHarness`` combining a
   built-in metric (``exact_match``) and an offline ``LLMJudge`` used as a metric.
4. Run a ``CoordinateAscentOptimizer`` and print baseline -> best.

Everything runs **offline with no API key**: the judge is backed by a
deterministic stub callable, and the agent is driven by a local runner that reads
the agent's live ``instructions`` so that optimizing the prompt actually changes
the output (a real run would call ``Runner.run_sync(agent, prompt)`` instead).

Run it with:

    python examples/openai_agents/03_evaluate_and_optimize.py
"""

from __future__ import annotations

try:
    import agents  # noqa: F401
except ImportError:
    raise SystemExit(
        "This example needs the OpenAI Agents SDK: "
        "pip install 'adapt-agent[openai-agents]'  (or: pip install openai-agents)"
    ) from None

from agents import Agent

from adapt_agent.optimization import (
    CoordinateAscentOptimizer,
    EvaluationHarness,
    GoldenDataset,
    LLMJudge,
    OptimizableAgent,
    exact_match,
)
from adapt_agent.optimization.introspection import detect, introspect

_CAPITALS = {
    "What is the capital of France?": "Paris",
    "What is the capital of Japan?": "Tokyo",
    "What is the capital of Italy?": "Rome",
    "What is the capital of Egypt?": "Cairo",
}


def make_runner(agent: Agent):
    """Return an offline runner that reads the agent's LIVE instructions.

    A real deployment would do ``return Runner.run_sync(agent, prompt).final_output``.
    Here we simulate that: the agent only answers cleanly (just the city name)
    when its instructions demand a terse answer -- so the optimizer can measurably
    improve quality by rewriting ``instructions``.
    """

    def run(prompt: str) -> str:
        terse = "ONLY" in agent.instructions or "city name" in agent.instructions.lower()
        answer = _CAPITALS.get(prompt, "unknown")
        if terse:
            return answer
        return f"Well, I believe the capital you are asking about is {answer}, a lovely city."

    return run


def _extract_fence(prompt: str, label: str) -> str:
    """Pull the text inside a <label>...</label> fence the judge builds."""
    start = prompt.find(f"<{label}>")
    end = prompt.find(f"</{label}>")
    if start == -1 or end == -1:
        return ""
    return prompt[start + len(label) + 2 : end].strip()


def deterministic_judge_stub(prompt: str) -> str:
    """Offline stand-in for a real LLM judge (no network, no key).

    The LLMJudge sends this callable the *user* prompt. We branch on what kind of
    prompt it is:

    * A *prompt-rewrite* prompt contains ``CURRENT INSTRUCTION:`` -> return an
      improved instruction so the runner answers tersely.
    * A *grading* prompt contains a ``<response>...</response>`` fence -> return a
      JSON score (higher for a single clean word).

    Swap in ``ClaudeJudge(model="claude-opus-4-8")`` (an LLMJudge subclass) for
    the real thing.
    """
    if "CURRENT INSTRUCTION:" in prompt:
        return "Answer with ONLY the capital city name, nothing else."
    response = _extract_fence(prompt, "response")
    score = 9 if response and " " not in response else 2
    return f'{{"score": {score}, "pass": {str(score >= 6).lower()}, "reasoning": "auto"}}'


def main() -> None:
    # 1. The smallest real OpenAI Agent. `instructions` is the tunable prompt.
    agent = Agent(
        name="Capitals",
        instructions="Answer the question.",
    )

    # 2. Discover what is tunable. `detect` names the framework; `introspect`
    #    returns Parameter objects bound to the live agent in place.
    print("Detected framework:", detect(agent))
    discovered = introspect(agent)
    print("Discovered knobs:")
    for p in discovered:
        print(f"  - {p.name}  (kind={p.kind.value})")

    # 3. Wrap as an OptimizableAgent. We pass an explicit offline `runner`; the
    #    agent itself is registered as a component so its instructions/model/tools
    #    are introspected into the search space automatically.
    target = OptimizableAgent.from_agent(agent, runner=make_runner(agent), name="capitals-agent")
    print("\nTunable parameters on the target:", [p.name for p in target.parameters])

    # 4. Golden dataset + harness (a metric AND an offline judge metric).
    data = GoldenDataset.from_list([{"input": q, "expected": a} for q, a in _CAPITALS.items()])
    judge = LLMJudge(deterministic_judge_stub)
    harness = EvaluationHarness(
        metrics=[exact_match(), judge.as_metric("quality")],
        primary_metric="quality",
    )

    print("\nBaseline:", harness.evaluate(target, data))
    print("Baseline answer for France:", target.run("What is the capital of France?"))

    # 5. Optimize. The judge proposes prompt rewrites from observed failures and
    #    the winning config is applied back onto the live agent in place.
    optimizer = CoordinateAscentOptimizer(harness, judge=judge, seed=0, verbose=False)
    result = optimizer.optimize(target, data)

    print("\nResult:", result)
    print("Improvement:", result.improvement)
    print("Best config:", result.best_config)
    print("Agent instructions now:", repr(agent.instructions))
    print("Answer for France now:", target.run("What is the capital of France?"))


if __name__ == "__main__":
    main()
