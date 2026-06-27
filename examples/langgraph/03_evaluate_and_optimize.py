"""LangGraph example 03: evaluate and optimize (offline, no API key).

The third rung enters the *training* half of ADAPT-Agent. LangGraph introspection
is a best-effort *structural* walk of a compiled graph; prompts that live inside a
node's closure (the common case) are not auto-discovered. The robust, documented
pattern is therefore to keep tunable prompts in a small live config object and
declare an explicit ``Parameter`` bound to it -- which is exactly what we do here.

We:

1. Keep the node's instruction in a live ``CONFIG`` dict and build a compiled
   graph whose node reads it.
2. Run ``detect`` / ``introspect`` on the compiled graph to show what the
   structural walk finds (often little for closure-based nodes -- hence step 3).
3. Wrap the graph as an ``OptimizableAgent`` and DECLARE the prompt as an explicit
   ``Parameter`` bound to ``CONFIG`` so the optimizer can read/rewrite it in place.
4. Score over a ``GoldenDataset`` with an ``EvaluationHarness`` (``exact_match`` +
   an offline ``LLMJudge`` metric) and run a ``CoordinateAscentOptimizer``.

Everything runs offline: the node is pure Python and the judge is a deterministic
stub. Swap ``LLMJudge(stub)`` for ``ClaudeJudge(...)`` / ``OpenAIJudge(...)`` and
point the node at a real model for production.

Run it with:

    python examples/langgraph/03_evaluate_and_optimize.py
"""

from __future__ import annotations

from typing import Any

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    raise SystemExit(
        "This example needs LangGraph: pip install 'adapt-agent[langgraph]'  "
        "(or: pip install langgraph)"
    ) from None

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
from adapt_agent.optimization.introspection import detect, introspect

# The single tunable knob, kept in a LIVE object the node reads on every run.
CONFIG: dict[str, str] = {"prompt": "Answer the question."}

_CAPITALS = {
    "What is the capital of France?": "Paris",
    "What is the capital of Japan?": "Tokyo",
    "What is the capital of Italy?": "Rome",
    "What is the capital of Egypt?": "Cairo",
}


def build_compiled_graph() -> Any:
    """A one-node graph whose behaviour depends on the live ``CONFIG['prompt']``."""

    def answer(state: dict[str, Any]) -> dict[str, Any]:
        question = state["question"]
        terse = "ONLY" in CONFIG["prompt"] or "city name" in CONFIG["prompt"].lower()
        city = _CAPITALS.get(question, "unknown")
        reply = city if terse else f"I think the capital is {city}, which is a lovely city."
        return {**state, "answer": reply}

    builder = StateGraph(dict)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile()


def _extract_fence(prompt: str, label: str) -> str:
    start, end = prompt.find(f"<{label}>"), prompt.find(f"</{label}>")
    if start == -1 or end == -1:
        return ""
    return prompt[start + len(label) + 2 : end].strip()


def deterministic_judge_stub(prompt: str) -> str:
    """Offline LLM-judge stand-in (no network, no key).

    The judge sends this callable its *user* prompt. A rewrite prompt contains
    ``CURRENT INSTRUCTION:``; a grading prompt contains a ``<response>`` fence.
    """
    if "CURRENT INSTRUCTION:" in prompt:
        return "Answer with ONLY the capital city name, nothing else."
    response = _extract_fence(prompt, "response")
    score = 9 if response and " " not in response else 2
    return f'{{"score": {score}, "pass": {str(score >= 6).lower()}, "reasoning": "auto"}}'


def main() -> None:
    graph = build_compiled_graph()

    # 2. What does the structural walk find? (Closure prompts are invisible.)
    print("Detected framework:", detect(graph))
    print("Structurally discovered knobs:", [p.name for p in introspect(graph)] or "(none)")

    # 3. Drive the graph offline and DECLARE the prompt knob explicitly.
    def run(question: str) -> str:
        return graph.invoke({"question": question})["answer"]

    prompt_param = Parameter(
        name="answer_node.prompt",
        kind=ParameterKind.PROMPT,
        getter=lambda: CONFIG["prompt"],
        setter=lambda v: CONFIG.__setitem__("prompt", v),
        component="answer_node",
    )
    target = OptimizableAgent.from_components(
        components={"graph": graph},
        runner=run,
        parameters=[prompt_param],
        name="capitals-graph",
    )
    print("Tunable parameters on the target:", [p.name for p in target.parameters])

    # 4. Golden dataset + harness (metric + offline judge), then optimize.
    data = GoldenDataset.from_list([{"input": q, "expected": a} for q, a in _CAPITALS.items()])
    judge = LLMJudge(deterministic_judge_stub)
    harness = EvaluationHarness(
        metrics=[exact_match(), judge.as_metric("quality")],
        primary_metric="quality",
    )

    print("\nBaseline:", harness.evaluate(target, data))
    print("Baseline answer for France:", run("What is the capital of France?"))

    result = CoordinateAscentOptimizer(harness, judge=judge, seed=0).optimize(target, data)

    print("\nResult:", result)
    print("Improvement:", result.improvement)
    print("Best config:", result.best_config)
    print("CONFIG['prompt'] now:", repr(CONFIG["prompt"]))
    print("Answer for France now:", run("What is the capital of France?"))


if __name__ == "__main__":
    main()
