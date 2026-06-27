"""LangGraph example 04: a multi-node router team + full training (offline).

The top rung. We build a realistic **supervisor / router** graph: a router node
inspects the question and conditionally routes to one of two specialist nodes
(geography or math). The whole graph is governed as ONE unit and optimized as a
whole -- every specialist's prompt, plus a tool allow-list, tuned together.

It shows three things, all offline (no API key):

1. **Govern the whole team** with ``LangGraphAdapter`` + a ``Firewall``.
2. **Optimize the whole team** with ``make_default_optimizer`` (few-shot ->
   prompts -> models/hparams -> tool/skill ablation) driven by an *adversarial*
   ``LLMJudge`` that also proposes NEW tools (surfaced on
   ``result.recommendations``).
3. **The declarative path**: the same run expressed as a config and executed with
   ``run_training`` -- here via an in-process module so it runs offline; see
   ``langgraph.train.yaml`` for the file-based, real-world template.

Run it with:

    python examples/langgraph/04_multi_agent_and_training.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    raise SystemExit(
        "This example needs LangGraph: pip install 'adapt-agent[langgraph]'  "
        "(or: pip install langgraph)"
    ) from None

from adapt_agent import Firewall
from adapt_agent.adapters import LangGraphAdapter
from adapt_agent.optimization import (
    EvaluationHarness,
    GoldenDataset,
    LLMJudge,
    OptimizableAgent,
    ParameterKind,
    exact_match,
    make_default_optimizer,
)
from adapt_agent.optimization.config import parse_training_config, run_training


# --- Live, tunable knobs the specialist nodes read on every run ------------- #
class Knobs:
    """Holds the team's tunable prompts + a tool allow-list as live attributes.

    Keeping knobs on a real object (rather than in node closures) lets both the
    Python API (explicit ``Parameter``/``add_tool_parameter``) and the YAML config
    (``attr``/``attr_path`` binding) read and rewrite them in place.
    """

    geo_prompt = "Answer the geography question."
    math_prompt = "Answer the math question."
    tools = ["atlas_lookup", "web_search", "calculator"]


KNOBS = Knobs()

_GEO = {"capital of france": "Paris", "capital of japan": "Tokyo", "capital of italy": "Rome"}
_MATH = {"2+2": "4", "3*5": "15", "10-7": "3"}


def _route(question: str) -> str:
    return "geo" if "capital" in question.lower() else "math"


def _geo_answer(question: str) -> str:
    terse = "ONLY" in KNOBS.geo_prompt or "city name" in KNOBS.geo_prompt.lower()
    key = question.lower().replace("what is the ", "").strip(" ?")
    city = _GEO.get(key, "unknown")
    return city if terse else f"I believe it is {city}."


def _math_answer(question: str) -> str:
    terse = "ONLY" in KNOBS.math_prompt or "number" in KNOBS.math_prompt.lower()
    val = _MATH.get(question.replace(" ", ""), "unknown")
    return val if terse else f"The result works out to {val}."


def build_compiled_graph() -> Any:
    """A router graph: START -> router -> (geo | math) -> END."""

    def router(state: dict[str, Any]) -> dict[str, Any]:
        return {**state, "route": _route(state["question"])}

    def geo(state: dict[str, Any]) -> dict[str, Any]:
        return {**state, "answer": _geo_answer(state["question"])}

    def math(state: dict[str, Any]) -> dict[str, Any]:
        return {**state, "answer": _math_answer(state["question"])}

    builder = StateGraph(dict)
    builder.add_node("router", router)
    builder.add_node("geo", geo)
    builder.add_node("math", math)
    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", lambda s: s["route"], {"geo": "geo", "math": "math"})
    builder.add_edge("geo", END)
    builder.add_edge("math", END)
    return builder.compile()


def run(question: str) -> str:
    """Drive the whole team for one question (the system entrypoint)."""
    return build_compiled_graph().invoke({"question": question})["answer"]


def _extract_fence(prompt: str, label: str) -> str:
    start, end = prompt.find(f"<{label}>"), prompt.find(f"</{label}>")
    return "" if start == -1 or end == -1 else prompt[start + len(label) + 2 : end].strip()


def adversarial_judge_stub(prompt: str) -> str:
    """Offline adversarial judge: grades harshly, rewrites prompts, suggests tools."""
    if "CURRENT INSTRUCTION:" in prompt:  # prompt-rewrite request
        return "Answer with ONLY the exact answer, nothing else."
    if "COMPONENT:" in prompt and "OBSERVED FAILURES" in prompt:  # tool-suggestion request
        return (
            '{"tools": [{"name": "unit_converter", "description": "convert units", '
            '"rationale": "several failures involved unit mismatches"}]}'
        )
    response = _extract_fence(prompt, "response")  # grading request
    score = 9 if response and " " not in response else 2
    return f'{{"score": {score}, "pass": {str(score >= 6).lower()}, "reasoning": "auto"}}'


GOLDEN = [
    {"input": "What is the capital of France?", "expected": "Paris"},
    {"input": "What is the capital of Japan?", "expected": "Tokyo"},
    {"input": "2 + 2", "expected": "4"},
    {"input": "10 - 7", "expected": "3"},
]


def govern_demo() -> None:
    print("=== 1. Govern the whole team ===")
    firewall = Firewall(max_content_length=10_000)
    guarded = LangGraphAdapter(firewall=firewall, agent_id="router-team").wrap_agent(
        build_compiled_graph()
    )
    out = guarded.execute({"question": "What is the capital of Italy?"})
    print("  team answered:", out["answer"])


def optimize_demo() -> None:
    print("\n=== 2. Optimize the whole team (programmatic, adversarial judge) ===")
    data = GoldenDataset.from_list(GOLDEN)
    judge = LLMJudge(adversarial_judge_stub, adversarial=True)
    harness = EvaluationHarness(
        [exact_match(), judge.as_metric("quality")], primary_metric="quality"
    )

    target = OptimizableAgent.from_callable(run, name="router-team")
    # Closure prompts are not auto-discoverable -> declare them, plus tool ablation.
    target.add_tool_parameter(
        name="geo.tools",
        kind=ParameterKind.TOOL,
        getter=lambda: KNOBS.tools,
        setter=lambda v: setattr(KNOBS, "tools", list(v)),
        candidate_tools=KNOBS.tools,
    )
    from adapt_agent.optimization import Parameter

    for comp, attr in (("geo", "geo_prompt"), ("math", "math_prompt")):
        target.add_parameter(
            Parameter(
                name=f"{comp}.prompt",
                kind=ParameterKind.PROMPT,
                getter=lambda a=attr: getattr(KNOBS, a),
                setter=lambda v, a=attr: setattr(KNOBS, a, v),
                component=comp,
            )
        )

    print("  baseline:", harness.evaluate(target, data).score)
    result = make_default_optimizer(harness, judge=judge, max_evals=40).optimize(target, data)
    print("  improvement:", round(result.improvement, 3))
    print("  recommendations (judge-proposed tools/skills):")
    for tip in result.recommendations:
        print("    -", tip)


def training_config_demo() -> None:
    print("\n=== 3. The declarative path (run_training, offline in-process) ===")
    # Reset knobs and write the golden data to a file the config can point at.
    KNOBS.geo_prompt, KNOBS.math_prompt = "Answer the geography question.", "Answer the math."
    data_path = Path(__file__).with_name("_golden.jsonl")
    data_path.write_text("\n".join(json.dumps(row) for row in GOLDEN), encoding="utf-8")

    # Expose this example's `run` + `KNOBS` as an importable module for the config.
    mod = types.ModuleType("lg_router_demo")
    mod.run = run
    mod.KNOBS = KNOBS
    sys.modules["lg_router_demo"] = mod

    config = parse_training_config(
        {
            "target": {
                "entrypoint": "lg_router_demo:run",
                "components": {"knobs": "lg_router_demo:KNOBS"},
            },
            "dataset": {"path": str(data_path), "format": "jsonl"},
            "metrics": ["exact_match"],
            "optimizer": {"type": "grid", "max_evals": 8},
            "parameters": [
                {
                    "name": "geo.prompt",
                    "kind": "prompt",
                    "component": "knobs",
                    "attr": "geo_prompt",
                    "candidates": ["Answer the geography question.", "Answer with ONLY the city."],
                }
            ],
        }
    )
    result = run_training(config)
    print("  declarative improvement:", round(result.improvement, 3))
    print("  best config:", result.best_config)
    data_path.unlink(missing_ok=True)

    # The file-based, real-world template (points at YOUR module):
    print("\n  See langgraph.train.yaml for the file-based template:")
    print("      from adapt_agent.optimization.config import run_training")
    print("      run_training('examples/langgraph/langgraph.train.yaml')")


def main() -> None:
    govern_demo()
    optimize_demo()
    training_config_demo()


if __name__ == "__main__":
    main()
