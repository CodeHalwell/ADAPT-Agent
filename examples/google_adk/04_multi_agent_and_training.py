"""Example 04 (Google ADK): a multi-agent system, governed and trained (offline).

A realistic Google ADK system is a *coordinator* ``LlmAgent`` with specialist
``sub_agents``. ADK routes a request to a sub-agent by name (the coordinator's
LLM decides, using each sub-agent's ``description``). ADAPT-Agent treats the whole
thing as ONE governed unit *and* introspects every agent in the tree:
``sub_agents`` are walked recursively, so each specialist's prompt/model/tools
become tunable knobs namespaced under the coordinator.

This example, fully offline (no API key, no network):

1. Builds an ADK-shaped coordinator with two sub-agents (a "math" specialist with
   tools and a "geo" specialist). The objects match ADK's duck type, so the
   ``google_adk`` introspector recognises them.
2. Wraps the whole system as one :class:`OptimizableAgent` via
   ``from_components`` with a single ``runner`` that drives the coordinator.
3. Adds an explicit TOOL ablation knob (the optimizer can drop a tool) and lets
   ``make_default_optimizer`` run the full pipeline (few-shot -> prompts ->
   models/hparams -> tools/skills), with the adversarial judge proposing brand
   new tools/skills surfaced on ``result.recommendations``.
4. Shows the parallel **YAML training** path: the same run encoded declaratively
   and executed with ``run_training`` (mirrors ``google_adk.train.yaml``).

Run it with:

    python examples/google_adk/04_multi_agent_and_training.py
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

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
from adapt_agent.optimization.introspection import introspect


# --- A Google-ADK-shaped multi-agent tree (no google.adk import) ----------- #
# Each agent matches ADK's LlmAgent duck type: name + instruction + sub_agents,
# and no foreign-framework attributes. The coordinator's sub_agents are walked
# recursively by the introspector.
def add(a: int, b: int) -> int:
    """A tool the math specialist can call."""
    return a + b


def multiply(a: int, b: int) -> int:
    """A second tool, so tool *selection* becomes a real search space."""
    return a * b


def build_system() -> SimpleNamespace:
    math_agent = SimpleNamespace(
        name="math_agent",
        model="gemini-flash-latest",
        instruction="Do the arithmetic.",
        global_instruction=None,
        generate_content_config=SimpleNamespace(temperature=0.3, top_p=1.0, max_output_tokens=128),
        tools=[add, multiply],  # 2+ tools -> drop-one ablation candidates
        sub_agents=[],
    )
    geo_agent = SimpleNamespace(
        name="geo_agent",
        model="gemini-flash-latest",
        instruction="Answer the geography question.",
        global_instruction=None,
        generate_content_config=SimpleNamespace(temperature=0.5, top_p=1.0, max_output_tokens=128),
        tools=[],
        sub_agents=[],
    )
    coordinator = SimpleNamespace(
        name="coordinator",
        model="gemini-flash-latest",
        instruction="Route the request to the right specialist.",
        global_instruction=None,
        generate_content_config=SimpleNamespace(temperature=0.2, top_p=1.0, max_output_tokens=64),
        tools=[],
        sub_agents=[math_agent, geo_agent],  # the routing tree
    )
    return coordinator


_CAPITALS = {"france": "Paris", "japan": "Tokyo", "italy": "Rome"}


def make_runner(coordinator: SimpleNamespace):
    """Drive the system; route to a sub-agent and honour its live prompt/tools."""
    math_agent, geo_agent = coordinator.sub_agents

    def run(question: str) -> str:
        q = question.lower()
        if any(op in question for op in ("+", "*", "-")) or "plus" in q or "times" in q:
            # The math specialist answers cleanly only when it still has the tool
            # it needs. There is no subtraction tool, so "-" always fails - the
            # gap the adversarial judge flags as a missing tool.
            tool_names = {getattr(t, "__name__", str(t)) for t in math_agent.tools}
            a, b = _two_numbers(question)
            if "+" in question or "plus" in q:
                return str(add(a, b)) if "add" in tool_names else "no add tool"
            if "*" in question or "times" in q:
                return str(multiply(a, b)) if "multiply" in tool_names else "no multiply tool"
            return "no subtraction tool"  # no tool for "-"
        country = q.replace("what is the capital of", "").strip(" ?")
        if "only" in geo_agent.instruction.lower():
            return _CAPITALS.get(country, "unknown")
        return f"The capital of {country} is a city."

    return run


def _two_numbers(text: str) -> tuple[int, int]:
    # Pull the first two integers regardless of operator spacing ("10-4" or "10 - 4").
    nums = [int(n) for n in re.findall(r"\d+", text)] + [0, 0]
    return (nums[0], nums[1])


def deterministic_judge_stub(prompt: str, system: str | None = None) -> str:
    """Offline judge: rewrites prompts, grades answers, and (adversarial) proposes tools."""
    rubric = system or ""
    if "agent architect" in rubric:
        # suggest_tools path: propose NEW capabilities from the failures. The
        # judge expects {"tools": [{name, description, rationale}, ...]}.
        return (
            '{"tools": [{"name": "subtract", "description": "Subtract two '
            'integers", "rationale": "The math agent has add and multiply but '
            'cannot handle subtraction inputs like 10 - 4."}]}'
        )
    if "prompt engineer" in rubric or "CURRENT INSTRUCTION" in prompt:
        return "Answer with ONLY the final value or city name, nothing else."
    import re

    match = re.search(r"<response>(.*?)</response>", prompt, re.DOTALL)
    response = (match.group(1) if match else prompt).strip()
    score = 9 if response and " " not in response and "no " not in response else 2
    return f'{{"score": {score}, "pass": {str(score >= 6).lower()}, "reasoning": "auto"}}'


def run_code_path() -> None:
    print("=== Code path: make_default_optimizer over the whole ADK tree ===")
    coordinator = build_system()

    # Introspection walks sub_agents recursively; note the namespaced names.
    print("Discovered knobs across the tree:")
    for p in introspect(coordinator):
        print(f"  - {p.name:42} kind={p.kind.value}")

    data = GoldenDataset.from_list(
        [
            {"input": "2 + 3", "expected": "5"},
            {"input": "4 * 5", "expected": "20"},
            {"input": "What is the capital of France?", "expected": "Paris"},
            {"input": "What is the capital of Japan?", "expected": "Tokyo"},
            # The math specialist has no subtraction tool, so this case keeps
            # failing no matter how we tune existing knobs - exactly the kind of
            # gap the adversarial judge flags as a missing tool/skill on
            # ``result.recommendations``.
            {"input": "10 - 4", "expected": "6"},
        ]
    )

    # Wrap the whole system as ONE optimizable unit. We pass all live agents as
    # components (so their knobs are tunable) and one runner that drives the tree.
    target = OptimizableAgent.from_components(
        components={
            "coordinator": coordinator,
            "math_agent": coordinator.sub_agents[0],
            "geo_agent": coordinator.sub_agents[1],
        },
        runner=make_runner(coordinator),
        name="adk-team",
    )

    # Introspection already exposed each agent's `tools` as a drop-one ablation
    # knob. To show how you declare a knob the framework does NOT expose, add a
    # SKILL-level selection knob for the math specialist (a higher-level "skill"
    # allow-list). `candidate_tools` becomes drop-one ablation subsets, so skill
    # *selection* is a real search space the optimizer explores.
    math_agent = coordinator.sub_agents[0]
    math_agent.skills = ["arithmetic", "word_problems"]
    target.add_tool_parameter(
        "math_agent.skills",
        kind=ParameterKind.SKILL,
        getter=lambda: math_agent.skills,
        setter=lambda skills: setattr(math_agent, "skills", list(skills)),
        candidate_tools=["arithmetic", "word_problems"],
        component="math_agent",
    )

    judge = LLMJudge(deterministic_judge_stub)
    harness = EvaluationHarness(
        metrics=[exact_match(), judge.as_metric("quality")],
        primary_metric="exact_match",
    )

    print("\nBaseline:", harness.evaluate(target, data))

    # The default optimizer runs the full pipeline and, with suggest_tools on,
    # asks the (adversarial) judge to propose NEW tools/skills from failures.
    optimizer = make_default_optimizer(
        harness, judge=judge, max_evals=60, seed=0, suggest_tools=True
    )
    result = optimizer.optimize(target, data)

    print("improvement:", round(result.improvement, 3))
    print("best_config:", result.best_config)
    if result.recommendations:
        print("Judge recommendations (advisory new tools/skills):")
        for tip in result.recommendations:
            print("  -", tip)


def run_yaml_path() -> None:
    print("\n=== YAML path: the same run encoded declaratively (run_training) ===")
    # A tiny golden dataset on disk.
    rows = [
        {"input": "France", "expected": "ANSWER:France"},
        {"input": "Japan", "expected": "ANSWER:Japan"},
    ]
    tmp = Path(tempfile.mkdtemp())
    data_path = tmp / "golden.jsonl"
    data_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    # The config mirrors google_adk.train.yaml. We point entrypoint/components at
    # objects in THIS module so it runs with no project install. A real config
    # uses "mypackage.module:attribute" references and a real judge provider.
    config = parse_training_config(
        {
            "target": {
                "entrypoint": "__main__:yaml_run",
                "components": {"cfg": "__main__:yaml_cfg"},
            },
            "dataset": {"path": str(data_path), "format": "jsonl"},
            "metrics": ["exact_match"],
            "optimizer": {"type": "coordinate_ascent", "max_evals": 10, "seed": 0},
            "parameters": [
                {
                    "name": "cfg.prefix",
                    "kind": "prompt",
                    "component": "cfg",
                    "attr": "prefix",
                    "candidates": ["", "ANSWER:"],
                }
            ],
        }
    )
    print("Baseline:", yaml_run("France"))
    result = run_training(config)
    print("improvement:", round(result.improvement, 3))
    print("best_config:", result.best_config)
    print("prefix now:", repr(yaml_cfg.prefix), "-> answer:", yaml_run("France"))


# Module-level objects the YAML entrypoint/components resolve to.
yaml_cfg = SimpleNamespace(prefix="")


def yaml_run(country: str) -> str:
    return f"{yaml_cfg.prefix}{country}"


def main() -> None:
    run_code_path()
    run_yaml_path()


if __name__ == "__main__":
    main()
