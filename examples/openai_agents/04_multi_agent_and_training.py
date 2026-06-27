"""OpenAI Agents SDK example 04: a multi-agent system, governed and trained.

The top rung. We build a realistic OpenAI Agents SDK topology -- a *triage*
``Agent`` that ``handoffs`` to two specialist ``Agent``s, each carrying a
``@function_tool`` -- then:

1. **Govern the whole system as ONE unit** by wrapping the triage agent with the
   ``OpenAIAgentsAdapter`` (the firewall screens every request regardless of which
   specialist ultimately answers).
2. **Optimize the whole system AND each agent individually.** ``introspect()``
   recurses through the ``handoffs`` graph, so the search space spans the triage
   agent's routing plus every specialist's instructions / model / tools. We then
   run ``make_default_optimizer`` (few-shot -> prompts -> models/hparams/routing
   -> tools/skills) including drop-one **tool ablation** and judge-driven new-tool
   **recommendations**.
3. **Show the parallel YAML-config path** (``run_training``) that encodes the same
   run declaratively -- see ``openai_agents.train.yaml`` in this directory.

Runs **offline, no API key**: the judge is a deterministic stub and the system is
driven by a local runner that reads the live agents' instructions (a real run
would use ``Runner.run_sync(triage, prompt)``).

Run it with:

    python examples/openai_agents/04_multi_agent_and_training.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

try:
    import agents  # noqa: F401
except ImportError:
    raise SystemExit(
        "This example needs the OpenAI Agents SDK: "
        "pip install 'adapt-agent[openai-agents]'  (or: pip install openai-agents)"
    ) from None

from agents import Agent, function_tool

from adapt_agent import Firewall
from adapt_agent.adapters import OpenAIAgentsAdapter
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

# --------------------------------------------------------------------------- #
# 1. The multi-agent system (a triage agent with handoffs to specialists).    #
# --------------------------------------------------------------------------- #


@function_tool
def capital_lookup(country: str) -> str:
    """Return the capital city of a country."""
    return {"France": "Paris", "Japan": "Tokyo"}.get(country, "unknown")


@function_tool
def population_lookup(country: str) -> str:
    """Return the (rough) population of a country."""
    return {"France": "68M", "Japan": "125M"}.get(country, "unknown")


@function_tool
def area_lookup(country: str) -> str:
    """Return the land area of a country (an unhelpful extra tool for ablation)."""
    return {"France": "551,695 km2", "Japan": "377,975 km2"}.get(country, "unknown")


geography_agent = Agent(
    name="Geography",
    instructions="Answer geography questions.",
    handoff_description="Handles questions about capitals and locations.",
    # Two tools, so introspection turns `geography.tools` into a real drop-one
    # ABLATION search space (the optimizer can learn that area_lookup is dead weight).
    tools=[capital_lookup, area_lookup],
)

demographics_agent = Agent(
    name="Demographics",
    instructions="Answer population questions.",
    handoff_description="Handles questions about population and demographics.",
    tools=[population_lookup],
)

triage_agent = Agent(
    name="Triage",
    instructions="Route the question to the right specialist.",
    handoffs=[geography_agent, demographics_agent],
)


# --------------------------------------------------------------------------- #
# 2. An OFFLINE runner so the example needs no API key. It reads the LIVE      #
#    agents (instructions + which tools are still attached) so optimizing them #
#    measurably changes behaviour. A real deployment uses Runner.run_sync.     #
# --------------------------------------------------------------------------- #


def run_system(question: str) -> str:
    """Drive the triage -> specialist flow deterministically and offline."""
    is_pop = "population" in question.lower()
    specialist = demographics_agent if is_pop else geography_agent
    country = "Japan" if "Japan" in question else "France"

    # The specialist can only answer if it still has its lookup tool (drop-one
    # ablation may remove it) and its instructions ask for a terse reply.
    tool_names = {getattr(t, "name", getattr(t, "__name__", "")) for t in specialist.tools}
    needed = "population_lookup" if is_pop else "capital_lookup"
    terse = "ONLY" in specialist.instructions or "concise" in specialist.instructions.lower()
    if needed not in tool_names:
        return "I cannot answer that without my lookup tool."
    table = {"France": "68M", "Japan": "125M"} if is_pop else {"France": "Paris", "Japan": "Tokyo"}
    value = table.get(country, "unknown")
    return value if terse else f"The answer is probably {value}, give or take."


def _extract_fence(prompt: str, label: str) -> str:
    """Pull the text inside a <label>...</label> fence the judge builds."""
    start = prompt.find(f"<{label}>")
    end = prompt.find(f"</{label}>")
    if start == -1 or end == -1:
        return ""
    return prompt[start + len(label) + 2 : end].strip()


def deterministic_judge_stub(prompt: str) -> str:
    """Offline LLM-judge stub. Branches on the kind of user prompt it receives:

    * ``CURRENT INSTRUCTION:`` -> a prompt-rewrite request; return a terser prompt.
    * ``OBSERVED FAILURES`` + ``COMPONENT:`` -> a tool-suggestion request; return a
      JSON list of proposed new tools (surfaced on ``result.recommendations``).
    * otherwise a grading request with a ``<response>`` fence -> return a JSON score.
    """
    if "CURRENT INSTRUCTION:" in prompt:
        return "Answer concisely with ONLY the value, nothing else."
    if "COMPONENT:" in prompt and "OBSERVED FAILURES" in prompt:
        return (
            '{"tools": [{"name": "currency_lookup", '
            '"description": "Look up a country\'s currency", '
            '"rationale": "finance questions currently fail"}]}'
        )
    response = _extract_fence(prompt, "response")
    score = 9 if response and len(response.split()) <= 3 else 3
    return f'{{"score": {score}, "pass": {str(score >= 6).lower()}, "reasoning": "auto"}}'


def build_dataset() -> GoldenDataset:
    return GoldenDataset.from_list(
        [
            {"input": "What is the capital of France?", "expected": "Paris"},
            {"input": "What is the capital of Japan?", "expected": "Tokyo"},
            {"input": "What is the population of France?", "expected": "68M"},
            {"input": "What is the population of Japan?", "expected": "125M"},
        ]
    )


def optimize_in_process() -> None:
    """Optimize the whole multi-agent system (and every agent within it)."""
    print("=== Guard the whole system as ONE unit ===")
    firewall = Firewall(max_content_length=10_000)
    firewall.add_blocked_pattern(r"ignore (all|previous) instructions", flags=2)  # re.IGNORECASE
    adapter = OpenAIAgentsAdapter(firewall=firewall, agent_id="triage-system")
    # Wrapping the triage agent governs every handoff path behind it.
    guarded = adapter.wrap_agent(run_system)  # offline runner; swap in triage_agent live
    print("  Guarded runner ready:", guarded is not None)

    print("\n=== Introspection recurses through handoffs ===")
    # introspect(triage_agent) walks triage -> geography -> demographics, so we
    # get the whole system's knobs (routing on triage, prompts/tools per agent).
    for p in introspect(triage_agent):
        print(f"  - {p.name}  (kind={p.kind.value})")

    # Register all three agents as components so each is tuned individually, and
    # provide the runner that drives the whole system end to end.
    target = OptimizableAgent.from_components(
        components={
            "triage": triage_agent,
            "geography": geography_agent,
            "demographics": demographics_agent,
        },
        runner=run_system,
        name="triage-team",
    )

    # Tool/skill ablation. Introspection already made `geography.tools` a drop-one
    # search space because the agent has >=2 tools. We ADDITIONALLY declare a
    # higher-level SKILL knob (a name no framework attribute exposes) to show how
    # to bind a custom ablation space via getter/setter. The candidate sets are
    # the full skill list first, then each drop-one subset.
    skills = {"geography": ["map_render", "distance_calc"]}
    target.add_tool_parameter(
        "geography.skills",
        kind=ParameterKind.SKILL,
        getter=lambda: skills["geography"],
        setter=lambda v: skills.__setitem__("geography", v),
        candidate_tools=["map_render", "distance_calc"],
        component="geography",
    )

    print("\n=== Optimize whole-system + per-agent ===")
    judge = LLMJudge(deterministic_judge_stub)
    harness = EvaluationHarness(
        metrics=[exact_match(), judge.as_metric("quality")],
        primary_metric="quality",
    )
    data = build_dataset()
    print("Baseline:", harness.evaluate(target, data))

    optimizer = make_default_optimizer(harness, judge=judge, max_evals=24, seed=0)
    result = optimizer.optimize(target, data)
    print("\nResult:", result)
    print("Improvement:", result.improvement)
    print("Best config:", result.best_config)
    if result.recommendations:
        print("\nJudge recommendations (advisory new tools/skills):")
        for tip in result.recommendations:
            print("  -", tip)


# --------------------------------------------------------------------------- #
# 3. The parallel YAML-config path. The same run, encoded declaratively.       #
#    We build the config in-process pointing at THIS module's objects so it    #
#    runs offline; in a project you would ship openai_agents.train.yaml and    #
#    call run_training("openai_agents.train.yaml").                            #
# --------------------------------------------------------------------------- #


def optimize_from_config() -> None:
    print("\n=== Same run via a declarative training config (offline) ===")
    rows = [
        {"input": "What is the capital of France?", "expected": "Paris"},
        {"input": "What is the population of Japan?", "expected": "125M"},
    ]
    tmp = Path(tempfile.mkdtemp())
    data_path = tmp / "golden.jsonl"
    data_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    config = parse_training_config(
        {
            "target": {
                "entrypoint": "__main__:run_system",
                "components": {
                    "triage": "__main__:triage_agent",
                    "geography": "__main__:geography_agent",
                    "demographics": "__main__:demographics_agent",
                },
            },
            "dataset": {"path": str(data_path), "format": "jsonl"},
            # Offline + deterministic: score with the built-in exact_match metric
            # and NO judge, so this needs no network or API key. To train for real,
            # add a `judge:` block (provider: anthropic / openai, model: ...) -- see
            # the `judge:` section in openai_agents.train.yaml.
            "metrics": ["exact_match"],
            "optimizer": {"type": "coordinate_ascent", "max_evals": 8, "seed": 0},
        }
    )
    result = run_training(config)
    print("  Config-driven result:", result)
    print("  Best config:", result.best_config)


def main() -> None:
    optimize_in_process()
    optimize_from_config()


if __name__ == "__main__":
    main()
