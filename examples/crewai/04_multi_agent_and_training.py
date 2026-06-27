"""Example 04 (CrewAI): a multi-agent crew, governed AND optimized end to end.

This is the capstone. It builds a realistic two-agent CrewAI crew -- a
*researcher* (with tools) handing off to a *writer* -- and then:

1. Governs the WHOLE crew as one unit with ``CrewAIAdapter`` (the same wrap as
   example 01, but over a multi-agent crew).
2. Optimizes the whole crew with ``make_default_optimizer`` -- the full pipeline
   that tunes few-shot blocks, then prompts, then models/hyperparameters, then
   **tools/skills** (drop-one ablation), guided by an adversarial offline judge
   that also proposes NEW tools/skills (``result.recommendations``).
3. Shows the **parallel YAML-config path**: the same run expressed declaratively
   in ``crewai.train.yaml`` and executed with ``run_training(...)``.

CrewAI is multi-agent by construction, so "optimize the whole system" and
"optimize each agent" are the same call here: the CrewAI introspector flattens
the crew into per-agent knobs (role/goal/backstory, llm model+temp+max_tokens,
tools, max_iter) plus per-task knobs (description/expected_output), and the
optimizer searches across all of them at once.

Everything runs offline -- a deterministic local LLM for the crew and a
deterministic stub judge -- so there is NO API key and NO network.

Run it with:

    python examples/crewai/04_multi_agent_and_training.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from crewai import LLM, Agent, Crew, Process, Task
    from crewai.tools import tool
except ImportError:
    raise SystemExit(
        "This example needs CrewAI: pip install 'adapt-agent[crewai]'\n" "(or: pip install crewai)"
    )

from adapt_agent import Firewall
from adapt_agent.adapters import CrewAIAdapter
from adapt_agent.optimization import (
    EvaluationHarness,
    GoldenDataset,
    LLMJudge,
    OptimizableAgent,
    exact_match,
    make_default_optimizer,
    token_f1,
)
from adapt_agent.optimization.introspection import introspect
from adapt_agent.optimization.providers import ModelProvider, register_provider

# --------------------------------------------------------------------------- #
# A tiny knowledge base + CrewAI tools.                                        #
# --------------------------------------------------------------------------- #
_FACTS = {
    "France": "France's capital is Paris, on the Seine.",
    "Japan": "Japan's capital is Tokyo, on Honshu.",
    "Italy": "Italy's capital is Rome, on the Tiber.",
    "Egypt": "Egypt's capital is Cairo, on the Nile.",
}


@tool("Knowledge Base Lookup")
def kb_lookup(country: str) -> str:
    """Return a factual sentence about a country's capital from the knowledge base."""
    for name, fact in _FACTS.items():
        if name.lower() in country.lower():
            return fact
    return "No fact found."


@tool("Distracting Web Search")
def distracting_search(query: str) -> str:
    """A noisy tool whose output is unhelpful -- the optimizer should learn to drop it."""
    return "Top result: 10 surprising travel hacks you won't believe!"


# --------------------------------------------------------------------------- #
# Deterministic, network-free crew model.                                     #
# --------------------------------------------------------------------------- #
class OfflineLLM(LLM):
    """A deterministic LLM whose answer quality depends on the live prompts.

    It produces a clean ``"<Country>: <City>"`` answer only when the writer's
    goal has been tightened to demand a terse format, giving the optimizer a real
    signal. A real crew swaps in ``LLM(model="openai/gpt-4o", temperature=0.2)``.
    """

    writer_goal: str = ""

    def __init__(self) -> None:
        super().__init__(model="offline/echo")

    def call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
        text = _stringify(messages)
        country = next((c for c in _FACTS if c in text), "")
        city = {
            "France": "Paris",
            "Japan": "Tokyo",
            "Italy": "Rome",
            "Egypt": "Cairo",
        }.get(country, "unknown")
        terse = "only" in self.writer_goal.lower() or "format" in self.writer_goal.lower()
        if terse:
            return f"{country}: {city}"
        return f"After much research, the capital appears to be the great city of {city}."


def _stringify(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        return " ".join(_stringify(m) for m in messages)
    if isinstance(messages, dict):
        return " ".join(str(v) for v in messages.values())
    return str(messages)


# --------------------------------------------------------------------------- #
# Build the multi-agent crew (module-level so the YAML path can resolve it).   #
# --------------------------------------------------------------------------- #
def build_crew() -> tuple[Crew, OfflineLLM, Agent]:
    llm = OfflineLLM()

    researcher = Agent(
        role="Researcher",
        goal="Find the factual capital of the given country using your tools.",
        backstory="A diligent fact-checker who relies on the knowledge base.",
        llm=llm,
        tools=[kb_lookup, distracting_search],  # one useful, one distracting
        max_iter=4,
        verbose=False,
    )
    writer = Agent(
        role="Writer",
        # Deliberately loose -- the optimizer should tighten this.
        goal="Write the answer.",
        backstory="An editor who turns research notes into the final answer.",
        llm=llm,
        max_iter=3,
        verbose=False,
    )
    llm.writer_goal = writer.goal

    research_task = Task(
        description="Research the capital of: {question}",
        expected_output="A factual note about the capital.",
        agent=researcher,
    )
    write_task = Task(
        description="Write the final answer about {question} from the research note.",
        expected_output="The final answer.",
        agent=writer,
    )
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=False,
    )
    return crew, llm, writer


# Module-level singletons so ``crewai.train.yaml`` can reference them as
# ``examples.crewai.04_multi_agent_and_training:CREW`` / ``:run``.
CREW, _LLM, _WRITER = build_crew()


def run(question: str) -> str:
    """Entrypoint that drives the whole crew and returns plain text for scoring."""
    _LLM.writer_goal = _WRITER.goal  # re-sync after any in-place prompt rewrite
    out = CREW.kickoff(inputs={"question": question})
    return getattr(out, "raw", str(out))


# --------------------------------------------------------------------------- #
# Offline judge: a deterministic stub usable both directly and via the YAML.   #
# --------------------------------------------------------------------------- #
def judge_stub(prompt: str) -> str:
    """Grade terse 'Country: City' answers highly; also rewrite prompts/propose tools."""
    if "Rewrite" in prompt:
        return "Write ONLY 'Country: City' with no extra words."
    if "propose" in prompt.lower() or "new tool" in prompt.lower():
        return "Consider adding a citation-checking tool to verify each fact."
    response = prompt.split("RESPONSE:")[-1].split("REFERENCE")[0].strip()
    score = 9 if ":" in response and len(response.split()) <= 4 else 3
    return f'{{"score": {score}, "pass": {str(score >= 6).lower()}, "reasoning": "auto"}}'


class StubProvider(ModelProvider):
    """Wraps :func:`judge_stub` as a named provider so the YAML can select it.

    Registered under ``"stub"`` below; the YAML's ``judge.provider: stub`` then
    resolves to this offline, network-free provider.
    """

    name = "stub"

    def __init__(self, model: str = "stub", **kw: Any):
        super().__init__(model, **kw)

    def complete(self, prompt: str, **overrides: Any) -> str:
        return judge_stub(prompt)


register_provider("stub", StubProvider)


def _reset_crew() -> None:
    """Rebuild the module-level crew so a fresh run starts from the baseline."""
    global CREW, _LLM, _WRITER
    CREW, _LLM, _WRITER = build_crew()


# --------------------------------------------------------------------------- #
# Main: govern, then optimize (Python path), then optimize (YAML path).        #
# --------------------------------------------------------------------------- #
def main() -> None:
    # === Part A: govern the whole multi-agent crew as one unit ============== #
    print("=== Govern the whole crew ===")
    adapter = CrewAIAdapter(
        firewall=Firewall(max_content_length=10_000),
        agent_id="demo-crewai-research-crew",
        block_on_violation=True,
    )
    guarded = adapter.wrap_agent(CREW)
    out = guarded.execute(
        {
            "messages": [{"role": "user", "content": "Capital of France?"}],
            "question": "France",
        }
    )
    print("  guarded crew output:", getattr(out, "raw", out))

    # === Part B: optimize the whole crew (Python path) ===================== #
    print("\n=== Optimize the whole crew (make_default_optimizer) ===")
    target = OptimizableAgent.from_components(
        components={"crew": CREW},
        runner=run,
        name="crewai-research-crew",
    )
    print("Discovered knobs across BOTH agents + tasks:")
    for p in introspect(CREW):
        print(f"  - {p.name:<28} kind={p.kind.value}")

    data = GoldenDataset.from_list(
        [
            {"input": "France", "expected": "France: Paris"},
            {"input": "Japan", "expected": "Japan: Tokyo"},
            {"input": "Italy", "expected": "Italy: Rome"},
            {"input": "Egypt", "expected": "Egypt: Cairo"},
        ]
    )

    # adversarial=True: the judge grades like a harsh critic AND, with the
    # optimizer's suggest_tools on, proposes new tools/skills from failures.
    judge = LLMJudge(judge_stub, adversarial=True)
    harness = EvaluationHarness(
        metrics=[exact_match(), token_f1(), judge.as_metric("quality")],
        primary_metric="quality",
    )

    print("\nBaseline:", harness.evaluate(target, data))
    print("Baseline answer for France:", run("France"))

    optimizer = make_default_optimizer(harness, judge=judge, max_evals=20, seed=0)
    result = optimizer.optimize(target, data)

    print("\nResult:", result)
    print("Improvement:", result.improvement)
    print("Final writer goal applied in place:", repr(_WRITER.goal))
    print("Answer for France now:", run("France"))
    if result.recommendations:
        print("\nJudge recommendations (advisory new tools/skills):")
        for tip in result.recommendations:
            print("  -", tip)

    # === Part C: the SAME run via the declarative YAML config ============== #
    print("\n=== Optimize via crewai.train.yaml (run_training) ===")
    # Reset the crew so the YAML path starts from the loose baseline again.
    _reset_crew()

    here = Path(__file__).resolve().parent
    golden = here / "_golden.jsonl"
    golden.write_text(
        "\n".join(
            json.dumps({"input": k, "expected": f"{k}: {v}"})
            for k, v in {
                "France": "Paris",
                "Japan": "Tokyo",
                "Italy": "Rome",
                "Egypt": "Cairo",
            }.items()
        ),
        encoding="utf-8",
    )
    try:
        from adapt_agent.optimization.config import run_training

        yaml_result = run_training(str(here / "crewai.train.yaml"))
        print("YAML result:", yaml_result)
        print("Improvement:", yaml_result.improvement)
    finally:
        golden.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
