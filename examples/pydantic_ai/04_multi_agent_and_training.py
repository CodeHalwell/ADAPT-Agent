"""Example 04: A multi-agent Pydantic AI system, governed and trained.

Pydantic AI has **no built-in orchestrator**. The idiomatic way to build a
multi-agent system is to write a plain Python function that routes a query to
several specialist ``Agent`` objects and combines their results (the docs call
this "programmatic hand-off"; "agent delegation" -- one agent calling another
inside a tool -- is the other pattern). This example uses an orchestrator
*function* driving two specialists:

* ``researcher`` -- gathers a few candidate facts for the question.
* ``writer``     -- turns those facts into one crisp final answer.

We then show BOTH halves of ADAPT-Agent on that system:

* **Guard:** wrap the whole orchestrator as ONE governed unit with the
  ``PydanticAIAdapter`` so the entire pipeline is screened and traced.
* **Train:** wrap the same system in an ``OptimizableAgent.from_components`` and
  optimize it with ``make_default_optimizer`` -- the full pipeline (few-shot ->
  prompts -> models/hparams -> tools/skills) with tool/skill ablation and
  judge-driven *recommendations*. We optimize the whole system AND see per-agent
  knobs surface for each specialist. Finally we run the SAME training from a
  declarative YAML file via ``run_training`` (``pydantic_ai.train.yaml``).

Everything runs offline (Pydantic AI ``FunctionModel`` + a deterministic judge
stub); no API key or network is needed.

Run it with:

    python examples/pydantic_ai/04_multi_agent_and_training.py
"""

from __future__ import annotations

import logging
from pathlib import Path

# --- Friendly skip if Pydantic AI is not installed ------------------------- #
try:
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
except ImportError:
    raise SystemExit(
        "This example needs Pydantic AI: pip install 'adapt-agent[pydantic-ai]'\n"
        "(or: pip install pydantic-ai)"
    ) from None

from adapt_agent import AdversarialDefense, AgentObserver, Firewall
from adapt_agent.adapters import PydanticAIAdapter
from adapt_agent.optimization import (
    EvaluationHarness,
    GoldenDataset,
    LLMJudge,
    OptimizableAgent,
    Parameter,
    ParameterKind,
    exact_match,
    make_default_optimizer,
)
from adapt_agent.optimization.config import load_training_config, run_training

# --------------------------------------------------------------------------- #
# A tiny offline "knowledge base" the researcher draws on.                    #
# --------------------------------------------------------------------------- #
_FACTS = {
    "France": [
        "Paris is the capital of France.",
        "France is in Europe.",
        "The Seine flows through it.",
    ],
    "Japan": [
        "Tokyo is the capital of Japan.",
        "Japan is an island nation.",
        "The yen is its currency.",
    ],
    "Italy": [
        "Rome is the capital of Italy.",
        "Italy is shaped like a boot.",
        "Pasta is popular there.",
    ],
    "Egypt": ["Cairo is the capital of Egypt.", "The Nile runs through Egypt.", "It has pyramids."],
}
_CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Egypt": "Cairo"}

# A routing knob the framework does not expose: how many facts to gather. The
# orchestrator reads it live, so the optimizer can tune it in place.
ROUTING = {"n_facts": 1}


class WriterConfig:
    """A live, YAML-bindable knob holding the writer's working instruction.

    The YAML training path (Part C) tunes this attribute over a small candidate
    list. It is a plain attribute on a module-level object, which is exactly what
    a YAML ``parameter`` with ``attr_path`` can bind a getter/setter to.
    """

    instruction = "Write an answer using the facts."


writer_cfg = WriterConfig()


def _country_of(question: str) -> str:
    return question.replace("What is the capital of", "").strip(" ?")


def _system_prompt_of(messages: list) -> str:
    for message in messages:
        for part in getattr(message, "parts", []):
            if type(part).__name__ == "SystemPromptPart":
                return getattr(part, "content", "") or ""
    return ""


def _user_text_of(messages: list) -> str:
    text = ""
    for message in messages:
        for part in getattr(message, "parts", []):
            if type(part).__name__ == "UserPromptPart":
                content = getattr(part, "content", "")
                text = content if isinstance(content, str) else str(content)
    return text


# --------------------------------------------------------------------------- #
# Two specialist agents backed by offline FunctionModels.                     #
# --------------------------------------------------------------------------- #
def _researcher_model(messages: list, info: AgentInfo) -> ModelResponse:
    """Echo back the gathered facts (joined). Quality is fine regardless."""
    return ModelResponse(parts=[TextPart(_user_text_of(messages))])


def _writer_model(messages: list, info: AgentInfo) -> ModelResponse:
    """Compose a final answer; crispness depends on the writer's system prompt.

    With a vague prompt the writer parrots the raw facts; once the prompt says to
    answer with ONLY the capital city name it extracts the bare city. That is the
    signal the optimizer chases when it rewrites the writer's prompt.
    """
    # The writer is "crisp" when EITHER its Pydantic AI system prompt (tuned by
    # the Python API path) OR its YAML-bound working instruction says to answer
    # with only the city name.
    system = _system_prompt_of(messages) + " " + writer_cfg.instruction
    facts_blob = _user_text_of(messages)
    crisp = "ONLY" in system or "city name" in system.lower()
    if crisp:
        # Extract the capital from the first fact like "X is the capital of Y."
        first = facts_blob.split(".")[0]
        answer = (
            first.split(" is the capital")[0].strip() if " is the capital" in first else facts_blob
        )
    else:
        answer = facts_blob  # rambling: returns all the facts verbatim
    return ModelResponse(parts=[TextPart(answer)])


def build_researcher() -> Agent:
    return Agent(
        FunctionModel(_researcher_model),
        system_prompt="Gather relevant facts for the question.",
        name="researcher",
    )


def build_writer() -> Agent:
    return Agent(
        FunctionModel(_writer_model),
        system_prompt="Write an answer using the facts.",
        name="writer",
    )


# --------------------------------------------------------------------------- #
# The orchestrator function: route -> researcher -> writer.                    #
# --------------------------------------------------------------------------- #
def make_orchestrator(researcher: Agent, writer: Agent):
    """Return a ``run(question) -> answer`` closure binding the two agents."""

    def run(question: str) -> str:
        country = _country_of(question)
        facts = _FACTS.get(country, [f"{country} is a place."])
        chosen = facts[: max(1, ROUTING["n_facts"])]
        # 1. Researcher condenses the facts.
        researched = researcher.run_sync(" ".join(chosen)).output
        # 2. Writer composes the final answer from the researched facts.
        return writer.run_sync(researched).output

    return run


def deterministic_judge_stub(prompt: str) -> str:
    """Offline judge: high score for a single clean word; rewrites vague prompts.

    Real usage: ``LLMJudge(ClaudeJudge(model="claude-opus-4-8"), adversarial=True)``.
    """
    if "CURRENT TOOLS/SKILLS" in prompt or "COMPONENT:" in prompt:
        # Advisory new-tool suggestions surfaced on result.recommendations.
        return (
            '{"tools": [{"name": "capital_lookup", '
            '"description": "Return the capital city of a country.", '
            '"rationale": "The team rambled instead of naming the capital."}]}'
        )
    if "CURRENT INSTRUCTION" in prompt or "FAILURES" in prompt:
        return "Answer with ONLY the capital city name, nothing else."
    response = ""
    if "<response>" in prompt and "</response>" in prompt:
        response = prompt.split("<response>")[1].split("</response>")[0].strip()
    score = 9 if response and " " not in response else 2
    return f'{{"score": {score}, "pass": {str(score >= 6).lower()}, "reasoning": "auto"}}'


def golden() -> GoldenDataset:
    return GoldenDataset.from_list(
        [
            {"input": "What is the capital of France?", "expected": "Paris"},
            {"input": "What is the capital of Japan?", "expected": "Tokyo"},
            {"input": "What is the capital of Italy?", "expected": "Rome"},
            {"input": "What is the capital of Egypt?", "expected": "Cairo"},
        ]
    )


# --------------------------------------------------------------------------- #
# Part A: guard the whole orchestrator as ONE governed unit.                   #
# --------------------------------------------------------------------------- #
def demo_guard() -> None:
    print("=== Part A: guard the whole multi-agent system ===")
    researcher, writer = build_researcher(), build_writer()
    orchestrator = make_orchestrator(researcher, writer)

    firewall = Firewall(max_content_length=10_000)
    firewall.add_blocked_pattern(r"(?i)ignore (all|previous) instructions")
    observer = AgentObserver()
    adapter = PydanticAIAdapter(
        firewall=firewall,
        defense=AdversarialDefense(),
        observer=observer,
        agent_id="pydantic-ai-research-team",
        block_on_violation=True,
    )
    # The adapter wraps any object with a callable run_sync/run. Our orchestrator
    # is a function, so we expose it on a tiny shim with a run_sync method.
    guarded = adapter.wrap_agent(_RunSyncShim(orchestrator))

    out = guarded.execute(
        {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
    )
    print("  team answer:", out)
    for trace in observer.get_traces():
        print(f"  trace {trace['trace_id'][:8]} status={trace['status']}")


class _RunSyncShim:
    """Expose an orchestrator function as an object with ``run_sync(prompt)``."""

    def __init__(self, fn):
        self._fn = fn

    def run_sync(self, prompt: str):
        return self._fn(prompt)


# --------------------------------------------------------------------------- #
# Part B: optimize the whole system + per-agent knobs (Python API).           #
# --------------------------------------------------------------------------- #
def demo_optimize() -> None:
    print("\n=== Part B: optimize the whole system (make_default_optimizer) ===")
    # Quiet the harmless 'read-only model_name' warnings from FunctionModel.
    logging.getLogger("adapt_agent.optimization.parameters").setLevel(logging.ERROR)

    researcher, writer = build_researcher(), build_writer()
    orchestrator = make_orchestrator(researcher, writer)

    # Wrap as a multi-component target. Each specialist is introspected and its
    # knobs are namespaced (researcher.*, writer.*). The runner drives the system.
    target = OptimizableAgent.from_components(
        components={"researcher": researcher, "writer": writer},
        runner=orchestrator,
        name="research-team",
    )

    # Declare the routing knob the framework can't expose (number of facts), bound
    # live to the ROUTING dict so the optimizer can tune it in place.
    target.add_parameter(
        Parameter(
            name="orchestrator.n_facts",
            kind=ParameterKind.ROUTING,
            value=ROUTING["n_facts"],
            bounds=(1, 3),
            step=1,
            getter=lambda: ROUTING["n_facts"],
            setter=lambda v: ROUTING.__setitem__("n_facts", int(v)),
            component="orchestrator",
        )
    )

    # Make the writer's TOOL set optimizable via drop-one ablation. (Our offline
    # writer has no real tools; we model a small pool so the ablation + new-tool
    # *recommendations* path is exercised. getter/setter bind to a live list.)
    writer_tools = {"tools": ["summarize", "format_city"]}
    target.add_tool_parameter(
        "writer.tools",
        kind=ParameterKind.TOOL,
        getter=lambda: list(writer_tools["tools"]),
        setter=lambda v: writer_tools.__setitem__("tools", list(v)),
        candidate_tools=["summarize", "format_city"],
        component="writer",
    )

    print("Tunable knobs across the team:")
    for param in target.parameters:
        print(f"  - {param.name:28} kind={param.kind.value}")

    judge = LLMJudge(deterministic_judge_stub)
    harness = EvaluationHarness(
        metrics=[exact_match(), judge.as_metric("quality")],
        primary_metric="quality",
    )

    print("\nBaseline:", harness.evaluate(target, golden()))
    print("Baseline answer:", orchestrator("What is the capital of France?"))

    # The full pipeline: few-shot -> prompts -> models/hparams -> tools/skills.
    # suggest_tools auto-enables (a judge is set), surfacing advisory new tools.
    optimizer = make_default_optimizer(harness, judge=judge, max_evals=40, seed=0)
    result = optimizer.optimize(target, golden())

    print("\nResult:", result)
    print("Improvement:", round(result.improvement, 4))
    print("Best config:", result.best_config)
    print("Answer now:", orchestrator("What is the capital of France?"))
    if result.recommendations:
        print("\nJudge recommendations (advisory new tools/skills):")
        for tip in result.recommendations:
            print("  -", tip)


# --------------------------------------------------------------------------- #
# Part C: the SAME training from a declarative YAML config.                    #
# --------------------------------------------------------------------------- #
# These module-level objects are what the YAML resolves via "module:attribute".
researcher = build_researcher()
writer = build_writer()
yaml_orchestrator = make_orchestrator(researcher, writer)


def yaml_run(question: str) -> str:
    """Entrypoint the YAML config points at (``__main__:yaml_run``)."""
    return yaml_orchestrator(question)


def demo_yaml() -> None:
    print("\n=== Part C: run the same training from pydantic_ai.train.yaml ===")
    logging.getLogger("adapt_agent.optimization.parameters").setLevel(logging.ERROR)

    yaml_path = Path(__file__).with_name("pydantic_ai.train.yaml")
    # The YAML references a golden dataset on disk; write a tiny one next to it.
    data_path = yaml_path.with_name("_golden.jsonl")
    data_path.write_text(
        "\n".join(
            f'{{"input": "What is the capital of {c}?", "expected": "{cap}"}}'
            for c, cap in _CAPITALS.items()
        ),
        encoding="utf-8",
    )

    # Load the YAML into a TrainingConfig, then run it. (For an in-process dict
    # instead of a file, use config.parse_training_config -- see the docs.)
    config = load_training_config(yaml_path)
    # Dataset paths in the YAML are relative to the process CWD, not the file, so
    # point it at the absolute path we just wrote next to the config.
    config.dataset.path = str(data_path)
    print("Baseline answer:", yaml_run("What is the capital of France?"))
    result = run_training(config)
    print("Result:", result)
    print("Best config:", result.best_config)
    print("Answer now:", yaml_run("What is the capital of France?"))
    data_path.unlink(missing_ok=True)


def main() -> None:
    demo_guard()
    demo_optimize()
    demo_yaml()


if __name__ == "__main__":
    main()
