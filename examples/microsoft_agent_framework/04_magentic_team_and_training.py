"""Example 04: A Magentic team, governed as one unit and trained as a whole.

This is the realistic, end-to-end scenario for Microsoft Agent Framework:

* An **orchestrator** (the Magentic *manager*) coordinates **four specialist
  ``ChatAgent``s** (researcher, writer, coder, reviewer) via dynamic planning.
* The whole team is built with ``MagenticBuilder`` and exposed as ONE agent with
  ``workflow.as_agent(name=...)`` -- a ``WorkflowAgent`` with an async ``run``
  whose result carries ``.text``.
* That single object is **guarded** by ``MicrosoftAgentFrameworkAdapter`` (one
  firewall/policy/observer wraps the entire team), and **trained** offline by
  ADAPT-Agent in two complementary ways:
    1. as a whole (register all 5 agents as ``components``; the runner drives the
       wrapped workflow agent), and
    2. each agent individually (introspect each ``ChatAgent`` on its own).

The real builder call (verified against Microsoft Learn) is::

    from agent_framework import MagenticBuilder
    from agent_framework.openai import OpenAIChatClient

    client = OpenAIChatClient(model_id="gpt-4o")
    researcher = client.create_agent(name="researcher", instructions="...")
    writer     = client.create_agent(name="writer",     instructions="...")
    coder      = client.create_agent(name="coder",      instructions="...")
    reviewer   = client.create_agent(name="reviewer",   instructions="...")
    manager    = client.create_agent(name="manager",    instructions="...")

    workflow = (
        MagenticBuilder()
        .participants(researcher=researcher, writer=writer, coder=coder, reviewer=reviewer)
        .with_standard_manager(
            agent=manager,
            max_round_count=8,      # total coordination rounds
            max_stall_count=3,      # consecutive no-progress rounds before replan
            max_reset_count=2,      # full resets before giving up
        )
        .build()                    # -> Workflow
    )
    team_agent = workflow.as_agent(name="research-team")   # -> WorkflowAgent
    response = await team_agent.run("Write a short report on X")
    print(response.text)

TWO GOTCHAS THIS EXAMPLE TEACHES
--------------------------------
1. ``as_agent()`` exposes NO introspectable knobs -- it is an opaque wrapper. So
   to tune the team you register the five underlying ``ChatAgent``s as
   ``components`` and run the *workflow agent* as the runner.
2. The Magentic routing limits (``max_round_count`` etc.) live on the manager
   config, not on any agent attribute, so they are NOT auto-discovered. We
   declare them as explicit ROUTING ``Parameter``s whose **setter rebuilds the
   workflow** (the only way to change them) and re-points the runner.

Runs fully offline (no API key, no network) using stand-ins that mirror the real
API surface. Swap in the real classes from the snippet above with no other
changes. Run it with:

    python examples/microsoft_agent_framework/04_magentic_team_and_training.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from adapt_agent import AgentObserver, Firewall
from adapt_agent.adapters import MicrosoftAgentFrameworkAdapter
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
from adapt_agent.optimization.introspection import introspect

# =========================================================================== #
# OFFLINE STAND-INS that mirror the real agent_framework API surface.
# Replace this whole block with the real imports/classes from the docstring.
# =========================================================================== #


@dataclass
class AgentRunResponse:
    text: str


@dataclass
class FakeChatClient:
    """Stands in for OpenAIChatClient: holds the model + sampling settings."""

    model_id: str = "gpt-4o-mini"
    temperature: float = 0.5

    def create_agent(self, **kwargs) -> ChatAgent:
        return ChatAgent(chat_client=self, **kwargs)


@dataclass
class ChatAgent:
    """Offline stand-in for a Microsoft ``ChatAgent`` (introspectable shape)."""

    instructions: str
    chat_client: FakeChatClient = field(default_factory=FakeChatClient)
    name: str = "agent"
    tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)

    async def run(self, prompt: str) -> AgentRunResponse:  # pragma: no cover - toy
        return AgentRunResponse(text=f"[{self.name}] {prompt}")


@dataclass
class _Workflow:
    """Stand-in for a built Magentic ``Workflow`` (only what we need)."""

    participants: dict[str, ChatAgent]
    manager: ChatAgent
    max_round_count: int

    def as_agent(self, name: str | None = None) -> WorkflowAgent:
        return WorkflowAgent(self, name=name or "magentic-team")


class WorkflowAgent:
    """Stand-in for the ``WorkflowAgent`` returned by ``Workflow.as_agent()``.

    Mirrors the real one: an opaque agent with an async ``run(prompt)`` returning
    something with ``.text``. Crucially it exposes NO ``instructions`` /
    ``chat_client``, so ADAPT-Agent cannot introspect it (gotcha #1).
    """

    def __init__(self, workflow: _Workflow, *, name: str) -> None:
        self._workflow = workflow
        self.name = name

    async def run(self, prompt: str) -> AgentRunResponse:
        # Toy "orchestration": the team answers cleanly only when the writer has
        # been instructed to be terse AND the manager is allowed enough rounds to
        # consult the reviewer. This makes prompt + routing knobs both matter.
        wf = self._workflow
        writer = wf.participants["writer"]
        terse = "ONLY" in writer.instructions or "one line" in writer.instructions.lower()
        enough_rounds = wf.max_round_count >= 6
        country = prompt.replace("Capital of", "").strip(" ?")
        capitals = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Egypt": "Cairo"}
        if terse and enough_rounds:
            return AgentRunResponse(text=capitals.get(country, "unknown"))
        return AgentRunResponse(
            text=f"After much deliberation, the capital of {country} is a city."
        )


class MagenticBuilder:
    """Offline stand-in for ``agent_framework.MagenticBuilder`` (fluent API)."""

    def __init__(self) -> None:
        self._participants: dict[str, ChatAgent] = {}
        self._manager: ChatAgent | None = None
        self._max_round_count = 8

    def participants(self, **participants: ChatAgent) -> MagenticBuilder:
        self._participants.update(participants)
        return self

    def with_standard_manager(
        self,
        manager: ChatAgent | None = None,
        *,
        agent: ChatAgent | None = None,
        max_round_count: int | None = None,
        max_stall_count: int = 3,
        max_reset_count: int | None = None,
    ) -> MagenticBuilder:
        self._manager = manager or agent
        if max_round_count is not None:
            self._max_round_count = max_round_count
        return self

    def build(self) -> _Workflow:
        assert self._manager is not None, "with_standard_manager(...) is required"
        return _Workflow(
            participants=dict(self._participants),
            manager=self._manager,
            max_round_count=self._max_round_count,
        )


# =========================================================================== #
# The team: five ChatAgents + a builder. Mutable container so ROUTING setters
# can rebuild the workflow in place and re-point the runner.
# =========================================================================== #


def build_agents() -> dict[str, ChatAgent]:
    client = FakeChatClient(model_id="gpt-4o-mini")
    return {
        "researcher": client.create_agent(
            name="researcher",
            instructions="Gather facts.",
            tools=["web_search", "wiki"],
        ),
        "writer": client.create_agent(name="writer", instructions="Write the answer."),
        "coder": client.create_agent(
            name="coder", instructions="Write code if needed.", tools=["python", "shell"]
        ),
        "reviewer": client.create_agent(
            name="reviewer", instructions="Check the answer.", skills=["fact_check"]
        ),
        "manager": client.create_agent(name="manager", instructions="Coordinate the team."),
    }


class Team:
    """Holds the live agents + the current routing limits and (re)builds the
    workflow + the ``as_agent`` wrapper. A ROUTING setter calls ``rebuild()``."""

    def __init__(self) -> None:
        self.agents = build_agents()
        self.max_round_count = 8
        self.max_stall_count = 3
        self.max_reset_count = 2
        self.workflow: _Workflow | None = None
        self.team_agent: WorkflowAgent | None = None
        self.rebuild()

    def rebuild(self) -> None:
        specialists = {k: v for k, v in self.agents.items() if k != "manager"}
        self.workflow = (
            MagenticBuilder()
            .participants(**specialists)
            .with_standard_manager(
                agent=self.agents["manager"],
                max_round_count=self.max_round_count,
                max_stall_count=self.max_stall_count,
                max_reset_count=self.max_reset_count,
            )
            .build()
        )
        self.team_agent = self.workflow.as_agent(name="research-team")

    def run(self, question: str) -> str:
        """The runner: drive the wrapped workflow agent synchronously."""
        assert self.team_agent is not None
        return asyncio.run(self.team_agent.run(question)).text


# =========================================================================== #
# Offline judge stub (matches the real judge prompt structure; see example 03).
# =========================================================================== #


def _fenced(prompt: str, label: str) -> str:
    start = prompt.find(f"<{label}>")
    end = prompt.find(f"</{label}>")
    if start == -1 or end == -1:
        return ""
    return prompt[start + len(label) + 2 : end].strip()


def judge_stub(prompt: str) -> str:
    if "CURRENT INSTRUCTION" in prompt:
        # Rewrite a writer/agent instruction so the team answers tersely.
        return "Write ONLY the answer in one line, nothing else."
    response = _fenced(prompt, "response")
    score = 9 if response and " " not in response and response != "unknown" else 2
    return f'{{"score": {score}, "pass": {str(score >= 6).lower()}, "reasoning": "auto"}}'


# =========================================================================== #
# Part A: guard the whole team as ONE unit.
# =========================================================================== #


def guard_the_team(team: Team) -> None:
    print("=== Part A: guard the Magentic team as one governed unit ===")
    firewall = Firewall(max_content_length=10_000)
    firewall.add_blocked_pattern(r"(?i)ignore (all|previous) instructions")
    adapter = MicrosoftAgentFrameworkAdapter(
        firewall=firewall,
        observer=AgentObserver(),
        agent_id="research-team",
        block_on_violation=True,
    )
    # One adapter wraps the single WorkflowAgent -> the entire team is governed.
    guarded = adapter.wrap_agent(team.team_agent)
    out = guarded.execute({"messages": [{"role": "user", "content": "Capital of France?"}]})
    print("  team answer:", out)


# =========================================================================== #
# Part B: train the whole team (components = all 5 agents) + routing knobs.
# =========================================================================== #


def build_target(team: Team) -> OptimizableAgent:
    """Wrap the team as an OptimizableAgent.

    * ``components`` = all five live ``ChatAgent``s -> each is introspected for
      its prompt/model/hyperparams/tools/skills (gotcha #1: we register the
      agents, not the opaque ``as_agent`` wrapper).
    * ``runner`` drives the wrapped *workflow agent* (``team.run``).
    * Routing limits are declared explicitly (gotcha #2): their setters rebuild
      the workflow so the change actually takes effect on the next run.
    """
    target = OptimizableAgent.from_components(
        components=dict(team.agents),  # researcher, writer, coder, reviewer, manager
        runner=team.run,
        name="research-team",
    )

    def _make_routing(attr: str, bounds: tuple[int, int]) -> Parameter:
        def setter(value: int, _attr: str = attr) -> None:
            setattr(team, _attr, int(value))
            team.rebuild()  # the ONLY way to apply a Magentic routing change

        return Parameter(
            name=f"manager.{attr}",
            kind=ParameterKind.ROUTING,
            value=getattr(team, attr),
            bounds=bounds,
            step=1,
            getter=lambda _attr=attr: getattr(team, _attr),
            setter=setter,
            component="manager",
        )

    target.add_parameter(_make_routing("max_round_count", (4, 10)))
    target.add_parameter(_make_routing("max_stall_count", (1, 5)))

    # Tool ablation on a specialist. The researcher's existing ``tools`` list is
    # already introspected automatically (as ``researcher.tools`` with drop-one
    # candidates). Here we ALSO declare an explicit knob over a *wider* candidate
    # pool that includes a tool the agent does not yet have ("arxiv"), so the
    # optimizer can search adding/removing tools, not just dropping current ones.
    # We give it a distinct name to avoid colliding with the introspected one.
    researcher = team.agents["researcher"]
    target.add_tool_parameter(
        "researcher.tool_pool",
        kind=ParameterKind.TOOL,
        getter=lambda: researcher.tools,
        setter=lambda v: setattr(researcher, "tools", list(v)),
        candidate_tools=["web_search", "wiki", "arxiv"],  # full pool to ablate from
    )
    return target


def train_whole(team: Team) -> None:
    print("\n=== Part B: train the WHOLE team (prompts + routing + tools) ===")
    data = GoldenDataset.from_list(
        [
            {"input": "Capital of France?", "expected": "Paris"},
            {"input": "Capital of Japan?", "expected": "Tokyo"},
            {"input": "Capital of Italy?", "expected": "Rome"},
            {"input": "Capital of Egypt?", "expected": "Cairo"},
        ]
    )
    # adversarial=True -> the judge grades like a harsh critic AND proposes new
    # tools/skills from failures (surfaced on result.recommendations).
    judge = LLMJudge(judge_stub, adversarial=True)
    harness = EvaluationHarness(
        metrics=[exact_match(), judge.as_metric("quality")],
        primary_metric="exact_match",
    )

    target = build_target(team)
    print("Tunable parameters discovered + declared:")
    for p in target.parameters:
        print(f"  - {p.name:<28} kind={p.kind.value}")

    print("\nBaseline:", harness.evaluate(target, data))
    print("Baseline answer:", team.run("Capital of France?"))

    # make_default_optimizer = few-shot -> prompts -> models/hparams/routing ->
    # tools/skills, with judge-driven new-tool suggestions enabled (adversarial).
    optimizer = make_default_optimizer(harness, judge=judge, max_evals=40, seed=0)
    result = optimizer.optimize(target, data)

    print(
        f"\nbaseline={result.baseline_score:.3f}  best={result.best_score:.3f}  "
        f"improvement={result.improvement:+.3f}"
    )
    print("Best config:", result.best_config)
    print("Team answer now:", team.run("Capital of France?"))
    print("max_round_count now:", team.max_round_count)
    if result.recommendations:
        print("\nJudge recommendations (advisory new tools/skills):")
        for tip in result.recommendations:
            print("  -", tip)


# =========================================================================== #
# Part C: train each agent individually (per-agent introspection).
# =========================================================================== #


def train_each_agent_individually(team: Team) -> None:
    print("\n=== Part C: introspect each ChatAgent individually ===")
    for name, agent in team.agents.items():
        params = introspect(agent)
        knobs = ", ".join(f"{p.name}({p.kind.value})" for p in params)
        print(f"  {name:<11} -> {knobs}")
    print(
        "  (Each agent can be wrapped in its own OptimizableAgent.from_agent(...) "
        "and tuned in isolation, e.g. to fix one specialist without touching the team.)"
    )


# =========================================================================== #
# Part D: the same run from a declarative YAML config (run_training).
# =========================================================================== #


def _register_yaml_app() -> None:
    """Expose the team to ``run_training`` as an importable module.

    ``run_training`` resolves ``"module:attribute"`` references via
    ``importlib.import_module``. This example file cannot be imported by path (its
    name starts with a digit), so we synthesize a small module ``magentic_team_app``
    in ``sys.modules`` that the YAML's entrypoint/components point at. In a real
    project this is just your normal package, e.g. ``myapp.app:run``.

    The synthetic module exposes the five agents AND a ``manager`` that carries the
    routing limits as attributes (so the YAML can bind ROUTING parameters to
    ``manager.max_round_count``); ``run`` rebuilds the workflow from those limits
    on every call, which is how a Magentic routing change actually takes effect.
    """
    import sys
    import types

    agents = build_agents()
    manager = agents["manager"]
    manager.max_round_count = 8  # routing knobs live on the manager object here
    manager.max_stall_count = 3
    manager.max_reset_count = 2

    def run(question: str) -> str:
        specialists = {k: v for k, v in agents.items() if k != "manager"}
        workflow = (
            MagenticBuilder()
            .participants(**specialists)
            .with_standard_manager(
                agent=manager,
                max_round_count=manager.max_round_count,
                max_stall_count=manager.max_stall_count,
                max_reset_count=manager.max_reset_count,
            )
            .build()
        )
        return asyncio.run(workflow.as_agent(name="research-team").run(question)).text

    module = types.ModuleType("magentic_team_app")
    module.run = run
    for name, agent in agents.items():
        setattr(module, name, agent)
    sys.modules["magentic_team_app"] = module


def train_from_yaml() -> None:
    print("\n=== Part D: the YAML-config training path (run_training) ===")
    # Register a deterministic offline provider so the YAML judge needs no network.
    # A provider is a ModelProvider subclass; get_provider(name, **kw) instantiates
    # it, so our subclass hardcodes the offline stub and accepts (and ignores) the
    # usual provider kwargs like ``model``.
    from adapt_agent.optimization.providers import CallableProvider, register_provider

    class OfflineStubProvider(CallableProvider):
        def __init__(self, *, model: str = "offline_stub", **kw):
            super().__init__(judge_stub, model=model)

    register_provider("offline_stub", OfflineStubProvider)
    _register_yaml_app()

    # The YAML uses dataset.path "golden.jsonl" (resolved relative to the cwd) and
    # the offline "offline_stub" provider. To avoid writing into the source tree,
    # we run from a temp dir into which we copy the real magentic.train.yaml and
    # write the golden data. In a real project you would just run the YAML in place.
    import os
    import shutil
    import tempfile

    yaml_path = Path(__file__).with_name("magentic.train.yaml")
    rows = [
        '{"input": "Capital of France?", "expected": "Paris"}',
        '{"input": "Capital of Japan?", "expected": "Tokyo"}',
        '{"input": "Capital of Italy?", "expected": "Rome"}',
        '{"input": "Capital of Egypt?", "expected": "Cairo"}',
    ]

    from adapt_agent.optimization.config import run_training

    print(f"  running: run_training({yaml_path.name})")
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "golden.jsonl").write_text("\n".join(rows), encoding="utf-8")
        shutil.copy(yaml_path, tmp_path / "magentic.train.yaml")
        try:
            os.chdir(tmp_path)
            result = run_training("magentic.train.yaml")
        finally:
            os.chdir(cwd)
    print(
        f"  baseline={result.baseline_score:.3f}  best={result.best_score:.3f}  "
        f"improvement={result.improvement:+.3f}"
    )
    print("  best config:", result.best_config)


def main() -> None:
    team = Team()
    guard_the_team(team)
    train_whole(team)
    train_each_agent_individually(team)
    train_from_yaml()


if __name__ == "__main__":
    main()
