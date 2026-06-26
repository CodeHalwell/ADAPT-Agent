"""Tests for the CrewAI optimization introspector.

These tests use plain fake classes mimicking CrewAI's ``Crew``/``Agent``/``Task``
attribute shape; the real ``crewai`` package is never imported.
"""

from adapt_agent.optimization.introspection import detect, introspect
from adapt_agent.optimization.parameters import ParameterKind


class FakeLLM:
    def __init__(self, model: str, temperature: float, max_tokens: int) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens


class FakeAgent:
    def __init__(self, role, goal, backstory, llm, tools, max_iter) -> None:
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm = llm
        self.tools = tools
        self.max_iter = max_iter


class FakeTask:
    def __init__(self, description, expected_output) -> None:
        self.description = description
        self.expected_output = expected_output


class FakeCrew:
    def __init__(self, agents, tasks) -> None:
        self.agents = agents
        self.tasks = tasks

    def kickoff(self, inputs=None):  # pragma: no cover - never called
        return None


def _make_crew() -> FakeCrew:
    researcher = FakeAgent(
        role="Senior Researcher",
        goal="Find facts",
        backstory="A seasoned analyst",
        llm=FakeLLM("gpt-4o", 0.3, 512),
        tools=["search"],
        max_iter=10,
    )
    writer = FakeAgent(
        role="Writer",
        goal="Write a report",
        backstory="A skilled writer",
        llm="gpt-4o-mini",
        tools=[],
        max_iter=5,
    )
    task = FakeTask(description="Research the topic", expected_output="A summary")
    return FakeCrew(agents=[researcher, writer], tasks=[task])


def test_detect_routes_to_crewai() -> None:
    assert detect(_make_crew()) == "crewai"


def test_predicate_false_for_unrelated_object() -> None:
    assert detect(object()) is None


def test_predicate_false_for_other_framework() -> None:
    class HandoffAgent:
        def __init__(self) -> None:
            self.agents = []
            self.handoffs = []

        def kickoff(self):  # pragma: no cover
            return None

    assert detect(HandoffAgent()) is None


def test_predicate_false_for_microsoft_chat_agent_like_object() -> None:
    # A Microsoft Agent Framework ``ChatAgent`` (``chat_client`` + callable
    # ``run``) might expose an ``agents`` list; the crewai introspector (registered
    # earlier) must never hijack it.
    class FakeChatAgent:
        def __init__(self) -> None:
            self.agents = []
            self.instructions = "be helpful"
            self.chat_client = object()

        def kickoff(self):  # pragma: no cover - never called
            return None

        def run(self, *a, **k):  # pragma: no cover - never called
            return None

    assert detect(FakeChatAgent()) != "crewai"


def test_introspect_param_names_and_kinds() -> None:
    params = introspect(_make_crew())
    by_name = {p.name: p for p in params}

    # Researcher: object LLM -> model/temperature/max_tokens introspected.
    assert by_name["senior_researcher.role"].kind is ParameterKind.PROMPT
    assert by_name["senior_researcher.goal"].kind is ParameterKind.PROMPT
    assert by_name["senior_researcher.backstory"].kind is ParameterKind.PROMPT
    assert by_name["senior_researcher.model"].kind is ParameterKind.MODEL
    assert by_name["senior_researcher.temperature"].kind is ParameterKind.HYPERPARAM
    assert by_name["senior_researcher.temperature"].bounds == (0.0, 2.0)
    assert by_name["senior_researcher.max_tokens"].kind is ParameterKind.HYPERPARAM
    assert by_name["senior_researcher.tools"].kind is ParameterKind.TOOL
    assert by_name["senior_researcher.max_iter"].kind is ParameterKind.HYPERPARAM
    assert by_name["senior_researcher.max_iter"].bounds == (1, 50)

    # Writer: string LLM -> single MODEL param, no temperature/max_tokens.
    assert by_name["writer.model"].kind is ParameterKind.MODEL
    assert by_name["writer.model"].read() == "gpt-4o-mini"
    assert "writer.temperature" not in by_name

    # Task params.
    assert by_name["task_0.description"].kind is ParameterKind.PROMPT
    assert by_name["task_0.expected_output"].kind is ParameterKind.PROMPT


def test_prompt_setter_round_trip() -> None:
    crew = _make_crew()
    params = introspect(crew)
    by_name = {p.name: p for p in params}

    param = by_name["senior_researcher.goal"]
    param.write("Find even better facts")
    assert crew.agents[0].goal == "Find even better facts"
    assert param.read() == "Find even better facts"


def test_string_llm_setter_round_trip() -> None:
    crew = _make_crew()
    params = introspect(crew)
    by_name = {p.name: p for p in params}

    by_name["writer.model"].write("gpt-4.1")
    assert crew.agents[1].llm == "gpt-4.1"


def test_agent_tools_param_is_optimizable_with_drop_one_candidates() -> None:
    agent = FakeAgent(
        role="Senior Researcher",
        goal="Find facts",
        backstory="A seasoned analyst",
        llm="gpt-4o-mini",
        tools=["search", "calculator", "wiki"],
        max_iter=10,
    )
    crew = FakeCrew(agents=[agent], tasks=[])
    by_name = {p.name: p for p in introspect(crew)}

    tools = by_name["senior_researcher.tools"]
    assert tools.kind is ParameterKind.TOOL
    assert tools.candidates[0] == ["search", "calculator", "wiki"]
    assert ["calculator", "wiki"] in tools.candidates
    assert ["search", "wiki"] in tools.candidates
    assert ["search", "calculator"] in tools.candidates
    assert len(tools.candidates) == 4


def test_agent_tools_param_not_searchable_with_single_tool() -> None:
    # The default researcher in ``_make_crew`` has a single tool.
    by_name = {p.name: p for p in introspect(_make_crew())}
    assert by_name["senior_researcher.tools"].candidates == []


def test_max_tokens_bounds_present() -> None:
    by_name = {p.name: p for p in introspect(_make_crew())}
    assert by_name["senior_researcher.max_tokens"].bounds == (1, 32000)


def test_fallback_component_name_without_role() -> None:
    class RolelessAgent:
        def __init__(self) -> None:
            self.goal = "do things"

    crew = FakeCrew(agents=[RolelessAgent()], tasks=[])
    params = introspect(crew)
    assert any(p.name == "agent_0.goal" for p in params)
