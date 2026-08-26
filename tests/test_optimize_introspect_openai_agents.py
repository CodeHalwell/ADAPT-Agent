"""Tests for OpenAI Agents SDK introspection.

These tests use FAKE objects mimicking the SDK's attribute shape (the real
``agents`` package is not installed). They verify registry routing, parameter
extraction (including recursion into handoff sub-agents), prompt setter
round-trip, and that the predicate rejects unrelated objects.
"""

# Importing the module triggers self-registration with the introspection registry.
import adapt_agent.optimization.introspection.openai_agents as module
from adapt_agent.optimization.introspection import detect, introspect
from adapt_agent.optimization.parameters import ParameterKind


class FakeModelSettings:
    """Mimics ``agents.ModelSettings`` (temperature / top_p / max_tokens)."""

    def __init__(self, temperature=0.7, top_p=0.9, max_tokens=512):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens


class FakeAgent:
    """Mimics ``agents.Agent`` (instructions / model / settings / tools / handoffs)."""

    def __init__(
        self,
        name,
        instructions,
        model="gpt-4o",
        handoffs=None,
        tools=None,
        handoff_description=None,
    ):
        self.name = name
        self.instructions = instructions
        self.handoff_description = handoff_description
        self.model = model
        self.model_settings = FakeModelSettings()
        self.tools = tools if tools is not None else []
        self.handoffs = handoffs if handoffs is not None else []


class FakeHandoff:
    """Mimics ``agents.Handoff``: a wrapper with no reachable agent object."""

    def __init__(self, tool_name, tool_description, agent_name):
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.agent_name = agent_name
        self.on_invoke_handoff = lambda *a, **k: None


def _make_orchestrator():
    sub = FakeAgent("Research Agent", "You research things.", model="gpt-4o-mini")
    root = FakeAgent(
        "Triage Agent",
        "You triage requests.",
        model="gpt-4o",
        handoffs=[sub],
        tools=["search_tool"],
    )
    return root, sub


def test_detect_routes_to_openai_agents():
    root, _ = _make_orchestrator()
    assert detect(root) == "openai_agents"


def test_introspect_param_names_and_kinds():
    root, _ = _make_orchestrator()
    params = introspect(root)
    by_name = {p.name: p for p in params}

    # Root agent ("triage_agent") params.
    assert by_name["triage_agent.instructions"].kind is ParameterKind.PROMPT
    assert by_name["triage_agent.model"].kind is ParameterKind.MODEL
    assert by_name["triage_agent.temperature"].kind is ParameterKind.HYPERPARAM
    assert by_name["triage_agent.temperature"].bounds == (0.0, 2.0)
    assert by_name["triage_agent.top_p"].kind is ParameterKind.HYPERPARAM
    assert by_name["triage_agent.top_p"].bounds == (0.0, 1.0)
    assert by_name["triage_agent.max_tokens"].kind is ParameterKind.HYPERPARAM
    assert by_name["triage_agent.tools"].kind is ParameterKind.TOOL
    assert by_name["triage_agent.handoffs"].kind is ParameterKind.ROUTING


def test_recursion_into_handoff_subagent():
    root, _ = _make_orchestrator()
    params = introspect(root)
    by_name = {p.name: p for p in params}

    # The handed-off sub-agent ("research_agent") must contribute its own params.
    assert by_name["research_agent.instructions"].kind is ParameterKind.PROMPT
    assert by_name["research_agent.model"].kind is ParameterKind.MODEL
    assert by_name["research_agent.model"].read() == "gpt-4o-mini"
    assert by_name["research_agent.temperature"].kind is ParameterKind.HYPERPARAM


def test_prompt_setter_round_trips():
    root, _ = _make_orchestrator()
    params = introspect(root)
    by_name = {p.name: p for p in params}

    prompt = by_name["triage_agent.instructions"]
    prompt.write("Updated instructions.")
    assert root.instructions == "Updated instructions."
    assert prompt.read() == "Updated instructions."


def test_callable_instructions_emit_no_prompt():
    root = FakeAgent("Dynamic Agent", lambda ctx, agent: "computed", model="gpt-4o")
    params = introspect(root)
    names = {p.name for p in params}
    assert "dynamic_agent.instructions" not in names
    # Other params still appear.
    assert "dynamic_agent.model" in names


def test_model_object_introspected():
    class FakeModel:
        def __init__(self):
            self.model = "claude-via-litellm"

    root = FakeAgent("Obj Agent", "Hi.", model=FakeModel())
    params = introspect(root)
    by_name = {p.name: p for p in params}
    assert by_name["obj_agent.model"].kind is ParameterKind.MODEL
    assert by_name["obj_agent.model"].read() == "claude-via-litellm"


def test_unnamed_agent_falls_back_to_agent_component():
    root = FakeAgent(None, "No name here.")
    params = introspect(root)
    names = {p.name for p in params}
    assert "agent.instructions" in names


def test_predicate_rejects_unrelated_object():
    assert module._predicate(object()) is False
    assert detect(object()) != "openai_agents"


def test_predicate_rejects_foreign_chat_client():
    # A Microsoft-ChatAgent-shaped object carrying chat_client must NOT be claimed
    # by the OpenAI introspector, even if it also happens to expose handoffs/tools.
    class FakeChatAgent:
        def __init__(self):
            self.instructions = "You orchestrate."
            self.tools = []
            self.handoffs = []
            self.chat_client = object()

    assert module._predicate(FakeChatAgent()) is False


def test_predicate_rejects_kickoff_and_sub_agents():
    class CrewLike:
        instructions = "x"
        tools: list = []
        handoffs: list = []

        def kickoff(self):  # CrewAI marker
            return None

    class AdkLike:
        instructions = "x"
        tools: list = []
        handoffs: list = []
        sub_agents: list = []  # Google ADK marker

    assert module._predicate(CrewLike()) is False
    assert module._predicate(AdkLike()) is False


def test_predicate_requires_handoffs_to_be_a_sequence():
    class NotASequence:
        instructions = "x"
        tools: list = []
        handoffs = None  # present but not a list/tuple

    assert module._predicate(NotASequence()) is False


def test_tools_carry_drop_one_candidates():
    root = FakeAgent("Tooled Agent", "Hi.", tools=["a", "b", "c"])
    params = introspect(root)
    by_name = {p.name: p for p in params}
    tools = by_name["tooled_agent.tools"]
    assert tools.candidates is not None
    assert tools.candidates[0] == ["a", "b", "c"]
    assert ["b", "c"] in tools.candidates


def test_recursion_guards_against_cycles():
    a = FakeAgent("A Agent", "a")
    b = FakeAgent("B Agent", "b")
    a.handoffs = [b]
    b.handoffs = [a]  # cycle
    params = introspect(a)  # must terminate
    names = {p.name for p in params}
    assert "a_agent.instructions" in names
    assert "b_agent.instructions" in names


def test_handoff_description_bound_as_prompt():
    sub = FakeAgent(
        "Research Agent",
        "You research things.",
        handoff_description="Specialist agent for research questions",
    )
    root = FakeAgent("Triage Agent", "You triage.", handoffs=[sub])
    params = introspect(root)
    by_name = {p.name: p for p in params}

    description = by_name["research_agent.handoff_description"]
    assert description.kind is ParameterKind.PROMPT
    description.write("Expert for deep research and citations")
    assert sub.handoff_description == "Expert for deep research and citations"
    assert description.read() == "Expert for deep research and citations"


def test_unset_handoff_description_emits_no_prompt():
    root = FakeAgent("Plain Agent", "Hi.")  # handoff_description stays None
    names = {p.name for p in introspect(root)}
    assert "plain_agent.handoff_description" not in names


def test_model_settings_max_tokens_is_a_real_search_space():
    # Boundless, the knob enumerated to just its current value -- a parameter
    # the optimizer could see but never move.
    root = FakeAgent("Bounded Agent", "Hi.")
    by_name = {p.name: p for p in introspect(root)}
    max_tokens = by_name["bounded_agent.max_tokens"]
    assert max_tokens.bounds == (1, 32000)
    assert len(max_tokens.enumerate_candidates()) > 1


def test_handoff_wrapper_tool_description_bound_as_prompt():
    wrapper = FakeHandoff(
        tool_name="transfer_to_research_agent",
        tool_description="Handles research questions",
        agent_name="Research Agent",
    )
    root = FakeAgent("Triage Agent", "You triage.", handoffs=[wrapper])
    params = introspect(root)
    by_name = {p.name: p for p in params}

    description = by_name["research_agent_handoff.tool_description"]
    assert description.kind is ParameterKind.PROMPT
    description.write("Handles research and literature review requests")
    assert wrapper.tool_description == "Handles research and literature review requests"

    # The wrapper contributes only its routing text -- there is no reachable
    # agent behind it to introspect.
    wrapper_params = [p for p in params if p.component == "research_agent_handoff"]
    assert len(wrapper_params) == 1


def test_handoff_wrapper_shared_by_two_agents_bound_once():
    wrapper = FakeHandoff("transfer", "Shared specialist", "Shared Agent")
    left = FakeAgent("Left Agent", "l", handoffs=[wrapper])
    root = FakeAgent("Root Agent", "r", handoffs=[left, wrapper])
    params = introspect(root)
    names = [p.name for p in params]
    assert names.count("shared_agent_handoff.tool_description") == 1
