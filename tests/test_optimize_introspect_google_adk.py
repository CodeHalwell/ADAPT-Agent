"""Tests for Google ADK introspection (offline, no ``google.adk`` import).

Builds fake ``LlmAgent``-shaped objects (plain Python classes) that mimic the
ADK attribute layout -- ``name``, ``instruction``, ``model``,
``generate_content_config``, ``tools``, ``sub_agents`` -- and verifies that the
registry routes them, that introspection produces the expected parameter
names/kinds (including recursion into a sub-agent and a temperature
hyperparameter), that a prompt setter round-trips onto the live object, and that
the predicate rejects unrelated objects.
"""

import adapt_agent.optimization.introspection.google_adk as adk
from adapt_agent.optimization.introspection import detect, introspect
from adapt_agent.optimization.parameters import ParameterKind


class FakeGenerateConfig:
    """Mimics an ADK ``generate_content_config`` object."""

    def __init__(self, temperature=0.7, top_p=0.9, max_output_tokens=512):
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens


class FakeLlmAgent:
    """Mimics an ADK ``LlmAgent``: has ``sub_agents`` and ``instruction``."""

    def __init__(
        self,
        name,
        instruction,
        model="gemini-2.0-flash",
        tools=None,
        sub_agents=None,
        generate_content_config=None,
        global_instruction=None,
    ):
        self.name = name
        self.instruction = instruction
        self.model = model
        self.tools = tools or []
        self.sub_agents = sub_agents or []
        self.generate_content_config = generate_content_config
        if global_instruction is not None:
            self.global_instruction = global_instruction


def _make_agent():
    child = FakeLlmAgent(
        name="researcher",
        instruction="Research the topic thoroughly.",
        model="gemini-2.0-flash",
        generate_content_config=FakeGenerateConfig(temperature=0.3),
    )
    parent = FakeLlmAgent(
        name="Coordinator Agent",
        instruction="You are a coordinator.",
        model="gemini-2.0-pro",
        tools=["search", "calc"],
        sub_agents=[child],
        generate_content_config=FakeGenerateConfig(temperature=0.9),
        global_instruction="Be helpful and concise.",
    )
    return parent


def test_registry_routes_adk_agent():
    agent = _make_agent()
    assert detect(agent) == "google_adk"


def test_introspect_names_and_kinds():
    agent = _make_agent()
    params = introspect(agent)
    by_name = {p.name: p for p in params}

    # Parent-level params.
    assert by_name["coordinator_agent.instruction"].kind is ParameterKind.PROMPT
    assert by_name["coordinator_agent.global_instruction"].kind is ParameterKind.PROMPT
    assert by_name["coordinator_agent.model"].kind is ParameterKind.MODEL
    assert by_name["coordinator_agent.temperature"].kind is ParameterKind.HYPERPARAM
    assert by_name["coordinator_agent.top_p"].kind is ParameterKind.HYPERPARAM
    assert by_name["coordinator_agent.max_output_tokens"].kind is ParameterKind.HYPERPARAM
    assert by_name["coordinator_agent.tools"].kind is ParameterKind.TOOL
    assert by_name["coordinator_agent.sub_agents"].kind is ParameterKind.ROUTING

    # Temperature bounds.
    assert by_name["coordinator_agent.temperature"].bounds == (0.0, 2.0)
    assert by_name["coordinator_agent.top_p"].bounds == (0.0, 1.0)


def test_recursion_into_sub_agent():
    agent = _make_agent()
    params = introspect(agent)
    names = {p.name for p in params}

    # The child's params are namespaced under the parent component.
    assert "coordinator_agent.researcher.instruction" in names
    assert "coordinator_agent.researcher.temperature" in names

    child_temp = next(p for p in params if p.name == "coordinator_agent.researcher.temperature")
    assert child_temp.kind is ParameterKind.HYPERPARAM
    assert child_temp.read() == 0.3


def test_prompt_setter_round_trips():
    agent = _make_agent()
    params = introspect(agent)
    prompt = next(p for p in params if p.name == "coordinator_agent.instruction")
    prompt.write("New coordinator instruction.")
    assert agent.instruction == "New coordinator instruction."
    assert prompt.read() == "New coordinator instruction."


def test_instruction_provider_callable_not_emitted_as_prompt():
    # When instruction is a callable provider, no PROMPT should be emitted for it.
    def provider(ctx):
        return "dynamic"

    agent = FakeLlmAgent(name="dyn", instruction=provider)
    params = introspect(agent)
    names = {p.name for p in params}
    assert "dyn.instruction" not in names
    # The agent is still detected/introspected (model etc. present).
    assert detect(agent) == "google_adk"
    assert "dyn.model" in names


def test_two_nameless_sub_agents_do_not_collide():
    # Two sub-agents without a usable ``name`` previously both collapsed to the
    # component "agent", producing duplicate parameter names. The positional
    # index in the fallback now keeps them distinct.
    child_a = FakeLlmAgent(name="", instruction="First nameless child.")
    child_b = FakeLlmAgent(name="", instruction="Second nameless child.")
    parent = FakeLlmAgent(
        name="root",
        instruction="Coordinate the nameless children.",
        sub_agents=[child_a, child_b],
    )
    params = introspect(parent)
    names = [p.name for p in params]

    # No duplicate parameter names overall.
    assert len(names) == len(set(names))
    # Each nameless child got a distinct, index-based component namespace.
    assert "root.agent_0.instruction" in names
    assert "root.agent_1.instruction" in names


def test_tools_are_optimizable_with_drop_one_candidates():
    agent = _make_agent()  # parent has tools ["search", "calc"]
    params = introspect(agent)
    tools = next(p for p in params if p.name == "coordinator_agent.tools")

    assert tools.kind is ParameterKind.TOOL
    assert tools.candidates is not None
    # Full set first, then each drop-one subset.
    assert tools.candidates[0] == ["search", "calc"]
    assert ["calc"] in tools.candidates
    assert ["search"] in tools.candidates
    # With candidates the knob is enumerable beyond its single current value.
    assert len(tools.enumerate_candidates()) >= 2


def test_single_tool_has_no_ablation_candidates():
    agent = FakeLlmAgent(name="solo", instruction="hi", tools=["only"])
    params = introspect(agent)
    tools = next(p for p in params if p.name == "solo.tools")
    # Fewer than two tools -> no meaningful subset search.
    assert tools.candidates is None


def test_max_output_tokens_is_bounded():
    agent = _make_agent()
    params = introspect(agent)
    mot = next(p for p in params if p.name == "coordinator_agent.max_output_tokens")
    assert mot.bounds == (1, 32000)


def test_predicate_rejects_unrelated_object():
    assert adk._predicate(object()) is False


def test_predicate_rejects_foreign_frameworks():
    class CrewishAgent:
        instruction = "x"
        sub_agents: list = []
        kickoff = lambda self: None  # noqa: E731

    assert adk._predicate(CrewishAgent()) is False
