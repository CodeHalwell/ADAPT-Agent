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

    def __init__(self, name, instructions, model="gpt-4o", handoffs=None, tools=None):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.model_settings = FakeModelSettings()
        self.tools = tools if tools is not None else []
        self.handoffs = handoffs if handoffs is not None else []


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


def test_recursion_guards_against_cycles():
    a = FakeAgent("A Agent", "a")
    b = FakeAgent("B Agent", "b")
    a.handoffs = [b]
    b.handoffs = [a]  # cycle
    params = introspect(a)  # must terminate
    names = {p.name for p in params}
    assert "a_agent.instructions" in names
    assert "b_agent.instructions" in names
