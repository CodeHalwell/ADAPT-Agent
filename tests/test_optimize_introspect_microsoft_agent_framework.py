"""Tests for the Microsoft Agent Framework optimization introspector.

These tests use plain fake classes mimicking a Microsoft ``ChatAgent`` attribute
shape (an ``instructions`` string, a ``chat_client`` object, a callable ``run``);
the real ``agent_framework`` package is never imported.
"""

from adapt_agent.optimization.introspection import detect, introspect
from adapt_agent.optimization.parameters import ParameterKind


class FakeChatClient:
    def __init__(self, model_id: str, temperature: float) -> None:
        self.model_id = model_id
        self.temperature = temperature


class FakeChatAgent:
    def __init__(self, name, instructions, chat_client, tools=None) -> None:
        self.name = name
        self.instructions = instructions
        self.chat_client = chat_client
        self.tools = tools

    async def run(self, prompt):  # pragma: no cover - never called
        return None


def _make_agent() -> FakeChatAgent:
    return FakeChatAgent(
        name="Support Bot",
        instructions="You are a helpful support agent.",
        chat_client=FakeChatClient("gpt-4o", 0.5),
        tools=["search"],
    )


def test_detect_routes_to_microsoft_agent_framework() -> None:
    assert detect(_make_agent()) == "microsoft_agent_framework"


def test_predicate_false_for_unrelated_object() -> None:
    assert detect(object()) is None


def test_predicate_false_for_openai_style_handoffs() -> None:
    class HandoffAgent:
        def __init__(self) -> None:
            self.instructions = "You orchestrate."
            self.chat_client = FakeChatClient("gpt-4o", 0.2)
            self.handoffs = []

        async def run(self, prompt):  # pragma: no cover
            return None

    assert detect(HandoffAgent()) is None


def test_introspect_param_names_and_kinds() -> None:
    params = introspect(_make_agent())
    by_name = {p.name: p for p in params}

    assert by_name["support_bot.instructions"].kind is ParameterKind.PROMPT
    assert by_name["support_bot.model"].kind is ParameterKind.MODEL
    assert by_name["support_bot.model"].read() == "gpt-4o"
    assert by_name["support_bot.temperature"].kind is ParameterKind.HYPERPARAM
    assert by_name["support_bot.temperature"].bounds == (0.0, 2.0)
    assert by_name["support_bot.tools"].kind is ParameterKind.TOOL


def test_instructions_setter_round_trip() -> None:
    agent = _make_agent()
    params = introspect(agent)
    by_name = {p.name: p for p in params}

    param = by_name["support_bot.instructions"]
    param.write("You are a concise support agent.")
    assert agent.instructions == "You are a concise support agent."
    assert param.read() == "You are a concise support agent."


def test_fallback_component_name_without_name() -> None:
    agent = FakeChatAgent(
        name=None,
        instructions="Be helpful.",
        chat_client=FakeChatClient("gpt-4o-mini", 0.0),
    )
    params = introspect(agent)
    assert any(p.name == "agent.instructions" for p in params)


def test_top_p_and_max_tokens_on_agent() -> None:
    agent = _make_agent()
    agent.top_p = 0.9
    agent.max_tokens = 1024
    params = introspect(agent)
    by_name = {p.name: p for p in params}

    assert by_name["support_bot.top_p"].kind is ParameterKind.HYPERPARAM
    assert by_name["support_bot.top_p"].bounds == (0.0, 1.0)
    assert by_name["support_bot.max_tokens"].kind is ParameterKind.HYPERPARAM
