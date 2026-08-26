"""Tests for the Microsoft Agent Framework optimization introspector.

Most tests here use plain fake classes mimicking a Microsoft agent's attribute
shape. ``test_real_agent_framework_agent_is_introspectable`` is the exception:
it imports the real ``agent_framework`` when installed, because a fake can only
encode what the SDK looked like when the fake was written -- which is exactly
how the ``.client`` / ``default_options`` move slipped past this suite.
"""

import importlib.util

import pytest

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
    assert by_name["support_bot.max_tokens"].bounds == (1, 32000)


def test_tools_are_optimizable_with_drop_one_candidates() -> None:
    agent = FakeChatAgent(
        name="Magentic Worker",
        instructions="Do work.",
        chat_client=FakeChatClient("gpt-4o", 0.3),
        tools=["search", "calc", "lookup"],
    )
    params = introspect(agent)
    by_name = {p.name: p for p in params}

    tools = by_name["magentic_worker.tools"]
    assert tools.kind is ParameterKind.TOOL
    assert tools.candidates is not None
    # Full set first, then each drop-one ablation subset.
    assert tools.candidates[0] == ["search", "calc", "lookup"]
    assert ["calc", "lookup"] in tools.candidates
    # More than one candidate makes it a real search space (optimizable).
    assert len(tools.candidates) > 1


def test_single_tool_has_no_ablation_candidates() -> None:
    # The shared _make_agent has a single tool: bound for visibility but no
    # drop-one candidates (nothing to ablate to).
    params = introspect(_make_agent())
    by_name = {p.name: p for p in params}
    assert by_name["support_bot.tools"].kind is ParameterKind.TOOL
    assert by_name["support_bot.tools"].candidates is None


def test_skills_are_introspected_as_skill_kind() -> None:
    agent = FakeChatAgent(
        name="Skilled Bot",
        instructions="Help out.",
        chat_client=FakeChatClient("gpt-4o", 0.1),
    )
    agent.skills = ["summarize", "translate"]
    params = introspect(agent)
    by_name = {p.name: p for p in params}

    skills = by_name["skilled_bot.skills"]
    assert skills.kind is ParameterKind.SKILL
    assert skills.candidates is not None
    assert skills.candidates[0] == ["summarize", "translate"]


# -- current SDK layout -------------------------------------------------------
#
# The fakes above mirror the *older* attribute shape. Current releases moved the
# client to `.client` and put the prompt, tools and sampling settings inside a
# `default_options` mapping -- and because the fakes only knew the old shape,
# the suite stayed green while real agents yielded nothing tunable at all. These
# tests pin the current layout; `test_real_agent_framework_agent_is_introspectable`
# below pins it against the actual SDK, which is the only thing that catches the
# next rename.


class FakeClient:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id


class FakeOptionsAgent:
    """Mirrors `agent_framework.Agent`: `.client`, no `.instructions`, options dict."""

    def __init__(self, name, default_options, client=None) -> None:
        self.name = name
        self.client = client if client is not None else FakeClient("gpt-4o-mini")
        self.default_options = default_options

    async def run(self, prompt):  # pragma: no cover - never called
        return None


def _options_agent(**options) -> FakeOptionsAgent:
    base = {"instructions": "You are a triage bot.", "temperature": 0.3}
    base.update(options)
    return FakeOptionsAgent(name="Triage Bot", default_options=base)


def test_detect_routes_agent_using_client_and_default_options() -> None:
    assert detect(_options_agent()) == "microsoft_agent_framework"


def test_prompt_is_discovered_from_default_options() -> None:
    params = introspect(_options_agent())
    prompts = [p for p in params if p.kind is ParameterKind.PROMPT]
    assert [p.name for p in prompts] == ["triage_bot.instructions"]
    assert prompts[0].value == "You are a triage bot."


def test_prompt_setter_writes_back_into_default_options() -> None:
    agent = _options_agent()
    prompt = next(p for p in introspect(agent) if p.kind is ParameterKind.PROMPT)
    prompt.write("Be terse.")
    assert agent.default_options["instructions"] == "Be terse."


def test_sampling_settings_are_discovered_from_default_options() -> None:
    agent = _options_agent(top_p=0.9, max_tokens=512)
    params = {p.name: p for p in introspect(agent)}
    assert params["triage_bot.temperature"].value == 0.3
    assert params["triage_bot.top_p"].value == 0.9
    assert params["triage_bot.max_tokens"].value == 512
    params["triage_bot.temperature"].write(0.9)
    assert agent.default_options["temperature"] == 0.9


def test_tools_are_discovered_from_default_options_with_ablation() -> None:
    agent = _options_agent(tools=["search", "calculator", "lookup"])
    tools = next(p for p in introspect(agent) if p.kind is ParameterKind.TOOL)
    assert tools.value == ["search", "calculator", "lookup"]
    # full set first, then each drop-one subset
    assert tools.candidates[0] == ["search", "calculator", "lookup"]
    assert ["calculator", "lookup"] in tools.candidates
    tools.write(["search"])
    assert agent.default_options["tools"] == ["search"]


def test_model_is_discovered_from_the_client_attribute() -> None:
    params = {p.name: p for p in introspect(_options_agent())}
    assert params["triage_bot.model"].value == "gpt-4o-mini"
    assert params["triage_bot.model"].kind is ParameterKind.MODEL


def test_attribute_layout_wins_over_default_options() -> None:
    """An agent carrying both shapes must not yield two prompt parameters."""
    agent = _options_agent()
    agent.instructions = "Attribute wins."
    prompts = [p for p in introspect(agent) if p.kind is ParameterKind.PROMPT]
    assert len(prompts) == 1
    assert prompts[0].value == "Attribute wins."


def test_penalties_discovered_from_default_options() -> None:
    agent = _options_agent(frequency_penalty=0.2, presence_penalty=-0.1)
    params = {p.name: p for p in introspect(agent)}

    frequency = params["triage_bot.frequency_penalty"]
    assert frequency.kind is ParameterKind.HYPERPARAM
    assert frequency.bounds == (-2.0, 2.0)
    assert frequency.value == 0.2
    presence = params["triage_bot.presence_penalty"]
    assert presence.bounds == (-2.0, 2.0)
    presence.write(0.5)
    assert agent.default_options["presence_penalty"] == 0.5


def test_penalties_discovered_from_client_attributes() -> None:
    client = FakeChatClient("gpt-4o", 0.5)
    client.frequency_penalty = 0.1
    agent = FakeChatAgent(name="Penalty Bot", instructions="Help.", chat_client=client)
    params = {p.name: p for p in introspect(agent)}

    frequency = params["penalty_bot.frequency_penalty"]
    assert frequency.kind is ParameterKind.HYPERPARAM
    assert frequency.bounds == (-2.0, 2.0)


def test_model_id_from_default_options_when_client_exposes_none() -> None:
    """A per-agent model override in the options mapping is the model knob.

    Some clients expose no model attribute at all; without the mapping fallback
    the whole agent had no MODEL parameter even though ``model_id`` sat right in
    ``default_options``.
    """
    agent = FakeOptionsAgent(
        name="Override Bot",
        default_options={"instructions": "Hi.", "model_id": "gpt-4o"},
        client=object(),  # exposes none of the model attributes
    )
    params = {p.name: p for p in introspect(agent)}

    model = params["override_bot.model"]
    assert model.kind is ParameterKind.MODEL
    assert model.value == "gpt-4o"
    model.write("gpt-4o-mini")
    assert agent.default_options["model_id"] == "gpt-4o-mini"


def test_client_model_wins_over_default_options_model_id() -> None:
    agent = _options_agent(model_id="options-model")  # client also has model_id
    params = [p for p in introspect(agent) if p.kind is ParameterKind.MODEL]
    assert len(params) == 1
    assert params[0].value == "gpt-4o-mini"  # the client's, not the mapping's


def test_pydantic_ai_style_agent_is_not_claimed() -> None:
    """The client check is load-bearing: Pydantic AI also has instructions+run.

    Without it this introspector would swallow a Pydantic AI ``Agent`` and hand
    back Microsoft-shaped parameter names for it.
    """

    class PydanticAIStyleAgent:
        def __init__(self) -> None:
            self.instructions = "You are helpful."
            self.model = "openai:gpt-4o"

        async def run(self, prompt):  # pragma: no cover - never called
            return None

    from adapt_agent.optimization.introspection.microsoft_agent_framework import _predicate

    assert _predicate(PydanticAIStyleAgent()) is False


@pytest.mark.skipif(
    importlib.util.find_spec("agent_framework") is None,
    reason="agent-framework is not installed",
)
def test_real_agent_framework_agent_is_introspectable() -> None:
    """Introspect an actual SDK agent, not a fake of one.

    Every fake in this module is a guess about the SDK's shape, and a guess that
    goes stale fails silently: `detect` returns None, `introspect` returns [],
    and `OptimizableAgent.from_agent` reports no tunable knobs -- which reads as
    "nothing to optimize here" rather than as a broken introspector. This is the
    only test that would have caught that.
    """
    from agent_framework import BaseChatClient  # type: ignore[import-not-found]
    from agent_framework._agents import Agent  # type: ignore[import-not-found]

    class _Client(BaseChatClient):  # type: ignore[misc]
        model_id = "gpt-4o-mini"

        async def _inner_get_response(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

        def _inner_get_streaming_response(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    agent = Agent(
        client=_Client(),
        instructions="You are a triage bot.",
        name="Triage Bot",
        default_options={"temperature": 0.3},
    )

    assert detect(agent) == "microsoft_agent_framework"

    params = {p.name: p for p in introspect(agent)}
    prompt = params["triage_bot.instructions"]
    assert prompt.kind is ParameterKind.PROMPT
    assert prompt.value == "You are a triage bot."
    assert params["triage_bot.model"].value == "gpt-4o-mini"
    assert params["triage_bot.temperature"].value == 0.3

    # The knobs must be writable, not merely discoverable.
    prompt.write("Be terse.")
    params["triage_bot.temperature"].write(0.9)
    assert agent.default_options["instructions"] == "Be terse."
    assert agent.default_options["temperature"] == 0.9
