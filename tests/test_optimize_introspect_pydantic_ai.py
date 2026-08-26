"""Offline tests for Pydantic AI introspection.

These build fake objects mimicking a Pydantic AI ``Agent`` (the real framework
is not installed) and assert the registry routes them, the discovered parameters
have the expected names/kinds, the system-prompt setter mutates the live object,
and the predicate rejects unrelated objects. No network, no framework import.
"""

import importlib.util

import pytest

from adapt_agent.optimization.introspection import detect, introspect
from adapt_agent.optimization.introspection.pydantic_ai import _predicate
from adapt_agent.optimization.parameters import ParameterKind


class FakeAgent:
    """Mimics the attribute shape of a Pydantic AI ``Agent``."""

    def __init__(self) -> None:
        self.model = "openai:gpt-4o"
        self._system_prompts = ("be helpful",)
        self.model_settings = {"temperature": 0.7}

    def run_sync(self, prompt: str) -> str:  # pragma: no cover - never called
        return prompt


def test_detect_routes_to_pydantic_ai() -> None:
    assert detect(FakeAgent()) == "pydantic_ai"


def test_introspect_param_names_and_kinds() -> None:
    params = {p.name: p for p in introspect(FakeAgent())}
    assert params["agent.system_prompt"].kind is ParameterKind.PROMPT
    assert params["agent.model"].kind is ParameterKind.MODEL
    assert "agent.temperature" in params
    assert params["agent.temperature"].kind is ParameterKind.HYPERPARAM
    assert params["agent.temperature"].bounds == (0.0, 2.0)


def test_system_prompt_setter_replaces_tuple() -> None:
    agent = FakeAgent()
    params = {p.name: p for p in introspect(agent)}
    prompt = params["agent.system_prompt"]
    assert prompt.read() == "be helpful"
    prompt.write("be concise")
    assert agent._system_prompts == ("be concise",)
    assert prompt.read() == "be concise"


def test_temperature_setter_roundtrips() -> None:
    agent = FakeAgent()
    params = {p.name: p for p in introspect(agent)}
    params["agent.temperature"].write(0.2)
    assert agent.model_settings["temperature"] == 0.2


def test_penalties_in_model_settings_bound() -> None:
    agent = FakeAgent()
    agent.model_settings["frequency_penalty"] = 0.3
    agent.model_settings["presence_penalty"] = -0.2
    params = {p.name: p for p in introspect(agent)}

    frequency = params["agent.frequency_penalty"]
    assert frequency.kind is ParameterKind.HYPERPARAM
    assert frequency.bounds == (-2.0, 2.0)
    presence = params["agent.presence_penalty"]
    assert presence.bounds == (-2.0, 2.0)
    presence.write(0.1)
    assert agent.model_settings["presence_penalty"] == 0.1


def test_unset_penalties_emit_no_parameters() -> None:
    params = {p.name for p in introspect(FakeAgent())}
    assert "agent.frequency_penalty" not in params
    assert "agent.presence_penalty" not in params


def test_predicate_rejects_unrelated_object() -> None:
    assert _predicate(object()) is False


def test_predicate_rejects_microsoft_chat_agent_like_object() -> None:
    # A Microsoft Agent Framework ``ChatAgent`` carries ``instructions`` +
    # ``chat_client`` + callable ``run``; the pydantic_ai introspector (registered
    # earlier) must never hijack it.
    class FakeChatAgent:
        model = "x"
        _system_prompts = ("hi",)

        def __init__(self) -> None:
            self.instructions = "be helpful"
            self.chat_client = object()

        def run_sync(self, prompt: str) -> str:  # pragma: no cover
            return prompt

        def run(self, *a, **k):  # pragma: no cover
            return None

    assert _predicate(FakeChatAgent()) is False
    assert detect(FakeChatAgent()) != "pydantic_ai"


def test_tools_param_is_optimizable_with_drop_one_candidates() -> None:
    class AgentWithTools:
        def __init__(self) -> None:
            self.model = "openai:gpt-4o"
            self._system_prompts = ("be helpful",)
            self.tools = ["search", "calculator", "wiki"]

        def run_sync(self, prompt: str) -> str:  # pragma: no cover
            return prompt

    params = {p.name: p for p in introspect(AgentWithTools())}
    tools = params["agent.tools"]
    assert tools.kind is ParameterKind.TOOL
    assert tools.candidates[0] == ["search", "calculator", "wiki"]
    assert ["calculator", "wiki"] in tools.candidates
    assert ["search", "wiki"] in tools.candidates
    assert ["search", "calculator"] in tools.candidates
    assert len(tools.candidates) == 4


def test_tools_param_not_searchable_with_single_tool() -> None:
    class AgentWithOneTool:
        def __init__(self) -> None:
            self.model = "openai:gpt-4o"
            self._system_prompts = ("be helpful",)
            self.tools = ["search"]

        def run_sync(self, prompt: str) -> str:  # pragma: no cover
            return prompt

    params = {p.name: p for p in introspect(AgentWithOneTool())}
    assert params["agent.tools"].candidates == []


def test_predicate_rejects_multi_agent_object() -> None:
    class Orchestrator:
        model = "x"
        _system_prompts = ("hi",)
        handoffs = []

        def run_sync(self, prompt: str) -> str:  # pragma: no cover
            return prompt

    assert _predicate(Orchestrator()) is False


def test_public_system_prompt_attr() -> None:
    class PublicAgent:
        model = "openai:gpt-4o"
        system_prompt = "be helpful"

        def run_sync(self, prompt: str) -> str:  # pragma: no cover
            return prompt

    agent = PublicAgent()
    assert detect(agent) == "pydantic_ai"
    params = {p.name: p for p in introspect(agent)}
    params["agent.system_prompt"].write("new")
    assert agent.system_prompt == "new"


@pytest.mark.skipif(
    importlib.util.find_spec("pydantic_ai") is None,
    reason="pydantic-ai is not installed",
)
@pytest.mark.parametrize("kwarg", ["instructions", "system_prompt"])
def test_a_real_agent_binds_whichever_prompt_field_it_was_built_with(kwarg: str) -> None:
    """Pydantic AI has two prompt fields and fills only the one you used.

    `Agent(system_prompt=...)` fills `_system_prompts`; `Agent(instructions=...)`
    -- the modern spelling -- fills `_instructions` and leaves the other empty.
    Binding `_system_prompts` unconditionally was worse than finding nothing on
    an `instructions=` agent: the knob started at `''` while the instruction the
    user wrote stayed fixed and still applied, so every candidate was measured
    on top of it and none of them tuned the prompt the agent runs on. (Writes
    to `_system_prompts` do reach the model -- checked against a captured
    request -- which is what makes the failure quiet rather than obvious.)
    """
    import pydantic_ai  # type: ignore[import-not-found]

    agent = pydantic_ai.Agent("test", **{kwarg: "You are terse."})

    prompts = [p for p in introspect(agent) if p.kind is ParameterKind.PROMPT]
    assert len(prompts) == 1
    prompt = prompts[0]
    assert prompt.value == "You are terse.", "bound the empty field, not the populated one"

    prompt.write("Be brief.")
    assert prompt.read() == "Be brief."
    populated = "_instructions" if kwarg == "instructions" else "_system_prompts"
    assert list(getattr(agent, populated)) == ["Be brief."]


# -- a prompt field may hold callables as well as strings ----------------------
#
# A *dynamic* instruction is a function evaluated per run, and either field can
# mix one with static text. Requiring every element to be a string read such a
# list as empty, so it lost the tie to a field that really was empty -- the
# same bug as binding the wrong field, one layer down.


def _dynamic(*args: object, **kwargs: object) -> str:  # pragma: no cover - never called
    return "and be British"


class MixedInstructionsAgent:
    """A Pydantic AI agent whose instructions mix static text with a callable."""

    def __init__(self, instructions: list[object] | None = None) -> None:
        self.model = "openai:gpt-4o"
        self._system_prompts: tuple[object, ...] = ()
        self._instructions: list[object] = (
            ["Be concise", _dynamic] if instructions is None else instructions
        )

    def run_sync(self, prompt: str) -> str:  # pragma: no cover - never called
        return prompt


def test_a_mixed_instruction_list_is_static_text_and_wins_the_tie() -> None:
    agent = MixedInstructionsAgent()
    prompt = next(p for p in introspect(agent) if p.kind is ParameterKind.PROMPT)
    assert prompt.value == "Be concise", "bound the empty field over a mixed one"
    assert prompt.metadata["source"] == "attr:_instructions"


def test_writing_a_prompt_keeps_the_dynamic_instructions() -> None:
    """Replacing the sequence wholesale would delete the callable."""
    agent = MixedInstructionsAgent()
    prompt = next(p for p in introspect(agent) if p.kind is ParameterKind.PROMPT)
    prompt.write("Be terse.")
    assert agent._instructions == ["Be terse.", _dynamic]
    assert prompt.read() == "Be terse."


def test_the_new_text_takes_the_place_of_the_old_one() -> None:
    """Order matters: instructions are concatenated in the order they sit in."""
    agent = MixedInstructionsAgent([_dynamic, "Be concise"])
    prompt = next(p for p in introspect(agent) if p.kind is ParameterKind.PROMPT)
    prompt.write("Be terse.")
    assert agent._instructions == [_dynamic, "Be terse."]


def test_a_field_holding_only_callables_still_beats_an_empty_one() -> None:
    """There is no static text either way, so bind where the agent was configured."""
    agent = MixedInstructionsAgent([_dynamic])
    prompt = next(p for p in introspect(agent) if p.kind is ParameterKind.PROMPT)
    assert prompt.value == ""
    assert prompt.metadata["source"] == "attr:_instructions"
    prompt.write("Be terse.")
    assert agent._instructions == ["Be terse.", _dynamic]


def test_system_prompts_still_breaks_the_tie_when_both_are_empty() -> None:
    agent = MixedInstructionsAgent([])
    prompt = next(p for p in introspect(agent) if p.kind is ParameterKind.PROMPT)
    assert prompt.metadata["source"] == "attr:_system_prompts"


@pytest.mark.skipif(
    importlib.util.find_spec("pydantic_ai") is None,
    reason="pydantic-ai is not installed",
)
@pytest.mark.parametrize("kwarg", ["instructions", "system_prompt"])
def test_a_real_agent_with_a_dynamic_instruction_binds_its_static_text(kwarg: str) -> None:
    import pydantic_ai  # type: ignore[import-not-found]

    agent = pydantic_ai.Agent("test", **{kwarg: ["You are terse.", _dynamic]})

    prompts = [p for p in introspect(agent) if p.kind is ParameterKind.PROMPT]
    assert len(prompts) == 1
    assert prompts[0].value == "You are terse.", "bound the empty field, not the mixed one"

    prompts[0].write("Be brief.")
    populated = "_instructions" if kwarg == "instructions" else "_system_prompts"
    assert list(getattr(agent, populated)) == ["Be brief.", _dynamic]


# -- static text can sit on both sides of a callable ---------------------------
#
# Reading every string and writing them back as one collapsed the interleaving:
# `["before", dynamic, "after"]` became `["before\nafter", dynamic]`, so a plain
# read-then-write reordered the user's agent and a tuned write deleted "after".
# The knob is one contiguous run now, and everything past a callable stays put.


class InterleavedAgent:
    """A Pydantic AI agent with static text on both sides of a callable."""

    def __init__(self, instructions: list[object]) -> None:
        self.model = "openai:gpt-4o"
        self._system_prompts: tuple[object, ...] = ()
        self._instructions: list[object] = instructions

    def run_sync(self, prompt: str) -> str:  # pragma: no cover - never called
        return prompt


@pytest.mark.parametrize(
    ("instructions", "expected_value"),
    [
        (["before", _dynamic, "after"], "before"),
        ([_dynamic, "mid", _dynamic], "mid"),
        (["a", "b", _dynamic], "a\nb"),
        (["only"], "only"),
    ],
    ids=["between", "surrounded", "adjacent-pair", "single"],
)
def test_the_prompt_is_the_first_run_of_static_text(
    instructions: list[object], expected_value: str
) -> None:
    agent = InterleavedAgent(list(instructions))
    prompt = next(p for p in introspect(agent) if p.kind is ParameterKind.PROMPT)
    assert prompt.value == expected_value


@pytest.mark.parametrize(
    ("instructions", "after_write"),
    [
        (["before", _dynamic, "after"], ["TUNED", _dynamic, "after"]),
        ([_dynamic, "mid", _dynamic], [_dynamic, "TUNED", _dynamic]),
        (["a", "b", _dynamic], ["TUNED", _dynamic]),
        ([_dynamic], ["TUNED", _dynamic]),
    ],
    ids=["between", "surrounded", "adjacent-pair", "callable-only"],
)
def test_a_write_replaces_the_run_and_moves_nothing_else(
    instructions: list[object], after_write: list[object]
) -> None:
    agent = InterleavedAgent(list(instructions))
    prompt = next(p for p in introspect(agent) if p.kind is ParameterKind.PROMPT)
    prompt.write("TUNED")
    assert agent._instructions == after_write


def test_reading_and_writing_back_changes_nothing() -> None:
    """`Optimizer.optimize` restores its snapshot, so this runs on every sweep.

    A knob whose round-trip is not the identity mutates the user's agent just
    by being optimized over, whether or not any candidate wins.
    """
    for instructions in (
        ["before", _dynamic, "after"],
        [_dynamic, "mid", _dynamic],
        ["only"],
        [_dynamic],
    ):
        agent = InterleavedAgent(list(instructions))
        prompt = next(p for p in introspect(agent) if p.kind is ParameterKind.PROMPT)
        prompt.write(prompt.value)
        assert agent._instructions == instructions


@pytest.mark.skipif(
    importlib.util.find_spec("pydantic_ai") is None,
    reason="pydantic-ai is not installed",
)
@pytest.mark.parametrize(
    "instructions",
    [["before", _dynamic, "after"], [_dynamic, "mid", _dynamic], ["a", "b", _dynamic], ["only"]],
    ids=["between", "surrounded", "adjacent-pair", "single"],
)
def test_a_real_round_trip_leaves_the_rendered_prompt_identical(
    instructions: list[object],
) -> None:
    """The property that matters is the text the model receives, not the list.

    An adjacent pair does collapse into one element -- two strings cannot be
    recovered from their join -- and that is safe only because `\n` is the
    separator Pydantic AI puts between them, which this asserts rather than
    assumes.
    """
    import pydantic_ai  # type: ignore[import-not-found]
    from pydantic_ai.models.test import TestModel  # type: ignore[import-not-found]

    def rendered(agent: object) -> list[str]:
        captured: list[object] = []

        class _Capture(TestModel):  # type: ignore[misc]
            async def request(self, messages, *args, **kwargs):  # type: ignore[no-untyped-def]
                captured.append(messages)
                return await super().request(messages, *args, **kwargs)

        agent.model = _Capture()  # type: ignore[attr-defined]
        agent.run_sync("hi")  # type: ignore[attr-defined]
        return [m.instructions for m in captured[0] if getattr(m, "instructions", None)]

    before = rendered(pydantic_ai.Agent("test", instructions=list(instructions)))
    agent = pydantic_ai.Agent("test", instructions=list(instructions))
    prompt = next(p for p in introspect(agent) if p.kind is ParameterKind.PROMPT)
    prompt.write(prompt.value)
    assert rendered(agent) == before
