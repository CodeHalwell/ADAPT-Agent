"""Offline tests for Pydantic AI introspection.

These build fake objects mimicking a Pydantic AI ``Agent`` (the real framework
is not installed) and assert the registry routes them, the discovered parameters
have the expected names/kinds, the system-prompt setter mutates the live object,
and the predicate rejects unrelated objects. No network, no framework import.
"""

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


def test_predicate_rejects_unrelated_object() -> None:
    assert _predicate(object()) is False


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
