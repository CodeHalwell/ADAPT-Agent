"""Tests for the LangGraph optimization introspector.

These tests build plain fake objects mimicking a *compiled* LangGraph graph's
attribute shape (``invoke`` + ``nodes`` mapping with node runnables carrying a
``system_prompt`` and a bound model). The real ``langgraph`` package is never
imported.
"""

from adapt_agent.optimization.introspection import detect, introspect
from adapt_agent.optimization.introspection.langgraph import _predicate
from adapt_agent.optimization.parameters import ParameterKind


class FakeModel:
    def __init__(self, model_name: str, temperature: float) -> None:
        self.model_name = model_name
        self.temperature = temperature


class FakeRunnable:
    def __init__(self, system_prompt: str, model: FakeModel) -> None:
        self.system_prompt = system_prompt
        self.model = model


class FakeNode:
    def __init__(self, runnable: FakeRunnable) -> None:
        self.runnable = runnable


class FakeCompiledGraph:
    def __init__(self, nodes) -> None:
        self.nodes = nodes

    def invoke(self, state):  # pragma: no cover - never called
        return state


def _make_graph() -> FakeCompiledGraph:
    node = FakeNode(
        FakeRunnable(
            system_prompt="You are a careful researcher.",
            model=FakeModel(model_name="gpt-4o", temperature=0.4),
        )
    )
    return FakeCompiledGraph(nodes={"researcher": node})


def test_detect_routes_to_langgraph() -> None:
    assert detect(_make_graph()) == "langgraph"


def test_predicate_false_for_unrelated_object() -> None:
    assert _predicate(object()) is False
    assert detect(object()) is None


def test_predicate_false_for_crewai_like_object() -> None:
    class FakeCrew:
        def __init__(self) -> None:
            self.agents = []
            self.nodes = {}

        def invoke(self, state):  # pragma: no cover - never called
            return state

        def kickoff(self):  # pragma: no cover - never called
            return None

    assert _predicate(FakeCrew()) is False
    assert detect(FakeCrew()) != "langgraph"


def test_introspect_emits_expected_params() -> None:
    params = {p.name: p for p in introspect(_make_graph())}

    assert "researcher.system_prompt" in params
    assert params["researcher.system_prompt"].kind is ParameterKind.PROMPT
    assert params["researcher.system_prompt"].component == "researcher"

    assert "researcher.model" in params
    assert params["researcher.model"].kind is ParameterKind.MODEL
    assert params["researcher.model"].read() == "gpt-4o"

    assert "researcher.temperature" in params
    assert params["researcher.temperature"].kind is ParameterKind.HYPERPARAM
    assert params["researcher.temperature"].bounds == (0.0, 2.0)


def test_prompt_setter_round_trips() -> None:
    graph = _make_graph()
    params = {p.name: p for p in introspect(graph)}
    prompt = params["researcher.system_prompt"]

    assert prompt.read() == "You are a careful researcher."
    prompt.write("You are a meticulous fact-checker.")

    assert graph.nodes["researcher"].runnable.system_prompt == (
        "You are a meticulous fact-checker."
    )
    # A fresh introspection observes the mutated value too.
    refreshed = {p.name: p for p in introspect(graph)}
    assert refreshed["researcher.system_prompt"].read() == "You are a meticulous fact-checker."
