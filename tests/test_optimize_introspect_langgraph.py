"""Tests for the LangGraph optimization introspector.

These tests build plain fake objects mimicking a *compiled* LangGraph graph's
attribute shape (``invoke`` + ``nodes`` mapping with node runnables carrying a
``system_prompt`` and a bound model). The real ``langgraph`` package is never
imported.
"""

import importlib.util

import pytest

from adapt_agent.optimization.introspection import detect, introspect
from adapt_agent.optimization.introspection.langgraph import _predicate
from adapt_agent.optimization.parameters import ParameterKind


class FakeModel:
    def __init__(self, model_name: str, temperature: float) -> None:
        self.model_name = model_name
        self.temperature = temperature


class FakeRunnable:
    def __init__(self, system_prompt: str, model: FakeModel, tools=None) -> None:
        self.system_prompt = system_prompt
        self.model = model
        if tools is not None:
            self.tools = tools


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


def test_predicate_false_for_microsoft_chat_agent_like_object() -> None:
    # A Microsoft Agent Framework ``ChatAgent`` carries ``instructions`` +
    # ``chat_client`` + a callable ``run``; the langgraph introspector (registered
    # earlier) must never hijack it via its ``invoke``/``nodes`` shape.
    class FakeChatAgent:
        def __init__(self) -> None:
            self.instructions = "be helpful"
            self.chat_client = object()
            self.nodes = {}

        def invoke(self, state):  # pragma: no cover - never called
            return state

        def run(self, *a, **k):  # pragma: no cover - never called
            return None

    assert _predicate(FakeChatAgent()) is False
    assert detect(FakeChatAgent()) != "langgraph"


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


def test_tools_param_is_optimizable_with_drop_one_candidates() -> None:
    node = FakeNode(
        FakeRunnable(
            system_prompt="You are a careful researcher.",
            model=FakeModel(model_name="gpt-4o", temperature=0.4),
            tools=["search", "calculator", "wiki"],
        )
    )
    graph = FakeCompiledGraph(nodes={"researcher": node})
    params = {p.name: p for p in introspect(graph)}

    tools = params["researcher.tools"]
    assert tools.kind is ParameterKind.TOOL
    # Full set first, then each drop-one subset.
    assert tools.candidates[0] == ["search", "calculator", "wiki"]
    assert ["calculator", "wiki"] in tools.candidates
    assert ["search", "wiki"] in tools.candidates
    assert ["search", "calculator"] in tools.candidates
    assert len(tools.candidates) == 4


def test_tools_param_not_searchable_with_single_tool() -> None:
    node = FakeNode(
        FakeRunnable(
            system_prompt="You are a careful researcher.",
            model=FakeModel(model_name="gpt-4o", temperature=0.4),
            tools=["search"],
        )
    )
    graph = FakeCompiledGraph(nodes={"researcher": node})
    params = {p.name: p for p in introspect(graph)}
    # A single tool yields no meaningful subset to search.
    assert params["researcher.tools"].candidates == []


def test_introspect_reads_nodes_via_get_graph() -> None:
    # Some compiled graphs expose nodes only through ``get_graph().nodes``.
    class GraphView:
        def __init__(self, nodes) -> None:
            self.nodes = nodes

    class GraphWithGetGraph:
        def __init__(self, nodes) -> None:
            self._nodes = nodes

        def invoke(self, state):  # pragma: no cover - never called
            return state

        def get_graph(self):
            return GraphView(self._nodes)

    node = FakeNode(
        FakeRunnable(
            system_prompt="You are a careful researcher.",
            model=FakeModel(model_name="gpt-4o", temperature=0.3),
        )
    )
    graph = GraphWithGetGraph(nodes={"researcher": node})

    assert detect(graph) == "langgraph"
    params = {p.name: p for p in introspect(graph)}
    assert "researcher.system_prompt" in params
    assert params["researcher.system_prompt"].kind is ParameterKind.PROMPT


@pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None,
    reason="langgraph is not installed",
)
def test_a_real_compiled_graph_yields_tunable_parameters() -> None:
    """A compiled graph does not hand back the callable you registered.

    `PregelNode.bound` is a `RunnableCallable` wrapper; the node object you
    added sits one hop further down at `.func`. Walking only as far as `.bound`
    inspected the wrapper -- which exposes no prompt and no model -- so every
    realistic graph introspected to zero parameters while still being *detected*
    as LangGraph. That reads as "this graph has nothing to tune" rather than as
    a broken walk, which is why no fake in this module caught it.
    """
    from typing import Annotated, TypedDict

    from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]
    from langgraph.graph.message import add_messages  # type: ignore[import-not-found]

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    class _Model:
        model_name = "gpt-4o-mini"
        temperature = 0.2

    class ChatNode:
        def __init__(self) -> None:
            self.system_prompt = "You are a helpful agent."
            self.model = _Model()

        def __call__(self, state):  # pragma: no cover - never invoked
            return {"messages": []}

    builder = StateGraph(State)
    builder.add_node("chat", ChatNode())
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    graph = builder.compile()

    assert detect(graph) == "langgraph"

    params = {p.name: p for p in introspect(graph)}
    assert params["chat.system_prompt"].kind is ParameterKind.PROMPT
    assert params["chat.system_prompt"].value == "You are a helpful agent."
    assert params["chat.model"].value == "gpt-4o-mini"
    assert params["chat.temperature"].value == 0.2

    # Writable, not merely discoverable: the write must reach the node object.
    params["chat.system_prompt"].write("Be terse.")
    assert graph.nodes["chat"].bound.func.system_prompt == "Be terse."
