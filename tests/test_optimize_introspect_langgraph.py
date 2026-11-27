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


def test_model_top_p_bound_when_present() -> None:
    model = FakeModel(model_name="gpt-4o", temperature=0.4)
    model.top_p = 0.9
    graph = FakeCompiledGraph(
        nodes={"writer": FakeNode(FakeRunnable(system_prompt="Write.", model=model))}
    )
    by_name = {p.name: p for p in introspect(graph)}

    top_p = by_name["writer.top_p"]
    assert top_p.kind is ParameterKind.HYPERPARAM
    assert top_p.bounds == (0.0, 1.0)
    top_p.write(0.5)
    assert model.top_p == 0.5


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


# -- prebuilt agents: the prompt, model and tools live in closures --------------
#
# ``create_react_agent`` and ``create_agent`` are the most common LangGraph
# agents and, being closures over their configuration, introspected to nothing.


def _recording_model():
    """An offline chat model that records the messages and tool names it is given."""
    pytest.importorskip("langchain_core")
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class Recording(BaseChatModel):
        model_name: str = "fake-model"
        temperature: float = 0.5
        seen: list = []
        tools_offered: list = []

        @property
        def _llm_type(self) -> str:
            return "recording"

        def bind_tools(self, tools, **kwargs):
            self.tools_offered.append([getattr(t, "name", str(t)) for t in tools])
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.seen.append([m.content for m in messages])
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])

    model = Recording()
    model.seen, model.tools_offered = [], []
    return model


def _two_tools():
    pytest.importorskip("langchain_core")
    from langchain_core.tools import tool

    @tool
    def fetch(url: str) -> str:
        """Fetch a url."""
        return url

    @tool
    def delete_db(name: str) -> str:
        """Delete a database."""
        return name

    return [fetch, delete_db]


def _prebuilt_agent(kind: str, model):
    import warnings

    tools = _two_tools()
    if kind == "react":
        prebuilt = pytest.importorskip("langgraph.prebuilt")
        with warnings.catch_warnings():  # create_react_agent is deprecated upstream
            warnings.simplefilter("ignore")
            return prebuilt.create_react_agent(model, tools, prompt="You are terse.")
    agents = pytest.importorskip("langchain.agents")
    return agents.create_agent(model, tools, system_prompt="You are terse.")


@pytest.mark.parametrize("kind", ["react", "create_agent"])
def test_a_prebuilt_agent_prompt_and_model_are_tunable_and_reach_the_model(kind: str) -> None:
    import asyncio

    model = _recording_model()
    graph = _prebuilt_agent(kind, model)
    assert detect(graph) == "langgraph"

    params = {p.name.split(".", 1)[1]: p for p in introspect(graph)}
    prompt = params["system_prompt"]
    assert prompt.kind is ParameterKind.PROMPT
    assert prompt.read() == "You are terse."
    assert params["model"].read() == "fake-model"
    assert params["temperature"].bounds == (0.0, 2.0)

    prompt.write("Be verbose.")
    params["temperature"].write(0.1)
    params["model"].write("other-model")
    graph.invoke({"messages": [("user", "hi")]})
    assert model.seen[-1] == ["Be verbose.", "hi"]
    assert (model.temperature, model.model_name) == (0.1, "other-model")

    # The async entry point closes over the same cells, so it sees the write too.
    asyncio.run(graph.ainvoke({"messages": [("user", "again")]}))
    assert model.seen[-1] == ["Be verbose.", "again"]


@pytest.mark.parametrize("kind", ["react", "create_agent"])
def test_a_failed_prebuilt_prompt_write_raises_and_changes_nothing(kind: str) -> None:
    model = _recording_model()
    graph = _prebuilt_agent(kind, model)
    prompt = next(p for p in introspect(graph) if p.name.endswith("system_prompt"))
    assert prompt.setter is not None
    with pytest.raises(TypeError):
        prompt.setter(None)
    graph.invoke({"messages": [("user", "hi")]})
    assert model.seen[-1] == ["You are terse.", "hi"]


def test_create_agent_tool_list_can_be_narrowed_and_the_model_sees_it() -> None:
    model = _recording_model()
    graph = _prebuilt_agent("create_agent", model)
    tools = next(p for p in introspect(graph) if p.kind is ParameterKind.TOOL)
    assert tools.optimizable
    assert [t.name for t in tools.read()] == ["fetch", "delete_db"]

    tools.write(tools.candidates[1])  # drop `fetch`
    graph.invoke({"messages": [("user", "hi")]})
    assert model.tools_offered[-1] == ["delete_db"]

    tools.write(tools.candidates[0])
    graph.invoke({"messages": [("user", "hi")]})
    assert model.tools_offered[-1] == ["fetch", "delete_db"]

    assert tools.setter is not None
    with pytest.raises(ValueError):
        tools.setter(["not a tool"])
    assert [t.name for t in tools.read()] == ["fetch", "delete_db"]


def test_create_react_agent_tools_are_reported_but_not_writable() -> None:
    """Its tool schemas are baked into the bound model, so narrowing would half-apply."""
    graph = _prebuilt_agent("react", _recording_model())
    tools = next(p for p in introspect(graph) if p.kind is ParameterKind.TOOL)
    assert [t.name for t in tools.read()] == ["fetch", "delete_db"]
    assert not tools.optimizable
