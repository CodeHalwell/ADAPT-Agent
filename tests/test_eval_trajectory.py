"""Trajectory (tool-call) evaluation: extraction from real framework results + metrics.

Framework tests build genuine result objects offline -- scripted models, no
network -- and skip when the framework is not installed.
"""

from __future__ import annotations

import asyncio
import logging
import warnings

import pytest

from adapt_agent.optimization.dataset import Example, GoldenDataset
from adapt_agent.optimization.evaluation import EvaluationHarness
from adapt_agent.optimization.extractors import (
    ToolCall,
    Trajectory,
    extract_output_text,
    extract_trajectory,
    extract_usage,
)
from adapt_agent.optimization.metrics import checks, exact_match, get_metric
from adapt_agent.optimization.trajectory import (
    no_forbidden_tools,
    tool_call_args_match,
    tool_call_budget,
    tool_called,
    tool_sequence,
)


def _calls(trajectory: Trajectory) -> list[tuple[str, dict]]:
    return [(c.name, c.arguments) for c in trajectory.tool_calls]


# -- the custom-agent dict form ---------------------------------------------------


def test_custom_dict_form_is_extracted() -> None:
    result = {
        "output": "sunny",
        "tool_calls": [
            {"name": "search", "arguments": {"q": "weather"}, "result": "r1"},
            {"name": "fetch", "args": '{"url": "u"}'},
        ],
        "steps": 3,
    }
    trajectory = extract_trajectory(result)
    assert _calls(trajectory) == [("search", {"q": "weather"}), ("fetch", {"url": "u"})]
    assert trajectory.tool_calls[0].result == "r1"
    assert trajectory.steps == 3


def test_openai_chat_completions_messages_pair_results_by_id() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "s", "arguments": '{"a": 1}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "found"},
        {"role": "assistant", "content": "done"},
    ]
    trajectory = extract_trajectory({"messages": messages})
    assert _calls(trajectory) == [("s", {"a": 1})]
    assert trajectory.tool_calls[0].result == "found"
    assert trajectory.steps == 2


def test_unrecognised_value_yields_an_empty_trajectory() -> None:
    assert extract_trajectory("just text").tool_calls == []
    assert extract_trajectory(object()).tool_calls == []


# -- metrics -------------------------------------------------------------------------

_RUN = Trajectory(
    tool_calls=[
        ToolCall("search", {"q": "Weather in Paris", "limit": 3}),
        ToolCall("fetch", {"url": "https://x"}),
        ToolCall("search", {"q": "London"}),
    ],
    steps=4,
)


def _row(**meta: object) -> Example:
    return Example(inputs="q", expected="a", metadata=dict(meta))


def test_tool_called_fixed_name_and_row_declared() -> None:
    assert tool_called("fetch")(_RUN, None, _row()) == 1.0
    assert tool_called("delete")(_RUN, None, _row()) == 0.0
    assert tool_called("search", min_times=3)(_RUN, None, _row()) == 0.0
    row = _row(expected_tools=["search", "missing"])
    assert tool_called()(_RUN, None, row) == 0.5


def test_args_match_is_a_subset_with_text_matchers() -> None:
    metric = tool_call_args_match()
    # Subset (extra `limit` ignored) and normalised text (case, edge punctuation).
    row = _row(expected_tools=[{"name": "search", "args": {"q": "weather in paris."}}])
    assert metric(_RUN, None, row) == 1.0
    row = _row(
        expected_tools=[{"name": "search", "args": {"q": {"$check": "contains", "value": "paris"}}}]
    )
    assert metric(_RUN, None, row) == 1.0
    row = _row(expected_tools=[{"name": "search", "args": {"q": "Berlin"}}])
    assert metric(_RUN, None, row) == 0.0
    # Each actual call satisfies one expectation only.
    row = _row(expected_tools=[{"name": "fetch", "args": {}}, {"name": "fetch", "args": {}}])
    assert metric(_RUN, None, row) == 0.5
    # A bool expectation does not match an int argument.
    run = Trajectory([ToolCall("t", {"flag": 1})])
    assert metric(run, None, _row(expected_tools=[{"name": "t", "args": {"flag": True}}])) == 0.0


@pytest.mark.parametrize(
    ("mode", "expected_tools", "score"),
    [
        ("exact", ["search", "fetch", "search"], 1.0),
        ("exact", ["search", "fetch"], 0.0),
        ("in_order", ["search", "search"], 1.0),
        ("in_order", ["fetch", "search"], 1.0),
        ("in_order", ["search", "fetch", "fetch"], 0.0),
        ("any_order", ["fetch", "search"], 1.0),
        ("any_order", ["fetch", "fetch"], 0.0),
        ("no_extra", ["fetch", "search", "search"], 1.0),
        ("no_extra", ["search", "search"], 0.0),
    ],
)
def test_tool_sequence_modes(mode: str, expected_tools: list[str], score: float) -> None:
    assert tool_sequence(mode=mode)(_RUN, None, _row(expected_tools=expected_tools)) == score


def test_tool_sequence_can_check_arguments() -> None:
    row = _row(expected_tools=[{"name": "search", "args": {"q": "London"}}])
    assert tool_sequence(check_args=True)(_RUN, None, row) == 1.0
    row = _row(expected_tools=[{"name": "search", "args": {"q": "Rome"}}])
    assert tool_sequence(check_args=True)(_RUN, None, row) == 0.0
    assert tool_sequence(check_args=False)(_RUN, None, row) == 1.0


def test_forbidden_tools_and_budgets() -> None:
    assert no_forbidden_tools(["delete_*"])(_RUN, None, _row()) == 1.0
    assert no_forbidden_tools()(_RUN, None, _row(forbidden_tools=["fet*"])) == 0.0
    assert tool_call_budget(3)(_RUN, None, _row()) == 1.0
    assert tool_call_budget(2)(_RUN, None, _row()) == 0.0
    assert tool_call_budget(max_steps=3)(_RUN, None, _row()) == 0.0
    assert tool_call_budget()(_RUN, None, _row(max_tool_calls=5, max_steps=4)) == 1.0
    with pytest.raises(ValueError, match="step count"):
        tool_call_budget(max_steps=3)(Trajectory([ToolCall("a")]), None, _row())


def test_a_row_without_expectations_is_an_error_not_a_pass() -> None:
    harness = EvaluationHarness([tool_sequence()])
    report = harness.evaluate(lambda x: {"output": "x", "tool_calls": []}, GoldenDataset([_row()]))
    assert report.score == 0.0
    assert "tool_sequence" in report.results[0].metric_errors
    assert report.n_metric_errors == 1


def test_harness_reads_the_raw_result_despite_text_extraction() -> None:
    def agent(_: object) -> dict:
        return {"output": "Paris", "tool_calls": [{"name": "lookup", "args": {"city": "Paris"}}]}

    data = GoldenDataset.from_list(
        [
            {
                "input": "capital?",
                "expected": "Paris",
                "expected_tools": [{"name": "lookup", "args": {"city": "paris"}}],
                "forbidden_tools": ["delete_*"],
            }
        ]
    )
    harness = EvaluationHarness(
        [exact_match(), tool_call_args_match(), no_forbidden_tools()],
        output_extractor=extract_output_text,
    )
    report = harness.evaluate(agent, data)
    assert report.aggregate == {
        "exact_match": 1.0,
        "tool_call_args_match": 1.0,
        "no_forbidden_tools": 1.0,
    }


def test_per_row_checks_can_name_trajectory_metrics() -> None:
    def agent(_: object) -> dict:
        return {"output": "ok", "tool_calls": [{"name": "a"}, {"name": "b"}]}

    data = GoldenDataset.from_list(
        [
            {"input": "1", "expected": "ok"},
            {
                "input": "2",
                "check": {"name": "tool_sequence", "mode": "exact"},
                "expected_tools": ["a", "b"],
            },
            {"input": "3", "check": "tool_sequence", "expected_tools": ["b", "a"]},
        ]
    )
    report = EvaluationHarness([checks()]).evaluate(agent, data)
    assert [r.scores["checks"] for r in report.results] == [1.0, 1.0, 0.0]
    assert get_metric("tool_called").name == "tool_called"


def test_concurrent_rows_each_see_their_own_trajectory() -> None:
    def agent(x: str) -> dict:
        return {"output": x, "tool_calls": [{"name": f"tool_{x}"}]}

    data = GoldenDataset.from_list(
        [{"input": str(i), "expected": str(i), "expected_tools": [f"tool_{i}"]} for i in range(20)]
    )
    report = EvaluationHarness([tool_sequence(mode="exact")]).evaluate(agent, data, concurrency=8)
    assert report.score == 1.0


# -- real framework results ----------------------------------------------------------


def test_langgraph_react_agent_trajectory() -> None:
    pytest.importorskip("langgraph")
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    @tool
    def search(q: str) -> str:
        """Search."""
        return "result for " + q

    class Model(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):  # type: ignore[no-untyped-def]
            return self

    model = Model(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search", "args": {"q": "weather"}, "id": "c1"}],
                    usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                ),
                AIMessage(content="It is sunny"),
            ]
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = create_react_agent(model, [search])
        state = graph.invoke({"messages": [{"role": "user", "content": "weather?"}]})
    trajectory = extract_trajectory(state)
    assert _calls(trajectory) == [("search", {"q": "weather"})]
    assert trajectory.tool_calls[0].result == "result for weather"
    assert trajectory.steps == 2
    assert extract_usage(state) == {"input_tokens": 7.0, "output_tokens": 3.0, "total_tokens": 10.0}


def test_pydantic_ai_trajectory() -> None:
    pytest.importorskip("pydantic_ai")
    import os

    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    def respond(messages, info):  # type: ignore[no-untyped-def]
        if len(messages) == 1:
            return ModelResponse(
                parts=[ToolCallPart("search", '{"q": "weather"}', tool_call_id="t1")]
            )
        return ModelResponse(parts=[TextPart("sunny")])

    agent = Agent(FunctionModel(respond))

    @agent.tool_plain
    def search(q: str) -> str:
        return "r:" + q

    result = agent.run_sync("weather?")
    trajectory = extract_trajectory(result)
    assert _calls(trajectory) == [("search", {"q": "weather"})]
    assert trajectory.tool_calls[0].result == "r:weather"
    assert trajectory.steps == 2
    usage = extract_usage(result)
    assert usage is not None and usage["input_tokens"] > 0


def test_openai_agents_trajectory() -> None:
    pytest.importorskip("agents")
    from agents import Agent, Runner, function_tool, set_tracing_disabled
    from agents.testing import ScriptedModel, assistant_message, function_call

    set_tracing_disabled(True)

    @function_tool
    def search(q: str) -> str:
        "search"
        return "r:" + q

    agent = Agent(
        name="A",
        instructions="x",
        tools=[search],
        model=ScriptedModel(
            [
                [function_call("search", {"q": "weather"}, call_id="c1")],
                [assistant_message("sunny")],
            ]
        ),
    )
    result = Runner.run_sync(agent, "hi")
    trajectory = extract_trajectory(result)
    assert _calls(trajectory) == [("search", {"q": "weather"})]
    assert trajectory.tool_calls[0].result == "r:weather"
    assert trajectory.steps == 2


def test_google_adk_trajectory() -> None:
    pytest.importorskip("google.adk")
    from google.adk.agents import LlmAgent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    class Fake(BaseLlm):
        script: list = []

        async def generate_content_async(self, llm_request, stream=False):  # type: ignore[no-untyped-def]
            step = self.script.pop(0)
            parts = step if isinstance(step, list) else [types.Part(text=step)]
            yield LlmResponse(content=types.Content(role="model", parts=parts))

    def search(q: str) -> dict:
        "search"
        return {"body": "r:" + q}

    call = types.Part(
        function_call=types.FunctionCall(name="search", args={"q": "weather"}, id="f1")
    )
    agent = LlmAgent(
        name="a",
        model=Fake(model="fake", script=[[call], "sunny"]),
        instruction="x",
        tools=[search],
    )

    async def run() -> list:
        runner = InMemoryRunner(agent=agent, app_name="t")
        await runner.session_service.create_session(app_name="t", user_id="u", session_id="s")
        message = types.Content(role="user", parts=[types.Part(text="hi")])
        return [e async for e in runner.run_async(user_id="u", session_id="s", new_message=message)]

    logging.disable(logging.ERROR)
    try:
        events = asyncio.run(run())
    finally:
        logging.disable(logging.NOTSET)
    trajectory = extract_trajectory(events)
    assert _calls(trajectory) == [("search", {"q": "weather"})]
    assert trajectory.tool_calls[0].result == {"body": "r:weather"}
    assert trajectory.steps == 2


def test_microsoft_agent_framework_trajectory() -> None:
    af = pytest.importorskip("agent_framework")

    class FakeClient(af.FunctionInvocationLayer, af.ChatMiddlewareLayer, af.BaseChatClient):
        def __init__(self, script, **kw):  # type: ignore[no-untyped-def]
            super().__init__(**kw)
            self.script = list(script)

        def _inner_get_response(self, *, messages, stream, options, **kwargs):  # type: ignore[no-untyped-def]
            step = self.script.pop(0)
            contents = step if isinstance(step, list) else [af.Content.from_text(step)]

            async def _respond():  # type: ignore[no-untyped-def]
                return af.ChatResponse(
                    messages=[af.Message(role="assistant", contents=contents)], response_id="r"
                )

            return _respond()

    @af.tool
    def search(q: str) -> str:
        "search"
        return "r:" + q

    call = af.Content.from_function_call(call_id="m1", name="search", arguments={"q": "weather"})
    agent = af.Agent(FakeClient([[call], "sunny"]), "be nice", name="a", tools=[search])
    response = asyncio.run(agent.run("hi"))
    trajectory = extract_trajectory(response)
    assert _calls(trajectory) == [("search", {"q": "weather"})]
    assert trajectory.tool_calls[0].result == "r:weather"
    assert trajectory.steps == 2


def test_claude_agent_sdk_trajectory() -> None:
    sdk = pytest.importorskip("claude_agent_sdk")
    stream = [
        sdk.AssistantMessage(
            content=[
                sdk.TextBlock(text="Let me look."),
                sdk.ToolUseBlock(id="tu1", name="Read", input={"file_path": "/a"}),
            ],
            model="m",
        ),
        sdk.UserMessage(content=[sdk.ToolResultBlock(tool_use_id="tu1", content="contents")]),
        sdk.AssistantMessage(content=[sdk.TextBlock(text="Done.")], model="m"),
        sdk.ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=2,
            session_id="s",
            total_cost_usd=0.01,
            usage={"input_tokens": 5, "output_tokens": 2},
            result="Done.",
        ),
    ]
    trajectory = extract_trajectory(stream)
    assert _calls(trajectory) == [("Read", {"file_path": "/a"})]
    assert trajectory.tool_calls[0].result == "contents"
    assert trajectory.steps == 2
    assert extract_usage(stream) == {"input_tokens": 5.0, "output_tokens": 2.0, "cost_usd": 0.01}
