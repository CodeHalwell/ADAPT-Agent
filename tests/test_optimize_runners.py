"""Offline tests for ``adapt_agent.optimization.runners``.

Framework shapes are mimicked with stubs (the same duck-typing the runners rely
on); no agent framework is imported or required.
"""

import pytest

from adapt_agent.optimization.runners import (
    AUTO,
    adk_runner,
    framework_runner,
    langgraph_inputs,
)

# -- langgraph_inputs ----------------------------------------------------------


def test_langgraph_inputs_wraps_string():
    assert langgraph_inputs("hi") == {"messages": [{"role": "user", "content": "hi"}]}


def test_langgraph_inputs_wraps_message_list():
    messages = [{"role": "user", "content": "hi"}]
    assert langgraph_inputs(messages) == {"messages": messages}


def test_langgraph_inputs_passes_mappings_through():
    state = {"messages": [], "context": {}}
    assert langgraph_inputs(state) is state


# -- framework_runner ----------------------------------------------------------


class _AgentRunResult:
    """Pydantic-AI-shaped run result."""

    def __init__(self, output):
        self.output = output

    def all_messages(self):  # pragma: no cover - presence is what matters
        return []


def test_framework_runner_extracts_from_plain_callable():
    run = framework_runner(lambda q: _AgentRunResult(f"answer:{q}"))
    assert run("x") == "answer:x"


def test_framework_runner_awaits_async_callable():
    async def agent(q):
        return _AgentRunResult(q.upper())

    run = framework_runner(agent)
    assert run("paris") == "PARIS"


def test_framework_runner_uses_run_sync_method():
    class PydanticAIAgentish:
        def run_sync(self, q):
            return _AgentRunResult(f"4 for {q}")

    run = framework_runner(PydanticAIAgentish())
    assert run("2+2") == "4 for 2+2"


def test_framework_runner_awaits_maf_style_run():
    class ChatMessage:
        def __init__(self, text):
            self.role = "assistant"
            self.text = text
            self.contents = []

    class AgentRunResponse:
        def __init__(self, text):
            self.messages = [ChatMessage(text)]
            self.text = text

    class MAFAgentish:
        async def run(self, q):
            return AgentRunResponse(f"echo {q}")

    run = framework_runner(MAFAgentish())
    assert run("hello") == "echo hello"


class _Graph:
    """Compiled-LangGraph-shaped object: callable ``invoke`` + ``nodes``."""

    def __init__(self):
        self.nodes = {}
        self.seen_inputs = []

    def invoke(self, state):
        self.seen_inputs.append(state)
        question = state["messages"][-1]["content"]
        return {
            "messages": [*state["messages"], {"role": "assistant", "content": f"re: {question}"}]
        }


def test_framework_runner_auto_adapts_langgraph_string_inputs():
    graph = _Graph()
    run = framework_runner(graph)
    assert run("what is 2+2?") == "re: what is 2+2?"
    # The bare string was wrapped into message-state form before invoke.
    assert graph.seen_inputs[0] == {"messages": [{"role": "user", "content": "what is 2+2?"}]}


def test_framework_runner_auto_leaves_native_state_untouched():
    graph = _Graph()
    run = framework_runner(graph)
    state = {"messages": [{"role": "user", "content": "hi"}]}
    assert run(state) == "re: hi"
    assert graph.seen_inputs[0] is state


def test_framework_runner_input_adapter_none_disables_adaptation():
    graph = _Graph()
    run = framework_runner(graph, input_adapter=None)
    with pytest.raises(TypeError):
        run("bare string reaches invoke unadapted")


def test_framework_runner_custom_input_adapter():
    run = framework_runner(lambda payload: payload["q"], input_adapter=lambda s: {"q": s})
    assert run("hi") == "hi"


def test_framework_runner_output_extractor_none_keeps_raw():
    result = _AgentRunResult("raw")
    run = framework_runner(lambda q: result, output_extractor=None)
    assert run("x") is result


def test_framework_runner_rejects_bad_input_adapter():
    with pytest.raises(ValueError):
        framework_runner(lambda q: q, input_adapter="magic")


def test_auto_sentinel_exported():
    assert AUTO == "auto"


# -- adk_runner ------------------------------------------------------------------


class _Part:
    def __init__(self, text=None):
        self.text = text


class _Content:
    def __init__(self, parts, role="model"):
        self.parts = parts
        self.role = role


class _Event:
    def __init__(self, text):
        self.content = _Content([_Part(text=text)])
        self.author = "agent"


class _FakeSessionService:
    def __init__(self, asynchronous=False):
        self.created = []
        self.asynchronous = asynchronous

    def create_session(self, *, app_name, user_id, session_id):
        if self.asynchronous:
            return self._create_async(app_name, user_id, session_id)
        self.created.append((app_name, user_id, session_id))
        return object()

    async def _create_async(self, app_name, user_id, session_id):
        self.created.append((app_name, user_id, session_id))
        return object()


class _FakeRunner:
    """Google-ADK-Runner-shaped: kwargs-driven ``run`` generator + session service."""

    def __init__(self, app_name="fake-app", asynchronous_sessions=False):
        self.app_name = app_name
        self.session_service = _FakeSessionService(asynchronous=asynchronous_sessions)
        self.calls = []

    def run(self, *, user_id, session_id, new_message):
        self.calls.append({"user_id": user_id, "session_id": session_id, "message": new_message})
        yield _Event(None)  # a no-text event (e.g. a tool call)
        yield _Event(f"final: {new_message}")


def test_adk_runner_drives_prebuilt_runner():
    fake = _FakeRunner()
    run = adk_runner(fake, message_factory=lambda s: s)
    assert run("what is 2+2?") == "final: what is 2+2?"
    # Session was created on the runner's own app_name before running.
    assert fake.session_service.created[0][0] == "fake-app"
    assert fake.calls[0]["session_id"] == fake.session_service.created[0][2]


def test_adk_runner_fresh_session_per_call():
    fake = _FakeRunner()
    run = adk_runner(fake, message_factory=lambda s: s)
    run("a")
    run("b")
    ids = [session_id for _, _, session_id in fake.session_service.created]
    assert len(ids) == 2 and ids[0] != ids[1]
    assert [c["session_id"] for c in fake.calls] == ids


def test_adk_runner_awaits_async_session_creation():
    fake = _FakeRunner(asynchronous_sessions=True)
    run = adk_runner(fake, message_factory=lambda s: s)
    assert run("hi") == "final: hi"
    assert len(fake.session_service.created) == 1


def test_adk_runner_custom_user_id():
    fake = _FakeRunner()
    run = adk_runner(fake, user_id="tester", message_factory=lambda s: s)
    run("x")
    assert fake.calls[0]["user_id"] == "tester"
    assert fake.session_service.created[0][1] == "tester"


def test_adk_runner_content_shaped_input_passes_through():
    fake = _FakeRunner()
    run = adk_runner(fake)  # default message factory, no google-genai needed here
    message = _Content([_Part(text="prepacked")], role="user")
    run(message)
    assert fake.calls[0]["message"] is message


def test_adk_runner_string_without_genai_raises_helpful_error():
    try:
        import google.genai  # noqa: F401

        pytest.skip("google-genai installed; the lazy-import error path is not reachable")
    except ImportError:
        pass
    fake = _FakeRunner()
    run = adk_runner(fake)
    with pytest.raises(ImportError, match="message_factory"):
        run("needs packing")


def test_adk_runner_output_extractor_none_returns_events():
    fake = _FakeRunner()
    run = adk_runner(fake, message_factory=lambda s: s, output_extractor=None)
    events = run("x")
    assert isinstance(events, list) and len(events) == 2


def test_adk_runner_bare_agent_without_sdk_raises_helpful_error():
    try:
        import google.adk  # noqa: F401

        pytest.skip("google-adk installed; the lazy-import error path is not reachable")
    except ImportError:
        pass

    class BareAgent:
        instruction = "hi"
        sub_agents = []

    with pytest.raises(ImportError, match="google-adk"):
        adk_runner(BareAgent())
