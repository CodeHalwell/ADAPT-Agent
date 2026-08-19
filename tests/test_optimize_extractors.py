"""Offline tests for ``adapt_agent.optimization.extractors``.

Each framework's run-result shape is mimicked with small stub classes (the same
duck-typing the extractor relies on), so no agent framework is imported.
"""

from types import SimpleNamespace

from adapt_agent.optimization.extractors import (
    available_extractors,
    extract_output_text,
    register_extractor,
)

# -- scalars and passthrough ---------------------------------------------------


def test_string_passthrough():
    assert extract_output_text("Paris") == "Paris"


def test_none_becomes_empty_string():
    assert extract_output_text(None) == ""


def test_bytes_decoded():
    assert extract_output_text(b"Paris") == "Paris"


def test_number_passthrough():
    assert extract_output_text(4) == 4
    assert extract_output_text(3.14) == 3.14


def test_unrecognised_object_unchanged():
    class Opaque:
        __slots__ = ()

    obj = Opaque()
    assert extract_output_text(obj) is obj


def test_unrecognised_mapping_unchanged():
    payload = {"foo": 1, "bar": 2}
    assert extract_output_text(payload) is payload


def test_structured_list_unchanged():
    payload = [{"a": 1}, {"b": 2}]
    assert extract_output_text(payload) is payload


def test_empty_stream_is_empty_text():
    assert extract_output_text([]) == ""


def test_cyclic_structure_terminates():
    payload: dict = {}
    payload["output"] = payload  # self-referential
    # Must terminate (bounded depth) and hand something back without raising.
    extract_output_text(payload)


# -- Pydantic AI ----------------------------------------------------------------


class _AgentRunResult:
    """Shape of ``pydantic_ai.agent.AgentRunResult``."""

    def __init__(self, output):
        self.output = output

    def all_messages(self):  # pragma: no cover - presence is what matters
        return []


class _LegacyRunResult:
    """Older Pydantic AI results exposed ``.data`` instead of ``.output``."""

    def __init__(self, data):
        self.data = data

    def new_messages(self):  # pragma: no cover - presence is what matters
        return []


def test_pydantic_ai_output():
    assert extract_output_text(_AgentRunResult("Paris")) == "Paris"


def test_pydantic_ai_legacy_data():
    assert extract_output_text(_LegacyRunResult("Tokyo")) == "Tokyo"


def test_pydantic_ai_structured_output_unchanged():
    structured = SimpleNamespace(city="Paris", population=2_100_000)
    result = extract_output_text(_AgentRunResult(structured))
    # The wrapper is removed; the structured payload itself is not text and has
    # no conventional output attribute, so it survives unchanged.
    assert result is structured


# -- OpenAI Agents SDK -----------------------------------------------------------


class _RunResult:
    """Shape of ``agents.result.RunResult``."""

    def __init__(self, final_output):
        self.final_output = final_output
        self.new_items = []


def test_openai_agents_final_output():
    assert extract_output_text(_RunResult("Rome")) == "Rome"


def test_openai_agents_numeric_final_output():
    assert extract_output_text(_RunResult(42)) == 42


# -- CrewAI ----------------------------------------------------------------------


class _CrewOutput:
    def __init__(self, raw):
        self.raw = raw
        self.tasks_output = []


def test_crewai_raw():
    assert extract_output_text(_CrewOutput("Cairo")) == "Cairo"


# -- Claude Agent SDK --------------------------------------------------------------


class _ResultMessage:
    def __init__(self, result):
        self.subtype = "success"
        self.result = result


class _TextBlock:
    def __init__(self, text):
        self.text = text


class _AssistantMessage:
    def __init__(self, *blocks):
        self.content = list(blocks)


def test_claude_result_message():
    assert extract_output_text(_ResultMessage("Berlin")) == "Berlin"


def test_claude_drained_message_stream():
    stream = [
        SimpleNamespace(subtype="init", data={}),
        _AssistantMessage(_TextBlock("Madrid")),
        _ResultMessage("Madrid"),
    ]
    assert extract_output_text(stream) == "Madrid"


def test_claude_assistant_message_blocks():
    message = _AssistantMessage(_TextBlock("Lisbon"))
    assert extract_output_text(message) == "Lisbon"


# -- Microsoft Agent Framework -----------------------------------------------------


class _ChatMessage:
    def __init__(self, role, text):
        self.role = role
        self.text = text
        self.contents = [SimpleNamespace(text=text)]


class _AgentRunResponse:
    """Shape of ``agent_framework.AgentRunResponse``: messages + text property."""

    def __init__(self, messages, text=None):
        self.messages = messages
        self._text = text

    @property
    def text(self):
        if self._text is not None:
            return self._text
        return "".join(m.text for m in self.messages)


def test_maf_response_text():
    response = _AgentRunResponse([_ChatMessage("assistant", "Oslo")])
    assert extract_output_text(response) == "Oslo"


def test_maf_response_empty_text_falls_back_to_messages():
    response = _AgentRunResponse([_ChatMessage("assistant", "Helsinki")], text="")
    assert extract_output_text(response) == "Helsinki"


def test_maf_single_chat_message():
    assert extract_output_text(_ChatMessage("assistant", "Bern")) == "Bern"


# -- LangGraph / LangChain ----------------------------------------------------------


class _LCMessage:
    """Shape of a LangChain ``BaseMessage`` (content + type)."""

    def __init__(self, type_, content):
        self.type = type_
        self.content = content


def test_langgraph_state_last_message():
    state = {
        "messages": [
            _LCMessage("human", "What is 2+2?"),
            _LCMessage("ai", "4"),
        ]
    }
    assert extract_output_text(state) == "4"


def test_langgraph_state_with_output_channel_wins():
    state = {"messages": [_LCMessage("ai", "ignored")], "output": "42"}
    assert extract_output_text(state) == "42"


def test_langchain_message_content_blocks():
    message = _LCMessage("ai", [{"type": "text", "text": "Vienna"}])
    assert extract_output_text(message) == "Vienna"


def test_langchain_multi_block_content_joined():
    message = _LCMessage("ai", [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert extract_output_text(message) == "a\nb"


def test_openai_style_message_dicts():
    state = {
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    }
    assert extract_output_text(state) == "hello"


# -- Google ADK / GenAI ----------------------------------------------------------------


class _Part:
    def __init__(self, text=None, function_call=None):
        self.text = text
        self.function_call = function_call


class _Content:
    def __init__(self, parts, role="model"):
        self.parts = parts
        self.role = role


class _Event:
    def __init__(self, content, author="agent"):
        self.content = content
        self.author = author

    def is_final_response(self):  # pragma: no cover - presence mirrors ADK
        return True


def test_adk_single_event():
    event = _Event(_Content([_Part(text="Ottawa")]))
    assert extract_output_text(event) == "Ottawa"


def test_adk_event_stream_takes_last_text_event():
    events = [
        _Event(_Content([_Part(function_call=object())])),  # tool call, no text
        _Event(_Content([_Part(text="intermediate")])),
        _Event(_Content([_Part(text="Canberra")])),
    ]
    assert extract_output_text(events) == "Canberra"


def test_adk_multi_part_content_joined():
    event = _Event(_Content([_Part(text="part one"), _Part(text="part two")]))
    assert extract_output_text(event) == "part one\npart two"


def test_genai_content_direct():
    assert extract_output_text(_Content([_Part(text="Nairobi")])) == "Nairobi"


# -- generic mappings and attributes -----------------------------------------------------


def test_mapping_output_key():
    assert extract_output_text({"output": "Quito"}) == "Quito"


def test_mapping_skips_empty_values():
    assert extract_output_text({"output": "", "text": "Lima"}) == "Lima"


def test_mapping_nested_unwrap():
    assert extract_output_text({"response": {"content": "Accra"}}) == "Accra"


def test_object_output_text_attr():
    assert extract_output_text(SimpleNamespace(output_text="Dakar")) == "Dakar"


def test_object_callable_attrs_skipped():
    class WithCallableResult:
        def result(self):  # a future-like API, not a value
            return "not this"

        text = "Hanoi"

    assert extract_output_text(WithCallableResult()) == "Hanoi"


def test_tuple_stream():
    assert extract_output_text(("meta", {"content": "Seoul"})) == "Seoul"


# -- registry -------------------------------------------------------------------


def test_register_extractor_priority_and_replace():
    class Wrapped:
        def __init__(self, inner):
            self.inner = inner
            self.text = "should not win"

    try:
        register_extractor("wrapped", lambda v: isinstance(v, Wrapped), lambda v: v.inner)
        assert "wrapped" in available_extractors()
        # The custom entry outranks the generic attr fallback ("text").
        assert extract_output_text(Wrapped("Custom")) == "Custom"

        # Re-registering the same name replaces the previous unwrap.
        register_extractor("wrapped", lambda v: isinstance(v, Wrapped), lambda v: v.inner * 2)
        assert extract_output_text(Wrapped("x")) == "xx"

        # Declining (returning None) falls through to the built-ins.
        register_extractor("wrapped", lambda v: isinstance(v, Wrapped), lambda v: None)
        assert extract_output_text(Wrapped("y")) == "should not win"
    finally:
        from adapt_agent.optimization import extractors as _mod

        _mod._CUSTOM_EXTRACTORS = [e for e in _mod._CUSTOM_EXTRACTORS if e[0] != "wrapped"]


def test_raising_predicate_is_skipped():
    def bad_predicate(value):
        raise RuntimeError("boom")

    try:
        register_extractor("bad", bad_predicate, lambda v: "never")
        assert extract_output_text({"output": "safe"}) == "safe"
    finally:
        from adapt_agent.optimization import extractors as _mod

        _mod._CUSTOM_EXTRACTORS = [e for e in _mod._CUSTOM_EXTRACTORS if e[0] != "bad"]
