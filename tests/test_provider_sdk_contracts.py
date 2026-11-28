"""Contract tests: providers against the *real* installed SDK classes, offline.

The provider unit tests use permissive fakes (``def create(**kwargs)``), which
accept any keyword and so cannot notice that an SDK release removed one. That
is exactly how an ``anthropic`` upgrade without ``temperature`` turned every
judged row into a silent zero. These tests build the real SDK client on an
in-process mock transport and check two things for each provider whose SDK is
importable:

* every keyword the provider sends binds to the SDK method's signature (and the
  request then goes through the SDK unchanged);
* every response attribute the provider reads exists on the SDK's response type.

No network is used: the transport answers in-process. Each SDK is
``importorskip``-ed, so a dev environment without it stays green.
"""

from __future__ import annotations

import importlib
import inspect
import json
from typing import Any

import pytest

from adapt_agent.optimization.providers import (
    ANTHROPIC_DEFAULT_MODEL,
    GEMINI_DEFAULT_MODEL,
    OPENAI_DEFAULT_MODEL,
    AnthropicProvider,
    GeminiProvider,
    GroqProvider,
    OpenAIProvider,
    RefusedResponseError,
    TruncatedResponseError,
    _model_rejects_sampling,
)


def _http_module(sdk: Any) -> Any:
    """The HTTP library (``httpx`` or ``httpx2``) this SDK version is built on."""
    base = next(cls for cls in sdk.DefaultHttpxClient.__mro__[1:] if cls.__name__ == "Client")
    return importlib.import_module(base.__module__.split(".")[0])


def _bind(fn: Any, kwargs: dict[str, Any]) -> None:
    """Fail unless ``kwargs`` binds to ``fn``'s real signature."""
    inspect.signature(fn).bind(**kwargs)


# -- anthropic ---------------------------------------------------------------------

#: Every current Anthropic model family, and older ones still served.
ANTHROPIC_MODELS = [
    ANTHROPIC_DEFAULT_MODEL,
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "claude-fable-5-2",
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
]


def _anthropic_client(sdk: Any, body: dict[str, Any], seen: list[dict[str, Any]]) -> Any:
    http = _http_module(sdk)

    def handler(request: Any) -> Any:
        sent = json.loads(request.content)
        seen.append(sent)
        message = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": sent["model"],
            "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 7},
            **body,
        }
        return http.Response(200, json=message)

    return sdk.Anthropic(
        api_key="test",
        max_retries=0,
        http_client=http.Client(transport=http.MockTransport(handler)),
    )


@pytest.mark.parametrize("model", ANTHROPIC_MODELS)
@pytest.mark.parametrize("structured", [False, True])
def test_anthropic_request_binds_to_real_sdk(model: str, structured: bool) -> None:
    sdk = pytest.importorskip("anthropic")
    seen: list[dict[str, Any]] = []
    body = {"content": [{"type": "text", "text": '{"score": 9}'}], "stop_reason": "end_turn"}
    client = _anthropic_client(sdk, body, seen)
    provider = AnthropicProvider(model, client=client, temperature=0.3)

    create, kwargs = provider._request("hi", {"system": "rubric"})
    _bind(create, kwargs)
    if structured:
        out = provider.complete_json("hi", {"type": "object"}, system="rubric")
        assert seen[-1]["output_config"]["format"]["type"] == "json_schema"
    else:
        out = provider.complete("hi", system="rubric")
    assert out == '{"score": 9}'
    assert len(seen) == 1  # one HTTP call, no local TypeError and no retry
    accepted = inspect.signature(client.messages.create).parameters
    # A sampling key goes out only if this SDK takes it and the model allows it.
    for key in ("temperature", "top_p", "top_k"):
        expected = key == "temperature" and key in accepted and not _model_rejects_sampling(model)
        assert (key in seen[0]) == expected, key
    assert provider.last_usage is not None
    assert (provider.last_usage.input_tokens, provider.last_usage.output_tokens) == (11, 7)


def test_anthropic_response_attributes_exist_on_sdk_types() -> None:
    pytest.importorskip("anthropic")
    from anthropic.types import Message, TextBlock, Usage

    for field in ("content", "stop_reason", "stop_details", "usage"):
        assert field in Message.model_fields, field
    assert "text" in TextBlock.model_fields and "type" in TextBlock.model_fields
    assert {"input_tokens", "output_tokens"} <= set(Usage.model_fields)
    stop_reasons = str(Message.model_fields["stop_reason"].annotation)
    for reason in ("max_tokens", "refusal", "end_turn"):
        assert reason in stop_reasons


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (
            {
                "content": [{"type": "thinking", "thinking": "", "signature": "s"}],
                "stop_reason": "max_tokens",
            },
            TruncatedResponseError,
        ),
        (
            {
                "content": [],
                "stop_reason": "refusal",
                "stop_details": {"type": "refusal", "category": "cyber", "explanation": "x"},
            },
            RefusedResponseError,
        ),
    ],
)
def test_anthropic_stop_reasons_through_real_sdk(body: dict[str, Any], error: type) -> None:
    sdk = pytest.importorskip("anthropic")
    provider = AnthropicProvider(client=_anthropic_client(sdk, body, []))
    with pytest.raises(error):
        provider.complete("hi")


def test_anthropic_thinking_and_text_blocks_join_only_text() -> None:
    sdk = pytest.importorskip("anthropic")
    body = {
        "content": [
            {"type": "thinking", "thinking": "secret", "signature": "s"},
            {"type": "text", "text": "answer"},
        ],
        "stop_reason": "end_turn",
    }
    assert AnthropicProvider(client=_anthropic_client(sdk, body, [])).complete("q") == "answer"


def test_anthropic_client_built_without_sdk_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert AnthropicProvider()._get_client().max_retries == 0
    injected = sdk.Anthropic(api_key="test")  # SDK default: 2
    assert AnthropicProvider(client=injected)._get_client().max_retries == 0
    assert AnthropicProvider(client=injected, sdk_max_retries=None)._get_client() is injected


# -- openai ------------------------------------------------------------------------


def _openai_client(sdk: Any, message: dict[str, Any], finish: str, seen: list[Any]) -> Any:
    http = _http_module(sdk)

    def handler(request: Any) -> Any:
        sent = json.loads(request.content)
        seen.append(sent)
        return http.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": sent["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish,
                        "message": {"role": "assistant", **message},
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    return sdk.OpenAI(
        api_key="test",
        max_retries=0,
        http_client=http.Client(transport=http.MockTransport(handler)),
    )


@pytest.mark.parametrize("cls", [OpenAIProvider, GroqProvider])
@pytest.mark.parametrize("structured", [False, True])
def test_openai_request_binds_to_real_sdk(cls: type, structured: bool) -> None:
    sdk = pytest.importorskip("openai")
    seen: list[Any] = []
    client = _openai_client(sdk, {"content": "hello"}, "stop", seen)
    provider = cls(client=client)
    create, kwargs = provider._request("hi", {"system": "rubric"})
    _bind(create, kwargs)
    if structured and cls is OpenAIProvider:
        assert provider.complete_json("hi", {"type": "object"}) == "hello"
        assert seen[-1]["response_format"]["type"] == "json_schema"
    else:
        assert provider.complete("hi", system="rubric") == "hello"
    assert len(seen) == 1
    accepted = inspect.signature(client.chat.completions.create).parameters
    if cls is OpenAIProvider and "max_completion_tokens" in accepted:
        assert "max_completion_tokens" in seen[0] and "max_tokens" not in seen[0]
    else:
        assert "max_tokens" in seen[0]
    assert provider.last_usage is not None and provider.last_usage.output_tokens == 2


@pytest.mark.parametrize(
    ("message", "finish", "error"),
    [
        ({"content": None}, "length", TruncatedResponseError),
        ({"content": None, "refusal": "I can't help with that"}, "stop", RefusedResponseError),
        ({"content": None}, "content_filter", RefusedResponseError),
    ],
)
def test_openai_finish_reasons_through_real_sdk(
    message: dict[str, Any], finish: str, error: type
) -> None:
    sdk = pytest.importorskip("openai")
    provider = OpenAIProvider(client=_openai_client(sdk, message, finish, []))
    with pytest.raises(error):
        provider.complete("hi")


def test_openai_response_attributes_exist_on_sdk_types() -> None:
    pytest.importorskip("openai")
    from openai.types import CompletionUsage
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice

    assert {"choices", "usage"} <= set(ChatCompletion.model_fields)
    assert {"finish_reason", "message"} <= set(Choice.model_fields)
    assert {"content", "refusal"} <= set(ChatCompletionMessage.model_fields)
    assert {"prompt_tokens", "completion_tokens"} <= set(CompletionUsage.model_fields)
    finish = str(Choice.model_fields["finish_reason"].annotation)
    assert "length" in finish and "content_filter" in finish


def test_openai_client_built_without_sdk_retries() -> None:
    sdk = pytest.importorskip("openai")
    assert OpenAIProvider(api_key="test")._get_client().max_retries == 0
    assert OpenAIProvider(client=sdk.OpenAI(api_key="test"))._get_client().max_retries == 0


def test_default_openai_model_is_known_to_the_sdk() -> None:
    pytest.importorskip("openai")
    chat_model = pytest.importorskip("openai.types.shared.chat_model")
    if OPENAI_DEFAULT_MODEL not in str(chat_model.ChatModel):
        pytest.skip("installed openai SDK predates the default model")
    assert OPENAI_DEFAULT_MODEL in str(chat_model.ChatModel)


# -- google-genai ------------------------------------------------------------------


def _gemini_fake(types: Any, response: Any) -> Any:
    """A client whose ``generate_content`` validates config like the real SDK."""

    class Models:
        last: dict[str, Any] = {}

        def generate_content(self, *, model: str, contents: Any, config: Any = None) -> Any:
            types.GenerateContentConfig.model_validate(config)
            Models.last = {"model": model, "contents": contents, "config": config}
            return response

    class Client:
        models = Models()

    return Client()


@pytest.mark.parametrize("structured", [False, True])
def test_gemini_request_binds_to_real_sdk(structured: bool) -> None:
    genai = pytest.importorskip("google.genai")
    from google.genai import types

    real = genai.Client(api_key="test").models.generate_content
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                finish_reason="STOP",
                content=types.Content(role="model", parts=[types.Part(text="ok")]),
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=4, candidates_token_count=1
        ),
    )
    client = _gemini_fake(types, response)
    provider = GeminiProvider(client=client)
    if structured:
        assert provider.complete_json("data", {"type": "object"}, system="rubric") == "ok"
    else:
        assert provider.complete("data", system="rubric") == "ok"
    sent = client.models.last
    _bind(real, {"model": sent["model"], "contents": sent["contents"], "config": sent["config"]})
    assert sent["model"] == GEMINI_DEFAULT_MODEL
    assert sent["contents"] == "data"  # rubric NOT concatenated into the data
    assert sent["config"]["system_instruction"] == "rubric"
    assert provider.last_usage is not None and provider.last_usage.input_tokens == 4


def test_gemini_max_tokens_is_truncation() -> None:
    pytest.importorskip("google.genai")
    from google.genai import types

    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                finish_reason="MAX_TOKENS", content=types.Content(role="model", parts=[])
            )
        ]
    )
    with pytest.raises(TruncatedResponseError):
        GeminiProvider(client=_gemini_fake(types, response)).complete("q")


def test_gemini_response_attributes_exist_on_sdk_types() -> None:
    pytest.importorskip("google.genai")
    from google.genai import types

    assert {"candidates", "prompt_feedback", "usage_metadata"} <= set(
        types.GenerateContentResponse.model_fields
    )
    assert "finish_reason" in types.Candidate.model_fields
    assert "block_reason" in types.GenerateContentResponsePromptFeedback.model_fields
    usage = set(types.GenerateContentResponseUsageMetadata.model_fields)
    assert {"prompt_token_count", "candidates_token_count", "thoughts_token_count"} <= usage
    assert {"MAX_TOKENS", "SAFETY"} <= set(types.FinishReason.__members__)
    config = set(types.GenerateContentConfig.model_fields)
    assert {"system_instruction", "response_mime_type", "response_json_schema"} <= config
