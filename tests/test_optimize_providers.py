"""Offline, deterministic tests for adapt_agent.optimization.providers.

Vendor providers are exercised with INJECTED fake clients that mimic the SDK
shape, so no real SDK is imported and no network call is made. _get_client() is
never invoked without an injected client.
"""

import pytest

from adapt_agent.optimization import providers as P
from adapt_agent.optimization.providers import (
    AnthropicProvider,
    AzureOpenAIProvider,
    BedrockProvider,
    CallableProvider,
    CohereProvider,
    EchoProvider,
    GeminiProvider,
    GroqProvider,
    HuggingFaceProvider,
    MistralProvider,
    ModelProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    TogetherProvider,
    _join_text_blocks,
    as_provider,
    available_providers,
    get_provider,
    register_provider,
)

# -- ModelProvider base --------------------------------------------------------


class _Concrete(ModelProvider):
    name = "concrete"

    def complete(self, prompt, **overrides):
        cfg = self._resolved(overrides)
        return f"{prompt}|{cfg['temperature']}|{cfg['max_tokens']}|{cfg['system']}"


def test_model_provider_is_callable():
    p = _Concrete("m", temperature=0.5, max_tokens=10, system="sys")
    assert p("hi") == "hi|0.5|10|sys"


def test_resolved_defaults():
    p = _Concrete("m", temperature=0.2, max_tokens=7, system="base")
    assert p._resolved({}) == {"temperature": 0.2, "max_tokens": 7, "system": "base"}


def test_resolved_overrides():
    p = _Concrete("m", temperature=0.2, max_tokens=7, system="base")
    resolved = p._resolved({"temperature": 0.9, "max_tokens": 99, "system": "override"})
    assert resolved == {"temperature": 0.9, "max_tokens": 99, "system": "override"}


def test_complete_with_overrides_via_call():
    p = _Concrete("m")
    assert p("hi", temperature=1.0) == "hi|1.0|1024|None"


# -- CallableProvider ----------------------------------------------------------


def test_callable_provider_passthrough():
    p = CallableProvider(lambda prompt: f"echo:{prompt}")
    assert p("hello") == "echo:hello"


def test_callable_provider_accepts_kwargs():
    def fn(prompt, **kw):
        return f"{prompt}:{kw.get('temperature')}"

    p = CallableProvider(fn)
    assert p("hi", temperature=0.7) == "hi:0.7"


def test_callable_provider_typeerror_fallback():
    # fn rejects kwargs -> TypeError -> retried plainly.
    calls = []

    def fn(prompt):
        calls.append(prompt)
        return "ok"

    p = CallableProvider(fn)
    assert p("hi", temperature=0.5) == "ok"
    assert calls == ["hi"]


def test_callable_provider_dropping_system_logs_a_warning(caplog):
    # A callable that cannot accept `system` is still supported, but silently
    # dropping the grading rubric is the worst failure mode for a judge -- it
    # keeps returning confident-looking scores while grading blind. The drop
    # must be logged, not swallowed.
    def fn(prompt):
        return "ok"

    p = CallableProvider(fn)
    with caplog.at_level("WARNING"):
        result = p("hi", system="grade strictly against this rubric")
    assert result == "ok"
    assert any("system" in r.message for r in caplog.records)


def test_callable_provider_dropping_non_system_override_is_silent(caplog):
    # Only `system` carries information a judge cannot function without;
    # a callable ignoring e.g. `temperature` is an ordinary, unremarkable stub.
    def fn(prompt):
        return "ok"

    p = CallableProvider(fn)
    with caplog.at_level("WARNING"):
        result = p("hi", temperature=0.7)
    assert result == "ok"
    assert caplog.records == []


def test_callable_provider_accepting_system_never_warns(caplog):
    def fn(prompt, **kw):
        return f"{prompt}:{kw.get('system')}"

    p = CallableProvider(fn)
    with caplog.at_level("WARNING"):
        result = p("hi", system="rubric")
    assert result == "hi:rubric"
    assert caplog.records == []


def test_callable_provider_coerces_non_str_result():
    p = CallableProvider(lambda prompt: 12345)
    assert p("x") == "12345"


def test_callable_provider_rejects_non_callable():
    with pytest.raises(TypeError):
        CallableProvider(42)


def test_callable_provider_default_model():
    p = CallableProvider(lambda prompt: "x")
    assert p.model == "callable"


# -- EchoProvider --------------------------------------------------------------


def test_echo_provider_default_template():
    p = EchoProvider()
    assert p("hello") == "hello"


def test_echo_provider_custom_template_with_placeholder():
    p = EchoProvider(template="<<{prompt}>>")
    assert p("x") == "<<x>>"


def test_echo_provider_literal_brace_template_passthrough():
    # Template with literal braces but no {prompt} placeholder -> returned as-is.
    p = EchoProvider(template='{"score": 8}')
    assert p("anything") == '{"score": 8}'


def test_echo_provider_no_placeholder_returns_template():
    p = EchoProvider(template="constant")
    assert p("ignored") == "constant"


# -- as_provider ---------------------------------------------------------------


def test_as_provider_passthrough_model_provider():
    p = EchoProvider()
    assert as_provider(p) is p


def test_as_provider_from_string_name():
    p = as_provider("echo")
    assert isinstance(p, EchoProvider)


def test_as_provider_from_callable():
    p = as_provider(lambda prompt: "x")
    assert isinstance(p, CallableProvider)
    assert p("y") == "x"


def test_as_provider_invalid_raises_typeerror():
    with pytest.raises(TypeError):
        as_provider(123)


# -- registry: get_provider / register_provider / available_providers ----------


def test_get_provider_known():
    assert isinstance(get_provider("echo"), EchoProvider)


def test_get_provider_case_insensitive():
    assert isinstance(get_provider("ECHO"), EchoProvider)


def test_get_provider_forwards_kwargs():
    p = get_provider("echo", template="t-{prompt}")
    assert p("x") == "t-x"


def test_get_provider_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_provider("nope_not_real")


def test_available_providers_sorted_and_complete():
    names = available_providers()
    assert names == sorted(names)
    for expected in ("anthropic", "openai", "echo", "callable", "gemini", "bedrock"):
        assert expected in names


def test_register_provider_accepts_subclass():
    class CustomProvider(ModelProvider):
        name = "custom_test"

        def __init__(self, model="custom-default", **kw):
            super().__init__(model, **kw)

        def complete(self, prompt, **overrides):
            return "custom"

    register_provider("custom_test_key", CustomProvider)
    try:
        assert "custom_test_key" in available_providers()
        assert isinstance(get_provider("custom_test_key"), CustomProvider)
    finally:
        P._PROVIDERS.pop("custom_test_key", None)


def test_register_provider_lowercases_name():
    class CustomProvider(ModelProvider):
        def __init__(self, model="m", **kw):
            super().__init__(model, **kw)

        def complete(self, prompt, **overrides):
            return "x"

    register_provider("MixedCase", CustomProvider)
    try:
        assert "mixedcase" in P._PROVIDERS
    finally:
        P._PROVIDERS.pop("mixedcase", None)


def test_register_provider_rejects_non_subclass():
    class NotAProvider:
        pass

    with pytest.raises(TypeError):
        register_provider("bad", NotAProvider)


def test_register_provider_rejects_non_type():
    with pytest.raises(TypeError):
        register_provider("bad", lambda: None)


# -- _join_text_blocks ---------------------------------------------------------


def test_join_text_blocks_str():
    assert _join_text_blocks("hello") == "hello"


def test_join_text_blocks_list_of_objects():
    class Block:
        def __init__(self, text):
            self.text = text

    assert _join_text_blocks([Block("a"), Block("b")]) == "ab"


def test_join_text_blocks_list_of_dicts():
    assert _join_text_blocks([{"text": "x"}, {"text": "y"}]) == "xy"


def test_join_text_blocks_mixed_and_skips_non_text():
    class Block:
        def __init__(self, text):
            self.text = text

    blocks = [Block("a"), {"text": "b"}, {"no_text": 1}, object()]
    assert _join_text_blocks(blocks) == "ab"


def test_join_text_blocks_fallback_str():
    assert _join_text_blocks(42) == "42"


# -- Vendor providers with INJECTED fake clients -------------------------------


class FakeAnthropicMessage:
    class _Block:
        def __init__(self, text):
            self.text = text

    def __init__(self, text):
        self.content = [self._Block(text)]


class FakeAnthropicMessages:
    def __init__(self, text):
        self._text = text
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeAnthropicMessage(self._text)


class FakeAnthropicClient:
    def __init__(self, text="claude-says-hi"):
        self.messages = FakeAnthropicMessages(text)


def test_anthropic_provider_with_injected_client():
    client = FakeAnthropicClient("hello from claude")
    p = AnthropicProvider(client=client)
    assert p.complete("prompt") == "hello from claude"
    kw = client.messages.last_kwargs
    assert kw["model"] == "claude-opus-4-8"
    assert kw["messages"] == [{"role": "user", "content": "prompt"}]
    assert "system" not in kw  # no system by default


def test_anthropic_provider_includes_system():
    client = FakeAnthropicClient("x")
    p = AnthropicProvider(client=client, system="be brief")
    p.complete("prompt")
    assert client.messages.last_kwargs["system"] == "be brief"


def test_anthropic_provider_default_model():
    p = AnthropicProvider(client=FakeAnthropicClient())
    assert p.model == "claude-opus-4-8"


def test_anthropic_opus_4_8_omits_sampling_params():
    # The default model (opus-4-8) rejects user-set sampling -> omit entirely.
    client = FakeAnthropicClient("x")
    p = AnthropicProvider(client=client, temperature=0.7)
    p.complete("prompt", top_p=0.5)
    kw = client.messages.last_kwargs
    assert "temperature" not in kw
    assert "top_p" not in kw
    assert "top_k" not in kw


def test_anthropic_opus_4_7_omits_sampling_params():
    client = FakeAnthropicClient("x")
    p = AnthropicProvider(client=client, model="claude-opus-4-7", temperature=0.9)
    p.complete("prompt")
    assert "temperature" not in client.messages.last_kwargs


def test_anthropic_fable_5_omits_sampling_params():
    client = FakeAnthropicClient("x")
    p = AnthropicProvider(client=client, model="claude-fable-5", temperature=1.0)
    p.complete("prompt")
    assert "temperature" not in client.messages.last_kwargs


def test_anthropic_legacy_model_sends_clamped_temperature():
    # A model NOT in the no-sampling family still gets a (clamped) temperature.
    client = FakeAnthropicClient("x")
    p = AnthropicProvider(client=client, model="claude-3-5-sonnet-20241022", temperature=0.5)
    p.complete("prompt")
    assert client.messages.last_kwargs["temperature"] == 0.5


def test_anthropic_over_range_temperature_clamped_and_warns(caplog):
    # Anthropic max_temperature is 1.0; a request of 1.7 must clamp to 1.0 + warn.
    client = FakeAnthropicClient("x")
    p = AnthropicProvider(client=client, model="claude-3-5-sonnet-20241022")
    with caplog.at_level("WARNING", logger="adapt_agent.optimization.providers"):
        p.complete("prompt", temperature=1.7)
    assert client.messages.last_kwargs["temperature"] == 1.0
    assert any("exceeds max" in rec.message for rec in caplog.records)


def test_openai_over_range_temperature_clamped():
    # OpenAI max_temperature is 2.0; 3.5 clamps to 2.0.
    client = FakeOpenAIClient("x")
    p = OpenAIProvider(client=client)
    p.complete("prompt", temperature=3.5)
    assert client.chat.completions.last_kwargs["temperature"] == 2.0


def test_openai_top_p_clamped_into_unit_interval():
    client = FakeOpenAIClient("x")
    p = OpenAIProvider(client=client)
    p.complete("prompt", top_p=1.9)
    assert client.chat.completions.last_kwargs["top_p"] == 1.0


def test_negative_temperature_clamped_to_zero():
    client = FakeOpenAIClient("x")
    p = OpenAIProvider(client=client)
    p.complete("prompt", temperature=-0.5)
    assert client.chat.completions.last_kwargs["temperature"] == 0.0


# OpenAI-style fakes ----------------------------------------------------------


class FakeOpenAIMessage:
    def __init__(self, content):
        self.content = content


class FakeOpenAIChoice:
    def __init__(self, content):
        self.message = FakeOpenAIMessage(content)


class FakeOpenAIResponse:
    def __init__(self, content):
        self.choices = [FakeOpenAIChoice(content)]


class FakeChatCompletions:
    def __init__(self, content):
        self._content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeOpenAIResponse(self._content)


class FakeChat:
    def __init__(self, content):
        self.completions = FakeChatCompletions(content)


class FakeOpenAIClient:
    def __init__(self, content="gpt-says-hi"):
        self.chat = FakeChat(content)


def test_openai_provider_with_injected_client():
    client = FakeOpenAIClient("hello gpt")
    p = OpenAIProvider(client=client)
    assert p.complete("prompt") == "hello gpt"
    kw = client.chat.completions.last_kwargs
    assert kw["model"] == "gpt-4o-mini"
    assert kw["messages"] == [{"role": "user", "content": "prompt"}]


def test_openai_provider_with_system():
    client = FakeOpenAIClient("x")
    p = OpenAIProvider(client=client, system="sys")
    p.complete("prompt")
    msgs = client.chat.completions.last_kwargs["messages"]
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1] == {"role": "user", "content": "prompt"}


def test_openai_provider_none_content_returns_empty():
    client = FakeOpenAIClient(None)
    p = OpenAIProvider(client=client)
    assert p.complete("prompt") == ""


def test_openai_defaults():
    p = OpenAIProvider(client=FakeOpenAIClient())
    assert p.api_key_env == "OPENAI_API_KEY"
    assert p.default_base_url is None
    assert p._base_url is None


# OpenAI-compatible subclasses: base_url / api_key_env -------------------------


def test_groq_config():
    p = GroqProvider(client=FakeOpenAIClient())
    assert p.api_key_env == "GROQ_API_KEY"
    assert p.default_base_url == "https://api.groq.com/openai/v1"
    assert p._base_url == "https://api.groq.com/openai/v1"
    assert p.model == "llama-3.3-70b-versatile"
    assert p.complete("x") == "gpt-says-hi"


def test_together_config():
    p = TogetherProvider(client=FakeOpenAIClient())
    assert p.api_key_env == "TOGETHER_API_KEY"
    assert p.default_base_url == "https://api.together.xyz/v1"
    assert p._base_url == "https://api.together.xyz/v1"
    assert p.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"


def test_openrouter_config():
    p = OpenRouterProvider(client=FakeOpenAIClient())
    assert p.api_key_env == "OPENROUTER_API_KEY"
    assert p.default_base_url == "https://openrouter.ai/api/v1"
    assert p._base_url == "https://openrouter.ai/api/v1"
    assert p.model == "openai/gpt-4o-mini"


def test_ollama_config_and_placeholder_key():
    p = OllamaProvider(client=FakeOpenAIClient())
    assert p.api_key_env == "OLLAMA_API_KEY"
    assert p.default_base_url == "http://localhost:11434/v1"
    assert p._base_url == "http://localhost:11434/v1"
    assert p.model == "llama3.1"
    # Ollama injects a placeholder api key.
    assert p._api_key == "ollama"


def test_ollama_explicit_api_key():
    p = OllamaProvider(client=FakeOpenAIClient(), api_key="real-key")
    assert p._api_key == "real-key"


def test_azure_openai_config():
    p = AzureOpenAIProvider(
        client=FakeOpenAIClient("azure-hi"),
        azure_endpoint="https://x.openai.azure.com",
        api_version="2024-99-99",
    )
    assert p.api_key_env == "AZURE_OPENAI_API_KEY"
    assert p._azure_endpoint == "https://x.openai.azure.com"
    assert p._api_version == "2024-99-99"
    assert p.complete("p") == "azure-hi"


# Gemini ----------------------------------------------------------------------


class FakeGeminiResponse:
    def __init__(self, text):
        self.text = text


class FakeGeminiModels:
    def __init__(self, text):
        self._text = text
        self.last = None

    def generate_content(self, model, contents, config):
        self.last = {"model": model, "contents": contents, "config": config}
        return FakeGeminiResponse(self._text)


class FakeGeminiClient:
    def __init__(self, text="gemini-hi"):
        self.models = FakeGeminiModels(text)


def test_gemini_provider_with_injected_client():
    client = FakeGeminiClient("hello gemini")
    p = GeminiProvider(client=client)
    assert p.complete("prompt") == "hello gemini"
    assert client.models.last["model"] == "gemini-2.0-flash"
    assert client.models.last["contents"] == "prompt"
    assert client.models.last["config"]["temperature"] == 0.0


def test_gemini_provider_prepends_system():
    client = FakeGeminiClient("x")
    p = GeminiProvider(client=client, system="SYS")
    p.complete("prompt")
    assert client.models.last["contents"] == "SYS\n\nprompt"


def test_gemini_provider_none_text_returns_empty():
    client = FakeGeminiClient(None)
    p = GeminiProvider(client=client)
    assert p.complete("p") == ""


# Mistral ---------------------------------------------------------------------


class FakeMistralChat:
    def __init__(self, content):
        self._content = content
        self.last = None

    def complete(self, **kwargs):
        self.last = kwargs
        return FakeOpenAIResponse(self._content)


class FakeMistralClient:
    def __init__(self, content="mistral-hi"):
        self.chat = FakeMistralChat(content)


def test_mistral_provider_with_injected_client():
    client = FakeMistralClient("hello mistral")
    p = MistralProvider(client=client)
    assert p.complete("prompt") == "hello mistral"
    assert client.chat.last["model"] == "mistral-large-latest"


def test_mistral_provider_with_system():
    client = FakeMistralClient("x")
    p = MistralProvider(client=client, system="sys")
    p.complete("prompt")
    assert client.chat.last["messages"][0] == {"role": "system", "content": "sys"}


def test_mistral_none_content_returns_empty():
    client = FakeMistralClient(None)
    assert MistralProvider(client=client).complete("p") == ""


# Cohere ----------------------------------------------------------------------


class FakeCohereMessage:
    def __init__(self, content):
        self.content = content


class FakeCohereResponse:
    def __init__(self, content):
        self.message = FakeCohereMessage(content)


class FakeCohereClient:
    def __init__(self, content):
        self._content = content
        self.last = None

    def chat(self, **kwargs):
        self.last = kwargs
        return FakeCohereResponse(self._content)


def test_cohere_provider_with_injected_client_string_content():
    client = FakeCohereClient("hello cohere")
    p = CohereProvider(client=client)
    assert p.complete("prompt") == "hello cohere"
    assert client.last["model"] == "command-r-plus"


def test_cohere_provider_list_content_blocks():
    class Block:
        def __init__(self, text):
            self.text = text

    client = FakeCohereClient([Block("a"), Block("b")])
    p = CohereProvider(client=client)
    assert p.complete("prompt") == "ab"


def test_cohere_provider_with_system():
    client = FakeCohereClient("x")
    p = CohereProvider(client=client, system="sys")
    p.complete("p")
    assert client.last["messages"][0] == {"role": "system", "content": "sys"}


# Bedrock ---------------------------------------------------------------------


class FakeBedrockClient:
    def __init__(self, text="bedrock-hi"):
        self._text = text
        self.last = None

    def converse(self, **kwargs):
        self.last = kwargs
        return {"output": {"message": {"content": [{"text": self._text}]}}}


def test_bedrock_provider_with_injected_client():
    client = FakeBedrockClient("hello bedrock")
    p = BedrockProvider(client=client)
    assert p.complete("prompt") == "hello bedrock"
    assert client.last["modelId"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert client.last["messages"][0]["content"][0]["text"] == "prompt"
    assert "system" not in client.last


def test_bedrock_provider_with_system():
    client = FakeBedrockClient("x")
    p = BedrockProvider(client=client, system="sys")
    p.complete("prompt")
    assert client.last["system"] == [{"text": "sys"}]


def test_bedrock_provider_joins_multiple_blocks():
    class MultiBlockBedrock:
        def converse(self, **kwargs):
            return {"output": {"message": {"content": [{"text": "a"}, {"text": "b"}, {}]}}}

    p = BedrockProvider(client=MultiBlockBedrock())
    assert p.complete("p") == "ab"


# HuggingFace -----------------------------------------------------------------


class FakeHFClient:
    def __init__(self, content="hf-hi"):
        self._content = content
        self.last = None

    def chat_completion(self, messages, **kwargs):
        self.last = {"messages": messages, **kwargs}
        return FakeOpenAIResponse(self._content)


def test_huggingface_provider_with_injected_client():
    client = FakeHFClient("hello hf")
    p = HuggingFaceProvider(client=client)
    assert p.complete("prompt") == "hello hf"
    assert client.last["model"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert client.last["messages"][-1] == {"role": "user", "content": "prompt"}


def test_huggingface_provider_with_system():
    client = FakeHFClient("x")
    p = HuggingFaceProvider(client=client, system="sys")
    p.complete("prompt")
    assert client.last["messages"][0] == {"role": "system", "content": "sys"}


def test_huggingface_none_content_returns_empty():
    client = FakeHFClient(None)
    assert HuggingFaceProvider(client=client).complete("p") == ""


# -- Every vendor provider is registered and constructs offline ----------------


# -- Sampling-param retry fallback ---------------------------------------------


class _SamplingRejectMessages:
    """Fake messages API that 400s once on any sampling param, then succeeds."""

    def __init__(self, text="ok"):
        self._text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if any(k in kwargs for k in ("temperature", "top_p", "top_k")):
            raise ValueError(
                "Error code: 400 - invalid_request_error: temperature is not supported"
            )
        return FakeAnthropicMessage(self._text)


class _SamplingRejectAnthropicClient:
    def __init__(self, text="ok"):
        self.messages = _SamplingRejectMessages(text)


def test_anthropic_retries_without_sampling_on_400(caplog):
    # A model that DOES accept sampling locally, but whose SDK rejects it at
    # runtime -> the call retries once without temperature/top_p/top_k.
    client = _SamplingRejectAnthropicClient("recovered")
    p = AnthropicProvider(client=client, model="claude-3-5-sonnet-20241022", temperature=0.5)
    with caplog.at_level("WARNING", logger="adapt_agent.optimization.providers"):
        result = p.complete("prompt")
    assert result == "recovered"
    # Two attempts: first with temperature, second stripped.
    assert len(client.messages.calls) == 2
    assert "temperature" in client.messages.calls[0]
    assert "temperature" not in client.messages.calls[1]
    assert any("retrying without" in rec.message for rec in caplog.records)


def test_non_sampling_error_is_not_retried():
    # An unrelated error must propagate without a retry.
    class _Boom:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise RuntimeError("network down")

    class _Client:
        def __init__(self):
            self.messages = _Boom()

    client = _Client()
    p = AnthropicProvider(client=client, model="claude-3-5-sonnet-20241022")
    with pytest.raises(RuntimeError, match="network down"):
        p.complete("prompt")
    assert client.messages.calls == 1


def test_call_with_sampling_retry_helper_strips_only_sampling_keys():
    seen = []

    def fn(**kwargs):
        seen.append(kwargs)
        if "temperature" in kwargs:
            raise ValueError("400 invalid_request: temperature unsupported")
        return "done"

    result = P._call_with_sampling_retry(fn, {"model": "m", "temperature": 0.3, "max_tokens": 5})
    assert result == "done"
    assert seen[1] == {"model": "m", "max_tokens": 5}


def test_get_provider_constructs_all_vendor_classes():
    # Constructing (without calling complete) must not import any SDK. The
    # "callable" provider requires a ``fn`` argument, so supply one; every other
    # provider defines its own default model.
    for name in available_providers():
        if name == "callable":
            p = get_provider(name, fn=lambda prompt: "x")
        else:
            p = get_provider(name)
        assert isinstance(p, ModelProvider)
