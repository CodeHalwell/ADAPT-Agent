"""Offline, deterministic tests for adapt_agent.optimization.judges.

Provider-specific judges are driven through INJECTED fake clients so the vendor
SDK is never imported and no network call is made.
"""

import pytest

from adapt_agent.optimization import judges as J
from adapt_agent.optimization.judge import _DEFAULT_RUBRIC, JudgeVerdict, LLMJudge
from adapt_agent.optimization.judges import (
    JUDGE_REGISTRY,
    AnthropicJudge,
    AzureOpenAIJudge,
    BedrockJudge,
    ClaudeJudge,
    CohereJudge,
    GeminiJudge,
    GoogleJudge,
    GroqJudge,
    HuggingFaceJudge,
    MistralJudge,
    OllamaJudge,
    OpenAIJudge,
    OpenRouterJudge,
    ProviderJudge,
    TogetherJudge,
    _split_kwargs,
    get_judge,
)
from adapt_agent.optimization.providers import (
    AnthropicProvider,
    AzureOpenAIProvider,
    BedrockProvider,
    CohereProvider,
    GeminiProvider,
    GroqProvider,
    HuggingFaceProvider,
    MistralProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    TogetherProvider,
)

# -- Fake clients (reused shapes) ---------------------------------------------


class _Block:
    def __init__(self, text):
        self.text = text


class FakeAnthropicClient:
    def __init__(self, text):
        self._text = text

        class _Msgs:
            def __init__(self, t):
                self._t = t

            def create(self, **kwargs):
                class _Msg:
                    content = [_Block(t) for t in [self_outer]]

                # Build a message with a single text block.
                m = type("M", (), {})()
                m.content = [_Block(self._t)]
                return m

        self_outer = text
        self.messages = _Msgs(text)


class _OpenAIMsg:
    def __init__(self, content):
        self.content = content


class _OpenAIChoice:
    def __init__(self, content):
        self.message = _OpenAIMsg(content)


class _OpenAIResp:
    def __init__(self, content):
        self.choices = [_OpenAIChoice(content)]


class FakeOpenAIClient:
    def __init__(self, content):
        outer = content

        class _Comp:
            def create(self, **kwargs):
                return _OpenAIResp(outer)

        class _Chat:
            completions = _Comp()

        self.chat = _Chat()


class FakeGeminiClient:
    def __init__(self, text):
        outer = text

        class _Models:
            def generate_content(self, model, contents, config):
                return type("R", (), {"text": outer})()

        self.models = _Models()


class FakeMistralClient:
    def __init__(self, content):
        outer = content

        class _Chat:
            def complete(self, **kwargs):
                return _OpenAIResp(outer)

        self.chat = _Chat()


class FakeCohereClient:
    def __init__(self, content):
        self._content = content

    def chat(self, **kwargs):
        return type("R", (), {"message": type("M", (), {"content": self._content})()})()


class FakeBedrockClient:
    def __init__(self, text):
        self._text = text

    def converse(self, **kwargs):
        return {"output": {"message": {"content": [{"text": self._text}]}}}


class FakeHFClient:
    def __init__(self, content):
        self._content = content

    def chat_completion(self, messages, **kwargs):
        return _OpenAIResp(self._content)


SCORE_JSON = '{"score": 8, "pass": true, "reasoning": "good"}'


# -- ProviderJudge base behavior ----------------------------------------------


def test_provider_judge_is_llmjudge_subclass():
    assert issubclass(ProviderJudge, LLMJudge)


def test_claude_judge_scores_offline():
    judge = ClaudeJudge(client=FakeAnthropicClient(SCORE_JSON))
    v = judge.score("in", "out")
    assert isinstance(v, JudgeVerdict)
    assert v.score == pytest.approx(0.8)
    assert v.passed is True


def test_claude_judge_provider_property():
    judge = ClaudeJudge(client=FakeAnthropicClient(SCORE_JSON))
    assert isinstance(judge.provider, AnthropicProvider)
    assert judge.provider is judge.complete


def test_anthropic_judge_is_claude_judge_alias():
    assert AnthropicJudge is ClaudeJudge


def test_openai_judge_scores_offline():
    judge = OpenAIJudge(client=FakeOpenAIClient(SCORE_JSON))
    assert judge.score("i", "o").score == pytest.approx(0.8)
    assert isinstance(judge.provider, OpenAIProvider)


def test_gemini_judge_scores_offline():
    judge = GeminiJudge(client=FakeGeminiClient(SCORE_JSON))
    assert judge.score("i", "o").score == pytest.approx(0.8)
    assert isinstance(judge.provider, GeminiProvider)


def test_google_judge_alias():
    assert GoogleJudge is GeminiJudge


def test_mistral_judge_scores_offline():
    judge = MistralJudge(client=FakeMistralClient(SCORE_JSON))
    assert judge.score("i", "o").score == pytest.approx(0.8)
    assert isinstance(judge.provider, MistralProvider)


def test_cohere_judge_scores_offline():
    judge = CohereJudge(client=FakeCohereClient(SCORE_JSON))
    assert judge.score("i", "o").score == pytest.approx(0.8)
    assert isinstance(judge.provider, CohereProvider)


def test_bedrock_judge_scores_offline():
    judge = BedrockJudge(client=FakeBedrockClient(SCORE_JSON))
    assert judge.score("i", "o").score == pytest.approx(0.8)
    assert isinstance(judge.provider, BedrockProvider)


def test_huggingface_judge_scores_offline():
    judge = HuggingFaceJudge(client=FakeHFClient(SCORE_JSON))
    assert judge.score("i", "o").score == pytest.approx(0.8)
    assert isinstance(judge.provider, HuggingFaceProvider)


def test_groq_judge_scores_offline():
    judge = GroqJudge(client=FakeOpenAIClient(SCORE_JSON))
    assert judge.score("i", "o").score == pytest.approx(0.8)
    assert isinstance(judge.provider, GroqProvider)


def test_together_judge_scores_offline():
    judge = TogetherJudge(client=FakeOpenAIClient(SCORE_JSON))
    assert judge.score("i", "o").score == pytest.approx(0.8)
    assert isinstance(judge.provider, TogetherProvider)


def test_openrouter_judge_scores_offline():
    judge = OpenRouterJudge(client=FakeOpenAIClient(SCORE_JSON))
    assert judge.score("i", "o").score == pytest.approx(0.8)
    assert isinstance(judge.provider, OpenRouterProvider)


def test_ollama_judge_scores_offline():
    judge = OllamaJudge(client=FakeOpenAIClient(SCORE_JSON))
    assert judge.score("i", "o").score == pytest.approx(0.8)
    assert isinstance(judge.provider, OllamaProvider)


def test_azure_judge_scores_offline():
    judge = AzureOpenAIJudge(
        client=FakeOpenAIClient(SCORE_JSON),
        azure_endpoint="https://x.openai.azure.com",
    )
    assert judge.score("i", "o").score == pytest.approx(0.8)
    assert isinstance(judge.provider, AzureOpenAIProvider)


# -- default_model values ------------------------------------------------------


@pytest.mark.parametrize(
    "judge_cls,expected_model",
    [
        (ClaudeJudge, "claude-opus-4-8"),
        (OpenAIJudge, "gpt-4o-mini"),
        (AzureOpenAIJudge, "gpt-4o-mini"),
        (GeminiJudge, "gemini-2.0-flash"),
        (MistralJudge, "mistral-large-latest"),
        (CohereJudge, "command-r-plus"),
        (GroqJudge, "llama-3.3-70b-versatile"),
        (TogetherJudge, "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        (OpenRouterJudge, "openai/gpt-4o-mini"),
        (OllamaJudge, "llama3.1"),
        (BedrockJudge, "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        (HuggingFaceJudge, "meta-llama/Llama-3.3-70B-Instruct"),
    ],
)
def test_default_model_attribute(judge_cls, expected_model):
    assert judge_cls.default_model == expected_model


def test_default_model_applied_to_provider():
    judge = ClaudeJudge(client=FakeAnthropicClient(SCORE_JSON))
    assert judge.provider.model == "claude-opus-4-8"


def test_explicit_model_overrides_default():
    judge = ClaudeJudge(model="claude-haiku-x", client=FakeAnthropicClient(SCORE_JSON))
    assert judge.provider.model == "claude-haiku-x"


# -- get_judge routing ---------------------------------------------------------


@pytest.mark.parametrize(
    "name,cls",
    [
        ("anthropic", ClaudeJudge),
        ("claude", ClaudeJudge),
        ("openai", OpenAIJudge),
        ("azure_openai", AzureOpenAIJudge),
        ("gemini", GeminiJudge),
        ("google", GeminiJudge),
        ("mistral", MistralJudge),
        ("cohere", CohereJudge),
        ("groq", GroqJudge),
        ("together", TogetherJudge),
        ("openrouter", OpenRouterJudge),
        ("ollama", OllamaJudge),
        ("bedrock", BedrockJudge),
        ("huggingface", HuggingFaceJudge),
    ],
)
def test_get_judge_routing(name, cls):
    judge = get_judge(name, client=FakeOpenAIClient(SCORE_JSON))
    assert isinstance(judge, cls)


def test_get_judge_case_insensitive():
    judge = get_judge("CLAUDE", client=FakeAnthropicClient(SCORE_JSON))
    assert isinstance(judge, ClaudeJudge)


def test_get_judge_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_judge("not_a_real_provider")


def test_get_judge_passes_model_and_runs():
    judge = get_judge("claude", model="custom-model", client=FakeAnthropicClient(SCORE_JSON))
    assert judge.provider.model == "custom-model"
    assert judge.score("i", "o").score == pytest.approx(0.8)


def test_judge_registry_keys():
    for key in (
        "anthropic",
        "claude",
        "openai",
        "azure_openai",
        "gemini",
        "google",
        "mistral",
        "cohere",
        "groq",
        "together",
        "openrouter",
        "ollama",
        "bedrock",
        "huggingface",
    ):
        assert key in JUDGE_REGISTRY


# -- _split_kwargs partition ---------------------------------------------------


def test_split_kwargs_partitions_judge_vs_provider():
    judge_kw, provider_kw = _split_kwargs(
        {
            "rubric": "R",
            "pass_threshold": 0.9,
            "scale": 5,
            "max_failures": 3,
            "on_error": 0.1,
            "api_key": "k",
            "temperature": 0.7,
            "model": "m",
        }
    )
    assert judge_kw == {
        "rubric": "R",
        "pass_threshold": 0.9,
        "scale": 5,
        "max_failures": 3,
        "on_error": 0.1,
    }
    assert provider_kw == {"api_key": "k", "temperature": 0.7, "model": "m"}


def test_split_kwargs_empty():
    assert _split_kwargs({}) == ({}, {})


def test_split_kwargs_all_provider():
    judge_kw, provider_kw = _split_kwargs({"client": object(), "base_url": "x"})
    assert judge_kw == {}
    assert set(provider_kw) == {"client", "base_url"}


def test_judge_kwargs_routed_correctly_end_to_end():
    judge = ClaudeJudge(
        client=FakeAnthropicClient(SCORE_JSON),
        pass_threshold=0.95,
        scale=10,
        rubric="custom rubric",
        on_error=0.42,
    )
    assert judge.pass_threshold == 0.95
    assert judge.scale == 10
    assert judge.rubric == "custom rubric"
    assert judge.on_error == 0.42
    # provider got the client, judge got the options.
    assert isinstance(judge.provider, AnthropicProvider)


def test_provider_kwargs_routed_to_provider():
    judge = OpenAIJudge(
        client=FakeOpenAIClient(SCORE_JSON),
        temperature=0.33,
        max_tokens=42,
    )
    assert judge.provider.temperature == 0.33
    assert judge.provider.max_tokens == 42


# -- adversarial / score_is_normalized threading -------------------------------


def _capture_base_init(monkeypatch):
    """Patch the LLMJudge base __init__ to record the kwargs it receives.

    Independent of whether judge.py (agent A2) has finished adding the new
    keyword-only params: we only assert that ProviderJudge forwards them to the
    base constructor, which is judges.py's responsibility.
    """
    captured: dict = {}

    def fake_init(self, complete, **kw):
        captured["complete"] = complete
        captured["kw"] = kw

    monkeypatch.setattr(LLMJudge, "__init__", fake_init)
    return captured


def test_adversarial_threads_to_base(monkeypatch):
    captured = _capture_base_init(monkeypatch)
    ClaudeJudge(client=FakeAnthropicClient(SCORE_JSON), adversarial=True)
    assert captured["kw"].get("adversarial") is True


def test_score_is_normalized_threads_to_base(monkeypatch):
    captured = _capture_base_init(monkeypatch)
    ClaudeJudge(client=FakeAnthropicClient(SCORE_JSON), score_is_normalized=True)
    assert captured["kw"].get("score_is_normalized") is True


def test_both_new_kwargs_thread_and_default_off(monkeypatch):
    captured = _capture_base_init(monkeypatch)
    # When not supplied, judges.py must NOT inject them (defaults live on the base).
    ClaudeJudge(client=FakeAnthropicClient(SCORE_JSON))
    assert "adversarial" not in captured["kw"]
    assert "score_is_normalized" not in captured["kw"]


def test_new_kwargs_thread_through_get_judge(monkeypatch):
    captured = _capture_base_init(monkeypatch)
    get_judge(
        "openai",
        client=FakeOpenAIClient(SCORE_JSON),
        adversarial=True,
        score_is_normalized=True,
    )
    assert captured["kw"].get("adversarial") is True
    assert captured["kw"].get("score_is_normalized") is True


def test_new_kwargs_are_judge_kwargs_not_provider_kwargs():
    judge_kw, provider_kw = _split_kwargs(
        {"adversarial": True, "score_is_normalized": True, "api_key": "k"}
    )
    assert judge_kw == {"adversarial": True, "score_is_normalized": True}
    assert provider_kw == {"api_key": "k"}


def test_positional_model_used_over_default():
    # The positional ``model`` arg must beat the class default_model.
    via_default = ClaudeJudge(client=FakeAnthropicClient(SCORE_JSON))
    assert via_default.provider.model == "claude-opus-4-8"
    via_positional = ClaudeJudge("pos-model", client=FakeAnthropicClient(SCORE_JSON))
    assert via_positional.provider.model == "pos-model"


def test_model_in_kwargs_used_when_no_positional():
    judge = ClaudeJudge(client=FakeAnthropicClient(SCORE_JSON), model="kw-model")
    assert judge.provider.model == "kw-model"


# -- import safety -------------------------------------------------------------


def test_default_rubric_reexported():
    assert J._DEFAULT_RUBRIC is _DEFAULT_RUBRIC


def test_judges_import_with_nothing_installed():
    # Re-importing the module must not require any vendor SDK; the import at the
    # top of this file already proves it, but assert the public surface exists.
    import importlib

    mod = importlib.reload(J)
    assert hasattr(mod, "ClaudeJudge")
    assert hasattr(mod, "get_judge")
    assert hasattr(mod, "JUDGE_REGISTRY")


def test_provider_judge_base_is_not_directly_constructible():
    # The base ProviderJudge declares ``provider_cls`` annotation-only (concrete
    # subclasses set it), so constructing the base directly fails with
    # AttributeError -- it is not meant to be instantiated on its own.
    with pytest.raises(AttributeError):
        ProviderJudge()
