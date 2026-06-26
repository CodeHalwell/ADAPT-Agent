"""Provider-specific :class:`~adapt_agent.optimization.judge.LLMJudge` subclasses.

These are thin conveniences: each wires a vendor
:class:`~adapt_agent.optimization.providers.ModelProvider` into an
:class:`LLMJudge` so you can write ``ClaudeJudge(model=...)`` instead of
constructing a provider by hand. They inherit every capability of ``LLMJudge``
(scoring, pairwise comparison, critique, prompt improvement, ``as_metric``) and
differ only in which provider backs them.

All judges stay import-safe: the vendor SDK is imported lazily by the provider on
first use, so ``from adapt_agent.optimization.judges import ClaudeJudge`` works
with nothing installed.

Example::

    from adapt_agent.optimization.judges import ClaudeJudge, OpenAIJudge, GeminiJudge

    judge = ClaudeJudge(model="claude-opus-4-8")        # ANTHROPIC_API_KEY from env
    judge.score("What is 2+2?", "4", "4").score          # -> ~1.0

    # Use a different vendor for grading vs. the agent under optimization:
    grader = GeminiJudge(model="gemini-2.0-flash")
    harness = EvaluationHarness([grader.as_metric()], primary_metric="llm_judge")
"""

from __future__ import annotations

from typing import Any

from adapt_agent.optimization.judge import _DEFAULT_RUBRIC, LLMJudge
from adapt_agent.optimization.providers import (
    AnthropicProvider,
    AzureOpenAIProvider,
    BedrockProvider,
    CohereProvider,
    GeminiProvider,
    GroqProvider,
    HuggingFaceProvider,
    MistralProvider,
    ModelProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    TogetherProvider,
)

# Judge-level keyword arguments (everything except ``complete``), shared by the
# provider-specific constructors so their signatures stay consistent.
_JUDGE_KW = ("rubric", "pass_threshold", "scale", "max_failures", "on_error")


def _split_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition kwargs into (judge_kwargs, provider_kwargs)."""
    judge_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in _JUDGE_KW}
    return judge_kw, kwargs


class ProviderJudge(LLMJudge):
    """Base for provider-specific judges.

    Subclasses set :attr:`provider_cls`; the constructor instantiates it with the
    given model/credentials and hands it to :class:`LLMJudge`. Judge-level options
    (``rubric``, ``pass_threshold``, ``scale``, ``max_failures``, ``on_error``)
    and provider options (``api_key``, ``temperature``, ``client``, ...) may be
    mixed freely as keyword arguments.
    """

    # Concrete subclasses set this to a concrete provider class. Declared
    # annotation-only so the abstract base is never assigned here.
    provider_cls: type[ModelProvider]
    default_model: str | None = None

    def __init__(self, model: str | None = None, **kwargs: Any):
        judge_kw, provider_kw = _split_kwargs(dict(kwargs))
        chosen = model or self.default_model
        if chosen is not None:
            provider_kw.setdefault("model", chosen)
        provider = self.provider_cls(**provider_kw)
        super().__init__(provider, **judge_kw)

    @property
    def provider(self) -> ModelProvider:
        """The underlying provider instance (the judge's ``complete`` target)."""
        return self.complete  # type: ignore[return-value]


class ClaudeJudge(ProviderJudge):
    """LLM judge backed by Anthropic Claude."""

    provider_cls = AnthropicProvider
    default_model = "claude-opus-4-8"


# Backwards/everyday-friendly alias.
AnthropicJudge = ClaudeJudge


class OpenAIJudge(ProviderJudge):
    """LLM judge backed by OpenAI."""

    provider_cls = OpenAIProvider
    default_model = "gpt-4o-mini"


class AzureOpenAIJudge(ProviderJudge):
    """LLM judge backed by Azure OpenAI (``model`` is the deployment name)."""

    provider_cls = AzureOpenAIProvider
    default_model = "gpt-4o-mini"


class GeminiJudge(ProviderJudge):
    """LLM judge backed by Google Gemini."""

    provider_cls = GeminiProvider
    default_model = "gemini-2.0-flash"


# Friendly alias.
GoogleJudge = GeminiJudge


class MistralJudge(ProviderJudge):
    """LLM judge backed by Mistral."""

    provider_cls = MistralProvider
    default_model = "mistral-large-latest"


class CohereJudge(ProviderJudge):
    """LLM judge backed by Cohere."""

    provider_cls = CohereProvider
    default_model = "command-r-plus"


class GroqJudge(ProviderJudge):
    """LLM judge backed by Groq (OpenAI-compatible)."""

    provider_cls = GroqProvider
    default_model = "llama-3.3-70b-versatile"


class TogetherJudge(ProviderJudge):
    """LLM judge backed by Together AI (OpenAI-compatible)."""

    provider_cls = TogetherProvider
    default_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


class OpenRouterJudge(ProviderJudge):
    """LLM judge backed by OpenRouter (OpenAI-compatible)."""

    provider_cls = OpenRouterProvider
    default_model = "openai/gpt-4o-mini"


class OllamaJudge(ProviderJudge):
    """LLM judge backed by a local Ollama server (OpenAI-compatible)."""

    provider_cls = OllamaProvider
    default_model = "llama3.1"


class BedrockJudge(ProviderJudge):
    """LLM judge backed by AWS Bedrock (Converse API)."""

    provider_cls = BedrockProvider
    default_model = "anthropic.claude-3-5-sonnet-20241022-v2:0"


class HuggingFaceJudge(ProviderJudge):
    """LLM judge backed by the Hugging Face Inference API."""

    provider_cls = HuggingFaceProvider
    default_model = "meta-llama/Llama-3.3-70B-Instruct"


#: Registry of provider-specific judges, keyed by short name.
JUDGE_REGISTRY: dict[str, type[ProviderJudge]] = {
    "anthropic": ClaudeJudge,
    "claude": ClaudeJudge,
    "openai": OpenAIJudge,
    "azure_openai": AzureOpenAIJudge,
    "gemini": GeminiJudge,
    "google": GeminiJudge,
    "mistral": MistralJudge,
    "cohere": CohereJudge,
    "groq": GroqJudge,
    "together": TogetherJudge,
    "openrouter": OpenRouterJudge,
    "ollama": OllamaJudge,
    "bedrock": BedrockJudge,
    "huggingface": HuggingFaceJudge,
}


def get_judge(provider: str, model: str | None = None, **kwargs: Any) -> LLMJudge:
    """Construct a provider-specific judge by name.

    Args:
        provider: One of :data:`JUDGE_REGISTRY` (e.g. ``"claude"``, ``"openai"``,
            ``"gemini"``).
        model: Model identifier (defaults to the provider's recommended model).
        **kwargs: Mixed judge/provider keyword arguments.
    """
    key = provider.lower()
    cls = JUDGE_REGISTRY.get(key)
    if cls is None:
        raise KeyError(f"Unknown judge provider {provider!r}. Available: {sorted(JUDGE_REGISTRY)}")
    return cls(model=model, **kwargs)


__all__ = [
    "ProviderJudge",
    "ClaudeJudge",
    "AnthropicJudge",
    "OpenAIJudge",
    "AzureOpenAIJudge",
    "GeminiJudge",
    "GoogleJudge",
    "MistralJudge",
    "CohereJudge",
    "GroqJudge",
    "TogetherJudge",
    "OpenRouterJudge",
    "OllamaJudge",
    "BedrockJudge",
    "HuggingFaceJudge",
    "JUDGE_REGISTRY",
    "get_judge",
    "_DEFAULT_RUBRIC",
]
