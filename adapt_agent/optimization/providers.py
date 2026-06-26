"""Provider-agnostic model access for the evaluation / optimization stack.

Everything that needs an LLM -- the :class:`~adapt_agent.optimization.judge.LLMJudge`
for scoring and prompt improvement, and any LLM-driven proposer -- talks to a
:class:`ModelProvider`, never to a vendor SDK directly. A provider exposes one
method, :meth:`ModelProvider.complete` (``prompt -> text``), and is itself
callable, so it is a drop-in for the ``Callable[[str], str]`` completion function
used throughout this package.

This keeps ``adapt_agent`` import-safe and dependency-free: importing this module
imports no vendor SDK. Concrete providers import their SDK *lazily* on first use,
exactly like the framework adapters. Register your own with :func:`register_provider`
and select by name with :func:`get_provider`.

Examples::

    from adapt_agent.optimization.providers import get_provider, CallableProvider

    # Vendor-backed (SDK imported lazily, API key from env or argument):
    provider = get_provider("anthropic", model="claude-opus-4-8")
    provider("Say hello")            # -> "Hello!"

    # Wrap any function you already have (or a deterministic stub in tests):
    provider = CallableProvider(lambda prompt: "stub answer")

    from adapt_agent.optimization import LLMJudge
    judge = LLMJudge(provider)        # providers are accepted anywhere a
                                      # completion function is.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Callable


class ModelProvider(ABC):
    """Base class for provider-agnostic text completion.

    Subclasses implement :meth:`complete`. Instances are callable so they can be
    passed wherever a ``Callable[[str], str]`` completion function is expected
    (e.g. :class:`~adapt_agent.optimization.judge.LLMJudge`).

    Args:
        model: The model / deployment identifier this provider targets.
        temperature: Default sampling temperature.
        max_tokens: Default maximum tokens to generate.
        system: Optional default system instruction applied to every call.
    """

    name: str = "provider"

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system = system

    @abstractmethod
    def complete(self, prompt: str, **overrides: Any) -> str:
        """Return a text completion for ``prompt``.

        ``overrides`` may include ``temperature``, ``max_tokens`` or ``system``
        to vary behaviour per call.
        """

    def __call__(self, prompt: str, **overrides: Any) -> str:
        return self.complete(prompt, **overrides)

    # -- helpers for subclasses ------------------------------------------------

    def _resolved(self, overrides: dict[str, Any]) -> dict[str, Any]:
        return {
            "temperature": overrides.get("temperature", self.temperature),
            "max_tokens": overrides.get("max_tokens", self.max_tokens),
            "system": overrides.get("system", self.system),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(model={self.model!r})"


class CallableProvider(ModelProvider):
    """Adapt an existing ``Callable[[str], str]`` into a :class:`ModelProvider`.

    The most direct way to plug in a custom client, a cached function, or a
    deterministic stub in tests. The wrapped callable may optionally accept
    keyword overrides; if it does not, they are ignored.
    """

    name = "callable"

    def __init__(self, fn: Callable[..., str], *, model: str = "callable", **kw: Any):
        super().__init__(model, **kw)
        if not callable(fn):
            raise TypeError("CallableProvider requires a callable")
        self._fn = fn

    def complete(self, prompt: str, **overrides: Any) -> str:
        try:
            result = self._fn(prompt, **overrides)
        except TypeError:
            # The wrapped callable does not accept overrides; call plainly.
            result = self._fn(prompt)
        return result if isinstance(result, str) else str(result)


class EchoProvider(ModelProvider):
    """Deterministic, offline provider that echoes a templated response.

    Useful for smoke tests and documentation examples where no network or API
    key is available. Never makes a network call.
    """

    name = "echo"

    def __init__(self, *, template: str = "{prompt}", model: str = "echo", **kw: Any):
        super().__init__(model, **kw)
        self.template = template

    def complete(self, prompt: str, **overrides: Any) -> str:
        # Only substitute when the placeholder is present, so templates that
        # contain literal braces (e.g. JSON) pass through untouched.
        if "{prompt}" in self.template:
            return self.template.replace("{prompt}", prompt)
        return self.template


class AnthropicProvider(ModelProvider):
    """Anthropic Claude provider (SDK imported lazily on first call).

    Requires the ``anthropic`` package and an API key (argument or
    ``ANTHROPIC_API_KEY``). A pre-built ``client`` may be injected to reuse a
    configured SDK client or for testing.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        *,
        api_key: str | None = None,
        client: Any = None,
        **kw: Any,
    ):
        super().__init__(model, **kw)
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic  # lazy: importing this module never needs the SDK
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise ImportError(
                    "AnthropicProvider requires the 'anthropic' package: " "pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, prompt: str, **overrides: Any) -> str:
        cfg = self._resolved(overrides)
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": cfg["max_tokens"],
            "temperature": cfg["temperature"],
            "messages": [{"role": "user", "content": prompt}],
        }
        if cfg["system"]:
            kwargs["system"] = cfg["system"]
        message = client.messages.create(**kwargs)
        return _join_text_blocks(getattr(message, "content", message))


class OpenAIProvider(ModelProvider):
    """OpenAI Chat Completions provider (SDK imported lazily on first call).

    Requires the ``openai`` package and an API key (argument or
    ``OPENAI_API_KEY``). A pre-built ``client`` may be injected.
    """

    name = "openai"
    #: Env var consulted for the API key (overridden by OpenAI-compatible subclasses).
    api_key_env = "OPENAI_API_KEY"
    #: Default base URL (``None`` -> the SDK default). Subclasses point this at
    #: OpenAI-compatible gateways (Groq, Together, Ollama, OpenRouter, ...).
    default_base_url: str | None = None

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
        **kw: Any,
    ):
        super().__init__(model, **kw)
        self._api_key = api_key or os.environ.get(self.api_key_env)
        self._base_url = base_url or self.default_base_url
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai  # lazy
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise ImportError(
                    "OpenAIProvider requires the 'openai' package: pip install openai"
                ) from exc
            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def complete(self, prompt: str, **overrides: Any) -> str:
        cfg = self._resolved(overrides)
        client = self._get_client()
        messages = []
        if cfg["system"]:
            messages.append({"role": "system", "content": cfg["system"]})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )
        return response.choices[0].message.content or ""


class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI provider (uses ``openai.AzureOpenAI``, imported lazily).

    ``model`` is the Azure *deployment* name. Endpoint/version come from
    arguments or ``AZURE_OPENAI_ENDPOINT`` / ``OPENAI_API_VERSION``.
    """

    name = "azure_openai"
    api_key_env = "AZURE_OPENAI_API_KEY"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        client: Any = None,
        **kw: Any,
    ):
        super().__init__(model, api_key=api_key, client=client, **kw)
        self._azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self._api_version = api_version or os.environ.get("OPENAI_API_VERSION", "2024-06-01")

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai  # lazy
            except ImportError as exc:  # pragma: no cover
                raise ImportError("AzureOpenAIProvider requires 'openai'") from exc
            self._client = openai.AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )
        return self._client


class GroqProvider(OpenAIProvider):
    """Groq (OpenAI-compatible API), SDK imported lazily."""

    name = "groq"
    api_key_env = "GROQ_API_KEY"
    default_base_url = "https://api.groq.com/openai/v1"

    def __init__(self, model: str = "llama-3.3-70b-versatile", **kw: Any):
        super().__init__(model, **kw)


class TogetherProvider(OpenAIProvider):
    """Together AI (OpenAI-compatible API), SDK imported lazily."""

    name = "together"
    api_key_env = "TOGETHER_API_KEY"
    default_base_url = "https://api.together.xyz/v1"

    def __init__(self, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo", **kw: Any):
        super().__init__(model, **kw)


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter (OpenAI-compatible API), SDK imported lazily."""

    name = "openrouter"
    api_key_env = "OPENROUTER_API_KEY"
    default_base_url = "https://openrouter.ai/api/v1"

    def __init__(self, model: str = "openai/gpt-4o-mini", **kw: Any):
        super().__init__(model, **kw)


class OllamaProvider(OpenAIProvider):
    """Local Ollama server via its OpenAI-compatible endpoint.

    Defaults to ``http://localhost:11434/v1`` and a placeholder key (Ollama
    ignores auth). Override ``base_url`` for a remote host.
    """

    name = "ollama"
    api_key_env = "OLLAMA_API_KEY"
    default_base_url = "http://localhost:11434/v1"

    def __init__(self, model: str = "llama3.1", *, api_key: str | None = None, **kw: Any):
        super().__init__(model, api_key=api_key or "ollama", **kw)


class GeminiProvider(ModelProvider):
    """Google Gemini provider (uses the ``google-genai`` SDK, imported lazily).

    Falls back to the legacy ``google-generativeai`` package if ``google-genai``
    is unavailable. API key from argument or ``GEMINI_API_KEY`` /
    ``GOOGLE_API_KEY``.
    """

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        *,
        api_key: str | None = None,
        client: Any = None,
        **kw: Any,
    ):
        super().__init__(model, **kw)
        self._api_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        self._client = client
        self._legacy = False

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai  # lazy, modern SDK

                self._client = genai.Client(api_key=self._api_key)
            except ImportError:
                try:
                    import google.generativeai as genai_legacy  # lazy, legacy SDK
                except ImportError as exc:  # pragma: no cover
                    raise ImportError(
                        "GeminiProvider requires 'google-genai' (or 'google-generativeai')"
                    ) from exc
                genai_legacy.configure(api_key=self._api_key)
                self._client = genai_legacy
                self._legacy = True
        return self._client

    def complete(self, prompt: str, **overrides: Any) -> str:
        cfg = self._resolved(overrides)
        client = self._get_client()
        full_prompt = f"{cfg['system']}\n\n{prompt}" if cfg["system"] else prompt
        if self._legacy:  # google.generativeai
            model = client.GenerativeModel(self.model)
            response = model.generate_content(full_prompt)
            return getattr(response, "text", "") or ""
        # google-genai
        config = {"temperature": cfg["temperature"], "max_output_tokens": cfg["max_tokens"]}
        response = client.models.generate_content(
            model=self.model, contents=full_prompt, config=config
        )
        return getattr(response, "text", "") or ""


class MistralProvider(ModelProvider):
    """Mistral provider (uses the ``mistralai`` SDK, imported lazily)."""

    name = "mistral"

    def __init__(
        self,
        model: str = "mistral-large-latest",
        *,
        api_key: str | None = None,
        client: Any = None,
        **kw: Any,
    ):
        super().__init__(model, **kw)
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from mistralai import Mistral  # lazy
            except ImportError as exc:  # pragma: no cover
                raise ImportError("MistralProvider requires 'mistralai'") from exc
            self._client = Mistral(api_key=self._api_key)
        return self._client

    def complete(self, prompt: str, **overrides: Any) -> str:
        cfg = self._resolved(overrides)
        client = self._get_client()
        messages = []
        if cfg["system"]:
            messages.append({"role": "system", "content": cfg["system"]})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )
        return response.choices[0].message.content or ""


class CohereProvider(ModelProvider):
    """Cohere provider (uses the ``cohere`` SDK v2, imported lazily)."""

    name = "cohere"

    def __init__(
        self,
        model: str = "command-r-plus",
        *,
        api_key: str | None = None,
        client: Any = None,
        **kw: Any,
    ):
        super().__init__(model, **kw)
        self._api_key = api_key or os.environ.get("COHERE_API_KEY")
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import cohere  # lazy
            except ImportError as exc:  # pragma: no cover
                raise ImportError("CohereProvider requires 'cohere'") from exc
            self._client = cohere.ClientV2(api_key=self._api_key)
        return self._client

    def complete(self, prompt: str, **overrides: Any) -> str:
        cfg = self._resolved(overrides)
        client = self._get_client()
        messages = []
        if cfg["system"]:
            messages.append({"role": "system", "content": cfg["system"]})
        messages.append({"role": "user", "content": prompt})
        response = client.chat(model=self.model, messages=messages, temperature=cfg["temperature"])
        return _join_text_blocks(getattr(response.message, "content", ""))


class BedrockProvider(ModelProvider):
    """AWS Bedrock provider via the Converse API (uses ``boto3``, imported lazily)."""

    name = "bedrock"

    def __init__(
        self,
        model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        *,
        region_name: str | None = None,
        client: Any = None,
        **kw: Any,
    ):
        super().__init__(model, **kw)
        self._region = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import boto3  # lazy
            except ImportError as exc:  # pragma: no cover
                raise ImportError("BedrockProvider requires 'boto3'") from exc
            self._client = boto3.client("bedrock-runtime", region_name=self._region)
        return self._client

    def complete(self, prompt: str, **overrides: Any) -> str:
        cfg = self._resolved(overrides)
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "modelId": self.model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "temperature": cfg["temperature"],
                "maxTokens": cfg["max_tokens"],
            },
        }
        if cfg["system"]:
            kwargs["system"] = [{"text": cfg["system"]}]
        response = client.converse(**kwargs)
        blocks = response["output"]["message"]["content"]
        return "".join(b.get("text", "") for b in blocks)


class HuggingFaceProvider(ModelProvider):
    """Hugging Face Inference provider (uses ``huggingface_hub``, imported lazily)."""

    name = "huggingface"

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.3-70B-Instruct",
        *,
        api_key: str | None = None,
        client: Any = None,
        **kw: Any,
    ):
        super().__init__(model, **kw)
        self._api_key = (
            api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        )
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from huggingface_hub import InferenceClient  # lazy
            except ImportError as exc:  # pragma: no cover
                raise ImportError("HuggingFaceProvider requires 'huggingface_hub'") from exc
            self._client = InferenceClient(model=self.model, token=self._api_key)
        return self._client

    def complete(self, prompt: str, **overrides: Any) -> str:
        cfg = self._resolved(overrides)
        client = self._get_client()
        messages = []
        if cfg["system"]:
            messages.append({"role": "system", "content": cfg["system"]})
        messages.append({"role": "user", "content": prompt})
        response = client.chat_completion(
            messages, model=self.model, temperature=cfg["temperature"], max_tokens=cfg["max_tokens"]
        )
        return response.choices[0].message.content or ""


# -- registry -----------------------------------------------------------------

_PROVIDERS: dict[str, type[ModelProvider]] = {
    "callable": CallableProvider,
    "echo": EchoProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "azure_openai": AzureOpenAIProvider,
    "gemini": GeminiProvider,
    "mistral": MistralProvider,
    "cohere": CohereProvider,
    "groq": GroqProvider,
    "together": TogetherProvider,
    "openrouter": OpenRouterProvider,
    "ollama": OllamaProvider,
    "bedrock": BedrockProvider,
    "huggingface": HuggingFaceProvider,
}


def register_provider(name: str, provider_cls: type[ModelProvider]) -> None:
    """Register a custom provider class under ``name`` for :func:`get_provider`."""
    if not (isinstance(provider_cls, type) and issubclass(provider_cls, ModelProvider)):
        raise TypeError("provider_cls must be a ModelProvider subclass")
    _PROVIDERS[name.lower()] = provider_cls


def available_providers() -> list[str]:
    """Return the names of all registered providers."""
    return sorted(_PROVIDERS)


def get_provider(name: str, **kwargs: Any) -> ModelProvider:
    """Instantiate a registered provider by name.

    Args:
        name: Provider key (e.g. ``"anthropic"``, ``"openai"``, ``"echo"``).
        **kwargs: Forwarded to the provider constructor (``model``, ``api_key``,
            ``temperature``, ...).
    """
    key = name.lower()
    cls = _PROVIDERS.get(key)
    if cls is None:
        raise KeyError(f"Unknown provider {name!r}. Available: {available_providers()}")
    return cls(**kwargs)


def as_provider(obj: Any) -> ModelProvider:
    """Coerce a provider, a callable, or a provider name into a :class:`ModelProvider`."""
    if isinstance(obj, ModelProvider):
        return obj
    if isinstance(obj, str):
        return get_provider(obj)
    if callable(obj):
        return CallableProvider(obj)
    raise TypeError(f"Cannot coerce {type(obj)!r} into a ModelProvider")


def _join_text_blocks(content: Any) -> str:
    """Concatenate text from an Anthropic-style content block list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


__all__ = [
    "ModelProvider",
    "CallableProvider",
    "EchoProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "register_provider",
    "available_providers",
    "get_provider",
    "as_provider",
]
