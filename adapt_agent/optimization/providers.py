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

A completion that the model did not finish is never returned as if it were an
answer. A response cut off at the token limit raises :class:`TruncatedResponseError`,
a safety refusal raises :class:`RefusedResponseError`, and a response with no
text at all raises :class:`EmptyResponseError` -- all subclasses of
:class:`ProviderResponseError`. Returning ``""`` or a half-written JSON object
instead lets a caller such as the judge score it, and a truncated grade is not
a grade.

Token usage is recorded without changing the ``-> str`` contract:
:attr:`ModelProvider.last_usage` holds the latest call's counts and
:attr:`ModelProvider.usage` accumulates them, for every provider whose SDK
response shape is verified (Anthropic, OpenAI and OpenAI-compatible, Gemini).

Examples::

    from adapt_agent.optimization.providers import get_provider, CallableProvider

    # Vendor-backed (SDK imported lazily, API key from env or argument):
    provider = get_provider("anthropic", model="claude-opus-5-5")
    provider("Say hello")            # -> "Hello!"
    provider.usage                   # -> TokenUsage(input_tokens=..., ...)

    # Wrap any function you already have (or a deterministic stub in tests):
    provider = CallableProvider(lambda prompt: "stub answer")

    from adapt_agent.optimization import LLMJudge
    judge = LLMJudge(provider)        # providers are accepted anywhere a
                                      # completion function is.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# -- default model identifiers -------------------------------------------------
#
# The single place every default model ID lives; ``judges.py`` imports these
# rather than repeating them, so a refresh is one edit.
#
# How to refresh: only change a value on evidence of the vendor's *current* ID,
# never from memory. Good evidence, in order: the vendor's own model list
# (Anthropic ``client.models.list()``, the OpenAI ``ChatModel`` literal in
# ``openai/types/shared/chat_model.py``), then a default the vendor's own SDK
# ships (``agents.models.default_models.get_default_model()`` in the OpenAI
# Agents SDK, ``google.adk.agents.llm_agent.LlmAgent.DEFAULT_MODEL`` in Google
# ADK). Then run ``tests/test_provider_sdk_contracts.py`` with the SDKs installed.
# Values marked "unverified" were carried forward because no such evidence was
# available when this table was last refreshed; they may be stale.

#: Anthropic: Claude Opus 5.5, the current Opus.
ANTHROPIC_DEFAULT_MODEL = "claude-opus-5-5"
#: OpenAI: the OpenAI Agents SDK's own default model (openai-agents 0.26).
OPENAI_DEFAULT_MODEL = "gpt-5.6-luna"
#: Azure OpenAI takes a *deployment* name the operator chose. Unverified.
AZURE_OPENAI_DEFAULT_DEPLOYMENT = "gpt-4o-mini"
#: Google: Google ADK's ``LlmAgent.DEFAULT_MODEL`` (google-adk 2.15).
GEMINI_DEFAULT_MODEL = "gemini-3.5-flash"
#: Unverified (SDK not installed where this table was refreshed).
MISTRAL_DEFAULT_MODEL = "mistral-large-latest"
#: Unverified (SDK not installed where this table was refreshed).
COHERE_DEFAULT_MODEL = "command-r-plus"
#: Unverified (gateway catalogue, no SDK-shipped evidence).
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
#: Unverified (gateway catalogue, no SDK-shipped evidence).
TOGETHER_DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
#: Unverified (gateway catalogue, no SDK-shipped evidence).
OPENROUTER_DEFAULT_MODEL = "openai/gpt-4o-mini"
#: Unverified (whatever the local Ollama server has pulled).
OLLAMA_DEFAULT_MODEL = "llama3.1"
#: Unverified (boto3 not installed where this table was refreshed).
BEDROCK_DEFAULT_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
#: Unverified (Hub model id, no SDK-shipped evidence).
HUGGINGFACE_DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

#: Every default above, keyed by provider name.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": ANTHROPIC_DEFAULT_MODEL,
    "openai": OPENAI_DEFAULT_MODEL,
    "azure_openai": AZURE_OPENAI_DEFAULT_DEPLOYMENT,
    "gemini": GEMINI_DEFAULT_MODEL,
    "mistral": MISTRAL_DEFAULT_MODEL,
    "cohere": COHERE_DEFAULT_MODEL,
    "groq": GROQ_DEFAULT_MODEL,
    "together": TOGETHER_DEFAULT_MODEL,
    "openrouter": OPENROUTER_DEFAULT_MODEL,
    "ollama": OLLAMA_DEFAULT_MODEL,
    "bedrock": BEDROCK_DEFAULT_MODEL,
    "huggingface": HUGGINGFACE_DEFAULT_MODEL,
}

#: Substrings identifying Anthropic models that reject user-set sampling
#: parameters (``temperature`` / ``top_p`` / ``top_k``) with a 400 error.
#:
#: This is the *secondary* guard. The primary one is the installed SDK itself:
#: a sampling parameter is only sent when the SDK's ``messages.create`` accepts
#: it (see :func:`_accepts_keyword`), and current SDKs accept none of them. The
#: list only matters for an older SDK that still takes ``temperature`` while
#: the model behind it refuses it. It covers Opus 4.7 and later, the whole
#: Fable / Mythos line, and the 5-series Sonnet and Haiku (which reject any
#: non-default value).
_NO_SAMPLING_MODEL_MARKERS: tuple[str, ...] = (
    "opus-4-7",
    "opus-4-8",
    "opus-5",  # opus-5, opus-5-5, ...
    "sonnet-5",  # sonnet-5, sonnet-5-5, ...
    "haiku-5",  # haiku-5-5, ...
    "fable-5",  # fable-5, fable-5-1, fable-5-2, ...
    "mythos",
)


def _model_rejects_sampling(model: str | None) -> bool:
    """Return ``True`` if ``model`` rejects user-set sampling parameters.

    Matched case-insensitively against :data:`_NO_SAMPLING_MODEL_MARKERS`.
    """
    if not model:
        return False
    lowered = model.lower()
    return any(marker in lowered for marker in _NO_SAMPLING_MODEL_MARKERS)


def _sampling_error(exc: Exception) -> bool:
    """Heuristic: does ``exc`` look like a 400/invalid-request about sampling?

    Detects errors raised by an SDK when ``temperature`` / ``top_p`` / ``top_k``
    are unsupported, so the call can be retried once without those params.
    A local ``TypeError`` for an unknown keyword is deliberately *not* matched:
    that is a mismatch between this module and the installed SDK, prevented
    up front by :func:`_accepts_keyword`, and retrying would hide it.
    """
    if isinstance(exc, TypeError):
        return False
    text = str(exc).lower()
    mentions_param = any(p in text for p in ("temperature", "top_p", "top_k", "top-p", "top-k"))
    looks_like_400 = any(
        token in text
        for token in ("400", "invalid_request", "invalid request", "unsupported", "not supported")
    )
    return mentions_param and looks_like_400


_SAMPLING_KEYS: tuple[str, ...] = ("temperature", "top_p", "top_k")


def _call_with_sampling_retry(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    """Call ``fn(**kwargs)``; on a sampling-related 400, retry once without them.

    Never lets a sampling parameter crash a completion: if the SDK raises an
    error that looks like an invalid-request about ``temperature`` / ``top_p`` /
    ``top_k`` (see :func:`_sampling_error`), the offending keys are stripped and
    the call is retried exactly once. Any other error propagates unchanged.
    """
    try:
        return fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is a sampling error
        retry = _strip_sampling(kwargs) if _sampling_error(exc) else None
        if retry is None:
            raise
        logger.warning(
            "SDK rejected sampling params (%s); retrying without temperature/top_p/top_k.",
            exc,
        )
        return fn(**retry)


def _strip_sampling(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """``kwargs`` without sampling keys, or ``None`` if it carried none.

    Sampling params may sit at the top level or nested inside a ``config``
    mapping (google-genai passes temperature/top_p/top_k inside ``config=``).
    """
    nested = kwargs.get("config")
    has_top_level = any(key in kwargs for key in _SAMPLING_KEYS)
    has_nested = isinstance(nested, dict) and any(key in nested for key in _SAMPLING_KEYS)
    if not has_top_level and not has_nested:
        return None
    retry = {k: v for k, v in kwargs.items() if k not in _SAMPLING_KEYS}
    if isinstance(retry.get("config"), dict):
        retry["config"] = {k: v for k, v in retry["config"].items() if k not in _SAMPLING_KEYS}
    return retry


def _rejected_param(exc: Exception) -> str | None:
    """The request parameter a 400 says is unsupported, if it names one.

    The OpenAI SDK carries it as ``exc.param``; otherwise the message is read
    for the shapes the API uses ("Unsupported parameter: 'max_tokens' ...",
    "'max_tokens' is not supported"). A message naming *two* parameters --
    "'max_tokens' is not supported ... use 'max_completion_tokens'" -- names
    the rejected one first, which is the one returned.
    """
    if isinstance(exc, TypeError):
        return None
    param = getattr(exc, "param", None)
    if isinstance(param, str) and param:
        return param
    text = str(exc)
    match = re.search(
        r"unsupported (?:parameter|value):\s*'([A-Za-z_]+)'|'([A-Za-z_]+)' is not supported",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1) or match.group(2)
    return None


_KEYWORD_KINDS = (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)


def _accepted_keywords(fn: Any) -> frozenset[str] | None:
    """The keyword names ``fn`` accepts, or ``None`` if it accepts any.

    ``None`` also when the signature cannot be read (some builtins).

    Read from the signature rather than discovered by calling and catching
    ``TypeError``: a ``TypeError`` raised *inside* the call (an SDK rejecting a
    keyword, or a bug in a user's function) is indistinguishable from "this
    callable has no such parameter" once caught, and treating the first as the
    second is how an SDK upgrade turned into a judge that silently scored zero.
    Not cached: callables here are often per-call lambdas, and reading a
    signature costs far less than the network call it guards.
    """
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):  # builtins / C callables without a signature
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return None
    return frozenset(p.name for p in params if p.kind in _KEYWORD_KINDS)


def _accepts_keyword(fn: Any, name: str) -> bool:
    """Whether ``fn`` can be called with keyword ``name``."""
    accepted = _accepted_keywords(fn)
    return accepted is None or name in accepted


# -- typed response failures -----------------------------------------------------


class ProviderResponseError(RuntimeError):
    """The model answered, but not with a usable completion.

    Distinct from an HTTP or network error: the call succeeded and the model
    stopped for a reason that makes its text unusable as an answer. Messages
    never quote the model's text (``partial_text`` carries it), so a retry
    classifier reading the message cannot be steered by what the model wrote.

    Attributes:
        kind: ``"truncated"``, ``"refused"`` or ``"empty"``.
        provider: The provider name.
        model: The model identifier.
        stop_reason: The vendor's stop / finish reason, as reported.
        partial_text: Whatever text the model did produce (may be ``""``).
    """

    kind: str = "invalid_response"

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        stop_reason: Any = None,
        partial_text: str = "",
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.stop_reason = stop_reason
        self.partial_text = partial_text


class TruncatedResponseError(ProviderResponseError):
    """The model hit its output-token limit before finishing."""

    kind = "truncated"


class RefusedResponseError(ProviderResponseError):
    """The model (or the vendor's safety system) declined to answer."""

    kind = "refused"


class EmptyResponseError(ProviderResponseError):
    """The model finished without producing any text."""

    kind = "empty"


class StructuredOutputUnsupported(NotImplementedError):
    """This provider (or its installed SDK) cannot constrain output to a schema.

    Raised by :meth:`ModelProvider.complete_json` before any request is made,
    so a caller can fall back to plain :meth:`ModelProvider.complete`.
    """


# -- token usage ----------------------------------------------------------------


@dataclass
class TokenUsage:
    """Token counts for one call (``last_usage``) or a running total (``usage``).

    ``output_tokens`` includes reasoning / thinking tokens where the vendor
    bills them as output.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


class ModelProvider(ABC):
    """Base class for provider-agnostic text completion.

    Subclasses implement :meth:`complete`. Instances are callable so they can be
    passed wherever a ``Callable[[str], str]`` completion function is expected
    (e.g. :class:`~adapt_agent.optimization.judge.LLMJudge`).

    Args:
        model: The model / deployment identifier this provider targets.
        temperature: Default sampling temperature.
        max_tokens: Default maximum tokens to generate (the class's
            :attr:`default_max_tokens` when omitted).
        system: Optional default system instruction applied to every call.
    """

    name: str = "provider"

    #: Highest sampling temperature this provider's API accepts. Requested
    #: temperatures are clamped into ``[0, max_temperature]`` before the call.
    max_temperature: float = 2.0

    #: Output-token budget when the caller sets none.
    default_max_tokens: int = 1024

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        system: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = self.default_max_tokens if max_tokens is None else max_tokens
        self.system = system
        #: Usage of the most recent completed call, or ``None``.
        self.last_usage: TokenUsage | None = None
        #: Running total across every call on this instance.
        self.usage = TokenUsage()
        self._usage_lock = threading.Lock()
        #: Set once the API rejected a sampling parameter for this model, so
        #: later calls stop sending it instead of paying a failed request each.
        self._sampling_rejected = False
        #: Set once the API rejected this instance's output-token parameter
        #: name; the alternative it accepted is used from then on.
        self._token_key_override: str | None = None

    @abstractmethod
    def complete(self, prompt: str, **overrides: Any) -> str:
        """Return a text completion for ``prompt``.

        ``overrides`` may include ``temperature``, ``max_tokens`` or ``system``
        to vary behaviour per call.

        Raises:
            ProviderResponseError: The model did not finish its answer
                (truncated, refused or empty).
        """

    def complete_json(self, prompt: str, schema: dict[str, Any], **overrides: Any) -> str:
        """Return a completion constrained to the JSON ``schema``, as JSON text.

        Returns the raw text (not a parsed object) so callers keep the exact
        completion for auditing and parse it the same way as free text.
        Providers whose SDK supports structured output override this; the base
        raises :class:`StructuredOutputUnsupported` without making a request.
        """
        raise StructuredOutputUnsupported(f"{type(self).__name__} has no structured output")

    @property
    def supports_structured_output(self) -> bool:
        """Whether :meth:`complete_json` is implemented by this class."""
        return type(self).complete_json is not ModelProvider.complete_json

    async def acomplete(self, prompt: str, **overrides: Any) -> str:
        """Async twin of :meth:`complete`, run on a worker thread.

        Keeps an event loop responsive while a provider call is in flight. It
        reuses the synchronous client, so usage accounting and remembered
        parameter rejections are shared with :meth:`complete`.
        """
        return await asyncio.to_thread(self.complete, prompt, **overrides)

    def __call__(self, prompt: str, **overrides: Any) -> str:
        return self.complete(prompt, **overrides)

    # -- helpers for subclasses ------------------------------------------------

    def _resolved(self, overrides: dict[str, Any]) -> dict[str, Any]:
        return {
            "temperature": overrides.get("temperature", self.temperature),
            "max_tokens": overrides.get("max_tokens", self.max_tokens),
            "system": overrides.get("system", self.system),
        }

    def _sampling(self, overrides: dict[str, Any], fn: Any = None) -> dict[str, Any]:
        """Resolve sampling params (``temperature``/``top_p``) safely.

        Returns a dict containing only the sampling keys that should be sent to
        the SDK:

        * Models that reject user-set sampling (see
          :data:`_NO_SAMPLING_MODEL_MARKERS`), and models whose API already
          rejected it on this instance, get an EMPTY dict.
        * When ``fn`` (the SDK method about to be called) is given, a key its
          signature does not accept is left out: sending it would raise a local
          ``TypeError`` before any request is made.
        * Otherwise ``temperature`` is clamped into ``[0, max_temperature]`` and
          ``top_p`` (if supplied) into ``[0, 1]``. A requested temperature above
          the maximum is logged and clamped rather than crashing the call.
        """
        if self._sampling_rejected or _model_rejects_sampling(self.model):
            return {}

        temperature = overrides.get("temperature", self.temperature)
        result: dict[str, Any] = {}
        if temperature is not None:
            if temperature > self.max_temperature:
                logger.warning(
                    "Requested temperature %s exceeds max %s for model %r; clamping.",
                    temperature,
                    self.max_temperature,
                    self.model,
                )
            result["temperature"] = min(max(temperature, 0.0), self.max_temperature)

        top_p = overrides.get("top_p")
        if top_p is not None:
            result["top_p"] = min(max(top_p, 0.0), 1.0)
        if fn is not None:
            result = {k: v for k, v in result.items() if _accepts_keyword(fn, k)}
        return result

    def _call_remembering(
        self, fn: Callable[..., Any], kwargs: dict[str, Any], *, token_keys: tuple[str, ...] = ()
    ) -> Any:
        """Call ``fn(**kwargs)``, recovering once from a rejected parameter.

        Two rejections are recoverable, and both are *remembered* on this
        instance so the next call does not repeat the failed request:

        * a sampling parameter (``temperature`` / ``top_p`` / ``top_k``): the
          keys are dropped;
        * the output-token parameter, when ``token_keys`` names the
          alternatives (``("max_completion_tokens", "max_tokens")``): the
          value moves to the other name.

        Anything else propagates unchanged.
        """
        for _ in range(len(token_keys) + 2):
            try:
                return fn(**kwargs)
            except Exception as exc:  # noqa: BLE001 - re-raised unless recoverable
                rejected = _rejected_param(exc)
                swapped = self._swap_token_key(kwargs, rejected, token_keys)
                if swapped is not None:
                    logger.warning(
                        "%s rejected %r for model %r; using %r from now on.",
                        self.name,
                        rejected,
                        self.model,
                        swapped,
                    )
                    continue
                stripped = _strip_sampling(kwargs) if _sampling_error(exc) else None
                if stripped is None:
                    raise
                logger.warning(
                    "SDK rejected sampling params (%s); retrying without temperature/top_p/top_k "
                    "and not sending them again for model %r.",
                    exc,
                    self.model,
                )
                self._sampling_rejected = True
                kwargs = stripped
        return fn(**kwargs)

    def _swap_token_key(
        self, kwargs: dict[str, Any], rejected: str | None, token_keys: tuple[str, ...]
    ) -> str | None:
        """Move the token limit to the alternative key; return its new name."""
        if not rejected or rejected not in token_keys or rejected not in kwargs:
            return None
        alternatives = [k for k in token_keys if k != rejected]
        if not alternatives:
            return None
        replacement = alternatives[0]
        kwargs[replacement] = kwargs.pop(rejected)
        self._token_key_override = replacement
        return replacement

    def _record_usage(self, input_tokens: Any, output_tokens: Any) -> None:
        """Record one call's token counts in ``last_usage`` and ``usage``."""
        current = TokenUsage(_int(input_tokens), _int(output_tokens), 1)
        with self._usage_lock:
            self.last_usage = current
            self.usage = TokenUsage(
                self.usage.input_tokens + current.input_tokens,
                self.usage.output_tokens + current.output_tokens,
                self.usage.calls + 1,
            )

    def _response_error(
        self,
        cls: type[ProviderResponseError],
        detail: str,
        *,
        stop_reason: Any = None,
        partial_text: str = "",
    ) -> ProviderResponseError:
        return cls(
            f"{self.name} model {self.model!r} {detail}",
            provider=self.name,
            model=self.model,
            stop_reason=stop_reason,
            partial_text=partial_text,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(model={self.model!r})"


def _configure_sdk_retries(client: Any, sdk_max_retries: int | None) -> Any:
    """Return ``client`` with the SDK's own retry count set to ``sdk_max_retries``.

    The SDKs retry 408/409/429/5xx twice by default, and the judge and the
    harness each run a :class:`~adapt_agent.optimization.retry.RetryPolicy`
    above them. Stacked, one row under a persistent 429 made nine HTTP calls
    with the backoff reset at every layer. The policy layer is the one that
    honours ``Retry-After``, classifies transients and excludes rows, so the SDK
    layer is turned off by default.

    Only applied to a client that reports an ``int`` ``max_retries`` and has
    ``with_options`` (the Anthropic and OpenAI clients); anything else -- a
    test double, another library -- is returned untouched. ``None`` leaves the
    client as configured.
    """
    if sdk_max_retries is None:
        return client
    current = getattr(client, "max_retries", None)
    if not isinstance(current, int) or isinstance(current, bool) or current == sdk_max_retries:
        return client
    with_options = getattr(client, "with_options", None)
    if not callable(with_options):
        return client
    return with_options(max_retries=sdk_max_retries)


class CallableProvider(ModelProvider):
    """Adapt an existing ``Callable[[str], str]`` into a :class:`ModelProvider`.

    The most direct way to plug in a custom client, a cached function, or a
    deterministic stub in tests. The wrapped callable may optionally accept
    keyword overrides; those it does not accept (read from its signature) are
    not passed -- **except** that dropping ``system``, which callers such as
    :class:`~adapt_agent.optimization.judge.LLMJudge` use to carry the grading
    rubric, is logged rather than silent: a judge that grades without its rubric
    returns confident-looking scores indistinguishable from a working judge,
    which is a worse failure than a loud one.

    A ``TypeError`` raised *by* the callable propagates. It used to be caught
    and the call repeated without keywords, which ran the function twice and
    reported its own bug as "does not accept ``system``".
    """

    name = "callable"

    def __init__(self, fn: Callable[..., str], *, model: str = "callable", **kw: Any):
        super().__init__(model, **kw)
        if not callable(fn):
            raise TypeError("CallableProvider requires a callable")
        self._fn = fn

    def complete(self, prompt: str, **overrides: Any) -> str:
        accepted = _accepted_keywords(self._fn)
        if accepted is None:
            passed = overrides
        else:
            passed = {k: v for k, v in overrides.items() if k in accepted}
            if "system" in overrides and "system" not in passed:
                logger.warning(
                    "CallableProvider: the wrapped callable does not accept a "
                    "`system` keyword, so the system/rubric text for this call "
                    "was dropped and the completion was generated without it. "
                    "Give the callable a `system=None` parameter (or `**kwargs`) "
                    "to receive it -- see CompletionFn in adapt_agent.optimization.judge."
                )
        result = self._fn(prompt, **passed)
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

    What is sent is decided by the installed SDK: a parameter is only included
    when ``client.messages.create`` accepts it. Current SDKs take no sampling
    parameters at all, and the current models reject them, so none are sent.

    Args:
        sdk_max_retries: Retry count for the SDK's own retry layer (default
            ``0``; see :func:`_configure_sdk_retries`). Applied to an injected
            client too. ``None`` leaves the client as configured.
    """

    name = "anthropic"
    max_temperature = 1.0
    #: Current models always think, and thinking counts against
    #: ``max_tokens``; a small budget truncates the answer.
    default_max_tokens = 16000

    def __init__(
        self,
        model: str = ANTHROPIC_DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        client: Any = None,
        sdk_max_retries: int | None = 0,
        **kw: Any,
    ):
        super().__init__(model, **kw)
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._sdk_max_retries = sdk_max_retries
        self._client = None if client is None else _configure_sdk_retries(client, sdk_max_retries)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic  # lazy: importing this module never needs the SDK
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise ImportError(
                    "AnthropicProvider requires the 'anthropic' package: " "pip install anthropic"
                ) from exc
            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._sdk_max_retries is not None:
                kwargs["max_retries"] = self._sdk_max_retries
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def _request(self, prompt: str, overrides: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        cfg = self._resolved(overrides)
        create = self._get_client().messages.create
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": cfg["max_tokens"],
            "messages": [{"role": "user", "content": prompt}],
            **self._sampling(overrides, create),
        }
        if cfg["system"]:
            kwargs["system"] = cfg["system"]
        return create, kwargs

    def complete(self, prompt: str, **overrides: Any) -> str:
        create, kwargs = self._request(prompt, overrides)
        return self._finish(self._call_remembering(create, kwargs))

    def complete_json(self, prompt: str, schema: dict[str, Any], **overrides: Any) -> str:
        create, kwargs = self._request(prompt, overrides)
        if not _accepts_keyword(create, "output_config"):
            raise StructuredOutputUnsupported(
                "the installed anthropic SDK's messages.create has no output_config"
            )
        kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        return self._finish(self._call_remembering(create, kwargs))

    def _finish(self, message: Any) -> str:
        """Check the stop reason, record usage, and return the text."""
        usage = getattr(message, "usage", None)
        if usage is not None:
            self._record_usage(
                getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0)
            )
        stop = getattr(message, "stop_reason", None)
        text = _join_text_blocks(getattr(message, "content", message))
        if stop == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None)
            raise self._response_error(
                RefusedResponseError,
                f"declined to answer (stop_reason=refusal, category={category!r})",
                stop_reason=stop,
                partial_text=text,
            )
        if stop in ("max_tokens", "model_context_window_exceeded"):
            raise self._response_error(
                TruncatedResponseError,
                f"stopped at its token limit (stop_reason={stop}, max_tokens="
                f"{self.max_tokens}) before finishing; raise max_tokens",
                stop_reason=stop,
                partial_text=text,
            )
        if isinstance(stop, str) and not text.strip():
            raise self._response_error(
                EmptyResponseError,
                f"returned no text (stop_reason={stop})",
                stop_reason=stop,
            )
        return text


class OpenAIProvider(ModelProvider):
    """OpenAI Chat Completions provider (SDK imported lazily on first call).

    Requires the ``openai`` package and an API key (argument or
    ``OPENAI_API_KEY``). A pre-built ``client`` may be injected.

    Chat Completions rather than the Responses API on purpose: this class is
    also the base of every OpenAI-*compatible* gateway here (Groq, Together,
    OpenRouter, Ollama), which implement Chat Completions and not, in general,
    Responses. One request shape keeps all of them on one tested path.

    The output-token limit is sent as ``max_completion_tokens`` where the SDK
    accepts it (``max_tokens`` is deprecated there and rejected by reasoning
    models); OpenAI-compatible gateways keep ``max_tokens``, which is what they
    document. A 400 rejecting either name, or a sampling parameter, is
    recovered once and remembered for this instance.

    Args:
        sdk_max_retries: Retry count for the SDK's own retry layer (default
            ``0``; see :func:`_configure_sdk_retries`). ``None`` leaves an
            injected client as configured.
    """

    name = "openai"
    max_temperature = 2.0
    #: Reasoning models spend output tokens thinking before they answer.
    default_max_tokens = 16000
    #: Env var consulted for the API key (overridden by OpenAI-compatible subclasses).
    api_key_env = "OPENAI_API_KEY"
    #: Default base URL (``None`` -> the SDK default). Subclasses point this at
    #: OpenAI-compatible gateways (Groq, Together, Ollama, OpenRouter, ...).
    default_base_url: str | None = None
    #: Preferred name for the output-token limit.
    token_limit_param = "max_completion_tokens"
    #: Whether the endpoint implements ``response_format={"type": "json_schema"}``.
    supports_json_schema = True

    def __init__(
        self,
        model: str = OPENAI_DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
        sdk_max_retries: int | None = 0,
        **kw: Any,
    ):
        super().__init__(model, **kw)
        self._api_key = api_key or os.environ.get(self.api_key_env)
        self._base_url = base_url or self.default_base_url
        self._sdk_max_retries = sdk_max_retries
        self._client = None if client is None else _configure_sdk_retries(client, sdk_max_retries)

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self._sdk_max_retries is not None:
            kwargs["max_retries"] = self._sdk_max_retries
        return kwargs

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai  # lazy
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise ImportError(
                    "OpenAIProvider requires the 'openai' package: pip install openai"
                ) from exc
            kwargs: dict[str, Any] = {"api_key": self._api_key, **self._client_kwargs()}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def _token_key(self, create: Any) -> str:
        if self._token_key_override is not None:
            return self._token_key_override
        if self.token_limit_param != "max_tokens" and _accepts_keyword(
            create, self.token_limit_param
        ):
            return self.token_limit_param
        return "max_tokens"

    def _request(self, prompt: str, overrides: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        cfg = self._resolved(overrides)
        create = self._get_client().chat.completions.create
        messages = []
        if cfg["system"]:
            messages.append({"role": "system", "content": cfg["system"]})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self._token_key(create): cfg["max_tokens"],
            **self._sampling(overrides, create),
        }
        return create, kwargs

    def _send(self, create: Any, kwargs: dict[str, Any]) -> str:
        response = self._call_remembering(
            create, kwargs, token_keys=("max_completion_tokens", "max_tokens")
        )
        return self._finish(response)

    def complete(self, prompt: str, **overrides: Any) -> str:
        create, kwargs = self._request(prompt, overrides)
        return self._send(create, kwargs)

    def complete_json(self, prompt: str, schema: dict[str, Any], **overrides: Any) -> str:
        create, kwargs = self._request(prompt, overrides)
        if not self.supports_json_schema or not _accepts_keyword(create, "response_format"):
            raise StructuredOutputUnsupported(
                f"{type(self).__name__} does not use response_format json_schema"
            )
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "result", "schema": schema, "strict": True},
        }
        return self._send(create, kwargs)

    def _finish(self, response: Any) -> str:
        """Check the finish reason, record usage, and return the text."""
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._record_usage(
                getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)
            )
        choice = response.choices[0]
        finish = getattr(choice, "finish_reason", None)
        message = choice.message
        content = getattr(message, "content", None)
        refusal = getattr(message, "refusal", None)
        if isinstance(refusal, str) and refusal:
            raise self._response_error(
                RefusedResponseError,
                f"declined to answer (finish_reason={finish})",
                stop_reason=finish,
                partial_text=refusal,
            )
        if finish == "content_filter":
            raise self._response_error(
                RefusedResponseError,
                "output was withheld by the content filter (finish_reason=content_filter)",
                stop_reason=finish,
                partial_text=content if isinstance(content, str) else "",
            )
        if finish == "length":
            raise self._response_error(
                TruncatedResponseError,
                "stopped at its token limit (finish_reason=length) before finishing; "
                "raise max_tokens",
                stop_reason=finish,
                partial_text=content if isinstance(content, str) else "",
            )
        if content is None:
            raise self._response_error(
                EmptyResponseError, f"returned no text (finish_reason={finish})", stop_reason=finish
            )
        return content if isinstance(content, str) else str(content)


class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI provider (uses ``openai.AzureOpenAI``, imported lazily).

    ``model`` is the Azure *deployment* name. Endpoint/version come from
    arguments or ``AZURE_OPENAI_ENDPOINT`` / ``OPENAI_API_VERSION``.

    There is no built-in default ``api_version``: the SDK ships none and a
    hard-coded date goes stale silently. Without one the SDK raises a clear
    error naming ``OPENAI_API_VERSION`` when the client is first built.
    """

    name = "azure_openai"
    api_key_env = "AZURE_OPENAI_API_KEY"
    #: Depends on the deployment's model and api_version; not assumed.
    supports_json_schema = False

    def __init__(
        self,
        model: str = AZURE_OPENAI_DEFAULT_DEPLOYMENT,
        *,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        client: Any = None,
        **kw: Any,
    ):
        super().__init__(model, api_key=api_key, client=client, **kw)
        self._azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self._api_version = api_version or os.environ.get("OPENAI_API_VERSION")

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai  # lazy
            except ImportError as exc:  # pragma: no cover
                raise ImportError("AzureOpenAIProvider requires 'openai'") from exc
            kwargs: dict[str, Any] = {
                "api_key": self._api_key,
                "api_version": self._api_version,
                **self._client_kwargs(),
            }
            if self._azure_endpoint:
                kwargs["azure_endpoint"] = self._azure_endpoint
            self._client = openai.AzureOpenAI(**kwargs)
        return self._client


class _OpenAICompatibleProvider(OpenAIProvider):
    """An OpenAI-compatible gateway: documented request shape only.

    Gateways document ``max_tokens`` and a 1024-token default was their
    long-standing behaviour here; structured output is not assumed.
    """

    token_limit_param = "max_tokens"
    supports_json_schema = False
    default_max_tokens = 1024


class GroqProvider(_OpenAICompatibleProvider):
    """Groq (OpenAI-compatible API), SDK imported lazily."""

    name = "groq"
    api_key_env = "GROQ_API_KEY"
    default_base_url = "https://api.groq.com/openai/v1"

    def __init__(self, model: str = GROQ_DEFAULT_MODEL, **kw: Any):
        super().__init__(model, **kw)


class TogetherProvider(_OpenAICompatibleProvider):
    """Together AI (OpenAI-compatible API), SDK imported lazily."""

    name = "together"
    api_key_env = "TOGETHER_API_KEY"
    default_base_url = "https://api.together.xyz/v1"

    def __init__(self, model: str = TOGETHER_DEFAULT_MODEL, **kw: Any):
        super().__init__(model, **kw)


class OpenRouterProvider(_OpenAICompatibleProvider):
    """OpenRouter (OpenAI-compatible API), SDK imported lazily."""

    name = "openrouter"
    api_key_env = "OPENROUTER_API_KEY"
    default_base_url = "https://openrouter.ai/api/v1"

    def __init__(self, model: str = OPENROUTER_DEFAULT_MODEL, **kw: Any):
        super().__init__(model, **kw)


class OllamaProvider(_OpenAICompatibleProvider):
    """Local Ollama server via its OpenAI-compatible endpoint.

    Defaults to ``http://localhost:11434/v1`` and a placeholder key (Ollama
    ignores auth). Override ``base_url`` for a remote host.
    """

    name = "ollama"
    api_key_env = "OLLAMA_API_KEY"
    default_base_url = "http://localhost:11434/v1"

    def __init__(self, model: str = OLLAMA_DEFAULT_MODEL, *, api_key: str | None = None, **kw: Any):
        super().__init__(model, api_key=api_key or "ollama", **kw)


#: Gemini finish reasons meaning the vendor withheld or blocked the output.
_GEMINI_BLOCKED = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "LANGUAGE",
    }
)


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "value", value)
    return name if isinstance(name, str) else str(name)


class GeminiProvider(ModelProvider):
    """Google Gemini provider (uses the ``google-genai`` SDK, imported lazily).

    Falls back to the legacy ``google-generativeai`` package if ``google-genai``
    is unavailable. API key from argument or ``GEMINI_API_KEY`` /
    ``GOOGLE_API_KEY``.

    The system text travels in ``config.system_instruction``, never
    concatenated into ``contents``: the judge relies on the system/data split
    to keep grading instructions apart from the untrusted text being graded.
    (The legacy package, which is not maintained, keeps its old behaviour.)
    google-genai does not retry unless ``http_options.retry_options`` is set,
    so no retry layer stacks under the library's own.
    """

    name = "gemini"

    def __init__(
        self,
        model: str = GEMINI_DEFAULT_MODEL,
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

    def _config(self, overrides: dict[str, Any]) -> dict[str, Any]:
        cfg = self._resolved(overrides)
        sampling = self._sampling(overrides)
        config: dict[str, Any] = {"max_output_tokens": cfg["max_tokens"]}
        if cfg["system"]:
            config["system_instruction"] = cfg["system"]
        if "temperature" in sampling:
            config["temperature"] = sampling["temperature"]
        if "top_p" in sampling:
            config["top_p"] = sampling["top_p"]
        return config

    def complete(self, prompt: str, **overrides: Any) -> str:
        client = self._get_client()
        if self._legacy:  # google.generativeai: not installed in CI, kept as it was
            cfg = self._resolved(overrides)
            full_prompt = f"{cfg['system']}\n\n{prompt}" if cfg["system"] else prompt
            model = client.GenerativeModel(self.model)
            response = model.generate_content(full_prompt)
            return getattr(response, "text", "") or ""
        return self._generate(client, prompt, self._config(overrides))

    def complete_json(self, prompt: str, schema: dict[str, Any], **overrides: Any) -> str:
        client = self._get_client()
        if self._legacy:
            raise StructuredOutputUnsupported("legacy google-generativeai path")
        config = self._config(overrides)
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = schema
        return self._generate(client, prompt, config)

    def _generate(self, client: Any, prompt: str, config: dict[str, Any]) -> str:
        response = self._call_remembering(
            lambda **kw: client.models.generate_content(model=self.model, **kw),
            {"contents": prompt, "config": config},
        )
        return self._finish(response)

    def _finish(self, response: Any) -> str:
        """Check block / finish reasons, record usage, and return the text."""
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            output = _int(getattr(usage, "candidates_token_count", 0)) + _int(
                getattr(usage, "thoughts_token_count", 0)
            )
            self._record_usage(getattr(usage, "prompt_token_count", 0), output)
        feedback = getattr(response, "prompt_feedback", None)
        blocked = _enum_name(getattr(feedback, "block_reason", None))
        if blocked:
            raise self._response_error(
                RefusedResponseError,
                f"blocked the prompt (block_reason={blocked})",
                stop_reason=blocked,
            )
        candidates = getattr(response, "candidates", None)
        finish = None
        if isinstance(candidates, list) and candidates:
            finish = _enum_name(getattr(candidates[0], "finish_reason", None))
        text = getattr(response, "text", None)
        text = text if isinstance(text, str) else ""
        if finish in _GEMINI_BLOCKED:
            raise self._response_error(
                RefusedResponseError,
                f"withheld its output (finish_reason={finish})",
                stop_reason=finish,
                partial_text=text,
            )
        if finish == "MAX_TOKENS":
            raise self._response_error(
                TruncatedResponseError,
                "stopped at its token limit (finish_reason=MAX_TOKENS) before finishing; "
                "raise max_tokens",
                stop_reason=finish,
                partial_text=text,
            )
        if not text:
            raise self._response_error(
                EmptyResponseError, f"returned no text (finish_reason={finish})", stop_reason=finish
            )
        return text


class MistralProvider(ModelProvider):
    """Mistral provider (uses the ``mistralai`` SDK, imported lazily).

    The SDK is not verified against this code (it is not a test dependency),
    so the request and response shapes are unchanged and no stop reason is read.
    """

    name = "mistral"

    def __init__(
        self,
        model: str = MISTRAL_DEFAULT_MODEL,
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
        sampling = self._sampling(overrides)
        client = self._get_client()
        messages = []
        if cfg["system"]:
            messages.append({"role": "system", "content": cfg["system"]})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": cfg["max_tokens"],
            **sampling,
        }
        response = self._call_remembering(client.chat.complete, kwargs)
        return response.choices[0].message.content or ""


class CohereProvider(ModelProvider):
    """Cohere provider (uses the ``cohere`` SDK v2, imported lazily).

    The SDK is not verified against this code (it is not a test dependency),
    so the request and response shapes are unchanged and no stop reason is read.
    """

    name = "cohere"
    max_temperature = 1.0

    def __init__(
        self,
        model: str = COHERE_DEFAULT_MODEL,
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
        sampling = self._sampling(overrides)
        client = self._get_client()
        messages = []
        if cfg["system"]:
            messages.append({"role": "system", "content": cfg["system"]})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, **sampling}
        response = self._call_remembering(client.chat, kwargs)
        return _join_text_blocks(getattr(response.message, "content", ""))


class BedrockProvider(ModelProvider):
    """AWS Bedrock provider via the Converse API (uses ``boto3``, imported lazily).

    ``boto3`` is not verified against this code (it is not a test dependency),
    so the request and response shapes are unchanged and no stop reason is read.
    """

    name = "bedrock"
    max_temperature = 1.0

    def __init__(
        self,
        model: str = BEDROCK_DEFAULT_MODEL,
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
        sampling = self._sampling(overrides)
        client = self._get_client()
        inference_config: dict[str, Any] = {"maxTokens": cfg["max_tokens"]}
        if "temperature" in sampling:
            inference_config["temperature"] = sampling["temperature"]
        if "top_p" in sampling:
            inference_config["topP"] = sampling["top_p"]
        kwargs: dict[str, Any] = {
            "modelId": self.model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": inference_config,
        }
        if cfg["system"]:
            kwargs["system"] = [{"text": cfg["system"]}]

        def _converse(**call_kwargs: Any) -> Any:
            # Bedrock nests sampling under inferenceConfig; on a sampling-related
            # 400 strip those keys (keeping maxTokens) and retry once.
            try:
                return client.converse(**call_kwargs)
            except Exception as exc:  # noqa: BLE001
                if not _sampling_error(exc):
                    raise
                cfg_only = {"maxTokens": call_kwargs["inferenceConfig"].get("maxTokens")}
                logger.warning("Bedrock rejected sampling params (%s); retrying without them.", exc)
                self._sampling_rejected = True
                retry = {**call_kwargs, "inferenceConfig": cfg_only}
                return client.converse(**retry)

        response = _converse(**kwargs)
        blocks = response["output"]["message"]["content"]
        return "".join(b.get("text", "") for b in blocks)


class HuggingFaceProvider(ModelProvider):
    """Hugging Face Inference provider (uses ``huggingface_hub``, imported lazily)."""

    name = "huggingface"

    def __init__(
        self,
        model: str = HUGGINGFACE_DEFAULT_MODEL,
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
        sampling = self._sampling(overrides, client.chat_completion)
        messages = []
        if cfg["system"]:
            messages.append({"role": "system", "content": cfg["system"]})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": cfg["max_tokens"], **sampling}
        response = self._call_remembering(
            lambda **kw: client.chat_completion(messages, **kw), kwargs
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
    """Concatenate the *text* blocks of an Anthropic-style content block list.

    Only blocks of ``type == "text"`` count (or, for a block without a
    ``type``, one carrying a string ``text``). Thinking, redacted-thinking,
    tool-use and server-tool blocks are skipped: none of them is the answer,
    and a tool result block can carry a ``text`` field of its own.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                kind = block.get("type")
                text = block.get("text")
            else:
                kind = getattr(block, "type", None)
                text = getattr(block, "text", None)
            if kind is not None and kind != "text":
                continue
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return str(content)


__all__ = [
    "ModelProvider",
    "CallableProvider",
    "EchoProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "ProviderResponseError",
    "TruncatedResponseError",
    "RefusedResponseError",
    "EmptyResponseError",
    "StructuredOutputUnsupported",
    "TokenUsage",
    "DEFAULT_MODELS",
    "register_provider",
    "available_providers",
    "get_provider",
    "as_provider",
]
