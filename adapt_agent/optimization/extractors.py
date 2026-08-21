"""Framework-native output extraction for evals.

Agent frameworks rarely return plain strings: LangGraph's ``invoke`` returns the
final state mapping, Pydantic AI's ``run_sync`` an ``AgentRunResult``, Microsoft
Agent Framework's ``run`` an ``AgentResponse``, a Google ADK ``Runner`` a
stream of events, CrewAI a ``CrewOutput``, and so on. Text-level checks
(``exact_match``, ``contains``, ``numeric_close``, an LLM-as-judge) need the
*final response text* out of those shapes -- otherwise they end up comparing the
expected answer against ``repr(AgentRunResult(...))``.

:func:`extract_output_text` performs that extraction. Following the design rules
of the introspection package it is **structural and dependency-free**: no
framework is ever imported, every shape is duck-typed via ``getattr``. Known
shapes are unwrapped recursively (bounded depth, so cyclic objects terminate);
values that are not recognised are returned **unchanged**, so pipelines scoring
structured outputs (e.g. ``json_subset`` over a dict) are not disturbed.

Recognised shapes include:

* Pydantic AI ``AgentRunResult`` (``.output`` / legacy ``.data``)
* OpenAI Agents SDK ``RunResult`` (``.final_output``)
* CrewAI ``CrewOutput`` (``.raw``)
* Claude Agent SDK ``ResultMessage`` (``.result``) and content-block messages
* Microsoft Agent Framework ``AgentResponse`` (``AgentRunResponse`` before
  1.14) / ``ChatResponse``
  (``.text`` / ``.messages``)
* Google ADK / GenAI events and ``Content`` objects (``.content.parts[*].text``)
* LangChain / chat messages (``.content`` as text or content-part lists)
* Mappings with conventional keys (``output`` / ``response`` / ``messages`` /
  ``content`` / ...), e.g. a LangGraph final state
* Message / event *streams* (lists are scanned from the end for the final
  extractable text -- how ADK events and drained async generators arrive)

Extraction is extensible: :func:`register_extractor` installs a
``(predicate, unwrap)`` pair tried *before* the built-ins, so an unsupported
framework can be plugged in without touching this module.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

#: Decides whether an unwrapper applies to a value.
ExtractorPredicate = Callable[[Any], bool]

#: Unwraps one layer of a recognised value. Returning ``None`` means "could not
#: unwrap" and the next registered extractor is tried.
ExtractorFn = Callable[[Any], Any]

#: Maximum unwrap depth. Generous for real framework nesting (a state mapping ->
#: message list -> message -> content parts -> text is depth five) while
#: guaranteeing termination on cyclic or degenerate structures.
_MAX_DEPTH = 8

#: Conventional mapping keys carrying the final output, tried in order.
_MAPPING_KEYS = (
    "output",
    "final_output",
    "output_text",
    "response",
    "answer",
    "result",
    "text",
    "content",
    "completion",
    "reply",
    "messages",
    "message",
)

#: Where LangGraph's ``create_react_agent(response_format=...)`` puts the
#: declared structured output -- see ``AgentStateWithStructuredResponse``.
#: Attribute marking a governed adapter's own ``{"result": ...}`` wrapper. See
#: ``adapt_agent.adapters._governed.GovernedEnvelope``: the shape alone cannot
#: say whether such a mapping is a wrapper or a one-field answer, so the
#: producer says. Duck-typed, so this module imports no adapter.
_GOVERNED_ENVELOPE_MARKER = "__adapt_governed_envelope__"

_STRUCTURED_STATE_KEY = "structured_response"
_MESSAGES_KEY = "messages"

#: Conventional attribute names carrying the final output, tried in order.
_ATTR_NAMES = (
    "output_text",
    "final_output",
    "output",
    "text",
    "raw",
    "content",
    "response",
    "answer",
    "result",
    "data",
    "message",
)


# -- built-in unwrappers -------------------------------------------------------


def _is_pydantic_ai_result(value: Any) -> bool:
    """Pydantic AI ``AgentRunResult``: run output plus a message-history API."""
    if not (hasattr(value, "output") or hasattr(value, "data")):
        return False
    return callable(getattr(value, "all_messages", None)) or callable(
        getattr(value, "new_messages", None)
    )


def _unwrap_pydantic_ai_result(value: Any) -> Any:
    if hasattr(value, "output"):
        return getattr(value, "output", None)
    return getattr(value, "data", None)


def _is_openai_agents_result(value: Any) -> bool:
    """OpenAI Agents SDK ``RunResult``: carries ``final_output``."""
    return hasattr(value, "final_output")


def _unwrap_openai_agents_result(value: Any) -> Any:
    return getattr(value, "final_output", None)


def _is_crewai_output(value: Any) -> bool:
    """CrewAI ``CrewOutput``: raw text plus per-task outputs."""
    return hasattr(value, "raw") and hasattr(value, "tasks_output")


def _unwrap_crewai_output(value: Any) -> Any:
    return getattr(value, "raw", None)


def _is_claude_result_message(value: Any) -> bool:
    """Claude Agent SDK ``ResultMessage``: final ``result`` text + ``subtype``."""
    if not hasattr(value, "subtype"):
        return False
    result = getattr(value, "result", None)
    return isinstance(result, str)


def _unwrap_claude_result_message(value: Any) -> Any:
    return getattr(value, "result", None)


def _is_maf_response(value: Any) -> bool:
    """Microsoft Agent Framework ``AgentResponse`` / ``ChatResponse``.

    ``AgentResponse`` was called ``AgentRunResponse`` before agent-framework
    1.14; both are matched, because the check is on shape, not name.

    Both carry a ``messages`` list and a ``text`` convenience property that
    concatenates the text content of the messages.
    """
    if not (hasattr(value, "messages") and hasattr(value, "text")):
        return False
    return not callable(getattr(value, "text", None))


def _unwrap_maf_response(value: Any) -> Any:
    text = getattr(value, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    return getattr(value, "messages", None)


def _is_chat_message(value: Any) -> bool:
    """A chat message: ``content`` plus a ``role``/``type`` discriminator.

    Covers LangChain ``BaseMessage`` (``content`` + ``type``), Anthropic SDK
    messages (``content`` + ``role``), and similar shapes.
    """
    if not hasattr(value, "content"):
        return False
    return hasattr(value, "role") or hasattr(value, "type")


def _unwrap_chat_message(value: Any) -> Any:
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        return _parts_text(content)
    return content


def _is_adk_event(value: Any) -> bool:
    """Google ADK ``Event``: wraps a GenAI ``Content`` under ``.content``."""
    content = getattr(value, "content", None)
    return content is not None and hasattr(content, "parts")


def _unwrap_adk_event(value: Any) -> Any:
    return getattr(value, "content", None)


def _is_content_parts(value: Any) -> bool:
    """A GenAI ``Content``-like object: a ``parts`` list of text-bearing parts."""
    parts = getattr(value, "parts", None)
    return isinstance(parts, Sequence) and not isinstance(parts, (str, bytes))


def _unwrap_content_parts(value: Any) -> Any:
    return _parts_text(getattr(value, "parts", ()))


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _unwrap_mapping(value: Any) -> Any:
    for key in _MAPPING_KEYS:
        if key not in value:
            continue
        candidate = value[key]
        if candidate is None or (isinstance(candidate, str) and not candidate):
            continue
        return candidate
    return None


def _is_object(value: Any) -> bool:
    return True


def _unwrap_object_attrs(value: Any) -> Any:
    for name in _ATTR_NAMES:
        try:
            candidate = getattr(value, name, None)
        except Exception:
            continue
        if candidate is None or callable(candidate):
            continue
        if isinstance(candidate, str) and not candidate:
            continue
        return candidate
    return None


def _parts_text(parts: Sequence[Any]) -> str | None:
    """Join the text carried by content parts (dicts, part objects, or strings).

    Returns ``None`` when no part carries text, so callers can fall through to
    other extraction strategies (e.g. a parts list of pure function calls).
    """
    texts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            text: Any = part
        elif isinstance(part, Mapping):
            text = part.get("text")
        else:
            text = getattr(part, "text", None)
        if isinstance(text, str) and text:
            texts.append(text)
    if not texts:
        return None
    return "\n".join(texts)


# Built-in entries, most-specific first. Each is (name, predicate, unwrap); an
# unwrap returning ``None`` means "no extraction" and the next entry is tried.
_BUILTIN_EXTRACTORS: list[tuple[str, ExtractorPredicate, ExtractorFn]] = [
    ("pydantic_ai_result", _is_pydantic_ai_result, _unwrap_pydantic_ai_result),
    ("openai_agents_result", _is_openai_agents_result, _unwrap_openai_agents_result),
    ("crewai_output", _is_crewai_output, _unwrap_crewai_output),
    ("claude_result_message", _is_claude_result_message, _unwrap_claude_result_message),
    ("microsoft_agent_framework_response", _is_maf_response, _unwrap_maf_response),
    ("chat_message", _is_chat_message, _unwrap_chat_message),
    ("google_adk_event", _is_adk_event, _unwrap_adk_event),
    ("content_parts", _is_content_parts, _unwrap_content_parts),
    ("mapping_keys", _is_mapping, _unwrap_mapping),
    ("object_attrs", _is_object, _unwrap_object_attrs),
]

#: User-registered entries; always tried before the built-ins.
_CUSTOM_EXTRACTORS: list[tuple[str, ExtractorPredicate, ExtractorFn]] = []


def register_extractor(name: str, predicate: ExtractorPredicate, unwrap: ExtractorFn) -> None:
    """Register a custom output extractor tried before the built-ins.

    Args:
        name: Stable identifier. Re-registering an existing name replaces it.
        predicate: ``value -> bool`` deciding whether ``unwrap`` applies.
        unwrap: ``value -> inner`` removing one wrapper layer. Return ``None``
            to decline (the next extractor is tried); any other value is
            extracted recursively.
    """
    global _CUSTOM_EXTRACTORS
    _CUSTOM_EXTRACTORS = [entry for entry in _CUSTOM_EXTRACTORS if entry[0] != name]
    _CUSTOM_EXTRACTORS.append((name, predicate, unwrap))


def available_extractors() -> list[str]:
    """Return the names of all registered extractors (custom first)."""
    return [name for name, _, _ in (*_CUSTOM_EXTRACTORS, *_BUILTIN_EXTRACTORS)]


def extract_output_text(value: Any) -> Any:
    """Extract the final response text from a framework-native run result.

    Best-effort and total: when ``value`` (or something nested inside it) is a
    recognised framework shape the extracted text is returned; a value that is
    already a string comes back as-is; anything unrecognised is returned
    **unchanged** (never stringified), so structured outputs still reach metrics
    like ``json_subset`` intact. ``None`` becomes ``""`` (no output).

    Lists and tuples are treated as message/event streams and scanned from the
    end for the last element yielding non-empty text -- matching how Google ADK
    event streams and drained async generators (e.g. Claude Agent SDK messages)
    present the final response.
    """
    return _extract(value, _MAX_DEPTH)


def extract_output_payload(value: Any) -> Any:
    """Unwrap the framework envelope but **keep the structure**.

    :func:`extract_output_text` is the right default for text and number checks,
    but it flattens a structured answer: a Microsoft ``AgentResponse`` is a
    recognised shape, so it becomes ``.text`` and the object is gone. The only
    escape used to be ``output_extractor=None``, which unwraps nothing at all
    and leaves a metric scoring a ``repr()``. Neither is what a structured-output
    agent needs.

    This extractor takes the middle path -- strip the framework wrapper, then
    recover the payload:

    * a mapping or sequence comes back unchanged;
    * a declared structured output -- a Pydantic model or a dataclass -- becomes
      a dict, so the field names survive for `field_match` even when one of them
      is called ``answer`` or ``result``; framework wrappers are recognised
      first, because several (Pydantic AI's ``AgentRunResult``, the OpenAI SDK's
      ``RunResult``) are themselves dataclasses;
    * a JSON string is parsed back into an object, including one wrapped in a
      `````json`` fence, which is how most models emit structured answers;
    * anything that is not JSON is returned as the extracted text, so a
      partially-structured pipeline degrades to text rather than to an error.

    Use it with :func:`~adapt_agent.optimization.metrics.json_subset` or
    :func:`~adapt_agent.optimization.metrics.field_match`;
    :func:`~adapt_agent.optimization.evals.evaluate_agent` selects it
    automatically when the metrics you asked for are structural.
    """
    unwrapped = _unwrap_envelope(value, _MAX_DEPTH)
    if isinstance(unwrapped, Mapping) or (
        isinstance(unwrapped, Sequence) and not isinstance(unwrapped, (str, bytes))
    ):
        return unwrapped
    as_dict = _model_to_dict(unwrapped)
    if as_dict is not None:
        return as_dict

    text = extract_output_text(unwrapped)
    if isinstance(text, str):
        parsed = _parse_json(text)
        if parsed is not None:
            return parsed
    return text


def _unwrap_envelope(value: Any, depth: int) -> Any:
    """Peel framework wrappers off ``value`` without collapsing it to text.

    Applies the same registered extractors as :func:`extract_output_text`, but
    stops as soon as the inner value is structured rather than continuing down
    to a string.
    """
    if value is None or isinstance(value, (str, bytes)) or depth <= 0:
        return value
    if isinstance(value, Mapping):
        # A governed adapter marks its own wrapper, so the ambiguity below does
        # not arise for one: `{"result": [...]}` from `execute` is definitively
        # an envelope, while the identical shape from an agent is definitively
        # an answer. Nothing is imported -- the marker is duck-typed.
        if getattr(value, _GOVERNED_ENVELOPE_MARKER, False) and len(value) == 1:
            (key,) = value.keys()
            return _unwrap_envelope(value[key], depth - 1)
        # A governed adapter returns an envelope -- ``execute`` wraps a non-dict
        # framework result as ``{"result": <payload>}``. Returning that as the
        # payload makes every structural metric score 0.0 against the real
        # fields. Peel a *single* conventional key, and only when what it holds
        # is itself structured: an envelope wraps a payload, whereas a genuine
        # one-field answer like ``{"result": "granted"}`` or ``{"answer":
        # "Paris"}`` holds a scalar and must survive intact, or `field_match`
        # would find its field gone.
        # LangGraph is the exception to the multi-key rule: a graph built with
        # ``response_format=`` returns a *state* -- ``{"messages": [...],
        # "structured_response": <the answer>}`` -- so the single-key test below
        # never fires and every field metric would score 0.0 against the state
        # rather than the answer inside it. The state must be recognised as one,
        # not merely carry the key: on its own ``structured_response`` is an
        # ordinary field name, and peeling ``{"structured_response": {...},
        # "request_id": "42"}`` would lose the sibling columns -- the mistake the
        # multi-key rule exists to avoid.
        if _is_graph_state(value):
            declared = value.get(_STRUCTURED_STATE_KEY)
            if declared is not None and not _is_scalar(declared):
                return _unwrap_envelope(declared, depth - 1)
        if len(value) == 1:
            (key,) = value.keys()
            inner = value[key]
            if key in _MAPPING_KEYS and not _is_scalar(inner):
                # An envelope wraps something that still needs unwrapping. If
                # the value inside is already the final payload, the mapping is
                # a one-field *answer* and peeling it deletes the column a
                # metric scores. This matches what the governed adapter
                # actually builds: `execute` returns a dict result untouched
                # and wraps only a non-dict, so `{"result": {...}}` is never an
                # envelope, and `{"result": ["A", "B"]}` is one only when those
                # elements are framework messages rather than plain values.
                peeled = _unwrap_envelope(inner, depth - 1)
                if peeled is not inner:
                    return peeled
        return value
    if isinstance(value, Sequence):
        return _unwrap_stream(value, depth)
    for _, predicate, unwrap in (*_CUSTOM_EXTRACTORS, *_BUILTIN_EXTRACTORS):
        if unwrap is _unwrap_object_attrs:
            # The catch-all matches everything, so it runs last -- after the
            # declared-model check below. Order matters in both directions and
            # each side has a real failure: run it first and a dataclass
            # ``Answer(answer="Paris")`` is peeled to ``"Paris"`` by its
            # conventional field name, losing the field `field_match` scores;
            # run the declared-model check ahead of the *specific* extractors
            # instead and framework results are mangled, because several are
            # themselves declared -- Pydantic AI's ``AgentRunResult`` and the
            # OpenAI SDK's ``RunResult`` are dataclasses, so the payload would
            # come back as the wrapper's own fields (``output``, ``_state``,
            # ...) rather than the answer inside it.
            continue
        try:
            if not predicate(value):
                continue
            inner = unwrap(value)
        except Exception:
            continue
        if inner is None or inner is value:
            continue
        return _unwrap_envelope(inner, depth - 1)
    # No framework wrapper recognised. A *declared* structured output -- a
    # Pydantic model or dataclass -- is the payload itself, so convert it
    # rather than letting the catch-all peel one of its fields.
    declared = _declared_model_to_dict(value)
    if declared is not None:
        return declared
    inner = _unwrap_object_attrs(value)
    if inner is not None and inner is not value:
        return _unwrap_envelope(inner, depth - 1)
    return value


def _unwrap_stream(value: Sequence[Any], depth: int) -> Any:
    """Find the final payload in a drained message/event stream.

    An async framework's result is materialised into a *list* -- a Claude Agent
    SDK message stream, a Google ADK event stream. Returning that list as the
    payload makes every structural metric score 0.0 against the real fields,
    because the answer is still inside the last message.

    Scanned from the end, like :func:`_extract_sequence`, and only elements that
    a registered extractor actually *recognises* count: a genuine structured
    list (a list of records the dataset expects) contains nothing recognisable
    and is returned unchanged.
    """
    for item in reversed(value):
        if item is None or isinstance(item, (str, bytes, int, float, bool)):
            continue
        if _is_declared_model(item) and not _is_recognised_wrapper(item):
            # A list of declared *records* is the answer, not a stream with the
            # answer in its last element. Converting one and returning it would
            # silently drop every other row.
            return value
        unwrapped = _unwrap_envelope(item, depth - 1)
        if unwrapped is not item and unwrapped is not None and unwrapped != "":
            return unwrapped
    return value


def _is_scalar(value: Any) -> bool:
    """Values that cannot themselves be a wrapped payload."""
    return value is None or isinstance(value, (str, bytes, int, float, bool))


def _is_recognised_wrapper(value: Any) -> bool:
    """Whether a *specific* framework extractor claims ``value``.

    Excludes the generic attribute catch-all, which matches everything. Used to
    tell a framework object that happens to be declared (a CrewAI ``CrewOutput``
    is a Pydantic model) from a plain record of the same kind.
    """
    for _, predicate, unwrap in (*_CUSTOM_EXTRACTORS, *_BUILTIN_EXTRACTORS):
        if unwrap is _unwrap_object_attrs:
            continue
        try:
            if predicate(value):
                return True
        except Exception:
            continue
    return False


def _is_graph_state(value: Mapping[Any, Any]) -> bool:
    """Whether ``value`` is a LangGraph state rather than an answer of that shape.

    ``messages`` alone is too weak a signal -- a structured answer may have a
    field of that name. The discriminator is what LangGraph guarantees about it:
    ``add_messages`` coerces every entry to a ``BaseMessage``, so a real state's
    ``messages`` is a non-empty list of *message objects*, never of plain
    strings or records.

    ``remaining_steps`` deliberately plays no part. It is declared
    ``NotRequired`` and managed by ``RemainingStepsManager``, so it is absent
    from the output state -- a real run returns exactly ``['messages',
    'structured_response']``, and requiring it would reject every genuine
    LangGraph result.
    """
    messages = value.get(_MESSAGES_KEY)
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return False
    return bool(messages) and all(_is_chat_message(item) for item in messages)


def _is_declared_model(value: Any) -> bool:
    """Whether ``value`` *advertises* its own fields -- a dataclass or a model.

    Deliberately narrower than "has a ``dict()`` method": a framework result
    object that merely happens to expose one must still go through the
    registered extractors, or unwrapping would stop one layer too early.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return True
    return hasattr(value, "model_fields") or hasattr(value, "__fields__")  # pydantic v2 / v1


def _declared_model_to_dict(value: Any) -> dict[str, Any] | None:
    """Convert an object that declares itself structured, else ``None``."""
    if not _is_declared_model(value):
        return None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return {f.name: getattr(value, f.name, None) for f in dataclasses.fields(value)}
        except Exception:
            return None
    return _model_to_dict(value)


def _model_to_dict(value: Any) -> dict[str, Any] | None:
    """Convert a Pydantic (v1 or v2) model or dataclass-like object to a dict."""
    for attr in ("model_dump", "dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                dumped = method()
            except Exception:
                continue
            if isinstance(dumped, dict):
                return dumped
    return None


def _parse_json(text: str) -> Any | None:
    """Parse ``text`` as JSON, tolerating a ``` fence. ``None`` if it is not JSON."""
    candidate = text.strip()
    if candidate.startswith("```"):
        # ```json\n{...}\n``` -- drop the fence lines and keep the body.
        lines = candidate.splitlines()
        if len(lines) >= 2:
            body = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
            candidate = "\n".join(body).strip()
    if not candidate or candidate[0] not in "{[":
        return None
    try:
        return json.loads(candidate)
    except ValueError:
        return None


def _extract(value: Any, depth: int) -> Any:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if depth <= 0:
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _extract_sequence(value, depth)
    for _, predicate, unwrap in (*_CUSTOM_EXTRACTORS, *_BUILTIN_EXTRACTORS):
        if unwrap is _unwrap_object_attrs:
            continue  # the catch-all runs last, after the declared-model check
        try:
            if not predicate(value):
                continue
            inner = unwrap(value)
        except Exception:
            continue
        if inner is None or inner is value:
            continue
        return _extract(inner, depth - 1)
    # A declared structured output is not a wrapper around text, so it comes
    # back **unchanged** -- the documented behaviour for anything unrecognised,
    # and what lets a per-row `field_match` still see its fields. Only the
    # generic attribute peel would flatten it, and only when a field happens to
    # be *named* like an envelope: `Answer(answer="Paris")` became `"Paris"`
    # while `Triage(lane="NOS")` survived, which is no contract at all. The
    # check sits after the specific extractors because plenty of recognised
    # shapes are declared too -- a LangChain `AIMessage` is a Pydantic model,
    # and it must still be peeled to its content.
    if _is_declared_model(value):
        return value
    inner = _unwrap_object_attrs(value)
    if inner is not None and inner is not value:
        return _extract(inner, depth - 1)
    return value


def _extract_sequence(value: Sequence[Any], depth: int) -> Any:
    """Scan a message/event stream from the end for the final extractable text."""
    if not value:
        return ""
    for item in reversed(value):
        extracted = _extract(item, depth - 1)
        if isinstance(extracted, str) and extracted.strip():
            return extracted
    # No element yielded text: hand the sequence back unchanged so structured
    # outputs (e.g. a list of records scored by a custom metric) survive.
    return value


__all__ = [
    "ExtractorFn",
    "ExtractorPredicate",
    "available_extractors",
    "extract_output_payload",
    "extract_output_text",
    "register_extractor",
]
