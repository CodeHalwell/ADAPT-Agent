"""Optional OpenTelemetry bridge for :class:`~adapt_agent.observability.AgentObserver`.

Every agent framework ADAPT governs now emits OpenTelemetry spans of its own. A
governance layer that keeps its traces only in process memory is invisible in
the tools people actually look at, so the observer can mirror each trace,
refusal and tool verdict into a real span -- a child of whatever span is
current, so a governance span nests under the framework's agent span (or, when
an adapter wraps the framework from outside, the framework's spans nest under
it).

This module imports without ``opentelemetry`` installed: only constructing an
:class:`OTelBridge` needs the API package, and the observer never constructs
one unless asked to. Nothing here is a security control, so a failure inside
the OpenTelemetry pipeline (a broken processor, an exporter that raises) is
counted and logged by the observer rather than allowed to fail the request.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# --------------------------------------------------------------------------- #
# Attribute names
# --------------------------------------------------------------------------- #
# OpenTelemetry GenAI semantic conventions. Checked against the constants in
# ``opentelemetry-semantic-conventions`` 0.66b2
# (``opentelemetry.semconv._incubating.attributes.gen_ai_attributes`` and the
# stable ``error_attributes``); ``tests/test_observability_otel.py`` re-checks
# them whenever that package is importable. The GenAI conventions are still
# "development" status and have renamed attributes before, which is why they
# live in one block and are not imported from the (incubating) package itself.
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_AGENT_ID = "gen_ai.agent.id"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
ERROR_TYPE = "error.type"
#: ``gen_ai.operation.name`` values (``GenAiOperationNameValues``).
OPERATION_INVOKE_AGENT = "invoke_agent"
OPERATION_EXECUTE_TOOL = "execute_tool"
#: The convention's fallback ``error.type`` when no more specific type is known.
ERROR_TYPE_OTHER = "_OTHER"

# Governance-specific attributes. The GenAI conventions have no vocabulary for
# a policy verdict, so these use the library's own namespace.
ADAPT_TRACE_ID = "adapt_agent.trace_id"
ADAPT_OPERATION = "adapt_agent.operation"
ADAPT_STATUS = "adapt_agent.status"
ADAPT_STAGE = "adapt_agent.stage"
ADAPT_DECISION = "adapt_agent.decision"
ADAPT_THREATS = "adapt_agent.threats"
ADAPT_POLICY_RULES = "adapt_agent.policy.rules"
ADAPT_REPORT_ONLY = "adapt_agent.report_only"
ADAPT_DESCRIPTION = "adapt_agent.description"
#: Span event added when a trace or tool span closes carrying threat labels.
EVENT_THREAT_DETECTED = "adapt_agent.threat_detected"

#: ``error.type`` for a run refused by a governance control without an
#: exception object to name (the exception, when there is one, names itself).
ERROR_TYPE_BLOCKED = "blocked"

#: Threat labels of the form ``policy:<rule>`` name the policy rule that fired.
_POLICY_PREFIX = "policy:"


def policy_rules(threats: Sequence[str]) -> list[str]:
    """The policy rule names inside a list of threat labels."""
    return [t[len(_POLICY_PREFIX) :] for t in threats if t.startswith(_POLICY_PREFIX)]


def to_attribute_value(value: Any) -> Any:
    """Coerce ``value`` into something OpenTelemetry accepts as an attribute.

    Attribute values must be ``str``/``bool``/``int``/``float`` or a homogeneous
    sequence of one of those; anything else is dropped by the SDK with a
    warning. Values reaching here have already been escaped and capped by the
    observer, so stringifying is safe.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value]
    return str(value)


def to_attributes(values: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce a mapping into an attribute dict, dropping ``None`` values."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        coerced = to_attribute_value(value)
        if coerced is not None:
            out[str(key)] = coerced
    return out


class SpanHandle:
    """A started span plus the context token that made it current, if any."""

    __slots__ = ("span", "token")

    def __init__(self, span: Any, token: Any) -> None:
        self.span = span
        self.token = token


class OTelBridge:
    """Creates OpenTelemetry spans on behalf of an :class:`AgentObserver`.

    Args:
        tracer: An OpenTelemetry ``Tracer``. Defaults to
            ``opentelemetry.trace.get_tracer("adapt_agent", ...)`` on the
            globally configured (or the given) tracer provider.
        tracer_provider: Used only when ``tracer`` is omitted.
        agent_operation_name: ``gen_ai.operation.name`` for agent traces and
            refusals. ``"invoke_agent"`` by default. A framework that already
            emits its own ``invoke_agent`` span will then show two per run (its
            own and the governance one nested in or around it); a backend that
            counts agent invocations from spans should filter on the
            ``adapt_agent.trace_id`` attribute, or pass another value here.
        activate: Make each trace's span the *current* span between
            ``start_trace`` and ``end_trace``, so spans the framework opens in
            between become its children. On by default.

    Raises:
        MissingDependencyError: If ``opentelemetry-api`` is not installed --
            loudly, because an observer asked to export spans that silently
            exported nothing would be the worst outcome.
    """

    def __init__(
        self,
        tracer: Any = None,
        *,
        tracer_provider: Any = None,
        agent_operation_name: str = OPERATION_INVOKE_AGENT,
        activate: bool = True,
    ) -> None:
        try:
            from opentelemetry import context as otel_context
            from opentelemetry import trace as otel_trace
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            from adapt_agent.exceptions import MissingDependencyError

            raise MissingDependencyError("opentelemetry-api", "otel") from exc

        self._trace = otel_trace
        self._context = otel_context
        self._status_cls = otel_trace.Status
        self._error_code = otel_trace.StatusCode.ERROR
        self._internal = otel_trace.SpanKind.INTERNAL
        if tracer is None:
            import adapt_agent

            tracer = otel_trace.get_tracer(
                "adapt_agent", adapt_agent.__version__, tracer_provider=tracer_provider
            )
        elif not callable(getattr(tracer, "start_span", None)):
            raise TypeError(f"tracer must be an OpenTelemetry Tracer, got {type(tracer).__name__}")
        self.tracer = tracer
        self.agent_operation_name = agent_operation_name
        self.activate = activate

    def start(
        self,
        name: str,
        attributes: Mapping[str, Any],
        *,
        parent: SpanHandle | None = None,
        activate: bool | None = None,
    ) -> SpanHandle:
        """Start a span as a child of ``parent``, or of the current context."""
        ctx = self._trace.set_span_in_context(parent.span) if parent is not None else None
        span = self.tracer.start_span(
            name, context=ctx, kind=self._internal, attributes=to_attributes(attributes)
        )
        token = None
        if self.activate if activate is None else activate:
            token = self._context.attach(self._trace.set_span_in_context(span))
        return SpanHandle(span, token)

    def ids(self, handle: SpanHandle) -> tuple[str, str] | None:
        """``(trace_id, span_id)`` as lowercase hex, or ``None`` for a no-op span."""
        ctx = handle.span.get_span_context()
        if not getattr(ctx, "is_valid", False):
            return None
        return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")

    def add_event(self, handle: SpanHandle, name: str, attributes: Mapping[str, Any]) -> None:
        """Attach a span event (an observer event or a threat detection)."""
        handle.span.add_event(name, attributes=to_attributes(attributes))

    def end(
        self,
        handle: SpanHandle,
        attributes: Mapping[str, Any],
        *,
        error_type: str | None = None,
        description: str | None = None,
    ) -> None:
        """Set final attributes and status, end the span, restore the context.

        The status is left UNSET on success, as the conventions recommend for
        instrumentation; ``error_type`` marks it ERROR.

        The context token is detached only when this span is still the current
        one *here*. ``start_trace`` and ``end_trace`` are separate calls, and a
        caller may end a trace from another thread or task than the one that
        started it; detaching a token there raises inside OpenTelemetry (which
        logs it as an error) and cannot restore anything meaningful anyway --
        the context the token belongs to is the other thread's.
        """
        span = handle.span
        if attributes:
            span.set_attributes(to_attributes(attributes))
        if error_type is not None:
            span.set_attribute(ERROR_TYPE, error_type)
            span.set_status(self._status_cls(self._error_code, description))
        span.end()
        token, handle.token = handle.token, None
        if token is not None and self._trace.get_current_span() is span:
            self._context.detach(token)


__all__ = [
    "ADAPT_DECISION",
    "ADAPT_DESCRIPTION",
    "ADAPT_OPERATION",
    "ADAPT_POLICY_RULES",
    "ADAPT_REPORT_ONLY",
    "ADAPT_STAGE",
    "ADAPT_STATUS",
    "ADAPT_THREATS",
    "ADAPT_TRACE_ID",
    "ERROR_TYPE",
    "ERROR_TYPE_BLOCKED",
    "ERROR_TYPE_OTHER",
    "EVENT_THREAT_DETECTED",
    "GEN_AI_AGENT_ID",
    "GEN_AI_AGENT_NAME",
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_TOOL_CALL_ID",
    "GEN_AI_TOOL_NAME",
    "OPERATION_EXECUTE_TOOL",
    "OPERATION_INVOKE_AGENT",
    "OTelBridge",
    "SpanHandle",
    "policy_rules",
]
