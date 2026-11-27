"""Observability and monitoring for LLM agents."""

from __future__ import annotations

import contextlib
import logging
import math
import random
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, cast

logger = logging.getLogger(__name__)

#: Characters that are escaped in every string the observer stores.
#:
#: All of Unicode categories ``Cc`` (C0/C1 controls, DEL -- this covers ``\n``,
#: ``\r``, ``\v``, ``\f``, ``\x1c``-``\x1e``, ``\x85`` and ESC), ``Zl`` (U+2028)
#: and ``Zp`` (U+2029). Escaping only ``\n`` and ``\r`` left eight more
#: characters that :meth:`str.splitlines` -- and so every line-oriented log
#: consumer -- treats as a line break, which let a logged value forge a log
#: line; and a raw ESC lets it rewrite a terminal (``\x1b[2K\x1b[1A`` erases the
#: line above). The bidirectional-override controls are added for the same
#: reason: they reorder what a human reads without changing the bytes
#: ("Trojan Source"). Other format characters (``Cf``: zero-width joiners, soft
#: hyphens) are left alone because ordinary text in many scripts uses them.
#: ``tests/test_observability_hardening.py`` re-derives the ``Cc``/``Zl``/``Zp``
#: part over the whole of Unicode.
_UNSAFE_CHARS_RE = re.compile(
    "[\x00-\x1f\x7f-\x9f\u2028\u2029\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]"
)
_NAMED_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}

#: Statuses that mean the run did not succeed (the trace's span is an ERROR).
ERROR_STATUSES = frozenset({"error", "failed", "blocked"})

#: How deep :meth:`AgentObserver._clean_value` descends into nested metadata,
#: and how many items of one container it keeps. Metadata is caller-supplied;
#: without both bounds one deeply nested or huge value is an unbounded copy.
_MAX_DEPTH = 4
_MAX_ITEMS = 100
#: Cap for identifiers (agent ids, event types, levels, threat labels). They
#: are escaped but never passed to the redaction hook, which could otherwise
#: rewrite them into something a filter no longer matches.
_MAX_ID_LENGTH = 256
_TRUNCATION_MARK = "...[truncated]"


def _escape_char(match: re.Match[str]) -> str:
    char = match.group()
    named = _NAMED_ESCAPES.get(char)
    if named is not None:
        return named
    code = ord(char)
    return f"\\x{code:02x}" if code <= 0xFF else f"\\u{code:04x}"


def escape_text(text: str) -> str:
    """Escape every control, line/paragraph separator and bidi-override char.

    The result contains no character a log viewer, terminal or
    ``str.splitlines`` treats as a line break or a control sequence.
    """
    if _UNSAFE_CHARS_RE.search(text) is None:  # the common case: nothing to do
        return text
    return _UNSAFE_CHARS_RE.sub(_escape_char, text)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= len(_TRUNCATION_MARK):
        return text[:limit]
    return text[: limit - len(_TRUNCATION_MARK)] + _TRUNCATION_MARK


#: Patterns :func:`redact_secrets` scrubs. Deliberately a short list of shapes
#: that are unambiguous credentials; a deployment with its own token formats
#: should pass its own ``redactor``.
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),  # OpenAI / Anthropic style API keys
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key ids
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub tokens
    re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"),  # Slack tokens
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd)(\s*[:=]\s*)[^\s,;]+"),
)


def redact_secrets(text: str) -> str:
    """A ready-made ``redactor``: replace common credential shapes with ``[REDACTED]``.

    Not the default -- the default stores text escaped and truncated but
    otherwise verbatim, because no pattern list is complete and a default that
    looked like a guarantee would be worse than none. Pass it (or your own
    callable) as ``AgentObserver(redactor=redact_secrets)``.
    """
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda m: m.group(1) + m.group(2) + "[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_span_id() -> str:
    # Not a secret: correlation only. getrandbits is what the OpenTelemetry SDK
    # itself uses for span ids, and is several times cheaper than uuid4.
    return f"{random.getrandbits(64):016x}"


class ToolSpan:
    """Handle yielded by :meth:`AgentObserver.tool_span`.

    Call :meth:`record` with each :class:`~adapt_agent.core.governance.ToolDecision`
    the gate returns for this tool call (the ``check_tool_call`` verdict and,
    after the tool ran, the ``check_tool_result`` one). The span closes with the
    most severe verdict recorded.
    """

    __slots__ = ("agent_id", "tool_name", "span_id", "parent_id", "tool_call_id", "_decisions")

    def __init__(
        self,
        agent_id: str,
        tool_name: str,
        span_id: str,
        parent_id: str | None,
        tool_call_id: str | None,
    ) -> None:
        self.agent_id = agent_id
        self.tool_name = tool_name
        self.span_id = span_id
        self.parent_id = parent_id
        self.tool_call_id = tool_call_id
        self._decisions: list[tuple[str, Any]] = []

    def record(self, decision: Any, phase: str = "call") -> None:
        """Record a verdict. ``phase`` is ``"call"`` or ``"result"``.

        Duck-typed: anything with ``action`` (``"allow"``/``"block"``/``"ask"``)
        and optionally ``threats``, ``report_only`` and ``tool_name`` works.
        """
        self._decisions.append((phase, decision))


#: Severity order used to pick the verdict a tool span closes with.
_ACTION_RANK = {"allow": 0, "ask": 1, "block": 2}


class AgentObserver:
    """Provides observability and monitoring for LLM agents.

    Tracks agent execution, logs interactions, and provides debugging and
    monitoring capabilities. Every stored string is escaped (no control, line or
    paragraph separator, or bidi-override character survives), length-capped,
    and -- for free text -- passed through the optional ``redactor`` first.

    The observer is safe to share between threads: the governed adapters call
    it from worker threads (``asyncio.to_thread``) while monitoring code reads
    it. Readers get snapshots, never the live records.

    Args:
        max_logs: Maximum number of logs to store in memory.
        max_traces: Maximum number of traces to store in memory. Completed
            traces are evicted first; an active trace is evicted only when every
            stored trace is active, and that is counted in :meth:`get_stats`.
            Refusals and tool decisions each keep their own buffer of this size.
        max_metrics: Maximum number of values stored per metric name.
        max_events_per_trace: Maximum number of events to store per trace.
        max_metric_names: Maximum number of metric names to store in memory.
        max_text_length: Cap for a stored free-text string (log messages,
            event descriptions, metadata values, results).
        max_error_length: Cap for error text: the ``result`` of a trace that
            ended in an error status and an exception's message. Exception text
            routinely quotes the prompt, the model's output or a credential, so
            it is kept short by default.
        redactor: Callable applied to every stored free-text string *before*
            escaping and truncation, e.g. :func:`redact_secrets`. A redactor
            that raises makes the text be stored as ``"[redaction failed]"``,
            never verbatim.
        otel: Mirror traces into OpenTelemetry spans. ``True`` uses the global
            tracer provider; an OpenTelemetry ``Tracer`` or an
            :class:`~adapt_agent.observability.otel.OTelBridge` is used as
            given. ``False`` (the default) never imports OpenTelemetry.
    """

    def __init__(
        self,
        max_logs: int = 1000,
        max_traces: int = 1000,
        max_metrics: int = 1000,
        max_events_per_trace: int = 1000,
        max_metric_names: int = 1000,
        *,
        max_text_length: int = 10000,
        max_error_length: int = 200,
        redactor: Callable[[str], str] | None = None,
        otel: Any = False,
    ):
        for name, value in (
            ("max_logs", max_logs),
            ("max_traces", max_traces),
            ("max_metrics", max_metrics),
            ("max_events_per_trace", max_events_per_trace),
            ("max_metric_names", max_metric_names),
            ("max_text_length", max_text_length),
            ("max_error_length", max_error_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if redactor is not None and not callable(redactor):
            raise TypeError("redactor must be a callable taking and returning a str")
        self.max_logs = max_logs
        self.max_traces = max_traces
        self.max_metrics = max_metrics
        self.max_events_per_trace = max_events_per_trace
        self.max_metric_names = max_metric_names
        self.max_text_length = max_text_length
        self.max_error_length = max_error_length
        self.redactor = redactor
        self.otel = self._make_bridge(otel)
        # One lock for all state. The critical sections are a few dict/deque
        # operations each; text cleaning and every OpenTelemetry call (span
        # processors may export synchronously) happen outside it.
        self._lock = threading.Lock()
        # ⚡ Bolt: Using a dict instead of list for O(1) trace lookups
        self._traces: dict[str, dict[str, Any]] = {}
        # Completed trace ids in completion order (an ordered set), so eviction
        # can drop the oldest *completed* trace in O(1) instead of a FIFO that
        # discarded still-running traces.
        self._completed: dict[str, None] = {}
        self._clock: dict[str, int] = {}
        self._spans: dict[str, Any] = {}
        # ⚡ Bolt: Use deque for O(1) eviction instead of O(N) list.pop(0)
        self._logs: deque[dict[str, Any]] = deque(maxlen=max_logs)
        self._metrics: dict[str, deque[float]] = {}
        self._refusals: deque[dict[str, Any]] = deque(maxlen=max_traces)
        self._tool_decisions: deque[dict[str, Any]] = deque(maxlen=max_traces)
        self._stats: dict[str, int] = {
            "traces_evicted": 0,
            "active_traces_evicted": 0,
            "duplicate_trace_ids": 0,
            "end_trace_unknown": 0,
            "logs_evicted": 0,
            "refusals_recorded": 0,
            "tool_decisions_recorded": 0,
            "otel_errors": 0,
        }

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _make_bridge(otel: Any) -> Any:
        if otel is False or otel is None:
            return None
        from adapt_agent.observability.otel import OTelBridge

        if otel is True:
            return OTelBridge()
        if isinstance(otel, OTelBridge):
            return otel
        if callable(getattr(otel, "start_span", None)):
            return OTelBridge(tracer=otel)
        raise TypeError(
            "otel must be True/False, an OpenTelemetry Tracer, or an OTelBridge; "
            f"got {type(otel).__name__}"
        )

    # --------------------------------------------------------------- cleaning

    def _clean(self, text: Any, limit: int) -> str:
        """Redact, escape and cap one free-text value.

        Truncating to ``limit`` first bounds the cost of the redactor and of
        escaping on a huge input; truncating again afterwards bounds what
        escaping (up to 6 characters per input character) or the redactor added.
        """
        if not isinstance(text, str):
            text = self._stringify(text)
        text = text[:limit]
        if self.redactor is not None:
            try:
                text = str(self.redactor(text))
            except Exception:  # noqa: BLE001 - fail closed on a broken hook
                logger.warning("AgentObserver redactor raised; text withheld", exc_info=True)
                return "[redaction failed]"
        return _truncate(escape_text(text), limit)

    @staticmethod
    def _clean_id(value: Any) -> str:
        """Escape and cap an identifier. Identifiers are never redacted."""
        if not isinstance(value, str):
            value = AgentObserver._stringify(value)
        return _truncate(escape_text(value[: _MAX_ID_LENGTH * 2]), _MAX_ID_LENGTH)

    @staticmethod
    def _stringify(value: Any) -> str:
        try:
            return str(value)
        except Exception:  # noqa: BLE001 - a hostile __str__ must not break logging
            return f"<unrepresentable {type(value).__name__}>"

    def _clean_value(self, value: Any, limit: int, depth: int = 0) -> Any:
        """Copy caller-supplied data with every string cleaned, to a bounded depth.

        Scalars that cannot carry text (``None``, ``bool``, ``int``, ``float``)
        are kept as they are; containers become plain ``dict``/``list`` copies
        of at most ``_MAX_ITEMS`` items; anything else is stored as its cleaned
        ``str()``, so the observer never retains a reference to caller objects.
        """
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return self._clean(value, limit)
        if depth >= _MAX_DEPTH:
            return "[truncated: nested too deep]"
        if isinstance(value, Mapping):
            return {
                self._clean_id(k): self._clean_value(v, limit, depth + 1)
                for k, v in list(value.items())[:_MAX_ITEMS]
            }
        if isinstance(value, (list, tuple, set, frozenset, deque)):
            return [self._clean_value(v, limit, depth + 1) for v in list(value)[:_MAX_ITEMS]]
        return self._clean(value, limit)

    def _clean_metadata(self, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        if not metadata:
            return {}
        return cast("dict[str, Any]", self._clean_value(dict(metadata), self.max_text_length))

    def _clean_threats(self, threats: Any) -> list[str]:
        if not threats:
            return []
        if isinstance(threats, str):
            threats = [threats]
        return [self._clean_id(t) for t in list(threats)[:_MAX_ITEMS]]

    # ------------------------------------------------------- otel plumbing

    def _otel_call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run one bridge call; a failure is counted and logged, never raised.

        The bridge is telemetry, not a control: an exporter or span processor
        that raises must not turn into a failed agent request.
        """
        try:
            return fn(*args, **kwargs)
        except Exception:  # noqa: BLE001
            with self._lock:
                self._stats["otel_errors"] += 1
            logger.warning("AgentObserver OpenTelemetry bridge call failed", exc_info=True)
            return None

    def _end_orphan_span(self, handle: Any, reason: str) -> None:
        """End the span of a trace that is leaving the store without end_trace."""
        if handle is not None and self.otel is not None:
            self._otel_call(
                self.otel.end,
                handle,
                {},
                error_type=reason,
                description=f"trace {reason} before end_trace",
            )

    # ---------------------------------------------------------------- traces

    def start_trace(
        self,
        trace_id: str,
        agent_id: str,
        operation: str,
        metadata: dict[str, Any] | None = None,
        *,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        """Start a new trace.

        Args:
            trace_id: Unique identifier for the trace
            agent_id: Agent identifier
            operation: Operation being traced
            metadata: Optional metadata
            parent_id: Optional id of the enclosing trace. With the OpenTelemetry
                bridge, a known parent's span becomes the parent span; otherwise
                the span is a child of the current context.

        Returns:
            A snapshot of the new trace record.
        """
        from_otel = self.otel
        record: dict[str, Any] = {
            "trace_id": self._clean_id(trace_id),
            "span_id": _new_span_id(),
            "parent_id": None if parent_id is None else self._clean_id(parent_id),
            "agent_id": self._clean_id(agent_id),
            "operation": self._clean_id(operation),
            "start_time": _now(),
            "end_time": None,
            "duration_ms": None,
            "metadata": self._clean_metadata(metadata),
            "events": [],
            "status": "active",
            "result": None,
            "threats": [],
            "decision": None,
            "report_only": False,
            "error_type": None,
        }

        handle = None
        if from_otel is not None:
            from adapt_agent.observability import otel as sc

            with self._lock:
                parent = self._spans.get(parent_id) if parent_id is not None else None
            attrs = {
                sc.GEN_AI_OPERATION_NAME: from_otel.agent_operation_name,
                sc.GEN_AI_AGENT_ID: record["agent_id"],
                sc.GEN_AI_AGENT_NAME: record["agent_id"],
                sc.ADAPT_TRACE_ID: record["trace_id"],
                sc.ADAPT_OPERATION: record["operation"],
            }
            handle = self._otel_call(
                from_otel.start,
                f"{from_otel.agent_operation_name} {record['agent_id']}",
                attrs,
                parent=parent,
            )
            ids = self._otel_call(from_otel.ids, handle) if handle is not None else None
            if ids is not None:
                record["otel_trace_id"], record["span_id"] = ids

        orphans: list[tuple[Any, str]] = []
        with self._lock:
            old = self._traces.pop(trace_id, None)
            if old is not None:
                # A reused id replaces the old record, as it always has; the
                # count makes the collision visible.
                self._stats["duplicate_trace_ids"] += 1
                self._completed.pop(trace_id, None)
                orphans.append((self._spans.pop(trace_id, None), "replaced"))
            self._traces[trace_id] = record
            self._clock[trace_id] = time.perf_counter_ns()
            if handle is not None:
                self._spans[trace_id] = handle

            # SECURITY: Prevent unbounded memory growth
            if len(self._traces) > self.max_traces:
                if self._completed:
                    victim = next(iter(self._completed))
                    del self._completed[victim]
                    self._stats["traces_evicted"] += 1
                else:
                    victim = next(iter(self._traces))
                    self._stats["active_traces_evicted"] += 1
                    orphans.append((self._spans.pop(victim, None), "evicted"))
                    logger.warning(
                        "AgentObserver evicted an active trace (all %d stored traces are "
                        "active); its end_trace will be counted as unknown",
                        self.max_traces,
                    )
                del self._traces[victim]
                self._clock.pop(victim, None)
            snapshot = self._snapshot(record)

        for orphan, reason in orphans:
            self._end_orphan_span(orphan, reason)
        return snapshot

    def end_trace(
        self,
        trace_id: str,
        status: str = "completed",
        result: Any | None = None,
        *,
        threats: Sequence[str] | None = None,
        error: BaseException | None = None,
        report_only: bool = False,
    ) -> None:
        """End a trace.

        Args:
            trace_id: Trace identifier
            status: Final status. ``"error"``, ``"failed"`` and ``"blocked"``
                mark the span as an ERROR.
            result: Optional result data (cleaned; an error status caps a string
                result at ``max_error_length``).
            threats: Threat labels behind the outcome (e.g. the
                ``SecurityBlockedError.threats`` of an output block).
            error: The exception that ended the run. Its type is recorded as
                ``error_type``; its message, capped at ``max_error_length``,
                becomes the result if none is given; a ``threats`` attribute
                (``SecurityBlockedError``) supplies the threat labels.
            report_only: The threats were detected in report-only mode, so the
                run was not refused.
        """
        failed = status in ERROR_STATUSES
        error_type: str | None = None
        if error is not None:
            error_type = type(error).__name__
            if threats is None:
                found = getattr(error, "threats", None)
                if isinstance(found, (list, tuple)):
                    threats = found
            if result is None:
                result = self._clean(error, self.max_error_length)
        labels = self._clean_threats(threats)
        if failed and isinstance(result, str):
            result = self._clean(result, self.max_error_length)
        else:
            result = self._clean_value(result, self.max_text_length)
        if failed and error_type is None:
            error_type = "blocked" if labels and not report_only else "_OTHER"
        decision = None
        if labels:
            decision = "block" if failed and not report_only else "allow"

        end_ns = time.perf_counter_ns()
        # ⚡ Bolt: Replaced O(N) loop with O(1) dictionary lookup
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                self._stats["end_trace_unknown"] += 1
                handle = self._spans.pop(trace_id, None)
            else:
                start_ns = self._clock.pop(trace_id, None)
                trace["end_time"] = _now()
                if start_ns is not None:
                    trace["duration_ms"] = round((end_ns - start_ns) / 1e6, 3)
                trace["status"] = self._clean_id(status)
                trace["result"] = result
                trace["threats"] = labels
                trace["decision"] = decision
                trace["report_only"] = bool(report_only)
                trace["error_type"] = error_type if failed else None
                self._completed.pop(trace_id, None)
                self._completed[trace_id] = None
                handle = self._spans.pop(trace_id, None)

        if handle is not None and self.otel is not None:
            from adapt_agent.observability import otel as sc

            if labels:
                self._otel_call(
                    self.otel.add_event,
                    handle,
                    sc.EVENT_THREAT_DETECTED,
                    {sc.ADAPT_THREATS: labels, sc.ADAPT_REPORT_ONLY: bool(report_only)},
                )
            attrs: dict[str, Any] = {
                sc.ADAPT_STATUS: self._clean_id(status),
                sc.ADAPT_REPORT_ONLY: bool(report_only),
            }
            if labels:
                attrs[sc.ADAPT_THREATS] = labels
                attrs[sc.ADAPT_DECISION] = decision
                rules = sc.policy_rules(labels)
                if rules:
                    attrs[sc.ADAPT_POLICY_RULES] = rules
            description = result if failed and isinstance(result, str) else None
            self._otel_call(
                self.otel.end,
                handle,
                attrs,
                error_type=error_type if failed else None,
                description=description,
            )

    # -------------------------------------------------------------- refusals

    def record_refusal(
        self,
        agent_id: str,
        operation: str,
        threats: Sequence[str] | None = None,
        *,
        stage: str = "input",
        reason: str | None = None,
        error: BaseException | None = None,
        report_only: bool = False,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a request refused before any trace was opened.

        An input block happens before the governed run starts, so it never had a
        trace to end -- and recorded nothing at all. This stores it in its own
        bounded buffer (:meth:`get_refusals`), so trace counts keep meaning "runs
        that started", and emits a zero-length ERROR span with the threat labels
        when the OpenTelemetry bridge is on.

        Args:
            agent_id: Agent identifier.
            operation: Operation label, as given to :meth:`start_trace`.
            threats: Threat labels behind the refusal (taken from
                ``error.threats`` when omitted).
            stage: Where the request was refused (``"input"``, ``"policy"``...).
            reason: Optional short, content-free reason.
            error: Optional exception; its type is recorded as ``error_type``.
            report_only: The controls only reported; nothing was refused.
            parent_id: Optional id of an enclosing trace.
            metadata: Optional metadata.

        Returns:
            A snapshot of the stored refusal record.
        """
        if threats is None and error is not None:
            found = getattr(error, "threats", None)
            if isinstance(found, (list, tuple)):
                threats = found
        if reason is None and error is not None:
            reason = getattr(error, "reason", None)
            if not isinstance(reason, str):
                reason = None
        labels = self._clean_threats(threats)
        error_type = type(error).__name__ if error is not None else "blocked"
        record: dict[str, Any] = {
            "span_id": _new_span_id(),
            "parent_id": None if parent_id is None else self._clean_id(parent_id),
            "agent_id": self._clean_id(agent_id),
            "operation": self._clean_id(operation),
            "stage": self._clean_id(stage),
            "time": _now(),
            "threats": labels,
            "decision": "allow" if report_only else "block",
            "report_only": bool(report_only),
            "reason": None if reason is None else self._clean(reason, self.max_error_length),
            "error_type": None if report_only else error_type,
            "metadata": self._clean_metadata(metadata),
        }

        bridge = self.otel
        if bridge is not None:
            from adapt_agent.observability import otel as sc

            with self._lock:
                parent = self._spans.get(parent_id) if parent_id is not None else None
            attrs = {
                sc.GEN_AI_OPERATION_NAME: bridge.agent_operation_name,
                sc.GEN_AI_AGENT_ID: record["agent_id"],
                sc.GEN_AI_AGENT_NAME: record["agent_id"],
                sc.ADAPT_OPERATION: record["operation"],
                sc.ADAPT_STAGE: record["stage"],
                sc.ADAPT_STATUS: "refused",
                sc.ADAPT_DECISION: record["decision"],
                sc.ADAPT_REPORT_ONLY: bool(report_only),
                sc.ADAPT_THREATS: labels,
                sc.ADAPT_POLICY_RULES: sc.policy_rules(labels) or None,
            }
            handle = self._otel_call(
                bridge.start,
                f"{bridge.agent_operation_name} {record['agent_id']}",
                attrs,
                parent=parent,
                activate=False,
            )
            if handle is not None:
                ids = self._otel_call(bridge.ids, handle)
                if ids is not None:
                    record["otel_trace_id"], record["span_id"] = ids
                if labels:
                    self._otel_call(
                        bridge.add_event,
                        handle,
                        sc.EVENT_THREAT_DETECTED,
                        {sc.ADAPT_THREATS: labels, sc.ADAPT_STAGE: record["stage"]},
                    )
                self._otel_call(
                    bridge.end,
                    handle,
                    {},
                    error_type=record["error_type"],
                    description=record["reason"],
                )

        with self._lock:
            self._refusals.append(record)
            self._stats["refusals_recorded"] += 1
            return dict(record)

    def get_refusals(
        self, agent_id: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Refusals recorded with :meth:`record_refusal`, oldest first."""
        return self._select(self._refusals, "agent_id", agent_id, limit)

    # ----------------------------------------------------------- tool spans

    @contextlib.contextmanager
    def tool_span(
        self,
        agent_id: str,
        tool_name: str,
        *,
        parent_id: str | None = None,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[ToolSpan]:
        """Time one tool call and record the gate's verdicts on it.

        Usage in a framework binding::

            with observer.tool_span(agent_id, name, parent_id=trace_id) as span:
                decision = gate.check_tool_call(name, args)
                span.record(decision, "call")
                if not decision.allowed:
                    return decision.reason
                result = run_tool(args)
                span.record(gate.check_tool_result(name, result), "result")

        With the OpenTelemetry bridge the span (``execute_tool <name>``) is
        current inside the block, so spans the tool opens nest under it. The
        span ends as an ERROR when a verdict blocked (outside report-only mode)
        or the block raised.
        """
        handle_span = ToolSpan(
            self._clean_id(agent_id),
            self._clean_id(tool_name),
            _new_span_id(),
            None if parent_id is None else self._clean_id(parent_id),
            None if tool_call_id is None else self._clean_id(tool_call_id),
        )
        bridge = self.otel
        handle = None
        if bridge is not None:
            from adapt_agent.observability import otel as sc

            with self._lock:
                parent = self._spans.get(parent_id) if parent_id is not None else None
            attrs = {
                sc.GEN_AI_OPERATION_NAME: sc.OPERATION_EXECUTE_TOOL,
                sc.GEN_AI_TOOL_NAME: handle_span.tool_name,
                sc.GEN_AI_TOOL_CALL_ID: handle_span.tool_call_id,
                sc.GEN_AI_AGENT_ID: handle_span.agent_id,
                sc.GEN_AI_AGENT_NAME: handle_span.agent_id,
            }
            handle = self._otel_call(
                bridge.start,
                f"{sc.OPERATION_EXECUTE_TOOL} {handle_span.tool_name}",
                attrs,
                parent=parent,
            )
            ids = self._otel_call(bridge.ids, handle) if handle is not None else None
            if ids is not None:
                handle_span.span_id = ids[1]
        started_at = _now()
        start_ns = time.perf_counter_ns()
        try:
            yield handle_span
        except BaseException as exc:  # CancelledError derives from BaseException
            self._finish_tool(handle_span, handle, started_at, start_ns, metadata, exc)
            raise
        self._finish_tool(handle_span, handle, started_at, start_ns, metadata, None)

    def record_tool_decision(
        self,
        agent_id: str | None,
        decision: Any,
        *,
        phase: str = "call",
        parent_id: str | None = None,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record one tool verdict as a zero-length tool span.

        The one-call form of :meth:`tool_span`, for a binding that only reports
        the verdict (for instance a block, where the tool never runs).

        Args:
            agent_id: Agent identifier; ``None`` uses ``decision.agent_id``.
            decision: A :class:`~adapt_agent.core.governance.ToolDecision`, or
                anything with ``action`` and ``tool_name`` (duck-typed).
            phase: ``"call"`` (``check_tool_call``) or ``"result"``
                (``check_tool_result``).
            parent_id: Optional id of the enclosing agent trace; the verdict is
                also logged as a ``tool_decision`` event on it.
            tool_call_id: The framework's tool-call id, if it has one.
            metadata: Optional metadata.

        Returns:
            A snapshot of the stored tool record.
        """
        if agent_id is None:
            agent_id = getattr(decision, "agent_id", None) or "unknown"
        tool_name = getattr(decision, "tool_name", None) or "unknown"
        with self.tool_span(
            agent_id, tool_name, parent_id=parent_id, tool_call_id=tool_call_id, metadata=metadata
        ) as span:
            span.record(decision, phase)
        with self._lock:
            for record in reversed(self._tool_decisions):
                if record["span_id"] == span.span_id:
                    return dict(record)
        return {}  # pragma: no cover - evicted between the two statements

    def _finish_tool(
        self,
        span: ToolSpan,
        handle: Any,
        started_at: str,
        start_ns: int,
        metadata: dict[str, Any] | None,
        exc: BaseException | None,
    ) -> None:
        action = None
        threats: list[str] = []
        report_only = False
        phases: list[str] = []
        for phase, decision in span._decisions:
            this = self._clean_id(getattr(decision, "action", "unknown"))
            phases.append(self._clean_id(phase))
            if action is None or _ACTION_RANK.get(this, 3) > _ACTION_RANK.get(action, 3):
                action = this
            for label in self._clean_threats(getattr(decision, "threats", None)):
                if label not in threats:
                    threats.append(label)
            report_only = report_only or bool(getattr(decision, "report_only", False))
        blocked = action is not None and action != "allow" and action != "ask"
        if exc is not None:
            status, error_type = "error", type(exc).__name__
        elif blocked and not report_only:
            status, error_type = "blocked", "blocked"
        elif action == "ask":
            status, error_type = "approval_required", None
        else:
            status, error_type = "completed", None

        record: dict[str, Any] = {
            "span_id": span.span_id,
            "parent_id": span.parent_id,
            "agent_id": span.agent_id,
            "tool_name": span.tool_name,
            "tool_call_id": span.tool_call_id,
            "phases": phases,
            "decision": action,
            "threats": threats,
            "report_only": report_only,
            "status": status,
            "error_type": error_type,
            "start_time": started_at,
            "end_time": _now(),
            "duration_ms": round((time.perf_counter_ns() - start_ns) / 1e6, 3),
            "metadata": self._clean_metadata(metadata),
        }
        with self._lock:
            self._tool_decisions.append(record)
            self._stats["tool_decisions_recorded"] += 1

        if span.parent_id is not None:
            self.log_event(
                span.parent_id,
                "tool_decision",
                f"{span.tool_name}: {action or 'unrecorded'}",
                metadata={"threats": threats, "status": status, "span_id": span.span_id},
            )

        if handle is not None and self.otel is not None:
            from adapt_agent.observability import otel as sc

            if threats:
                self._otel_call(
                    self.otel.add_event,
                    handle,
                    sc.EVENT_THREAT_DETECTED,
                    {sc.ADAPT_THREATS: threats, sc.ADAPT_REPORT_ONLY: report_only},
                )
            attrs = {
                sc.ADAPT_STATUS: status,
                sc.ADAPT_DECISION: action,
                sc.ADAPT_STAGE: ["tool_" + p for p in phases] or None,
                sc.ADAPT_THREATS: threats or None,
                sc.ADAPT_POLICY_RULES: sc.policy_rules(threats) or None,
                sc.ADAPT_REPORT_ONLY: report_only,
            }
            description = None
            if exc is not None:
                description = self._clean(exc, self.max_error_length)
            elif error_type is not None:
                description = f"tool call blocked: {span.tool_name}"
            self._otel_call(
                self.otel.end, handle, attrs, error_type=error_type, description=description
            )

    def get_tool_decisions(
        self,
        agent_id: str | None = None,
        tool_name: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Tool records from :meth:`tool_span` / :meth:`record_tool_decision`."""
        records = self._select(self._tool_decisions, "agent_id", agent_id, None)
        if tool_name is not None:
            wanted = self._clean_id(tool_name)
            records = [r for r in records if r["tool_name"] == wanted]
        if limit is not None:
            records = records[-limit:] if limit > 0 else []
        return records

    # ---------------------------------------------------------------- events

    def log_event(
        self,
        trace_id: str,
        event_type: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an event within a trace.

        Args:
            trace_id: Trace identifier
            event_type: Type of event
            description: Event description
            metadata: Optional metadata
        """
        event: dict[str, Any] = {
            "event_type": self._clean_id(event_type),
            "description": self._clean(description, self.max_text_length),
            "timestamp": _now(),
            "metadata": self._clean_metadata(metadata),
        }

        # ⚡ Bolt: Replaced O(N) loop with O(1) dictionary lookup
        with self._lock:
            if trace_id not in self._traces:
                return
            self._traces[trace_id]["events"].append(event)
            # SECURITY: Prevent unbounded memory growth
            if len(self._traces[trace_id]["events"]) > self.max_events_per_trace:
                self._traces[trace_id]["events"].pop(0)
            handle = self._spans.get(trace_id)

        if handle is not None and self.otel is not None:
            from adapt_agent.observability import otel as sc

            attrs: dict[str, Any] = {sc.ADAPT_DESCRIPTION: event["description"]}
            for key, value in event["metadata"].items():
                attrs[f"adapt_agent.event.{key}"] = value
            self._otel_call(self.otel.add_event, handle, event["event_type"], attrs)

    def log(
        self,
        level: str,
        message: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a message.

        Args:
            level: Log level (debug, info, warning, error)
            message: Log message
            agent_id: Optional agent identifier
            metadata: Optional metadata
        """
        log_entry = {
            "level": self._clean_id(level),
            "message": self._clean(message, self.max_text_length),
            "agent_id": None if agent_id is None else self._clean_id(agent_id),
            "timestamp": _now(),
            "metadata": self._clean_metadata(metadata),
        }
        with self._lock:
            if len(self._logs) == self.max_logs:
                self._stats["logs_evicted"] += 1
            self._logs.append(log_entry)

    # --------------------------------------------------------------- metrics

    def record_metric(
        self,
        metric_name: str,
        value: float,
    ) -> None:
        """Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value

        Raises:
            TypeError: If ``value`` is not an ``int``/``float`` (``bool`` included).
            ValueError: If ``value`` is NaN or infinite. One NaN would make every
                later ``min``/``max``/``avg`` of the series meaningless (``min``
                and ``max`` become order-dependent, the mean NaN), so it is
                refused at the door rather than stored.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"metric value must be a number, got {type(value).__name__}")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"metric value must be finite, got {value!r}")
        with self._lock:
            if metric_name not in self._metrics:
                # ⚡ Bolt: Use deque for O(1) eviction instead of O(N) list.pop(0)
                self._metrics[metric_name] = deque(maxlen=self.max_metrics)
                # SECURITY: Prevent memory exhaustion from unbounded dictionary
                if len(self._metrics) > self.max_metric_names:
                    oldest_metric = next(iter(self._metrics))
                    del self._metrics[oldest_metric]

            self._metrics[metric_name].append(value)

    def get_metric_stats(self, metric_name: str) -> dict[str, float]:
        """Get statistics for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            ``count``, ``min``, ``max`` and ``avg``; ``{}`` for an unknown or
            empty series.
        """
        with self._lock:
            series = self._metrics.get(metric_name)
            if not series:
                return {}
            values = list(series)
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            # fsum: the plain sum of many values of mixed magnitude drifts.
            "avg": math.fsum(values) / len(values),
        }

    # --------------------------------------------------------------- queries

    @staticmethod
    def _snapshot(trace: dict[str, Any]) -> dict[str, Any]:
        """A copy a reader can hold while writers keep updating the record.

        ``metadata`` and event dicts are created by the observer and never
        mutated afterwards, so sharing them is safe; the record itself and its
        ``events``/``threats`` lists are mutated, so they are copied.
        """
        snap = dict(trace)
        snap["events"] = list(trace["events"])
        snap["threats"] = list(trace["threats"])
        return snap

    def get_traces(
        self,
        agent_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get traces.

        Args:
            agent_id: Filter by agent ID
            status: Filter by status
            limit: Maximum number of traces to return

        Returns:
            Snapshots of the matching traces, oldest first.
        """
        if agent_id:
            agent_id = self._clean_id(agent_id)
        if status:
            status = self._clean_id(status)
        with self._lock:
            # ⚡ Bolt: Using reversed() with early-exit limits O(N) operations to O(limit)
            if limit is not None:
                results = []
                for t in reversed(self._traces.values()):
                    if agent_id and t["agent_id"] != agent_id:
                        continue
                    if status and t["status"] != status:
                        continue
                    results.append(t)
                    if len(results) >= limit:
                        break
                results.reverse()
            # ⚡ Bolt: Use a single-pass list comprehension with combined conditions
            elif agent_id and status:
                results = [
                    t
                    for t in self._traces.values()
                    if t["agent_id"] == agent_id and t["status"] == status
                ]
            elif agent_id:
                results = [t for t in self._traces.values() if t["agent_id"] == agent_id]
            elif status:
                results = [t for t in self._traces.values() if t["status"] == status]
            else:
                results = list(self._traces.values())
            return [self._snapshot(t) for t in results]

    def get_logs(
        self,
        level: str | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get logs.

        Args:
            level: Filter by log level
            agent_id: Filter by agent ID
            limit: Maximum number of logs to return

        Returns:
            List of log entries
        """
        if level:
            level = self._clean_id(level)
        if agent_id:
            agent_id = self._clean_id(agent_id)
        with self._lock:
            # ⚡ Bolt: Using reversed() with early-exit limits O(N) operations to O(limit)
            if limit is not None:
                results = []
                for log in reversed(self._logs):
                    if level and log["level"] != level:
                        continue
                    if agent_id and log.get("agent_id") != agent_id:
                        continue
                    results.append(log)
                    if len(results) >= limit:
                        break
                results.reverse()
                return results

            # ⚡ Bolt: Use a single-pass list comprehension with combined conditions
            if level and agent_id:
                return [
                    log
                    for log in self._logs
                    if log["level"] == level and log.get("agent_id") == agent_id
                ]
            if level:
                return [log for log in self._logs if log["level"] == level]
            if agent_id:
                return [log for log in self._logs if log.get("agent_id") == agent_id]

            return list(self._logs)

    def _select(
        self,
        buffer: deque[dict[str, Any]],
        key: str,
        wanted: str | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        if wanted is not None:
            wanted = self._clean_id(wanted)
        with self._lock:
            records = [dict(r) for r in buffer if wanted is None or r[key] == wanted]
        if limit is not None:
            records = records[-limit:] if limit > 0 else []
        return records

    def get_stats(self) -> dict[str, int]:
        """Counters for everything the observer dropped, replaced or failed to export.

        ``active_traces_evicted`` and ``end_trace_unknown`` in particular should
        stay at zero; a non-zero value means ``max_traces`` is too small for the
        number of concurrently running traces (or a caller never ends its
        traces), and some runs were not recorded to completion.
        """
        with self._lock:
            stats = dict(self._stats)
            stats["traces_stored"] = len(self._traces)
            stats["traces_active"] = len(self._traces) - len(self._completed)
            stats["logs_stored"] = len(self._logs)
            stats["metric_names_stored"] = len(self._metrics)
        return stats


__all__ = ["AgentObserver", "ToolSpan", "escape_text", "redact_secrets"]
