"""Transient-error classification and backoff for evaluation runs.

Why this exists: under concurrency, provider rate limiting is *expected*, and an
evaluation harness that scores a throttled example zero does not merely lose a
data point -- it corrupts the comparison. A 429 is indistinguishable from a bad
prompt once it lands as ``error`` plus a zero score, and the damage is
systematic rather than random: whichever candidate happens to be evaluated while
the provider is busiest scores lowest, so an optimizer can select a prompt for
having been lucky with rate limits. That is a measurement bug, not a robustness
nicety.

Two pieces, both duck-typed so no provider SDK is ever imported:

* :func:`is_transient_error` -- does this exception look like throttling or a
  transient server/network fault, as opposed to the agent being broken?
* :class:`RetryPolicy` -- how many attempts, and how long to wait between them,
  honouring a server-supplied ``Retry-After`` when there is one.

The classifier is deliberately generous. A false positive costs one slow retry;
a false negative silently biases every score the run produces.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

#: HTTP status codes worth retrying: throttling, request timeouts, and the
#: 5xx family that means "the server, not your request, is having a problem".
_TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: Attributes that may carry an HTTP status, in the order providers use them.
_STATUS_ATTRS = ("status_code", "status", "http_status", "code")

#: Exception *type name* fragments that mark a transient fault. Matched
#: case-insensitively against the class name and its bases, so provider-specific
#: subclasses (``RateLimitError``, ``APITimeoutError``, ``ServiceUnavailable``,
#: ``ThrottlingException``, ...) are all caught without importing any of them.
_TRANSIENT_TYPE_FRAGMENTS = (
    "ratelimit",
    "toomanyrequests",
    "throttl",
    "timeout",
    "timedout",
    "serviceunavailable",
    "unavailable",
    "overloaded",
    "apiconnection",
    "connectionerror",
    "connectionreset",
    "internalserver",
    "badgateway",
    "temporarilyunavailable",
)

#: Message fragments, matched case-insensitively. Bare status numbers are
#: matched as whole words so a token count or an id containing "429" does not
#: trip them.
_TRANSIENT_MESSAGE_FRAGMENTS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "throttl",
    "service unavailable",
    "temporarily unavailable",
    "overloaded",
    "server had an error",
    "internal server error",
    "bad gateway",
    "gateway timeout",
    "connection reset",
    "connection aborted",
    "timed out",
    "timeout",
)

#: A status number in *status context* -- "Error code: 429", "HTTP 503",
#: "429 Too Many Requests". A bare number is not enough: "order 500 not found"
#: and "expected 429 items, got 3" are application errors, not throttling.
_STATUS_CODES = "408|409|425|429|500|502|503|504"
_STATUS_REASONS = (
    "too many requests|request timeout|conflict|too early|internal server error|"
    "bad gateway|service unavailable|gateway timeout"
)
_STATUS_WORD = re.compile(
    rf"\b(?:error|status|statuscode|status_code|code|http|https?error)\b\W{{0,12}}"
    rf"(?:{_STATUS_CODES})\b"
    rf"|\b(?:{_STATUS_CODES})\b\W{{0,3}}(?:{_STATUS_REASONS})"
    rf"|^\s*(?:{_STATUS_CODES})\s*$"
)

#: Exception types whose message says nothing about a provider. These are
#: raised by deterministic logic -- argument validation, a missing key, a failed
#: assertion -- so a message that happens to contain "timeout" or "429" is
#: describing the *subject* of the error, not a fault to retry.
#: ``ValueError("timeout must be positive")`` is the canonical case.
#:
#: Matched on the *exact* type, never ``isinstance``: a provider that subclasses
#: ``ValueError`` for its own errors keeps message matching, and only the bare
#: builtin is excluded. Type-name and HTTP-status checks still run for these --
#: this gate is on the message heuristic alone, which is the weakest signal.
_DETERMINISTIC_TYPES = frozenset(
    {
        ArithmeticError,
        AssertionError,
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
        LookupError,
        ModuleNotFoundError,
        NameError,
        NotImplementedError,
        RecursionError,
        StopAsyncIteration,
        StopIteration,
        SyntaxError,
        TypeError,
        UnboundLocalError,
        ValueError,
        ZeroDivisionError,
    }
)


def _status_of(exc: BaseException) -> int | None:
    """Return an HTTP status carried by ``exc`` (directly or on a response)."""
    for source in (exc, getattr(exc, "response", None)):
        if source is None:
            continue
        for attr in _STATUS_ATTRS:
            value = getattr(source, attr, None)
            if isinstance(value, bool):  # bool is an int; never a status
                continue
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
    return None


def _type_names(exc: BaseException) -> list[str]:
    """Lowercased class names of ``exc`` and its bases, punctuation stripped."""
    return [cls.__name__.replace("_", "").lower() for cls in type(exc).__mro__]


def is_transient_error(exc: BaseException) -> bool:
    """Return ``True`` when ``exc`` looks like throttling or a transient fault.

    Checked in order of decreasing reliability: an explicit HTTP status, then
    the exception's type name, then its message. Never imports a provider SDK.

    The message is only consulted for exceptions that could plausibly be
    provider-shaped -- see :data:`_DETERMINISTIC_TYPES`. A provider that signals
    throttling with a bare :class:`ValueError` and nothing else to go on would
    be missed; pass ``RetryPolicy(is_transient=...)`` for that.

    A :class:`KeyboardInterrupt` / :class:`SystemExit` / ``CancelledError`` is
    never transient -- those mean *stop*, not *try again*.
    """
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return False
    if type(exc).__name__ == "CancelledError":
        return False

    status = _status_of(exc)
    if status is not None:
        return status in _TRANSIENT_STATUS

    names = _type_names(exc)
    if any(fragment in name for name in names for fragment in _TRANSIENT_TYPE_FRAGMENTS):
        return True

    # The message is the weakest signal, so it only speaks for exceptions that
    # could plausibly have come from a provider. A deterministic builtin that
    # merely mentions a timeout is a defect to score, not a fault to retry --
    # and retrying it wastes the budget, then drops the row from the score,
    # hiding the very bug the run should surface.
    if type(exc) in _DETERMINISTIC_TYPES:
        return False

    message = str(exc).lower()
    if any(fragment in message for fragment in _TRANSIENT_MESSAGE_FRAGMENTS):
        return True
    return bool(_STATUS_WORD.search(message))


def retry_after_seconds(exc: BaseException) -> float | None:
    """Return the server's requested wait, if it supplied one.

    Reads a ``retry_after`` attribute or a ``Retry-After`` header off an
    attached response. Only the delay-seconds form is understood; the HTTP-date
    form returns ``None`` so the caller falls back to its own backoff. Negative
    or absurd values are rejected rather than trusted.
    """
    candidates: list[Any] = [getattr(exc, "retry_after", None)]
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            candidates.append(headers.get("retry-after", headers.get("Retry-After")))
        except Exception:  # a headers object that does not behave like a mapping
            pass

    for value in candidates:
        if isinstance(value, bool) or value is None:
            continue
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= seconds <= 300.0:
            return seconds
    return None


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff for transient provider errors.

    Args:
        attempts: Total tries per example, including the first. ``1`` disables
            retrying without disabling classification, so a throttled example is
            still *reported* as transient rather than as a bad answer.
        initial_backoff: Seconds to wait after the first failure.
        multiplier: Growth factor applied per subsequent failure.
        max_backoff: Ceiling on a single wait.
        jitter: Fraction of the computed delay to randomise, in ``[0, 1]``.
            Full-zero makes waits deterministic (useful in tests); the default
            spreads retries so concurrent workers do not resynchronise into the
            thundering herd that caused the throttling.
        is_transient: Classifier override. Defaults to
            :func:`is_transient_error`; supply your own to widen or narrow it.
        respect_retry_after: Prefer a server-supplied ``Retry-After`` over the
            computed backoff when one is present.

    Frozen on purpose. :data:`DEFAULT_RETRY_POLICY` is shared by every
    default-constructed harness, so a mutable policy would let
    ``harness.retry.attempts = 1`` silently reconfigure retrying for every other
    evaluation in the process. Rebind instead -- ``harness.retry =
    RetryPolicy(attempts=1)`` -- or build the harness with ``retry=``; mutating
    one now raises immediately rather than leaking.
    """

    attempts: int = 3
    initial_backoff: float = 0.5
    multiplier: float = 2.0
    max_backoff: float = 30.0
    jitter: float = 0.25
    is_transient: Callable[[BaseException], bool] = field(default=is_transient_error)
    respect_retry_after: bool = True

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        """Whether a failure on 1-based ``attempt`` earns another try."""
        if attempt >= max(1, self.attempts):
            return False
        try:
            return bool(self.is_transient(exc))
        except Exception:  # a broken custom classifier must not abort the run
            return False

    def delay_for(self, exc: BaseException, attempt: int) -> float:
        """Seconds to wait before retry number ``attempt`` (1-based)."""
        if self.respect_retry_after:
            requested = retry_after_seconds(exc)
            if requested is not None:
                return min(requested, self.max_backoff)
        delay = min(self.initial_backoff * (self.multiplier ** (attempt - 1)), self.max_backoff)
        if self.jitter > 0.0:
            spread = delay * min(self.jitter, 1.0)
            delay = max(0.0, delay + random.uniform(-spread, spread))
        return delay


#: Attribute stamped on an exception whose retries a *lower* layer already
#: spent. Nested retry loops otherwise multiply: a judge that tries three times
#: and re-raises, inside a harness that tries three times, is nine provider
#: calls for one row with the backoff reset twice -- piling on load precisely
#: while the provider is throttling.
_EXHAUSTED_MARKER = "__adapt_retries_exhausted__"


def mark_retries_exhausted(exc: BaseException) -> BaseException:
    """Record that ``exc`` already used up a retry budget. Returns ``exc``."""
    try:
        setattr(exc, _EXHAUSTED_MARKER, True)
    except Exception:  # an exception type that refuses attributes
        pass
    return exc


def retries_already_exhausted(exc: BaseException) -> bool:
    """Whether a lower layer already spent this error's retries."""
    return bool(getattr(exc, _EXHAUSTED_MARKER, False))


#: Used when a harness is constructed without an explicit policy.
DEFAULT_RETRY_POLICY = RetryPolicy()


__all__ = [
    "RetryPolicy",
    "mark_retries_exhausted",
    "retries_already_exhausted",
    "DEFAULT_RETRY_POLICY",
    "is_transient_error",
    "retry_after_seconds",
]
