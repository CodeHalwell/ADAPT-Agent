"""Shared plumbing for the native-hook integrations."""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from adapt_agent.core.governance import GovernanceGate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from adapt_agent.adversarial import AdversarialDefense
    from adapt_agent.core.policy import PolicyEnforcer
    from adapt_agent.security.firewall import Firewall


#: Default ``agent_id`` for a binding that does not name itself. A gate passed
#: in keeps its own label unless the binding overrides it -- see `build_gate`.
_DEFAULT_AGENT_ID = "agent"


def build_gate(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = _DEFAULT_AGENT_ID,
) -> GovernanceGate:
    """Return the supplied gate, or build one from individual controls.

    Every integration factory accepts both forms: pass ``gate=`` to share one
    configured gate across several agents, or pass the controls directly for a
    one-off. Passing ``gate=`` alongside controls ignores the controls -- the
    gate already carries its own.

    ``agent_id`` is the exception, because it identifies the *binding* rather
    than the controls. Sharing one gate across a graph is exactly the case where
    each agent needs its own label: returning the gate unchanged raised errors
    naming the shared gate while the hook traced the same invocation under the
    binding's id, so one violation was attributed two different ways.
    """
    if gate is not None:
        if agent_id == _DEFAULT_AGENT_ID or agent_id == gate.agent_id:
            return gate
        return GovernanceGate(
            firewall=gate.firewall,
            defense=gate.defense,
            policy_enforcer=gate.policy_enforcer,
            block_on_violation=gate.block_on_violation,
            agent_id=agent_id,
        )
    return GovernanceGate(
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
    )


@contextlib.contextmanager
def traced(observer: Any, agent_id: str, operation: str) -> Iterator[None]:
    """Open an :class:`~adapt_agent.observability.AgentObserver` span, if given.

    A native hook sits *inside* the framework's execution, so the framework owns
    running the agent -- but the observer stage of the governance pipeline still
    applies, and inside a multi-agent graph it is what attributes a span to the
    specific agent that produced it. ``observer=None`` makes this a no-op.
    """
    if observer is None:
        yield
        return
    trace_id = uuid.uuid4().hex
    observer.start_trace(trace_id, agent_id, operation)
    try:
        yield
    except BaseException as exc:
        # BaseException, not Exception: `asyncio.CancelledError` derives from it,
        # so a cancelled request would otherwise leave the span open forever.
        observer.end_trace(trace_id, status="error", result=str(exc))
        raise
    observer.end_trace(trace_id, status="completed")


def optional_import(module: str, extra: str, needed_for: str) -> Any:
    """Import a framework module, or raise a message that names the fix.

    Integrations are duck-typed and work without their SDK installed; this is
    only for the few places that genuinely need a framework *type* (constructing
    an ADK refusal response, wrapping an OpenAI guardrail object).
    """
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as exc:  # pragma: no cover - exercised without the SDK
        raise ImportError(
            f"{needed_for} needs the {module!r} package: " f"pip install 'adapt-agent[{extra}]'."
        ) from exc


def context_state(context: Any) -> dict[str, Any]:
    """Extract a framework callback context's own state mapping, if it has one.

    A policy rule typically gates on session data -- ``state['trust_score']`` --
    which lives on the framework's context, not in the model request. Without
    this the key is simply absent, and a fail-open enforcer reads that as "no
    violation": the rule never fires and nothing says so.
    """
    # ADK: `Context.state`; some versions expose it via the session instead.
    state = getattr(context, "state", None)
    if isinstance(state, Mapping):
        return dict(state)
    session = getattr(context, "session", None)
    session_state = getattr(session, "state", None)
    if isinstance(session_state, Mapping):
        return dict(session_state)
    # OpenAI Agents: `RunContextWrapper.context` is whatever the caller passed to
    # `Runner.run(..., context=...)` -- the conventional home for authorization
    # data a policy rule gates on.
    runtime = getattr(context, "context", None)
    if isinstance(runtime, Mapping):
        return dict(runtime)
    return {}


def as_state(messages: Any, /, **extra: Any) -> dict[str, Any]:
    """Shape framework messages into the mapping a policy rule expects.

    :meth:`GovernanceGate.review_input <adapt_agent.core.governance.GovernanceGate.review_input>`
    only evaluates policy when it is given a ``state``. Every hook must pass one,
    or a configured ``policy_enforcer`` is silently inert -- accepted, documented
    and doing nothing, which is the worst possible failure mode for a security
    control.

    ``extra`` is the framework's own session or runtime state, splatted in by the
    hooks. Its keys are arbitrary -- an app is free to keep one called
    ``messages`` -- so the first parameter is **positional-only**: named
    ``messages`` it would collide, and every benign request through the ADK and
    OpenAI hooks would raise ``TypeError`` before governance ever ran. For the
    same reason the messages being *screened* win the merge: a session's own
    ``messages`` entry must not quietly replace the ones under inspection, or a
    message-based rule evaluates something other than the request in hand.
    """
    if isinstance(messages, dict):
        state = dict(messages)
        state.setdefault("messages", [])
        state.setdefault("context", {})
    elif isinstance(messages, (list, tuple)):
        state = {"messages": list(messages), "context": {}}
    elif messages is None:
        state = {"messages": [], "context": {}}
    else:
        state = {"messages": [messages], "context": {}}
    screened = state["messages"]
    state.update(extra)
    state["messages"] = screened
    return state


__all__ = ["as_state", "build_gate", "context_state", "optional_import", "traced"]
