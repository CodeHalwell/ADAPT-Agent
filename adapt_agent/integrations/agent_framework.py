"""Governance as Microsoft Agent Framework middleware.

MAF's interception point is an **agent middleware**: an async callable

.. code-block:: python

    async def middleware(context: AgentContext, call_next) -> None: ...

registered with ``Agent(..., middleware=[...])``. ``call_next`` takes no
arguments; the response lands on ``context.result`` once it returns.

This is the recommended way to govern a MAF application, and the only practical
way to govern a **workflow**. ``WorkflowBuilder(...).build().as_agent()`` will
duck-type through :class:`~adapt_agent.adapters.MicrosoftAgentFrameworkAdapter`,
but wrapping it governs only the outer boundary -- the raw request in, the final
answer out. Attaching middleware to each specialist instead means an intake
router and a specialist reading untrusted content can carry different rules, and
governance composes with the token-usage middleware an app already stacks:

.. code-block:: python

    from adapt_agent.integrations.agent_framework import governance_middleware

    agent = chat_client.create_agent(
        instructions="...",
        middleware=[
            usage_middleware("nos"),                       # the app's own
            governance_middleware(firewall=fw, agent_id="nos"),
        ],
    )

Being async-native, this path never touches the sync/async bridge that
``execute`` needs, so it works unchanged inside a running event loop.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from adapt_agent.core.governance import GovernanceGate
from adapt_agent.integrations._common import build_gate, traced

if TYPE_CHECKING:  # pragma: no cover - typing only
    from adapt_agent.adversarial import AdversarialDefense
    from adapt_agent.core.policy import PolicyEnforcer
    from adapt_agent.security.firewall import Firewall


def governance_middleware(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    screen_output: bool = True,
    observer: Any = None,
    operation: str = "agent_framework.run",
) -> Callable[..., Awaitable[None]]:
    """Build MAF agent middleware that applies ADAPT-Agent governance.

    Args:
        gate: A pre-built :class:`~adapt_agent.core.governance.GovernanceGate`
            to share across agents. When omitted one is built from the controls
            below.
        firewall: Screens the inbound messages and the outbound response.
        defense: Adversarial/prompt-injection analysis of inbound messages.
        policy_enforcer: Evaluated against the messages as agent state. Content
            rules belong on the ``firewall``; policy rules gate on state.
        block_on_violation: ``True`` (default) raises
            :class:`~adapt_agent.exceptions.SecurityBlockedError` on a hit.
            ``False`` runs the agent anyway -- useful to shake out false
            positives before enforcing.
        agent_id: Named in the raised error, so a refusal inside a workflow
            identifies which agent produced it.
        screen_output: Also screen ``context.result`` after the agent responds.
        observer: Optional :class:`~adapt_agent.observability.AgentObserver`.
            Inside a graph this is what attributes a span to the specific agent,
            since one outer wrapper cannot tell the specialists apart.
        operation: Trace operation label recorded by the observer.

    Returns:
        An async ``(context, call_next)`` callable for ``Agent(middleware=[...])``.
        When ``agent_framework`` is installed the returned function is tagged
        with its ``@agent_middleware`` decorator so MAF categorises it without
        needing a type annotation; without the SDK the plain callable is
        returned and remains directly unit-testable.
    """
    resolved = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
    )

    async def adapt_governance(context: Any, call_next: Callable[[], Awaitable[None]]) -> None:
        # ``context.messages`` is a list[Message]; Message.text is picked up by
        # the gate's structural text extraction, so no MAF import is needed.
        resolved.review_input(list(getattr(context, "messages", []) or []))
        with traced(observer, agent_id, operation):
            await call_next()
        if screen_output:
            resolved.review_output(getattr(context, "result", None))

    return _mark_agent_middleware(adapt_governance)


def _mark_agent_middleware(func: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
    """Tag ``func`` as agent middleware when MAF is importable.

    MAF decides whether a bare callable is agent/chat/function middleware from
    either its ``@agent_middleware`` decorator marker or the *name* of its first
    parameter's annotation. A duck-typed integration has neither, so without
    this tag MAF raises ``MiddlewareException: Cannot determine middleware type``.
    Applying the real decorator is exact; the import is deferred to call time and
    is optional, which keeps this module importable with no SDK installed.
    """
    try:
        from agent_framework import agent_middleware
    except Exception:  # pragma: no cover - exercised without the SDK
        return func
    return agent_middleware(func)  # type: ignore[no-any-return]


__all__ = ["governance_middleware"]
