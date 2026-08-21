"""Governance as Claude Agent SDK hooks.

The SDK's interception point is a hook registered against an event:

.. code-block:: python

    from adapt_agent.integrations.claude_agent import governance_hooks

    options = ClaudeAgentOptions(
        hooks=governance_hooks(firewall=fw, agent_id="assistant"),
    )

A hook is ``async (input, tool_use_id, context) -> HookJSONOutput``; returning
``{"decision": "block", "reason": ...}`` stops the action and tells the model
why, which is a materially better outcome than an exception: the agent sees the
refusal and can respond to it rather than the run dying.

Three events are governed by default:

* ``UserPromptSubmit`` -- screens the prompt before the model sees it.
* ``PreToolUse`` -- screens **tool inputs**, which an outer wrapper cannot reach
  at all.
* ``PostToolUse`` -- screens **tool results**. A Claude agent loops through many
  tool calls inside a single ``query()``, and whatever a tool fetched from the
  open web comes back to the model here. ``PreToolUse`` catches an injection
  only if the model copies it into a *subsequent* tool call; if it simply reads
  the page and answers, nothing else sees the content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from adapt_agent.core.governance import GovernanceGate
from adapt_agent.integrations._common import build_gate, optional_import

if TYPE_CHECKING:  # pragma: no cover - typing only
    from adapt_agent.adversarial import AdversarialDefense
    from adapt_agent.core.policy import PolicyEnforcer
    from adapt_agent.security.firewall import Firewall

#: Hook events governed unless ``events=`` narrows them.
DEFAULT_EVENTS = ("UserPromptSubmit", "PreToolUse", "PostToolUse")

#: Events whose ``matcher`` is a *tool name* (``"Bash"``, ``"Write|Edit"``).
#: A matcher is meaningless on any other event and must not be attached to one:
#: a tool-name matcher on ``UserPromptSubmit`` describes a prompt event that can
#: never match, which would silently disable prompt screening.
_TOOL_SCOPED_EVENTS = ("PreToolUse", "PostToolUse")


def governance_hooks(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    agent_id: str = "agent",
    events: tuple[str, ...] = DEFAULT_EVENTS,
    matcher: str | None = None,
) -> dict[str, Any]:
    """Build Claude Agent SDK hooks applying ADAPT-Agent governance.

    Args:
        gate: A pre-built gate to share; otherwise built from the controls below.
            A gate's ``block_on_violation`` is unused -- the hook's ``block``
            decision is the refusal, and it is always reported to the model.
        firewall: Screens prompts and tool inputs.
        defense: Adversarial analysis of the same.
        policy_enforcer: Evaluated against the hook input as agent state.
        agent_id: Included in the block reason.
        events: Which hook events to register. Defaults to ``UserPromptSubmit``
            and ``PreToolUse``.
        matcher: Optional tool-name matcher, applied to tool-scoped events
            (``PreToolUse``/``PostToolUse``) only -- a prompt event has no tool
            name, so attaching one there would filter the hook out and leave the
            prompt unscreened. E.g. ``"Bash"`` or
            ``"Write|Edit"``. ``None`` matches everything.

    Returns:
        A ``{event: [HookMatcher]}`` dict for ``ClaudeAgentOptions(hooks=...)``.

    Raises:
        ImportError: If ``claude-agent-sdk`` is not installed -- real
            ``HookMatcher`` objects are required.
    """
    sdk = optional_import("claude_agent_sdk", "claude-agent", "governance_hooks")

    resolved = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=False,  # the hook's block decision is the refusal
        agent_id=agent_id,
    )

    async def adapt_governance_hook(
        hook_input: Any, tool_use_id: Any, context: Any
    ) -> dict[str, Any]:
        threats = resolved.scan_input(hook_input)
        if isinstance(hook_input, dict):
            threats.extend(f"policy:{n}" for n in resolved.policy_violations(hook_input))
        if not threats:
            return {}
        return {
            "decision": "block",
            "reason": (
                f"Blocked by ADAPT-Agent [{resolved.agent_id}]: {', '.join(sorted(set(threats)))}. "
                "Do not retry this content; tell the user it was refused."
            ),
        }

    return {
        event: [
            sdk.HookMatcher(
                # Only a tool-scoped event has a tool name to match on.
                matcher=matcher if event in _TOOL_SCOPED_EVENTS else None,
                hooks=[adapt_governance_hook],
            )
        ]
        for event in events
    }


__all__ = ["DEFAULT_EVENTS", "governance_hooks"]
