"""Governance as LangGraph pre/post-model hooks.

``create_react_agent`` accepts two hook *nodes* that run either side of the
model call. A node takes the graph state and returns a state update (``{}``
meaning "no change"), which is all governance needs:

.. code-block:: python

    from adapt_agent.integrations.langgraph import governance_hooks

    agent = create_react_agent(
        model, tools,
        **governance_hooks(firewall=fw, agent_id="researcher"),
    )

Attaching hooks per node is what an outer wrapper cannot do. A supervisor graph
runs several ``create_react_agent`` nodes; wrapping the compiled graph screens
only the first input and the last output, while these hooks fire on **every**
model call inside it -- including one whose input came from a tool result rather
than from the user, which is exactly where injected content arrives.

For a hand-built ``StateGraph`` there is no hook parameter; use
:func:`governance_node` and add it as a node yourself, or wrap the compiled
graph with :class:`~adapt_agent.adapters.LangGraphAdapter`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from adapt_agent.core.governance import GovernanceGate
from adapt_agent.integrations._common import build_gate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from adapt_agent.adversarial import AdversarialDefense
    from adapt_agent.core.policy import PolicyEnforcer
    from adapt_agent.security.firewall import Firewall


def governance_hooks(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    screen_output: bool = True,
) -> dict[str, Callable[[Any], dict[str, Any]]]:
    """Build ``pre_model_hook`` / ``post_model_hook`` nodes applying governance.

    Args:
        gate: A pre-built gate to share across nodes; otherwise built from the
            controls below.
        firewall: Screens the state's messages before and after the model call.
        defense: Adversarial analysis of the inbound messages.
        policy_enforcer: Evaluated against the graph state -- which here is the
            *real* graph state, so a rule may reference any key the graph
            carries, not just the ``messages``/``context`` an adapter exposes.
        block_on_violation: ``False`` scans without raising.
        agent_id: Named in the raised error.
        screen_output: Also register a ``post_model_hook``.

    Returns:
        A dict of ``create_react_agent(...)`` kwargs.
    """
    resolved = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
    )

    hooks: dict[str, Callable[[Any], dict[str, Any]]] = {
        "pre_model_hook": governance_node(resolved, direction="input")
    }
    if screen_output:
        hooks["post_model_hook"] = governance_node(resolved, direction="output")
    return hooks


def governance_node(
    gate: GovernanceGate, *, direction: str = "input"
) -> Callable[[Any], dict[str, Any]]:
    """Return a LangGraph node that screens the state and returns no update.

    Usable directly in a hand-built graph::

        builder.add_node("guard", governance_node(gate))
        builder.add_edge("guard", "model")

    Args:
        gate: The configured :class:`~adapt_agent.core.governance.GovernanceGate`.
        direction: ``"input"`` screens with the input controls and evaluates
            policy; ``"output"`` screens with the output controls only.

    Raises:
        ValueError: If ``direction`` is not ``"input"`` or ``"output"``.
    """
    if direction not in ("input", "output"):
        raise ValueError(f"direction must be 'input' or 'output', got {direction!r}")

    def _node(state: Any) -> dict[str, Any]:
        if direction == "input":
            gate.review_input(_messages(state), state=state if isinstance(state, dict) else None)
        else:
            gate.review_output(_messages(state))
        return {}  # governance never rewrites state

    _node.__name__ = f"adapt_governance_{direction}"
    return _node


def _messages(state: Any) -> Any:
    """The messages of a graph state, or the whole state when it has none."""
    if isinstance(state, dict) and "messages" in state:
        return state["messages"]
    return state


__all__ = ["governance_hooks", "governance_node"]
