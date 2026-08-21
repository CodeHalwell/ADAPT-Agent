"""Governance as CrewAI kickoff callbacks and task guardrails.

CrewAI exposes two list-valued hooks on a ``Crew`` -- both synchronous, both
free to rewrite what passes through:

* ``before_kickoff_callbacks``: ``(inputs: dict) -> dict``
* ``after_kickoff_callbacks``: ``(result) -> result``

.. code-block:: python

    from adapt_agent.integrations.crewai import governance_callbacks

    crew = Crew(
        agents=[...], tasks=[...],
        **governance_callbacks(firewall=fw, agent_id="research-crew"),
    )

Being lists, these compose with callbacks the app already registers.

Crew-level callbacks fire once per kickoff, so for per-task screening inside a
long crew run, add :func:`governance_guardrail` to the individual ``Task``
objects that handle untrusted content -- a task guardrail returns
``(ok, value_or_message)`` and CrewAI retries or fails the task on ``False``.
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


def governance_callbacks(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    screen_output: bool = True,
) -> dict[str, list[Callable[[Any], Any]]]:
    """Build CrewAI kickoff callbacks applying ADAPT-Agent governance.

    Args:
        gate: A pre-built gate to share; otherwise built from the controls below.
        firewall: Screens the kickoff inputs and the crew output.
        defense: Adversarial analysis of the inputs.
        policy_enforcer: Evaluated against the kickoff inputs as agent state.
        block_on_violation: ``False`` scans without raising.
        agent_id: Named in the raised error.
        screen_output: Also register an after-kickoff callback.

    Returns:
        A dict of ``Crew(...)`` kwargs: ``before_kickoff_callbacks`` and (when
        ``screen_output``) ``after_kickoff_callbacks``.
    """
    resolved = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
    )

    def before_kickoff(inputs: Any) -> Any:
        resolved.review_input(inputs, state=inputs if isinstance(inputs, dict) else None)
        return inputs  # governance never rewrites the inputs

    callbacks: dict[str, list[Callable[[Any], Any]]] = {
        "before_kickoff_callbacks": [before_kickoff]
    }

    if screen_output:

        def after_kickoff(result: Any) -> Any:
            resolved.review_output(result)
            return result

        callbacks["after_kickoff_callbacks"] = [after_kickoff]

    return callbacks


def governance_guardrail(gate: GovernanceGate) -> Callable[[Any], tuple[bool, Any]]:
    """Return a CrewAI ``Task(guardrail=...)`` that screens a task's output.

    A task guardrail returns ``(True, output)`` to accept or ``(False, message)``
    to reject -- CrewAI feeds the message back and retries up to
    ``guardrail_max_retries``. Use this on the tasks that touch untrusted
    content, where a crew-level callback would fire too late.
    """

    def _guardrail(output: Any) -> tuple[bool, Any]:
        threats = gate.scan_output(output)
        if threats:
            return False, (
                f"Output blocked by ADAPT-Agent [{gate.agent_id}]: "
                f"{', '.join(sorted(set(threats)))}. Rewrite without the flagged content."
            )
        return True, output

    return _guardrail


__all__ = ["governance_callbacks", "governance_guardrail"]
