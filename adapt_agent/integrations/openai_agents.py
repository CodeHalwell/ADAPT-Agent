"""Governance as OpenAI Agents SDK guardrails.

The Agents SDK already has a first-class concept for exactly this job:

.. code-block:: python

    from adapt_agent.integrations.openai_agents import governance_guardrails

    agent = Agent(
        name="triage", instructions="...",
        **governance_guardrails(firewall=fw, agent_id="triage"),
    )

A guardrail function returns ``GuardrailFunctionOutput(output_info,
tripwire_triggered)``; a tripped input guardrail makes the SDK raise
``InputGuardrailTripwireTriggered`` (``OutputGuardrailTripwireTriggered`` on the
way out) and abort the run. That is the framework's own refusal path, so a
handoff chain reports the block the same way it reports any other, and
guardrails run concurrently with the agent rather than serialising in front of
it.

``output_info`` carries the ADAPT-Agent threat list, so a caught tripwire can
report *what* matched: ``exc.guardrail_result.output.output_info["threats"]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from adapt_agent.core.governance import GovernanceGate
from adapt_agent.integrations._common import (
    as_state,
    build_gate,
    context_state,
    optional_import,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from adapt_agent.adversarial import AdversarialDefense
    from adapt_agent.core.policy import PolicyEnforcer
    from adapt_agent.security.firewall import Firewall


def governance_guardrails(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    agent_id: str = "agent",
    screen_output: bool = True,
) -> dict[str, Any]:
    """Build OpenAI Agents guardrails applying ADAPT-Agent governance.

    Args:
        gate: A pre-built gate to share across agents; otherwise built from the
            controls below. Note that a gate's ``block_on_violation`` is not
            used here -- tripping the tripwire *is* the block, and the SDK owns
            what happens next.
        firewall: Screens the agent input and final output.
        defense: Adversarial analysis of the input.
        policy_enforcer: Evaluated against the input *and the run's runtime
            context* (``Runner.run(..., context=...)``), so a rule gating on
            authorization data such as ``state['trust_score']`` sees it.
        agent_id: Recorded in ``output_info`` so a tripwire names the agent.
        screen_output: Also register an output guardrail.

    Returns:
        A dict of ``Agent(...)`` kwargs: ``input_guardrails`` and (when
        ``screen_output``) ``output_guardrails``.

    Raises:
        ImportError: If ``openai-agents`` is not installed -- unlike the other
            integrations this one must build real ``InputGuardrail`` objects.
    """
    agents = optional_import("agents", "openai-agents", "governance_guardrails")

    # Scanning must not raise here: the SDK's tripwire is the refusal mechanism,
    # so the gate reports threats and the guardrail converts them.
    resolved = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=False,
        agent_id=agent_id,
    )

    async def adapt_input_guardrail(context: Any, agent: Any, agent_input: Any) -> Any:
        threats = resolved.scan_input(agent_input)
        # The runtime context must be merged in: a rule gating on
        # `state['trust_score']` reads what the caller passed to
        # `Runner.run(..., context=...)`, not the agent input.
        state = as_state(agent_input, **context_state(context))
        threats.extend(f"policy:{n}" for n in resolved.policy_violations(state))
        return agents.GuardrailFunctionOutput(
            output_info={"agent_id": agent_id, "threats": threats},
            tripwire_triggered=bool(threats),
        )

    guardrails: dict[str, Any] = {
        "input_guardrails": [agents.InputGuardrail(adapt_input_guardrail, name="adapt-agent")]
    }

    if screen_output:

        async def adapt_output_guardrail(context: Any, agent: Any, agent_output: Any) -> Any:
            threats = resolved.scan_output(agent_output)
            return agents.GuardrailFunctionOutput(
                output_info={"agent_id": agent_id, "threats": threats},
                tripwire_triggered=bool(threats),
            )

        guardrails["output_guardrails"] = [
            agents.OutputGuardrail(adapt_output_guardrail, name="adapt-agent")
        ]

    return guardrails


__all__ = ["governance_guardrails"]
