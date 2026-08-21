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

**Guardrails do not cover a handoff target.** The SDK runs input guardrails for
the *starting* agent only -- ``run.py`` gates them on ``current_turn == 0`` --
so a specialist reached by a handoff never runs its own, and content transferred
to it is screened by whatever the entry agent's guardrail saw, not by the
specialist's rules. Verified against the installed SDK by driving a real handoff:

.. code-block:: text

    input guardrails that ran: ['triage']
    handoff happened: True (final agent: specialist)
    specialist's own input guardrail ran: False

For a specialist, use :func:`governance_agent_hooks` instead, which binds to
``AgentHooks.on_llm_start`` -- per agent, per model call, and therefore also on
the way back from a tool.
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

    Note:
        Input guardrails run for the *starting* agent of a run only. An agent
        reachable by handoff needs :func:`governance_agent_hooks`, or its input
        governance silently never runs.

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
            output_info={"agent_id": resolved.agent_id, "threats": threats},
            tripwire_triggered=bool(threats),
        )

    guardrails: dict[str, Any] = {
        "input_guardrails": [agents.InputGuardrail(adapt_input_guardrail, name="adapt-agent")]
    }

    if screen_output:

        async def adapt_output_guardrail(context: Any, agent: Any, agent_output: Any) -> Any:
            threats = resolved.scan_output(agent_output)
            return agents.GuardrailFunctionOutput(
                output_info={"agent_id": resolved.agent_id, "threats": threats},
                tripwire_triggered=bool(threats),
            )

        guardrails["output_guardrails"] = [
            agents.OutputGuardrail(adapt_output_guardrail, name="adapt-agent")
        ]

    return guardrails


def governance_agent_hooks(
    *,
    gate: GovernanceGate | None = None,
    firewall: Any = None,
    defense: Any = None,
    policy_enforcer: Any = None,
    agent_id: str = "agent",
    screen_output: bool = True,
    inner: Any = None,
) -> Any:
    """Governance on ``AgentHooks``, which runs for **every** agent in a run.

    :func:`governance_guardrails` is the SDK's own idiom and the right default
    for a run's entry point, but the SDK gates input guardrails on
    ``current_turn == 0``: a specialist reached by a handoff never runs its own.
    A control that is configured, documented, and silently skipped is the worst
    shape a security control can take, so this is the seam for those agents.

    ``on_llm_start`` fires before every model call *for the agent it is attached
    to* and receives the input items about to be sent; ``on_end`` fires when the
    agent finishes and receives its answer, so both directions are covered. Confirmed against the
    installed SDK, including through a handoff::

        hooks fired: ['triage:on_start', 'triage:on_llm_start(items=1)',
                      'SPECIALIST:on_start', 'SPECIALIST:on_llm_start(items=3)']

    Because it runs per model call rather than per run, it also screens tool
    results on their way back to the model -- the path that carries whatever a
    tool fetched from the open web.

    A block raises :class:`~adapt_agent.exceptions.SecurityBlockedError`, which
    aborts the run. That is a harder stop than a tripwire; use guardrails where
    you want the SDK's own refusal path.

    Args:
        gate: A pre-built gate to share across agents; otherwise built from the
            controls below.
        firewall: Screens the items about to reach the model.
        defense: Adversarial analysis of those items.
        policy_enforcer: Evaluated against the items plus the run's runtime
            context (``Runner.run(..., context=...)``).
        agent_id: Named in the raised error.
        screen_output: Also screen the agent's final answer, on ``on_end``.
        inner: An existing ``AgentHooks`` to delegate to, so an app's own
            lifecycle hooks still run. Governance is applied first.

    Returns:
        An ``AgentHooks`` instance for ``Agent(hooks=...)``.

    Raises:
        ImportError: If ``openai-agents`` is not installed.
    """
    agents = optional_import("agents", "openai-agents", "governance_agent_hooks")
    resolved = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=True,
        agent_id=agent_id,
    )

    base: Any = agents.AgentHooks  # resolved at call time; never imported at module load

    class AdaptGovernanceHooks(base):
        async def on_llm_start(
            self, context: Any, agent: Any, system_prompt: Any, input_items: Any
        ) -> None:
            resolved.review_input(
                input_items, state=as_state(input_items, **context_state(context))
            )
            if inner is not None:
                await inner.on_llm_start(context, agent, system_prompt, input_items)

        async def on_end(self, context: Any, agent: Any, output: Any) -> None:
            # The other half of the seam. Guardrails give a handoff target
            # neither input *nor* output screening; covering only the input made
            # this factory the very thing it was added to replace -- a control
            # that looks complete and is half missing.
            if screen_output:
                resolved.review_output(output)
            if inner is not None:
                await inner.on_end(context, agent, output)

    hooks = AdaptGovernanceHooks()
    if inner is not None:
        # Every *other* lifecycle hook belongs to the app. `__getattr__` cannot
        # forward them: the base class defines them, so lookup never misses and
        # the app's would be silently shadowed by the SDK's no-ops -- configured,
        # and doing nothing. Bind them on the instance, where they win.
        governed = {"on_llm_start", "on_end"}
        for name in dir(base):
            if name.startswith("on_") and name not in governed and hasattr(inner, name):
                setattr(hooks, name, getattr(inner, name))
    return hooks


__all__ = ["governance_agent_hooks", "governance_guardrails"]
