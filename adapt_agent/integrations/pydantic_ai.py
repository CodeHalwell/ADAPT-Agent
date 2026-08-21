"""Governance as a Pydantic AI output validator.

Pydantic AI is the one supported framework with only *half* a seam, and this
module says so rather than inventing the other half.

**Output: native.** ``@agent.output_validator`` runs on every result, may be
async, and raising from it aborts the run:

.. code-block:: python

    from adapt_agent.integrations.pydantic_ai import install_governance

    agent = Agent("openai:gpt-4o", output_type=Triage)
    install_governance(agent, firewall=fw, agent_id="triage")

**Input: no native hook.** ``Agent.__init__`` has no pre-run interception point
(``instructions`` and ``system_prompt`` shape the prompt but cannot refuse one),
so there is nothing to plug into. Screen inputs with the adapter instead --
:class:`~adapt_agent.adapters.PydanticAIAdapter` gives you ``execute`` and
``aexecute`` with the full pipeline -- or call
:meth:`GovernanceGate.review_input <adapt_agent.core.governance.GovernanceGate.review_input>`
yourself before ``agent.run``. Combining both is fine and is the recommended
setup: the adapter screens what goes in, the validator screens what comes out.

A structured ``output_type`` is screened field by field: the gate reaches the
text inside a Pydantic model, so a prompt-injection string smuggled into one
field of a structured answer is still caught.

**Unsupported controls are refused, not ignored.** Passing ``policy_enforcer``
or ``defense`` here raises. A policy rule gates on agent state and adversarial
defense analyses *input*; an output-only seam sees neither, so honouring them is
impossible, and accepting them quietly would leave you believing a control was
active when nothing ever consulted it. Enforce both through the adapter; the two
compose.
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


def governance_output_validator(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
) -> Callable[[Any], Any]:
    """Build an output validator that screens a Pydantic AI result.

    Returns:
        A single-argument callable returning the output unchanged, or raising
        :class:`~adapt_agent.exceptions.SecurityBlockedError`. Pydantic AI
        accepts validators with or without a leading ``RunContext``; this one
        takes just the output, which is the form it applies to any single-arg
        callable.

    Register it with :func:`install_governance`, or by hand::

        agent.output_validator(governance_output_validator(firewall=fw))

    Raises:
        ValueError: If a ``policy_enforcer`` or ``defense`` is supplied
            (directly or on a ``gate``). See the module docstring -- this seam
            cannot honour either, and silently ignoring one would be worse than
            refusing it.
    """
    resolved = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
    )
    _reject_unsupported(resolved)

    def adapt_governance_validator(output: Any) -> Any:
        resolved.review_output(output)
        return output

    return adapt_governance_validator


def _reject_unsupported(gate: GovernanceGate) -> None:
    """Refuse the controls this output-only seam cannot honour.

    An output validator sees only the result, so two of the gate's controls are
    structurally out of reach here:

    * ``policy_enforcer`` gates on agent *state*, which is never in scope;
    * ``defense`` is *input* analysis -- :meth:`GovernanceGate.scan_output` runs
      the firewall only, deliberately, since adversarial-instruction detection
      over an answer flags an agent legitimately quoting an instruction.

    Accepting either and quietly doing nothing is the failure mode that makes a
    security control dangerous: it looks configured. So this raises instead of
    pretending. Use :class:`~adapt_agent.adapters.PydanticAIAdapter` for both --
    the adapter and this validator compose.
    """
    unsupported = [
        name
        for name, value in (
            ("policy_enforcer", gate.policy_enforcer),
            ("defense", gate.defense),
        )
        if value is not None
    ]
    if unsupported:
        raise ValueError(
            f"Pydantic AI's output validator cannot honour {', '.join(unsupported)}: "
            "a policy rule gates on agent state and adversarial defense analyses "
            "*input*, neither of which an output-only seam sees. Use "
            "PydanticAIAdapter (execute/aexecute) for those, and keep this "
            "validator for output screening -- the two compose. See "
            "adapt_agent.integrations.pydantic_ai."
        )


def install_governance(
    agent: Any,
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
) -> Any:
    """Register output governance on ``agent`` in place and return it.

    Remember this covers the **output** only -- see the module docstring for why,
    and what to pair it with for inputs.

    Raises:
        TypeError: If ``agent`` has no ``output_validator`` (i.e. it is not a
            Pydantic AI agent).
        ValueError: If a ``policy_enforcer`` or ``defense`` is supplied -- see
            :func:`governance_output_validator`.
    """
    register = getattr(agent, "output_validator", None)
    if not callable(register):
        raise TypeError(
            "install_governance expects a Pydantic AI Agent (with an "
            f"`output_validator` method), got {type(agent).__name__}."
        )
    register(
        governance_output_validator(
            gate=gate,
            firewall=firewall,
            defense=defense,
            policy_enforcer=policy_enforcer,
            block_on_violation=block_on_violation,
            agent_id=agent_id,
        )
    )
    return agent


__all__ = ["governance_output_validator", "install_governance"]
