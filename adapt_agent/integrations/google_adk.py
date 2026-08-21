"""Governance as Google ADK agent callbacks.

ADK's interception points are callbacks on ``LlmAgent``. Two matter here, and
both may be async and both accept a *list* of callbacks, so governance appends
to whatever the app already registered:

* ``before_model_callback(context, llm_request) -> LlmResponse | None`` --
  returning an ``LlmResponse`` **short-circuits the model call**.
* ``after_model_callback(context, llm_response) -> LlmResponse | None`` --
  returning a response replaces the model's.

.. code-block:: python

    from adapt_agent.integrations.google_adk import governance_callbacks

    agent = LlmAgent(
        name="intake", model="gemini-2.0-flash",
        **governance_callbacks(firewall=fw, agent_id="intake"),
    )

Because ADK runs a *tree* of agents, attaching callbacks per agent is the only
way to give a sub-agent reading untrusted content stricter rules than the
router above it -- an outer wrapper around the ``Runner`` cannot see inside.

Blocking mode
-------------
``on_block="raise"`` (default) raises
:class:`~adapt_agent.exceptions.SecurityBlockedError` out of the run, matching
every other ADAPT-Agent entry point. ``on_block="refuse"`` instead returns a
refusal ``LlmResponse``, which is ADK's own idiom: the agent answers with the
refusal text and the surrounding graph keeps running. Prefer ``refuse`` inside a
multi-agent tree where one blocked branch should not abort the whole run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from adapt_agent.core.governance import GovernanceGate
from adapt_agent.exceptions import SecurityBlockedError
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

#: Text returned to the model's caller when ``on_block="refuse"``.
DEFAULT_REFUSAL = "This request was blocked by a security policy."


def governance_callbacks(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    screen_output: bool = True,
    on_block: str = "raise",
    refusal_text: str = DEFAULT_REFUSAL,
) -> dict[str, Any]:
    """Build ADK callbacks applying ADAPT-Agent governance.

    Args:
        gate: A pre-built gate to share across agents; otherwise built from the
            controls below.
        firewall: Screens the outgoing request contents and the model response.
        defense: Adversarial analysis of the request contents.
        policy_enforcer: Evaluated against the request *and the callback's
            session state*, so a rule gating on session data such as
            ``state['trust_score']`` sees it.
        block_on_violation: ``False`` scans and records without blocking.
        agent_id: Named in the raised error, and in a refusal's
            ``custom_metadata["adapt_agent"]``, identifying which agent
            in the tree refused.
        screen_output: Also screen the model response.
        on_block: ``"raise"`` (default) or ``"refuse"`` -- see the module
            docstring.
        refusal_text: Body of the refusal response when ``on_block="refuse"``.

    Returns:
        A dict of ADK callback kwargs -- splat it into ``LlmAgent(...)``. Keys:
        ``before_model_callback`` and (when ``screen_output``)
        ``after_model_callback``.

    Raises:
        ValueError: If ``on_block`` is not ``"raise"`` or ``"refuse"``.
    """
    if on_block not in ("raise", "refuse"):
        raise ValueError(f"on_block must be 'raise' or 'refuse', got {on_block!r}")

    resolved = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
    )

    async def before_model_callback(context: Any, llm_request: Any) -> Any:
        # ``llm_request.contents`` is a list of genai Content objects whose
        # ``parts[*].text`` the gate reaches structurally -- no ADK import here.
        try:
            contents = getattr(llm_request, "contents", None)
            # `state=` is required, or a configured policy_enforcer never runs --
            # and the session state must be merged in, since that is where a rule
            # gating on `state['trust_score']` reads from.
            resolved.review_input(contents, state=as_state(contents, **context_state(context)))
        except SecurityBlockedError as exc:
            if on_block == "raise":
                raise
            return _refusal_response(refusal_text, resolved.agent_id, exc.threats)
        return None  # None = proceed to the model

    callbacks: dict[str, Any] = {"before_model_callback": before_model_callback}

    if screen_output:

        async def after_model_callback(context: Any, llm_response: Any) -> Any:
            try:
                resolved.review_output(getattr(llm_response, "content", llm_response))
            except SecurityBlockedError as exc:
                if on_block == "raise":
                    raise
                return _refusal_response(refusal_text, resolved.agent_id, exc.threats)
            return None  # None = keep the model's own response

        callbacks["after_model_callback"] = after_model_callback

    return callbacks


def _refusal_response(text: str, agent_id: str, threats: list[str]) -> Any:
    """Build an ADK ``LlmResponse`` carrying ``text``, short-circuiting the model.

    The refusal is an ordinary response, so unlike the raising path it carries no
    exception for the surrounding graph to inspect: two agents refusing for
    different reasons produced byte-identical objects. Attribution therefore goes
    in ``custom_metadata``, under a namespaced key so it cannot collide with the
    app's own. It stays out of ``text`` deliberately -- that is the caller's copy
    for the end user, and an agent id is internal topology.
    """
    llm_response = optional_import(
        "google.adk.models.llm_response", "google-adk", "on_block='refuse'"
    )
    genai_types = optional_import("google.genai.types", "google-adk", "on_block='refuse'")
    return llm_response.LlmResponse(
        content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)]),
        custom_metadata={"adapt_agent": {"agent_id": agent_id, "threats": list(threats)}},
    )


__all__ = ["DEFAULT_REFUSAL", "governance_callbacks"]
