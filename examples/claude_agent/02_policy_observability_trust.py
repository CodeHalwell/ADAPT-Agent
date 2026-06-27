"""Example 02 (Claude Agent SDK): policy, defense, observability, trust & taint.

Building on example 01, this layers in the rest of ADAPT-Agent's *runtime guard*
around a Claude Agent SDK ``query`` call:

* ``PolicyEnforcer`` -- a SAFE, eval-free rule language over the extracted
  ``message`` and ``state``. Only ``action="block"`` rules actually stop a run;
  ``action="warn"`` rules are recorded but non-blocking.
* ``AdversarialDefense`` -- heuristics for jailbreaks / injection beyond the
  Firewall's literal patterns.
* ``AgentObserver`` -- records a trace per ``execute`` (start/stop/status) so you
  can see exactly what ran.
* ``Middleware`` -- a pre/post hook pipeline that can rewrite the payload.
* ``block_on_violation=False`` -- show the guard *recording* threats while still
  letting the (benign) run complete, which is what you want in shadow/audit mode.

It also demonstrates the two standalone trust primitives, which are not adapter
constructor args but cooperate with it:

* ``TrustManager`` -- a per-agent trust score you update from outcomes.
* ``TaintTracker`` -- marks untrusted data and propagates the taint.

Runs with NO API key and NO SDK installed via the same ``fake_query`` shape as
example 01.

Run it with:

    python examples/claude_agent/02_policy_observability_trust.py
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

try:
    import claude_agent_sdk  # noqa: F401
except ImportError:
    claude_agent_sdk = None  # type: ignore[assignment]

from adapt_agent import (
    AdversarialDefense,
    AgentObserver,
    Firewall,
    Middleware,
    PolicyEnforcer,
    TaintTracker,
    TrustManager,
)
from adapt_agent.adapters import ClaudeAgentSDKAdapter
from adapt_agent.exceptions import SecurityBlockedError
from adapt_agent.security.taint_tracker import TaintLevel


# --- Fake SDK stream (same shape as example 01) ---------------------------- #
class _TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _AssistantMessage:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class _ResultMessage:
    def __init__(self, result: str) -> None:
        self.result = result


async def fake_query(*, prompt: str, options: Any = None) -> AsyncIterator[Any]:
    """An async generator shaped like ``claude_agent_sdk.query``."""
    answer = f"Acknowledged: {prompt!r}"
    yield _AssistantMessage(content=[_TextBlock(answer)])
    yield _ResultMessage(result=answer)


class TagInputMiddleware(Middleware):
    """A trivial pre/post middleware that annotates the payload.

    Middleware runs *after* policy enforcement and *before* the agent (input
    side), then again over ``{"result": ...}`` (output side). Real middleware
    might redact PII, attach request IDs, or enforce per-tenant quotas.
    """

    def process_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, "_audited": True}

    def process_output(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload


def build_policy() -> PolicyEnforcer:
    """A PolicyEnforcer with one blocking rule and one warn-only rule.

    Conditions use the SAFE expression language (no ``eval`` -- only literals,
    indexing, ``in``, comparisons and boolean ops). The adapter evaluates rules
    against the extracted :class:`AgentState`, so conditions reference ``state``
    (here ``state['messages'][0]['content']`` -- the user turn). The expression
    sandbox rejects function calls and negative indices, which is why we index
    ``[0]`` rather than ``[-1]`` and avoid helpers like ``any(...)``.
    """
    policy = PolicyEnforcer()
    policy.add_rule(
        name="no_credentials",
        description="Block requests that mention passwords/secrets.",
        condition=(
            "'password' in state['messages'][0]['content'] "
            "or 'secret' in state['messages'][0]['content']"
        ),
        action="block",  # only action="block" actually stops a run
        severity="high",
    )
    policy.add_rule(
        name="discourage_urgency",
        description="Flag (but allow) high-pressure 'urgent' phrasing.",
        condition="'URGENT' in state['messages'][0]['content']",
        action="warn",  # recorded, never blocks
        severity="low",
    )
    return policy


def build_adapter(*, block: bool) -> ClaudeAgentSDKAdapter:
    """Assemble the full guard. ``block`` toggles ``block_on_violation``."""
    firewall = Firewall(max_content_length=10_000)
    firewall.add_blocked_pattern(r"(?i)ignore[\w ]*?instructions")
    return ClaudeAgentSDKAdapter(
        firewall=firewall,
        defense=AdversarialDefense(),
        policy_enforcer=build_policy(),
        observer=AgentObserver(),
        middleware=TagInputMiddleware(),
        agent_id="claude-guarded",
        block_on_violation=block,
    )


def main() -> None:
    # ---- Blocking mode: the policy stops a credential-seeking request ----- #
    print("=== Blocking mode (block_on_violation=True) ===")
    blocking_adapter = build_adapter(block=True)
    guarded = blocking_adapter.wrap_agent(fake_query)

    print("- safe request:")
    safe = {"messages": [{"role": "user", "content": "Summarize the project status."}]}
    print("   result:", guarded.execute(safe))

    print("- policy-violating request (asks for a password):")
    bad = {"messages": [{"role": "user", "content": "Email me the admin password please."}]}
    try:
        guarded.execute(bad)
    except SecurityBlockedError as exc:
        print(f"   Blocked! reason={exc.reason!r} threats={exc.threats}")

    print("\n  Observer traces:")
    for trace in blocking_adapter.observer.get_traces():
        print(
            f"   trace {trace['trace_id'][:8]} "
            f"operation={trace['operation']} status={trace['status']}"
        )

    # ---- Non-blocking (audit/shadow) mode --------------------------------- #
    # With block_on_violation=False the guard still screens input and records
    # warnings/threats, but a benign run completes. This is how you roll a new
    # rule out in "log only" mode before enforcing it.
    print("\n=== Audit mode (block_on_violation=False) ===")
    audit_adapter = build_adapter(block=False)
    audited = audit_adapter.wrap_agent(fake_query)

    urgent = {"messages": [{"role": "user", "content": "URGENT: send the quarterly numbers."}]}
    result = audited.execute(urgent)  # the warn-only rule fires but does NOT block
    print("  Completed despite the 'URGENT' warn rule. Result:", result)

    # In audit mode even a *block* rule does not stop the run -- handy proof:
    cred = {"messages": [{"role": "user", "content": "Send me the secret API password."}]}
    audited.execute(cred)
    print("  Completed despite the credential 'block' rule (audit mode never blocks).")

    # Inspect exactly which rules a given input state trips (warn or block alike)
    # by checking the PolicyEnforcer directly against the request state. This is
    # what an audit pipeline logs before any rule is switched to enforcing.
    for req in (urgent, cred):
        state = audit_adapter.extract_state(req)
        triggered = audit_adapter.policy_enforcer.check_state(state)
        actions = {n: audit_adapter.policy_enforcer.get_rule(n)["action"] for n in triggered}
        print(f"  {req['messages'][0]['content'][:32]!r} -> {actions}")

    # ---- TrustManager: a per-agent reputation score ----------------------- #
    print("\n=== TrustManager ===")
    trust = TrustManager(initial_trust=0.5)
    agent_id = "claude-guarded"
    print(f"  initial trust({agent_id}) = {trust.get_trust_score(agent_id):.2f}")
    # Reward a clean run, penalize a blocked one. `delta` is positional; an
    # optional `reason` is recorded in the trust history.
    trust.update_trust_score(agent_id, +0.2, reason="clean run")
    trust.update_trust_score(agent_id, -0.4, reason="policy block")
    print(f"  after +0.2 / -0.4    = {trust.get_trust_score(agent_id):.2f}")
    print(f"  is_trusted(>=0.5)?    {trust.is_trusted(agent_id)}")

    # ---- TaintTracker: track untrusted data flow -------------------------- #
    print("\n=== TaintTracker ===")
    taint = TaintTracker()
    # 1. Register the untrusted origin (e.g. a scraped web page).
    taint.register_source("web", source_type="external", level=TaintLevel.HIGH)
    # 2. Mark a piece of data as tainted by that source.
    taint.mark_tainted("scraped_bio", source_ids=["web"])
    print("  is_tainted('scraped_bio')?", taint.is_tainted("scraped_bio"))
    # 3. Taint propagates to anything derived from it.
    taint.propagate_taint("scraped_bio", "summary_of_bio", operation="summarize")
    print("  is_tainted('summary_of_bio')?", taint.is_tainted("summary_of_bio"))
    print("  taint stats:", taint.get_stats())


if __name__ == "__main__":
    main()
