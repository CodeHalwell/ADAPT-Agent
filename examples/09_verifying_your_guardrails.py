"""Example 09: Check that your guardrails actually do something.

A guardrail that can never fire is worse than none: it looks installed. This
example shows the tools for catching that *before* production, and the
fail-closed behaviour that catches it *in* production.

1.  ``Firewall.dry_run`` -- what would the firewall decide, and why, without
    recording anything. Use it to show each pattern blocks something real.
2.  ``Firewall.get_pattern_hits`` -- which blocked patterns have never matched.
3.  ``PolicyEnforcer.add_rule`` rejects a rule it could never evaluate.
4.  ``PolicyEnforcer.evaluate_rules`` -- dry-run rules against example states.
5.  Fail-closed by default: a rule that cannot reach a verdict blocks.
6.  ``PolicyEnforcer.vacuous_rules`` -- rules that have never seen their data.
7.  A refused request leaves the wrapped agent exactly as it was.

No optional dependencies and no network. Run it with:

    python examples/09_verifying_your_guardrails.py
"""

import logging

from adapt_agent.adapters import LangGraphAdapter
from adapt_agent.core import PolicyEnforcer
from adapt_agent.exceptions import SecurityBlockedError
from adapt_agent.security import Firewall

# The enforcer logs a warning every time a rule cannot be evaluated. That is
# useful in an application; here we show the same facts explicitly.
logging.getLogger("adapt_agent").setLevel(logging.ERROR)


def firewall_checks() -> None:
    print("== 1-2. Firewall: does each pattern match anything? ==")
    firewall = Firewall(max_content_length=200)
    firewall.add_blocked_pattern(r"\b\d{16}\b")  # a card-number shape
    firewall.add_blocked_pattern(r"SECRET-[A-Z]{40}")  # mistyped: real keys are 8 long

    # dry_run records no event and moves no counter, so it is safe in a test.
    for text in ["my card is 4111111111111111", "what time is it?", "x" * 201]:
        verdict = firewall.dry_run(text)
        print(f"  {text[:32]!r:36} -> allowed={verdict['allowed']} ({verdict['reason']})")
    assert firewall.get_security_events() == []

    # After real traffic, a pattern still at zero has never matched anything.
    for text in ["card 4111111111111111", "key SECRET-ABCDEFGH", "hello"]:
        firewall.check_input(text)
    print("  blocks per pattern:", firewall.get_pattern_hits())


def policy_registration() -> PolicyEnforcer:
    print("\n== 3. Policy: rules that could never be evaluated are rejected ==")
    policy = PolicyEnforcer()
    for condition in [
        "len(state['messages']) > 20",  # function calls are not in the language
        "stat['trust_score'] < 0.5",  # a typo: there is no `stat`
        "state.trust_score < 0.5",  # attribute access is not in the language
    ]:
        try:
            policy.add_rule("bad", "never registers", condition, action="block")
        except ValueError as exc:
            print(f"  rejected {condition!r}: {str(exc)[:58]}")

    policy.add_rule(
        "low_trust", "Block low-trust callers", "state['trust_score'] < 0.5", action="block"
    )
    policy.add_rule("free_tier", "Flag free-tier callers", "state['tier'] == 'free'")
    return policy


def policy_dry_run(policy: PolicyEnforcer) -> None:
    print("\n== 4. Policy: dry-run rules against example states ==")
    examples = {
        "trusted, paid": {"trust_score": 0.9, "tier": "paid"},
        "untrusted, free": {"trust_score": 0.2, "tier": "free"},
        "no trust_score at all": {"tier": "paid"},
    }
    for label, state in examples.items():
        print(f"  {label:24} -> {policy.evaluate_rules(state=state)}")
    # In your own tests: assert each blocking rule reports "violation" for one
    # realistic example and "pass" for another. "error" means the rule could
    # not see what it reads.


def fail_closed_and_vacuity(policy: PolicyEnforcer) -> None:
    print("\n== 5. Fail-closed: a rule that cannot reach a verdict blocks ==")
    state_without_score = {"messages": [], "context": {}}
    print("  default enforcer :", policy.check_state(state_without_score))

    lenient = PolicyEnforcer(fail_closed=False)
    lenient.add_rule("low_trust", "d", "state['trust_score'] < 0.5", action="block")
    print("  fail_closed=False:", lenient.check_state(state_without_score), "(silently passes)")

    print("\n== 6. vacuous_rules: rules that have never seen their data ==")
    # `free_tier` did not error above -- a missing key compares as None, so
    # `== 'free'` is quietly False. It is still a rule that has decided nothing
    # on evidence, and the enforcer has been counting.
    print("  stats  :", policy.get_rule_stats())
    print("  vacuous:", policy.vacuous_rules())
    policy.check_state({"messages": [], "context": {}, "trust_score": 0.9, "tier": "paid"})
    print("  after one request that carries both keys:", policy.vacuous_rules())


def refusal_leaves_no_trace() -> None:
    print("\n== 7. A refused request does not become the agent's state ==")
    policy = PolicyEnforcer()
    policy.add_rule("low_trust", "d", "state['trust_score'] < 0.5", action="block")

    def fake_graph(state: dict) -> dict:
        return {"messages": [{"role": "assistant", "content": "done"}]}

    agent = LangGraphAdapter(policy_enforcer=policy).wrap_agent(fake_graph)

    agent.execute({"messages": [{"role": "user", "content": "hi"}], "trust_score": 0.9})
    served = agent.get_state()
    try:
        agent.execute({"messages": [{"role": "user", "content": "again"}], "trust_score": 0.1})
    except SecurityBlockedError as exc:
        print("  refused:", exc.threats)
    print("  state unchanged by the refusal:", agent.get_state() == served)


def main() -> None:
    firewall_checks()
    policy = policy_registration()
    policy_dry_run(policy)
    fail_closed_and_vacuity(policy)
    refusal_leaves_no_trace()


if __name__ == "__main__":
    main()
