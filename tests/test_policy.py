"""Tests for the PolicyEnforcer."""

import pytest

from adapt_agent.core import PolicyEnforcer


def test_add_rule_and_get_rule():
    enforcer = PolicyEnforcer()
    enforcer.add_rule(
        name="r1",
        description="desc",
        condition="True == True",
        action="warn",
        severity="low",
    )
    rule = enforcer.get_rule("r1")
    assert rule is not None
    assert rule["name"] == "r1"
    assert rule["condition"] == "True == True"
    assert enforcer.get_rule("missing") is None


def test_add_rule_condition_too_long_raises():
    enforcer = PolicyEnforcer()
    long_condition = "1" * 1025
    with pytest.raises(ValueError):
        enforcer.add_rule(
            name="big",
            description="d",
            condition=long_condition,
        )


def test_remove_rule_true_and_false():
    enforcer = PolicyEnforcer()
    enforcer.add_rule(name="r", description="d", condition="True")
    assert enforcer.remove_rule("r") is True
    assert enforcer.remove_rule("r") is False


def test_list_rules():
    enforcer = PolicyEnforcer()
    assert enforcer.list_rules() == []
    enforcer.add_rule(name="a", description="d", condition="True")
    enforcer.add_rule(name="b", description="d", condition="True")
    names = {r["name"] for r in enforcer.list_rules()}
    assert names == {"a", "b"}


def test_register_handler_invoked_on_violation():
    enforcer = PolicyEnforcer()
    handled = []

    def handler(rule):
        handled.append(rule["name"])

    enforcer.register_handler("warn", handler)
    enforcer.add_rule(
        name="always",
        description="d",
        condition="1 == 1",
        action="warn",
    )
    violations = enforcer.check_state({"messages": [], "context": {}})
    assert violations == ["always"]
    assert handled == ["always"]


def test_check_message_returns_violated_rule_names_and_records():
    enforcer = PolicyEnforcer()
    enforcer.add_rule(
        name="password_leak",
        description="d",
        condition="'password' in message['content']",
        action="block",
        severity="high",
    )
    message = {"role": "user", "content": "my password is x"}
    violations = enforcer.check_message(message)
    assert violations == ["password_leak"]

    recorded = enforcer.get_violations()
    assert len(recorded) == 1
    assert recorded[0]["rule_name"] == "password_leak"
    assert recorded[0]["violation_type"] == "message"
    assert recorded[0]["severity"] == "high"


def test_check_state_returns_violated_rule_names():
    enforcer = PolicyEnforcer()
    enforcer.add_rule(
        name="low_trust",
        description="d",
        condition="state['trust_score'] < 0.5",
        action="warn",
        severity="medium",
    )
    state = {"messages": [], "context": {}, "trust_score": 0.3}
    violations = enforcer.check_state(state)
    assert violations == ["low_trust"]
    recorded = enforcer.get_violations()
    assert recorded[0]["violation_type"] == "state"


def test_check_state_no_violation():
    enforcer = PolicyEnforcer()
    enforcer.add_rule(
        name="low_trust",
        description="d",
        condition="state['trust_score'] < 0.5",
    )
    state = {"messages": [], "context": {}, "trust_score": 0.9}
    assert enforcer.check_state(state) == []
    assert enforcer.get_violations() == []


def test_get_violations_severity_filter_and_limit():
    enforcer = PolicyEnforcer()
    enforcer.add_rule(name="high1", description="d", condition="True", severity="high")
    enforcer.add_rule(name="low1", description="d", condition="True", severity="low")
    enforcer.add_rule(name="high2", description="d", condition="True", severity="high")

    enforcer.check_state({"messages": [], "context": {}})

    high = enforcer.get_violations(severity="high")
    assert {v["rule_name"] for v in high} == {"high1", "high2"}

    low = enforcer.get_violations(severity="low")
    assert {v["rule_name"] for v in low} == {"low1"}

    # severity + limit fast path
    limited = enforcer.get_violations(severity="high", limit=1)
    assert len(limited) == 1
    assert limited[0]["severity"] == "high"

    # limit only
    last_two = enforcer.get_violations(limit=2)
    assert len(last_two) == 2


def test_evaluate_condition_binop_arithmetic():
    enforcer = PolicyEnforcer()
    assert enforcer._evaluate_condition("1 + 1 == 2", {}) is True
    assert enforcer._evaluate_condition("3 * 2 == 6", {}) is True
    assert enforcer._evaluate_condition("10 - 4 == 5", {}) is False
    assert enforcer._evaluate_condition("8 / 2 == 4", {}) is True


def test_evaluate_condition_subscript_missing_key_returns_none():
    enforcer = PolicyEnforcer()
    context = {"state": {"a": 1}}
    # Missing key -> subscript returns None -> None == None is True
    assert enforcer._evaluate_condition("state['missing'] == None", context) is True
    assert enforcer._evaluate_condition("state['a'] == 1", context) is True


def test_evaluate_condition_unsupported_node_returns_false():
    enforcer = PolicyEnforcer()
    # Lambda is an unsupported AST node -> exception caught -> False
    assert enforcer._evaluate_condition("(lambda: 1)() == 1", {}) is False


def test_evaluate_condition_unsupported_operator_returns_false():
    enforcer = PolicyEnforcer()
    # Modulo is not in _BINOPS -> ValueError -> caught -> False
    assert enforcer._evaluate_condition("5 % 2 == 1", {}) is False


def test_evaluate_condition_depth_limit_returns_false():
    enforcer = PolicyEnforcer()
    # Deeply nested arithmetic to exceed depth limit (>50)
    deep = "+".join(["1"] * 60) + " == 60"
    assert enforcer._evaluate_condition(deep, {}) is False


def test_evaluate_condition_bool_ops():
    enforcer = PolicyEnforcer()
    context = {"message": {"role": "user", "content": "hello"}}
    cond = "message['role'] == 'user' and 'hello' in message['content']"
    assert enforcer._evaluate_condition(cond, context) is True
    cond_or = "message['role'] == 'admin' or 'hello' in message['content']"
    assert enforcer._evaluate_condition(cond_or, context) is True


def test_max_violations_bounding():
    enforcer = PolicyEnforcer(max_violations=2)
    enforcer.add_rule(name="r", description="d", condition="True", severity="low")
    for _ in range(5):
        enforcer.check_state({"messages": [], "context": {}})
    assert len(enforcer.get_violations()) == 2


# --- A14: fail-open vs fail-closed on evaluation errors -----------------------


def test_evaluation_error_default_fails_open_but_logs(caplog):
    """Default behaviour: an evaluation error is logged and is NOT a violation."""
    enforcer = PolicyEnforcer()  # fail_closed defaults to False
    assert enforcer.fail_closed is False
    # ``missing_var`` is not in the context -> unknown variable -> eval error.
    with caplog.at_level("WARNING"):
        result = enforcer._evaluate_condition("missing_var == 1", {}, rule_name="r")
    assert result is False
    assert any("missing_var" in rec.getMessage() for rec in caplog.records)
    assert any("r" in rec.getMessage() for rec in caplog.records)


def test_evaluation_error_fail_closed_is_violation(caplog):
    """fail_closed=True turns an evaluation error into a violation (still logged)."""
    enforcer = PolicyEnforcer(fail_closed=True)
    assert enforcer.fail_closed is True
    with caplog.at_level("WARNING"):
        result = enforcer._evaluate_condition("missing_var == 1", {}, rule_name="r")
    assert result is True
    assert any("missing_var" in rec.getMessage() for rec in caplog.records)


def test_fail_closed_block_rule_fires_on_malformed_input():
    """A block rule referencing a missing key fires under fail_closed."""
    enforcer = PolicyEnforcer(fail_closed=True)
    enforcer.add_rule(
        name="needs_field",
        description="d",
        condition="state['trust_score'] < 0.5",
        action="block",
        severity="high",
    )
    # No 'trust_score' key -> subscript returns None -> None < 0.5 raises
    # TypeError -> evaluation error -> violation because fail_closed.
    state = {"messages": [], "context": {}}
    violations = enforcer.check_state(state)
    assert violations == ["needs_field"]


def test_fail_open_block_rule_does_not_fire_on_malformed_input():
    """The same malformed input is silently allowed under the default fail-open."""
    enforcer = PolicyEnforcer()  # fail-open
    enforcer.add_rule(
        name="needs_field",
        description="d",
        condition="state['trust_score'] < 0.5",
        action="block",
        severity="high",
    )
    state = {"messages": [], "context": {}}
    assert enforcer.check_state(state) == []
    assert enforcer.get_violations() == []


# --- A14: node-count cap ------------------------------------------------------


def test_node_count_cap_rejects_oversized_condition():
    """A flat-but-large literal is rejected at add time even though it is shallow."""
    enforcer = PolicyEnforcer()
    # Flat 250-element list literal: shallow (depth 1) but many nodes.
    big_list = "[" + ", ".join(["1"] * 250) + "] == []"
    with pytest.raises(ValueError, match="node count"):
        enforcer.add_rule(name="big", description="d", condition=big_list)
    # The rule must not have been registered.
    assert enforcer.get_rule("big") is None


def test_node_count_cap_allows_normal_condition():
    """A normal, small condition is accepted."""
    enforcer = PolicyEnforcer()
    enforcer.add_rule(
        name="ok",
        description="d",
        condition="state['trust_score'] < 0.5",
    )
    assert enforcer.get_rule("ok") is not None
