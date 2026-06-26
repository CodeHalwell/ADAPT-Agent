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
