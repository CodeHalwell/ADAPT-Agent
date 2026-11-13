"""Tests for the PolicyEnforcer."""

import ast

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


# -- registration-time validation: a rule that can never fire is rejected ------


@pytest.mark.parametrize(
    "condition",
    [
        "stat['trust_score'] < 0.5",  # typo'd name
        "len(state['messages']) > 5",  # function call
        "state.trust_score < 0.5",  # attribute access
        "not state['ok']",  # unary not
        "state['messages'][-1] == 'x'",  # unary minus
        "state['messages'][0:2] == []",  # slice
        "state['n'] % 2 == 0",  # unsupported arithmetic operator
        "state['n'] ** 2 > 4",  # power: unbounded CPU and memory
        "state['callback']() == 1",  # a call, even with only known names
        "state['obj'].secret == 1",  # attribute access on a looked-up value
        "state['n'] is None",  # unsupported comparison
        "state['n'] >",  # syntax error
    ],
)
def test_add_rule_rejects_conditions_that_could_never_evaluate(condition):
    enforcer = PolicyEnforcer()
    with pytest.raises(ValueError):
        enforcer.add_rule(name="r", description="d", condition=condition, action="block")
    assert enforcer.get_rule("r") is None


@pytest.mark.parametrize(
    "condition",
    [
        "state['trust_score'] < 0.5",
        "'password' in message['content'] or 'credentials' in message['content']",
        "0 < state['n'] <= 10 and state['context'] != {}",
        "state['a'] + state['b'] * 2 - 1 > state['c'] / 2",
        "message['role'] in ['system', 'tool'] and message['role'] not in ('user',)",
        "state['context'] == {'k': [1, 2], 's': {3}}",
        "True",
    ],
)
def test_add_rule_accepts_every_supported_construct(condition):
    PolicyEnforcer().add_rule(name="r", description="d", condition=condition)


def test_every_condition_accepted_at_registration_is_evaluable():
    """validate_condition and _eval_node must agree on the supported grammar.

    If the validator accepted a node the evaluator cannot handle, the rule
    would register and then silently never fire -- the exact bug the validator
    exists to prevent.
    """
    enforcer = PolicyEnforcer()
    state = {"n": 3, "a": 1, "b": 2, "c": 4, "context": {"k": 1}, "messages": [{"content": "x"}]}
    conditions = {
        "cmp": "0 < state['n'] <= 10",
        "bool": "state['n'] == 3 and state['context'] != {} or state['a'] == 9",
        "arith": "state['a'] + state['b'] * 2 - 1 > state['c'] / 2",
        "member": "state['n'] in [1, 2, 3] and state['n'] not in (7,) and state['n'] in {3}",
        "nested": "state['messages'][0]['content'] == 'x'",
        "dict": "state['context'] == {'k': 1}",
    }
    for name, condition in conditions.items():
        enforcer.add_rule(name=name, description="d", condition=condition)
    assert enforcer.evaluate_rules(state=state) == dict.fromkeys(conditions, "violation")


def test_evaluate_rules_reports_violation_pass_and_error_without_side_effects():
    enforcer = PolicyEnforcer()
    handled = []
    enforcer.register_handler("block", handled.append)
    enforcer.add_rule(
        name="low_trust", description="d", condition="state['trust_score'] < 0.5", action="block"
    )
    enforcer.add_rule(name="msg", description="d", condition="'x' in message['content']")

    assert enforcer.evaluate_rules(state={"trust_score": 0.2}) == {
        "low_trust": "violation",
        "msg": "error",  # `message` is not in scope for a state-only example
    }
    assert enforcer.evaluate_rules(state={"trust_score": 0.9})["low_trust"] == "pass"
    # A missing key is the run-time vacuity registration cannot see.
    assert enforcer.evaluate_rules(state={})["low_trust"] == "error"
    assert enforcer.evaluate_rules(message={"role": "user", "content": "x"})["msg"] == "violation"

    assert enforcer.get_violations() == []
    assert handled == []


@pytest.mark.parametrize(
    "condition", ["2 ** 3 == 8", "state['f'](1) == 1", "state['o'].real == 1", "9 ** 9 ** 9 > 0"]
)
def test_evaluator_itself_refuses_calls_attributes_and_power(condition):
    """Defence in depth behind add_rule: the evaluator never runs these.

    Registration rejects them, but the evaluator is the actual sandbox -- a
    rule injected past add_rule must still not be able to call, reach an
    attribute, or burn CPU on an exponent tower.
    """
    enforcer = PolicyEnforcer(fail_closed=True)
    state = {"f": lambda x: x, "o": 1}
    # fail_closed turns "could not evaluate" into True, and a successful
    # evaluation of any of these would also be truthy -- so assert on the node
    # walk directly rather than on the boolean.
    tree = ast.parse(condition, mode="eval")
    with pytest.raises(ValueError):
        enforcer._eval_node(tree.body, {"state": state})


# -- run-time vacuity: a rule that has only ever errored ------------------------


def test_rule_stats_count_each_outcome():
    enforcer = PolicyEnforcer()
    enforcer.add_rule("low_trust", "d", "state['trust_score'] < 0.5", action="block")
    enforcer.check_state({"trust_score": 0.2})
    enforcer.check_state({"trust_score": 0.9})
    enforcer.check_state({"trust_score": 0.8})
    enforcer.check_state({})  # the key is missing: cannot be evaluated
    assert enforcer.get_rule_stats() == {
        "low_trust": {"violation": 1, "pass": 2, "error": 1, "unresolved": 1}
    }


def test_vacuous_rules_finds_a_rule_that_has_never_been_evaluable():
    """The documented trap: a `message` rule under something that only checks state.

    Under the default fail-open setting it has silently done nothing since it
    was installed. Nothing fails; the rule simply never fires.
    """
    enforcer = PolicyEnforcer()
    enforcer.add_rule("no_secrets", "d", "'password' in message['content']", action="block")
    enforcer.add_rule("needs_tier", "d", "state['tier'] == 'free'", action="block")
    enforcer.add_rule("low_trust", "d", "state['trust_score'] < 0.5", action="block")
    enforcer.add_rule("never_checked", "d", "state['x'] == 1")

    assert enforcer.vacuous_rules() == [], "no evidence yet: nothing has been evaluated"
    enforcer.remove_rule("never_checked")

    for trust in (0.9, 0.2, 0.7):
        assert "no_secrets" not in enforcer.check_state({"trust_score": trust})

    assert enforcer.vacuous_rules() == ["no_secrets", "needs_tier"]


def test_a_missing_key_is_a_quiet_pass_but_is_counted_as_unresolved():
    """`==` against an absent key does not error -- which is what hides it."""
    enforcer = PolicyEnforcer()
    enforcer.add_rule("free_tier", "d", "state['tier'] == 'free'", action="block")
    assert enforcer.check_state({"trust_score": 0.9}) == []
    assert enforcer.get_rule_stats()["free_tier"] == {
        "violation": 0,
        "pass": 1,
        "error": 0,
        "unresolved": 1,
    }
    assert enforcer.vacuous_rules() == ["free_tier"]

    enforcer.check_state({"tier": "paid"})  # now it has seen real data once
    assert enforcer.vacuous_rules() == []


def test_a_guarded_lookup_that_short_circuits_is_not_a_missed_lookup():
    enforcer = PolicyEnforcer()
    enforcer.add_rule(
        "admin", "d", "'admin' in state['context'] and state['context']['admin'] == True"
    )
    enforcer.check_state({"context": {}})
    assert enforcer.get_rule_stats()["admin"]["unresolved"] == 0


def test_a_rule_that_is_sometimes_out_of_scope_is_not_vacuous():
    """Mixing check_message and check_state on one enforcer is legitimate."""
    enforcer = PolicyEnforcer()
    enforcer.add_rule("msg", "d", "'x' in message['content']")
    enforcer.add_rule("st", "d", "state['n'] > 1")
    enforcer.check_state({"n": 0})
    enforcer.check_message({"role": "user", "content": "y"})
    assert enforcer.vacuous_rules() == []


def test_rule_stats_follow_the_rule_not_the_name():
    enforcer = PolicyEnforcer()
    enforcer.add_rule("r", "d", "state['missing'] == 1")
    enforcer.check_state({})
    assert enforcer.vacuous_rules() == ["r"]

    enforcer.add_rule("r", "d", "state['n'] == 1")  # replaced: old counts are about the old rule
    assert enforcer.get_rule_stats() == {
        "r": {"violation": 0, "pass": 0, "error": 0, "unresolved": 0}
    }
    enforcer.remove_rule("r")
    assert enforcer.get_rule_stats() == {}


def test_dry_run_is_not_counted_in_rule_stats():
    enforcer = PolicyEnforcer()
    enforcer.add_rule("r", "d", "state['n'] == 1")
    enforcer.evaluate_rules(state={"n": 1})
    assert enforcer.get_rule_stats() == {
        "r": {"violation": 0, "pass": 0, "error": 0, "unresolved": 0}
    }


@pytest.mark.parametrize(
    "condition, expected",
    [
        ("(0 or 2) == 2", "violation"),  # `or` yields the operand, not True
        ("(1 and state['s']) == 'ab'", "violation"),  # so does `and`
        ("(0 and 1) == 0", "violation"),
        ("('' or state['none']) == None", "violation"),
        ("0 < (0 or state['none'])", "error"),  # Python refuses 0 < None; so must we
        ("state['z'] or state['n']", "violation"),
        ("state['z'] and state['n']", "pass"),
    ],
)
def test_and_or_have_python_semantics(condition, expected):
    enforcer = PolicyEnforcer()
    enforcer.add_rule("r", "d", condition)
    state = {"s": "ab", "z": 0, "n": 2, "none": None}
    assert enforcer.evaluate_rules(state=state)["r"] == expected
