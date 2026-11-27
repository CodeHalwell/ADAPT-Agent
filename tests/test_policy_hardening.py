"""Hardening of the policy engine, trust manager, taint tracker and friends.

Each test pins a way a control used to be accepted and then quietly do nothing
(or to be bypassed), so a regression is a failure rather than a silent pass.
"""

from __future__ import annotations

import logging
import math
import threading
import time

import pytest

from adapt_agent import PatchManager, PolicyEnforcer, TrustManager
from adapt_agent.core.memory import MemorySystem
from adapt_agent.core.policy import (
    VALID_ACTIONS,
    VALID_SEVERITIES,
    validate_condition,
)
from adapt_agent.exceptions import SecurityBlockedError
from adapt_agent.security.taint_tracker import TaintLevel, TaintTracker

# ---------------------------------------------------------------- action / severity


def test_valid_sets_are_the_documented_ones():
    assert VALID_ACTIONS == {"warn", "block", "modify", "require_approval"}
    assert VALID_SEVERITIES == {"low", "medium", "high", "critical"}


@pytest.mark.parametrize(
    "action", ["Block", "BLOCK", "deny", "blok", "reject", "", "log", " block"]
)
def test_add_rule_rejects_an_unknown_action(action):
    enforcer = PolicyEnforcer()
    with pytest.raises(ValueError, match="block"):
        enforcer.add_rule("r", "d", "state['x'] == 1", action=action)
    assert enforcer.list_rules() == []


@pytest.mark.parametrize("severity", ["High", "urgent", "", "severe"])
def test_add_rule_rejects_an_unknown_severity(severity):
    with pytest.raises(ValueError, match="critical"):
        PolicyEnforcer().add_rule("r", "d", "state['x'] == 1", severity=severity)


@pytest.mark.parametrize("action", sorted(VALID_ACTIONS - {"require_approval"}))
@pytest.mark.parametrize("severity", sorted(VALID_SEVERITIES))
def test_add_rule_accepts_every_valid_pair(action, severity):
    PolicyEnforcer().add_rule("r", "d", "state['x'] == 1", action=action, severity=severity)


def test_require_approval_is_refused_on_a_rule_that_is_not_tool_scoped():
    enforcer = PolicyEnforcer()
    with pytest.raises(ValueError, match="tool"):
        enforcer.add_rule("r", "d", "state['x'] == 1", action="require_approval")
    enforcer.add_rule("ok", "d", "tool['name'] == 'refund'", action="require_approval")


def test_register_handler_rejects_an_action_that_could_never_fire():
    with pytest.raises(ValueError, match="Invalid action"):
        PolicyEnforcer().register_handler("Block", lambda rule: None)


# --------------------------------------------------------------- sandbox robustness


@pytest.mark.parametrize(
    "condition",
    [
        "'a' * 400000000 == state['x']",
        "[0] * 50000000 == state['x']",
        "state['msg'] * state['n'] == ''",
    ],
)
def test_sequence_repetition_is_not_evaluated(condition, caplog):
    enforcer = PolicyEnforcer()
    enforcer.add_rule("r", "d", condition, action="block")
    started = time.perf_counter()
    with caplog.at_level(logging.WARNING):
        outcome = enforcer.evaluate_rules(state={"x": 1, "msg": "A" * 1000, "n": 500000})
    assert outcome == {"r": "error"}  # an evaluation error, so fail-closed -- not an allocation
    assert time.perf_counter() - started < 0.5


def test_huge_repetition_does_not_reach_the_allocator():
    enforcer = PolicyEnforcer()
    enforcer.add_rule("oom", "d", "'a' * 99999999999 == state['x']", action="block")
    assert enforcer.evaluate_rules(state={"x": 1}) == {"oom": "error"}


def test_numeric_multiplication_and_power_rules_are_unchanged():
    enforcer = PolicyEnforcer()
    enforcer.add_rule("m", "d", "state['a'] * 2 + 1 == 7", action="block")
    assert enforcer.check_state({"a": 3}) == ["m"]
    with pytest.raises(ValueError):
        enforcer.add_rule("p", "d", "state['a'] ** 99999999 == 1")


def test_a_condition_the_evaluator_could_never_evaluate_for_depth_is_rejected():
    deep = "state" + "['a']" * 51 + " == 1"
    with pytest.raises(ValueError, match="deeper"):
        validate_condition(deep)
    with pytest.raises(ValueError):
        PolicyEnforcer().add_rule("deep", "d", deep, action="block")


def test_the_deepest_evaluable_condition_is_still_accepted():
    ok = "state" + "['a']" * 49 + " == 1"
    enforcer = PolicyEnforcer()
    enforcer.add_rule("ok", "d", ok)
    nested: dict = {"a": 1}
    for _ in range(48):
        nested = {"a": nested}
    assert enforcer.evaluate_rules(state=nested) == {"ok": "violation"}  # evaluated, not errored


def test_a_hostile_nesting_depth_is_a_value_error_not_a_crash():
    with pytest.raises(ValueError):
        validate_condition("[" * 900 + "]" * 900)


def test_a_raising_handler_neither_aborts_the_loop_nor_hides_the_violation(caplog):
    enforcer = PolicyEnforcer()
    enforcer.add_rule("a_warn", "d", "state['x'] == 1", action="warn")
    enforcer.add_rule("b_block", "d", "state['x'] == 1", action="block")
    enforcer.register_handler("warn", lambda rule: 1 / 0)
    with caplog.at_level(logging.ERROR):
        result = enforcer.check_state({"x": 1})
    assert result == ["a_warn", "b_block"]
    assert [v["rule_name"] for v in enforcer.get_violations()] == ["a_warn", "b_block"]
    assert "a_warn" in caplog.text


def test_a_handler_may_still_refuse_deliberately():
    enforcer = PolicyEnforcer()
    enforcer.add_rule("r", "d", "state['x'] == 1", action="block")

    def refuse(rule):
        raise SecurityBlockedError("no")

    enforcer.register_handler("block", refuse)
    with pytest.raises(SecurityBlockedError):
        enforcer.check_state({"x": 1})


def test_rules_can_be_added_and_removed_while_checking():
    enforcer = PolicyEnforcer()
    for i in range(200):
        enforcer.add_rule(f"r{i}", "d", "state['x'] == 2", action="block")
    errors: list[BaseException] = []
    stop = threading.Event()

    def churn():
        i = 0
        while not stop.is_set():
            enforcer.add_rule(f"tmp{i % 50}", "d", "state['x'] == 3")
            enforcer.remove_rule(f"tmp{i % 50}")
            i += 1

    def check():
        for _ in range(500):
            try:
                enforcer.check_state({"x": 1})
                enforcer.get_violations()
                enforcer.vacuous_rules()
            except Exception as exc:  # noqa: BLE001 - the point is that none escapes
                errors.append(exc)

    churner = threading.Thread(target=churn)
    checkers = [threading.Thread(target=check) for _ in range(4)]
    churner.start()
    for t in checkers:
        t.start()
    for t in checkers:
        t.join()
    stop.set()
    churner.join()
    assert errors == []


def test_rule_counters_do_not_lose_updates_under_threads():
    enforcer = PolicyEnforcer()
    enforcer.add_rule("r", "d", "state['x'] == 1")

    def hammer():
        for _ in range(5000):
            enforcer.check_state({"x": 1})

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert enforcer.get_rule_stats()["r"]["violation"] == 40000


def test_violation_history_is_a_bounded_snapshot_not_the_live_state():
    enforcer = PolicyEnforcer()
    enforcer.add_rule("r", "d", "state['x'] == 1", severity="high")
    state = {"x": 1, "messages": [{"role": "user", "content": "my SSN is 078-05-1120"}]}
    enforcer.check_state(state)  # type: ignore[arg-type]
    state["messages"].append({"role": "assistant", "content": "a later turn"})
    state["huge"] = "Z" * 10_000_000

    (record,) = enforcer.get_violations()
    assert set(record) == {"rule_name", "violation_type", "severity", "timestamp", "data"}
    assert record["data"] is not state
    rendered = repr(record["data"])
    assert "078-05-1120" not in rendered and "later turn" not in rendered
    assert len(rendered) < 1000
    assert record["data"]["keys"] == ["x", "messages"]


def test_violation_summary_caps_keys_and_strings():
    enforcer = PolicyEnforcer()
    enforcer.add_rule("r", "d", "state['x'] == 1")
    enforcer.check_state({"x": 1, **{f"k{i}" * 50: i for i in range(500)}})  # type: ignore[arg-type]
    data = enforcer.get_violations()[0]["data"]
    assert len(data["keys"]) == 20
    assert all(len(k) <= 64 for k in data["keys"])
    assert data["size"] == 501


def test_tool_violation_keeps_the_tool_name_but_not_its_arguments():
    enforcer = PolicyEnforcer()
    enforcer.add_rule("r", "d", "tool['name'] == 'send'", action="block")
    enforcer.check_tool({"name": "send", "arguments": {"body": "secret-token"}})
    data = enforcer.get_violations()[0]["data"]
    assert data["name"] == "send"
    assert "secret-token" not in repr(data)


@pytest.mark.parametrize("op", [">", "<", ">=", "==", "!="])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_comparison_with_a_non_finite_number_is_an_evaluation_error(op, bad):
    enforcer = PolicyEnforcer()
    enforcer.add_rule("r", "d", f"state['x'] {op} 100", action="block")
    assert enforcer.evaluate_rules(state={"x": bad}) == {"r": "error"}
    assert enforcer.check_state({"x": bad}) == ["r"]  # fail-closed
    assert PolicyEnforcer(fail_closed=False).check_state({"x": bad}) == []


def test_nan_that_arrives_through_arithmetic_is_caught_too():
    enforcer = PolicyEnforcer()
    enforcer.add_rule("r", "d", "state['x'] * 1e308 * 10 > 1", action="block")
    assert enforcer.evaluate_rules(state={"x": 5.0}) == {"r": "error"}


def test_finite_floats_still_compare_normally():
    enforcer = PolicyEnforcer()
    enforcer.add_rule("r", "d", "state['x'] > 0.5", action="block")
    assert enforcer.check_state({"x": 0.9}) == ["r"]
    assert enforcer.check_state({"x": 0.1}) == []


def test_negative_literals_are_supported():
    enforcer = PolicyEnforcer()
    enforcer.add_rule("lt", "d", "state['n'] < -1", action="block")
    enforcer.add_rule("idx", "d", "state['messages'][-1] == 'x'", action="block")
    enforcer.add_rule("plus", "d", "state['n'] == +2")
    enforcer.add_rule("f", "d", "state['t'] > -0.5")
    assert enforcer.check_state({"n": -3, "messages": ["a", "x"], "t": 0.0}) == ["lt", "idx", "f"]
    assert enforcer.check_state({"n": 2, "messages": ["x", "a"], "t": -1.0}) == ["plus"]


@pytest.mark.parametrize(
    "condition",
    ["-state['n'] > 1", "not state['n']", "~1 == state['n']", "- -1 == 1"],
)
def test_unary_operators_on_anything_but_a_numeric_literal_are_rejected(condition):
    with pytest.raises(ValueError):
        validate_condition(condition)


# ------------------------------------------------------------------------- trust


@pytest.mark.parametrize("delta", [math.nan, math.inf, -math.inf])
def test_a_non_finite_trust_delta_is_rejected(delta):
    trust = TrustManager()
    trust.update_trust_score("a", -1.0)
    with pytest.raises(ValueError, match="finite"):
        trust.update_trust_score("a", delta)
    assert trust.get_trust_score("a") == 0.0
    assert trust.is_trusted("a") is False


@pytest.mark.parametrize("delta", ["0.1", None, True])
def test_a_non_numeric_trust_delta_is_rejected(delta):
    with pytest.raises(ValueError):
        TrustManager().update_trust_score("a", delta)  # type: ignore[arg-type]


def test_churning_throwaway_ids_cannot_launder_a_distrusted_agent():
    trust = TrustManager(max_agents=100)  # default distrust_floor 0.25 < initial_trust 0.5
    trust.update_trust_score("villain", -0.2)
    assert trust.get_trust_score("villain") == pytest.approx(0.3)
    for i in range(500):
        trust.update_trust_score(f"throwaway-{i}", 0.1)
    assert len(trust._trust_scores) <= 100
    assert trust.get_trust_score("villain") == pytest.approx(0.3)
    assert trust.is_trusted("villain") is False


def test_the_cap_still_holds_when_every_slot_is_distrusted(caplog):
    trust = TrustManager(max_agents=5)
    with caplog.at_level(logging.WARNING):
        for i in range(20):
            trust.update_trust_score(f"bad-{i}", -0.1 - i * 0.01)
    assert len(trust._trust_scores) == 5
    assert "distrusted" in caplog.text


def test_the_agent_being_updated_is_not_its_own_eviction_victim():
    trust = TrustManager(max_agents=2)
    trust.update_trust_score("a", -0.3)
    trust.update_trust_score("b", -0.2)
    trust.update_trust_score("c", -0.4)  # everyone is distrusted; c must survive its own update
    assert trust.get_trust_score("c") == pytest.approx(0.1)


# ------------------------------------------------------------------------- taint


def _fill(tracker: TaintTracker, n: int, source: str = "benign") -> None:
    for i in range(n):
        tracker.mark_tainted(f"filler-{i}", [source])


def test_evicted_critical_data_does_not_read_clean():
    tracker = TaintTracker()
    tracker.register_source("evil", "web", TaintLevel.CRITICAL)
    tracker.register_source("benign", "web", TaintLevel.LOW)
    tracker.mark_tainted("secret", ["evil"])
    _fill(tracker, 1500)
    assert tracker.is_tainted("secret")
    assert tracker.get_taint_level("secret") is TaintLevel.CRITICAL
    assert [s.level for s in tracker.get_taint_sources("secret")] == [TaintLevel.CRITICAL]


def test_evicted_data_keeps_its_level_after_its_source_is_evicted_too():
    tracker = TaintTracker(max_tracked_items=5)
    tracker.register_source("evil", "web", TaintLevel.HIGH)
    tracker.mark_tainted("secret", ["evil"])
    _fill(tracker, 6, source="other")
    for i in range(10):  # evil is now unreferenced and can be evicted invisibly
        tracker.register_source(f"pad-{i}", "web", TaintLevel.LOW)
    assert "evil" not in tracker._taint_sources
    assert tracker.get_taint_level("secret") is TaintLevel.HIGH


def test_taint_survives_overflow_of_the_eviction_memory_by_saturating():
    tracker = TaintTracker(max_tracked_items=5)
    tracker.register_source("evil", "web", TaintLevel.CRITICAL)
    tracker.register_source("benign", "web", TaintLevel.LOW)
    tracker.mark_tainted("secret", ["evil"])
    _fill(tracker, 50)
    assert tracker.get_stats()["saturated"] is True
    assert tracker.get_taint_level("secret") is TaintLevel.CRITICAL
    # An id it has never heard of reads as tainted rather than clean.
    assert tracker.is_tainted("never-seen")
    assert tracker.get_taint_level("never-seen") is TaintLevel.CRITICAL
    tracker.reset()
    assert tracker.get_stats()["saturated"] is False
    assert not tracker.is_tainted("never-seen")


def test_sanitize_clears_an_evicted_item():
    tracker = TaintTracker(max_tracked_items=5)
    tracker.register_source("evil", "web", TaintLevel.HIGH)
    tracker.mark_tainted("secret", ["evil"])
    _fill(tracker, 6, source="evil")
    assert tracker.get_taint_level("secret") is TaintLevel.HIGH
    tracker.sanitize("secret")
    assert not tracker.is_tainted("secret")


def test_propagation_from_an_evicted_item_carries_its_taint():
    tracker = TaintTracker(max_tracked_items=5)
    tracker.register_source("evil", "web", TaintLevel.CRITICAL)
    tracker.register_source("benign", "web", TaintLevel.LOW)
    tracker.mark_tainted("secret", ["evil"])
    _fill(tracker, 6)
    tracker.propagate_taint("secret", "derived", "concat")
    assert tracker.get_taint_level("derived") is TaintLevel.CRITICAL


def test_empty_marks_are_ignored_and_cannot_evict_real_taint():
    tracker = TaintTracker(max_tracked_items=5)
    tracker.register_source("evil", "web", TaintLevel.CRITICAL)
    tracker.mark_tainted("secret", ["evil"])
    for i in range(1000):
        tracker.mark_tainted(f"empty-{i}", [])
    assert tracker.get_stats()["tainted_data_count"] == 1
    assert tracker.get_stats()["evicted_data_count"] == 0
    assert not tracker.is_tainted("empty-0")
    assert tracker.get_taint_level("secret") is TaintLevel.CRITICAL


def test_a_source_cannot_be_re_registered_at_a_lower_level():
    tracker = TaintTracker()
    tracker.register_source("s", "web", TaintLevel.CRITICAL)
    tracker.mark_tainted("d", ["s"])
    with pytest.raises(ValueError, match="re-register"):
        tracker.register_source("s", "web", TaintLevel.LOW)
    assert tracker.get_taint_level("d") is TaintLevel.CRITICAL
    # Same level or higher is fine.
    tracker.register_source("s", "web", TaintLevel.CRITICAL, metadata={"v": 2})
    tracker.register_source("t", "web", TaintLevel.LOW)
    tracker.register_source("t", "web", TaintLevel.HIGH)
    assert tracker.get_taint_level("t") is TaintLevel.UNTAINTED  # nothing marked yet


def test_an_evicted_referenced_source_cannot_come_back_lower():
    tracker = TaintTracker(max_tracked_items=3, unknown_source_level=TaintLevel.CRITICAL)
    tracker.register_source("s0", "web", TaintLevel.HIGH)
    tracker.mark_tainted("d", ["s0"])
    for i in range(1, 6):
        tracker.register_source(f"s{i}", "web", TaintLevel.HIGH)
        tracker.mark_tainted(f"d{i}", [f"s{i}"])
    assert "s0" not in tracker._taint_sources
    with pytest.raises(ValueError):
        tracker.register_source("s0", "web", TaintLevel.LOW)
    tracker.register_source("s0", "web", TaintLevel.HIGH)
    assert tracker.get_taint_level("d") is not TaintLevel.UNTAINTED


def test_the_tracker_is_safe_to_use_from_several_threads():
    tracker = TaintTracker(max_tracked_items=50)
    errors: list[BaseException] = []

    def work(n: int) -> None:
        try:
            for i in range(300):
                tracker.register_source(f"s{n}-{i}", "web", TaintLevel.MEDIUM)
                tracker.mark_tainted(f"d{n}-{i}", [f"s{n}-{i}"])
                tracker.get_taint_level(f"d{n}-{i}")
                tracker.get_stats()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


# --------------------------------------------------------------- memory / patches


def test_memory_is_safe_to_search_while_another_thread_stores():
    memory = MemorySystem(short_term_capacity=50, long_term_capacity=50)
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            memory.store_short_term(f"k{i}", f"v{i}")
            i += 1

    def reader():
        for _ in range(500):
            try:
                memory.search("v")
                memory.retrieve("k1")
                memory.consolidate()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

    w = threading.Thread(target=writer)
    readers = [threading.Thread(target=reader) for _ in range(3)]
    w.start()
    for t in readers:
        t.start()
    for t in readers:
        t.join()
    stop.set()
    w.join()
    assert errors == []


def test_memory_rejects_a_negative_capacity():
    with pytest.raises(ValueError):
        MemorySystem(short_term_capacity=-1)


def test_a_patch_is_applied_once_even_when_applied_concurrently():
    manager = PatchManager()
    calls: list[int] = []

    def slow_patch(target):
        calls.append(1)
        time.sleep(0.01)

    manager.register_patch("p", "fw", "d", slow_patch)
    threads = [threading.Thread(target=manager.apply_patch, args=("p", object())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1


def test_re_registering_a_patch_id_forgets_that_it_was_applied():
    manager = PatchManager()
    ran: list[str] = []
    manager.register_patch("p", "fw", "old", lambda t: ran.append("old"))
    assert manager.apply_patch("p", object())
    manager.register_patch("p", "fw", "new", lambda t: ran.append("new"))
    assert manager.is_applied("p") is False
    assert manager.apply_patch("p", object())
    assert ran == ["old", "new"]
