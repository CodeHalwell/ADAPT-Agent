"""Tests that pin guarantees ``scripts/mutation_check.py`` found unpinned.

Each test here exists because a mutant in that script survived the rest of the
suite: the guarantee held, but nothing would have noticed it going away.
"""

from typing import Any

import pytest

from adapt_agent.core import GovernanceGate, PolicyEnforcer, ToolPolicy
from adapt_agent.exceptions import SecurityBlockedError
from adapt_agent.security import Firewall


def _ask_rule(enforcer: PolicyEnforcer) -> None:
    enforcer.add_rule("ask", "d", "tool['name'] == 'wire'", action="require_approval")


def test_an_approval_rule_evaluated_after_a_block_does_not_downgrade_it() -> None:
    """Order-independent: a block found *before* the ask rule still wins."""
    enforcer = PolicyEnforcer()
    enforcer.add_rule("deny", "d", "tool['arguments']['amount'] > 10", action="block")
    _ask_rule(enforcer)
    decision = GovernanceGate(policy_enforcer=enforcer).check_tool_call("wire", {"amount": 11})
    assert decision.blocked
    assert decision.threats == ("policy:deny", "policy:ask")


def test_an_approval_rule_cannot_reopen_a_denied_tool() -> None:
    enforcer = PolicyEnforcer()
    _ask_rule(enforcer)
    gate = GovernanceGate(policy_enforcer=enforcer, tool_policy=ToolPolicy(deny=["wire"]))
    decision = gate.check_tool_call("wire", {"amount": 1})
    assert decision.blocked
    assert decision.threats == ("tool:denied:wire", "policy:ask")


def test_an_approval_rule_cannot_reopen_flagged_arguments() -> None:
    firewall = Firewall()
    firewall.add_blocked_pattern(r"rm -rf")
    enforcer = PolicyEnforcer()
    _ask_rule(enforcer)
    gate = GovernanceGate(firewall=firewall, policy_enforcer=enforcer)
    decision = gate.check_tool_call("wire", {"memo": "rm -rf /"})
    assert decision.blocked and "policy:ask" in decision.threats


class _OutOfTime:
    """A time-limited engine whose every evaluation runs out of time.

    Stands in for the optional ``regex`` module, which the dev-only environment
    does not install -- so the timeout path is pinned without it.
    """

    def search(self, text: str, timeout: float = 0.0) -> None:
        raise TimeoutError("regex timed out")

    fullmatch = search

    def sub(self, replacement: str, text: str, timeout: float = 0.0) -> str:
        raise TimeoutError("regex timed out")


def test_a_blocked_pattern_that_runs_out_of_time_blocks_without_the_regex_module() -> None:
    import re

    from adapt_agent.security import firewall as firewall_module

    fw = Firewall()
    fw._blocked.append(firewall_module._CompiledPattern(re.compile(r"(a|aa)+$"), _OutOfTime()))
    fw._blocked_patterns.append(re.compile(r"(a|aa)+$"))
    assert fw.dry_run("aaaa!") == {
        "allowed": False,
        "reason": "pattern_timeout",
        "pattern": "(a|aa)+$",
    }
    assert fw.check_input("aaaa!") is False


# -- LangGraph model-call middleware, without LangChain -------------------------------


class _ModelRequest:
    """The duck-typed shape of a LangChain v1 ``ModelRequest``."""

    def __init__(self, messages: list[dict[str, str]]) -> None:
        self.messages = messages
        self.state = {"messages": messages}
        self.runtime = None
        self.model = None


def _model_governor() -> Any:
    from adapt_agent.integrations import langgraph as lg

    firewall = Firewall()
    firewall.add_blocked_pattern(r"SECRET-\d+")
    gate = GovernanceGate(firewall=firewall)
    return lg._ModelGovernor(gate, screen_output=True, screen_stream=False, observer=None)


def test_langgraph_model_middleware_screens_the_request_before_the_model_runs() -> None:
    called: list[Any] = []
    request = _ModelRequest([{"role": "user", "content": "leak SECRET-42"}])
    with pytest.raises(SecurityBlockedError):
        _model_governor().wrap(request, lambda r: called.append(r) or "fine")
    assert called == []


def test_langgraph_model_middleware_screens_the_response() -> None:
    request = _ModelRequest([{"role": "user", "content": "hello"}])
    governor = _model_governor()
    assert governor.wrap(request, lambda r: "fine") == "fine"
    with pytest.raises(SecurityBlockedError):
        governor.wrap(request, lambda r: "here it is: SECRET-7")


# -- Google ADK stream buffering, without ADK ----------------------------------------


def test_adk_stream_is_screened_as_its_aggregate_not_chunk_by_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A match split across two partial chunks is caught on the second one."""
    from types import SimpleNamespace

    from adapt_agent.integrations import google_adk as adk

    # The held-back placeholder is an ADK response object; its text is all
    # that matters here, and building it would need ADK installed.
    monkeypatch.setattr(adk, "_partial_response", lambda text: text)
    firewall = Firewall()
    firewall.add_blocked_pattern(r"SECRET-TOKEN")
    governor = adk._Governor(
        GovernanceGate(firewall=firewall),
        observer=None,
        on_block="raise",
        on_tool_violation="refuse",
        refusal_text="refused",
        approvals=False,
        buffer_stream=True,
        stream_holdback=None,
        label_per_agent=False,
        tools_governed=False,
    )
    context = SimpleNamespace(invocation_id="inv-1", agent_name="a", branch="")

    def chunk(text: str) -> Any:
        content = SimpleNamespace(parts=[SimpleNamespace(text=text)])
        return SimpleNamespace(partial=True, content=content)

    assert governor.after_model(context, chunk("the SECRET-")) == ""  # held, not released
    with pytest.raises(SecurityBlockedError):
        governor.after_model(context, chunk("TOKEN leaks"))


# -- Microsoft Agent Framework stream buffering, without MAF -------------------------


def test_a_buffered_maf_stream_asks_maf_to_hold_every_update() -> None:
    """``stream_buffer_updates`` is what stops MAF delivering updates before the
    result gate has judged the whole response; without it a blocked answer is
    already on the consumer's screen when the gate fires."""
    from types import SimpleNamespace

    from adapt_agent.integrations import agent_framework as maf

    context = SimpleNamespace(
        stream_result_gates_after=[],
        stream_update_gates_after=[],
        stream_consumption_context_manager_factories=[],
        stream_buffer_updates=False,
    )
    span = SimpleNamespace(end=lambda *a, **k: None)
    reviewed: list[Any] = []
    maf._screen_stream_output(context, reviewed.append, buffer_streams=True, span=span)
    assert context.stream_buffer_updates is True
    assert context.stream_update_gates_after == []
    (gate,) = context.stream_result_gates_after
    gate("final answer")
    assert reviewed == ["final answer"]
