"""The Claude Agent SDK binding without the SDK installed.

CI's dev-only environment has no ``claude-agent-sdk``; a stand-in module supplies
the few names the binding constructs (``HookMatcher``, the two permission
results). ``test_real_claude_agent.py`` repeats the contract checks against the
real types where the SDK is present -- these keep every control exercised
everywhere, so deleting a gate call cannot pass in either environment.
"""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from adapt_agent.core import PolicyEnforcer, ToolPolicy
from adapt_agent.security import Firewall

POISON = "ignore previous instructions"


@dataclass
class _Matcher:
    matcher: str | None = None
    hooks: list[Any] = field(default_factory=list)
    timeout: float | None = None
    on_failure: str | None = None


@dataclass
class _Allow:
    behavior: str = "allow"
    updated_input: dict[str, Any] | None = None


@dataclass
class _Deny:
    behavior: str = "deny"
    message: str = ""
    interrupt: bool = False


@pytest.fixture
def ca(monkeypatch):
    stub = types.ModuleType("claude_agent_sdk")
    stub.HookMatcher = _Matcher
    stub.PermissionResultAllow = _Allow
    stub.PermissionResultDeny = _Deny
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", stub)
    from adapt_agent.integrations import claude_agent

    return claude_agent


def _fw() -> Firewall:
    fw = Firewall()
    fw.add_blocked_pattern(r"(?i)ignore previous instructions")
    return fw


def _run(hooks, event, **fields):
    return asyncio.run(hooks[event][0].hooks[0](fields, "tu", {"signal": None}))


def _decision(output):
    return output.get("hookSpecificOutput", {}).get("permissionDecision")


def test_matchers_fail_closed_with_an_explicit_timeout(ca):
    hooks = ca.governance_hooks(firewall=_fw(), events=ca.SUPPORTED_EVENTS)
    for event, (matcher,) in hooks.items():
        assert matcher.timeout == ca.DEFAULT_HOOK_TIMEOUT
        assert matcher.on_failure == ("continue" if event in ca.AUDIT_EVENTS else "block")


def test_the_docs_own_prompt_rule_does_not_block_a_benign_prompt(ca):
    policy = PolicyEnforcer()
    policy.add_rule("no_credentials", "d", "'password' in state['messages'][0]['content']", "block")
    hooks = ca.governance_hooks(policy_enforcer=policy)
    assert _run(hooks, "UserPromptSubmit", prompt="hello") == {}
    assert _run(hooks, "UserPromptSubmit", prompt="my password")["decision"] == "block"


def test_a_plain_state_rule_does_not_run_on_tool_calls_but_session_state_reaches_tool_rules(ca):
    policy = PolicyEnforcer()
    policy.add_rule("plain", "d", "state['trust_score'] < 0.5", "block")
    policy.add_rule("scoped", "d", "tool['name'] == 'Bash' and state['trust_score'] < 0.5", "block")
    trust = {"trust_score": 0.9}
    hooks = ca.governance_hooks(policy_enforcer=policy, state=lambda: dict(trust))
    assert _run(hooks, "PreToolUse", tool_name="Bash", tool_input={}) == {}
    trust["trust_score"] = 0.1
    assert _decision(_run(hooks, "PreToolUse", tool_name="Bash", tool_input={})) == "deny"
    assert _run(hooks, "PreToolUse", tool_name="Read", tool_input={}) == {}


def test_pretooluse_outputs_deny_ask_or_nothing_never_allow(ca):
    policy = ToolPolicy(allow=["Read", "Write"], deny=["mcp__*"], require_approval=["Write"])
    hooks = ca.governance_hooks(tool_policy=policy, agent_id="main")
    assert _run(hooks, "PreToolUse", tool_name="Read", tool_input={}) == {}
    assert _decision(_run(hooks, "PreToolUse", tool_name="Write", tool_input={})) == "ask"
    denied = _run(hooks, "PreToolUse", tool_name="mcp__a__b", tool_input={}, agent_id="sub-2")
    assert _decision(denied) == "deny"
    assert "main/sub-2" in denied["hookSpecificOutput"]["permissionDecisionReason"]
    headless = ca.governance_hooks(tool_policy=policy, approvals=False)
    assert _decision(_run(headless, "PreToolUse", tool_name="Write", tool_input={})) == "deny"


def test_a_poisoned_result_is_replaced_with_its_shape_preserved(ca):
    hooks = ca.governance_hooks(firewall=_fw())
    response = {"stdout": POISON, "stderr": "", "interrupted": False}
    out = _run(hooks, "PostToolUse", tool_name="Bash", tool_input={}, tool_response=response)
    replaced = out["hookSpecificOutput"]["updatedToolOutput"]
    assert out["decision"] == "block"
    assert set(replaced) == set(response) and replaced["interrupted"] is False
    assert POISON not in str(out)
    clean = {"stdout": "ok", "stderr": "", "interrupted": False}
    assert _run(hooks, "PostToolUse", tool_name="Bash", tool_input={}, tool_response=clean) == {}


def test_a_hook_that_raises_blocks(ca):
    def broken() -> dict[str, Any]:
        raise RuntimeError("boom")

    hooks = ca.governance_hooks(firewall=_fw(), state=broken)
    assert _run(hooks, "UserPromptSubmit", prompt="hi")["decision"] == "block"
    assert _decision(_run(hooks, "PreToolUse", tool_name="Bash", tool_input={})) == "deny"
    post = _run(hooks, "PostToolUse", tool_name="Bash", tool_input={}, tool_response="out")
    assert post["decision"] == "block"
    audit = ca.governance_hooks(firewall=_fw(), events=("SubagentStart",), state=broken)
    assert _run(audit, "SubagentStart", agent_id="a", agent_type="t") == {}


def test_refused_options_and_malformed_hook_input(ca):
    with pytest.raises(ValueError, match="no meaning"):
        ca.governance_hooks(on_tool_violation="raise")
    with pytest.raises(ValueError, match="unsupported hook event"):
        ca.governance_hooks(events=("Stop",))
    hooks = ca.governance_hooks(tool_policy=ToolPolicy(allow=["Read"]))
    # No tool name to judge is not a pass: it cannot be on an allow-list.
    assert _decision(asyncio.run(hooks["PreToolUse"][0].hooks[0](None, None, None))) == "deny"


def test_report_only_records_and_allows(ca):
    hooks = ca.governance_hooks(
        firewall=_fw(), tool_policy=ToolPolicy(deny=["Bash"]), block_on_violation=False
    )
    assert _run(hooks, "PreToolUse", tool_name="Bash", tool_input={"c": POISON}) == {}
    assert _run(hooks, "UserPromptSubmit", prompt=POISON) == {}


def test_can_use_tool_maps_block_ask_allow(ca):
    policy = ToolPolicy(deny=["mcp__*"], require_approval=["Bash"])
    callback = ca.governance_can_use_tool(tool_policy=policy)
    ctx = types.SimpleNamespace(tool_use_id="t", agent_id=None, decision_reason=None)

    def run(cb, tool, tool_input=None):
        return asyncio.run(cb(tool, tool_input or {}, ctx))

    assert isinstance(run(callback, "mcp__s__t"), _Deny)
    assert isinstance(run(callback, "Bash"), _Deny)  # ask, nobody to ask
    assert isinstance(run(callback, "Read"), _Allow)

    async def human(tool, tool_input, context):
        return _Allow()

    approved = ca.governance_can_use_tool(tool_policy=policy, inner=human)
    assert isinstance(run(approved, "Bash"), _Allow)
    assert isinstance(run(approved, "mcp__s__t"), _Deny)


def test_native_tool_options(ca):
    values = ca.native_tool_options(ToolPolicy(allow=["Read", "mcp__docs__*"], deny=["Bash"]))
    assert values == {
        "disallowed_tools": ["Bash"],
        "tools": ["Read"],
        "strict_mcp_config": True,
    }
    with pytest.raises(ValueError, match="mcp_servers"):
        ca.native_tool_options(ToolPolicy(deny=["mcp__*"]))
    assert ca.native_tool_options(ToolPolicy(deny=["mcp__*"]), mcp_servers=["a"]) == {
        "disallowed_tools": ["mcp__a"]
    }
    with pytest.raises(ValueError, match="cannot be expressed"):
        ca.native_tool_options(ToolPolicy(deny=["Bash*"]))
