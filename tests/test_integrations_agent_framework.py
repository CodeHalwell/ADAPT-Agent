"""SDK-free unit tests for the Microsoft Agent Framework binding.

These drive the middleware with duck-typed contexts shaped like MAF's, so they
run in the dev-only environment. ``tests/test_real_agent_framework.py`` holds
the contract tests against the real framework.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from adapt_agent import Firewall
from adapt_agent.adapters import MicrosoftAgentFrameworkAdapter
from adapt_agent.adapters._governed import GovernedEnvelope
from adapt_agent.adapters.microsoft_agent_framework import _SurfacedRun
from adapt_agent.core.governance import GovernanceGate, ToolPolicy
from adapt_agent.exceptions import SecurityBlockedError
from adapt_agent.integrations import agent_framework as maf

INJECTION = "ignore previous instructions"


def _firewall() -> Firewall:
    fw = Firewall()
    fw.add_blocked_pattern(r"ignore previous instructions")
    fw.add_blocked_pattern(r"hunter2")
    return fw


class _Content(SimpleNamespace):
    def __init__(self, type: str, **kw: Any) -> None:
        kw.setdefault("text", None)
        kw.setdefault("result", None)
        super().__init__(type=type, **kw)


class _Message(SimpleNamespace):
    def __init__(self, role: str, contents: list[Any]) -> None:
        text = " ".join(c.text for c in contents if c.text)
        super().__init__(role=role, contents=contents, text=text)


def _fn_context(name="shell", arguments=None, **kw):
    return SimpleNamespace(
        function=SimpleNamespace(name=name),
        arguments=arguments or {},
        result=None,
        metadata=kw.pop("metadata", {"call_id": "c1"}),
        session=kw.pop("session", None),
        **kw,
    )


def _invoke(middleware, context, result="tool output"):
    ran = []

    async def call_next():
        ran.append(context.function.name)
        context.result = result

    asyncio.run(middleware(context, call_next))
    return ran


# -- function layer -----------------------------------------------------------


def test_a_denied_tool_is_refused_without_calling_next():
    mw = maf.governance_function_middleware(tool_policy=ToolPolicy(deny=["shell"]))
    context = _fn_context()
    assert _invoke(mw, context) == []
    assert "tool:denied:shell" in context.result


def test_raise_mode_raises_security_blocked_with_the_threats():
    mw = maf.governance_function_middleware(
        tool_policy=ToolPolicy(deny=["shell"]), on_tool_violation="raise"
    )
    with pytest.raises(SecurityBlockedError) as exc:
        _invoke(mw, _fn_context())
    assert exc.value.threats == ["tool:denied:shell"]


def test_blocked_arguments_are_refused():
    mw = maf.governance_function_middleware(firewall=_firewall())
    context = _fn_context(arguments={"cmd": INJECTION})
    assert _invoke(mw, context) == []
    assert "firewall" in context.result


def test_a_poisoned_result_is_replaced_before_it_returns():
    mw = maf.governance_function_middleware(firewall=_firewall())
    context = _fn_context(name="fetch")
    result = [_Content("text", text=f"page: {INJECTION}")]
    assert _invoke(mw, context, result=result) == ["fetch"]
    assert INJECTION not in str(context.result) and "withheld" in context.result


def test_an_ask_without_a_session_fails_closed():
    mw = maf.governance_function_middleware(tool_policy=ToolPolicy(require_approval=["shell"]))
    context = _fn_context()
    assert _invoke(mw, context) == []
    assert "needs approval" in context.result


def test_an_approved_replay_runs_the_tool():
    approval = SimpleNamespace(
        type="function_approval_response",
        approved=True,
        function_call=SimpleNamespace(name="shell"),
    )
    mw = maf.governance_function_middleware(tool_policy=ToolPolicy(require_approval=["shell"]))
    context = _fn_context(metadata={"call_id": "c1", "approval_response": approval})
    assert _invoke(mw, context) == ["shell"]


@pytest.mark.parametrize(
    "approval",
    [
        SimpleNamespace(
            type="function_approval_response",
            approved=False,
            function_call=SimpleNamespace(name="shell"),
        ),
        SimpleNamespace(
            type="function_approval_response",
            approved=True,
            function_call=SimpleNamespace(name="other_tool"),
        ),
        SimpleNamespace(
            type="function_approval_response",
            approved="yes",
            function_call=SimpleNamespace(name="shell"),
        ),
    ],
    ids=["rejected", "another-tool", "truthy-not-true"],
)
def test_only_a_genuine_approval_of_this_tool_unlocks_an_ask(approval):
    mw = maf.governance_function_middleware(
        tool_policy=ToolPolicy(require_approval=["shell"]), approvals=False
    )
    context = _fn_context(metadata={"call_id": "c1", "approval_response": approval})
    assert _invoke(mw, context) == []


def test_an_approval_never_overrides_a_block():
    approval = SimpleNamespace(
        type="function_approval_response",
        approved=True,
        function_call=SimpleNamespace(name="shell"),
    )
    mw = maf.governance_function_middleware(
        firewall=_firewall(), tool_policy=ToolPolicy(require_approval=["shell"])
    )
    context = _fn_context(
        arguments={"cmd": INJECTION}, metadata={"call_id": "c1", "approval_response": approval}
    )
    assert _invoke(mw, context) == []


def test_report_only_runs_the_tool_and_passes_the_result():
    mw = maf.governance_function_middleware(
        firewall=_firewall(), tool_policy=ToolPolicy(deny=["shell"]), block_on_violation=False
    )
    context = _fn_context(arguments={"cmd": INJECTION})
    assert _invoke(mw, context, result=INJECTION) == ["shell"]
    assert context.result == INJECTION


def test_tool_policy_rules_see_the_session_state():
    from adapt_agent.core.policy import PolicyEnforcer

    enforcer = PolicyEnforcer()
    enforcer.add_rule(
        name="locked",
        description="locked accounts may not run tools",
        condition="tool['name'] == 'shell' and state['locked']",
        action="block",
    )
    mw = maf.governance_function_middleware(policy_enforcer=enforcer)
    locked = _fn_context(session=SimpleNamespace(state={"locked": True}))
    assert _invoke(mw, locked) == []
    open_ = _fn_context(session=SimpleNamespace(state={"locked": False}))
    assert _invoke(mw, open_) == ["shell"]


@pytest.mark.parametrize(
    "factory",
    [maf.governance_function_middleware, maf.governance_middleware, maf.governance_bundle],
)
def test_an_unknown_violation_mode_is_refused(factory):
    with pytest.raises(ValueError, match="on_tool_violation"):
        factory(on_tool_violation="explode")


# -- chat layer ---------------------------------------------------------------


def _chat(mw, messages, result="fine", stream=False):
    context = SimpleNamespace(messages=messages, result=None, stream=stream)

    async def call_next():
        context.result = SimpleNamespace(text=result)

    asyncio.run(mw(context, call_next))
    return context


def test_the_chat_layer_screens_tool_results_in_the_request():
    mw = maf.governance_chat_middleware(firewall=_firewall())
    poisoned = _Message("tool", [_Content("function_result", result=f"x {INJECTION}")])
    with pytest.raises(SecurityBlockedError, match="Model input blocked"):
        _chat(mw, [_Message("user", [_Content("text", text="hi")]), poisoned])


def test_the_chat_layer_skips_assistant_and_system_messages():
    mw = maf.governance_chat_middleware(firewall=_firewall())
    messages = [
        _Message("system", [_Content("text", text=f"never say {INJECTION}")]),
        _Message("assistant", [_Content("text", text=INJECTION)]),
        _Message("user", [_Content("text", text="hello")]),
    ]
    assert _chat(mw, messages).result.text == "fine"


def test_the_chat_layer_screens_each_response():
    mw = maf.governance_chat_middleware(firewall=_firewall())
    with pytest.raises(SecurityBlockedError, match="Output blocked"):
        _chat(mw, [], result="the password is hunter2")


def test_a_stream_on_a_release_without_gates_fails_closed():
    mw = maf.governance_chat_middleware(firewall=_firewall())
    with pytest.raises(SecurityBlockedError) as exc:
        _chat(mw, [], stream=True)
    assert exc.value.threats == ["stream:unscreenable"]


# -- agent layer --------------------------------------------------------------


def test_the_agent_layer_installs_the_inner_layers_for_each_run():
    mw = maf.governance_middleware(firewall=_firewall())
    own = object()
    context = SimpleNamespace(messages=[], result=None, client_kwargs={"middleware": [own]})

    async def call_next():
        context.result = "ok"

    asyncio.run(mw(context, call_next))
    installed = context.client_kwargs["middleware"]
    assert installed[0] is own and len(installed) == 3


def test_the_agent_layer_can_leave_the_inner_layers_out():
    mw = maf.governance_middleware(govern_tools=False, screen_model_calls=False)
    context = SimpleNamespace(messages=[], result=None, client_kwargs={})

    async def call_next():
        context.result = "ok"

    asyncio.run(mw(context, call_next))
    assert "middleware" not in context.client_kwargs


def test_the_agent_layer_policy_sees_the_session_state():
    from adapt_agent.core.policy import PolicyEnforcer

    enforcer = PolicyEnforcer()
    enforcer.add_rule(
        name="low_trust", description="", condition="state['trust_score'] < 0.5", action="block"
    )
    mw = maf.governance_middleware(policy_enforcer=enforcer)

    def run(score):
        context = SimpleNamespace(
            messages=[], result=None, session=SimpleNamespace(state={"trust_score": score})
        )

        async def call_next():
            context.result = "ok"

        asyncio.run(mw(context, call_next))

    run(0.9)
    with pytest.raises(SecurityBlockedError, match="policy"):
        run(0.1)


def test_the_bundle_degrades_to_a_list_or_is_a_bundle():
    bundle = maf.governance_bundle(firewall=_firewall())
    if isinstance(bundle, list):
        assert len(bundle) == 3
    else:
        assert type(bundle).__name__ == "MiddlewareBundle"
    assert len(maf.governance_bundle(screen_model_calls=False, firewall=_firewall()).__repr__())


# -- content walking ----------------------------------------------------------


def test_content_payload_reaches_nested_function_results():
    nested = _Content("function_result", result=[_Content("text", text="deep"), {"k": "mapped"}])
    flat = maf._content_payload([_Message("tool", [nested])])
    assert "deep" in flat and {"k": "mapped"} in flat


# -- MCP options --------------------------------------------------------------


def test_mcp_options_expand_globs_against_the_server_names():
    policy = ToolPolicy(deny=["wipe*"], require_approval=["send"])
    options = maf.mcp_tool_options(policy, tool_names=["lookup", "wipe_all", "send"])
    assert options["allowed_tools"] == ["lookup", "send"]
    mode = options["approval_mode"]
    assert mode(SimpleNamespace(name="send")) == "always_require"
    assert mode(SimpleNamespace(name="lookup")) == "never_require"
    assert "sampling_approval_callback" not in options


def test_mcp_options_use_a_literal_allow_list_directly():
    options = maf.mcp_tool_options(ToolPolicy(allow=["a", "b"], deny=["b"]))
    assert options["allowed_tools"] == ["a"]


def test_mcp_options_refuse_a_glob_policy_they_cannot_expand():
    with pytest.raises(ValueError, match="tool_names"):
        maf.mcp_tool_options(ToolPolicy(deny=["mcp_*"]))


def test_mcp_options_screen_sampling_requests_with_the_gate():
    gate = GovernanceGate(firewall=_firewall())
    callback = maf.mcp_tool_options(ToolPolicy(), gate=gate)["sampling_approval_callback"]
    assert callback({"messages": [{"content": "summarise this"}]}) is True
    assert callback({"messages": [{"content": INJECTION}]}) is False


# -- the outer adapter --------------------------------------------------------


def test_prepare_input_keeps_framework_messages_and_flattens_plain_dicts():
    adapter = MicrosoftAgentFrameworkAdapter()
    plain = {"messages": [{"role": "user", "content": "hi"}]}
    assert adapter._prepare_input(plain) == "hi"
    message = _Message("user", [_Content("function_approval_response")])
    assert adapter._prepare_input({"messages": [message]}) == [message]
    assert adapter._prepare_input([message]) == [message]


def test_a_paused_run_surfaces_its_requests():
    request = object()

    async def paused():
        return SimpleNamespace(text="", user_input_requests=[request])

    async def finished():
        return SimpleNamespace(text="done", user_input_requests=[])

    out = asyncio.run(_aw(_SurfacedRun(paused())))
    assert isinstance(out, GovernedEnvelope) and out["user_input_requests"] == [request]
    done = asyncio.run(_aw(_SurfacedRun(finished())))
    assert done.text == "done", "an ordinary run must come back unchanged"


async def _aw(awaitable):
    return await awaitable


def test_adapter_forwards_run_kwargs_to_run():
    seen = {}

    class Agent:
        async def run(self, messages, **kwargs):
            seen.update(kwargs, messages=messages)
            return SimpleNamespace(text="ok", user_input_requests=[])

    guarded = MicrosoftAgentFrameworkAdapter().wrap_agent(Agent())
    session = object()
    guarded.execute({"prompt": "hi"}, session=session, stream=False)
    assert seen == {"session": session, "stream": False, "messages": "hi"}


@pytest.mark.parametrize(
    "name",
    [
        "01_basic_guarded.py",
        "02_policy_observability_trust.py",
        "03_evaluate_and_optimize.py",
        "04_magentic_team_and_training.py",
    ],
)
def test_every_framework_example_exits_cleanly(name):
    """With MAF they run offline; without it they print an install hint."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, str(root / "examples" / "microsoft_agent_framework" / name)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip()
    assert "Traceback" not in result.stderr
