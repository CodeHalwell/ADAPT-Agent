"""The OpenAI Agents bindings' own decisions, exercised without the SDK installed.

The default test environment -- CI's included -- does not install
``openai-agents``, so ``tests/test_real_openai_agents.py`` skips there. These
tests drive the real binding code against a stand-in ``agents`` module that has
only the names the bindings use, and assert on what the bindings *decide*: which
guardrail behaviour they return, whether they ask for approval, what a hook
raises, what the input filter sends, which stream events are released. That the
real SDK then honours those decisions is the real-framework file's job.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import types
from typing import Any

import pytest

from adapt_agent.core import PolicyEnforcer
from adapt_agent.core.governance import GovernanceGate, ToolPolicy
from adapt_agent.exceptions import SecurityBlockedError
from adapt_agent.observability import AgentObserver
from adapt_agent.security import Firewall

# -- a stand-in SDK -----------------------------------------------------------------


@dataclasses.dataclass
class _Output:
    output_info: Any
    behavior: dict[str, Any]

    @classmethod
    def allow(cls, output_info: Any = None) -> _Output:
        return cls(output_info, {"type": "allow"})

    @classmethod
    def reject_content(cls, message: str, output_info: Any = None) -> _Output:
        return cls(output_info, {"type": "reject_content", "message": message})

    @classmethod
    def raise_exception(cls, output_info: Any = None) -> _Output:
        return cls(output_info, {"type": "raise_exception"})


@dataclasses.dataclass
class _ToolGuardrail:
    guardrail_function: Any
    name: str | None = None


class _Guardrail:
    def __init__(self, guardrail_function: Any, name: str | None = None) -> None:
        self.guardrail_function = guardrail_function
        self.name = name


class _GuardrailFunctionOutput:
    def __init__(self, output_info: Any, tripwire_triggered: bool) -> None:
        self.output_info = output_info
        self.tripwire_triggered = tripwire_triggered


class _Hooks:
    async def on_llm_start(self, context, agent, system_prompt, input_items):
        return None

    async def on_agent_end(self, context, agent, output):
        return None

    async def on_end(self, context, agent, output):
        return None

    async def on_tool_start(self, context, agent, tool):
        return None

    async def on_tool_end(self, context, agent, tool, result):
        return None

    async def on_handoff(self, context, from_agent, to_agent):
        return None


@dataclasses.dataclass
class _FunctionTool:
    name: str
    tool_input_guardrails: list[Any] | None = None
    tool_output_guardrails: list[Any] | None = None
    needs_approval: Any = False


@dataclasses.dataclass
class _ShellTool:
    name: str = "shell"
    needs_approval: Any = False
    on_approval: Any = None
    environment: Any = None


class _ApplyPatchTool(_ShellTool):
    pass


class _CustomTool(_ShellTool):
    pass


@dataclasses.dataclass
class _HostedMCPTool:
    on_approval_request: Any = None
    name: str = "hosted_mcp"


@dataclasses.dataclass
class _Agent:
    name: str
    tools: list[Any] = dataclasses.field(default_factory=list)
    handoffs: list[Any] = dataclasses.field(default_factory=list)
    mcp_servers: list[Any] = dataclasses.field(default_factory=list)


class _Handoff:
    def __init__(self, agent: Any) -> None:
        self._agent_ref = lambda: agent


@dataclasses.dataclass
class _ModelInputData:
    input: list[Any]
    instructions: str | None


@pytest.fixture
def oa(monkeypatch):
    """The integration module, with a stand-in ``agents`` SDK importable."""
    stub = types.ModuleType("agents")
    stub.ToolGuardrailFunctionOutput = _Output
    stub.ToolInputGuardrail = _ToolGuardrail
    stub.ToolOutputGuardrail = _ToolGuardrail
    stub.GuardrailFunctionOutput = _GuardrailFunctionOutput
    stub.InputGuardrail = _Guardrail
    stub.OutputGuardrail = _Guardrail
    stub.AgentHooks = _Hooks
    stub.RunHooks = _Hooks
    stub.FunctionTool = _FunctionTool
    stub.ShellTool = _ShellTool
    stub.ApplyPatchTool = _ApplyPatchTool
    stub.CustomTool = _CustomTool
    stub.HostedMCPTool = _HostedMCPTool
    stub.Agent = _Agent
    stub.Handoff = _Handoff
    run_config = types.ModuleType("agents.run_config")
    run_config.ModelInputData = _ModelInputData
    stub.run_config = run_config
    monkeypatch.setitem(sys.modules, "agents", stub)
    monkeypatch.setitem(sys.modules, "agents.run_config", run_config)
    from adapt_agent.integrations import openai_agents

    return openai_agents


class _ToolContext:
    """What a tool guardrail / hook receives: name, JSON arguments, approvals."""

    def __init__(
        self,
        tool_name: str,
        arguments: Any = None,
        *,
        call_id: str = "c1",
        approved: bool | None = None,
        **context: Any,
    ) -> None:
        self.tool_name = tool_name
        self.tool_arguments = json.dumps(arguments or {})
        self.tool_call_id = call_id
        self.context = context
        self._approved = approved

    def is_tool_approved(self, tool_name: str, call_id: str) -> bool | None:
        return self._approved


def _firewall() -> Firewall:
    firewall = Firewall()
    firewall.add_blocked_pattern("FORBIDDEN")
    return firewall


def _input(guardrails: dict[str, Any], ctx: _ToolContext) -> _Output:
    guard = guardrails["tool_input_guardrails"][0]
    return asyncio.run(guard.guardrail_function(types.SimpleNamespace(context=ctx, agent=None)))


def _output(guardrails: dict[str, Any], ctx: _ToolContext, result: Any) -> _Output:
    guard = guardrails["tool_output_guardrails"][0]
    data = types.SimpleNamespace(context=ctx, agent=None, output=result)
    return asyncio.run(guard.guardrail_function(data))


# -- tool guardrails ---------------------------------------------------------------------


def test_a_clean_call_and_result_are_allowed(oa):
    guards = oa.governance_tool_guardrails(firewall=_firewall())
    assert _input(guards, _ToolContext("search", {"q": "weather"})).behavior == {"type": "allow"}
    assert _output(guards, _ToolContext("search"), "sunny").behavior == {"type": "allow"}


def test_a_flagged_argument_is_refused_with_a_content_free_message(oa):
    guards = oa.governance_tool_guardrails(firewall=_firewall(), agent_id="ops")
    verdict = _input(guards, _ToolContext("search", {"q": "FORBIDDEN"}))
    assert verdict.behavior["type"] == "reject_content"
    assert "FORBIDDEN" not in verdict.behavior["message"]
    assert verdict.output_info == {
        "agent_id": "ops",
        "tool": "search",
        "action": "block",
        "threats": ["firewall"],
        "report_only": False,
    }


def test_raise_mode_maps_onto_raise_exception(oa):
    guards = oa.governance_tool_guardrails(
        tool_policy=ToolPolicy(deny=["shell"]), on_tool_violation="raise"
    )
    verdict = _input(guards, _ToolContext("shell"))
    assert verdict.behavior == {"type": "raise_exception"}
    assert verdict.output_info["threats"] == ["tool:denied:shell"]


def test_a_flagged_result_is_replaced_not_passed_on(oa):
    guards = oa.governance_tool_guardrails(firewall=_firewall())
    verdict = _output(guards, _ToolContext("fetch"), {"page": "FORBIDDEN text"})
    assert verdict.behavior["type"] == "reject_content"
    assert "FORBIDDEN" not in verdict.behavior["message"]


def test_result_screening_can_be_switched_off(oa):
    assert oa.governance_tool_guardrails(screen_results=False)["tool_output_guardrails"] == []


def test_an_ask_runs_only_with_an_approval_on_record(oa):
    guards = oa.governance_tool_guardrails(tool_policy=ToolPolicy(require_approval=["wire"]))
    assert _input(guards, _ToolContext("wire")).behavior["type"] == "reject_content"
    assert _input(guards, _ToolContext("wire", approved=True)).behavior == {"type": "allow"}


def test_report_only_allows_and_still_reports(oa):
    guards = oa.governance_tool_guardrails(firewall=_firewall(), block_on_violation=False)
    verdict = _input(guards, _ToolContext("search", {"q": "FORBIDDEN"}))
    assert verdict.behavior == {"type": "allow"}
    assert verdict.output_info["report_only"] is True
    assert verdict.output_info["threats"] == ["firewall"]


def test_a_tool_rule_reads_parsed_arguments_and_the_run_context(oa):
    policy = PolicyEnforcer()
    policy.add_rule(
        "untrusted_wire",
        "d",
        "tool['name'] == 'wire' and tool['arguments']['amount'] > 10 and state['trust'] < 1",
        action="block",
    )
    guards = oa.governance_tool_guardrails(policy_enforcer=policy)
    assert _input(guards, _ToolContext("wire", {"amount": 50}, trust=0)).output_info["threats"] == [
        "policy:untrusted_wire"
    ]
    assert _input(guards, _ToolContext("wire", {"amount": 50}, trust=5)).behavior == {
        "type": "allow"
    }


def test_the_observer_records_each_decision(oa):
    observer = AgentObserver()
    guards = oa.governance_tool_guardrails(firewall=_firewall(), observer=observer)
    _input(guards, _ToolContext("search", {"q": "ok"}))
    _input(guards, _ToolContext("search", {"q": "FORBIDDEN"}))
    _output(guards, _ToolContext("search"), "fine")
    assert [(t["operation"], t["status"]) for t in observer.get_traces()] == [
        ("openai_agents.tool_call", "completed"),
        ("openai_agents.tool_call", "error"),
        ("openai_agents.tool_result", "completed"),
    ]


def test_unknown_modes_and_a_dropped_tool_policy_are_refused(oa):
    with pytest.raises(ValueError, match="on_tool_violation"):
        oa.governance_tool_guardrails(on_tool_violation="warn")
    with pytest.raises(ValueError, match="tool_policy"):
        oa.governance_tool_guardrails(gate=GovernanceGate(), tool_policy=ToolPolicy())
    with pytest.raises(ValueError, match="tool_policy"):
        oa.governance_run_hooks(gate=GovernanceGate(), tool_policy=ToolPolicy())
    with pytest.raises(ValueError, match="on_violation"):
        oa.governance_model_input_filter(on_violation="ignore")


# -- govern_tools ------------------------------------------------------------------------


def test_govern_tools_walks_handoffs_agents_as_tools_and_mcp_servers(oa):
    inner_tool = _FunctionTool("inner")
    inner_agent = _Agent("inner", tools=[inner_tool])
    as_tool = _FunctionTool("research")
    as_tool._agent_instance = inner_agent
    server = types.SimpleNamespace(
        tool_input_guardrails=None,
        tool_output_guardrails=None,
        list_tools=lambda: None,
        call_tool=lambda: None,
    )
    billing_tool, support_tool = _FunctionTool("refund"), _FunctionTool("lookup")
    billing = _Agent("billing", tools=[billing_tool])
    support = _Agent("support", tools=[support_tool])
    triage = _Agent("triage", tools=[as_tool], handoffs=[billing, _Handoff(support)])
    triage.mcp_servers = [server]
    billing.handoffs = [triage]  # a cycle

    oa.govern_tools(triage, firewall=_firewall())

    for tool in (inner_tool, as_tool, billing_tool, support_tool, server):
        assert [g.name for g in tool.tool_input_guardrails] == ["adapt-agent"]
        assert [g.name for g in tool.tool_output_guardrails] == ["adapt-agent"]


def test_govern_tools_installs_approval_only_where_the_gate_can_ask(oa):
    asks, plain = _FunctionTool("wire"), _FunctionTool("search")
    oa.govern_tools([asks, plain], tool_policy=ToolPolicy(require_approval=["wire"]))
    assert callable(asks.needs_approval) and plain.needs_approval is False

    oa.govern_tools(
        [asks, plain], tool_policy=ToolPolicy(require_approval=["wire"]), approvals=False
    )
    assert asks.needs_approval is False  # the earlier callable is unwrapped, not stacked


def test_a_rule_naming_one_tool_puts_approval_on_that_tool_only(oa):
    policy = PolicyEnforcer()
    policy.add_rule(
        "big", "d", "tool['name'] == 'refund' and tool['arguments']['n'] > 5", "require_approval"
    )
    refund, search = _FunctionTool("refund"), _FunctionTool("search")
    oa.govern_tools([refund, search], policy_enforcer=policy)
    assert callable(refund.needs_approval) and search.needs_approval is False


def test_the_approval_callable_asks_composes_and_never_asks_about_a_block(oa):
    policy = ToolPolicy(require_approval=["wire"], deny=["nuke"])
    wire, nuke, own = _FunctionTool("wire"), _FunctionTool("nuke"), _FunctionTool("own")
    own.needs_approval = True
    oa.govern_tools([wire, nuke, own], tool_policy=policy)

    async def check(tool: _FunctionTool, call_id: str) -> bool:
        return await tool.needs_approval(_ToolContext(tool.name), {}, call_id)

    assert asyncio.run(check(wire, "w1")) is True
    assert asyncio.run(check(own, "o1")) is True  # the tool's own setting survives
    nuke.needs_approval = oa._ToolGovernor(  # a blocked tool is refused, not sent to a human
        GovernanceGate(tool_policy=policy),
        observer=None,
        on_tool_violation="refuse",
        approvals=True,
        screen_results=True,
    ).needs_approval("nuke", True)
    assert asyncio.run(check(nuke, "n1")) is False


def test_a_pending_ask_is_not_refused_by_a_pre_approval_guardrail_pass(oa):
    wire = _FunctionTool("wire")
    oa.govern_tools([wire], tool_policy=ToolPolicy(require_approval=["wire"]))
    guards = {"tool_input_guardrails": wire.tool_input_guardrails}

    assert asyncio.run(wire.needs_approval(_ToolContext("wire"), {}, "c7")) is True
    assert _input(guards, _ToolContext("wire", call_id="c7")).behavior == {"type": "allow"}
    # A different call that was never sent for approval still fails closed.
    assert _input(guards, _ToolContext("wire", call_id="c8")).behavior["type"] == "reject_content"


def test_shell_tools_auto_reject_a_block_and_leave_an_ask_for_a_human(oa):
    shell = _ShellTool()
    oa.govern_tools(shell, firewall=_firewall(), tool_policy=ToolPolicy(require_approval=["shell"]))
    action = types.SimpleNamespace(commands=["ls"])

    async def flow(commands: list[str], call_id: str) -> tuple[bool, Any]:
        @dataclasses.dataclass
        class Action:
            commands: list[str]

        needs = await shell.needs_approval(_ToolContext("shell"), Action(commands), call_id)
        item = types.SimpleNamespace(call_id=call_id, arguments=None)
        return needs, await shell.on_approval(_ToolContext("shell"), item)

    assert action.commands == ["ls"]
    needs, verdict = asyncio.run(flow(["ls"], "s1"))
    assert needs is True and verdict == {}  # no verdict: a human decides
    needs, verdict = asyncio.run(flow(["FORBIDDEN"], "s2"))
    assert needs is True and verdict["approve"] is False and "FORBIDDEN" not in verdict["reason"]


def test_a_hosted_mcp_approval_callback_is_wrapped_and_others_are_reported(oa):
    calls = []

    def approve(request: Any) -> dict[str, Any]:
        calls.append(request)
        return {"approve": True}

    hosted = _HostedMCPTool(on_approval_request=approve)
    unknown = types.SimpleNamespace(name="web_search")
    with pytest.warns(RuntimeWarning, match="web_search"):
        oa.govern_tools([hosted, unknown], tool_policy=ToolPolicy(deny=["drop_*"]))

    def request(name: str) -> Any:
        data = types.SimpleNamespace(name=name, arguments="{}")
        return types.SimpleNamespace(data=data, ctx_wrapper=None)

    assert asyncio.run(hosted.on_approval_request(request("drop_db")))["approve"] is False
    assert asyncio.run(hosted.on_approval_request(request("search"))) == {"approve": True}
    assert len(calls) == 1


# -- hooks -------------------------------------------------------------------------------


class _RunContext:
    def __init__(self, **context: Any) -> None:
        self.context = context


def test_run_hooks_screen_model_input_and_output(oa):
    hooks = oa.governance_run_hooks(firewall=_firewall(), agent_id="run")
    with pytest.raises(SecurityBlockedError) as blocked:
        asyncio.run(hooks.on_llm_start(_RunContext(), None, "s", [{"content": "FORBIDDEN"}]))
    assert blocked.value.threats == ["firewall"] and "[run]" in blocked.value.reason
    with pytest.raises(SecurityBlockedError):
        asyncio.run(hooks.on_agent_end(_RunContext(), None, "FORBIDDEN"))
    asyncio.run(hooks.on_llm_start(_RunContext(), None, "s", [{"content": "hello"}]))


def test_the_tool_backstop_skips_governed_tools_and_stops_the_rest(oa):
    hooks = oa.governance_run_hooks(firewall=_firewall(), tool_policy=ToolPolicy(deny=["nuke"]))
    governed = _FunctionTool("nuke")
    oa.govern_tools(governed, tool_policy=ToolPolicy(deny=["nuke"]))

    asyncio.run(hooks.on_tool_start(_ToolContext("nuke"), None, governed))  # its own seam decides
    with pytest.raises(SecurityBlockedError):
        asyncio.run(hooks.on_tool_start(_ToolContext("nuke"), None, _FunctionTool("nuke")))
    with pytest.raises(SecurityBlockedError):
        asyncio.run(hooks.on_tool_start(_RunContext(), None, types.SimpleNamespace(name="nuke")))
    with pytest.raises(SecurityBlockedError):
        asyncio.run(
            hooks.on_tool_end(_ToolContext("x"), None, _FunctionTool("x"), "FORBIDDEN result")
        )


def test_the_tool_backstop_lets_an_approved_ask_through(oa):
    hooks = oa.governance_run_hooks(tool_policy=ToolPolicy(require_approval=["wire"]))
    tool = _FunctionTool("wire")
    with pytest.raises(SecurityBlockedError):
        asyncio.run(hooks.on_tool_start(_ToolContext("wire"), None, tool))
    asyncio.run(hooks.on_tool_start(_ToolContext("wire", approved=True), None, tool))


def test_run_hooks_delegate_to_the_apps_hooks_after_governance(oa):
    seen = []

    class Inner(_Hooks):
        async def on_llm_start(self, context, agent, system_prompt, input_items):
            seen.append("llm")

        async def on_agent_end(self, context, agent, output):
            seen.append("end")

        async def on_handoff(self, context, from_agent, to_agent):
            seen.append("handoff")

    hooks = oa.governance_run_hooks(firewall=_firewall(), inner=Inner())
    with pytest.raises(SecurityBlockedError):
        asyncio.run(hooks.on_llm_start(_RunContext(), None, "s", ["FORBIDDEN"]))
    asyncio.run(hooks.on_llm_start(_RunContext(), None, "s", ["hi"]))
    asyncio.run(hooks.on_agent_end(_RunContext(), None, "ok"))
    asyncio.run(hooks.on_handoff(_RunContext(), None, None))
    assert seen == ["llm", "end", "handoff"]


def test_agent_hooks_gain_the_tool_backstop_and_report_only(oa):
    hooks = oa.governance_agent_hooks(tool_policy=ToolPolicy(deny=["nuke"]))
    with pytest.raises(SecurityBlockedError):
        asyncio.run(hooks.on_tool_start(_ToolContext("nuke"), None, _FunctionTool("nuke")))
    quiet = oa.governance_agent_hooks(firewall=_firewall(), block_on_violation=False)
    asyncio.run(quiet.on_llm_start(_RunContext(), None, "s", ["FORBIDDEN"]))  # no raise


def test_guardrails_report_only_never_trip(oa):
    guards = oa.governance_guardrails(firewall=_firewall(), block_on_violation=False)
    guard = guards["input_guardrails"][0]
    verdict = asyncio.run(guard.guardrail_function(_RunContext(), None, "FORBIDDEN"))
    assert verdict.tripwire_triggered is False and verdict.output_info["threats"] == ["firewall"]


# -- the model-input filter ----------------------------------------------------------------


def _call_model_data(items: list[Any], **context: Any) -> Any:
    return types.SimpleNamespace(
        model_data=_ModelInputData(input=items, instructions="sys"), agent=None, context=context
    )


def test_the_filter_redacts_tool_results_and_messages(oa):
    flt = oa.governance_model_input_filter(firewall=_firewall(), on_violation="redact")
    items = [
        {"role": "user", "content": "hi"},
        {"type": "function_call_output", "call_id": "c1", "output": "FORBIDDEN page"},
        {"role": "user", "content": "FORBIDDEN"},
    ]
    sent = asyncio.run(flt(_call_model_data(items)))
    assert sent.instructions == "sys"
    assert sent.input[0] == {"role": "user", "content": "hi"}
    assert sent.input[1]["output"] == "[withheld by ADAPT-Agent: firewall]"
    assert sent.input[2]["content"] == "[withheld by ADAPT-Agent: firewall]"
    assert items[1]["output"] == "FORBIDDEN page"  # the SDK's own history is not mutated


def test_the_filter_fails_closed_on_what_it_cannot_redact(oa):
    flt = oa.governance_model_input_filter(firewall=_firewall(), on_violation="redact")
    call = {"type": "function_call", "name": "x", "arguments": "FORBIDDEN"}
    with pytest.raises(SecurityBlockedError):
        asyncio.run(flt(_call_model_data([call])))

    policy = PolicyEnforcer()
    policy.add_rule("low", "d", "state['trust'] < 1", action="block")
    flt = oa.governance_model_input_filter(policy_enforcer=policy, on_violation="redact")
    with pytest.raises(SecurityBlockedError) as blocked:
        asyncio.run(flt(_call_model_data([{"role": "user", "content": "hi"}], trust=0)))
    assert blocked.value.threats == ["policy:low"]
    sent = asyncio.run(flt(_call_model_data([{"role": "user", "content": "hi"}], trust=5)))
    assert sent.input == [{"role": "user", "content": "hi"}]


def test_the_filter_raises_by_default_and_runs_the_inner_filter_first(oa):
    def inner(data: Any) -> Any:
        return _ModelInputData(input=[{"role": "user", "content": "FORBIDDEN"}], instructions=None)

    flt = oa.governance_model_input_filter(firewall=_firewall(), inner=inner)
    with pytest.raises(SecurityBlockedError):
        asyncio.run(flt(_call_model_data([{"role": "user", "content": "hi"}])))


# -- streaming -----------------------------------------------------------------------------


def _raw(kind: str, **fields: Any) -> Any:
    return types.SimpleNamespace(
        type="raw_response_event", data=types.SimpleNamespace(type=kind, **fields)
    )


class _FakeStream:
    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.cancelled = False

    async def stream_events(self) -> Any:
        for event in self.events:
            if self.cancelled:
                return
            yield event

    def cancel(self, mode: str = "immediate") -> None:
        self.cancelled = True


def _response(text: str) -> list[Any]:
    item = types.SimpleNamespace(type="message")
    return [
        _raw("response.created"),
        _raw("response.output_item.added", item=item),
        _raw("response.output_text.delta", delta=text[:4]),
        _raw("response.output_text.delta", delta=text[4:]),
        _raw("response.output_text.done", text=text),
        _raw("response.completed"),
    ]


def _drain(stream: Any) -> tuple[list[Any], BaseException | None]:
    got: list[Any] = []

    async def main() -> None:
        async for event in stream:
            got.append(event)

    try:
        asyncio.run(main())
    except SecurityBlockedError as exc:
        return got, exc
    return got, None


def test_a_clean_response_is_released_in_order(oa):
    events = _response("hello world")
    got, error = _drain(oa.govern_stream(_FakeStream(events), firewall=_firewall()))
    assert error is None
    assert [e.data.type for e in got] == [e.data.type for e in events]


def test_a_flagged_response_is_withheld_and_the_run_cancelled(oa):
    source = _FakeStream(_response("FORBIDDEN text") + _response("later"))
    got, error = _drain(oa.govern_stream(source, firewall=_firewall()))
    assert isinstance(error, SecurityBlockedError)
    assert [e.data.type for e in got] == ["response.created"]  # no text of it delivered
    assert source.cancelled is True


def test_non_text_events_are_not_held(oa):
    call = types.SimpleNamespace(type="function_call")
    events = [
        _raw("response.created"),
        _raw("response.output_item.added", item=call),
        _raw("response.function_call_arguments.delta", delta='{"q": 1}'),
        _raw("response.output_text.delta", delta="note"),
    ]
    seen: list[str] = []

    async def main() -> None:
        stream = oa.govern_stream(_FakeStream(events), firewall=_firewall())
        async for event in stream:
            seen.append(event.data.type)
            if event.data.type == "response.function_call_arguments.delta":
                assert "response.output_text.delta" not in seen

    asyncio.run(main())
    # A stream that ends without a completion event is still screened, then released.
    assert seen[-1] == "response.output_text.delta"


def test_a_report_only_stream_releases_and_records(oa):
    stream = oa.govern_stream(
        _FakeStream(_response("FORBIDDEN")), firewall=_firewall(), block_on_violation=False
    )
    got, error = _drain(stream)
    assert error is None and len(got) == 6 and stream.threats == ["firewall"]


# -- helpers and the adapter ----------------------------------------------------------------


def test_find_security_block_walks_the_cause_chain(oa):
    inner = SecurityBlockedError("blocked", ["firewall"])
    try:
        try:
            raise inner
        except SecurityBlockedError as exc:
            raise RuntimeError("Error running tool x") from exc
    except RuntimeError as wrapped:
        assert oa.find_security_block(wrapped) is inner
    assert oa.find_security_block(ValueError("x")) is None
    assert oa.find_security_block(None) is None


class _FakeRunState:
    def approve(self, item: Any) -> None: ...

    def reject(self, item: Any) -> None: ...

    def to_json(self) -> dict[str, Any]:
        return {}


def test_the_adapter_passes_a_run_state_through_instead_of_stringifying_it():
    from adapt_agent.adapters import OpenAIAgentsAdapter

    seen: list[Any] = []

    def run(prompt: Any, **kwargs: Any) -> Any:
        seen.append((prompt, kwargs))
        return types.SimpleNamespace(final_output="ok", interruptions=[])

    state = _FakeRunState()
    guarded = OpenAIAgentsAdapter().wrap_agent(run)
    guarded.execute({"input": state}, session="S", max_turns=3)
    guarded.execute({"input": [{"role": "user", "content": "hi"}]})
    guarded.execute({"messages": [{"role": "user", "content": "hello"}]})
    assert seen[0] == (state, {"session": "S", "max_turns": 3})
    assert seen[1][0] == [{"role": "user", "content": "hi"}]
    assert seen[2][0] == "hello"
    assert OpenAIAgentsAdapter()._prepare_input(state) is state


def test_the_adapter_passes_a_bare_list_of_input_items_through_whole():
    """Given directly, a conversation used to be reduced to its latest user text."""
    from adapt_agent.adapters import OpenAIAgentsAdapter

    seen: list[Any] = []

    def run(prompt: Any, **kwargs: Any) -> Any:
        seen.append(prompt)
        return types.SimpleNamespace(final_output="ok", interruptions=[])

    items = [
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "Hello Ada."},
        {"role": "user", "content": "What is my name?"},
    ]
    guarded = OpenAIAgentsAdapter().wrap_agent(run)
    guarded.execute(items)
    guarded.execute({"input": items})
    assert seen == [items, items]


def test_a_bare_list_of_input_items_is_still_screened():
    from adapt_agent.adapters import OpenAIAgentsAdapter
    from adapt_agent.exceptions import SecurityBlockedError
    from adapt_agent.security import Firewall

    firewall = Firewall()
    firewall.add_blocked_pattern("SECRET-7")
    ran: list[Any] = []
    guarded = OpenAIAgentsAdapter(firewall=firewall).wrap_agent(
        lambda prompt, **kw: ran.append(prompt)
    )
    with pytest.raises(SecurityBlockedError):
        guarded.execute([{"role": "user", "content": "the code is SECRET-7"}])
    assert ran == []


def test_the_adapter_surfaces_interruptions_only_when_there_are_some():
    from adapt_agent.adapters import OpenAIAgentsAdapter

    pending = types.SimpleNamespace(name="wire")
    results = iter(
        [
            types.SimpleNamespace(final_output=None, interruptions=[pending]),
            types.SimpleNamespace(final_output="done", interruptions=[]),
        ]
    )
    guarded = OpenAIAgentsAdapter().wrap_agent(lambda prompt: next(results))

    paused = guarded.execute({"prompt": "wire"})
    assert paused["interruptions"] == [pending]
    assert paused.__adapt_governed_envelope__ is True

    done = guarded.execute({"prompt": "wire"})
    assert list(done) == ["result"]


def test_the_adapter_surfaces_interruptions_on_the_async_path():
    from adapt_agent.adapters import OpenAIAgentsAdapter

    pending = types.SimpleNamespace(name="wire")

    async def run(prompt: Any, **kwargs: Any) -> Any:
        return types.SimpleNamespace(final_output=None, interruptions=[pending])

    out = asyncio.run(OpenAIAgentsAdapter().wrap_agent(run).aexecute({"prompt": "x"}))
    assert out["interruptions"] == [pending]
