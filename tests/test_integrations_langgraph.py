"""SDK-free unit tests for the LangGraph / LangChain v1 binding.

These run in the dev-only environment (no LangChain installed) and pin the
parts of the binding that do not need the framework: decision parsing, policy
state, fail-closed paths, lazy imports, and the adapter's output handling.
The real-framework contract lives in ``tests/test_real_langgraph.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from adapt_agent.adapters import LangGraphAdapter
from adapt_agent.core.governance import GovernanceGate, ToolPolicy
from adapt_agent.core.policy import PolicyEnforcer
from adapt_agent.exceptions import SecurityBlockedError
from adapt_agent.integrations import langgraph as lg
from adapt_agent.security import Firewall


def _firewall() -> Firewall:
    fw = Firewall()
    fw.add_blocked_pattern(r"SECRET-\d+")
    return fw


class _Request:
    """The duck-typed shape of a LangGraph ``ToolCallRequest``."""

    def __init__(self, name: str, args: dict[str, Any], state: Any = None, runtime: Any = None):
        self.tool_call = {"name": name, "args": args, "id": "c1"}
        self.state = state if state is not None else {"messages": []}
        self.runtime = runtime

    def override(self, **kw: Any) -> _Request:
        new = _Request(self.tool_call["name"], self.tool_call["args"], self.state, self.runtime)
        new.tool_call = kw.get("tool_call", self.tool_call)
        return new


def _governor(gate: GovernanceGate, **kw: Any) -> Any:
    options = {"on_tool_violation": "raise", "approvals": True, "approval_args": True}
    options.update(kw)
    return lg._ToolGovernor(gate, observer=None, **options)


# -- decisions ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "kind"),
    [
        ({"type": "approve"}, "approve"),
        ({"type": "reject", "message": "no"}, "reject"),
        ({"type": "edit", "edited_action": {}}, "edit"),
        ({"type": "respond", "message": "x"}, "respond"),
        ({"decisions": [{"type": "approve"}]}, "approve"),
        ({"decisions": [{"type": "approve"}, {"type": "approve"}]}, "invalid"),
        ({"decisions": "approve"}, "invalid"),
        (True, "approve"),
        (False, "invalid"),
        ("approve", "approve"),
        ("reject", "reject"),
        ("yes please", "invalid"),
        ({"type": "APPROVE"}, "invalid"),
        (None, "invalid"),
        (1, "invalid"),
    ],
)
def test_resume_values_are_parsed_strictly(answer, kind):
    assert lg._parse_decision(answer)[0] == kind


def test_an_ask_without_langgraph_fails_closed():
    gate = GovernanceGate(tool_policy=ToolPolicy(require_approval=["wire"]))
    ran: list[str] = []
    with pytest.raises(SecurityBlockedError, match="refused") as info:
        _governor(gate).wrap(_Request("wire", {}), lambda r: ran.append("ran"))
    assert ran == [] and info.value.threats == ["tool:approval:wire"]


def test_a_denied_call_raises_before_the_handler():
    gate = GovernanceGate(tool_policy=ToolPolicy(deny=["rm"]))
    ran: list[str] = []
    with pytest.raises(SecurityBlockedError):
        _governor(gate).wrap(_Request("rm", {}), lambda r: ran.append("ran"))

    async def handler(r: Any) -> None:
        ran.append("ran")

    with pytest.raises(SecurityBlockedError):
        asyncio.run(_governor(gate).awrap(_Request("rm", {}), handler))
    assert ran == []


def test_a_poisoned_result_raises_in_raise_mode():
    gate = GovernanceGate(firewall=_firewall())
    with pytest.raises(SecurityBlockedError, match="withheld"):
        _governor(gate).wrap(_Request("fetch", {}), lambda r: "SECRET-1")


def test_report_only_runs_and_returns_everything():
    gate = GovernanceGate(
        firewall=_firewall(), tool_policy=ToolPolicy(deny=["rm"]), block_on_violation=False
    )
    assert _governor(gate).wrap(_Request("rm", {"a": "SECRET-1"}), lambda r: "SECRET-2") == (
        "SECRET-2"
    )


def test_unknown_violation_mode_is_refused_loudly():
    with pytest.raises(ValueError, match="on_tool_violation"):
        lg.governance_tool_wrapper(on_tool_violation="ignore")
    with pytest.raises(ValueError, match="on_tool_violation"):
        lg.governance_middleware(on_tool_violation="warn")


def test_tool_wrapper_returns_the_toolnode_keywords():
    wrappers = lg.governance_tool_wrapper()
    assert set(wrappers) == {"wrap_tool_call", "awrap_tool_call"}
    assert asyncio.iscoroutinefunction(wrappers["awrap_tool_call"])


def test_approval_request_carries_labels_not_withheld_content():
    gate = GovernanceGate(tool_policy=ToolPolicy(require_approval=["wire"]), agent_id="pay")
    decision = gate.check_tool_call("wire", {"to": "acme"})
    value = _governor(gate)._approval_request({"name": "wire", "args": {"to": "acme"}}, decision)
    assert value["type"] == "tool_approval" and value["threats"] == ["tool:approval:wire"]
    assert value["agent_id"] == "pay" and value["description"] == decision.reason
    hidden = _governor(gate, approval_args=False)._approval_request(
        {"name": "wire", "args": {"to": "acme"}}, decision
    )
    assert "acme" not in repr(hidden)


# -- policy state -----------------------------------------------------------------


@dataclasses.dataclass
class _DataState:
    messages: list[Any]
    trust_score: float


try:
    from pydantic import BaseModel
except ImportError:  # pydantic is not a dependency of the core package

    class _PydanticState:
        """A stand-in shaped like a Pydantic v2 model: declared fields, ``model_dump``."""

        model_fields: dict[str, Any] = {"messages": None, "trust_score": None}

        def __init__(self, messages: list[Any], trust_score: float) -> None:
            self.messages = messages
            self.trust_score = trust_score

        def model_dump(self) -> dict[str, Any]:
            return {"messages": self.messages, "trust_score": self.trust_score}

else:

    class _PydanticState(BaseModel):  # type: ignore[no-redef]
        messages: list[Any] = []
        trust_score: float = 0.0


class _Runtime:
    def __init__(self, context: Any):
        self.context = context


@pytest.mark.parametrize(
    "state",
    [
        {"messages": ["hi"], "trust_score": 0.2},
        _DataState(messages=["hi"], trust_score=0.2),
        _PydanticState(messages=["hi"], trust_score=0.2),
    ],
)
def test_policy_state_reads_every_state_schema(state):
    built = lg.policy_state(["screened"], state, _Runtime({"role": "intern"}))
    assert built == {"messages": ["screened"], "context": {"role": "intern"}, "trust_score": 0.2}


def test_policy_state_without_runtime_context_keeps_a_context_key():
    assert lg.policy_state([], {"messages": []}, _Runtime(None))["context"] == {}
    assert lg.policy_state([], {"context": {"a": 1}})["context"] == {"a": 1}


def test_governance_node_evaluates_policy_on_a_non_dict_state():
    """A Pydantic graph state used to reach the enforcer as `None`: policy off."""
    enforcer = PolicyEnforcer()
    enforcer.add_rule("low_trust", "d", "state['trust_score'] < 0.5", action="block")
    node = lg.governance_node(GovernanceGate(policy_enforcer=enforcer))
    with pytest.raises(SecurityBlockedError, match="policy"):
        node(_PydanticState(messages=["hi"], trust_score=0.1))
    assert node(_PydanticState(messages=["hi"], trust_score=0.9)) == {}


def test_governance_hooks_accept_a_tool_policy():
    hooks = lg.governance_hooks(tool_policy=ToolPolicy(deny=["x"]))
    assert set(hooks) == {"pre_model_hook", "post_model_hook"}


# -- model screening helpers --------------------------------------------------------


def test_screen_stream_refuses_a_model_it_cannot_tag():
    with pytest.raises(TypeError, match="screen_stream"):
        lg._without_token_stream(object())


def test_previous_model_turns_are_screened_by_text_only():
    class _AI:
        type = "ai"
        content = "earlier answer"
        tool_calls = [{"args": {"k": "SECRET-1"}}]

    assert lg._screenable(["user text", _AI()]) == ["user text", "earlier answer"]


# -- lazy imports ---------------------------------------------------------------------


def test_importing_the_binding_imports_no_framework():
    code = (
        "import sys; import adapt_agent.integrations.langgraph, adapt_agent.adapters.langgraph;"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('langchain','langgraph',"
        "'langchain_core')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# -- adapter ----------------------------------------------------------------------------


@dataclasses.dataclass
class GraphOutput:  # the name the adapter duck-types on
    value: Any
    interrupts: tuple[Any, ...] = ()


def test_v2_graph_output_is_unpacked_with_its_interrupts():
    agent = LangGraphAdapter().wrap_agent(lambda p: GraphOutput({"messages": []}, ("pending",)))
    assert agent.execute({"messages": []}) == {"messages": [], "__interrupt__": ["pending"]}


def test_v2_graph_output_is_unpacked_on_the_async_path():
    class Graph:
        def invoke(self, payload: Any, **kw: Any) -> Any:  # pragma: no cover - not used
            raise AssertionError

        async def ainvoke(self, payload: Any, **kw: Any) -> Any:
            return GraphOutput({"messages": ["done"]})

    out = asyncio.run(LangGraphAdapter().wrap_agent(Graph()).aexecute({"messages": []}))
    assert out == {"messages": ["done"]}


def test_run_kwargs_reach_invoke():
    seen: dict[str, Any] = {}

    class Graph:
        def invoke(self, payload: Any, **kw: Any) -> Any:
            seen.update(kw)
            return {"messages": []}

    LangGraphAdapter().wrap_agent(Graph()).execute(
        {"messages": []}, config={"configurable": {"thread_id": "t"}}
    )
    assert seen == {"config": {"configurable": {"thread_id": "t"}}}


@dataclasses.dataclass
class Command:  # the name the adapter duck-types on
    resume: Any = None
    update: Any = None


class _CheckpointedGraph:
    def __init__(self, values: dict[str, Any]):
        self.values = values
        self.inputs: list[Any] = []

    def get_state(self, config: Any) -> Any:
        return type("Snapshot", (), {"values": self.values})()

    def invoke(self, payload: Any, **kw: Any) -> Any:
        self.inputs.append(payload)
        return {"messages": []}


def test_a_resume_is_judged_against_the_checkpointed_state():
    enforcer = PolicyEnforcer()
    enforcer.add_rule("low_trust", "d", "state['trust_score'] < 0.5", action="block")
    config = {"configurable": {"thread_id": "t"}}

    graph = _CheckpointedGraph({"messages": [], "trust_score": 0.9})
    agent = LangGraphAdapter(policy_enforcer=enforcer).wrap_agent(graph)
    resume = Command(resume=True)
    agent.execute(resume, config=config)
    assert graph.inputs == [resume]  # the Command itself reaches the graph
    asyncio.run(agent.aexecute(resume, config=config))

    with pytest.raises(SecurityBlockedError, match="policy"):
        agent.execute(Command(resume=True, update={"trust_score": 0.1}), config=config)
    with pytest.raises(SecurityBlockedError, match="policy"):
        # No config: no thread to read, so the rule cannot pass (fail closed).
        agent.execute(resume)
    with pytest.raises(SecurityBlockedError, match="Input blocked by security"):
        LangGraphAdapter(firewall=_firewall()).wrap_agent(graph).execute(
            Command(resume={"type": "respond", "message": "SECRET-3"}), config=config
        )


class _Chunk:
    def __init__(self, content: str, id: str):
        self.content, self.id = content, id


class _StreamingGraph:
    def __init__(self, items: list[Any]):
        self.items = items

    def invoke(self, payload: Any, **kw: Any) -> Any:  # pragma: no cover - not used
        return {}

    def stream(self, payload: Any, **kw: Any) -> Any:
        yield from self.items

    async def astream(self, payload: Any, **kw: Any) -> Any:
        for item in self.items:
            yield item


def test_stream_catches_a_pattern_split_across_tokens():
    items = [(_Chunk("SEC", "m1"), {}), (_Chunk("RET-7", "m1"), {}), (_Chunk("bye", "m2"), {})]
    agent = LangGraphAdapter(firewall=_firewall()).wrap_agent(_StreamingGraph(items))
    released: list[Any] = []
    with pytest.raises(SecurityBlockedError):
        for item in agent.stream({"messages": []}, stream_mode="messages"):
            released.append(item)
    assert released == []


def test_stream_releases_clean_messages_in_order():
    items = [(_Chunk("he", "m1"), {}), (_Chunk("llo", "m1"), {}), (_Chunk("bye", "m2"), {})]
    agent = LangGraphAdapter(firewall=_firewall()).wrap_agent(_StreamingGraph(items))
    assert list(agent.stream({"messages": []}, stream_mode="messages")) == items


def test_stream_item_shapes_for_modes_subgraphs_and_v2():
    data = (_Chunk("SECRET-1", "m"), {})
    for kwargs, item in (
        ({"stream_mode": ["messages"]}, ("messages", data)),
        ({"stream_mode": "messages", "subgraphs": True}, (("sub",), data)),
        ({"stream_mode": ["messages"], "subgraphs": True}, (("sub",), "messages", data)),
        (
            {"stream_mode": "messages", "version": "v2"},
            {"type": "messages", "ns": (), "data": data},
        ),
        ({"stream_mode": "updates"}, {"model": {"messages": ["SECRET-1"]}}),
    ):
        agent = LangGraphAdapter(firewall=_firewall()).wrap_agent(_StreamingGraph([item]))
        with pytest.raises(SecurityBlockedError):
            list(agent.stream({"messages": []}, **kwargs))


def test_astream_screens_and_input_is_screened_before_the_graph_starts():
    agent = LangGraphAdapter(firewall=_firewall()).wrap_agent(
        _StreamingGraph([{"model": {"messages": ["SECRET-1"]}}])
    )

    async def collect() -> list[Any]:
        return [item async for item in agent.astream({"messages": []}, stream_mode="updates")]

    with pytest.raises(SecurityBlockedError):
        asyncio.run(collect())
    with pytest.raises(SecurityBlockedError):
        agent.astream({"messages": ["SECRET-2"]})
    with pytest.raises(SecurityBlockedError):
        agent.stream({"messages": ["SECRET-2"]})


def test_stream_report_only_releases_everything():
    items = [{"model": {"messages": ["SECRET-1"]}}]
    agent = LangGraphAdapter(firewall=_firewall(), block_on_violation=False).wrap_agent(
        _StreamingGraph(items)
    )
    assert list(agent.stream({"messages": []}, stream_mode="updates")) == items


# -- examples ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES = sorted((_ROOT / "examples" / "langgraph").glob("[0-9][0-9]_*.py"))


@pytest.mark.parametrize("script", _EXAMPLES, ids=lambda p: p.name)
def test_langgraph_example_exits_zero_with_or_without_the_framework(script):
    """Installed, the example runs for real; missing, it says so and exits 0."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_ROOT), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip()
    assert "Traceback" not in result.stderr
