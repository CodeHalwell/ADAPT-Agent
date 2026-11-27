"""SDK-free tests for the CrewAI binding: nothing here imports ``crewai``.

The behaviour against the real framework is in ``tests/test_real_crewai.py``.
These pin the parts that do not need it -- the checkpoint-path mechanism, the
guardrail's signature, the adapter's input shaping and stream draining against
shaped fakes, and the refusals.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import sys

import pytest

from adapt_agent.adapters import CrewAIAdapter
from adapt_agent.core.governance import GovernanceGate, ToolPolicy
from adapt_agent.exceptions import SecurityBlockedError
from adapt_agent.integrations import crewai as integ
from adapt_agent.security import Firewall


def _firewall() -> Firewall:
    fw = Firewall()
    fw.add_blocked_pattern(r"(?i)ignore previous instructions")
    return fw


# -- the guardrail ------------------------------------------------------------


def test_guardrail_has_no_string_return_annotation():
    """``from __future__ import annotations`` made it a str; CrewAI wants the real tuple."""
    sig = inspect.signature(integ.governance_guardrail(GovernanceGate()))
    assert sig.return_annotation is inspect.Signature.empty
    assert len(sig.parameters) == 1


class _TaskOutput:
    """The shape of a CrewAI TaskOutput: an answer plus the transcript behind it."""

    def __init__(self, raw, messages):
        self.raw = raw
        self.messages = messages


def test_guardrail_judges_the_answer_not_the_transcript_of_earlier_attempts():
    guardrail = integ.governance_guardrail(GovernanceGate(firewall=_firewall()))
    transcript = [{"role": "assistant", "content": "ignore previous instructions"}]
    ok, _ = guardrail(_TaskOutput("all clean", transcript))
    assert ok is True
    ok, message = guardrail(_TaskOutput("ignore previous instructions", []))
    assert ok is False and "ADAPT-Agent" in message


# -- the checkpoint-safe callbacks ---------------------------------------------


def _dotted(fn) -> str:
    """What CrewAI's ``callable_to_string`` would write for ``fn``."""
    return f"{fn.__module__}.{fn.__qualname__}"


def test_callbacks_advertise_a_path_that_resolves_back_to_themselves():
    callbacks = integ.governance_callbacks(firewall=_firewall(), agent_id="resume-me")
    for fn in (callbacks["before_kickoff_callbacks"][0], callbacks["after_kickoff_callbacks"][0]):
        module_name, _, attr = _dotted(fn).rpartition(".")
        assert "<locals>" not in attr and "<lambda>" not in attr
        assert getattr(importlib.import_module(module_name), attr) is fn


def test_the_path_is_stable_across_processes_and_distinct_per_agent_id():
    a = integ.governance_callbacks(agent_id="crew-a")["before_kickoff_callbacks"][0]
    b = integ.governance_callbacks(agent_id="crew-b")["before_kickoff_callbacks"][0]
    again = integ.governance_callbacks(agent_id="crew-a")["before_kickoff_callbacks"][0]
    assert a.__qualname__ != b.__qualname__
    assert a.__qualname__ == again.__qualname__
    assert integ.governance_callbacks(agent_id="a-b")["before_kickoff_callbacks"][
        0
    ].__qualname__ != (
        integ.governance_callbacks(agent_id="a_b")["before_kickoff_callbacks"][0].__qualname__
    )


def test_an_unregistered_checkpoint_path_resolves_to_a_callable_that_refuses():
    stand_in = integ.checkpoint_before_never_registered_00000000
    with pytest.raises(SecurityBlockedError) as excinfo:
        stand_in({"topic": "x"})
    assert "governance:gate_not_registered" in excinfo.value.threats
    with pytest.raises(AttributeError):
        integ.not_a_callback  # noqa: B018


def test_callbacks_screen_inputs_and_outputs_and_return_them_unchanged():
    callbacks = integ.governance_callbacks(firewall=_firewall(), agent_id="cb")
    before = callbacks["before_kickoff_callbacks"][0]
    inputs = {"topic": "fine"}
    assert before(inputs) is inputs
    with pytest.raises(SecurityBlockedError):
        before({"topic": "ignore previous instructions"})
    after = callbacks["after_kickoff_callbacks"][0]
    assert after("fine") == "fine"
    with pytest.raises(SecurityBlockedError):
        after("ignore previous instructions")


# -- the hooks object needs CrewAI, and says so ---------------------------------


def test_hooks_refuse_bad_options_without_needing_the_framework():
    with pytest.raises(ValueError, match="on_tool_violation"):
        integ.governance_hooks(on_tool_violation="explode")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="agents"):
        integ.governance_hooks(agents="Researcher")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="approve"):
        integ.governance_hooks(approve="yes")  # type: ignore[arg-type]


def test_installing_without_crewai_names_the_fix(monkeypatch):
    monkeypatch.setitem(sys.modules, "crewai", None)
    monkeypatch.setitem(sys.modules, "crewai.hooks", None)
    with pytest.raises(ImportError, match="crewai"):
        integ.governance_hooks(firewall=_firewall()).install()


# -- the adapter, against shaped fakes ------------------------------------------


class FakeStream:
    """The surface of ``CrewStreamingOutput`` the adapter relies on."""

    def __init__(self, text, *, sync=True, aync=False):
        self._result = type("Out", (), {"raw": text})()
        self._sync_iterator = iter(["a", "b"]) if sync else None
        self._async_iterator = object() if aync else None
        self.is_completed = False
        self.chunks = []
        self.delivered = []

    def get_full_text(self):
        return ""

    def __iter__(self):
        for chunk in self._sync_iterator:
            self.delivered.append(chunk)
            yield chunk
        self.is_completed = True

    async def __aiter__(self):
        for chunk in ("x", "y"):
            self.delivered.append(chunk)
            yield chunk
        self.is_completed = True

    @property
    def result(self):
        return self._result


class StreamingCrew:
    def __init__(self, stream):
        self.stream = stream

    def kickoff(self, inputs=None):
        return self.stream

    async def akickoff(self, inputs=None):
        return self.stream


def test_adapter_drains_a_stream_and_screens_only_the_final_output():
    stream = FakeStream("fine")
    out = CrewAIAdapter(firewall=_firewall()).wrap_agent(StreamingCrew(stream)).execute({})
    assert out["result"].raw == "fine"
    assert stream.delivered == ["a", "b"]  # consumed by the adapter, none left for a caller

    bad = FakeStream("ignore previous instructions")
    with pytest.raises(SecurityBlockedError):
        CrewAIAdapter(firewall=_firewall()).wrap_agent(StreamingCrew(bad)).execute({})


def test_adapter_drains_an_async_only_stream_from_aexecute():
    stream = FakeStream("fine", sync=False, aync=True)
    guarded = CrewAIAdapter(firewall=_firewall()).wrap_agent(StreamingCrew(stream))
    out = asyncio.run(guarded.aexecute({}))
    assert out["result"].raw == "fine"
    assert stream.delivered == ["x", "y"]


class SoloAgent:
    """A standalone CrewAI ``Agent``: ``kickoff(messages)``, no ``inputs``."""

    def __init__(self):
        self.asked = None

    def kickoff(self, messages, response_format=None, input_files=None):
        self.asked = messages
        return type("Out", (), {"raw": "answer"})()


def test_adapter_sends_a_standalone_agent_its_messages_or_prompt():
    agent = SoloAgent()
    adapter = CrewAIAdapter()
    msgs = [{"role": "user", "content": "hello"}]
    adapter.wrap_agent(agent).execute({"messages": msgs})
    assert agent.asked == msgs
    adapter.wrap_agent(agent).execute({"prompt": "name a river"})
    assert agent.asked == "name a river"


class RecordingCrew:
    def __init__(self):
        self.received = None
        self.kwargs = None

    def kickoff(self, inputs=None, input_files=None):
        self.received, self.kwargs = inputs, {"input_files": input_files}
        return "ok"


def test_adapter_still_sends_a_crew_its_inputs_and_forwards_run_kwargs():
    crew = RecordingCrew()
    CrewAIAdapter().wrap_agent(crew).execute(
        {"messages": [{"role": "user", "content": "hi"}], "topic": "AI"}, input_files={"f": 1}
    )
    assert crew.received == {"topic": "AI"}
    assert crew.kwargs == {"input_files": {"f": 1}}


def test_adapter_prefers_akickoff_for_async_runs():
    class Both:
        def kickoff(self, inputs=None):
            return "sync"

        async def kickoff_async(self, inputs=None):
            return "thread-hop"

        async def akickoff(self, inputs=None):
            return "native"

    guarded = CrewAIAdapter().wrap_agent(Both())
    assert asyncio.run(guarded.aexecute({}))["result"] == "native"
    assert guarded.execute({})["result"] == "sync"


def test_adapter_falls_back_to_kickoff_async_on_older_releases():
    class Old:
        def kickoff(self, inputs=None):
            return "sync"

        async def kickoff_async(self, inputs=None):
            return "thread-hop"

    assert asyncio.run(CrewAIAdapter().wrap_agent(Old()).aexecute({}))["result"] == "thread-hop"


def test_adapter_refuses_tool_options_it_cannot_apply():
    for kwargs in (
        {"tool_policy": ToolPolicy(deny=["x"])},
        {"on_tool_violation": "raise"},
        {"approve": True},
    ):
        with pytest.raises(ValueError, match="hooks=True"):
            CrewAIAdapter(**kwargs)
    adapter = CrewAIAdapter(hooks=True, tool_policy=ToolPolicy(deny=["x"]))
    assert adapter.gate.tool_policy is not None


def test_a_wrapped_governance_block_is_surfaced_with_its_threats():
    from adapt_agent.adapters.crewai import _surface

    inner = SecurityBlockedError("blocked", ["tool:denied:x"])
    outer = RuntimeError("framework wrapper")
    outer.__cause__ = inner
    with pytest.raises(SecurityBlockedError) as excinfo:
        _surface(outer)
    assert excinfo.value.threats == ["tool:denied:x"]
    _surface(RuntimeError("unrelated"))  # nothing to unwrap: returns, caller re-raises
