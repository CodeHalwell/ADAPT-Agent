"""Tests for the Claude Agent SDK optimization introspector.

Most tests here use plain fake objects mimicking the ``ClaudeAgentOptions``
attribute shape. ``test_real_claude_agent_options_is_introspectable`` is the
exception: it imports the real ``claude_agent_sdk`` when installed, because a
fake can only ever encode what the SDK looked like when the fake was written --
which is exactly how the ``agents`` field slipped past this suite.
"""

import importlib.util
import types

import pytest

from adapt_agent.optimization.introspection import detect, introspect
from adapt_agent.optimization.introspection.claude_agent import _predicate
from adapt_agent.optimization.parameters import ParameterKind


def _make_options() -> types.SimpleNamespace:
    """A ``ClaudeAgentOptions``-shaped object with a string ``system_prompt``."""
    return types.SimpleNamespace(
        system_prompt="You are a helpful assistant.",
        model="claude-sonnet-4-5",
        allowed_tools=["Read", "Grep"],
        max_turns=20,
    )


def test_detect_routes_to_claude_agent() -> None:
    assert detect(_make_options()) == "claude_agent"


def test_predicate_false_for_unrelated_object() -> None:
    assert _predicate(object()) is False
    assert detect(object()) is None


def test_predicate_false_for_other_framework() -> None:
    """A *populated* foreign marker still rejects the object.

    Populated, not merely present: see
    `test_an_unset_foreign_attribute_does_not_block_detection`.
    """
    foreign = types.SimpleNamespace(system_prompt="x", allowed_tools=[], handoffs=["billing_agent"])
    assert _predicate(foreign) is False
    assert detect(foreign) is None


def test_an_unset_foreign_attribute_does_not_block_detection() -> None:
    """`ClaudeAgentOptions` declares some foreign names itself, unset.

    The SDK grew an `agents` field (default None) for subagent definitions. A
    bare `hasattr` veto then rejected every real options object -- `detect`
    returned None, `introspect` returned [], and the whole framework looked like
    it had no tunable knobs. An attribute that exists but holds nothing is not
    evidence of another framework.

    The falsy markers here are ones still listed in `_FOREIGN_ATTRS`. `agents`
    was dropped from that list entirely, so asserting on it alone would pass
    against a `hasattr` veto and leave the populated-value rule unguarded.
    """
    options = types.SimpleNamespace(
        system_prompt="You are helpful.",
        allowed_tools=["Read"],
        agents=None,
        handoffs=[],
        sub_agents=None,
    )
    assert _predicate(options) is True
    assert detect(options) == "claude_agent"
    assert any(p.kind is ParameterKind.PROMPT for p in introspect(options))


@pytest.mark.skipif(
    importlib.util.find_spec("claude_agent_sdk") is None,
    reason="claude-agent-sdk is not installed",
)
def test_real_claude_agent_options_is_introspectable() -> None:
    """Introspect an actual SDK options object, not a fake of one.

    Every fake here is a guess at the SDK's shape, and a stale guess fails
    silently rather than loudly. This is the only test that catches the next
    field the SDK adds.
    """
    import claude_agent_sdk  # type: ignore[import-not-found]

    options = claude_agent_sdk.ClaudeAgentOptions(
        system_prompt="You are helpful.", allowed_tools=["Read", "Grep"]
    )
    assert detect(options) == "claude_agent"

    params = {p.name: p for p in introspect(options)}
    prompt = params["agent.system_prompt"]
    assert prompt.kind is ParameterKind.PROMPT
    assert prompt.value == "You are helpful."
    prompt.write("Be terse.")
    assert options.system_prompt == "Be terse."


def test_introspect_param_names_and_kinds() -> None:
    params = introspect(_make_options())
    by_name = {p.name: p for p in params}

    assert by_name["agent.system_prompt"].kind is ParameterKind.PROMPT
    assert by_name["agent.model"].kind is ParameterKind.MODEL
    assert by_name["agent.allowed_tools"].kind is ParameterKind.TOOL
    assert by_name["agent.max_turns"].kind is ParameterKind.HYPERPARAM
    assert by_name["agent.max_turns"].bounds == (1, 100)


def test_system_prompt_setter_round_trip() -> None:
    opts = _make_options()
    params = introspect(opts)
    by_name = {p.name: p for p in params}

    param = by_name["agent.system_prompt"]
    param.write("You are a meticulous assistant.")
    assert opts.system_prompt == "You are a meticulous assistant."
    assert param.read() == "You are a meticulous assistant."


def test_dict_form_system_prompt_binds_append() -> None:
    opts = types.SimpleNamespace(
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": "Always cite sources.",
        },
        model="claude-opus-4-8",
        allowed_tools=[],
        max_turns=5,
    )
    params = introspect(opts)
    by_name = {p.name: p for p in params}

    param = by_name["agent.system_prompt"]
    assert param.kind is ParameterKind.PROMPT
    assert param.read() == "Always cite sources."

    # Setter mutates only the "append" key, preserving the preset wrapper.
    param.write("Always cite primary sources.")
    assert opts.system_prompt["append"] == "Always cite primary sources."
    assert opts.system_prompt["preset"] == "claude_code"


def test_dict_form_without_append_exposes_no_prompt() -> None:
    opts = types.SimpleNamespace(
        system_prompt={"type": "preset", "preset": "claude_code"},
        model="claude-opus-4-8",
        allowed_tools=[],
        max_turns=5,
    )
    by_name = {p.name: p for p in introspect(opts)}
    assert "agent.system_prompt" not in by_name
    # Other params still discovered.
    assert by_name["agent.model"].kind is ParameterKind.MODEL


def test_optional_attrs_present_when_set() -> None:
    opts = types.SimpleNamespace(
        system_prompt="hi",
        model="claude-sonnet-4-5",
        allowed_tools=["Read"],
        disallowed_tools=["Bash"],
        max_turns=10,
        permission_mode="acceptEdits",
    )
    by_name = {p.name: p for p in introspect(opts)}
    assert by_name["agent.disallowed_tools"].kind is ParameterKind.TOOL
    assert by_name["agent.permission_mode"].kind is ParameterKind.ROUTING


def test_permission_mode_has_real_candidates() -> None:
    opts = types.SimpleNamespace(
        system_prompt="hi",
        model="claude-sonnet-4-5",
        allowed_tools=["Read"],
        max_turns=10,
        permission_mode="default",
    )
    by_name = {p.name: p for p in introspect(opts)}
    pm = by_name["agent.permission_mode"]
    assert pm.candidates == ["default", "acceptEdits", "plan", "bypassPermissions"]
    # With candidates the knob is actually optimizable/searchable.
    assert pm.optimizable
    assert pm.enumerate_candidates() == ["default", "acceptEdits", "plan", "bypassPermissions"]


def test_allowed_tools_optimizable_with_drop_one_candidates() -> None:
    # Two-plus allowed tools -> drop-one ablation candidates.
    opts = _make_options()  # allowed_tools=["Read", "Grep"]
    by_name = {p.name: p for p in introspect(opts)}
    tools = by_name["agent.allowed_tools"]

    assert tools.kind is ParameterKind.TOOL
    assert tools.candidates is not None
    assert tools.candidates[0] == ["Read", "Grep"]
    assert ["Grep"] in tools.candidates
    assert ["Read"] in tools.candidates
    assert tools.optimizable


def test_allowed_tools_single_has_no_candidates() -> None:
    opts = types.SimpleNamespace(
        system_prompt="hi",
        model="claude-sonnet-4-5",
        allowed_tools=["Read"],
        max_turns=10,
    )
    by_name = {p.name: p for p in introspect(opts)}
    assert by_name["agent.allowed_tools"].candidates is None


def test_configured_subagents_do_not_block_detection() -> None:
    """`agents` is the Claude SDK's own subagent field, not a foreign marker.

    Requiring a *populated* foreign value fixed the unset case and left this
    one: configure any subagent and `detect()` went back to None, taking every
    tunable prompt/model/tool setting with it.
    """
    options = types.SimpleNamespace(
        system_prompt="You are helpful.",
        allowed_tools=["Read"],
        agents={"researcher": types.SimpleNamespace(description="d", prompt="p")},
    )
    assert _predicate(options) is True
    assert detect(options) == "claude_agent"
    assert any(p.kind is ParameterKind.PROMPT for p in introspect(options))


@pytest.mark.skipif(
    importlib.util.find_spec("claude_agent_sdk") is None,
    reason="claude-agent-sdk is not installed",
)
def test_a_real_options_object_with_subagents_is_introspectable() -> None:
    import claude_agent_sdk  # type: ignore[import-not-found]

    options = claude_agent_sdk.ClaudeAgentOptions(
        system_prompt="You are helpful.", allowed_tools=["Read"]
    )
    options.agents = {"researcher": {"description": "d", "prompt": "p"}}
    assert detect(options) == "claude_agent"
    assert any(p.kind is ParameterKind.PROMPT for p in introspect(options))
