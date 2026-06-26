"""Tests for the Claude Agent SDK optimization introspector.

These tests use plain fake objects mimicking the ``ClaudeAgentOptions``
attribute shape; the real ``claude_agent_sdk`` package is never imported.
"""

import types

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
    # Other frameworks carry handoffs/agents/instructions/etc.
    foreign = types.SimpleNamespace(system_prompt="x", allowed_tools=[], handoffs=[])
    assert _predicate(foreign) is False
    assert detect(foreign) is None


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
