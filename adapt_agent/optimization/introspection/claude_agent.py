"""Introspection for the `Claude Agent SDK <https://github.com/anthropics/claude-agent-sdk-python>`_.

The Claude Agent SDK is driven by a ``query(prompt=..., options=...)`` call,
where ``options`` is a ``ClaudeAgentOptions`` object holding the agent's
configuration: its ``system_prompt``, ``model``, ``allowed_tools`` /
``disallowed_tools``, ``max_turns`` budget, and ``permission_mode``. This module
turns such an options object into a flat list of tunable
:class:`~adapt_agent.optimization.parameters.Parameter` objects without ever
importing ``claude_agent_sdk``: everything is discovered by duck-typing with
``getattr``/``hasattr``.

The ``system_prompt`` may be a plain string, or a preset mapping such as
``{"type": "preset", "preset": "claude_code", "append": "..."}``. When it is a
string we bind it directly; when it is a mapping carrying an ``"append"`` key we
bind that key so the appended instructions stay tunable while the preset itself
is preserved.

Importing this module registers the introspector under the ``"claude_agent"``
key.
"""

from __future__ import annotations

from typing import Any

from adapt_agent.optimization.introspection import (
    bind_attr,
    bind_item,
    register,
    tool_subset_candidates,
)
from adapt_agent.optimization.parameters import Parameter, ParameterKind

#: Attributes that identify *other* frameworks' agent objects; their presence
#: means the object is not a Claude Agent SDK options object.
_FOREIGN_ATTRS = ("handoffs", "sub_agents", "agents", "kickoff", "instructions")

#: The discrete set of Claude Agent SDK ``permission_mode`` values. Exposing
#: these as candidates makes the knob a real (searchable) routing parameter.
_PERMISSION_MODES = ["default", "acceptEdits", "plan", "bypassPermissions"]


def _predicate(obj: Any) -> bool:
    """Return ``True`` when ``obj`` looks like a ``ClaudeAgentOptions`` object.

    Carrying both ``system_prompt`` and ``allowed_tools`` is the discriminator,
    and it is unique among the supported frameworks. Objects with a *populated*
    ``handoffs``/``sub_agents``/``agents``/``kickoff``/``instructions`` are
    rejected as belonging elsewhere -- populated, not merely present, because
    the SDK's own options object declares some of those names itself. The body
    never raises.
    """
    try:
        if not (hasattr(obj, "system_prompt") and hasattr(obj, "allowed_tools")):
            return False
        for foreign in _FOREIGN_ATTRS:
            # Presence alone is not evidence of another framework, and treating
            # it as such is fragile: `ClaudeAgentOptions` grew an `agents` field
            # (defaulting to None) for subagent definitions, and a bare
            # `hasattr` check then rejected every real options object -- so
            # `detect` returned None and no knobs were found at all. Only a
            # populated value counts.
            if getattr(obj, foreign, None):
                return False
        return True
    except Exception:
        return False


def _introspect_system_prompt(obj: Any, component: str) -> list[Parameter]:
    """Bind the ``system_prompt`` (string, or preset mapping with ``"append"``)."""
    prompt = getattr(obj, "system_prompt", None)
    if isinstance(prompt, str):
        param = bind_attr(
            obj,
            "system_prompt",
            f"{component}.system_prompt",
            ParameterKind.PROMPT,
            component=component,
        )
        return [param] if param is not None else []
    if isinstance(prompt, dict) and "append" in prompt:
        param = bind_item(
            prompt,
            "append",
            f"{component}.system_prompt",
            ParameterKind.PROMPT,
            component=component,
        )
        return [param] if param is not None else []
    return []


def _introspect(obj: Any) -> list[Parameter]:
    """Walk a ``ClaudeAgentOptions`` object and return its tunable parameters.

    Best-effort: returns whatever was discovered (possibly ``[]``) on any
    unexpected error rather than raising.
    """
    component = "agent"
    params: list[Parameter] = []
    try:
        params.extend(_introspect_system_prompt(obj, component))

        # Tool allow-list: when two or more tools are present, offer drop-one
        # ablation candidates so the optimizer can actually search which tools
        # to keep instead of treating it as an inert knob.
        allowed = getattr(obj, "allowed_tools", None)
        allowed_candidates = (
            tool_subset_candidates(allowed) if isinstance(allowed, (list, tuple)) else None
        )

        candidates = [
            bind_attr(obj, "model", f"{component}.model", ParameterKind.MODEL, component=component),
            bind_attr(
                obj,
                "allowed_tools",
                f"{component}.allowed_tools",
                ParameterKind.TOOL,
                component=component,
                candidates=allowed_candidates or None,
            ),
            bind_attr(
                obj,
                "disallowed_tools",
                f"{component}.disallowed_tools",
                ParameterKind.TOOL,
                component=component,
            ),
            bind_attr(
                obj,
                "max_turns",
                f"{component}.max_turns",
                ParameterKind.HYPERPARAM,
                component=component,
                bounds=(1, 100),
            ),
            # ``permission_mode`` is a small discrete enum; giving it explicit
            # candidates turns it from a dead knob into a searchable one.
            bind_attr(
                obj,
                "permission_mode",
                f"{component}.permission_mode",
                ParameterKind.ROUTING,
                component=component,
                candidates=list(_PERMISSION_MODES),
            ),
        ]
        params.extend(p for p in candidates if p is not None)
    except Exception:
        return params
    return params


register("claude_agent", _predicate, _introspect)


__all__ = ["_predicate", "_introspect"]
