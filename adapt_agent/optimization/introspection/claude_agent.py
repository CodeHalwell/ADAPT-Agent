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

Subagents are covered too. ``options.agents`` maps a subagent name to a
definition -- an ``AgentDefinition`` dataclass or an equivalent plain mapping --
whose ``prompt`` and ``description`` are the prompts that steer the subagent and
the delegation to it, alongside its own ``model``, ``tools`` and ``skills``.
Each definition is introspected under its (slugged) subagent name, so a
multi-agent Claude setup exposes every specialist's knobs, not just the
orchestrator's.

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
#: Markers of *other* frameworks. `agents` is deliberately absent: it is the
#: Claude SDK's own subagent-definition field, so vetoing on it rejected the
#: very object this introspector exists for. Requiring a populated value only
#: narrowed that to options with subagents configured -- still a real, and
#: more advanced, Claude setup.
_FOREIGN_ATTRS = ("handoffs", "sub_agents", "kickoff", "instructions")

#: The discrete set of Claude Agent SDK ``permission_mode`` values. Exposing
#: these as candidates makes the knob a real (searchable) routing parameter.
_PERMISSION_MODES = ["default", "acceptEdits", "plan", "bypassPermissions"]


def _predicate(obj: Any) -> bool:
    """Return ``True`` when ``obj`` looks like a ``ClaudeAgentOptions`` object.

    Carrying both ``system_prompt`` and ``allowed_tools`` is the discriminator,
    and it is unique among the supported frameworks. Objects with a *populated*
    ``handoffs``/``sub_agents``/``kickoff``/``instructions`` are rejected as
    belonging elsewhere -- populated, not merely present, since an unset
    attribute is not evidence of anything. The body never raises.
    """
    try:
        if not (hasattr(obj, "system_prompt") and hasattr(obj, "allowed_tools")):
            return False
        for foreign in _FOREIGN_ATTRS:
            # Presence alone is not evidence of another framework: an
            # attribute that exists but holds nothing says nothing. (The field
            # that made this matter, `agents`, is no longer listed at all -- see
            # `_FOREIGN_ATTRS`.)
            if getattr(obj, foreign, None):
                return False
        return True
    except Exception:
        return False


def _slug(text: Any) -> str | None:
    """Slugify a subagent name (lowercase, spaces -> underscores). ``None`` if empty."""
    if not isinstance(text, str):
        return None
    slug = text.strip().lower().replace(" ", "_")
    return slug or None


def _definition_params(definition: Any, component: str) -> list[Parameter]:
    """Bind one subagent definition's tunable fields.

    A definition is an ``AgentDefinition`` dataclass in the real SDK, but the
    SDK also accepts plain mappings, so both shapes are handled: attributes for
    objects, keys for mappings. ``prompt`` and ``description`` are both PROMPT
    knobs -- the prompt steers what the subagent does, the description steers
    when the orchestrator delegates to it -- and ``tools``/``skills`` get the
    same drop-one ablation candidates as every other tool list so subagent tool
    selection is a real search space.
    """
    is_mapping = isinstance(definition, dict)

    def _value(field: str) -> Any:
        return definition.get(field) if is_mapping else getattr(definition, field, None)

    def _bind(
        field: str, name: str, kind: ParameterKind, candidates: list[Any] | None = None
    ) -> Parameter | None:
        if is_mapping:
            return bind_item(
                definition, field, name, kind, component=component, candidates=candidates
            )
        return bind_attr(definition, field, name, kind, component=component, candidates=candidates)

    params: list[Parameter] = []
    for field in ("prompt", "description"):
        if isinstance(_value(field), str):
            param = _bind(field, f"{component}.{field}", ParameterKind.PROMPT)
            if param is not None:
                params.append(param)
    if isinstance(_value("model"), str):
        param = _bind("model", f"{component}.model", ParameterKind.MODEL)
        if param is not None:
            params.append(param)
    for field, kind in (("tools", ParameterKind.TOOL), ("skills", ParameterKind.SKILL)):
        current = _value(field)
        if isinstance(current, (list, tuple)):
            param = _bind(
                field, f"{component}.{field}", kind, tool_subset_candidates(current) or None
            )
            if param is not None:
                params.append(param)
    return params


def _introspect_subagents(obj: Any, params: list[Parameter]) -> None:
    """Introspect the subagent definitions in ``options.agents``.

    Each definition is namespaced under its slugged dict key. Names already
    emitted are skipped rather than duplicated -- a subagent literally named
    ``"agent"`` would otherwise collide with the root component's own knobs and
    make ``SearchSpace.add`` raise downstream.
    """
    definitions = getattr(obj, "agents", None)
    if not isinstance(definitions, dict):
        return
    seen = {p.name for p in params}
    for index, (key, definition) in enumerate(definitions.items()):
        component = _slug(key) or f"subagent_{index}"
        for param in _definition_params(definition, component):
            if param.name in seen:
                continue
            seen.add(param.name)
            params.append(param)


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
            # The extended-thinking budget (present on SDK versions that carry
            # it as a flat option). The floor matches the API's minimum budget.
            bind_attr(
                obj,
                "max_thinking_tokens",
                f"{component}.max_thinking_tokens",
                ParameterKind.HYPERPARAM,
                component=component,
                bounds=(1024, 32000),
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

        _introspect_subagents(obj, params)
    except Exception:
        return params
    return params


register("claude_agent", _predicate, _introspect)


__all__ = ["_predicate", "_introspect"]
