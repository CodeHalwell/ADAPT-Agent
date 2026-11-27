"""Introspection for the `Claude Agent SDK <https://github.com/anthropics/claude-agent-sdk-python>`_.

The Claude Agent SDK is driven by a ``query(prompt=..., options=...)`` call,
where ``options`` is a ``ClaudeAgentOptions`` object holding the agent's
configuration: its ``system_prompt``, ``model``, ``allowed_tools`` /
``disallowed_tools``, ``max_turns`` budget, ``effort`` and ``permission_mode``.
Security-relevant settings are never searched: ``permission_mode`` and
``disallowed_tools`` are reported read-only, and ``allowed_tools`` can only be
narrowed. This module
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

#: Effort levels used when the installed SDK's own type cannot be read (the
#: object being introspected is then a stand-in for one).
_FALLBACK_EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"]


def _effort_levels() -> list[str]:
    """The ``effort`` values the installed SDK accepts, read from its own type.

    Reading the ``EffortLevel`` literal keeps the search space equal to what the
    SDK will actually take, whatever release is installed, instead of a list
    that rots as levels are added.
    """
    try:
        import typing

        from claude_agent_sdk.types import EffortLevel

        levels = [level for level in typing.get_args(EffortLevel) if isinstance(level, str)]
    except Exception:
        levels = []
    return levels or list(_FALLBACK_EFFORT_LEVELS)


def _read_only(
    obj: Any, attr: str, name: str, kind: ParameterKind, component: str, note: str
) -> Parameter | None:
    """Bind ``obj.<attr>`` for reporting only: no setter, never searched.

    Used for settings whose only possible *changes* are security decisions. The
    value shows up in reports and snapshots, but no optimizer can write it.
    """
    if not hasattr(obj, attr):
        return None
    return Parameter(
        name=name,
        kind=kind,
        value=getattr(obj, attr, None),
        getter=lambda: getattr(obj, attr, None),
        component=component,
        mutable=False,
        metadata={"source": f"attr:{attr}", "read_only_reason": note},
    )


def _tighten_only(param: Parameter | None) -> Parameter | None:
    """Restrict a tool-list parameter so a write can only *remove* entries.

    ``allowed_tools`` is the list of tools the agent may use without asking, so
    adding a tool widens what runs unattended. An optimizer scored on task
    quality has every reason to prefer more tools; letting it grow the list
    would trade safety for score without anyone deciding to. The setter
    therefore accepts only subsets of the list as introspected, and raises on
    anything else.
    """
    if param is None or param.setter is None:
        return param
    baseline = frozenset(param.read() or ())
    write = param.setter

    def _setter(value: Any) -> None:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{param.name} must be a list of tool names")
        added = set(value) - baseline
        if added:
            raise ValueError(
                f"{param.name} may only be narrowed: {sorted(added)!r} would grant "
                "tools the agent was configured without"
            )
        write(list(value))

    param.setter = _setter
    return param


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
    selection is a real search space (``tools`` can only be narrowed, like
    ``allowed_tools``).
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
            tightened = _tighten_only(param) if field == "tools" else param
            if tightened is not None:
                params.append(tightened)
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

        # ``effort`` replaced the flat thinking budget: current SDKs document
        # ``max_thinking_tokens`` as deprecated and treat it as an on/off
        # switch, so a numeric search over it explores nothing. It is still
        # offered on releases that have no ``effort`` field.
        has_effort = hasattr(obj, "effort")

        candidates = [
            bind_attr(obj, "model", f"{component}.model", ParameterKind.MODEL, component=component),
            _tighten_only(
                bind_attr(
                    obj,
                    "allowed_tools",
                    f"{component}.allowed_tools",
                    ParameterKind.TOOL,
                    component=component,
                    candidates=allowed_candidates or None,
                )
            ),
            _read_only(
                obj,
                "disallowed_tools",
                f"{component}.disallowed_tools",
                ParameterKind.TOOL,
                component,
                "the deny list is a security control; shrinking it is a policy decision",
            ),
            bind_attr(
                obj,
                "max_turns",
                f"{component}.max_turns",
                ParameterKind.HYPERPARAM,
                component=component,
                bounds=(1, 100),
            ),
            bind_attr(
                obj,
                "effort",
                f"{component}.effort",
                ParameterKind.HYPERPARAM,
                component=component,
                candidates=_effort_levels() if has_effort else None,
            ),
            (
                None
                if has_effort
                else bind_attr(
                    obj,
                    "max_thinking_tokens",
                    f"{component}.max_thinking_tokens",
                    ParameterKind.HYPERPARAM,
                    component=component,
                    bounds=(1024, 32000),
                )
            ),
            # ``permission_mode`` decides which tool calls run without a human.
            # The SDK documents its values but gives no ordering from which
            # "tighter" and "looser" can be read with confidence (``plan``,
            # ``dontAsk`` and ``auto`` are not points on one scale), and
            # ``bypassPermissions`` switches checking off entirely. A score-driven
            # search must not choose among them, so the mode is reported, never
            # written.
            _read_only(
                obj,
                "permission_mode",
                f"{component}.permission_mode",
                ParameterKind.ROUTING,
                component,
                "permission modes cannot be ordered from the SDK's documentation; "
                "choosing one is a security decision, not a tunable",
            ),
        ]
        params.extend(p for p in candidates if p is not None)

        _introspect_subagents(obj, params)
    except Exception:
        return params
    return params


register("claude_agent", _predicate, _introspect)


__all__ = ["_predicate", "_introspect"]
