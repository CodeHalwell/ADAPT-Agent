"""Introspection for the OpenAI Agents SDK.

The `OpenAI Agents SDK <https://openai.github.io/openai-agents-python>`_ models an
agent as an ``Agent`` object carrying ``instructions`` (a system prompt, either a
string or a callable), a ``handoff_description`` (the text a routing agent reads
when deciding to delegate to this agent), a ``model`` (a string identifier or a
``Model`` object), ``model_settings`` (temperature / top_p / max_tokens), a list
of ``tools``, and a list of ``handoffs`` to other agents or ``Handoff`` wrappers.
The handoffs list is what makes an orchestrator route work between sub-agents, so
it is also the signal that distinguishes this framework from the other supported
ones.

This module turns a live ``Agent`` into tunable
:class:`~adapt_agent.optimization.parameters.Parameter` objects bound to the agent
in place, recursing into handed-off sub-agents so multi-agent topologies are fully
covered. A handoff entry may also be a ``Handoff`` *wrapper* (built with
``handoff(...)``) rather than a bare agent; the wrapped agent lives inside an
``on_invoke_handoff`` closure and cannot be reached structurally, but the
wrapper's ``tool_description`` -- the text the routing LLM actually reads -- is
bound as a PROMPT so the routing surface stays tunable. It never imports
``agents``; everything is duck-typed via ``getattr`` / ``hasattr`` so importing
this module is safe with nothing installed.
"""

from __future__ import annotations

from typing import Any

from adapt_agent.optimization.introspection import (
    bind_attr,
    register,
    tool_subset_candidates,
)
from adapt_agent.optimization.parameters import Parameter, ParameterKind


def _slugify(name: Any) -> str:
    """Slugify an agent name into a component identifier (fallback ``"agent"``)."""
    if not isinstance(name, str):
        return "agent"
    slug = name.strip().lower().replace(" ", "_")
    return slug or "agent"


#: Markers that belong to *other* frameworks. An object carrying any of these is
#: not an OpenAI Agents ``Agent`` -- ``chat_client`` is the Microsoft Agent
#: Framework ``ChatAgent`` signal, ``kickoff`` is CrewAI, ``sub_agents`` is Google
#: ADK. Rejecting them stops this introspector (registered *before* Microsoft)
#: from hijacking a Microsoft ChatAgent / magentic orchestrator.
_FOREIGN_MARKERS = ("chat_client", "kickoff", "sub_agents")


def _predicate(obj: Any) -> bool:
    """Return ``True`` for an OpenAI Agents ``Agent``-shaped object.

    An ``Agent`` has ``instructions``, ``tools`` and -- distinctively -- a
    ``handoffs`` *list/tuple*, which separates it from the other frameworks.
    Objects carrying foreign markers (``chat_client``/``kickoff``/``sub_agents``)
    are rejected outright so a Microsoft ``ChatAgent`` or magentic orchestrator is
    never misrouted here. Wrapped so it never raises.
    """
    try:
        for foreign in _FOREIGN_MARKERS:
            if hasattr(obj, foreign):
                return False
        if not (hasattr(obj, "instructions") and hasattr(obj, "tools")):
            return False
        # Require handoffs to actually be a list/tuple, not merely present: a
        # bare/None ``handoffs`` attribute is too weak a signal and would let this
        # introspector claim objects from other frameworks.
        return isinstance(getattr(obj, "handoffs", None), (list, tuple))
    except Exception:
        return False


def _looks_like_agent(obj: Any) -> bool:
    """Return ``True`` if a handoff target is itself an introspectable agent."""
    try:
        return hasattr(obj, "instructions")
    except Exception:
        return False


def _looks_like_handoff(obj: Any) -> bool:
    """Return ``True`` if a handoff entry is a ``Handoff`` wrapper object.

    A ``Handoff`` carries ``tool_description`` and ``agent_name`` but no
    ``instructions`` (the wrapped agent is captured inside its
    ``on_invoke_handoff`` closure, not held as an attribute).
    """
    try:
        return (
            hasattr(obj, "tool_description")
            and hasattr(obj, "agent_name")
            and not hasattr(obj, "instructions")
        )
    except Exception:
        return False


def _introspect_model(obj: Any, component: str, params: list[Parameter]) -> None:
    """Append model-related parameters for ``obj.model`` (string or object)."""
    model = getattr(obj, "model", None)
    if isinstance(model, str):
        param = bind_attr(
            obj, "model", f"{component}.model", ParameterKind.MODEL, component=component
        )
        if param is not None:
            params.append(param)
        return
    if model is None:
        return
    # ``model`` is a Model object: expose its identifier attribute, whichever exists.
    for attr in ("model", "model_name"):
        if hasattr(model, attr):
            param = bind_attr(
                model, attr, f"{component}.model", ParameterKind.MODEL, component=component
            )
            if param is not None:
                params.append(param)
            break


def _introspect_model_settings(obj: Any, component: str, params: list[Parameter]) -> None:
    """Append hyperparameters from ``obj.model_settings`` (temperature/top_p/...)."""
    settings = getattr(obj, "model_settings", None)
    if settings is None:
        return
    temperature = bind_attr(
        settings,
        "temperature",
        f"{component}.temperature",
        ParameterKind.HYPERPARAM,
        component=component,
        bounds=(0.0, 2.0),
    )
    if temperature is not None:
        params.append(temperature)
    top_p = bind_attr(
        settings,
        "top_p",
        f"{component}.top_p",
        ParameterKind.HYPERPARAM,
        component=component,
        bounds=(0.0, 1.0),
    )
    if top_p is not None:
        params.append(top_p)
    # Bounded like every other framework's max_tokens: without bounds the knob
    # has no search space at all -- ``enumerate_candidates`` collapses to the
    # current value and the numeric proposer skips it -- so it silently sat in
    # every report as a parameter the optimizer could never actually move.
    max_tokens = bind_attr(
        settings,
        "max_tokens",
        f"{component}.max_tokens",
        ParameterKind.HYPERPARAM,
        component=component,
        bounds=(1, 32000),
    )
    if max_tokens is not None:
        params.append(max_tokens)


def _introspect_agent(obj: Any, params: list[Parameter], visited: set[int]) -> None:
    """Walk a single ``Agent`` (and its handoffs), appending parameters.

    ``visited`` tracks object ids to guard against cyclic handoff graphs.
    """
    if id(obj) in visited:
        return
    visited.add(id(obj))

    component = _slugify(getattr(obj, "name", None))

    # instructions -> PROMPT, but only when it is a plain string (it may be a
    # callable that computes instructions dynamically, which we cannot tune).
    if isinstance(getattr(obj, "instructions", None), str):
        prompt = bind_attr(
            obj,
            "instructions",
            f"{component}.instructions",
            ParameterKind.PROMPT,
            component=component,
        )
        if prompt is not None:
            params.append(prompt)

    # handoff_description -> PROMPT. This is the text a *routing* agent reads
    # when deciding whether to delegate here, so in a multi-agent topology it
    # steers routing quality as directly as the instructions steer answers.
    if isinstance(getattr(obj, "handoff_description", None), str):
        description = bind_attr(
            obj,
            "handoff_description",
            f"{component}.handoff_description",
            ParameterKind.PROMPT,
            component=component,
        )
        if description is not None:
            params.append(description)

    _introspect_model(obj, component, params)
    _introspect_model_settings(obj, component, params)

    # tools -> TOOL allow-list (bind the list attribute itself). When there are
    # >=2 tools, attach drop-one ablation candidates so tool selection is a real
    # search space rather than a single fixed value.
    if hasattr(obj, "tools"):
        current_tools = getattr(obj, "tools", None)
        candidates = (
            tool_subset_candidates(current_tools)
            if isinstance(current_tools, (list, tuple))
            else None
        )
        tools = bind_attr(
            obj,
            "tools",
            f"{component}.tools",
            ParameterKind.TOOL,
            component=component,
            candidates=candidates or None,
        )
        if tools is not None:
            params.append(tools)

    # handoffs -> ROUTING (bind the list attribute holding the topology).
    if hasattr(obj, "handoffs"):
        routing = bind_attr(
            obj,
            "handoffs",
            f"{component}.handoffs",
            ParameterKind.ROUTING,
            component=component,
        )
        if routing is not None:
            params.append(routing)

    # Recurse into handed-off sub-agents so orchestrator+subagent topologies are
    # covered; each child's params are namespaced under its own component name.
    # A ``Handoff`` wrapper cannot be recursed into (its agent is closed over,
    # not held), but its ``tool_description`` is still bound: that text is what
    # the routing LLM reads to pick the handoff, so it is the tunable part.
    handoffs = getattr(obj, "handoffs", None)
    if isinstance(handoffs, (list, tuple)):
        for child in handoffs:
            if _looks_like_agent(child):
                _introspect_agent(child, params, visited)
            elif _looks_like_handoff(child):
                _introspect_handoff(child, params, visited)


def _introspect_handoff(wrapper: Any, params: list[Parameter], visited: set[int]) -> None:
    """Bind a ``Handoff`` wrapper's ``tool_description`` as a PROMPT parameter.

    The component is ``"<agent_name>_handoff"`` so it can never collide with the
    wrapped agent's own component if that agent is also reachable bare elsewhere
    in the topology. ``visited`` keeps a wrapper that appears on several agents
    from binding (and colliding with) itself twice.
    """
    if id(wrapper) in visited:
        return
    visited.add(id(wrapper))
    if not isinstance(getattr(wrapper, "tool_description", None), str):
        return
    component = f"{_slugify(getattr(wrapper, 'agent_name', None))}_handoff"
    description = bind_attr(
        wrapper,
        "tool_description",
        f"{component}.tool_description",
        ParameterKind.PROMPT,
        component=component,
    )
    if description is not None:
        params.append(description)


def _introspect(obj: Any) -> list[Parameter]:
    """Return tunable parameters for an OpenAI Agents ``Agent`` (best effort)."""
    params: list[Parameter] = []
    try:
        _introspect_agent(obj, params, set())
    except Exception:
        # Best-effort: never raise out of an introspector.
        return params
    return params


register("openai_agents", _predicate, _introspect)


__all__ = ["_predicate", "_introspect"]
