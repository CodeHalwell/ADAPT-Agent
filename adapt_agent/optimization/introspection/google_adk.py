"""Introspection for `Google ADK <https://google.github.io/adk-docs>`_ agents.

A Google ADK ``LlmAgent`` (alias ``Agent``) carries a ``name``, an
``instruction`` (the system prompt -- either a string or an instruction-provider
callable), an optional ``global_instruction``, a ``model`` (a string identifier
or a model object), a ``tools`` allow-list, a list of ``sub_agents``, and an
optional ``generate_content_config`` holding generation hyperparameters. This
module turns a live agent into a flat list of tunable
:class:`~adapt_agent.optimization.parameters.Parameter` objects without ever
importing ``google.adk``: everything is discovered by duck-typing with
``getattr``.

For each agent we expose its ``instruction`` and ``global_instruction`` prompts
(only when they are plain strings), its model (a string identifier, or an
introspected model object's ``model``/``model_name``), the
``temperature``/``top_p``/``max_output_tokens`` hyperparameters from its
``generate_content_config``, its ``tools`` allow-list, and its ``sub_agents``
routing list. Nested ``sub_agents`` are introspected recursively and namespaced
under their own component names; an ``id()`` visited set guards against cycles.

Importing this module registers the introspector under the ``"google_adk"`` key.
"""

from __future__ import annotations

from typing import Any

from adapt_agent.optimization.introspection import (
    bind_attr,
    register,
)
from adapt_agent.optimization.parameters import Parameter, ParameterKind


def _predicate(obj: Any) -> bool:
    """Return ``True`` when ``obj`` looks like a Google ADK ``LlmAgent``.

    An ADK agent is distinguished by having both ``sub_agents`` and an
    ``instruction`` attribute. We explicitly reject objects belonging to other
    frameworks (those carrying ``handoffs``/``kickoff``/``allowed_tools``) to
    avoid false positives.
    """
    try:
        if not hasattr(obj, "sub_agents") or not hasattr(obj, "instruction"):
            return False
        for foreign in ("handoffs", "kickoff", "allowed_tools"):
            if hasattr(obj, foreign):
                return False
        return True
    except Exception:
        return False


def _slug(text: Any) -> str | None:
    """Slugify a name string (lowercase, spaces -> underscores). ``None`` if empty."""
    if not isinstance(text, str):
        return None
    slug = text.strip().lower().replace(" ", "_")
    return slug or None


def _introspect_model(model: Any, component: str) -> list[Parameter]:
    """Introspect a model *object* held on an agent's ``model`` attribute."""
    params: list[Parameter] = []
    model_param = bind_attr(
        model, "model", f"{component}.model", ParameterKind.MODEL, component=component
    )
    if model_param is None:
        model_param = bind_attr(
            model, "model_name", f"{component}.model", ParameterKind.MODEL, component=component
        )
    if model_param is not None:
        params.append(model_param)
    return params


def _introspect_generate_config(config: Any, component: str) -> list[Parameter]:
    """Introspect an ADK ``generate_content_config`` object's hyperparameters."""
    candidates = [
        bind_attr(
            config,
            "temperature",
            f"{component}.temperature",
            ParameterKind.HYPERPARAM,
            component=component,
            bounds=(0.0, 2.0),
        ),
        bind_attr(
            config,
            "top_p",
            f"{component}.top_p",
            ParameterKind.HYPERPARAM,
            component=component,
            bounds=(0.0, 1.0),
        ),
        bind_attr(
            config,
            "max_output_tokens",
            f"{component}.max_output_tokens",
            ParameterKind.HYPERPARAM,
            component=component,
        ),
    ]
    return [p for p in candidates if p is not None]


def _introspect_agent(agent: Any, index: int, visited: set[int]) -> list[Parameter]:
    """Introspect a single ADK ``LlmAgent`` and recurse into its ``sub_agents``."""
    if id(agent) in visited:
        return []
    visited.add(id(agent))

    component = _slug(getattr(agent, "name", None)) or "agent"
    params: list[Parameter] = []

    # Prompts: only emit when the value is a plain string (an instruction may be
    # a provider callable, which is not a tunable text knob).
    instruction = getattr(agent, "instruction", None)
    if isinstance(instruction, str):
        prompt = bind_attr(
            agent,
            "instruction",
            f"{component}.instruction",
            ParameterKind.PROMPT,
            component=component,
        )
        if prompt is not None:
            params.append(prompt)

    global_instruction = getattr(agent, "global_instruction", None)
    if isinstance(global_instruction, str):
        global_prompt = bind_attr(
            agent,
            "global_instruction",
            f"{component}.global_instruction",
            ParameterKind.PROMPT,
            component=component,
        )
        if global_prompt is not None:
            params.append(global_prompt)

    # Model: a string identifier, or an object to introspect.
    model = getattr(agent, "model", None)
    if isinstance(model, str):
        model_param = bind_attr(
            agent, "model", f"{component}.model", ParameterKind.MODEL, component=component
        )
        if model_param is not None:
            params.append(model_param)
    elif model is not None:
        params.extend(_introspect_model(model, component))

    # Generation hyperparameters.
    config = getattr(agent, "generate_content_config", None)
    if config is not None:
        params.extend(_introspect_generate_config(config, component))

    # Tools allow-list.
    tools = bind_attr(agent, "tools", f"{component}.tools", ParameterKind.TOOL, component=component)
    if tools is not None:
        params.append(tools)

    # Sub-agent routing list (structural, but exposed for visibility).
    routing = bind_attr(
        agent, "sub_agents", f"{component}.sub_agents", ParameterKind.ROUTING, component=component
    )
    if routing is not None:
        params.append(routing)

    # Recurse into sub-agents, namespacing each under its own component name.
    sub_agents = getattr(agent, "sub_agents", None) or []
    for i, sub in enumerate(sub_agents):
        for param in _introspect_agent(sub, i, visited):
            if not param.name.startswith(f"{component}."):
                param.name = f"{component}.{param.name}"
            params.append(param)

    return params


def _introspect(obj: Any) -> list[Parameter]:
    """Walk a Google ADK agent and return its tunable parameters (best-effort)."""
    try:
        return _introspect_agent(obj, 0, set())
    except Exception:
        return []


register("google_adk", _predicate, _introspect)


__all__ = ["_predicate", "_introspect"]
