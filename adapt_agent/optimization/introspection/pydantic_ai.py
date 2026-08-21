"""Introspection for Pydantic AI ``Agent`` objects.

`Pydantic AI <https://ai.pydantic.dev>`_ centres on an ``Agent`` that exposes a
callable ``run_sync`` and holds its configuration on attributes: the system
prompts in a private ``_system_prompts`` tuple, a ``model`` (a string identifier
or a model object), an optional ``model_settings`` dict (temperature / top_p /
max_tokens), and a function-tool registry (``_function_tools`` or ``tools``).

This module turns such an object into a flat list of tunable
:class:`~adapt_agent.optimization.parameters.Parameter` objects bound to live
getters/setters. It duck-types every attribute access and never imports
``pydantic_ai``; importing this module with nothing installed succeeds.
"""

from __future__ import annotations

from typing import Any

from adapt_agent.optimization.introspection import (
    bind_attr,
    bind_mapping_key,
    register,
    tool_subset_candidates,
)
from adapt_agent.optimization.parameters import Parameter, ParameterKind


def _predicate(obj: Any) -> bool:
    """Return ``True`` for objects shaped like a Pydantic AI ``Agent``.

    A match needs a callable ``run_sync`` and a ``model`` attribute, plus either
    the private ``_system_prompts`` tuple or a public ``system_prompt``. We
    explicitly reject objects that carry multi-agent / handoff markers (or a
    ``chat_client``, which a Microsoft Agent Framework ``ChatAgent`` carries) so
    we do not steal objects belonging to other frameworks' introspectors.
    """
    try:
        if not callable(getattr(obj, "run_sync", None)):
            return False
        if not hasattr(obj, "model"):
            return False
        if not (hasattr(obj, "_system_prompts") or hasattr(obj, "system_prompt")):
            return False
        for foreign in ("handoffs", "sub_agents", "agents", "allowed_tools", "chat_client"):
            if hasattr(obj, foreign):
                return False
        return True
    except Exception:
        return False


def _component_name(obj: Any) -> str:
    """Return the agent's ``name`` if it is a non-empty string, else ``"agent"``."""
    name = getattr(obj, "name", None)
    if isinstance(name, str) and name:
        return name
    return "agent"


def _sequence_of_strings(value: Any) -> bool:
    """Whether ``value`` is a non-empty sequence of strings."""
    if not isinstance(value, (list, tuple)) or not value:
        return False
    return all(isinstance(item, str) for item in value)


def _sequence_prompt_param(obj: Any, attr: str, wrap: Any, component: str) -> Parameter:
    """Bind a prompt held as a list/tuple of strings on ``obj.<attr>``."""

    def _getter() -> Any:
        prompts = getattr(obj, attr, None)
        if prompts is None:
            return None
        return "\n".join(prompts)

    def _setter(value: Any) -> None:
        setattr(obj, attr, wrap([value]))

    return Parameter(
        name="agent.system_prompt",
        kind=ParameterKind.PROMPT,
        value=_getter(),
        getter=_getter,
        setter=_setter,
        component=component,
        metadata={"source": f"attr:{attr}"},
    )


def _system_prompt_param(obj: Any, component: str) -> Parameter | None:
    """Build a prompt Parameter for the system prompt(s).

    Pydantic AI has *two* prompt fields and an agent uses whichever one it was
    built with: ``Agent(system_prompt=...)`` fills ``_system_prompts`` (a tuple)
    and ``Agent(instructions=...)`` -- the modern spelling -- fills
    ``_instructions`` (a list), leaving the other empty. Binding
    ``_system_prompts`` unconditionally is therefore worse than finding nothing
    on an ``instructions=`` agent: the optimizer gets a prompt knob whose value
    is ``''`` and whose writes land in a field the agent never reads, so a sweep
    runs to completion and reports improvements that cannot exist. The populated
    field wins; ``_system_prompts`` breaks the tie when both are empty.

    Neither is bindable through the generic helpers -- a single element of a
    tuple has no working setter -- so the getter/setter are hand-built: reads
    join the sequence, writes replace it wholesale.
    """
    for attr, wrap in (("_system_prompts", tuple), ("_instructions", list)):
        if _sequence_of_strings(getattr(obj, attr, None)):
            return _sequence_prompt_param(obj, attr, wrap, component)
    for attr, wrap in (("_system_prompts", tuple), ("_instructions", list)):
        if hasattr(obj, attr):
            return _sequence_prompt_param(obj, attr, wrap, component)
    if isinstance(getattr(obj, "system_prompt", None), str):
        return bind_attr(
            obj,
            "system_prompt",
            "agent.system_prompt",
            ParameterKind.PROMPT,
            component=component,
        )
    return None


def _model_params(obj: Any, component: str) -> list[Parameter]:
    """Build MODEL (and nested HYPERPARAM) parameters for ``obj.model``.

    A string model is bound directly; an object model is introspected for its
    ``model_name`` / ``model`` identifier and any ``temperature`` / ``top_p`` /
    ``max_tokens`` hyperparameters.
    """
    model = getattr(obj, "model", None)
    if isinstance(model, str):
        param = bind_attr(obj, "model", "agent.model", ParameterKind.MODEL, component=component)
        return [param] if param is not None else []

    if model is None:
        return []

    params: list[Parameter | None] = []
    if isinstance(getattr(model, "model_name", None), str):
        params.append(
            bind_attr(model, "model_name", "agent.model", ParameterKind.MODEL, component=component)
        )
    elif isinstance(getattr(model, "model", None), str):
        params.append(
            bind_attr(model, "model", "agent.model", ParameterKind.MODEL, component=component)
        )
    params.append(
        bind_attr(
            model,
            "temperature",
            "agent.temperature",
            ParameterKind.HYPERPARAM,
            component=component,
            bounds=(0.0, 2.0),
        )
    )
    params.append(
        bind_attr(
            model,
            "top_p",
            "agent.top_p",
            ParameterKind.HYPERPARAM,
            component=component,
            bounds=(0.0, 1.0),
        )
    )
    params.append(
        bind_attr(
            model,
            "max_tokens",
            "agent.max_tokens",
            ParameterKind.HYPERPARAM,
            component=component,
            bounds=(1, 32000),
        )
    )
    return [p for p in params if p is not None]


def _model_settings_params(obj: Any, component: str) -> list[Parameter]:
    """Bind HYPERPARAM parameters held in the ``model_settings`` dict."""
    specs = (
        ("temperature", "agent.temperature", (0.0, 2.0)),
        ("top_p", "agent.top_p", (0.0, 1.0)),
        ("max_tokens", "agent.max_tokens", (1, 32000)),
    )
    params: list[Parameter | None] = [
        bind_mapping_key(
            obj,
            "model_settings",
            key,
            name,
            ParameterKind.HYPERPARAM,
            component=component,
            bounds=bounds,
        )
        for key, name, bounds in specs
    ]
    return [p for p in params if p is not None]


def _tool_param(obj: Any, component: str) -> Parameter | None:
    """Expose a TOOL parameter bound to the agent's function-tool registry.

    When the registry is a list/tuple of two or more tools the parameter is made
    *optimizable* via drop-one ablation ``candidates`` so the optimizer can search
    tool subsets.
    """
    for attr in ("_function_tools", "tools"):
        if hasattr(obj, attr):
            current = getattr(obj, attr, None)
            return bind_attr(
                obj,
                attr,
                "agent.tools",
                ParameterKind.TOOL,
                component=component,
                candidates=(
                    tool_subset_candidates(current) if isinstance(current, (list, tuple)) else None
                ),
            )
    return None


def _introspect(obj: Any) -> list[Parameter]:
    """Walk a Pydantic AI ``Agent`` and return its tunable parameters.

    Best-effort and total: any unexpected shape yields ``[]`` rather than an
    error, and absent attributes simply contribute no parameter.
    """
    try:
        component = _component_name(obj)
        params: list[Parameter] = []

        prompt = _system_prompt_param(obj, component)
        if prompt is not None:
            params.append(prompt)

        params.extend(_model_params(obj, component))

        # ``model_settings`` only supplies hyperparameters not already exposed by
        # an object model above.
        existing = {p.name for p in params}
        for param in _model_settings_params(obj, component):
            if param.name not in existing:
                params.append(param)
                existing.add(param.name)

        tool = _tool_param(obj, component)
        if tool is not None:
            params.append(tool)

        return params
    except Exception:
        return []


register("pydantic_ai", _predicate, _introspect)


__all__ = ["_predicate", "_introspect"]
