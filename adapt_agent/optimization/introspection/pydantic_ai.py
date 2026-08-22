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


#: The two fields a Pydantic AI agent can hold a prompt in, in tie-break
#: order, with the sequence type each one expects back.
_PROMPT_FIELDS = (("_system_prompts", tuple), ("_instructions", list))


def _string_elements(value: Any) -> list[str]:
    """The string members of a prompt sequence, in order.

    Either field may hold callables as well as strings: a *dynamic* instruction
    is a function evaluated per run. Only the strings are static text, and only
    static text is something an optimizer can tune -- so a field is judged by
    the strings in it, not by whether every element is one.

    Requiring *all* elements to be strings read a mixed list as empty, which
    then lost the tie to a genuinely empty field.
    """
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, str)]


def _is_populated(value: Any) -> bool:
    """Whether ``value`` is a non-empty prompt sequence, of anything."""
    return isinstance(value, (list, tuple)) and bool(value)


def _first_static_run(value: Any) -> tuple[int, int]:
    """The half-open index range of the first unbroken run of strings.

    A prompt sequence can hold static text on *both* sides of a callable, and
    a single string cannot represent that without choosing an order. Reading
    all the strings and writing them back as one collapsed the interleaving --
    ``["before", dynamic, "after"]`` became ``["before\nafter", dynamic]``, so
    a plain read-then-write reordered the agent, and a tuned write deleted
    "after" outright.

    So the knob is one *run*: contiguous static text, with everything on the
    far side of a callable left exactly where the user put it. For the common
    shapes -- all strings, strings then callables, callables then strings --
    the run is the whole of the static text and nothing is left out.
    """
    if not isinstance(value, (list, tuple)):
        return (0, 0)
    start = next((i for i, item in enumerate(value) if isinstance(item, str)), None)
    if start is None:
        return (0, 0)
    end = start
    while end < len(value) and isinstance(value[end], str):
        end += 1
    return (start, end)


def _sequence_prompt_param(obj: Any, attr: str, wrap: Any, component: str) -> Parameter:
    """Bind the static prompt text held in the sequence at ``obj.<attr>``.

    Reads join the first run of static text; writes replace exactly that run,
    leaving callables -- and any static text beyond them -- where they were.
    Replacing the sequence wholesale would delete the agent's dynamic
    instructions, joining it wholesale would raise on the first callable, and
    collapsing every string into one would reorder the sequence and drop
    whatever sat past the callable. See :func:`_first_static_run`.

    ``"\n"`` is the separator Pydantic AI itself puts between consecutive
    static instructions, so reading a run and writing it back unchanged leaves
    the rendered prompt byte-identical.
    """

    def _getter() -> Any:
        prompts = getattr(obj, attr, None)
        if prompts is None:
            return None
        start, end = _first_static_run(prompts)
        return "\n".join(prompts[start:end])

    def _setter(value: Any) -> None:
        current = list(getattr(obj, attr, None) or ())
        start, end = _first_static_run(current)
        if start == end:  # no static text yet: the new prompt goes in front
            # ...unless there is no prompt either. Writing back the empty value
            # a callable-only field reads must leave that field untouched, or
            # the round trip an optimizer performs on every sweep grows an
            # empty instruction each time.
            replaced = [value, *current] if value else list(current)
        else:
            replaced = [*current[:start], value, *current[end:]]
        setattr(obj, attr, wrap(replaced))

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
    on an ``instructions=`` agent: the optimizer gets a prompt knob starting at
    ``''`` while the instruction the user actually wrote stays fixed and still
    applies, so every candidate is measured on top of it and the knob never
    tunes the prompt the agent runs on.

    Three tiers, because "populated" turned out to have two meanings. The
    field holding **static text** wins -- judged by the strings in it, so a
    list mixing an instruction with a dynamic callable counts, which it did
    not when every element had to be a string. Failing that, a field populated
    with *only* callables wins, since that is the field the agent was
    configured through and a new prompt belongs beside its siblings. Failing
    both, ``_system_prompts`` breaks the tie and the knob starts empty --
    correct, because then there is no prompt to tune, only one to introduce.

    Neither is bindable through the generic helpers -- a single element of a
    sequence has no working setter -- so the getter/setter are hand-built.
    """
    for populated in (_string_elements, _is_populated):
        for attr, wrap in _PROMPT_FIELDS:
            if populated(getattr(obj, attr, None)):
                return _sequence_prompt_param(obj, attr, wrap, component)
    for attr, wrap in _PROMPT_FIELDS:
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
