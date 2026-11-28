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
(only when they are plain strings; ``global_instruction`` only on releases
that do not deprecate it), its model (a string identifier, or an
introspected model object's ``model``/``model_name``), the
``temperature``/``top_p``/``top_k``/``max_output_tokens`` hyperparameters from
its ``generate_content_config``, its ``tools`` allow-list, and its
``sub_agents`` routing list. Nested ``sub_agents`` are introspected recursively and namespaced
under their own component names; an ``id()`` visited set guards against cycles.

Workflow agents (``SequentialAgent`` / ``ParallelAgent`` / ``LoopAgent``) and the
graph-based ``Workflow`` have no prompt of their own; they are recognised so
their sub-agents (or the agent nodes of the graph) are introspected.

Importing this module registers the introspector under the ``"google_adk"`` key.
"""

from __future__ import annotations

from typing import Any

from adapt_agent.optimization.introspection import (
    bind_attr,
    register,
    tool_subset_candidates,
)
from adapt_agent.optimization.parameters import Parameter, ParameterKind

#: Class names, defined in ``google.adk``, whose instances are ADK agents. A
#: workflow agent (``SequentialAgent`` / ``ParallelAgent`` / ``LoopAgent``) has
#: sub-agents but no ``instruction``, and a ``Workflow`` is a graph of nodes
#: rather than an agent at all, so neither can be told apart by attribute shape.
_ADK_BASE_NAMES = frozenset({"BaseAgent", "Workflow"})


def _is_adk_class_instance(obj: Any) -> bool:
    """Whether ``obj``'s class derives from an ADK agent / workflow base class.

    Judged by class name and module along the MRO, so the framework is never
    imported; a user subclass of ``BaseAgent`` still matches.
    """
    for cls in type(obj).__mro__:
        if cls.__name__ in _ADK_BASE_NAMES and cls.__module__.startswith("google.adk"):
            return True
    return False


def _predicate(obj: Any) -> bool:
    """Return ``True`` when ``obj`` looks like a Google ADK agent or ``Workflow``.

    An ``LlmAgent`` is distinguished by having both ``sub_agents`` and an
    ``instruction`` attribute; workflow agents and ``Workflow`` by deriving from
    the ADK base classes. We explicitly reject objects belonging to other
    frameworks (those carrying ``handoffs``/``kickoff``/``allowed_tools``) to
    avoid false positives.
    """
    try:
        for foreign in ("handoffs", "kickoff", "allowed_tools"):
            if hasattr(obj, foreign):
                return False
        if _is_adk_class_instance(obj):
            return True
        return hasattr(obj, "sub_agents") and hasattr(obj, "instruction")
    except Exception:
        return False


def _global_instruction_deprecated(agent: Any) -> bool:
    """Whether this ADK release marks ``global_instruction`` as deprecated.

    The framework signals it only by warning when the field is resolved (the
    field itself carries no pydantic ``deprecated`` marker), so the check looks
    for that warning in the resolving method. Tuning a knob the framework is
    about to remove -- and warns about on every run -- wastes the search, so the
    knob is withheld where the marker is present and kept on older releases
    (or when the source cannot be inspected, which reads as "not deprecated").
    """
    import inspect

    resolver = getattr(type(agent), "canonical_global_instruction", None)
    if resolver is None:
        return False
    try:
        return "DeprecationWarning" in inspect.getsource(resolver)
    except (OSError, TypeError):
        return False


def _child_agents(agent: Any) -> list[Any]:
    """The agents nested under ``agent``: ``sub_agents``, or a workflow's graph nodes.

    A ``Workflow`` keeps its steps in ``graph.nodes`` mixed with function nodes
    and the start marker; only the steps that are themselves agents or nested
    workflows carry anything tunable.
    """
    children = list(getattr(agent, "sub_agents", None) or [])
    graph = getattr(agent, "graph", None)
    nodes = getattr(graph, "nodes", None)
    if isinstance(nodes, (list, tuple)):
        children.extend(
            node for node in nodes if node is not agent and _is_adk_class_instance(node)
        )
    return children


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
        # Kept within (1, 40): valid for every Gemini generation, so a gridded
        # candidate never turns into a runtime rejection mid-search.
        bind_attr(
            config,
            "top_k",
            f"{component}.top_k",
            ParameterKind.HYPERPARAM,
            component=component,
            bounds=(1, 40),
        ),
        bind_attr(
            config,
            "max_output_tokens",
            f"{component}.max_output_tokens",
            ParameterKind.HYPERPARAM,
            component=component,
            bounds=(1, 32000),
        ),
    ]
    return [p for p in candidates if p is not None]


def _introspect_agent(agent: Any, index: int, visited: set[int]) -> list[Parameter]:
    """Introspect a single ADK ``LlmAgent`` and recurse into its ``sub_agents``."""
    if id(agent) in visited:
        return []
    visited.add(id(agent))

    # Use the positional index in the fallback so two name-less sub-agents do
    # not both collapse to "agent" and produce colliding parameter names.
    component = _slug(getattr(agent, "name", None)) or f"agent_{index}"
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
    if isinstance(global_instruction, str) and not _global_instruction_deprecated(agent):
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

    # Tools allow-list. When the agent carries two or more tools, expose
    # drop-one ablation candidates so tool *selection* becomes a real search
    # space the optimizer can explore.
    current_tools = getattr(agent, "tools", None)
    tool_candidates = (
        tool_subset_candidates(current_tools) if isinstance(current_tools, (list, tuple)) else None
    )
    tools = bind_attr(
        agent,
        "tools",
        f"{component}.tools",
        ParameterKind.TOOL,
        component=component,
        candidates=tool_candidates or None,
    )
    if tools is not None:
        params.append(tools)

    # Sub-agent routing list (structural, but exposed for visibility).
    routing = bind_attr(
        agent, "sub_agents", f"{component}.sub_agents", ParameterKind.ROUTING, component=component
    )
    if routing is not None:
        params.append(routing)

    # Recurse into sub-agents, namespacing each under this agent's component
    # name. A sub-agent already in ``visited`` (a cycle or a shared/diamond
    # child reachable by two parents) returns ``[]`` and is skipped silently, so
    # we never crash on shared topologies. We additionally de-duplicate emitted
    # names: re-prefixing nested parameters can otherwise collide, and a
    # duplicate name would make ``SearchSpace.add`` raise downstream.
    seen = {p.name for p in params}
    for i, sub in enumerate(_child_agents(agent)):
        for param in _introspect_agent(sub, i, visited):
            if not param.name.startswith(f"{component}."):
                param.name = f"{component}.{param.name}"
            if param.name in seen:
                continue
            seen.add(param.name)
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
