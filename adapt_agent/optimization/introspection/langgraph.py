"""Introspection for `LangGraph <https://langchain-ai.github.io/langgraph/>`_ graphs.

A *compiled* LangGraph graph exposes a callable ``invoke(state) -> state`` and a
``nodes`` mapping ``{node_name: node}`` (or a callable ``get_graph``). Unlike the
agent-centric frameworks, LangGraph keeps prompts and models *inside* compiled
node functions, so there is no canonical, stable place to read them from. This
module therefore performs a **best-effort structural walk**: for each node it
probes the runnable (``node`` / ``node.runnable`` / ``node.bound``) for a system
prompt, a bound chat model, and a tool list, binding whatever it finds to live
getters/setters.

Because real LangGraph internals vary across versions, the walk is written
defensively -- every attribute access is guarded and the introspector never
raises, returning ``[]`` when nothing recognisable is found. Introspection is
necessarily incomplete: prompts buried in closures cannot be reached
structurally. Users can always declare extra
:class:`~adapt_agent.optimization.parameters.Parameter` objects explicitly on the
``OptimizableAgent`` for prompts this structural walk cannot see.

This module duck-types everything and never imports ``langgraph``; importing it
with nothing installed succeeds. Importing it registers the introspector under
the ``"langgraph"`` key.
"""

from __future__ import annotations

from typing import Any

from adapt_agent.optimization.introspection import (
    bind_attr,
    register,
    tool_subset_candidates,
)
from adapt_agent.optimization.parameters import Parameter, ParameterKind

#: Attribute names that, if present, signal the object belongs to *another*
#: framework's introspector. We refuse to match anything carrying these so we do
#: not steal CrewAI crews, OpenAI/Pydantic agents, Claude SDK options, or a
#: Microsoft Agent Framework ``ChatAgent`` (``instructions`` + ``chat_client``).
_FOREIGN_MARKERS = (
    "agents",
    "handoffs",
    "sub_agents",
    "kickoff",
    "instructions",
    "allowed_tools",
    "chat_client",
)


def _predicate(obj: Any) -> bool:
    """Return ``True`` when ``obj`` looks like a compiled LangGraph graph.

    A compiled graph has a callable ``invoke`` and either a ``nodes`` attribute
    or a callable ``get_graph``. We reject objects carrying other frameworks'
    markers (``agents``/``handoffs``/``sub_agents``/``kickoff``/``instructions``/
    ``allowed_tools``/``chat_client``) to avoid false positives -- in particular a
    Microsoft Agent Framework ``ChatAgent`` (``instructions`` + ``chat_client``)
    must never be hijacked here.
    """
    try:
        if not callable(getattr(obj, "invoke", None)):
            return False
        has_nodes = hasattr(obj, "nodes")
        has_get_graph = callable(getattr(obj, "get_graph", None))
        if not (has_nodes or has_get_graph):
            return False
        for foreign in _FOREIGN_MARKERS:
            if hasattr(obj, foreign):
                return False
        return True
    except Exception:
        return False


def _runnable_of(node: Any) -> Any:
    """Return the runnable carried by a node (``node`` / ``.runnable`` / ``.bound``)."""
    for attr in ("runnable", "bound"):
        candidate = getattr(node, attr, None)
        if candidate is not None:
            return candidate
    return node


def _prompt_param(runnable: Any, component: str) -> Parameter | None:
    """Bind a PROMPT parameter from a ``system_prompt`` or ``prompt`` string attr."""
    for attr in ("system_prompt", "prompt"):
        if isinstance(getattr(runnable, attr, None), str):
            return bind_attr(
                runnable,
                attr,
                f"{component}.{attr}",
                ParameterKind.PROMPT,
                component=component,
            )
    return None


def _model_object(runnable: Any) -> Any:
    """Return a bound chat-model object held on the runnable, if any.

    A model object is one reachable via ``model`` / ``llm`` / ``bound`` that
    itself carries a ``model_name`` or ``model`` identifier. We skip plain string
    values (those are handled directly by the caller).
    """
    for attr in ("model", "llm", "bound"):
        candidate = getattr(runnable, attr, None)
        if candidate is None or isinstance(candidate, str):
            continue
        if isinstance(getattr(candidate, "model_name", None), str) or isinstance(
            getattr(candidate, "model", None), str
        ):
            return candidate
    return None


def _model_params(runnable: Any, component: str) -> list[Parameter]:
    """Build MODEL and nested HYPERPARAM parameters for a bound chat model.

    A string ``model``/``llm`` attribute is bound directly as a MODEL; an object
    model is introspected for its ``model_name``/``model`` identifier (MODEL) and
    its ``temperature``/``max_tokens`` hyperparameters (HYPERPARAM).
    """
    # A direct string model identifier on the runnable.
    for attr in ("model", "llm"):
        if isinstance(getattr(runnable, attr, None), str):
            param = bind_attr(
                runnable, attr, f"{component}.model", ParameterKind.MODEL, component=component
            )
            return [param] if param is not None else []

    model = _model_object(runnable)
    if model is None:
        return []

    params: list[Parameter | None] = []
    if isinstance(getattr(model, "model_name", None), str):
        params.append(
            bind_attr(
                model, "model_name", f"{component}.model", ParameterKind.MODEL, component=component
            )
        )
    elif isinstance(getattr(model, "model", None), str):
        params.append(
            bind_attr(
                model, "model", f"{component}.model", ParameterKind.MODEL, component=component
            )
        )
    params.append(
        bind_attr(
            model,
            "temperature",
            f"{component}.temperature",
            ParameterKind.HYPERPARAM,
            component=component,
            bounds=(0.0, 2.0),
        )
    )
    params.append(
        bind_attr(
            model,
            "max_tokens",
            f"{component}.max_tokens",
            ParameterKind.HYPERPARAM,
            component=component,
            bounds=(1, 32000),
        )
    )
    return [p for p in params if p is not None]


def _tool_param(runnable: Any, component: str) -> Parameter | None:
    """Expose a TOOL parameter bound to the runnable's ``tools`` list, if present.

    When the list holds two or more tools the parameter is made *optimizable* by
    attaching drop-one ablation ``candidates`` so the optimizer can search tool
    subsets.
    """
    current = getattr(runnable, "tools", None)
    if isinstance(current, (list, tuple)):
        return bind_attr(
            runnable,
            "tools",
            f"{component}.tools",
            ParameterKind.TOOL,
            component=component,
            candidates=tool_subset_candidates(current),
        )
    return None


def _introspect_node(name: Any, node: Any) -> list[Parameter]:
    """Introspect a single graph node, namespacing params under the node name."""
    component = name if isinstance(name, str) and name else "node"
    try:
        runnable = _runnable_of(node)
        if runnable is None:
            return []
        params: list[Parameter] = []

        prompt = _prompt_param(runnable, component)
        if prompt is not None:
            params.append(prompt)

        params.extend(_model_params(runnable, component))

        tool = _tool_param(runnable, component)
        if tool is not None:
            params.append(tool)

        return params
    except Exception:
        return []


def _introspect(obj: Any) -> list[Parameter]:
    """Walk a compiled LangGraph graph and return its tunable parameters.

    Best-effort and total: an unexpected shape yields ``[]`` rather than an
    error, and nodes that expose nothing recognisable simply contribute nothing.
    """
    try:
        nodes = getattr(obj, "nodes", None)
        if not isinstance(nodes, dict):
            # Some compiled graphs expose nodes only via ``get_graph().nodes``.
            get_graph = getattr(obj, "get_graph", None)
            if callable(get_graph):
                try:
                    nodes = getattr(get_graph(), "nodes", None)
                except Exception:
                    nodes = None
        if not isinstance(nodes, dict):
            return []
        params: list[Parameter] = []
        for name, node in nodes.items():
            params.extend(_introspect_node(name, node))
        return params
    except Exception:
        return []


register("langgraph", _predicate, _introspect)


__all__ = ["_predicate", "_introspect"]
