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
necessarily incomplete for hand-written nodes: prompts buried in *their*
closures cannot be reached structurally. The two prebuilt constructors --
``langgraph.prebuilt.create_react_agent`` and ``langchain.agents.create_agent``
-- are special-cased through their closure cells (see "prebuilt agents" below):
their system prompt and model are tunable, ``create_agent``'s tool list can be
narrowed, and ``create_react_agent``'s tools are reported read-only because
their schemas are baked into the bound model. Users can always declare extra
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
#: Microsoft Agent Framework ``Agent`` (formerly ``ChatAgent``) (``instructions`` + ``chat_client``).
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
    Microsoft Agent Framework ``Agent`` (formerly ``ChatAgent``) (``instructions`` + ``chat_client``)
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


#: Attributes to follow, in order, when digging a user's node object out of the
#: wrappers LangGraph puts around it.
_UNWRAP_ATTRS = ("runnable", "bound", "func", "afunc", "__self__")


def _has_tunable_surface(obj: Any) -> bool:
    """Whether ``obj`` exposes anything this introspector could bind."""
    return any(hasattr(obj, attr) for attr in ("system_prompt", "prompt", "model", "llm", "tools"))


def _runnable_of(node: Any) -> Any:
    """Return the object on a node that actually carries the tunable knobs.

    A compiled graph does not hand back the callable you registered: a
    ``PregelNode`` holds a ``RunnableCallable`` wrapper at ``.bound``, and *your*
    node object sits one further hop down at ``.func`` (or ``.func.__self__``
    when you registered a bound method). Stopping at ``.bound`` -- as this did --
    inspects the wrapper, which exposes no prompt and no model, so every
    realistic graph introspected to zero parameters while still being detected
    as LangGraph. That reads as "this graph has no knobs" rather than as a
    broken walk.

    Each hop is taken only when it leads somewhere with a bindable attribute, so
    unwrapping never overshoots past the object holding the knobs.
    """
    current = node
    for _ in range(len(_UNWRAP_ATTRS)):
        if _has_tunable_surface(current) and current is not node:
            return current
        for attr in _UNWRAP_ATTRS:
            candidate = getattr(current, attr, None)
            if candidate is not None and candidate is not current:
                current = candidate
                break
        else:
            break
    return current


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
    return _model_object_params(model, component)


def _model_object_params(model: Any, component: str) -> list[Parameter]:
    """MODEL and HYPERPARAM parameters for a chat-model *object* (not a string)."""
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
            "top_p",
            f"{component}.top_p",
            ParameterKind.HYPERPARAM,
            component=component,
            bounds=(0.0, 1.0),
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


# -- prebuilt agents ------------------------------------------------------------
#
# ``langgraph.prebuilt.create_react_agent`` and ``langchain.agents.create_agent``
# do not keep their prompt, model or tools on an attribute of anything the
# compiled graph exposes: the node function is a closure over them. Python
# closures hold their variables in *cells*, which can be read and rebound
# (``cell.cell_contents``), and that is the only route to them. It leans on the
# variable names those two constructors use, so every access is checked, a
# missing name simply yields no parameter, and each write is read back and
# undone if it did not take.


def _closure_cells(func: Any) -> dict[str, Any]:
    """``{free variable name: cell}`` for a plain function, else ``{}``."""
    code = getattr(func, "__code__", None)
    closure = getattr(func, "__closure__", None)
    if code is None or not closure:
        return {}
    return dict(zip(code.co_freevars, closure, strict=True))


def _cell_value(cell: Any) -> Any:
    try:
        return cell.cell_contents
    except ValueError:  # an empty cell
        return None


def _is_message(value: Any) -> bool:
    """Whether ``value`` is a chat message with plain-string content."""
    return hasattr(value, "model_copy") and isinstance(getattr(value, "content", None), str)


def _system_message_cell(node: Any) -> Any:
    """The closure cell holding a prebuilt agent's system message, if there is one.

    ``create_agent`` keeps it in ``system_message`` on the model node.
    ``create_react_agent`` builds the prompt step into the model sequence it
    closes over (``static_model``), as ``_system_message`` (a ``prompt=`` string)
    or ``prompt`` (a ``prompt=SystemMessage(...)``).
    """
    func = getattr(getattr(node, "bound", None), "func", None)
    cells = _closure_cells(func)
    cell = cells.get("system_message")
    if cell is not None and _is_message(_cell_value(cell)):
        return cell
    steps = (
        getattr(_cell_value(cells["static_model"]), "steps", None)
        if "static_model" in cells
        else None
    )
    first = steps[0] if isinstance(steps, (list, tuple)) and steps else None
    inner = _closure_cells(getattr(first, "func", None))
    for name in ("_system_message", "prompt"):
        cell = inner.get(name)
        if cell is not None and _is_message(_cell_value(cell)):
            return cell
    return None


def _system_prompt_param(cell: Any, component: str) -> Parameter:
    """A PROMPT parameter that rebinds the message in ``cell``.

    The new message is a copy of the old one with its content replaced, so
    message type, name and metadata survive. Both the sync and async node
    functions close over the *same* cell, so one write serves both.
    """

    def _getter() -> Any:
        value = _cell_value(cell)
        return value.content if _is_message(value) else None

    def _setter(value: Any) -> None:
        if not isinstance(value, str):
            raise TypeError(f"a prompt must be a str, got {type(value).__name__}")
        old = _cell_value(cell)
        if not _is_message(old):
            raise RuntimeError("the agent no longer holds a system message to rewrite")
        cell.cell_contents = old.model_copy(update={"content": value})
        if _getter() != value:
            cell.cell_contents = old
            raise RuntimeError("writing the system prompt did not take effect")

    return Parameter(
        name=f"{component}.system_prompt",
        kind=ParameterKind.PROMPT,
        value=_getter(),
        getter=_getter,
        setter=_setter,
        component=component,
        metadata={"source": "closure:system_message"},
    )


def _prebuilt_model(node: Any) -> Any:
    """The chat model a prebuilt agent's model node calls, if it can be found."""
    func = getattr(getattr(node, "bound", None), "func", None)
    cells = _closure_cells(func)
    if "model" in cells and not isinstance(_cell_value(cells["model"]), str):
        candidate = _cell_value(cells["model"])
    elif "static_model" in cells:
        steps = getattr(_cell_value(cells["static_model"]), "steps", None)
        candidate = steps[-1] if isinstance(steps, (list, tuple)) and steps else None
    else:
        return None
    # A model with tools bound is a ``RunnableBinding`` around the real one.
    for _ in range(4):
        if isinstance(getattr(candidate, "model_name", None), str) or isinstance(
            getattr(candidate, "model", None), str
        ):
            return candidate
        candidate = getattr(candidate, "bound", None)
        if candidate is None:
            return None
    return None


def _prebuilt_node_params(name: Any, node: Any) -> list[Parameter]:
    """Prompt and model knobs of a ``create_agent`` / ``create_react_agent`` model node."""
    component = name if isinstance(name, str) and name else "node"
    params: list[Parameter] = []
    cell = _system_message_cell(node)
    if cell is not None:
        params.append(_system_prompt_param(cell, component))
    model = _prebuilt_model(node)
    if model is not None:
        params.extend(_model_object_params(model, component))
    return params


def _tool_registry(bound: Any) -> dict[str, Any] | None:
    """A ``ToolNode``'s ``name -> tool`` dict, or ``None`` when ``bound`` is not one."""
    registry = getattr(bound, "_tools_by_name", None)
    return registry if isinstance(registry, dict) and registry else None


def _agent_tool_params(nodes: dict[Any, Any]) -> list[Parameter]:
    """The tool knob of a prebuilt agent, spread across two nodes.

    The tool *executor* is a ``ToolNode`` holding a ``name -> tool`` registry;
    what the model is *shown* is a separate list. ``create_agent`` rebuilds the
    model's tool binding from a list (``default_tools``) on every call, so
    narrowing both together is a real, complete write. ``create_react_agent``
    bakes the tool schemas into the bound model once at construction: narrowing
    only the executor would leave the model offering tools whose calls then
    fail, which is worse than not tuning. Its tools are therefore reported
    read-only.
    """
    for name, node in nodes.items():
        bound = getattr(node, "bound", None)
        registry = _tool_registry(bound)
        if registry is None:
            continue
        component = name if isinstance(name, str) and name else "tools"
        baseline = dict(registry)
        injected = getattr(bound, "_injected_args", None)
        baseline_injected = dict(injected) if isinstance(injected, dict) else None
        offered = _offered_tools(nodes)
        tools = list(baseline.values())
        param_name = f"{component}.tools"

        def _getter(registry: dict = registry) -> Any:
            return list(registry.values())

        if offered is None:
            return [
                Parameter(
                    name=param_name,
                    kind=ParameterKind.TOOL,
                    value=tools,
                    getter=_getter,
                    component=component,
                    mutable=False,
                    metadata={
                        "source": "closure:ToolNode",
                        "read_only_reason": "the model's tool schemas are fixed when the agent is built",
                    },
                )
            ]

        def _setter(
            value: Any,
            registry: dict = registry,
            offered: list = offered,
            baseline: dict = baseline,
            injected: Any = injected,
            baseline_injected: Any = baseline_injected,
        ) -> None:
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError("tools must be a non-empty list of the agent's own tools")
            chosen: dict[str, Any] = {}
            for tool in value:
                tool_name = getattr(tool, "name", None)
                if not isinstance(tool_name, str) or baseline.get(tool_name) is not tool:
                    raise ValueError(f"{tool!r} is not one of the agent's tools")
                chosen[tool_name] = tool
            ordered = {n: t for n, t in baseline.items() if n in chosen}
            before = (list(offered), dict(registry))
            # Anything in the offered list that is not a registry tool (a
            # provider built-in, passed as a dict) is not ours to drop.
            offered[:] = [t for t in offered if getattr(t, "name", None) not in baseline] + list(
                ordered.values()
            )
            registry.clear()
            registry.update(ordered)
            if isinstance(injected, dict) and baseline_injected is not None:
                injected.clear()
                injected.update({n: a for n, a in baseline_injected.items() if n in ordered})
            if list(registry) != list(ordered) or any(
                getattr(t, "name", None) in baseline and t not in ordered.values() for t in offered
            ):
                offered[:] = before[0]
                registry.clear()
                registry.update(before[1])
                raise RuntimeError("narrowing the agent's tools did not take effect")

        return [
            Parameter(
                name=param_name,
                kind=ParameterKind.TOOL,
                value=tools,
                candidates=tool_subset_candidates(tools),
                getter=_getter,
                setter=_setter,
                component=component,
                metadata={"source": "closure:default_tools+ToolNode"},
            )
        ]
    return []


def _offered_tools(nodes: dict[Any, Any]) -> list[Any] | None:
    """The list ``create_agent`` binds to the model on each call, if present."""
    for node in nodes.values():
        func = getattr(getattr(node, "bound", None), "func", None)
        cell = _closure_cells(func).get("default_tools")
        value = _cell_value(cell) if cell is not None else None
        if isinstance(value, list):
            return value
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

        seen = {p.name for p in params}
        params.extend(p for p in _prebuilt_node_params(name, node) if p.name not in seen)

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
        seen = {p.name for p in params}
        params.extend(p for p in _agent_tool_params(nodes) if p.name not in seen)
        return params
    except Exception:
        return []


register("langgraph", _predicate, _introspect)


__all__ = ["_predicate", "_introspect"]
