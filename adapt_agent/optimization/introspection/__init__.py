"""Framework introspection: turn a live agent object into tunable parameters.

This package is how ADAPT-Agent "goes deep" into each agent framework. Given a
framework object -- a LangGraph compiled graph, a CrewAI ``Crew``, a Pydantic AI
``Agent``, an OpenAI Agents ``Agent``, a Microsoft Agent Framework ``ChatAgent``,
a Google ADK agent, or a Claude Agent SDK options object -- an *introspector*
walks its structure and returns a flat list of
:class:`~adapt_agent.optimization.parameters.Parameter` objects, each bound to a
live getter/setter so optimizers can read and rewrite prompts, models,
hyperparameters, routing knobs, and tool allow-lists *in place*.

Design rules (mirroring the rest of ``adapt_agent``):

* **Structural, never importing the framework.** Introspectors duck-type with
  ``getattr``; importing this package never imports LangGraph/CrewAI/etc.
* **Best-effort and total.** An introspector returns ``[]`` rather than raising
  when it does not recognise an object, so :func:`introspect` can try each
  registered introspector in turn.
* **Lazy registration.** The per-framework modules register themselves on first
  use via :func:`_ensure_loaded`.

Reusable binding helpers (:func:`bind_attr`, :func:`bind_item`,
:func:`bind_mapping_key`) build the getter/setter pairs; framework modules should
use them so behaviour (and read-only fallback) stays consistent.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Optional

from adapt_agent.optimization.parameters import Parameter, ParameterKind

#: An introspector maps a framework object to a list of bound parameters.
Introspector = Callable[[Any], list[Parameter]]

#: Predicate deciding whether an introspector applies to an object.
Predicate = Callable[[Any], bool]

# Each entry is (name, predicate, introspector). Order matters: the first
# matching, non-empty introspector wins in :func:`introspect`.
_REGISTRY: list[tuple[str, Predicate, Introspector]] = []
_REGISTERED_NAMES: set[str] = set()

#: Per-framework modules to import lazily; each registers on import.
_FRAMEWORK_MODULES = (
    "adapt_agent.optimization.introspection.langgraph",
    "adapt_agent.optimization.introspection.crewai",
    "adapt_agent.optimization.introspection.pydantic_ai",
    "adapt_agent.optimization.introspection.openai_agents",
    "adapt_agent.optimization.introspection.microsoft_agent_framework",
    "adapt_agent.optimization.introspection.google_adk",
    "adapt_agent.optimization.introspection.claude_agent",
)
_loaded = False


def register(name: str, predicate: Predicate, introspector: Introspector) -> None:
    """Register a framework introspector.

    Re-registering an existing ``name`` replaces the previous entry (so modules
    are import-idempotent under reload).
    """
    global _REGISTRY
    if name in _REGISTERED_NAMES:
        _REGISTRY = [entry for entry in _REGISTRY if entry[0] != name]
    _REGISTRY.append((name, predicate, introspector))
    _REGISTERED_NAMES.add(name)


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    for module in _FRAMEWORK_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            # A genuinely missing/unimportable optional module is skipped; other
            # exceptions (e.g. a real bug in a first-party introspector) propagate
            # so they are visible rather than silently swallowed.
            continue


def available() -> list[str]:
    """Return the names of all registered framework introspectors."""
    _ensure_loaded()
    return [name for name, _, _ in _REGISTRY]


def detect(obj: Any) -> str | None:
    """Return the name of the first introspector matching ``obj`` (or ``None``)."""
    _ensure_loaded()
    for name, predicate, _ in _REGISTRY:
        try:
            if predicate(obj):
                return name
        except Exception:
            continue
    return None


def introspect(obj: Any, *, component: str | None = None) -> list[Parameter]:
    """Return tunable parameters discovered on ``obj``.

    Tries each registered introspector whose predicate matches and returns the
    first non-empty result. When ``component`` is given, every returned
    parameter's ``name`` is prefixed with ``"<component>."`` (unless it already
    starts with it) and its ``component`` field is set, so parameters from
    several objects can be merged without name clashes.
    """
    _ensure_loaded()
    params: list[Parameter] = []
    for _, predicate, introspector in _REGISTRY:
        try:
            if not predicate(obj):
                continue
            found = introspector(obj)
        except Exception:
            continue
        if found:
            params = found
            break
    if component:
        for p in params:
            if not p.name.startswith(f"{component}."):
                p.name = f"{component}.{p.name}"
            if p.component is None:
                p.component = component
    return params


def introspect_components(components: dict[str, Any]) -> list[Parameter]:
    """Introspect a mapping of ``{component_name: framework_object}``.

    Each object's parameters are namespaced under its component name. Objects
    that no introspector recognises contribute nothing (no error).
    """
    params: list[Parameter] = []
    seen: set[str] = set()
    for name, obj in components.items():
        for param in introspect(obj, component=name):
            if param.name in seen:
                continue
            seen.add(param.name)
            params.append(param)
    return params


# -- binding helpers ----------------------------------------------------------


def bind_attr(
    obj: Any,
    attr: str,
    name: str,
    kind: ParameterKind,
    *,
    component: str | None = None,
    candidates: list[Any] | None = None,
    bounds: tuple[float, float] | None = None,
    step: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> Parameter | None:
    """Build a :class:`Parameter` bound to ``obj.<attr>``.

    Returns ``None`` if the attribute is absent, so introspectors can simply
    filter ``None`` out of their candidate list. The setter writes through
    ``setattr``; if the attribute turns out to be read-only at write time the
    parameter's :meth:`~adapt_agent.optimization.parameters.Parameter.write`
    will surface the error to the optimizer (which skips it).
    """
    if not hasattr(obj, attr):
        return None
    current = getattr(obj, attr, None)

    def _getter() -> Any:
        return getattr(obj, attr, None)

    def _setter(value: Any) -> None:
        setattr(obj, attr, value)

    return Parameter(
        name=name,
        kind=kind,
        value=current,
        candidates=candidates,
        bounds=bounds,
        step=step,
        getter=_getter,
        setter=_setter,
        component=component,
        metadata={"source": f"attr:{attr}", **(metadata or {})},
    )


def bind_item(
    container: Any,
    key: Any,
    name: str,
    kind: ParameterKind,
    *,
    component: str | None = None,
    candidates: list[Any] | None = None,
    bounds: tuple[float, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Parameter | None:
    """Build a :class:`Parameter` bound to ``container[key]`` (dict/list item)."""
    try:
        current = container[key]
    except (KeyError, IndexError, TypeError):
        return None

    def _getter() -> Any:
        try:
            return container[key]
        except (KeyError, IndexError, TypeError):
            return None

    def _setter(value: Any) -> None:
        container[key] = value

    return Parameter(
        name=name,
        kind=kind,
        value=current,
        candidates=candidates,
        bounds=bounds,
        getter=_getter,
        setter=_setter,
        component=component,
        metadata={"source": f"item:{key}", **(metadata or {})},
    )


def bind_mapping_key(
    getter_obj: Any,
    mapping_attr: str,
    key: str,
    name: str,
    kind: ParameterKind,
    *,
    component: str | None = None,
    candidates: list[Any] | None = None,
    bounds: tuple[float, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Parameter | None:
    """Bind to ``getattr(getter_obj, mapping_attr)[key]`` (a dict held on an attr).

    Useful for frameworks that keep config in a ``dict`` attribute (e.g. model
    settings). Returns ``None`` if the mapping or key is missing.
    """
    mapping = getattr(getter_obj, mapping_attr, None)
    if not isinstance(mapping, dict) or key not in mapping:
        return None
    return bind_item(
        mapping,
        key,
        name,
        kind,
        component=component,
        candidates=candidates,
        bounds=bounds,
        metadata={"source": f"{mapping_attr}[{key}]", **(metadata or {})},
    )


__all__ = [
    "Introspector",
    "Predicate",
    "register",
    "available",
    "detect",
    "introspect",
    "introspect_components",
    "bind_attr",
    "bind_item",
    "bind_mapping_key",
]
