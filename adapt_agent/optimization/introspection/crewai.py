"""Introspection for `CrewAI <https://docs.crewai.com>`_ crews.

A CrewAI ``Crew`` orchestrates a list of ``Agent`` objects against a list of
``Task`` objects and is executed with ``crew.kickoff(...)``. This module turns a
live ``Crew`` into a flat list of tunable
:class:`~adapt_agent.optimization.parameters.Parameter` objects without ever
importing ``crewai``: everything is discovered by duck-typing with ``getattr``.

For each agent we expose its ``role``/``goal``/``backstory`` prompts, its model
(a string identifier, or an introspected LLM object's
``model``/``temperature``/``max_tokens``), its ``tools`` allow-list, and its
``max_iter`` hyperparameter. For each task we expose its ``description`` and
``expected_output`` prompts.

Importing this module registers the introspector under the ``"crewai"`` key.
"""

from __future__ import annotations

from typing import Any

from adapt_agent.optimization.introspection import (
    bind_attr,
    register,
)
from adapt_agent.optimization.parameters import Parameter, ParameterKind


def _predicate(obj: Any) -> bool:
    """Return ``True`` when ``obj`` looks like a CrewAI ``Crew``.

    A ``Crew`` has an ``agents`` list/tuple and a callable ``kickoff``. We
    explicitly reject objects belonging to other frameworks (those carrying
    ``handoffs``/``sub_agents``/``system_prompt``) to avoid false positives.
    """
    try:
        agents = getattr(obj, "agents", None)
        if not isinstance(agents, (list, tuple)):
            return False
        if not callable(getattr(obj, "kickoff", None)):
            return False
        for foreign in ("handoffs", "sub_agents", "system_prompt"):
            if hasattr(obj, foreign):
                return False
        return True
    except Exception:
        return False


def _slug(text: Any) -> str | None:
    """Slugify a role string (lowercase, spaces -> underscores). ``None`` if empty."""
    if not isinstance(text, str):
        return None
    slug = text.strip().lower().replace(" ", "_")
    return slug or None


def _introspect_llm(llm: Any, component: str) -> list[Parameter]:
    """Introspect an LLM *object* held on an agent's ``llm`` attribute."""
    params: list[Parameter] = []
    model = bind_attr(llm, "model", f"{component}.model", ParameterKind.MODEL, component=component)
    if model is None:
        model = bind_attr(
            llm, "model_name", f"{component}.model", ParameterKind.MODEL, component=component
        )
    candidates = [
        model,
        bind_attr(
            llm,
            "temperature",
            f"{component}.temperature",
            ParameterKind.HYPERPARAM,
            component=component,
            bounds=(0.0, 2.0),
        ),
        bind_attr(
            llm,
            "max_tokens",
            f"{component}.max_tokens",
            ParameterKind.HYPERPARAM,
            component=component,
        ),
    ]
    params.extend(p for p in candidates if p is not None)
    return params


def _introspect_agent(agent: Any, index: int) -> list[Parameter]:
    """Introspect a single CrewAI ``Agent``."""
    component = _slug(getattr(agent, "role", None)) or f"agent_{index}"
    params: list[Parameter] = []

    prompts = [
        bind_attr(agent, "role", f"{component}.role", ParameterKind.PROMPT, component=component),
        bind_attr(agent, "goal", f"{component}.goal", ParameterKind.PROMPT, component=component),
        bind_attr(
            agent,
            "backstory",
            f"{component}.backstory",
            ParameterKind.PROMPT,
            component=component,
        ),
    ]
    params.extend(p for p in prompts if p is not None)

    llm = getattr(agent, "llm", None)
    if isinstance(llm, str):
        model = bind_attr(
            agent, "llm", f"{component}.model", ParameterKind.MODEL, component=component
        )
        if model is not None:
            params.append(model)
    elif llm is not None:
        params.extend(_introspect_llm(llm, component))

    tools = bind_attr(agent, "tools", f"{component}.tools", ParameterKind.TOOL, component=component)
    if tools is not None:
        params.append(tools)

    max_iter = bind_attr(
        agent,
        "max_iter",
        f"{component}.max_iter",
        ParameterKind.HYPERPARAM,
        component=component,
        bounds=(1, 50),
    )
    if max_iter is not None:
        params.append(max_iter)

    return params


def _introspect_task(task: Any, index: int) -> list[Parameter]:
    """Introspect a single CrewAI ``Task``."""
    component = f"task_{index}"
    candidates = [
        bind_attr(
            task,
            "description",
            f"{component}.description",
            ParameterKind.PROMPT,
            component=component,
        ),
        bind_attr(
            task,
            "expected_output",
            f"{component}.expected_output",
            ParameterKind.PROMPT,
            component=component,
        ),
    ]
    return [p for p in candidates if p is not None]


def _introspect(obj: Any) -> list[Parameter]:
    """Walk a CrewAI ``Crew`` and return its tunable parameters (best-effort)."""
    params: list[Parameter] = []
    try:
        agents = getattr(obj, "agents", None) or []
        for i, agent in enumerate(agents):
            params.extend(_introspect_agent(agent, i))
        tasks = getattr(obj, "tasks", None) or []
        for i, task in enumerate(tasks):
            params.extend(_introspect_task(task, i))
    except Exception:
        return params
    return params


register("crewai", _predicate, _introspect)


__all__ = ["_predicate", "_introspect"]
