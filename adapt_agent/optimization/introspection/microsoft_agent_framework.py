"""Introspection for `Microsoft Agent Framework <https://github.com/microsoft/agent-framework>`_.

Microsoft Agent Framework is Microsoft's unified successor to Semantic Kernel and
AutoGen. Its primary runnable object is a ``ChatAgent`` (also exported as
``Agent``): it carries ``.instructions`` (the system prompt), a ``.chat_client``
object that holds the model (and sometimes sampling settings), an optional
``.name``, and a callable ``run`` coroutine. This module turns a live
``ChatAgent`` into a flat list of tunable
:class:`~adapt_agent.optimization.parameters.Parameter` objects without ever
importing ``agent_framework``: everything is discovered by duck-typing with
``getattr``.

The ``chat_client`` attribute is what distinguishes a ``ChatAgent`` from an
OpenAI Agents ``Agent`` (which uses ``handoffs``); the predicate keys off it and
explicitly rejects objects carrying ``handoffs``/``sub_agents``/``agents``/
``kickoff`` to avoid false positives.

Importing this module registers the introspector under the
``"microsoft_agent_framework"`` key.
"""

from __future__ import annotations

from typing import Any

from adapt_agent.optimization.introspection import (
    bind_attr,
    register,
)
from adapt_agent.optimization.parameters import Parameter, ParameterKind

#: chat_client attributes that may hold the model identifier, in priority order.
_MODEL_ATTRS = ("model_id", "model", "ai_model_id", "deployment_name")


def _predicate(obj: Any) -> bool:
    """Return ``True`` when ``obj`` looks like a Microsoft ``ChatAgent``.

    A ``ChatAgent`` has ``instructions`` and ``chat_client`` attributes plus a
    callable ``run``. The ``chat_client`` attribute is what tells it apart from
    an OpenAI Agents ``Agent``. We explicitly reject objects belonging to other
    frameworks (those carrying ``handoffs``/``sub_agents``/``agents``/
    ``kickoff``) to avoid false positives.
    """
    try:
        for foreign in ("handoffs", "sub_agents", "agents", "kickoff"):
            if hasattr(obj, foreign):
                return False
        if not hasattr(obj, "instructions"):
            return False
        if not hasattr(obj, "chat_client"):
            return False
        if not callable(getattr(obj, "run", None)):
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


def _introspect_hyperparams(source: Any, component: str) -> list[Parameter]:
    """Introspect temperature/top_p/max_tokens on ``source`` (agent or chat_client)."""
    candidates = [
        bind_attr(
            source,
            "temperature",
            f"{component}.temperature",
            ParameterKind.HYPERPARAM,
            component=component,
            bounds=(0.0, 2.0),
        ),
        bind_attr(
            source,
            "top_p",
            f"{component}.top_p",
            ParameterKind.HYPERPARAM,
            component=component,
            bounds=(0.0, 1.0),
        ),
        bind_attr(
            source,
            "max_tokens",
            f"{component}.max_tokens",
            ParameterKind.HYPERPARAM,
            component=component,
        ),
    ]
    return [p for p in candidates if p is not None]


def _introspect(obj: Any) -> list[Parameter]:
    """Walk a Microsoft ``ChatAgent`` and return its tunable parameters (best-effort)."""
    params: list[Parameter] = []
    try:
        component = _slug(getattr(obj, "name", None)) or "agent"

        instructions = bind_attr(
            obj,
            "instructions",
            f"{component}.instructions",
            ParameterKind.PROMPT,
            component=component,
        )
        if instructions is not None:
            params.append(instructions)

        chat_client = getattr(obj, "chat_client", None)
        if chat_client is not None:
            for attr in _MODEL_ATTRS:
                model = bind_attr(
                    chat_client,
                    attr,
                    f"{component}.model",
                    ParameterKind.MODEL,
                    component=component,
                )
                if model is not None:
                    params.append(model)
                    break
            params.extend(_introspect_hyperparams(chat_client, component))

        # Sampling settings may live on the agent itself rather than the client.
        seen = {p.name for p in params}
        for param in _introspect_hyperparams(obj, component):
            if param.name not in seen:
                params.append(param)
                seen.add(param.name)

        tools = getattr(obj, "tools", None)
        if isinstance(tools, (list, tuple)):
            tool_param = bind_attr(
                obj, "tools", f"{component}.tools", ParameterKind.TOOL, component=component
            )
            if tool_param is not None:
                params.append(tool_param)
    except Exception:
        return params
    return params


register("microsoft_agent_framework", _predicate, _introspect)


__all__ = ["_predicate", "_introspect"]
