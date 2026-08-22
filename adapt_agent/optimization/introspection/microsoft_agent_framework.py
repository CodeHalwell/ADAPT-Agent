"""Introspection for `Microsoft Agent Framework <https://github.com/microsoft/agent-framework>`_.

Microsoft Agent Framework is Microsoft's unified successor to Semantic Kernel and
AutoGen. Its runnable object is an ``Agent`` (historically ``ChatAgent``) with a
client object holding the model, an optional ``.name``, and a callable ``run``.
This module turns a live agent into a flat list of tunable
:class:`~adapt_agent.optimization.parameters.Parameter` objects without ever
importing ``agent_framework``: everything is discovered by duck-typing.

**Two layouts, both supported.** The SDK moved its configuration around, and an
introspector that only knows the old shape silently finds nothing -- which looks
exactly like "this agent has no tunable knobs" rather than like a bug:

======================  ==========================  ================================
what                    older layout                current layout (1.x)
======================  ==========================  ================================
client                  ``.chat_client``            ``.client``
system prompt           ``.instructions``           ``default_options["instructions"]``
tools                   ``.tools``                  ``default_options["tools"]``
sampling settings       client attrs / agent attrs  ``default_options[...]``
======================  ==========================  ================================

Every lookup tries the attribute first and falls back to the ``default_options``
mapping, so both layouts yield the same parameter names.

A client attribute is what distinguishes this agent from a Pydantic AI ``Agent``
-- which *also* carries ``.instructions`` and a callable ``run``, but no client
-- so the predicate requires one. It additionally rejects objects carrying
``handoffs``/``sub_agents``/``agents``/``kickoff`` (OpenAI Agents, Google ADK,
CrewAI) to avoid false positives.

Importing this module registers the introspector under the
``"microsoft_agent_framework"`` key.
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

#: client attributes that may hold the model identifier, in priority order.
_MODEL_ATTRS = ("model_id", "model", "ai_model_id", "deployment_name")

#: Attributes that may hold the chat client, newest layout last.
_CLIENT_ATTRS = ("chat_client", "client")

#: Mapping attribute holding per-agent defaults in current releases.
_OPTIONS_ATTR = "default_options"


def _client_of(obj: Any) -> Any:
    """Return the agent's chat client under whichever attribute holds it."""
    for attr in _CLIENT_ATTRS:
        client = getattr(obj, attr, None)
        if client is not None:
            return client
    return None


def _options_of(obj: Any) -> dict[str, Any] | None:
    """Return the ``default_options`` mapping when the agent carries one."""
    options = getattr(obj, _OPTIONS_ATTR, None)
    return options if isinstance(options, dict) else None


def _predicate(obj: Any) -> bool:
    """Return ``True`` when ``obj`` looks like a Microsoft Agent Framework agent.

    Requires a client (``.chat_client`` or ``.client``), a callable ``run``, and
    somewhere a system prompt *could* live -- an ``instructions`` attribute, or a
    ``default_options`` mapping.

    Note the mapping is not required to contain an ``instructions`` key. An
    agent constructed without instructions is still a Microsoft agent with a
    tunable model and sampling settings, and refusing to claim it would leave
    those undiscoverable rather than merely leave the prompt undiscoverable. The
    client check below, not this one, is what keeps other frameworks out.

    The client requirement is load-bearing, not incidental: a Pydantic AI
    ``Agent`` also has ``.instructions`` and a callable ``run``, and carries none
    of the foreign markers below, so dropping the client check would make this
    predicate swallow it. Objects carrying ``handoffs``/``sub_agents``/
    ``agents``/``kickoff`` (OpenAI Agents, Google ADK, CrewAI) are rejected
    outright.
    """
    try:
        for foreign in ("handoffs", "sub_agents", "agents", "kickoff"):
            if hasattr(obj, foreign):
                return False
        if _client_of(obj) is None:
            return False
        if not hasattr(obj, "instructions") and _options_of(obj) is None:
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


#: Sampling knobs and their bounds, shared by the attribute and mapping binders.
_HYPERPARAMS: tuple[tuple[str, tuple[float, float]], ...] = (
    ("temperature", (0.0, 2.0)),
    ("top_p", (0.0, 1.0)),
    ("max_tokens", (1, 32000)),
)


def _introspect_hyperparams(source: Any, component: str) -> list[Parameter]:
    """Introspect temperature/top_p/max_tokens as attributes of ``source``."""
    candidates = [
        bind_attr(
            source,
            attr,
            f"{component}.{attr}",
            ParameterKind.HYPERPARAM,
            component=component,
            bounds=bounds,
        )
        for attr, bounds in _HYPERPARAMS
    ]
    return [p for p in candidates if p is not None]


def _introspect_option_hyperparams(obj: Any, component: str) -> list[Parameter]:
    """Introspect the same sampling knobs as ``default_options`` keys.

    Current releases keep sampling settings in the options mapping rather than
    as attributes, so an attribute-only walk finds nothing to tune.
    """
    candidates = [
        bind_mapping_key(
            obj,
            _OPTIONS_ATTR,
            key,
            f"{component}.{key}",
            ParameterKind.HYPERPARAM,
            component=component,
            bounds=bounds,
        )
        for key, bounds in _HYPERPARAMS
    ]
    return [p for p in candidates if p is not None]


def _bind_tool_like(obj: Any, attr: str, component: str, kind: ParameterKind) -> Parameter | None:
    """Bind a ``tools``/``skills`` list/tuple attribute as an optimizable parameter.

    When the attribute holds a list/tuple with >=2 entries, attach drop-one
    ablation candidates (via :func:`tool_subset_candidates`) so the optimizer can
    actually search tool/skill selection. With <2 entries the parameter is still
    bound (for visibility) but carries no extra candidates.
    """
    current = getattr(obj, attr, None)
    if isinstance(current, (list, tuple)):
        return bind_attr(
            obj,
            attr,
            f"{component}.{attr}",
            kind,
            component=component,
            candidates=tool_subset_candidates(current) or None,
        )

    # Current releases keep tools in ``default_options`` rather than on the agent.
    options = _options_of(obj)
    if options is None or not isinstance(options.get(attr), (list, tuple)):
        return None
    return bind_mapping_key(
        obj,
        _OPTIONS_ATTR,
        attr,
        f"{component}.{attr}",
        kind,
        component=component,
        candidates=tool_subset_candidates(options[attr]) or None,
    )


def _introspect(obj: Any) -> list[Parameter]:
    """Walk a Microsoft ``ChatAgent`` and return its tunable parameters (best-effort)."""
    params: list[Parameter] = []
    try:
        component = _slug(getattr(obj, "name", None)) or "agent"

        # The system prompt is an attribute in older releases and a
        # ``default_options`` key in current ones. Miss the second and the
        # headline promise -- point it at your agent, it finds the knobs --
        # quietly yields no PROMPT parameter at all.
        instructions = bind_attr(
            obj,
            "instructions",
            f"{component}.instructions",
            ParameterKind.PROMPT,
            component=component,
        ) or bind_mapping_key(
            obj,
            _OPTIONS_ATTR,
            "instructions",
            f"{component}.instructions",
            ParameterKind.PROMPT,
            component=component,
        )
        if instructions is not None:
            params.append(instructions)

        client = _client_of(obj)
        if client is not None:
            for attr in _MODEL_ATTRS:
                model = bind_attr(
                    client,
                    attr,
                    f"{component}.model",
                    ParameterKind.MODEL,
                    component=component,
                )
                if model is not None:
                    params.append(model)
                    break
            params.extend(_introspect_hyperparams(client, component))

        # Sampling settings may live on the agent itself, or in its options
        # mapping, rather than on the client. First source found wins per knob.
        seen = {p.name for p in params}
        for param in (
            *_introspect_hyperparams(obj, component),
            *_introspect_option_hyperparams(obj, component),
        ):
            if param.name not in seen:
                params.append(param)
                seen.add(param.name)

        tool_param = _bind_tool_like(obj, "tools", component, ParameterKind.TOOL)
        if tool_param is not None:
            params.append(tool_param)

        # ``skills`` (Semantic-Kernel-style named skills/plugins) are optimized
        # the same way as tools, under ParameterKind.SKILL.
        skill_param = _bind_tool_like(obj, "skills", component, ParameterKind.SKILL)
        if skill_param is not None:
            params.append(skill_param)
    except Exception:
        return params
    return params


register("microsoft_agent_framework", _predicate, _introspect)


__all__ = ["_predicate", "_introspect"]
