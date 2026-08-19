"""Framework-aware runners: drive any supported agent with plain eval inputs.

:func:`~adapt_agent.optimization.evaluation.resolve_runner` already turns most
framework objects into a ``Callable[[input], output]`` (finding ``run_sync`` /
``invoke`` / ``kickoff`` / ``run`` and materialising async results). Two gaps
remain when running *evals* from a golden dataset of plain strings:

* the produced output is framework-native (an ``AgentRunResult``, a LangGraph
  state mapping, an event stream) rather than the final response text, and
* some frameworks do not accept a plain string input -- a LangGraph
  message-state graph wants ``{"messages": [...]}``, and a Google ADK agent is
  driven through a ``Runner`` with a session and a ``Content`` message.

This module closes both gaps:

* :func:`framework_runner` -- wrap any supported agent as
  ``Callable[[input], text]``: input adaptation (auto-detected for LangGraph),
  sync/async resolution, then output extraction via
  :func:`~adapt_agent.optimization.extractors.extract_output_text`.
* :func:`langgraph_inputs` -- the string -> message-state input adapter.
* :func:`adk_runner` -- drive a Google ADK agent (or prebuilt ``Runner``)
  synchronously with a fresh session per call, returning final response text.

Importing this module never imports an agent framework; ``google.adk`` /
``google.genai`` are imported lazily inside :func:`adk_runner` only when a bare
agent (rather than a prebuilt runner) or a plain-string message needs them.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from typing import Any

from adapt_agent.optimization.evaluation import resolve_runner
from adapt_agent.optimization.extractors import extract_output_text

#: Sentinel: pick the input adapter automatically from the detected framework.
AUTO = "auto"


def framework_runner(
    agent: Any,
    *,
    input_adapter: Callable[[Any], Any] | str | None = AUTO,
    output_extractor: Callable[[Any], Any] | None = extract_output_text,
) -> Callable[[Any], Any]:
    """Wrap ``agent`` as a plain ``input -> final output`` callable for evals.

    Resolution builds on :func:`~adapt_agent.optimization.evaluation.resolve_runner`
    (so anything with ``run_sync`` / ``invoke`` / ``kickoff`` / ``run`` /
    ``execute``, or a bare callable, works) and adds the two eval-specific
    steps: dataset inputs are adapted to the framework's native shape, and
    framework-native results are unwrapped to the final response text.

    Args:
        agent: A framework agent (LangGraph compiled graph, Microsoft Agent
            Framework ``ChatAgent``, Pydantic AI ``Agent``, CrewAI ``Crew``,
            ...), a governed adapter, or any callable. For Google ADK prefer
            :func:`adk_runner`, which handles sessions and message packing.
        input_adapter: ``"auto"`` (default) applies :func:`langgraph_inputs`
            when the agent is detected as a LangGraph graph and nothing
            otherwise. Pass a callable for custom adaptation or ``None`` to
            disable.
        output_extractor: Applied to each raw result;
            :func:`~adapt_agent.optimization.extractors.extract_output_text`
            by default. Pass ``None`` to keep framework-native outputs.

    Returns:
        A synchronous ``Callable[[input], output]`` (async agents are awaited /
        drained internally) suitable for
        :meth:`EvaluationHarness.evaluate <adapt_agent.optimization.evaluation.EvaluationHarness.evaluate>`.
    """
    runner = resolve_runner(agent)
    adapter = _resolve_input_adapter(agent, input_adapter)

    def _run(input_data: Any) -> Any:
        if adapter is not None:
            input_data = adapter(input_data)
        output = _materialize(runner(input_data))
        if output_extractor is not None:
            output = output_extractor(output)
        return output

    return _run


def langgraph_inputs(input_data: Any) -> Any:
    """Adapt a plain dataset input into LangGraph message-state form.

    A string becomes ``{"messages": [{"role": "user", "content": <s>}]}`` (the
    ``MessagesState`` convention used by ``create_react_agent`` and most
    graphs); a list is taken as a ready message list and wrapped the same way;
    mappings and anything else pass through unchanged (the dataset already
    carries graph-native state).
    """
    if isinstance(input_data, str):
        return {"messages": [{"role": "user", "content": input_data}]}
    if isinstance(input_data, (list, tuple)):
        return {"messages": list(input_data)}
    return input_data


def adk_runner(
    target: Any,
    *,
    app_name: str = "adapt-agent-eval",
    user_id: str = "adapt-eval-user",
    message_factory: Callable[[Any], Any] | None = None,
    output_extractor: Callable[[Any], Any] | None = extract_output_text,
) -> Callable[[Any], Any]:
    """Drive a Google ADK agent synchronously: ``input -> final response text``.

    Google ADK agents are not directly runnable -- they execute inside a
    ``Runner`` that needs an app name, a session, and a ``google.genai``
    ``Content`` message. This helper hides that machinery behind a plain
    callable the :class:`~adapt_agent.optimization.evaluation.EvaluationHarness`
    can drive. Every call creates a **fresh session**, so eval examples stay
    independent instead of leaking conversation history into each other.

    Args:
        target: A prebuilt ADK ``Runner`` (anything with a callable ``run`` and
            a ``session_service``), or a bare ADK agent -- in which case
            ``google.adk`` is imported (lazily, only here) to build an
            ``InMemoryRunner`` around it.
        app_name: Application name used when building an ``InMemoryRunner``
            and when creating sessions (a prebuilt runner's own ``app_name``
            wins for sessions).
        user_id: User id recorded on eval sessions.
        message_factory: Optional ``input -> new_message`` override. By default
            strings are packed into a ``google.genai`` user ``Content`` (values
            that already look like a ``Content`` pass through).
        output_extractor: Applied to the drained event list;
            :func:`~adapt_agent.optimization.extractors.extract_output_text`
            by default (yields the final response text). ``None`` returns the
            raw event list.

    Returns:
        A synchronous ``Callable[[input], output]``.

    Raises:
        ImportError: When ``google-adk`` (or ``google-genai`` for default
            message packing) is needed but not installed.
    """
    runner = target if _is_adk_runner_like(target) else _build_inmemory_runner(target, app_name)
    session_app_name = getattr(runner, "app_name", None) or app_name
    make_message = message_factory if message_factory is not None else _adk_user_message

    def _run(input_data: Any) -> Any:
        session_id = f"adapt-eval-{uuid.uuid4().hex}"
        _ensure_session(runner, session_app_name, user_id, session_id)
        events = runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=make_message(input_data),
        )
        output = _materialize(events)
        if output_extractor is not None:
            output = output_extractor(output)
        return output

    return _run


# -- internals -----------------------------------------------------------------


def _resolve_input_adapter(
    agent: Any, input_adapter: Callable[[Any], Any] | str | None
) -> Callable[[Any], Any] | None:
    if input_adapter is None:
        return None
    if callable(input_adapter):
        return input_adapter
    if input_adapter == AUTO:
        # Local import: introspection loads its (framework-free) sub-modules on
        # first use; keep that off this module's import path.
        from adapt_agent.optimization.introspection import detect

        try:
            detected = detect(agent)
        except Exception:
            detected = None
        return langgraph_inputs if detected == "langgraph" else None
    raise ValueError(f"input_adapter must be a callable, None, or 'auto', got {input_adapter!r}")


def _materialize(value: Any) -> Any:
    """Await coroutines and drain (a)sync generators into concrete values."""
    # Imported lazily; reuses the adapters' resolver. No framework SDK involved.
    from adapt_agent.adapters._governed import _resolve_result

    return _resolve_result(value)


def _is_adk_runner_like(obj: Any) -> bool:
    """A Google ADK ``Runner``: callable ``run`` plus a ``session_service``."""
    return callable(getattr(obj, "run", None)) and hasattr(obj, "session_service")


def _build_inmemory_runner(agent: Any, app_name: str) -> Any:
    try:
        from google.adk.runners import InMemoryRunner
    except ImportError as exc:  # pragma: no cover - exercised without the SDK
        raise ImportError(
            "adk_runner needs google-adk to build a Runner around a bare agent: "
            "pip install 'adapt-agent[google-adk]'. Alternatively pass a "
            "prebuilt google.adk Runner as the target."
        ) from exc
    return InMemoryRunner(agent=agent, app_name=app_name)


def _ensure_session(runner: Any, app_name: str, user_id: str, session_id: str) -> None:
    """Create the eval session on the runner's session service (sync or async)."""
    service = getattr(runner, "session_service", None)
    if service is None:
        return
    create = getattr(service, "create_session_sync", None)
    if not callable(create):
        create = getattr(service, "create_session", None)
    if not callable(create):
        return
    _materialize(create(app_name=app_name, user_id=user_id, session_id=session_id))


def _adk_user_message(input_data: Any) -> Any:
    """Pack a dataset input into a ``google.genai`` user ``Content`` message."""
    if not isinstance(input_data, str):
        # Already Content-shaped (or an explicit framework payload): pass through.
        if hasattr(input_data, "parts") or isinstance(input_data, Mapping):
            return input_data
        input_data = str(input_data)
    try:
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - exercised without the SDK
        raise ImportError(
            "adk_runner needs google-genai to pack a string into a Content "
            "message: pip install 'adapt-agent[google-adk]'. Alternatively pass "
            "message_factory= to build messages yourself."
        ) from exc
    return types.Content(role="user", parts=[types.Part(text=input_data)])


__all__ = ["AUTO", "adk_runner", "framework_runner", "langgraph_inputs"]
