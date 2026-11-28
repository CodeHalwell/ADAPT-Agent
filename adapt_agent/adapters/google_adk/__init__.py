"""Google ADK adapter for ADAPT-Agent.

The `Google Agent Development Kit <https://google.github.io/adk-docs>`_ runs an
agent through a ``Runner``: ``runner.run_async(user_id=..., session_id=...,
new_message=...)`` (an async generator of ``Event`` objects) or the sync
``runner.run(...)``. There is no single result object -- you iterate events and
take the one where ``event.is_final_response()`` is true.

The adapter accepts three wrap targets:

* **A ``Runner``** (``InMemoryRunner(app=app)``, or any ``Runner``). Each
  ``execute(payload, user_id=..., session_id=..., **run_kwargs)`` sends the
  payload's prompt as ``new_message`` and returns the run's events. The session
  is created on first use; omit ``session_id`` for a fresh session per call.
  ``run_config=``, ``state_delta=``, ``invocation_id=`` and a ready-made
  ``new_message=`` (a confirmation response, say) pass straight through --
  except that ``new_message`` is content, not configuration: it replaces the
  payload's prompt as what the model reads, so it is screened as input first.
* **An ``App``**, run on an ``InMemoryRunner`` the adapter creates for it once
  (so sessions persist across calls of the same wrapped agent).
* **A callable you write** that performs the run and returns (or yields) the
  events -- the full payload dict and every ``execute`` keyword argument are
  passed to it. Use this for anything the two forms above do not cover.

Whichever you wrap, a :class:`~adapt_agent.exceptions.SecurityBlockedError`
raised by :func:`~adapt_agent.integrations.google_adk.governance_plugin` inside
the run reaches you as itself, with its ``threats``: ADK re-raises plugin
errors as ``RuntimeError("Error in plugin ...")``, and the adapter unwraps them.

The adapter governs the boundary (the payload in, the events out). For tool
calls, sub-agents and streaming, register the governance plugin on the ``App``
as well. Importing this module never imports ``google.adk``.

Example
-------
>>> from google.adk.apps import App                  # doctest: +SKIP
>>> from google.adk.runners import InMemoryRunner     # doctest: +SKIP
>>> from adapt_agent.integrations.google_adk import governance_plugin
>>> app = App(name="app", root_agent=my_agent,        # doctest: +SKIP
...           plugins=[governance_plugin(firewall=fw)])
>>> from adapt_agent.adapters import GoogleADKAdapter
>>> from adapt_agent.security import Firewall
>>> adapter = GoogleADKAdapter(firewall=Firewall())
>>> guarded = adapter.wrap_agent(InMemoryRunner(app=app))  # doctest: +SKIP
>>> guarded.execute({"prompt": "hi"}, user_id="u", session_id="s")  # doctest: +SKIP
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Iterator
from typing import Any

from adapt_agent.adapters._governed import GovernedAdapter, Runner, _extract_prompt
from adapt_agent.exceptions import SecurityBlockedError
from adapt_agent.integrations._common import optional_import
from adapt_agent.integrations.google_adk import reraise_security_blocks

#: ``user_id`` used when ``execute`` is not given one for a ``Runner``/``App``.
DEFAULT_USER_ID = "user"


def _is_runner(obj: Any) -> bool:
    return (
        callable(getattr(obj, "run_async", None))
        and hasattr(obj, "session_service")
        and hasattr(obj, "app_name")
    )


def _is_app(obj: Any) -> bool:
    return (
        not _is_runner(obj)
        and hasattr(obj, "root_agent")
        and isinstance(getattr(obj, "name", None), str)
        and hasattr(obj, "plugins")
    )


class _RunnerCall:
    """Drives an ADK ``Runner`` from an ``execute`` payload.

    The adapter resolves the bound ``__call__``, which ``inspect`` recognises
    as an async-generator function, so ``aexecute`` calls it inline rather
    than in a worker thread -- ``run_async`` belongs in the caller's loop.
    """

    def __init__(self, runner: Any):
        self.runner = runner

    async def __call__(
        self,
        payload: Any,
        *,
        user_id: str = DEFAULT_USER_ID,
        session_id: str | None = None,
        new_message: Any = None,
        **run_kwargs: Any,
    ) -> AsyncIterator[Any]:
        runner = self.runner
        if session_id is None:
            session_id = uuid.uuid4().hex
        service = runner.session_service
        session = await service.get_session(
            app_name=runner.app_name, user_id=user_id, session_id=session_id
        )
        if session is None:
            await service.create_session(
                app_name=runner.app_name, user_id=user_id, session_id=session_id
            )
        if new_message is None:
            genai_types = optional_import("google.genai.types", "google-adk", "GoogleADKAdapter")
            new_message = genai_types.Content(
                role="user", parts=[genai_types.Part(text=_extract_prompt(payload))]
            )
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=new_message, **run_kwargs
        ):
            yield event


async def _unwrap_async(stream: AsyncIterable[Any]) -> AsyncIterator[Any]:
    with reraise_security_blocks():
        async for item in stream:
            yield item


def _unwrap_sync(stream: Iterator[Any]) -> Iterator[Any]:
    with reraise_security_blocks():
        yield from stream


async def _unwrap_awaitable(value: Any) -> Any:
    with reraise_security_blocks():
        result = await value
    return _unwrapping(result)


def _unwrapping(value: Any) -> Any:
    """Wrap a framework result so a plugin's block surfaces as itself.

    The run happens while the adapter *drains* the result, after
    ``_call_runner`` has returned, so catching around the call alone is not
    enough: each lazy shape is wrapped so the unwrap happens where it runs.
    """
    if isinstance(value, AsyncIterable):
        return _unwrap_async(value)
    if inspect.isawaitable(value):
        return _unwrap_awaitable(value)
    if inspect.isgenerator(value):
        return _unwrap_sync(value)
    return value


class GoogleADKAdapter(GovernedAdapter):
    """Adapter for integrating ADAPT-Agent with Google ADK agents.

    Wrap a ``Runner``, an ``App``, or a callable ``run(payload, **kwargs) ->
    events`` that drives a ``Runner`` (the events may be a sync generator, an
    async generator, or a list). See the module docstring for how each form
    uses ``execute``'s keyword arguments, and
    :class:`~adapt_agent.adapters._governed.GovernedAdapter` for the full
    constructor signature.
    """

    framework_name = "Google ADK"
    # An ADK run requires session/user arguments, so binding to a bare
    # ``runner.run`` would mismatch its signature; `_resolve_runner` adapts a
    # Runner or App itself, and anything else must be a callable.
    run_method_names = ()
    operation = "google_adk.run"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # One runner per wrapped App, so the sync and async entry points (and
        # repeated wraps) share its in-memory sessions. Keyed by id with the
        # App held alongside, so the id cannot be recycled while cached.
        self._app_runners: dict[int, tuple[Any, Any]] = {}

    def _runner_for_app(self, app: Any) -> Any:
        cached = self._app_runners.get(id(app))
        if cached is not None and cached[0] is app:
            return cached[1]
        runners = optional_import("google.adk.runners", "google-adk", "GoogleADKAdapter(App)")
        runner = runners.InMemoryRunner(app=app)
        self._app_runners[id(app)] = (app, runner)
        return runner

    def _resolve_runner(self, agent: Any) -> Runner | None:
        if _is_runner(agent):
            return _RunnerCall(agent).__call__
        if _is_app(agent):
            return _RunnerCall(self._runner_for_app(agent)).__call__
        return super()._resolve_runner(agent)

    def _call_runner(self, runner: Runner, payload: dict[str, Any], **run_kwargs: Any) -> Any:
        """Call ``runner`` and make a plugin-raised block surface as itself.

        ``new_message`` is the one run argument that is content rather than
        configuration: given explicitly it *replaces* the prompt the payload
        would have supplied, so it is what the model reads. It is screened as
        input here, or an override would walk past every control the payload
        went through.
        """
        new_message = run_kwargs.get("new_message")
        if new_message is not None:
            threats = self._screen_input(new_message)
            if threats and self.block_on_violation:
                raise SecurityBlockedError("Input blocked by security controls", threats)
        with reraise_security_blocks():
            raw = runner(self._prepare_input(payload), **run_kwargs)
        return _unwrapping(raw)


__all__ = ["DEFAULT_USER_ID", "GoogleADKAdapter"]
