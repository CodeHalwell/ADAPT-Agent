"""OpenAI Agents SDK adapter for ADAPT-Agent.

The `OpenAI Agents SDK <https://openai.github.io/openai-agents-python>`_ runs an
``Agent`` through the ``Runner`` class: ``Runner.run_sync(agent, input)`` (sync)
or ``await Runner.run(agent, input)`` (async). The result is a ``RunResult``
whose final text is on ``.final_output``.

You can wrap an OpenAI ``Agent`` directly -- the adapter lazily imports
``agents.Runner`` at execution time and drives it for you -- or wrap any plain
callable ``run(input) -> result`` if you want full control over the run
configuration. Importing this module never imports ``agents``.

Run configuration and resuming
------------------------------
Keyword arguments to ``execute`` / ``aexecute`` go to ``Runner.run`` untouched:
``session=``, ``context=``, ``run_config=``, ``hooks=``, ``max_turns=``. They
configure the run and are not screened as content.

A run that pauses for human approval (a tool with ``needs_approval``, or an
"ask" from :func:`~adapt_agent.integrations.openai_agents.govern_tools`) comes
back with ``out["interruptions"]``. Approve or reject on the state and pass the
state back -- it reaches the SDK as the run input, not as a prompt::

    out = guarded.execute({"prompt": "refund order 7"})
    state = out["result"].to_state()
    for item in out["interruptions"]:
        state.approve(item)            # or state.reject(item)
    out = guarded.execute({"input": state})

``{"input": [...]}`` passes a list of SDK input items through the same way.

Example
-------
>>> from agents import Agent                       # doctest: +SKIP
>>> agent = Agent(name="Assistant", instructions="...")  # doctest: +SKIP
>>> from adapt_agent.adapters import OpenAIAgentsAdapter
>>> from adapt_agent.security import Firewall
>>> adapter = OpenAIAgentsAdapter(firewall=Firewall())
>>> guarded = adapter.wrap_agent(agent)            # doctest: +SKIP
>>> guarded.execute({"messages": [{"role": "user", "content": "hi"}]})  # doctest: +SKIP
"""

import inspect
from collections.abc import Mapping
from typing import Any, cast

from adapt_agent.adapters._governed import (
    GovernedAdapter,
    GovernedEnvelope,
    Runner,
    _extract_prompt,
)
from adapt_agent.exceptions import AdapterError, MissingDependencyError


def _is_run_state(value: Any) -> bool:
    """Whether ``value`` is an SDK ``RunState`` (a paused run to resume).

    Duck-typed so this module never imports the SDK: a ``RunState`` is the one
    run input that carries ``approve``/``reject`` and serialises itself.
    """
    return all(callable(getattr(value, name, None)) for name in ("approve", "reject", "to_json"))


def _is_input_items(value: Any) -> bool:
    """Whether ``value`` is a list of SDK input items (``{"role"|"type": ...}``)."""
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(item, Mapping) and ("role" in item or "type" in item) for item in value)
    )


def _with_interruptions(result: Any) -> Any:
    """Surface a paused run's pending approvals next to the result.

    A ``RunResult`` that stopped for approval has ``final_output=None`` and the
    pending calls on ``.interruptions``; returned bare, the envelope reads like
    an empty answer. Only an interrupted run gets the extra key, so every other
    result keeps the one-key envelope extractors rely on.
    """
    interruptions = getattr(result, "interruptions", None)
    if isinstance(interruptions, list) and interruptions:
        envelope = GovernedEnvelope(result=result)
        envelope["interruptions"] = list(interruptions)
        return envelope
    return result


async def _awith_interruptions(awaitable: Any) -> Any:
    return _with_interruptions(await awaitable)


class OpenAIAgentsAdapter(GovernedAdapter):
    """Adapter for integrating ADAPT-Agent with the OpenAI Agents SDK.

    Wrap an OpenAI ``Agent`` (driven via a lazily-imported ``Runner``) or any
    plain callable ``run(input) -> result``. See
    :class:`~adapt_agent.adapters._governed.GovernedAdapter` for the full
    constructor signature, and the module docstring for run keyword arguments,
    resuming an approval interruption, and :meth:`run_streamed`.

    The adapter governs the boundary of the run. For every tool call, every
    agent reached by a handoff and every model call inside the run, combine it
    with :mod:`adapt_agent.integrations.openai_agents`: pass
    ``hooks=governance_run_hooks(...)`` through ``execute`` and govern the
    agent's tools with ``govern_tools(...)``.
    """

    framework_name = "OpenAI Agents"
    run_method_names = ("run_sync", "run")
    async_run_method_names = ("run", "run_sync")
    operation = "openai_agents.run"

    def _resolve_runner(self, agent: Any) -> Runner | None:
        # An object exposing run_sync/run (e.g. a custom runner) or a plain
        # callable is used as-is.
        runner = super()._resolve_runner(agent)
        if runner is not None:
            return runner
        # Otherwise treat ``agent`` as an OpenAI Agent and drive it through the
        # SDK's Runner, imported lazily so importing this module stays cheap.
        # Run keyword arguments (session, context, run_config, hooks, max_turns)
        # must reach `Runner.run_sync`: a lambda taking only the prompt turned
        # every one of them into a TypeError.
        return lambda prompt, **run_kwargs: self._run_with_sdk(agent, prompt, **run_kwargs)

    def _resolve_async_runner(self, agent: Any) -> Runner | None:
        # A custom runner exposing `run`/`run_sync` is handled by the base class.
        for name in self.async_run_method_names:
            candidate = getattr(agent, name, None)
            if callable(candidate):
                return cast(Runner, candidate)
        if callable(agent):
            return cast(Runner, agent)
        # A directly-wrapped SDK ``Agent`` exposes neither, so without this the
        # base class would fall back to the *synchronous* `Runner.run_sync`
        # lambda and `aexecute` would block the caller's event loop.
        return lambda prompt, **run_kwargs: self._arun_with_sdk(agent, prompt, **run_kwargs)

    @staticmethod
    def _arun_with_sdk(agent: Any, prompt: Any, **run_kwargs: Any) -> Any:
        """Return the SDK's ``Runner.run`` coroutine for ``aexecute`` to await."""
        try:
            from agents import Runner
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise MissingDependencyError("openai-agents", "openai-agents") from exc
        return Runner.run(agent, prompt, **run_kwargs)

    @staticmethod
    def _run_with_sdk(agent: Any, prompt: Any, **run_kwargs: Any) -> Any:
        try:
            from agents import Runner
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise MissingDependencyError("openai-agents", "openai-agents") from exc
        return Runner.run_sync(agent, prompt, **run_kwargs)

    def _prepare_input(self, payload: Any) -> Any:
        """Shape the payload into ``Runner.run``'s ``input``.

        A ``RunState`` -- a run paused for approval -- and a list of SDK input
        items are passed through untouched, either bare (once the governed
        pipeline accepts a non-dict payload) or under ``"input"``. Reduced to a
        prompt string, a resume silently became a *new* run whose prompt was the
        state's repr, and the approved tool never ran. Everything else becomes
        the latest user prompt, as before.
        """
        if _is_run_state(payload):
            return payload
        if isinstance(payload, Mapping):
            inner = payload.get("input")
            if _is_run_state(inner) or _is_input_items(inner):
                return inner
        return _extract_prompt(payload)

    def _call_runner(self, runner: Runner, payload: Any, **run_kwargs: Any) -> Any:
        raw = super()._call_runner(runner, payload, **run_kwargs)
        if inspect.isawaitable(raw):
            return _awith_interruptions(raw)
        return _with_interruptions(raw)

    def run_streamed(self, agent: Any, input_data: Any, **run_kwargs: Any) -> Any:
        """Start a governed ``Runner.run_streamed`` and return a :class:`GovernedStream`.

        The same pipeline as :meth:`execute` -- input screening, policy and
        pre-middleware before the run starts (a block raises here, before any
        model call) -- and then each model response's text is held until it has
        been screened by this adapter's firewall, so a blocked answer is never
        delivered. Post-middleware and final-output screening run when the
        stream is exhausted, closing the observer trace::

            stream = adapter.run_streamed(agent, {"prompt": "hi"}, context=ctx)
            async for event in stream:
                ...
            print(stream.result.final_output)

        Raises:
            AdapterError: If ``agent`` is not an SDK ``Agent`` -- a plain
                callable has no streamed entry point to govern.
            SecurityBlockedError: From here for a blocked input; from the
                iteration for a blocked response.
        """
        try:
            import agents as sdk
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise MissingDependencyError("openai-agents", "openai-agents") from exc
        from adapt_agent.integrations.openai_agents import GovernedStream

        if not isinstance(agent, sdk.Agent):
            raise AdapterError(
                f"{type(self).__name__}.run_streamed needs an agents.Agent, not "
                f"{type(agent).__name__}: only the SDK's own Runner.run_streamed "
                "produces a stream that can be governed."
            )
        governed = cast(Any, self.wrap_agent(agent))
        trace_id, payload = governed._before(input_data)
        try:
            result = sdk.Runner.run_streamed(agent, self._prepare_input(payload), **run_kwargs)
        except BaseException as exc:
            governed._trace_error(trace_id, exc)
            raise
        return GovernedStream(
            result,
            self.gate,
            on_finish=lambda: governed._after(trace_id, result),
            on_error=lambda exc: governed._trace_error(trace_id, exc),
        )


__all__ = ["OpenAIAgentsAdapter"]
