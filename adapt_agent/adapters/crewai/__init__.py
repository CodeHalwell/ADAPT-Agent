"""CrewAI adapter for ADAPT-Agent.

`CrewAI <https://docs.crewai.com>`_ orchestrates a ``Crew`` of agents that you
run with ``crew.kickoff(inputs=...)`` (sync) or ``await crew.akickoff(inputs=...)``.
The result is a ``CrewOutput`` whose final text is on ``.raw``. A standalone
``Agent`` runs from messages instead: ``agent.kickoff(messages)``.

This adapter wraps a ``Crew``, a ``Flow``, a standalone ``Agent`` (or anything
exposing a callable ``kickoff`` / ``akickoff`` / ``kickoff_async`` / ``run``) and
applies ADAPT-Agent's governance pipeline. For a crew or flow the payload's
``context`` (its non-``messages`` keys) is forwarded as the ``inputs`` mapping of
template variables; for a standalone agent the payload's messages (or, failing
that, its prompt) are what the agent is asked. Extra keyword arguments to
``execute`` reach ``kickoff`` untouched (``input_files=``, ``from_checkpoint=``).
Importing this module never imports ``crewai``.

Governing what happens *inside* the run
---------------------------------------
By default the adapter governs the run's boundary only. Pass ``hooks=True`` and
every ``execute`` / ``aexecute`` also runs inside CrewAI's interception layer
(:func:`adapt_agent.integrations.crewai.governance_hooks`): tool calls and tool
results, model calls and steps are screened for the run, and the hooks are gone
when it ends -- nothing is left registered process-wide. ``tool_policy``,
``on_tool_violation`` and ``approve`` configure that layer and are refused
without ``hooks=True`` rather than accepted and ignored.

A block raised by a hook comes out of ``execute`` as
:class:`~adapt_agent.exceptions.SecurityBlockedError`, threats attached.

Streaming
---------
A ``Crew(stream=True)`` returns a streaming wrapper whose chunks reach whoever
iterates it before any post-run check could fire. The adapter therefore drains
it itself, inside governance, and returns the final ``CrewOutput`` only once it
was screened: a blocked answer is never delivered, at the cost of the stream's
latency benefit.

Example
-------
>>> from crewai import Crew                      # doctest: +SKIP
>>> crew = Crew(agents=[...], tasks=[...])        # doctest: +SKIP
>>> from adapt_agent.adapters import CrewAIAdapter
>>> from adapt_agent.security import Firewall
>>> adapter = CrewAIAdapter(firewall=Firewall(), hooks=True)
>>> guarded = adapter.wrap_agent(crew)            # doctest: +SKIP
>>> guarded.execute({"messages": [{"role": "user", "content": "hi"}], "topic": "AI"})  # doctest: +SKIP
"""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from adapt_agent.adapters._governed import GovernedAdapter, Runner, _extract_prompt
from adapt_agent.adversarial import AdversarialDefense
from adapt_agent.core.governance import GovernanceGate, ToolPolicy
from adapt_agent.core.middleware import Middleware
from adapt_agent.core.policy import PolicyEnforcer
from adapt_agent.exceptions import SecurityBlockedError
from adapt_agent.observability import AgentObserver
from adapt_agent.security.firewall import Firewall


def _is_stream(value: Any) -> bool:
    """Whether ``value`` is a CrewAI streaming wrapper (crew or flow).

    Duck-typed so the adapter never imports ``crewai``: the wrappers expose
    ``result`` (available after exhaustion) plus ``is_completed`` and ``chunks``.
    """
    return all(hasattr(value, name) for name in ("is_completed", "chunks", "get_full_text"))


def _block_behind(exc: BaseException) -> SecurityBlockedError | None:
    """Find the governance block behind an exception CrewAI may have wrapped."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, SecurityBlockedError):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


class CrewAIAdapter(GovernedAdapter):
    """Adapter for integrating ADAPT-Agent with CrewAI crews, flows and agents.

    See :class:`~adapt_agent.adapters._governed.GovernedAdapter` for the base
    constructor arguments. Added here:

    Args:
        hooks: Also run each execution inside CrewAI's interception hooks (tool,
            model and step governance). Default ``False``.
        tool_policy: Allow / deny / approval globs over tool names (needs ``hooks``).
        on_tool_violation: ``"refuse"`` or ``"raise"`` (needs ``hooks``); see
            :func:`~adapt_agent.integrations.crewai.governance_hooks`.
        approve: How a tool call that needs approval is answered (needs ``hooks``).
    """

    framework_name = "CrewAI"
    run_method_names = ("kickoff", "kickoff_async", "run")
    #: ``akickoff`` is the native coroutine; ``kickoff_async`` is, on a Crew, a
    #: thread hop around ``kickoff`` and the only async entry on older releases.
    async_run_method_names = ("akickoff", "kickoff_async", "kickoff", "run")
    operation = "crewai.kickoff"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        firewall: Firewall | None = None,
        defense: AdversarialDefense | None = None,
        policy_enforcer: PolicyEnforcer | None = None,
        observer: AgentObserver | None = None,
        middleware: Middleware | None = None,
        agent_id: str | None = None,
        block_on_violation: bool = True,
        hooks: bool = False,
        tool_policy: ToolPolicy | None = None,
        on_tool_violation: str | None = None,
        approve: bool | Callable[..., bool] | None = None,
    ):
        super().__init__(
            config,
            firewall=firewall,
            defense=defense,
            policy_enforcer=policy_enforcer,
            observer=observer,
            middleware=middleware,
            agent_id=agent_id,
            block_on_violation=block_on_violation,
        )
        if not hooks and (tool_policy is not None or on_tool_violation is not None or approve):
            raise ValueError(
                "tool_policy, on_tool_violation and approve act through CrewAI's interception "
                "hooks; pass hooks=True or they would be accepted and do nothing."
            )
        self.hooks = hooks
        self.tool_policy = tool_policy
        self.on_tool_violation = on_tool_violation or "refuse"
        self.approve = approve

    @property
    def gate(self) -> GovernanceGate:
        """The gate over this adapter's controls, tool policy included."""
        return GovernanceGate(
            firewall=self.firewall,
            defense=self.defense,
            policy_enforcer=self.policy_enforcer,
            block_on_violation=self.block_on_violation,
            agent_id=self.agent_id,
            tool_policy=self.tool_policy,
        )

    # -- running ---------------------------------------------------------------

    def _prepare_input(self, payload: dict[str, Any]) -> Any:
        # CrewAI's ``inputs`` are template variables, not a chat transcript.
        return {k: v for k, v in payload.items() if k != "messages"}

    def _call_runner(self, runner: Runner, payload: dict[str, Any], **run_kwargs: Any) -> Any:
        call = self._bind(runner, payload, run_kwargs)
        if not self.hooks:
            return self._finish(call())

        from adapt_agent.integrations.crewai import governance_hooks

        governance = governance_hooks(
            gate=self.gate,
            observer=self.observer,
            on_tool_violation=self.on_tool_violation,  # type: ignore[arg-type]
            approve=self.approve,
        )
        try:
            with governance:
                raw = self._finish(call())
        except Exception as exc:
            _surface(exc)
            raise
        if inspect.isawaitable(raw):
            # A coroutine runs only when awaited, after the ``with`` above has
            # exited; the hooks have to be registered for the awaiting itself.
            return self._scoped(governance, raw)
        return raw

    def _bind(self, runner: Runner, payload: dict[str, Any], run_kwargs: dict[str, Any]) -> Any:
        """Return a thunk calling ``runner`` the way its signature wants."""
        if self._takes_messages(runner):
            # A standalone ``Agent`` runs ``kickoff(messages)``. Handing it the
            # crew-style inputs dict (messages stripped, so ``{}``) is a
            # validation error and drops the very question it must answer.
            messages = payload.get("messages")
            asked: Any = (
                messages if isinstance(messages, list) and messages else _extract_prompt(payload)
            )
            return lambda: runner(asked, **run_kwargs)
        inputs = self._prepare_input(payload)
        # Prefer the idiomatic ``kickoff(inputs=...)`` call when supported.
        if self._accepts_kwarg(runner, "inputs"):
            return lambda: runner(inputs=inputs, **run_kwargs)
        return lambda: runner(inputs, **run_kwargs)

    def _finish(self, raw: Any) -> Any:
        """Drain a streaming result inside governance; defer an awaitable one."""
        if inspect.isawaitable(raw):
            return self._finish_async(raw)
        if _is_stream(raw):
            return self._drain(raw)
        return raw

    @staticmethod
    def _drain(stream: Any) -> Any:
        """Consume a crew stream so no chunk is delivered before it was screened.

        Returns the final ``CrewOutput``, which the caller then screens like any
        other result.
        """
        if getattr(stream, "_sync_iterator", None) is None:
            raise RuntimeError(
                "This CrewAI stream can only be consumed asynchronously; use aexecute()."
            )
        for _ in stream:
            pass
        return stream.result

    async def _finish_async(self, awaitable: Any) -> Any:
        value = await awaitable
        if not _is_stream(value):
            return value
        if getattr(value, "_async_iterator", None) is not None:
            async for _ in value:
                pass
            return value.result
        return await asyncio.to_thread(self._drain, value)

    async def _scoped(self, governance: Any, awaitable: Any) -> Any:
        try:
            with governance:
                return await awaitable
        except Exception as exc:
            _surface(exc)
            raise

    @staticmethod
    def _takes_messages(runner: Runner) -> bool:
        """Whether ``runner`` is a standalone-agent ``kickoff(messages, ...)``."""
        try:
            params = inspect.signature(runner).parameters
        except (TypeError, ValueError):
            return False
        return "messages" in params and "inputs" not in params

    @staticmethod
    def _accepts_kwarg(runner: Runner, name: str) -> bool:
        try:
            signature = inspect.signature(runner)
        except (TypeError, ValueError):
            return False
        for param in signature.parameters.values():
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                return True
            if param.name == name and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                return True
        return False


def _surface(exc: Exception) -> None:
    """Re-raise the governance block behind ``exc``, if CrewAI wrapped one.

    CrewAI re-raises most errors untouched, but a block wrapped on the way out
    must still reach the caller as ``SecurityBlockedError`` with its threats, not
    as whatever the framework wrapped it in. Returns when there is nothing to
    unwrap, so the caller re-raises the original.
    """
    block = _block_behind(exc)
    if block is not None and block is not exc:
        raise block from exc


__all__ = ["CrewAIAdapter"]
