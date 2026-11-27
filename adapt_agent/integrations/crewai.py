"""Governance for CrewAI: interception hooks, kickoff callbacks and task guardrails.

CrewAI (1.15+) has one interception layer, ``crewai.hooks``, that fires around
every model call, every tool call, every task / flow step and the start and end
of every crew or flow run. :func:`governance_hooks` binds the gate to it:

.. code-block:: python

    from adapt_agent.integrations.crewai import governance_hooks
    from adapt_agent.core.governance import ToolPolicy

    hooks = governance_hooks(firewall=fw, tool_policy=ToolPolicy(deny=["delete_*"]))
    with hooks:                        # scoped to this block, leaks nowhere else
        crew.kickoff(inputs={"topic": "AI"})

    hooks.install()                    # ...or process-wide, until
    hooks.uninstall()

Every crew, agent and flow run inside the block is covered -- sub-agents and
delegation targets included, which a per-agent attachment would silently skip.

What is governed
----------------
* **Tool calls** (``pre_tool_call``): the gate's tool API
  (:meth:`~adapt_agent.core.governance.GovernanceGate.check_tool_call`) decides
  from the tool policy, the arguments and policy rules; a denied call never
  executes. MCP tools mounted with ``Agent(mcps=[...])`` are ``BaseTool``
  subclasses and run through the same hook, so they are covered with no extra
  step. A2A delegation (``Agent(a2a=...)``) only emits events and has **no**
  hook, so it is not governed by this binding.
* **Tool results** (``post_tool_call``): screened as input before the model sees
  them (the indirect-injection seam). A poisoned result is replaced, or the run
  aborts -- never passed on.
* **Model calls** (``pre_model_call`` / ``post_model_call``): the messages going
  in and the response coming out.
* **Run boundaries** (``execution_start``/``input``/``output``) and **steps**
  (``pre_step``/``post_step``, tasks and flow methods): inputs and the final
  output, which is the only seam that sees a Flow, whose methods need not call a
  model at all.

Why a block must be a ``HookAborted``
-------------------------------------
CrewAI deliberately swallows every exception a hook raises *except*
``crewai.hooks.HookAborted`` (a buggy user hook must not break a run). An
ordinary ``SecurityBlockedError`` raised from a tool hook was swallowed and the
dangerous tool ran. The binding therefore raises
:class:`GovernanceBlocked`, a subclass of **both** ``HookAborted`` (so CrewAI
honours it) and ``SecurityBlockedError`` (so the caller of ``kickoff`` catches
the exception, with its threats, they would have caught anywhere else). Any other
exception inside the binding's own hooks fails closed for the same reason.

Block modes (``on_tool_violation``)
-----------------------------------
``"refuse"`` (default) keeps the run going: the tool does not execute and the
model is told so (CrewAI's own treatment of a hook-blocked tool), so an agent can
pick another route. ``"raise"`` aborts the run: the first block raises
:class:`GovernanceBlocked` out of ``kickoff``. ``refuse`` is the default because
a refused tool call is an ordinary event in an agent loop and CrewAI models it as
such; choose ``raise`` where one block means the run is compromised. Blocked
*model* input or output and blocked run boundaries always raise: there is no
refusal to hand a model that has not answered yet.

Approvals
---------
A decision of ``"ask"`` (a ``ToolPolicy(require_approval=...)`` match or a
``require_approval`` rule) is routed to ``approve``: ``True`` uses CrewAI's own
``ctx.request_human_input`` console prompt, a callable receives
``(decision, ctx)`` and returns a bool. With ``approve=None`` an ask fails closed
to a block. The prompt blocks the calling thread.

Streaming
---------
CrewAI streams model chunks to the consumer *while* the model is still
answering, and its hooks fire only when a call completes, so no hook can hold
back a streamed chunk. A blocked response is aborted and raises, but chunks
already emitted have been delivered. Where a blocked answer must never be seen,
do not stream through the raw crew: :class:`~adapt_agent.adapters.CrewAIAdapter`
drains a ``stream=True`` crew inside governance and returns the final output only
after it was screened.

Checkpoints
-----------
:func:`governance_callbacks` returns callables CrewAI can write into a checkpoint
and rebuild (see :class:`_CheckpointCallback`); a closure would be dropped on
restore, silently removing the control. ``Task`` guardrails are different: CrewAI
never serialises a callable guardrail, so :func:`governance_guardrail` does not
survive a restore -- use hooks, which are process state you install again after
``Crew.from_checkpoint(...)``, for anything that must.
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import re
import threading
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from adapt_agent.core.governance import GovernanceGate, ToolDecision, ToolPolicy
from adapt_agent.exceptions import SecurityBlockedError
from adapt_agent.integrations._common import as_state, build_gate, optional_import

if TYPE_CHECKING:  # pragma: no cover - typing only
    from adapt_agent.adversarial import AdversarialDefense
    from adapt_agent.core.policy import PolicyEnforcer
    from adapt_agent.security.firewall import Firewall

#: How a blocked tool call is handled. See the module docstring.
ToolViolationMode = Literal["refuse", "raise"]

#: An approver: ``approve(decision, ctx) -> bool``. ``True`` lets the call run.
Approver = Callable[[ToolDecision, Any], bool]

_MODES = ("refuse", "raise")
_APPROVE_WORDS = frozenset({"y", "yes", "approve", "approved", "ok"})
_MODULE = "adapt_agent.integrations.crewai"


# --------------------------------------------------------------------------- #
# The abort exception
# --------------------------------------------------------------------------- #

_blocked_cls: type[BaseException] | None = None
_blocked_lock = threading.Lock()


def _blocked_class() -> type[BaseException]:
    """Build :class:`GovernanceBlocked` the first time it is needed.

    It must subclass ``crewai.hooks.HookAborted``, which cannot be imported
    without CrewAI, while this module has to import without it.
    """
    global _blocked_cls
    if _blocked_cls is None:
        with _blocked_lock:
            if _blocked_cls is None:
                hooks = optional_import("crewai.hooks", "crewai", "CrewAI governance hooks")

                class GovernanceBlocked(hooks.HookAborted, SecurityBlockedError):  # type: ignore[name-defined]
                    """A governance block that CrewAI's hook layer actually honours.

                    Both a ``crewai.hooks.HookAborted`` and an
                    :class:`~adapt_agent.exceptions.SecurityBlockedError`: the
                    first is the only exception CrewAI does not swallow from a
                    hook, the second is what a caller of ``kickoff`` expects.
                    """

                    def __init__(self, reason: str, threats: list[str] | None = None) -> None:
                        Exception.__init__(self, reason)
                        self.reason = reason
                        self.threats = list(threats or [])
                        self.source = "adapt_agent"

                GovernanceBlocked.__module__ = _MODULE
                _blocked_cls = GovernanceBlocked
    return _blocked_cls


def _blocked(reason: str, threats: Sequence[str]) -> BaseException:
    return _blocked_class()(reason, list(threats))


# --------------------------------------------------------------------------- #
# Per-run state
# --------------------------------------------------------------------------- #


@dataclass
class GovernanceEvent:
    """One governance verdict, kept on :attr:`CrewAIGovernance.events`.

    Attributes:
        point: The CrewAI interception point (``"pre_tool_call"``, ...).
        action: ``"allow"``, ``"block"``, ``"ask"`` or ``"error"``.
        name: The tool name, agent role or boundary the verdict is about.
        threats: What was found; populated for ``allow`` too under a report-only gate.
        report_only: The gate would have blocked but is in report-only mode.
    """

    point: str
    action: str
    name: str
    threats: tuple[str, ...] = ()
    report_only: bool = False


@dataclass
class _Run:
    """What one crew/flow run needs to remember between hooks."""

    inputs: dict[str, Any] = field(default_factory=dict)
    #: A raise-mode block waiting to abort the run at the next seam that can.
    pending: BaseException | None = None
    #: Refusal reasons keyed by the blocked call's argument dict, for the post hook.
    refusals: dict[int, str] = field(default_factory=dict)
    #: Open observer spans, keyed by the call they belong to.
    spans: dict[tuple[str, int], str] = field(default_factory=dict)


_RUN: ContextVar[_Run | None] = ContextVar("adapt_crewai_run", default=None)


def _current_run() -> _Run:
    run = _RUN.get()
    if run is None:
        run = _Run()
        _RUN.set(run)
    return run


def _answer(payload: Any) -> Any:
    """What a task or crew output *says*, without the transcript behind it.

    A ``TaskOutput`` / ``CrewOutput`` also carries the message history and every
    task's output; scanning those re-flags a tool call the gate already refused,
    and aborts a run that handled it correctly. The answer is ``raw`` plus the
    structured forms.
    """
    if hasattr(payload, "raw"):
        parts = [getattr(payload, name, None) for name in ("raw", "pydantic", "json_dict")]
        return [part for part in parts if part is not None]
    return payload


def _role(ctx: Any) -> str:
    agent = getattr(ctx, "agent", None)
    role = getattr(agent, "role", None) or getattr(ctx, "agent_role", None)
    return role if isinstance(role, str) else ""


# --------------------------------------------------------------------------- #
# The hooks object
# --------------------------------------------------------------------------- #


class CrewAIGovernance:
    """ADAPT-Agent governance as CrewAI interception hooks.

    Build it with :func:`governance_hooks`. It is inert until entered
    (``with hooks:``, scoped to the block and to the contexts it spawns) or
    :meth:`install`-ed (process-wide until :meth:`uninstall`). Prefer the scoped
    form: a globally registered hook governs every crew in the process, including
    unrelated ones and the next test.

    Attributes:
        gate: The gate whose controls are applied.
        events: The most recent verdicts (bounded), oldest first -- the audit
            trail for a report-only rollout.
    """

    def __init__(
        self,
        gate: GovernanceGate,
        *,
        observer: Any = None,
        on_tool_violation: ToolViolationMode = "refuse",
        approve: bool | Approver | None = None,
        screen_models: bool = True,
        screen_tools: bool = True,
        screen_boundaries: bool = True,
        screen_steps: bool = True,
        agents: Sequence[str] | None = None,
        max_events: int = 1000,
    ) -> None:
        if on_tool_violation not in _MODES:
            raise ValueError(
                f"on_tool_violation must be one of {_MODES}, got {on_tool_violation!r}"
            )
        if isinstance(agents, str):
            raise TypeError("agents must be a collection of agent roles, not a str")
        if approve not in (None, True, False) and not callable(approve):
            raise TypeError("approve must be True, a callable (decision, ctx) -> bool, or None")
        self.gate = gate
        self.observer = observer
        self.on_tool_violation: ToolViolationMode = on_tool_violation
        self._approve: bool | Approver | None = approve or None
        self._agents = frozenset(agents) if agents is not None else None
        self.events: collections.deque[GovernanceEvent] = collections.deque(maxlen=max_events)
        self._screen = {
            "models": screen_models,
            "tools": screen_tools,
            "boundaries": screen_boundaries,
            "steps": screen_steps,
        }
        self._installed: list[tuple[Any, Callable[[Any], Any]]] = []
        self._stack: list[contextlib.ExitStack] = []
        self._points: list[tuple[Any, Callable[[Any], Any]]] | None = None

    # -- lifecycle -----------------------------------------------------------

    def _hook_table(self) -> list[tuple[Any, Callable[[Any], Any]]]:
        if self._points is None:
            hooks = optional_import("crewai.hooks", "crewai", "CrewAI governance hooks")
            point = getattr(hooks, "InterceptionPoint", None)
            if point is None:
                # An older CrewAI has a `crewai.hooks` module without the
                # interception layer. Failing here with a bare AttributeError
                # would say nothing about the cause or the way out.
                raise RuntimeError(
                    "governance_hooks() needs CrewAI's interception layer "
                    "(crewai.hooks.InterceptionPoint), which the installed CrewAI does "
                    "not have. Upgrade CrewAI, or on this release use "
                    "governance_callbacks() and governance_guardrail(), which do not "
                    "need it. tool_inspection_supported() reports which applies."
                )
            table: list[tuple[Any, Callable[[Any], Any]]] = []
            g = self._guard
            if self._screen["boundaries"]:
                table += [
                    (point.EXECUTION_START, g(self._on_start)),
                    (point.INPUT, g(self._on_input)),
                    (point.OUTPUT, g(self._on_output)),
                    (point.EXECUTION_END, g(self._on_end)),
                ]
            if self._screen["steps"]:
                table += [
                    (point.PRE_STEP, g(self._on_pre_step)),
                    (point.POST_STEP, g(self._on_post_step)),
                ]
            if self._screen["models"]:
                table += [
                    (point.PRE_MODEL_CALL, g(self._on_pre_model)),
                    (point.POST_MODEL_CALL, g(self._on_post_model)),
                ]
            if self._screen["tools"]:
                table += [
                    (point.PRE_TOOL_CALL, g(self._on_pre_tool)),
                    (point.POST_TOOL_CALL, g(self._on_post_tool)),
                ]
            self._points = table
        return self._points

    def install(self) -> CrewAIGovernance:
        """Register the hooks process-wide. Idempotent; undo with :meth:`uninstall`."""
        if self._installed:
            return self
        hooks = optional_import("crewai.hooks", "crewai", "CrewAI governance hooks")
        for point, fn in self._hook_table():
            hooks.register_hook(point, fn)
            self._installed.append((point, fn))
        return self

    def uninstall(self) -> None:
        """Remove what :meth:`install` registered. A no-op when not installed."""
        if not self._installed:
            return
        hooks = optional_import("crewai.hooks", "crewai", "CrewAI governance hooks")
        for point, fn in self._installed:
            hooks.unregister_hook(point, fn)
        self._installed = []

    @property
    def installed(self) -> bool:
        """Whether :meth:`install` is in effect."""
        return bool(self._installed)

    @contextlib.contextmanager
    def scope(self) -> Iterator[CrewAIGovernance]:
        """Register the hooks for this block (and the contexts it spawns) only.

        Uses CrewAI's execution-scoped registry, which a worker thread or task
        started from inside the block inherits, and which vanishes on exit --
        nothing is left registered to govern the next, unrelated run.
        """
        dispatch = optional_import("crewai.hooks.dispatch", "crewai", "CrewAI governance hooks")
        with dispatch.scoped_hooks():
            for point, fn in self._hook_table():
                dispatch.register_scoped(point, fn)
            yield self

    def __enter__(self) -> CrewAIGovernance:
        stack = contextlib.ExitStack()
        stack.enter_context(self.scope())
        self._stack.append(stack)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._stack.pop().close()

    # -- plumbing ------------------------------------------------------------

    def _skip(self, ctx: Any) -> bool:
        if self._agents is None:
            return False
        role = _role(ctx)
        return bool(role) and role not in self._agents

    def _record(
        self,
        point: str,
        action: str,
        name: str,
        threats: Sequence[str] = (),
        report_only: bool = False,
    ) -> None:
        self.events.append(GovernanceEvent(point, action, name, tuple(threats), report_only))

    def _state(self, ctx: Any, messages: Any = None) -> dict[str, Any]:
        """The mapping a policy rule is evaluated against.

        Always carries ``messages`` and ``context``; plus ``agent_role``,
        ``task`` (the task description) and the run's kickoff inputs, each
        splatted as top-level keys. Inputs sit *under* the fixed keys so a
        kickoff input named ``messages`` cannot replace what is being screened.
        """
        run = _current_run()
        extra: dict[str, Any] = {**run.inputs, "agent_role": _role(ctx)}
        task = getattr(ctx, "task", None)
        description = getattr(task, "description", None)
        if isinstance(description, str):
            extra["task"] = description
        iterations = getattr(ctx, "iterations", None)
        if isinstance(iterations, int):
            extra["iterations"] = iterations
        return as_state(messages, **extra)

    def _span_open(self, run: _Run, kind: str, key: int, operation: str) -> None:
        if self.observer is None:
            return
        trace_id = uuid.uuid4().hex
        self.observer.start_trace(trace_id, self.gate.agent_id, operation)
        run.spans[(kind, key)] = trace_id

    def _span_close(self, run: _Run, kind: str, key: int, status: str, result: str = "") -> None:
        trace_id = run.spans.pop((kind, key), None)
        if trace_id is not None and self.observer is not None:
            self.observer.end_trace(trace_id, status=status, result=result or None)

    def _abort_pending(self, run: _Run) -> None:
        """Raise a stored raise-mode block, once, at a seam where it propagates."""
        if run.pending is not None:
            err, run.pending = run.pending, None
            raise err

    def _guard(self, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """Fail closed on any error inside a hook.

        CrewAI swallows every non-``HookAborted`` exception from a hook and
        carries on, so a bug here would otherwise quietly un-govern the call. A
        report-only gate must not break a run, so it records the error instead.
        """

        aborted = optional_import("crewai.hooks", "crewai", "CrewAI governance hooks").HookAborted

        def guarded(ctx: Any) -> Any:
            if self._skip(ctx):
                return None
            try:
                return fn(ctx)
            except BaseException as exc:
                if isinstance(exc, aborted) or not isinstance(exc, Exception):
                    raise
                self._record(getattr(fn, "__name__", "hook"), "error", type(exc).__name__)
                if not self.gate.block_on_violation:
                    return None
                err = _blocked(
                    f"Governance hook failed closed [{self.gate.agent_id}]: {type(exc).__name__}",
                    ["governance:error"],
                )
                if self.on_tool_violation == "raise" and "tool" in guarded.__name__:
                    # The tool is refused either way (a hook abort skips it); in raise
                    # mode the run must also stop, which the post hook or the next
                    # seam does once it sees this.
                    _current_run().pending = err
                raise err from exc

        guarded.__name__ = getattr(fn, "__name__", "hook")
        return guarded

    # -- run boundaries ------------------------------------------------------

    def _on_start(self, ctx: Any) -> Any:
        run = _Run()
        payload = ctx.payload if isinstance(ctx.payload, dict) else getattr(ctx, "inputs", None)
        run.inputs = dict(payload) if isinstance(payload, dict) else {}
        _RUN.set(run)
        self._screen_input("execution_start", ctx.payload, self._state(ctx), "run inputs")
        return None

    def _on_input(self, ctx: Any) -> Any:
        self._screen_input("input", ctx.payload, self._state(ctx), "run inputs")
        return None

    def _on_output(self, ctx: Any) -> Any:
        run = _current_run()
        self._abort_pending(run)
        self._screen_output("output", ctx.payload, "run output")
        return None

    def _on_end(self, ctx: Any) -> Any:
        run = _current_run()
        for kind, key in list(run.spans):
            self._span_close(run, kind, key, "error", "run ended with the call open")
        run.refusals.clear()
        run.pending = None if getattr(ctx, "status", "completed") == "failed" else run.pending
        self._abort_pending(run)
        return None

    def _on_pre_step(self, ctx: Any) -> Any:
        self._abort_pending(_current_run())
        if getattr(ctx, "kind", None) == "flow_method":
            self._screen_input("pre_step", ctx.payload, self._state(ctx), "flow step input")
        return None

    def _on_post_step(self, ctx: Any) -> Any:
        self._abort_pending(_current_run())
        self._screen_output("post_step", ctx.payload, "step output")
        return None

    def _screen_input(self, point: str, payload: Any, state: dict[str, Any], what: str) -> None:
        try:
            threats = self.gate.review_input(payload, state=state)
        except SecurityBlockedError as err:
            self._record(point, "block", what, err.threats)
            raise _blocked(err.reason, err.threats) from None
        if threats:
            self._record(point, "allow", what, threats, report_only=True)

    def _screen_output(self, point: str, payload: Any, what: str) -> None:
        try:
            threats = self.gate.review_output(_answer(payload))
        except SecurityBlockedError as err:
            self._record(point, "block", what, err.threats)
            raise _blocked(err.reason, err.threats) from None
        if threats:
            self._record(point, "allow", what, threats, report_only=True)

    # -- model calls ---------------------------------------------------------

    def _on_pre_model(self, ctx: Any) -> Any:
        run = _current_run()
        self._abort_pending(run)
        messages = getattr(ctx, "messages", None) or []
        # The model's own earlier answers are not input: they were screened as
        # output when produced, and an input scan over them flags a model that
        # merely quoted a flagged string back.
        incoming = [
            m for m in messages if not (isinstance(m, dict) and m.get("role") == "assistant")
        ]
        key = id(messages)
        self._span_open(run, "model", key, "crewai.model_call")
        try:
            self._screen_input(
                "pre_model_call", incoming, self._state(ctx, incoming), "model input"
            )
        except BaseException as exc:
            self._span_close(run, "model", key, "error", str(exc))
            raise
        return None

    def _on_post_model(self, ctx: Any) -> Any:
        run = _current_run()
        key = id(getattr(ctx, "messages", None))
        try:
            self._screen_output("post_model_call", ctx.response, "model output")
        except BaseException as exc:
            self._span_close(run, "model", key, "error", str(exc))
            raise
        self._span_close(run, "model", key, "completed")
        return None

    # -- tool calls ----------------------------------------------------------

    def _on_pre_tool(self, ctx: Any) -> Any:
        run = _current_run()
        name = ctx.tool_name
        key = id(ctx.tool_input)
        state = self._state(ctx)
        decision = self.gate.check_tool_call(name, ctx.tool_input, state=state)
        self._span_open(run, "tool", key, f"crewai.tool:{name}")

        if decision.needs_approval:
            decision = self._ask(decision, ctx)

        self._record("pre_tool_call", decision.action, name, decision.threats, decision.report_only)
        if not decision.blocked:
            return None

        run.refusals[key] = decision.reason
        if self.on_tool_violation == "raise":
            err = _blocked(decision.reason, decision.threats)
            run.pending = err
            raise err
        # refuse: HookAborted makes CrewAI skip the tool and tell the model.
        raise _blocked(decision.reason, decision.threats)

    def _ask(self, decision: ToolDecision, ctx: Any) -> ToolDecision:
        """Resolve an ``"ask"`` to ``"allow"`` or ``"block"``; never leave it open."""
        approver = self._approve
        supported = callable(approver) or (approver is True and hasattr(ctx, "request_human_input"))
        if decision.resolve(approvals_supported=supported) != "ask":
            return _as_block(decision)
        try:
            if approver is True:
                prompt = (
                    f"ADAPT-Agent: approve tool call {decision.tool_name!r} "
                    f"({', '.join(decision.threats)})? Type 'approve' to allow:"
                )
                approved = ctx.request_human_input(prompt).strip().lower() in _APPROVE_WORDS
            else:
                assert callable(approver)
                approved = bool(approver(decision, ctx))
        except Exception:  # EOFError with no stdin, a crashed approver: deny
            approved = False
        if approved:
            return ToolDecision(
                "allow", decision.tool_name, decision.threats, False, decision.agent_id
            )
        return _as_block(decision)

    def _on_post_tool(self, ctx: Any) -> Any:
        run = _current_run()
        name = ctx.tool_name
        key = id(ctx.tool_input)
        if self.on_tool_violation == "raise":
            self._abort_pending(run)
        refusal = run.refusals.pop(key, None)
        if refusal is not None:
            # The call was refused before it ran; CrewAI's own message says only
            # "blocked by hook". Hand the model the gate's reason instead -- and
            # in raise mode this is the seam that carries the abort out.
            self._span_close(run, "tool", key, "error", refusal)
            return refusal

        decision = self.gate.check_tool_result(name, ctx.tool_result, state=self._state(ctx))
        self._record(
            "post_tool_call", decision.action, name, decision.threats, decision.report_only
        )
        if decision.blocked:
            self._span_close(run, "tool", key, "error", decision.reason)
            if self.on_tool_violation == "raise":
                raise _blocked(decision.reason, decision.threats)
            return (
                f"Result of tool {name!r} withheld by ADAPT-Agent [{decision.agent_id}]: "
                f"{', '.join(decision.threats)}."
            )
        self._span_close(run, "tool", key, "completed")
        return None


def _as_block(decision: ToolDecision) -> ToolDecision:
    return ToolDecision("block", decision.tool_name, decision.threats, False, decision.agent_id)


def governance_hooks(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    tool_policy: ToolPolicy | None = None,
    observer: Any = None,
    on_tool_violation: ToolViolationMode = "refuse",
    approve: bool | Approver | None = None,
    screen_models: bool = True,
    screen_tools: bool = True,
    screen_boundaries: bool = True,
    screen_steps: bool = True,
    agents: Sequence[str] | None = None,
) -> CrewAIGovernance:
    """Build governance hooks for CrewAI's interception layer.

    Args:
        gate: A pre-built gate to share; otherwise built from the controls below.
        firewall: Screens model input and output, tool arguments and results.
        defense: Adversarial analysis of everything entering the model.
        policy_enforcer: Rules evaluated against the per-hook state (see
            :meth:`CrewAIGovernance._state`) and, for tool-scoped rules, the call.
        block_on_violation: ``False`` records without blocking (report-only).
        agent_id: Named in errors and spans.
        tool_policy: Allow / deny / approval globs over tool names.
        observer: An :class:`~adapt_agent.observability.AgentObserver`; a span
            opens per governed model call and tool call.
        on_tool_violation: ``"refuse"`` (default) or ``"raise"`` -- see the module
            docstring.
        approve: How an ``"ask"`` is answered: ``True`` for CrewAI's console
            prompt, ``callable(decision, ctx) -> bool``, or ``None`` to fail closed.
        screen_models, screen_tools, screen_boundaries, screen_steps: Turn a
            family of hooks off. Off means ungoverned there, not "allowed".
        agents: Govern only these agent roles. Calls with no agent (a flow step,
            a bare LLM call) are still governed.

    Returns:
        A :class:`CrewAIGovernance`. Use it as a context manager (scoped) or call
        :meth:`~CrewAIGovernance.install` (process-wide).
    """
    resolved = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
        tool_policy=tool_policy,
    )
    return CrewAIGovernance(
        resolved,
        observer=observer,
        on_tool_violation=on_tool_violation,
        approve=approve,
        screen_models=screen_models,
        screen_tools=screen_tools,
        screen_boundaries=screen_boundaries,
        screen_steps=screen_steps,
        agents=agents,
    )


def install_hooks(**kwargs: Any) -> CrewAIGovernance:
    """:func:`governance_hooks` followed by :meth:`~CrewAIGovernance.install`.

    Process-wide. Keep the returned object and call ``uninstall()`` when done.
    """
    return governance_hooks(**kwargs).install()


# --------------------------------------------------------------------------- #
# Kickoff callbacks (checkpoint-safe)
# --------------------------------------------------------------------------- #

_checkpoint_callbacks: dict[str, _CheckpointCallback] = {}


def _slug(agent_id: str) -> str:
    digest = hashlib.sha1(agent_id.encode("utf-8")).hexdigest()[:8]  # noqa: S324 - not security
    return f"{re.sub(r'[^0-9A-Za-z_]', '_', agent_id)}_{digest}"


class _CheckpointCallback:
    """A kickoff callback CrewAI can write into a checkpoint *and* rebuild.

    CrewAI serialises a crew callback as ``"<module>.<qualname>"`` and, on
    restore, imports that path and uses what it finds. A closure has no such path
    and a callable instance has none either, so both come back as ``None`` and
    are dropped -- after a restore the crew ran with the control silently gone.

    This instance advertises a path of its own: its ``__qualname__`` is
    ``checkpoint_<before|after>_<id>``, a name this module resolves again through
    its module-level ``__getattr__`` to the instance registered under the same
    ``agent_id``. In the original process that is this very object; in a new
    process it is whatever :func:`governance_callbacks` registered under that id
    before the restore. If nothing was registered the resolved callable refuses
    every call (see :class:`_MissingGate`) rather than letting the run go ungoverned.
    """

    def __init__(self, gate: GovernanceGate, kind: str) -> None:
        self._resolved = gate
        self._kind = kind
        # An instance attribute, deliberately: `callable_to_string` reads
        # `fn.__qualname__`, and only an instance attribute shadows the class's.
        self.__qualname__ = f"checkpoint_{kind}_{_slug(gate.agent_id)}"

    def __call__(self, value: Any) -> Any:
        if self._kind == "before":
            self._resolved.review_input(
                value, state=as_state(value) if isinstance(value, dict) else as_state(None)
            )
        else:
            self._resolved.review_output(value)
        return value  # governance never rewrites what it screens


class _MissingGate:
    """Stands in for a checkpointed callback whose gate this process never registered."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.__qualname__ = name

    def __call__(self, value: Any) -> Any:
        raise SecurityBlockedError(
            f"Checkpoint restored a governance callback ({self._name}) but no gate with that "
            "agent_id is registered in this process. Call governance_callbacks(..., "
            "agent_id=<the same id>) before Crew.from_checkpoint(...).",
            ["governance:gate_not_registered"],
        )


def __getattr__(name: str) -> Any:
    """Resolve a checkpointed callback path; also the lazy ``GovernanceBlocked``."""
    if name == "GovernanceBlocked":
        return _blocked_class()
    if name.startswith(("checkpoint_before_", "checkpoint_after_")):
        found = _checkpoint_callbacks.get(name)
        return found if found is not None else _MissingGate(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def governance_callbacks(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    screen_output: bool = True,
) -> dict[str, list[Callable[[Any], Any]]]:
    """Build CrewAI kickoff callbacks applying ADAPT-Agent governance.

    ``before_kickoff_callbacks`` is ``(inputs: dict) -> dict`` and
    ``after_kickoff_callbacks`` is ``(result) -> result``; both fire once per
    kickoff. They are the lightest binding (no tool or model-call coverage -- use
    :func:`governance_hooks` for that) and the one that survives a CrewAI
    checkpoint: see :class:`_CheckpointCallback`. To restore, set
    ``CREWAI_DESERIALIZE_CALLBACKS=1`` (CrewAI refuses to import callback paths
    otherwise) and call this again with the same ``agent_id`` first.

    Args:
        gate: A pre-built gate to share; otherwise built from the controls below.
        firewall: Screens the kickoff inputs and the crew output.
        defense: Adversarial analysis of the inputs.
        policy_enforcer: Evaluated against the kickoff inputs as agent state
            (``messages`` and ``context`` are added when absent).
        block_on_violation: ``False`` scans without raising.
        agent_id: Named in the raised error, and the key a restored checkpoint
            finds this gate by -- keep it stable across processes.
        screen_output: Also register an after-kickoff callback.

    Returns:
        A dict of ``Crew(...)`` kwargs: ``before_kickoff_callbacks`` and (when
        ``screen_output``) ``after_kickoff_callbacks``.
    """
    resolved = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
    )
    before = _CheckpointCallback(resolved, "before")
    _checkpoint_callbacks[before.__qualname__] = before
    callbacks: dict[str, list[Callable[[Any], Any]]] = {"before_kickoff_callbacks": [before]}

    if screen_output:
        after = _CheckpointCallback(resolved, "after")
        _checkpoint_callbacks[after.__qualname__] = after
        callbacks["after_kickoff_callbacks"] = [after]

    return callbacks


def governance_guardrail(gate: GovernanceGate) -> Callable[[Any], tuple[bool, Any]]:
    """Return a CrewAI ``Task(guardrail=...)`` that screens a task's output.

    A task guardrail returns ``(True, output)`` to accept or ``(False, message)``
    to reject -- CrewAI feeds the message back and retries up to
    ``guardrail_max_retries``. Use this on the tasks that touch untrusted
    content, where a crew-level callback would fire too late.

    The inner function carries **no return annotation** on purpose. This module
    uses ``from __future__ import annotations``, which makes any annotation a
    string, and ``Task`` accepts a guardrail's annotation only when it is the real
    ``tuple[bool, Any]`` object -- a string raised ``ValidationError`` and the
    guardrail could not be attached at all.

    A callable guardrail is never written into a CrewAI checkpoint, so it does
    not survive ``Crew.from_checkpoint``; see the module docstring.
    """

    def _guardrail(output: Any):
        # The answer only: a TaskOutput also carries the transcript of earlier
        # attempts, so scanning all of it re-flags the rejected draft forever.
        threats = gate.scan_output(_answer(output))
        if threats:
            return False, (
                f"Output blocked by ADAPT-Agent [{gate.agent_id}]: "
                f"{', '.join(sorted(set(threats)))}. Rewrite without the flagged content."
            )
        return True, output

    return _guardrail


def tool_inspection_supported() -> bool:
    """Whether the installed CrewAI has the interception layer :func:`governance_hooks` needs."""
    try:
        from crewai.hooks import InterceptionPoint  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = [
    "Approver",
    "CrewAIGovernance",
    "GovernanceEvent",
    "ToolViolationMode",
    "governance_callbacks",
    "governance_guardrail",
    "governance_hooks",
    "install_hooks",
    "tool_inspection_supported",
]
