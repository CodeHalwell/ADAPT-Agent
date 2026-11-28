"""Governance for LangChain v1 agents and LangGraph graphs.

Four seams, from most to least specific:

* :func:`governance_middleware` -- an ``AgentMiddleware`` for
  ``langchain.agents.create_agent``, the agent constructor of LangChain v1 and
  LangGraph v1. It screens every model call (input before, output after, both
  inside ``wrap_model_call``) and every tool call (``wrap_tool_call``): a denied
  tool never runs, screened arguments never reach it, a poisoned result never
  reaches the next model call, and an ``"ask"`` becomes a LangGraph
  ``interrupt()`` a human answers with ``Command(resume=...)``.

  .. code-block:: python

      from langchain.agents import create_agent
      from adapt_agent.integrations.langgraph import governance_middleware

      agent = create_agent(
          model, tools,
          middleware=[governance_middleware(firewall=fw, tool_policy=policy)],
          checkpointer=InMemorySaver(),   # needed only for approvals
      )

* :func:`governance_tool_wrapper` -- the same tool governance for a hand-built
  ``StateGraph``: ``ToolNode(tools, **governance_tool_wrapper(...))``.
* :func:`governance_callback_handler` -- a run-level net. Passed in
  ``config={"callbacks": [...]}`` it reaches **every** model and tool call of the
  run, sub-agents and nested graphs included, whether or not they carry the
  middleware. A callback cannot substitute a refusal or pause for a human, so it
  only aborts.
* :func:`governance_hooks` / :func:`governance_node` -- the original
  ``pre_model_hook`` / ``post_model_hook`` nodes for ``langgraph.prebuilt``'s
  ``create_react_agent``. That constructor is **deprecated upstream** (moved to
  ``langchain.agents.create_agent``, to be removed in LangGraph 2.0), and
  ``create_agent`` rejects the hook keywords with ``TypeError``. Kept working
  for existing graphs; new code uses the middleware.

Attaching per model call is what an outer wrapper cannot do: a supervisor runs
several agents, and wrapping the compiled graph screens only the first input
and the last output, while these seams fire on every model call -- including one
whose input came from a tool result rather than from the user, which is exactly
where injected content arrives.

Policy state
------------
A fail-closed :class:`~adapt_agent.core.policy.PolicyEnforcer` treats a rule it
cannot evaluate as violated, so the state each hook hands it is built
deliberately (see :func:`policy_state`):

* ``messages`` -- the messages being screened (the model request's messages, or
  the graph state's ``messages`` for a tool call).
* every other key of the **graph state** (custom ``state_schema`` fields
  included; a Pydantic or dataclass state is converted to a mapping).
* ``context`` -- the run's ``Runtime.context`` (``create_agent(context_schema=...)``
  plus ``invoke(..., context=...)``) when it is a mapping, dataclass or Pydantic
  model; otherwise the graph state's own ``context`` key, else ``{}``.

The run-level callback sees no graph state: its rules can rely on ``messages``
and ``context`` (``{}``) only.

Streaming
---------
LangGraph's ``stream_mode="messages"`` forwards model tokens to the consumer
from a callback, *while the model is still generating* -- before any post-model
check can run. With ``screen_stream=True`` the middleware tags the model call
``nostream``, so the token stream is suppressed and the complete message is
delivered from the model node's output instead, only after output screening has
passed it. The cost is latency: the consumer gets each model message whole,
when it is finished, rather than token by token. With ``screen_stream=False``
(the default) tokens stream as usual and a blocked response is stopped from
entering the graph state and reaching the caller's final result, but tokens
already streamed have been seen. ``stream_mode="updates"`` / ``"values"`` are
safe either way, because output is screened inside the model node, before it
writes to the state.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

from adapt_agent.core.governance import BLOCK, GovernanceGate, ToolDecision
from adapt_agent.exceptions import SecurityBlockedError
from adapt_agent.integrations._common import as_state, build_gate, optional_import, traced

if TYPE_CHECKING:  # pragma: no cover - typing only
    from adapt_agent.adversarial import AdversarialDefense
    from adapt_agent.core.governance import ToolPolicy
    from adapt_agent.core.policy import PolicyEnforcer
    from adapt_agent.security.firewall import Firewall

#: ``on_tool_violation`` values.
_TOOL_VIOLATION_MODES = ("raise", "refuse")

#: LangGraph's ``TAG_NOSTREAM``: a chat-model run carrying it is skipped by the
#: ``stream_mode="messages"`` handler, whose node-output path then delivers the
#: finished message instead. Spelled out so this module imports no LangGraph.
_TAG_NOSTREAM = "nostream"

#: ``langgraph._internal._constants.CONFIG_KEY_CHECKPOINTER``: where Pregel puts
#: the run's checkpointer in ``config["configurable"]``. An ``interrupt()``
#: without one pauses a run that can never be resumed.
_CONFIG_KEY_CHECKPOINTER = "__pregel_checkpointer"

#: Decisions an approver may return, in ``HumanInTheLoopMiddleware``'s terms.
_APPROVAL_DECISIONS = ("approve", "edit", "reject", "respond")


# -- policy state ---------------------------------------------------------------


def _as_mapping(value: Any) -> dict[str, Any] | None:
    """A mapping view of a graph state or runtime context, or ``None``.

    LangGraph accepts a ``TypedDict``, a Pydantic model or a dataclass as the
    state schema, and a ``context_schema`` of any of those. Passing anything
    but a mapping to the enforcer made policy silently inert (the old node
    passed ``None`` for a non-dict state), so each is converted explicitly.
    """
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: getattr(value, f.name, None) for f in dataclasses.fields(value)}
    dump = getattr(value, "model_dump", None)
    if callable(dump) and hasattr(value, "model_fields"):
        try:
            dumped = dump()
        except Exception:
            return None
        if isinstance(dumped, dict):
            return dumped
    return None


def policy_state(messages: Any, graph_state: Any = None, runtime: Any = None) -> dict[str, Any]:
    """Build the mapping a policy rule is evaluated against.

    Args:
        messages: What is being screened; always lands under ``messages``.
        graph_state: The graph's state (``request.state`` in a middleware hook,
            the node's input in a hand-built graph). Every key but ``messages``
            is copied in.
        runtime: A LangGraph ``Runtime`` / ``ToolRuntime``; its ``context`` is
            exposed as ``context`` when it can be read as a mapping.

    Returns:
        ``{"messages": ..., "context": ..., <graph state keys>}``.
    """
    extra = _as_mapping(graph_state) or {}
    extra.pop("messages", None)
    context = _as_mapping(getattr(runtime, "context", None))
    if context is not None:
        extra["context"] = context
    return as_state(messages, **extra)


def _state_messages(state: Any) -> Any:
    mapping = _as_mapping(state)
    if mapping is not None and "messages" in mapping:
        return mapping["messages"]
    return state


# -- shared tool-call governance -------------------------------------------------


def _check_mode(on_tool_violation: str) -> None:
    if on_tool_violation not in _TOOL_VIOLATION_MODES:
        raise ValueError(
            f"on_tool_violation must be one of {_TOOL_VIOLATION_MODES}, got {on_tool_violation!r}"
        )


def _has_checkpointer() -> bool:
    """Whether the running graph can be paused and resumed.

    ``interrupt()`` needs a checkpointer to resume from. Without one the run
    stops with an interrupt nobody can answer, so an ``"ask"`` fails closed to a
    refusal instead. Outside a graph (``get_config`` raises) there is nothing to
    pause either.
    """
    try:
        from langgraph.config import get_config

        configurable = get_config().get("configurable") or {}
    except Exception:
        return False
    return configurable.get(_CONFIG_KEY_CHECKPOINTER) is not None


def _record(observer: Any, decision: ToolDecision, stage: str) -> None:
    """Log a tool decision that found anything, including report-only ones."""
    if observer is None or not decision.threats:
        return
    observer.log(
        "warning",
        decision.reason,
        agent_id=decision.agent_id or None,
        metadata={
            "stage": stage,
            "tool": decision.tool_name,
            "action": decision.action,
            "threats": list(decision.threats),
            "report_only": decision.report_only,
        },
    )


def _record_threats(observer: Any, agent_id: str, stage: str, threats: list[str]) -> None:
    """Log model-call threats a report-only gate let through."""
    if observer is None or not threats:
        return
    observer.log(
        "warning",
        f"Model {stage} threats reported [{agent_id}]: {', '.join(threats)}.",
        agent_id=agent_id,
        metadata={"stage": f"model_{stage}", "threats": threats, "report_only": True},
    )


class _ToolGovernor:
    """The ``wrap_tool_call`` logic, shared by the middleware and ``ToolNode``.

    Framework-free at construction: LangChain types are imported when a call
    is governed, so importing this module never imports LangChain.
    """

    def __init__(
        self,
        gate: GovernanceGate,
        *,
        on_tool_violation: str,
        approvals: bool,
        approval_args: bool,
        observer: Any,
    ):
        _check_mode(on_tool_violation)
        self.gate = gate
        self.on_tool_violation = on_tool_violation
        self.approvals = approvals
        self.approval_args = approval_args
        self.observer = observer

    # -- the two entry points ----------------------------------------------------

    def wrap(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        name = request.tool_call["name"]
        with traced(self.observer, self.gate.agent_id, f"langgraph.tool:{name}"):
            state = self._state(request)
            outcome = self._before(request, state)
            if not _is_request(outcome):
                return outcome
            return self._after(outcome, handler(outcome), state)

    async def awrap(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        name = request.tool_call["name"]
        with traced(self.observer, self.gate.agent_id, f"langgraph.tool:{name}"):
            state = self._state(request)
            outcome = self._before(request, state)
            if not _is_request(outcome):
                return outcome
            return self._after(outcome, await handler(outcome), state)

    # -- stages ------------------------------------------------------------------

    def _state(self, request: Any) -> dict[str, Any]:
        graph_state = getattr(request, "state", None)
        return policy_state(
            _state_messages(graph_state), graph_state, getattr(request, "runtime", None)
        )

    def _before(self, request: Any, state: dict[str, Any]) -> Any:
        """The request to run (possibly edited by an approver), or a ToolMessage."""
        call = request.tool_call
        decision = self.gate.check_tool_call(call["name"], call.get("args"), state=state)
        _record(self.observer, decision, "call")
        if decision.needs_approval:
            if self.approvals and _has_checkpointer():
                return self._ask(request, decision, state)
            # No way to pause: an ask must not run the tool unseen.
            decision = dataclasses.replace(
                decision, action=decision.resolve(approvals_supported=False)
            )
        if decision.blocked:
            return self._refuse(request, decision)
        return request

    def _after(self, request: Any, result: Any, state: dict[str, Any]) -> Any:
        """Screen the tool's result before it can reach the next model call."""
        name = request.tool_call["name"]
        content = getattr(result, "content", result) if _is_tool_message(result) else result
        decision = self.gate.check_tool_result(name, content, state=state)
        _record(self.observer, decision, "result")
        if not decision.blocked:
            return result
        if self.on_tool_violation == "raise":
            raise SecurityBlockedError(_withheld(decision), list(decision.threats))
        # A `Command` result's state update (a handoff, a custom key) is dropped
        # with its content: half-applying a poisoned result is not withholding it.
        return _tool_message(request, _withheld(decision), status="error")

    def _refuse(self, request: Any, decision: ToolDecision) -> Any:
        if self.on_tool_violation == "raise":
            raise SecurityBlockedError(decision.reason, list(decision.threats))
        return _tool_message(request, decision.reason, status="error")

    def _ask(self, request: Any, decision: ToolDecision, state: dict[str, Any]) -> Any:
        """Pause for a human, then act on the answer. Fails closed on anything odd."""
        from langgraph.types import interrupt

        call = request.tool_call
        answer = interrupt(
            self._approval_request(call, decision), response_schema=_DECISION_JSON_SCHEMA
        )
        kind, body = _parse_decision(answer)
        if kind == "approve":
            return request
        if kind == "edit":
            edited = body.get("edited_action") if isinstance(body, Mapping) else None
            if (
                not isinstance(edited, Mapping)
                or edited.get("name", call["name"]) != call["name"]
                or not isinstance(edited.get("args"), Mapping)
            ):
                # Switching tools would bypass the name-based policy the human
                # was asked about; a malformed edit is not an approval.
                return self._refuse(request, _blocked(decision, "tool:approval:invalid_edit"))
            args = dict(edited["args"])
            recheck = self.gate.check_tool_call(call["name"], args, state=state)
            _record(self.observer, recheck, "call")
            if recheck.blocked:
                # The edit is screened like the model's own call: an approver
                # cannot paste a payload past the firewall. A remaining "ask"
                # is the approval list again, which this human just answered.
                return self._refuse(request, recheck)
            return request.override(tool_call={**call, "args": args})
        if kind == "respond":
            message = str(body.get("message", "")) if isinstance(body, Mapping) else ""
            screened = self.gate.check_tool_result(call["name"], message, state=state)
            _record(self.observer, screened, "result")
            if screened.blocked:
                return _tool_message(request, _withheld(screened), status="error")
            return _tool_message(request, message, status="success")
        reason = body.get("message") if isinstance(body, Mapping) else None
        if kind == "reject" and isinstance(reason, str) and reason:
            text = f"User rejected the tool call for `{call['name']}` with reason: {reason}"
        else:
            # A rejection, or an answer this binding cannot read: never a run.
            text = (
                f"User rejected the tool call for `{call['name']}`. The tool was not "
                "executed. Do not retry this tool call unless the user explicitly requests it."
            )
        return _tool_message(request, text, status="error")

    def _approval_request(self, call: Mapping[str, Any], decision: ToolDecision) -> dict[str, Any]:
        """The interrupt value, readable by ``HumanInTheLoopMiddleware`` UIs.

        It carries both of that middleware's shapes -- the per-call
        ``ToolApprovalRequest`` keys (``type``, ``tool_call_id``, ``name``,
        ``args``, ``description``) and the batched ``HITLRequest`` keys
        (``action_requests``, ``review_configs``) -- so an approval UI written
        for either renders it unchanged, plus ADAPT-Agent's own ``threats`` and
        ``agent_id``. ``description`` is :attr:`ToolDecision.reason`: the tool
        and threat labels only.

        ``args`` are included because an approver cannot judge a call without
        them, and an ``"ask"`` is only ever raised for arguments the firewall
        and defense already passed (a content hit is a block, which outranks an
        ask). ``approval_args=False`` replaces them with ``{}`` for tools whose
        arguments the approver must not see; ``edit`` is then not offered.
        """
        args = dict(call.get("args") or {}) if self.approval_args else {}
        allowed = list(_APPROVAL_DECISIONS if self.approval_args else ("approve", "reject"))
        description = decision.reason
        return {
            "type": "tool_approval",
            "tool_call_id": call.get("id"),
            "name": call["name"],
            "args": args,
            "description": description,
            "action_requests": [{"name": call["name"], "args": args, "description": description}],
            "review_configs": [{"action_name": call["name"], "allowed_decisions": allowed}],
            "source": "adapt_agent",
            "agent_id": decision.agent_id,
            "threats": list(decision.threats),
        }


#: Published with the interrupt (LangGraph does not enforce a dict schema), so
#: a client can render the choices. Answers are validated by `_parse_decision`,
#: which also accepts the batched ``{"decisions": [...]}`` shape a
#: ``HumanInTheLoopMiddleware`` UI sends -- a typed ``response_schema`` would
#: reject that with an unhandled ``ValidationError``.
_DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {"enum": list(_APPROVAL_DECISIONS)},
        "message": {"type": "string"},
        "edited_action": {"type": "object"},
    },
    "required": ["type"],
}


def _parse_decision(answer: Any) -> tuple[str, Any]:
    """Normalise a resume value to ``(kind, decision)``; unreadable means reject.

    Accepted: a ``HumanInTheLoopMiddleware`` decision (``{"type": "approve"}``,
    ``"reject"`` with an optional ``message``, ``"edit"`` with
    ``edited_action``, ``"respond"`` with ``message``); the batched
    ``{"decisions": [<one decision>]}``; ``True`` / ``False``; or the strings
    ``"approve"`` / ``"reject"``.
    """
    if answer is True:
        return "approve", {}
    if isinstance(answer, str) and answer in ("approve", "reject"):
        return answer, {}
    if isinstance(answer, Mapping):
        decisions = answer.get("decisions")
        if decisions is not None:
            # One interrupt is one call: several decisions cannot be matched to it.
            if isinstance(decisions, (list, tuple)) and len(decisions) == 1:
                return _parse_decision(decisions[0])
            return "invalid", {}
        kind = answer.get("type")
        if kind in _APPROVAL_DECISIONS:
            return str(kind), answer
    return "invalid", {}


def _blocked(decision: ToolDecision, label: str) -> ToolDecision:
    return dataclasses.replace(decision, action=BLOCK, threats=(*decision.threats, label))


def _withheld(decision: ToolDecision) -> str:
    who = f" [{decision.agent_id}]" if decision.agent_id else ""
    found = ", ".join(decision.threats) if decision.threats else "no threats"
    return f"Result of tool {decision.tool_name!r} withheld by ADAPT-Agent{who}: {found}."


def _is_request(value: Any) -> bool:
    return hasattr(value, "tool_call") and hasattr(value, "override")


def _is_tool_message(value: Any) -> bool:
    return type(value).__name__ == "ToolMessage" and hasattr(value, "tool_call_id")


def _tool_message(request: Any, content: str, *, status: str) -> Any:
    messages = optional_import("langchain_core.messages", "langgraph", "LangGraph tool governance")
    call = request.tool_call
    return messages.ToolMessage(
        content=content, name=call["name"], tool_call_id=call.get("id") or "", status=status
    )


# -- model-call governance -------------------------------------------------------


class _ModelGovernor:
    """The ``wrap_model_call`` logic: screen input, run, screen output."""

    def __init__(
        self, gate: GovernanceGate, *, screen_output: bool, screen_stream: bool, observer: Any
    ):
        self.controls = gate
        self.screen_output = screen_output
        self.screen_stream = screen_stream
        self.observer = observer

    def wrap(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        with traced(self.observer, self.controls.agent_id, "langgraph.model_call"):
            request = self._before(request)
            return self._after(handler(request))

    async def awrap(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        with traced(self.observer, self.controls.agent_id, "langgraph.model_call"):
            request = self._before(request)
            return self._after(await handler(request))

    def _before(self, request: Any) -> Any:
        # `request.messages` is exactly what this call sends, after every
        # earlier middleware's edits -- not the state as it was at node entry.
        state = policy_state(request.messages, request.state, request.runtime)
        threats = self.controls.review_input(_screenable(request.messages), state=state)
        _record_threats(self.observer, self.controls.agent_id, "input", threats)
        if self.screen_stream:
            request = request.override(model=_without_token_stream(request.model))
        return request

    def _after(self, response: Any) -> Any:
        # Inside the model node, before the response is written to the state
        # (and checkpointed) or returned: an `after_model` hook runs a node
        # later, when `stream_mode="updates"` has already delivered it.
        if self.screen_output:
            threats = self.controls.review_output(response)
            _record_threats(self.observer, self.controls.agent_id, "output", threats)
        return response


def _screenable(messages: Any) -> Any:
    """The model request's messages as input screening should see them.

    Every message is screened whole except the model's own turns, which are
    reduced to their text. A previous ``AIMessage`` carries its tool calls'
    arguments, and those were already screened where they belong -- by
    ``check_tool_call`` before the tool ran, and as model output when produced.
    Screening them again here as *input* turned every ``"refuse"`` of a blocked
    argument into an abort on the very next model call, because the refused
    call stays in the history the model is shown.
    """
    if not isinstance(messages, (list, tuple)):
        return messages
    return [
        getattr(message, "content", "") if getattr(message, "type", None) == "ai" else message
        for message in messages
    ]


def _without_token_stream(model: Any) -> Any:
    """A copy of ``model`` whose runs carry LangGraph's ``nostream`` tag.

    Raises:
        TypeError: When the model cannot be tagged -- ``screen_stream=True``
            promised the consumer never sees a blocked response, and silently
            streaming anyway would break that promise.
    """
    tags = list(getattr(model, "tags", None) or [])
    if _TAG_NOSTREAM in tags:
        return model
    copy = getattr(model, "model_copy", None)
    if callable(copy) and hasattr(model, "tags"):
        return copy(update={"tags": [*tags, _TAG_NOSTREAM]})
    raise TypeError(
        f"screen_stream=True needs a LangChain chat model it can tag; got "
        f"{type(model).__name__}. Pass a BaseChatModel instance, or screen_stream=False."
    )


# -- public factories ------------------------------------------------------------


def governance_middleware(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    tool_policy: ToolPolicy | None = None,
    on_tool_violation: str = "raise",
    screen_output: bool = True,
    screen_stream: bool = False,
    approvals: bool = True,
    approval_args: bool = True,
    observer: Any = None,
) -> Any:
    """An ``AgentMiddleware`` applying governance inside ``create_agent``.

    Per model call (``wrap_model_call`` / ``awrap_model_call``): the request's
    messages are screened with the input controls and policy evaluated before
    the model runs; the response is screened with the output controls before it
    is written to the graph state. A block raises
    :class:`~adapt_agent.exceptions.SecurityBlockedError`, which aborts the run.

    Per tool call (``wrap_tool_call`` / ``awrap_tool_call``):

    * :meth:`GovernanceGate.check_tool_call` before the tool runs. A block
      follows ``on_tool_violation``. An ``"ask"`` raises a LangGraph
      ``interrupt()`` whose value is compatible with
      ``HumanInTheLoopMiddleware``'s (see ``_ToolGovernor._approval_request``);
      the run resumes with ``Command(resume=<decision>)`` -- ``{"type":
      "approve"}`` runs the tool, ``{"type": "reject", "message": ...}``
      returns a refusal, ``{"type": "edit", "edited_action": {...}}`` runs the
      re-screened arguments, ``{"type": "respond", "message": ...}`` answers in
      the tool's place. Any other answer is a rejection. Without a checkpointer
      (or with ``approvals=False``) an ask fails closed to a block.
    * :meth:`GovernanceGate.check_tool_result` after it returns; a blocked
      result is replaced by a placeholder (``"refuse"``) or aborts the run
      (``"raise"``), so it never reaches the next model call.

    Tool calls from MCP servers go through the same ``ToolNode`` and are
    governed identically; there is no separate MCP path in ``create_agent``.

    Args:
        gate: A pre-built gate to share across agents; otherwise built from the
            controls below.
        firewall: Screens model input/output and tool arguments/results.
        defense: Adversarial analysis of model input, tool arguments and results.
        policy_enforcer: Evaluated against :func:`policy_state` -- the graph
            state, the runtime context and the messages.
        block_on_violation: ``False`` reports (to ``observer``) without blocking.
        agent_id: Names this agent in errors, refusals and traces.
        tool_policy: Name-based allow / deny / approval lists for tools.
        on_tool_violation: ``"raise"`` (default) aborts the run with
            ``SecurityBlockedError``; ``"refuse"`` returns an error
            ``ToolMessage`` carrying :attr:`ToolDecision.reason` so the model
            can carry on without the tool. ``"raise"`` is the default because it
            matches the model-call screens (which always raise) and because a
            refusal hands a possibly-injected model another turn to try a
            different route. Note: ``create_agent``'s ``ToolNode`` re-raises it;
            a hand-built ``ToolNode(handle_tool_errors=True)`` would convert it
            into an error message, i.e. a refusal.
        screen_output: Also screen each model response.
        screen_stream: Hold model tokens back from ``stream_mode="messages"``
            until the whole response has passed output screening; see the
            module docstring for the latency trade-off.
        approvals: Map ``"ask"`` to ``interrupt()`` when the graph has a
            checkpointer. ``False`` treats every ask as a block.
        approval_args: Include the tool's arguments in the approval request.
        observer: An :class:`~adapt_agent.observability.AgentObserver`; records
            a span per governed model call and tool call, and logs every tool
            decision that found anything (report-only ones included).

    Returns:
        An ``AgentMiddleware`` instance for ``create_agent(middleware=[...])``.

    Raises:
        ImportError: If ``langchain`` is not installed.
        ValueError: If ``on_tool_violation`` is not ``"raise"`` or ``"refuse"``.
    """
    _check_mode(on_tool_violation)
    middleware_module = optional_import(
        "langchain.agents.middleware", "langgraph", "governance_middleware"
    )
    resolved = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
        tool_policy=tool_policy,
    )
    tools = _ToolGovernor(
        resolved,
        on_tool_violation=on_tool_violation,
        approvals=approvals,
        approval_args=approval_args,
        observer=observer,
    )
    models = _ModelGovernor(
        resolved, screen_output=screen_output, screen_stream=screen_stream, observer=observer
    )

    base: Any = middleware_module.AgentMiddleware  # resolved at call time, never at import

    class AdaptGovernanceMiddleware(base):
        """ADAPT-Agent governance for one ``create_agent`` agent."""

        def __init__(self) -> None:
            super().__init__()
            self.gate = resolved
            self.on_tool_violation = on_tool_violation

        @property
        def name(self) -> str:
            # `create_agent` refuses two middleware with one name; a supervisor
            # and its sub-agents are separate agents, but one agent may stack
            # differently-labelled gates.
            return f"AdaptGovernanceMiddleware[{resolved.agent_id}]"

        def wrap_model_call(self, request: Any, handler: Any) -> Any:
            return models.wrap(request, handler)

        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            return await models.awrap(request, handler)

        def wrap_tool_call(self, request: Any, handler: Any) -> Any:
            return tools.wrap(request, handler)

        async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
            return await tools.awrap(request, handler)

    return AdaptGovernanceMiddleware()


def governance_tool_wrapper(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    tool_policy: ToolPolicy | None = None,
    on_tool_violation: str = "raise",
    approvals: bool = True,
    approval_args: bool = True,
    observer: Any = None,
) -> dict[str, Callable[..., Any]]:
    """Tool-call governance for a hand-built graph's ``ToolNode``.

    The same checks, approvals and result screening as
    :func:`governance_middleware`'s tool half::

        tools_node = ToolNode(tools, **governance_tool_wrapper(tool_policy=policy))

    Also the way to govern the tools of a (deprecated) ``create_react_agent``:
    pass it ``tools=ToolNode(tools, **governance_tool_wrapper(...))``.

    ``on_tool_violation="raise"`` needs the ``ToolNode``'s default
    ``handle_tool_errors``: ``handle_tool_errors=True`` turns the raised
    :class:`~adapt_agent.exceptions.SecurityBlockedError` into an error
    ``ToolMessage`` -- the tool still does not run, but the run continues as if
    ``"refuse"`` had been chosen.

    Returns:
        ``{"wrap_tool_call": ..., "awrap_tool_call": ...}`` -- the two
        ``ToolNode`` keyword arguments, sync and async.

    Raises:
        ValueError: If ``on_tool_violation`` is not ``"raise"`` or ``"refuse"``.
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
    governor = _ToolGovernor(
        resolved,
        on_tool_violation=on_tool_violation,
        approvals=approvals,
        approval_args=approval_args,
        observer=observer,
    )

    def wrap_tool_call(request: Any, handler: Callable[[Any], Any]) -> Any:
        return governor.wrap(request, handler)

    async def awrap_tool_call(request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        return await governor.awrap(request, handler)

    return {"wrap_tool_call": wrap_tool_call, "awrap_tool_call": awrap_tool_call}


def governance_callback_handler(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    tool_policy: ToolPolicy | None = None,
    screen_output: bool = True,
    observer: Any = None,
) -> Any:
    """A run-level net: a callback handler that aborts a run on any violation.

    LangChain callbacks passed in ``config={"callbacks": [handler]}`` propagate
    to every runnable the run starts -- each agent of a supervisor, a sub-agent
    called as a tool, a nested graph -- whether or not that agent carries the
    middleware. This is the seam for "govern everything this run does".

    It is deliberately abort-only. A callback cannot replace a tool's result or
    pause for a human, so a blocked or ``"ask"`` tool call (an ask fails closed
    to a block), a blocked tool result, a blocked model input or output all
    raise :class:`~adapt_agent.exceptions.SecurityBlockedError` (the handler sets
    ``raise_error = True``; without it LangChain logs and continues). Use the
    middleware where refusals or approvals are wanted, and this handler as a
    backstop.

    A tool result is screened in ``on_tool_end``, after the tool ran but before
    the result returns to the graph. Policy rules see ``messages`` and
    ``context`` (``{}``) only: a callback receives no graph state.

    Returns:
        A ``langchain_core`` ``BaseCallbackHandler``.

    Raises:
        ImportError: If ``langchain-core`` is not installed.
    """
    callbacks = optional_import(
        "langchain_core.callbacks", "langgraph", "governance_callback_handler"
    )
    resolved = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
        tool_policy=tool_policy,
    )
    tool_names: dict[Any, str] = {}

    base: Any = callbacks.BaseCallbackHandler

    class AdaptGovernanceCallbackHandler(base):
        raise_error = True  # an exception here must abort, not be logged and ignored
        run_inline = True  # screen in order, on the run's own thread

        def on_chat_model_start(
            self, serialized: Any, messages: Any, *, run_id: Any = None, **kwargs: Any
        ) -> None:
            batch = messages[0] if messages and isinstance(messages[0], list) else messages
            threats = resolved.review_input(batch, state=as_state(batch))
            _record_threats(observer, resolved.agent_id, "input", threats)

        def on_llm_end(self, response: Any, *, run_id: Any = None, **kwargs: Any) -> None:
            if screen_output:
                threats = resolved.review_output(getattr(response, "generations", response))
                _record_threats(observer, resolved.agent_id, "output", threats)

        def on_tool_start(
            self,
            serialized: Any,
            input_str: str,
            *,
            run_id: Any = None,
            inputs: Any = None,
            **kwargs: Any,
        ) -> None:
            name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
            tool_names[run_id] = name
            arguments = inputs if inputs is not None else input_str
            decision = resolved.check_tool_call(name, arguments, state=as_state([]))
            _record(observer, decision, "call")
            if decision.resolve(approvals_supported=False) == BLOCK:
                tool_names.pop(run_id, None)
                raise SecurityBlockedError(
                    dataclasses.replace(decision, action=BLOCK).reason, list(decision.threats)
                )

        def on_tool_end(self, output: Any, *, run_id: Any = None, **kwargs: Any) -> None:
            name = tool_names.pop(run_id, None) or kwargs.get("name") or "tool"
            content = getattr(output, "content", output)
            decision = resolved.check_tool_result(name, content)
            _record(observer, decision, "result")
            if decision.blocked:
                raise SecurityBlockedError(_withheld(decision), list(decision.threats))

        def on_tool_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
            tool_names.pop(run_id, None)

    return AdaptGovernanceCallbackHandler()


def governance_hooks(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    screen_output: bool = True,
    tool_policy: ToolPolicy | None = None,
) -> dict[str, Callable[[Any], dict[str, Any]]]:
    """``pre_model_hook`` / ``post_model_hook`` nodes for ``create_react_agent``.

    **Deprecated seam.** ``langgraph.prebuilt.create_react_agent`` is
    deprecated upstream (moved to ``langchain.agents.create_agent``, to be
    removed in LangGraph 2.0). ``create_agent`` has no hook parameters and
    raises ``TypeError: create_agent() got an unexpected keyword argument
    'pre_model_hook'`` if given these -- use :func:`governance_middleware`
    there. These hooks screen model calls only; for ``create_react_agent``'s
    tools pass ``tools=ToolNode(tools, **governance_tool_wrapper(...))``.

    Args:
        gate: A pre-built gate to share across nodes; otherwise built from the
            controls below.
        firewall: Screens the state's messages before and after the model call.
        defense: Adversarial analysis of the inbound messages.
        policy_enforcer: Evaluated against :func:`policy_state` of the graph
            state, so a rule may reference any key the graph carries.
        block_on_violation: ``False`` scans without raising.
        agent_id: Named in the raised error.
        screen_output: Also register a ``post_model_hook``.
        tool_policy: Carried on the gate for symmetry with the other factories;
            these hooks do not see tool calls (see above).

    Returns:
        A dict of ``create_react_agent(...)`` kwargs.
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

    hooks: dict[str, Callable[[Any], dict[str, Any]]] = {
        "pre_model_hook": governance_node(resolved, direction="input")
    }
    if screen_output:
        hooks["post_model_hook"] = governance_node(resolved, direction="output")
    return hooks


def governance_node(
    gate: GovernanceGate, *, direction: str = "input"
) -> Callable[[Any], dict[str, Any]]:
    """Return a LangGraph node that screens the state and returns no update.

    Usable directly in a hand-built graph::

        builder.add_node("guard", governance_node(gate))
        builder.add_edge("guard", "model")

    Also what :func:`governance_hooks` hands the deprecated
    ``create_react_agent``. A node sees whole states, not individual model or
    tool calls; in a ``create_agent`` graph use :func:`governance_middleware`.

    Args:
        gate: The configured :class:`~adapt_agent.core.governance.GovernanceGate`.
        direction: ``"input"`` screens with the input controls and evaluates
            policy against :func:`policy_state` of the node's state;
            ``"output"`` screens with the output controls only.

    Raises:
        ValueError: If ``direction`` is not ``"input"`` or ``"output"``.
    """
    if direction not in ("input", "output"):
        raise ValueError(f"direction must be 'input' or 'output', got {direction!r}")

    def _node(state: Any) -> dict[str, Any]:
        messages = _state_messages(state)
        if direction == "input":
            # Always a state: a Pydantic/dataclass graph state used to get
            # `None` here, which left a configured policy silently inert.
            gate.review_input(messages, state=policy_state(messages, state))
        else:
            gate.review_output(messages)
        return {}  # governance never rewrites state

    _node.__name__ = f"adapt_governance_{direction}"
    return _node


__all__ = [
    "governance_callback_handler",
    "governance_hooks",
    "governance_middleware",
    "governance_node",
    "governance_tool_wrapper",
    "policy_state",
]
