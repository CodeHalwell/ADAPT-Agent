"""Governance as Microsoft Agent Framework middleware.

Microsoft Agent Framework (MAF) has three middleware seams, and ADAPT-Agent binds
all three:

* **agent middleware** ``async (AgentContext, call_next)`` -- once per
  ``agent.run``: the caller's messages in, the final response out;
* **chat middleware** ``async (ChatContext, call_next)`` -- once per *model
  call*, inside the tool loop: everything the model is about to read (history,
  context-provider content, tool results) and everything it answers;
* **function middleware** ``async (FunctionInvocationContext, call_next)`` --
  once per *tool call*, MCP tools included: the name and arguments before the
  tool runs, its result before the model sees it.

The recommended entry point is :func:`governance_bundle`, which returns all
three as one MAF ``MiddlewareBundle``:

.. code-block:: python

    from agent_framework import Agent
    from adapt_agent.core.governance import ToolPolicy
    from adapt_agent.integrations.agent_framework import governance_bundle

    agent = Agent(
        client,
        "You are an operations assistant.",
        tools=[shell, fetch],
        middleware=[governance_bundle(firewall=fw, tool_policy=ToolPolicy(deny=["shell"]))],
    )

The same bundle works per run (``agent.run(..., middleware=[bundle])``) and on
every agent inside a workflow -- including ``WorkflowBuilder(...).build()
.as_agent()``, where wrapping the outer agent with the adapter would govern only
the outermost boundary. :func:`governance_middleware`, the original factory, is
a single agent middleware that installs the chat and function layers into each
run it governs, so existing code gains tool governance without changes.

What each layer sees (verified against the installed release):

* The agent layer sees only the caller's new messages -- not session history,
  not context-provider content. It evaluates the ``policy_enforcer`` (once per
  run) and screens the final response.
* The chat layer sees the full request sent to the model, including messages a
  ``ContextProvider`` injected (RAG, memory) and the tool results of the
  previous step. Screening the response *here* matters for sessions: MAF
  persists a response to the session's history before the agent middleware
  regains control, so a response blocked only at the agent layer still reached
  the next turn's model call. A block at the chat layer aborts the run before
  that persistence happens.
* The function layer is the only one that can stop a tool from running.

Tool decisions come from :meth:`GovernanceGate.check_tool_call
<adapt_agent.core.governance.GovernanceGate.check_tool_call>` and
:meth:`~adapt_agent.core.governance.GovernanceGate.check_tool_result`:

* **block**, ``on_tool_violation="refuse"`` (the default): the tool does not
  run; the model receives ``decision.reason`` as the tool's result and the run
  continues. This is MAF's own convention for a failed tool and lets the model
  explain the refusal instead of the whole run failing.
* **block**, ``on_tool_violation="raise"``: the run aborts. An ordinary
  exception raised in MAF function middleware is converted into a tool error
  *and fed back to the model* -- the run carries on -- so this raises an error
  that is both a MAF ``MiddlewareFailure`` (the loop's fail-closed escape) and a
  :class:`~adapt_agent.exceptions.SecurityBlockedError` carrying the threats,
  which is what reaches the caller of ``agent.run``.
* a **blocked result** is replaced by a content-free placeholder (or aborts, in
  raise mode) before the next model call.
* **ask** becomes a MAF ``function_approval_request``: the run pauses, the
  request appears on ``AgentResponse.user_input_requests``, and the caller
  resumes with ``request.to_function_approval_response(approved)`` on the *same*
  ``AgentSession``. MAF binds the response to the request recorded in that
  session (so the arguments cannot be swapped), and the gate re-screens the
  arguments on the approved replay. Without a session MAF cannot bind -- and
  ignores -- approval responses, so an ask there fails closed to a block, as it
  does with ``approvals=False``.

Streaming: with ``stream=True`` the response is a ``ResponseStream``, and the
output checks ride its gates. ``buffer_streams=True`` (default) holds every
update until the response has passed screening, so a blocked response is never
delivered -- at the cost of time-to-first-token. ``buffer_streams=False``
screens the accumulated text as each update arrives and stops the stream at the
update that completes a match; the text *before* that update has already been
delivered.

Being async-native, these hooks never touch the sync/async bridge ``execute``
needs, so they work unchanged inside a running event loop.
"""

from __future__ import annotations

import contextlib
import fnmatch
import uuid
import warnings
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from adapt_agent.core.governance import ALLOW, ASK, BLOCK, GovernanceGate, ToolDecision, ToolPolicy
from adapt_agent.exceptions import SecurityBlockedError
from adapt_agent.integrations._common import as_state, build_gate, context_state, traced

if TYPE_CHECKING:  # pragma: no cover - typing only
    from adapt_agent.adversarial import AdversarialDefense
    from adapt_agent.core.policy import PolicyEnforcer
    from adapt_agent.security.firewall import Firewall

#: ``on_tool_violation`` values.
_TOOL_VIOLATION_MODES = ("refuse", "raise")

#: Bounded walk over MAF ``Content`` trees (a function result can hold content
#: that holds content). Deep enough for any real payload; a cyclic or
#: pathological one stops here rather than recursing forever.
_MAX_CONTENT_DEPTH = 8

#: Message roles whose content the chat layer screens as model *input*.
#: ``assistant`` is the model's own earlier output (screened as output when it
#: was produced) and ``system`` is the developer's instructions -- screening
#: those with input rules mostly yields false positives on the app's own prompt.
_UNSCREENED_INPUT_ROLES = frozenset({"assistant", "system"})


# -- shared helpers -----------------------------------------------------------


def _role_of(message: Any) -> str:
    role = getattr(message, "role", None)
    role = getattr(role, "value", role)  # older releases used a Role enum
    return role.lower() if isinstance(role, str) else ""


def _content_payload(value: Any, depth: int = 0) -> list[Any]:
    """Flatten MAF ``Message`` / ``Content`` objects into screenable values.

    The gate's generic extractor reads ``Message.text`` but not the
    ``contents`` list behind it, so a ``function_result`` -- the tool output
    that is the classic indirect-injection carrier -- would go unscreened. This
    walks the content tree explicitly: text, function results, function-call
    arguments, and nested content. Anything unrecognised is passed through for
    the gate's own structural extraction, so nothing is silently dropped.
    """
    if depth > _MAX_CONTENT_DEPTH or value is None:
        return []
    if isinstance(value, (str, bytes, int, float, bool, Mapping)):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            out.extend(_content_payload(item, depth + 1))
        return out
    contents = getattr(value, "contents", None)
    if isinstance(contents, (list, tuple)):  # a Message
        return _content_payload(list(contents), depth + 1)
    kind = getattr(value, "type", None)
    if isinstance(kind, str):  # a Content
        out = []
        for attr in ("text", "result", "arguments", "function_call", "items"):
            inner = getattr(value, attr, None)
            if inner is not None:
                out.extend(_content_payload(inner, depth + 1))
        return out
    return [value]


#: Content kinds that are a *request to run a tool*, not something the agent
#: said. Their arguments are judged at the tool seam, which can refuse the one
#: call and let the run continue.
_TOOL_REQUEST_KINDS = frozenset(
    {"function_call", "function_approval_request", "function_approval_response"}
)


def _output_payload(result: Any, depth: int = 0) -> list[Any]:
    """The screenable content of a model or agent response, minus tool requests.

    Output screening judges what the agent *says*. A function call in the
    response is the model asking for a tool; the function layer screens its
    arguments and applies ``on_tool_violation``. Screening the same arguments
    here as output would abort the run before that refusal could be returned
    to the model, which is the difference between the two modes.

    Everything else is kept -- text, reasoning, tool results already in the
    transcript, a structured ``value`` -- and anything unrecognised is passed
    through whole for the gate's own extraction, so nothing is dropped by
    omission.
    """
    if depth > _MAX_CONTENT_DEPTH or result is None:
        return []
    if isinstance(result, (str, bytes, int, float, bool, Mapping)):
        return [result]
    if isinstance(result, (list, tuple)):
        out: list[Any] = []
        for item in result:
            out.extend(_output_payload(item, depth + 1))
        return out
    messages = getattr(result, "messages", None)
    if isinstance(messages, (list, tuple)):  # an AgentResponse / ChatResponse
        out = _output_payload(list(messages), depth + 1)
        value = getattr(result, "value", None)  # structured output, when present
        if value is not None and not isinstance(value, str):
            out.append(value)
        return out
    contents = getattr(result, "contents", None)
    if isinstance(contents, (list, tuple)):  # a Message
        return _output_payload(list(contents), depth + 1)
    kind = getattr(result, "type", None)
    if isinstance(kind, str):  # a Content
        if kind in _TOOL_REQUEST_KINDS:
            return []
        return _content_payload(result, depth)
    return [result]


def _review_output(gate: GovernanceGate) -> Callable[[Any], Any]:
    """``gate.review_output`` over :func:`_output_payload` of its argument."""

    def review(result: Any) -> Any:
        return gate.review_output(_output_payload(result))

    return review


def _arguments_dict(arguments: Any) -> Any:
    """Return tool arguments as a plain mapping where possible.

    MAF hands middleware either a mapping or a pydantic model; the gate scans
    either, but a ``ToolPolicy``/policy rule reads ``arguments`` as a mapping.
    """
    dump = getattr(arguments, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            return arguments
    if isinstance(arguments, Mapping):
        return dict(arguments)
    return arguments


def _blocked_error(reason: str, threats: Iterable[str]) -> SecurityBlockedError:
    """A ``SecurityBlockedError`` that MAF's tool loop will not swallow.

    MAF converts an ordinary exception raised in function middleware into a
    tool-error result and lets the model carry on -- fail-open for a security
    control. ``MiddlewareFailure`` is the loop's one fail-closed escape. The
    returned error is both, so the loop aborts and the caller can still
    ``except SecurityBlockedError`` and read ``.threats``. Without the SDK
    (unit tests) a plain ``SecurityBlockedError`` is returned.
    """
    cls = _maf_blocked_error_class()
    return cls(reason, list(threats))


_BLOCKED_ERROR_CLASS: type[SecurityBlockedError] | None = None


def _maf_blocked_error_class() -> type[SecurityBlockedError]:
    global _BLOCKED_ERROR_CLASS
    if _BLOCKED_ERROR_CLASS is None:
        try:
            from agent_framework import MiddlewareFailure
        except Exception:  # pragma: no cover - exercised without the SDK
            return SecurityBlockedError

        class ToolCallBlockedError(SecurityBlockedError, MiddlewareFailure):
            """A tool-seam block: a ``SecurityBlockedError`` and a ``MiddlewareFailure``."""

        _BLOCKED_ERROR_CLASS = ToolCallBlockedError
    return _BLOCKED_ERROR_CLASS


def _check_mode(on_tool_violation: str) -> None:
    if on_tool_violation not in _TOOL_VIOLATION_MODES:
        raise ValueError(
            f"on_tool_violation must be one of {_TOOL_VIOLATION_MODES}, got {on_tool_violation!r}"
        )


def _mark(
    decorator_name: str, func: Callable[..., Awaitable[None]]
) -> Callable[..., Awaitable[None]]:
    """Tag ``func`` with MAF's own middleware decorator when MAF is importable.

    MAF decides whether a bare callable is agent/chat/function middleware from
    either its decorator marker or the *name* of its first parameter's
    annotation. A duck-typed integration has neither, so without this tag MAF
    raises ``MiddlewareException: Cannot determine middleware type``. Applying
    the real decorator is exact; the import is deferred and optional, which
    keeps this module importable with no SDK installed.
    """
    try:
        import agent_framework
    except Exception:  # pragma: no cover - exercised without the SDK
        return func
    decorator = getattr(agent_framework, decorator_name, None)
    if decorator is None:  # pragma: no cover - a release without this seam
        return func
    return decorator(func)  # type: ignore[no-any-return]


class _Span:
    """An observer trace that can be closed after ``call_next`` returns, exactly once.

    For a streamed run ``call_next()`` returns before a single update has been
    produced -- the response is consumed later, by the caller. Closing the span
    when ``call_next`` returns recorded a completed run for output that had not
    yet been screened, so a streamed span stays open until consumption ends.
    """

    def __init__(self, observer: Any, agent_id: str, operation: str) -> None:
        self._observer = observer
        self._trace_id = uuid.uuid4().hex
        self._open = observer is not None
        if self._open:
            observer.start_trace(self._trace_id, agent_id, operation)

    def end(self, status: str, result: Any = None) -> None:
        if self._open:
            self._open = False
            if result is None:
                self._observer.end_trace(self._trace_id, status=status)
            else:
                self._observer.end_trace(self._trace_id, status=status, result=result)


def _screen_stream_output(
    context: Any,
    review: Callable[[Any], Any],
    *,
    buffer_streams: bool,
    span: _Span,
) -> None:
    """Attach output screening to a streamed MAF response.

    Uses the context's gate lists, which MAF applies to the ``ResponseStream``
    after the middleware chain unwinds. Gates must return ``None`` (or raise);
    a raised error surfaces to the consumer when the stream is iterated.

    The span is closed by a consumption context manager rather than by the
    gate: when an *inner* layer (the chat middleware, or the model itself)
    raises, MAF never runs this layer's gates, and a span closed only there
    stayed open forever.
    """
    if not hasattr(context, "stream_result_gates_after"):
        # An older release with no stream gates: the response would reach the
        # consumer unscreened. Refuse rather than pretend.
        span.end("error", "stream output cannot be screened on this release")
        raise SecurityBlockedError(
            "Streamed output cannot be screened on this agent_framework release; "
            "upgrade agent-framework-core or pass screen_output=False explicitly",
            ["stream:unscreenable"],
        )

    factories = getattr(context, "stream_consumption_context_manager_factories", None)
    if isinstance(factories, list):

        @contextlib.contextmanager
        def _span_scope() -> Iterator[None]:
            try:
                yield
            except BaseException as exc:
                span.end("error", str(exc))
                raise
            span.end("completed")

        factories.append(_span_scope)

    if buffer_streams:
        context.stream_buffer_updates = True

        def _result_gate(final: Any) -> None:
            review(final)

        context.stream_result_gates_after.append(_result_gate)
        return

    seen: list[str] = []

    def _update_gate(update: Any) -> None:
        text = getattr(update, "text", None)
        if isinstance(text, str) and text:
            seen.append(text)
        review("".join(seen))

    context.stream_update_gates_after.append(_update_gate)


# -- the three middleware -----------------------------------------------------


def governance_function_middleware(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    tool_policy: ToolPolicy | None = None,
    on_tool_violation: str = "refuse",
    approvals: bool = True,
    observer: Any = None,
    operation: str = "agent_framework.tool",
) -> Callable[..., Awaitable[None]]:
    """Build MAF *function* middleware: governs every tool call, MCP tools included.

    Before the tool runs the gate's :meth:`check_tool_call` decides on the name,
    the arguments and any tool-scoped policy rule; after it returns,
    :meth:`check_tool_result` screens the result before the model sees it. See
    the module docstring for what ``refuse``, ``raise`` and an ``ask`` do.

    Install it last among your function middleware (or let
    :func:`governance_middleware` install it for you, which places it
    innermost). MAF runs function middleware outermost-first, and an
    argument-repair middleware that runs *after* this one could change the
    arguments the gate has already approved.

    Policy rules evaluated here see ``state["tool"]`` (``{"name", "arguments"}``)
    plus the run's ``AgentSession.state`` keys and an empty ``messages`` list.

    Args:
        on_tool_violation: ``"refuse"`` (default) or ``"raise"``.
        approvals: Map an ``"ask"`` onto MAF's approval flow. ``False`` treats
            every ask as a block.
        observer: Records one span per governed tool call.
        operation: Span label.
    """
    _check_mode(on_tool_violation)
    resolved = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
        tool_policy=tool_policy,
    )

    def _refuse_or_raise(context: Any, decision: ToolDecision, refusal: str) -> None:
        if on_tool_violation == "raise":
            raise _blocked_error(refusal, decision.threats)
        context.result = refusal

    async def adapt_tool_governance(context: Any, call_next: Callable[[], Awaitable[None]]) -> None:
        function = getattr(context, "function", None)
        name = str(getattr(function, "name", None) or "unknown")
        arguments = _arguments_dict(getattr(context, "arguments", None))
        state = as_state([], **context_state(context))
        with traced(observer, resolved.agent_id, operation):
            decision = resolved.check_tool_call(name, arguments, state=state)
            action = decision.action
            if action == ASK:
                action = _resolve_ask(context, decision, name, approvals=approvals)
                if action == ASK:
                    return  # an approval request is now the call's result
            if action == BLOCK:
                _refuse_or_raise(context, decision, decision.reason)
                return

            await call_next()

            result_decision = resolved.check_tool_result(
                name, _content_payload(getattr(context, "result", None)), state=state
            )
            if result_decision.blocked:
                _refuse_or_raise(context, result_decision, _withheld(result_decision))

    return _mark("function_middleware", adapt_tool_governance)


def _withheld(decision: ToolDecision) -> str:
    """Placeholder that replaces a blocked tool result. Labels only, no content."""
    who = f" [{decision.agent_id}]" if decision.agent_id else ""
    found = ", ".join(decision.threats) or "policy"
    return f"Tool result from {decision.tool_name!r} withheld by ADAPT-Agent{who}: {found}."


def _approved_replay(context: Any, name: str) -> bool:
    """True when this invocation is MAF replaying a call a human approved.

    MAF puts the approval response it bound to this call on
    ``context.metadata["approval_response"]``. With an ``AgentSession`` that
    response has already been matched against the request MAF recorded --
    arguments included -- so it cannot be forged by editing the resume message.
    The name check is belt and braces: an approval of one tool never runs
    another.
    """
    metadata = getattr(context, "metadata", None)
    response = metadata.get("approval_response") if isinstance(metadata, Mapping) else None
    if response is None or getattr(response, "type", None) != "function_approval_response":
        return False
    if getattr(response, "approved", None) is not True:
        return False
    call = getattr(response, "function_call", None)
    return getattr(call, "name", None) == name


def _resolve_ask(context: Any, decision: ToolDecision, name: str, *, approvals: bool) -> str:
    """Turn an ``"ask"`` into ``allow`` (approved replay), ``block``, or a pending request."""
    if _approved_replay(context, name):
        return ALLOW
    metadata = getattr(context, "metadata", None)
    call_id = metadata.get("call_id") if isinstance(metadata, Mapping) else None
    # No session: MAF ignores approval responses it cannot bind to a recorded
    # request, so the run could never resume -- refuse now instead of pausing
    # forever. No call id: invoked outside MAF's tool loop, nothing to pause.
    can_pause = approvals and getattr(context, "session", None) is not None and bool(call_id)
    if not can_pause:
        return decision.resolve(approvals_supported=False)
    try:
        from agent_framework import Content
    except Exception:  # pragma: no cover - exercised without the SDK
        return decision.resolve(approvals_supported=False)
    occurrence = metadata.get("function_call_occurrence_id") if metadata else None
    call = Content.from_function_call(
        call_id=str(call_id),
        name=name,
        arguments=_arguments_dict(getattr(context, "arguments", None)),
        id=occurrence,
    )
    context.result = Content.from_function_approval_request(
        id=str(occurrence or call_id),
        function_call=call,
        additional_properties={
            "reason": decision.reason,
            "adapt_agent_threats": list(decision.threats),
        },
    )
    return ASK


def governance_chat_middleware(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    tool_policy: ToolPolicy | None = None,
    screen_output: bool = True,
    buffer_streams: bool = True,
    observer: Any = None,
    operation: str = "agent_framework.model_call",
) -> Callable[..., Awaitable[None]]:
    """Build MAF *chat* middleware: screens every model call inside the tool loop.

    Input: every non-``assistant``, non-``system`` message in the request --
    the user's turn, session history, ``ContextProvider`` content and tool
    results -- is screened with the firewall and adversarial defense. Output:
    each model response is screened before MAF persists it to the session or
    acts on it (streamed responses through the stream's gates).

    Policy rules are *not* evaluated here; they run once per run in the agent
    layer, so a ``warn``/``log`` rule is not re-fired on every tool step.
    ``tool_policy`` is accepted for a uniform signature and has no effect on
    this layer (tool calls are the function layer's job).

    Raises:
        SecurityBlockedError: From the model call, when a blocking gate finds a
            threat. MAF propagates chat-middleware exceptions to the caller.
    """
    model_gate = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
        tool_policy=tool_policy,
    )

    async def adapt_model_call_governance(
        context: Any, call_next: Callable[[], Awaitable[None]]
    ) -> None:
        messages = list(getattr(context, "messages", []) or [])
        screened = [m for m in messages if _role_of(m) not in _UNSCREENED_INPUT_ROLES]
        threats = model_gate.scan_input(_content_payload(screened))
        if threats and model_gate.block_on_violation:
            raise SecurityBlockedError(
                model_gate._reason("Model input blocked by security controls"), threats
            )

        streaming = bool(getattr(context, "stream", False))
        span = _Span(observer, model_gate.agent_id, operation)
        try:
            await call_next()
        except BaseException as exc:
            span.end("error", str(exc))
            raise
        if not screen_output:
            span.end("completed")
            return
        if streaming:
            _screen_stream_output(
                context,
                _review_output(model_gate),
                buffer_streams=buffer_streams,
                span=span,
            )
            return
        try:
            # The response's *text*. Function-call arguments in it are the tool
            # layer's to judge -- screening them here aborted the run before a
            # refusal could be returned to the model.
            model_gate.review_output(_output_payload(getattr(context, "result", None)))
        except BaseException as exc:
            span.end("error", str(exc))
            raise
        span.end("completed")

    return _mark("chat_middleware", adapt_model_call_governance)


def governance_middleware(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    screen_output: bool = True,
    observer: Any = None,
    operation: str = "agent_framework.run",
    tool_policy: ToolPolicy | None = None,
    on_tool_violation: str = "refuse",
    approvals: bool = True,
    govern_tools: bool = True,
    screen_model_calls: bool = True,
    buffer_streams: bool = True,
) -> Callable[..., Awaitable[None]]:
    """Build MAF agent middleware that applies ADAPT-Agent governance.

    On its own this governs the whole run: it screens the caller's messages and
    evaluates policy, screens the final response (streamed responses through
    the stream's gates), and -- unless ``govern_tools`` / ``screen_model_calls``
    are turned off -- installs :func:`governance_function_middleware` and
    :func:`governance_chat_middleware` into each run it governs, innermost, by
    appending them to the run's ``client_kwargs["middleware"]`` (the list MAF
    hands the chat client). Use :func:`governance_bundle` instead when you want
    the three layers as one explicit ``MiddlewareBundle``.

    Policy rules evaluated here see ``state["messages"]`` (the caller's new
    messages for this run) and ``state["context"]``, plus every key of the
    run's ``AgentSession.state`` -- the place to keep a ``trust_score`` or an
    account flag a rule gates on.

    Args:
        gate: A pre-built :class:`~adapt_agent.core.governance.GovernanceGate`
            to share across agents. When omitted one is built from the controls
            below.
        firewall: Screens the inbound messages and the outbound response.
        defense: Adversarial/prompt-injection analysis of inbound messages.
        policy_enforcer: Evaluated against the state described above. Content
            rules belong on the ``firewall``; policy rules gate on state.
        block_on_violation: ``True`` (default) raises
            :class:`~adapt_agent.exceptions.SecurityBlockedError` on a hit.
            ``False`` runs the agent anyway -- useful to shake out false
            positives before enforcing.
        agent_id: Named in the raised error, so a refusal inside a workflow
            identifies which agent produced it.
        screen_output: Screen the response (final and per model call).
        observer: Optional :class:`~adapt_agent.observability.AgentObserver`.
            Records a span per run, per model call and per tool call.
        operation: Trace operation label for the run span.
        tool_policy: Name-based allow/deny/approval lists for tool calls.
        on_tool_violation: ``"refuse"`` (default) or ``"raise"``; see the
            module docstring.
        approvals: Map a tool ``"ask"`` onto MAF's approval flow (needs an
            ``AgentSession``); ``False`` treats every ask as a block.
        govern_tools: Install the function (tool) layer into each run.
        screen_model_calls: Install the chat (per model call) layer.
        buffer_streams: Hold streamed updates until the response has passed
            screening (default). ``False`` screens incrementally; see the
            module docstring for what that can deliver.

    Returns:
        An async ``(context, call_next)`` callable for ``Agent(middleware=[...])``
        or ``agent.run(..., middleware=[...])``. When ``agent_framework`` is
        installed it is tagged with ``@agent_middleware`` so MAF categorises it
        without a type annotation; without the SDK the plain callable is
        returned and remains directly unit-testable.
    """
    _check_mode(on_tool_violation)
    resolved = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
        tool_policy=tool_policy,
    )
    inner: list[Callable[..., Awaitable[None]]] = []
    if govern_tools:
        inner.append(
            governance_function_middleware(
                gate=resolved,
                on_tool_violation=on_tool_violation,
                approvals=approvals,
                observer=observer,
            )
        )
    if screen_model_calls:
        inner.append(
            governance_chat_middleware(
                gate=resolved,
                screen_output=screen_output,
                buffer_streams=buffer_streams,
                observer=observer,
            )
        )
    return _agent_middleware(
        resolved,
        screen_output=screen_output,
        buffer_streams=buffer_streams,
        observer=observer,
        operation=operation,
        inject=inner,
    )


def _agent_middleware(
    resolved: GovernanceGate,
    *,
    screen_output: bool,
    buffer_streams: bool,
    observer: Any,
    operation: str,
    inject: list[Callable[..., Awaitable[None]]],
) -> Callable[..., Awaitable[None]]:
    async def adapt_governance(context: Any, call_next: Callable[[], Awaitable[None]]) -> None:
        # ``context.messages`` is a list[Message]. ``Message.text`` covers plain
        # turns; the explicit content walk adds what it hides (function results
        # or call arguments a caller passes in directly).
        messages = list(getattr(context, "messages", []) or [])
        # `state=` is required, or a configured policy_enforcer never runs.
        resolved.review_input(
            [messages, *_content_payload(messages)],
            state=as_state(messages, **context_state(context)),
        )
        if inject:
            client_kwargs = getattr(context, "client_kwargs", None)
            if isinstance(client_kwargs, dict):
                # A fresh list: the existing one may be shared with the agent's
                # own configuration, and this run's additions must not leak
                # into the next run.
                existing = list(client_kwargs.get("middleware") or [])
                client_kwargs["middleware"] = [*existing, *inject]

        streaming = bool(getattr(context, "stream", False))
        # `resolved.agent_id`, not the parameter: `build_gate` decides which
        # label wins when a shared gate and a binding both name themselves, and
        # the span must carry the same one the raised error does.
        span = _Span(observer, resolved.agent_id, operation)
        try:
            await call_next()
        except BaseException as exc:
            # BaseException: `asyncio.CancelledError` must close the span too.
            span.end("error", str(exc))
            raise
        if not screen_output:
            span.end("completed")
            return
        if streaming:
            _screen_stream_output(
                context,
                _review_output(resolved),
                buffer_streams=buffer_streams,
                span=span,
            )
            return
        # Inside the span, not after it. A blocked result is exactly the event
        # an observer exists to surface.
        try:
            resolved.review_output(_output_payload(getattr(context, "result", None)))
        except BaseException as exc:
            span.end("error", str(exc))
            raise
        span.end("completed")

    return _mark("agent_middleware", adapt_governance)


def governance_bundle(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    tool_policy: ToolPolicy | None = None,
    on_tool_violation: str = "refuse",
    approvals: bool = True,
    screen_output: bool = True,
    screen_model_calls: bool = True,
    buffer_streams: bool = True,
    observer: Any = None,
    operation: str = "agent_framework.run",
) -> Any:
    """All three ADAPT-Agent layers as one MAF ``MiddlewareBundle``.

    Members, in order: the agent middleware (input + policy, final output),
    the chat middleware (every model call; omitted with
    ``screen_model_calls=False``) and the function middleware (every tool
    call). A bundle cannot be installed partially -- MAF refuses to split it --
    which is the point: a binding that silently lost its tool layer would look
    configured and govern nothing that matters.

    Usable per agent (``Agent(..., middleware=[bundle])``), per run
    (``agent.run(..., middleware=[bundle])``) and on each agent of a workflow.
    Place it after any argument-repair function middleware of your own (MAF
    runs function middleware in list order, and arguments changed after the
    gate approved them would reach the tool unscreened).

    ``MiddlewareBundle`` is marked experimental in MAF (its ``AGENT_HOOKS``
    feature stage); its ``ExperimentalWarning`` is suppressed here because the
    choice to use it is ours, not the caller's. On a release without it, the
    members are returned as a plain list, which MAF categorises the same way.

    Arguments are those of :func:`governance_middleware`.
    """
    _check_mode(on_tool_violation)
    resolved = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
        tool_policy=tool_policy,
    )
    members: list[Callable[..., Awaitable[None]]] = [
        _agent_middleware(
            resolved,
            screen_output=screen_output,
            buffer_streams=buffer_streams,
            observer=observer,
            operation=operation,
            inject=[],
        )
    ]
    if screen_model_calls:
        members.append(
            governance_chat_middleware(
                gate=resolved,
                screen_output=screen_output,
                buffer_streams=buffer_streams,
                observer=observer,
            )
        )
    members.append(
        governance_function_middleware(
            gate=resolved,
            on_tool_violation=on_tool_violation,
            approvals=approvals,
            observer=observer,
        )
    )
    try:
        from agent_framework import MiddlewareBundle
    except Exception:
        return members
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*MiddlewareBundle is experimental.*")
        return MiddlewareBundle(members)


# -- MCP -----------------------------------------------------------------------

_GLOB_CHARS = frozenset("*?[")


def _is_literal(pattern: str) -> bool:
    return not (_GLOB_CHARS & set(pattern))


def mcp_tool_options(
    tool_policy: ToolPolicy,
    *,
    tool_names: Iterable[str] | None = None,
    gate: GovernanceGate | None = None,
) -> dict[str, Any]:
    """Keyword arguments that make an MAF MCP tool honour a :class:`ToolPolicy`.

    MCP tool calls already pass through function middleware (verified with a
    real ``MCPStdioTool``), so :func:`governance_function_middleware` enforces
    the policy on every call. This narrows what the model is *offered* as well:

    .. code-block:: python

        MCPStdioTool("files", command="...", **mcp_tool_options(policy, tool_names=[...]))

    * ``allowed_tools`` -- the server's tools the policy does not refuse. MAF
      matches exact names only, so a policy with glob patterns needs
      ``tool_names=`` (the server's tool names) to expand them; without it this
      raises rather than return a filter that quietly filters nothing.
    * ``approval_mode`` -- a callable MAF evaluates per discovered tool:
      ``"always_require"`` for tools the policy sends for approval.
    * ``sampling_approval_callback`` -- only with ``gate=``: approves a server's
      sampling request (the server asking *your* model to generate) only when
      the request passes the gate's input screening. MAF denies sampling by
      default; omit ``gate`` to keep it that way.

    Names are matched as MAF's MCP filters match them -- the server's raw tool
    names. With ``tool_name_prefix=`` the function middleware sees the
    *prefixed* name, so write the policy to match both, or not use a prefix.
    """
    options: dict[str, Any] = {}

    def approval_mode(item: Any) -> str:
        action, _ = tool_policy.verdict(str(getattr(item, "name", "")))
        return "always_require" if action != ALLOW else "never_require"

    options["approval_mode"] = approval_mode

    if tool_names is not None:
        names = list(tool_names)
        options["allowed_tools"] = [n for n in names if tool_policy.verdict(n)[0] != BLOCK]
    elif tool_policy.allow is not None or tool_policy.deny:
        literal = all(_is_literal(p) for p in (*(tool_policy.allow or ()), *tool_policy.deny))
        if tool_policy.allow is None or not literal:
            raise ValueError(
                "mcp_tool_options cannot turn this ToolPolicy into MAF's exact-name "
                "allowed_tools without the server's tool names: pass tool_names=[...]. "
                "governance_function_middleware enforces the policy per call either way."
            )
        options["allowed_tools"] = [
            n
            for n in tool_policy.allow
            if not any(fnmatch.fnmatchcase(n, d) for d in tool_policy.deny)
        ]

    if gate is not None:

        def sampling_approval_callback(params: Any) -> bool:
            threats = gate.scan_input(params)
            return not (threats and gate.block_on_violation)

        options["sampling_approval_callback"] = sampling_approval_callback
    return options


__all__ = [
    "governance_bundle",
    "governance_chat_middleware",
    "governance_function_middleware",
    "governance_middleware",
    "mcp_tool_options",
]
