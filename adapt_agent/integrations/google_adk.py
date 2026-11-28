"""Governance for Google ADK: a run-level plugin, or per-agent callbacks.

ADK offers two places to intercept a run, and this module binds both to the same
:class:`~adapt_agent.core.governance.GovernanceGate`:

* **A plugin** -- :func:`governance_plugin` returns a ``BasePlugin`` that you
  register once on the ``App``. ADK calls a plugin for *every* agent and *every*
  tool in the tree: sub-agents reached by ``transfer_to_agent``, agents wrapped
  in an ``AgentTool``, the nodes of a ``Workflow``, MCP tools. This is the
  binding to use: per-agent attachment silently leaves whatever you forgot to
  attach ungoverned.

  .. code-block:: python

      from google.adk.apps import App
      from google.adk.runners import InMemoryRunner
      from adapt_agent.integrations.google_adk import governance_plugin

      app = App(
          name="support",
          root_agent=router,  # its sub-agents and tools are covered too
          plugins=[governance_plugin(firewall=fw, tool_policy=policy)],
      )
      runner = InMemoryRunner(app=app)

* **Per-agent callbacks** -- :func:`governance_callbacks` returns the
  ``before_model_callback`` / ``after_model_callback`` /
  ``before_tool_callback`` / ``after_tool_callback`` kwargs for one
  ``LlmAgent``. Use them *in addition to* the plugin when one agent needs
  stricter controls than the rest (the specialist reading untrusted mail), or
  where you cannot touch the ``App``.

What each hook governs
----------------------
============================  ==================================================
Seam                          What happens
============================  ==================================================
user message (plugin only)    Screened before it enters the session. ``raise``
                              aborts the run; ``refuse`` stores a placeholder
                              instead of the message (so it cannot poison later
                              turns) and answers with the refusal.
model request                 ``llm_request.contents`` -- the whole history the
                              model is about to see, tool results included.
model response                The response content, before ADK acts on it (a
                              blocked function call is never executed).
tool call                     ``gate.check_tool_call(tool.name, args)`` before
                              the tool runs: :class:`~adapt_agent.core.governance.ToolPolicy`,
                              firewall/defense over the arguments, tool-scoped
                              policy rules.
tool result                   ``gate.check_tool_result(...)`` on what the tool
                              returned, before it reaches the next model call.
============================  ==================================================

Blocking modes
--------------
``on_block`` (model calls and the user message) and ``on_tool_violation``
(tool calls and results) each take ``"raise"`` or ``"refuse"``:

* ``on_block="raise"`` (default) aborts the run with
  :class:`~adapt_agent.exceptions.SecurityBlockedError`, matching every other
  ADAPT-Agent entry point. ``"refuse"`` returns a refusal ``LlmResponse``
  instead -- ADK's own idiom -- so one blocked branch of a tree answers with the
  refusal and the rest keeps running.
* ``on_tool_violation="refuse"`` (default) answers the call with
  ``{"error": decision.reason}``: the tool does **not** run and the model is
  told why, in a message that names the threat labels but never the content,
  so it can recover or explain. That is ADK's documented short-circuit for a
  before-tool callback, and a denied tool is routine enough (a model probing a
  tool it should not use) that aborting the whole run for it is usually wrong.
  A blocked *result* is replaced the same way, so the poisoned content never
  reaches the next model call. ``"raise"`` aborts the run instead.

ADK wraps an exception raised inside a **plugin** callback in
``RuntimeError("Error in plugin ...")``, with the original as ``__cause__``.
:class:`~adapt_agent.adapters.GoogleADKAdapter` unwraps it for you; when you
drive the ``Runner`` yourself, use :func:`reraise_security_blocks` (or
:func:`find_security_error`) to get the ``SecurityBlockedError`` and its
``threats`` back::

    with reraise_security_blocks():
        async for event in runner.run_async(user_id=..., session_id=..., new_message=msg):
            ...

Approvals ("ask")
-----------------
A tool call the gate answers with ``"ask"`` (a ``ToolPolicy(require_approval=...)``
match, or a ``require_approval`` policy rule) uses ADK's tool-confirmation
flow: the callback calls ``tool_context.request_confirmation(...)``, the tool
does not run, and the run ends with an ``adk_request_confirmation`` function
call for your client to show a human. The client answers with a
``FunctionResponse(name="adk_request_confirmation", id=<that call's id>,
response={"confirmed": True | False})`` as the next ``new_message``; ADK
re-runs the original call, the callback sees ``tool_context.tool_confirmation``
and lets an approved call run (still screening its result) or refuses a
rejected one. No ``resumability_config`` or ``require_confirmation=`` on the
tool is needed. Where the flow cannot complete -- ``approvals=False``, a live
(bidi-streaming) run, a context with no function-call id -- the ask fails
closed to a block. Inside an ``AgentTool`` the nested run cannot be resumed,
so ADK answers the outer call with an error: the tool still does not run.

For tools governed without the plugin, :func:`gate_confirmation` derives
``FunctionTool(func, require_confirmation=...)`` from the gate.

MCP
---
``McpToolset`` tools are ordinary ``BaseTool`` instances by the time they run,
so their calls and results pass through the same tool callbacks (verified
against a real stdio MCP server). :func:`mcp_toolset_kwargs` additionally
derives a ``tool_filter`` from the gate's ``ToolPolicy`` so a denied MCP tool is
never even offered to the model.

Streaming
---------
Under ``RunConfig(streaming_mode=StreamingMode.SSE)`` ADK yields every partial
chunk to the caller as it arrives, and checks each one on its own -- so text
before a blocked chunk is already delivered, and a pattern split across chunks
(``"SEC" + "RET-TOKEN"``) is invisible to every per-chunk check. With
``buffer_stream=True`` the after-model hook keeps a per-model-call buffer:
every partial is screened as part of the aggregate so far, and its content is
replaced by what is safe to release:

* ``stream_holdback=None`` (default): nothing is released until the complete
  response has passed. Strict, but the caller sees no incremental text -- the
  partial events still arrive, empty, and the final event carries the answer.
* ``stream_holdback=N``: everything except the last ``N`` characters is
  released once the aggregate passes. Incremental, at an ``N``-character delay;
  a match longer than ``N`` characters can have its start released before it
  completes, so set ``N`` to at least the longest string your rules must catch.

Because the hook rewrites the events the ``Runner`` yields, this works for a
bare ``Runner`` as well as through the adapter. Without ``buffer_stream`` the
limitation above applies in full. Live (bidi) runs are not buffered.

Policy state
------------
A fail-closed :class:`~adapt_agent.core.policy.PolicyEnforcer` treats a rule it
cannot evaluate as violated, so the state each hook builds is fixed:

* model request: ``messages`` (the request ``contents``) and ``context`` (``{}``),
  merged with the ADK session state (``callback_context.state``, including
  writes made earlier in the same invocation). The screened ``messages`` win
  over a session key of the same name.
* user message: the same shape, with ``messages`` = ``[user_message]``.
* tool call: the session state as-is, passed as the ``state`` of a tool-scoped
  rule (``tool['name']``, ``tool['arguments']``, ``state['trust_score']``).
"""

from __future__ import annotations

import collections
import contextlib
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from adapt_agent.core.governance import BLOCK, GovernanceGate, ToolDecision, ToolPolicy
from adapt_agent.exceptions import SecurityBlockedError
from adapt_agent.integrations._common import (
    as_state,
    build_gate,
    context_state,
    optional_import,
    traced,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from adapt_agent.adversarial import AdversarialDefense
    from adapt_agent.core.policy import PolicyEnforcer
    from adapt_agent.security.firewall import Firewall

#: Text returned to the model's caller when ``on_block="refuse"``.
DEFAULT_REFUSAL = "This request was blocked by a security policy."

#: What the session stores in place of a user message refused at run level.
WITHHELD_MESSAGE = "[message withheld by a security policy]"

#: The plugin's ``name``; ADK refuses two plugins of the same name on one runner.
DEFAULT_PLUGIN_NAME = "adapt_agent_governance"

_MODES = ("raise", "refuse")

#: How many tool decisions a plugin keeps in ``decisions``.
_DECISION_HISTORY = 256


# -- error unwrapping ------------------------------------------------------------


def find_security_error(exc: BaseException | None) -> SecurityBlockedError | None:
    """Return the :class:`SecurityBlockedError` behind ``exc``, if there is one.

    ADK re-raises an exception from a plugin callback as
    ``RuntimeError("Error in plugin ...")`` chained to the original, and an
    ``AgentTool`` or ``Workflow`` can wrap it again. A caller catching
    ``SecurityBlockedError`` would miss every one of those. This walks
    ``__cause__``, ``__context__`` and exception groups, guarding against cycles.
    """
    seen: set[int] = set()
    pending: list[BaseException] = [exc] if exc is not None else []
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, SecurityBlockedError):
            return current
        grouped = getattr(current, "exceptions", None)
        if isinstance(grouped, (list, tuple)):
            pending.extend(e for e in grouped if isinstance(e, BaseException))
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                pending.append(linked)
    return None


@contextlib.contextmanager
def reraise_security_blocks() -> Iterator[None]:
    """Re-raise a wrapped :class:`SecurityBlockedError` as itself.

    Works around ``for``/``async for`` alike, since it only inspects what
    propagates out of the block::

        with reraise_security_blocks():
            for event in runner.run(user_id="u", session_id="s", new_message=msg):
                ...

    Any other exception propagates unchanged.
    """
    try:
        yield
    except SecurityBlockedError:
        raise
    except Exception as exc:
        found = find_security_error(exc)
        if found is None:
            raise
        # `from None`: the original's own traceback already points at the hook
        # that blocked; chaining the ADK wrapper back on would make a cycle.
        raise found from None


# -- shared machinery ---------------------------------------------------------------


def _session_state(context: Any) -> dict[str, Any]:
    """The ADK session state visible to ``context``, pending writes included.

    ADK's ``State`` is not a ``Mapping`` -- it merges the committed session
    state with the current invocation's uncommitted delta -- so the generic
    `context_state` falls back to the committed ``session.state`` and misses a
    value an earlier callback in the same turn just set. ``to_dict()`` is the
    merged view.
    """
    state = getattr(context, "state", None)
    to_dict = getattr(state, "to_dict", None)
    if callable(to_dict):
        try:
            merged = to_dict()
        except Exception:  # pragma: no cover - defensive: fall back below
            merged = None
        if isinstance(merged, Mapping):
            return dict(merged)
    return context_state(context)


def _text_of(content: Any) -> str:
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(part, "text", None) or "" for part in parts)


def _check_mode(option: str, value: str) -> None:
    if value not in _MODES:
        raise ValueError(f"{option} must be 'raise' or 'refuse', got {value!r}")


class _Stream:
    __slots__ = ("blocked", "released", "text", "threats")

    def __init__(self) -> None:
        self.text = ""
        self.released = 0
        self.blocked = False
        self.threats: list[str] = []


class _Governor:
    """The hook bodies, shared by the plugin and the per-agent callbacks.

    One implementation, so the two bindings cannot drift; each review call
    site appears exactly once, so a hook that stops screening is caught.
    """

    def __init__(
        self,
        gate: GovernanceGate,
        *,
        observer: Any,
        on_block: str,
        on_tool_violation: str,
        refusal_text: str,
        approvals: bool,
        buffer_stream: bool,
        stream_holdback: int | None,
        label_per_agent: bool,
        tools_governed: bool,
    ):
        _check_mode("on_block", on_block)
        _check_mode("on_tool_violation", on_tool_violation)
        if stream_holdback is not None:
            if not buffer_stream:
                raise ValueError("stream_holdback needs buffer_stream=True")
            if isinstance(stream_holdback, bool) or not isinstance(stream_holdback, int):
                raise TypeError("stream_holdback must be an int or None")
            if stream_holdback < 1:
                raise ValueError("stream_holdback must be at least 1 (or None to hold all)")
        self.gate = gate
        self.observer = observer
        self.on_block = on_block
        self.on_tool_violation = on_tool_violation
        self.refusal_text = refusal_text
        self.approvals = approvals
        self.buffer_stream = buffer_stream
        self.stream_holdback = stream_holdback
        self.label_per_agent = label_per_agent
        self.tools_governed = tools_governed
        self._gates: dict[str, GovernanceGate] = {}
        self._streams: dict[tuple[str, ...], _Stream] = {}
        self._refused_invocations: set[str] = set()
        #: The most recent tool decisions, newest last -- report-only ones
        #: included, which is how a ``block_on_violation=False`` rollout sees
        #: what *would* have been refused. Bounded, so a long-lived plugin
        #: cannot grow without limit.
        self.decisions: collections.deque[ToolDecision] = collections.deque(
            maxlen=_DECISION_HISTORY
        )

    # -- attribution ----------------------------------------------------------

    def gate_for(self, context: Any) -> GovernanceGate:
        """The gate labelled for the agent ``context`` belongs to.

        A plugin covers the whole tree, so an error naming only the plugin
        would not say which agent tripped it.
        """
        agent_name = getattr(context, "agent_name", None)
        if not self.label_per_agent or not isinstance(agent_name, str) or not agent_name:
            return self.gate
        label = f"{self.gate.agent_id}/{agent_name}"
        found = self._gates.get(label)
        if found is None:
            found = self._gates[label] = build_gate(gate=self.gate, agent_id=label)
        return found

    # -- the single review call sites -----------------------------------------

    def _review_request(
        self, resolved: GovernanceGate, payload: Any, state: Mapping[str, Any]
    ) -> list[str]:
        # `state=` is required, or a configured policy_enforcer never runs.
        return resolved.review_input(payload, state=state)

    def _review_response(self, resolved: GovernanceGate, payload: Any) -> list[str]:
        return resolved.review_output(payload)

    # -- model calls ----------------------------------------------------------

    def before_model(self, context: Any, llm_request: Any) -> Any:
        resolved = self.gate_for(context)
        contents = getattr(llm_request, "contents", None)
        screened = _without_function_calls(contents) if self.tools_governed else contents
        try:
            with traced(self.observer, resolved.agent_id, "google_adk.model_request"):
                self._review_request(
                    resolved, screened, as_state(contents, **_session_state(context))
                )
        except SecurityBlockedError as exc:
            if self.on_block == "raise":
                raise
            return _refusal_response(self.refusal_text, resolved.agent_id, exc.threats)
        return None  # None = proceed to the model

    def after_model(self, context: Any, llm_response: Any) -> Any:
        resolved = self.gate_for(context)
        partial = getattr(llm_response, "partial", None) is True
        if self.buffer_stream and partial:
            return self._buffer_partial(context, resolved, llm_response)
        stream = self._streams.pop(_stream_key(context), None) if self.buffer_stream else None
        try:
            with traced(self.observer, resolved.agent_id, "google_adk.model_response"):
                if stream is not None and stream.blocked:
                    # The aggregate already failed; a final that somehow
                    # differs must not be what slips through.
                    raise SecurityBlockedError(
                        f"Output blocked by security controls [{resolved.agent_id}]",
                        stream.threats,
                    )
                self._review_response(resolved, getattr(llm_response, "content", llm_response))
        except SecurityBlockedError as exc:
            if self.on_block == "raise":
                raise
            return _refusal_response(self.refusal_text, resolved.agent_id, exc.threats)
        return None  # None = keep the model's own response

    def _buffer_partial(self, context: Any, resolved: GovernanceGate, llm_response: Any) -> Any:
        """Screen one streamed chunk as part of the aggregate; release what is safe."""
        key = _stream_key(context)
        stream = self._streams.setdefault(key, _Stream())
        if not stream.blocked:
            stream.text += _text_of(getattr(llm_response, "content", None))
            try:
                self._review_response(resolved, stream.text)
            except SecurityBlockedError as exc:
                stream.blocked = True
                stream.threats = list(exc.threats)
                if self.on_block == "raise":
                    self._streams.pop(key, None)
                    raise
        if stream.blocked:
            return _partial_response("")
        if self.stream_holdback is None:
            return _partial_response("")
        releasable = max(len(stream.text) - self.stream_holdback, stream.released)
        delta = stream.text[stream.released : releasable]
        stream.released = releasable
        return _partial_response(delta)

    # -- the user message ---------------------------------------------------

    def on_user_message(self, invocation_context: Any, user_message: Any) -> Any:
        resolved = self.gate
        session = getattr(invocation_context, "session", None)
        state = _session_state(session) if session is not None else {}
        try:
            with traced(self.observer, resolved.agent_id, "google_adk.user_message"):
                self._review_request(resolved, user_message, as_state([user_message], **state))
        except SecurityBlockedError:
            if self.on_block == "raise":
                raise
            invocation_id = getattr(invocation_context, "invocation_id", None)
            if isinstance(invocation_id, str):
                self._refused_invocations.add(invocation_id)
            # Replaced, not merely flagged: the session keeps whatever this
            # returns, and a stored injection would be re-screened -- and
            # refused -- on every later turn of the conversation.
            return _content("user", WITHHELD_MESSAGE)
        return None

    def before_run(self, invocation_context: Any) -> Any:
        invocation_id = getattr(invocation_context, "invocation_id", None)
        if invocation_id in self._refused_invocations:
            self._refused_invocations.discard(invocation_id)
            return _content("model", self.refusal_text)
        return None

    def after_run(self, invocation_context: Any) -> None:
        invocation_id = getattr(invocation_context, "invocation_id", None)
        self._refused_invocations.discard(invocation_id)
        # A model call that errored mid-stream never delivers its final chunk.
        for key in [k for k in self._streams if k and k[0] == invocation_id]:
            self._streams.pop(key, None)

    # -- tools ------------------------------------------------------------------

    def before_tool(self, tool: Any, args: Any, tool_context: Any) -> dict[str, Any] | None:
        resolved = self.gate_for(tool_context)
        name = _tool_name(tool)
        try:
            with traced(self.observer, resolved.agent_id, "google_adk.tool_call"):
                decision = resolved.check_tool_call(name, args, state=_session_state(tool_context))
                self.decisions.append(decision)
                if decision.needs_approval:
                    return self._ask(decision, tool_context)
                if decision.blocked:
                    raise SecurityBlockedError(decision.reason, list(decision.threats))
        except SecurityBlockedError as exc:
            if self.on_tool_violation == "raise":
                raise
            return {"error": exc.reason}
        return None  # None = run the tool

    def _ask(self, decision: ToolDecision, tool_context: Any) -> dict[str, Any] | None:
        """Route an "ask" to ADK's tool-confirmation flow, or fail closed."""
        confirmation = getattr(tool_context, "tool_confirmation", None)
        if confirmation is not None:
            if getattr(confirmation, "confirmed", False) is True:
                return None  # a human approved this exact call
            return {"error": f"Tool call {decision.tool_name!r} was rejected by a human reviewer."}
        if self._can_confirm(tool_context):
            try:
                tool_context.request_confirmation(hint=decision.reason)
            except Exception:
                pass  # no function-call id, or an ADK that cannot pause: block below
            else:
                actions = getattr(tool_context, "actions", None)
                if actions is not None:
                    # The pause is not a tool result for the model to summarise;
                    # without this ADK calls the model again, which calls the
                    # tool again.
                    actions.skip_summarization = True
                return {"error": f"{decision.reason} Awaiting a human decision; it has not run."}
        # Never an allow: an ask nobody can answer must not run unseen.
        if decision.resolve(approvals_supported=False) != BLOCK:  # pragma: no cover
            raise AssertionError("an unanswerable ask must resolve to a block")
        raise SecurityBlockedError(
            f"{decision.reason} No approval mechanism is available, so it was refused.",
            list(decision.threats),
        )

    def _can_confirm(self, tool_context: Any) -> bool:
        if not self.approvals or not callable(getattr(tool_context, "request_confirmation", None)):
            return False
        if not getattr(tool_context, "function_call_id", None):
            return False
        get_ic = getattr(tool_context, "get_invocation_context", None)
        if callable(get_ic):
            try:
                invocation = get_ic()
            except Exception:
                return False
            # A live (bidi) run relays the confirmation error to the model and
            # closes the call; it has no way to resume it.
            if getattr(invocation, "live_request_queue", None) is not None:
                return False
        return True

    def after_tool(
        self, tool: Any, args: Any, tool_context: Any, result: Any
    ) -> dict[str, Any] | None:
        resolved = self.gate_for(tool_context)
        name = _tool_name(tool)
        try:
            with traced(self.observer, resolved.agent_id, "google_adk.tool_result"):
                decision = resolved.check_tool_result(
                    name, result, state=_session_state(tool_context)
                )
                self.decisions.append(decision)
                if decision.blocked:
                    raise SecurityBlockedError(
                        f"Tool result from {name!r} withheld by ADAPT-Agent"
                        f" [{resolved.agent_id}]: {', '.join(decision.threats)}.",
                        list(decision.threats),
                    )
        except SecurityBlockedError as exc:
            if self.on_tool_violation == "raise":
                raise
            # A replacement, never the original: the next model call must not
            # see the content that was blocked.
            return {"error": exc.reason}
        return None  # None = keep the tool's own result


def _without_function_calls(contents: Any) -> Any:
    """The request history minus the model's own function calls.

    Those calls were the model's *output*: screened as such when produced, and
    their arguments again by the tool seam. Re-screening them as input to every
    later model call meant one call the tool seam had already refused --
    leaving the run to continue, as ``on_tool_violation="refuse"`` promises --
    sat in the history and blocked every following request. Everything else
    (user turns, tool results, the model's text) is screened as before.
    """
    if not isinstance(contents, (list, tuple)):
        return contents
    kept: list[Any] = []
    for content in contents:
        parts = getattr(content, "parts", None)
        if not isinstance(parts, (list, tuple)):
            kept.append(content)
            continue
        kept.extend(part for part in parts if getattr(part, "function_call", None) is None)
    return kept


def _tool_name(tool: Any) -> str:
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else type(tool).__name__


def _stream_key(context: Any) -> tuple[str, ...]:
    """One buffer per model call: the invocation, the agent and its branch."""
    return (
        str(getattr(context, "invocation_id", "")),
        str(getattr(context, "agent_name", "")),
        str(getattr(context, "branch", "") or ""),
    )


def _content(role: str, text: str) -> Any:
    genai_types = optional_import("google.genai.types", "google-adk", "the ADK governance plugin")
    return genai_types.Content(role=role, parts=[genai_types.Part(text=text)])


def _partial_response(text: str) -> Any:
    """A partial ``LlmResponse`` carrying only ``text`` (possibly none)."""
    llm_response = optional_import("google.adk.models.llm_response", "google-adk", "buffer_stream")
    genai_types = optional_import("google.genai.types", "google-adk", "buffer_stream")
    parts = [genai_types.Part(text=text)] if text else []
    return llm_response.LlmResponse(
        content=genai_types.Content(role="model", parts=parts), partial=True
    )


def _refusal_response(text: str, agent_id: str, threats: list[str]) -> Any:
    """Build an ADK ``LlmResponse`` carrying ``text``, short-circuiting the model.

    The refusal is an ordinary response, so unlike the raising path it carries no
    exception for the surrounding graph to inspect: two agents refusing for
    different reasons produced byte-identical objects. Attribution therefore goes
    in ``custom_metadata``, under a namespaced key so it cannot collide with the
    app's own. It stays out of ``text`` deliberately -- that is the caller's copy
    for the end user, and an agent id is internal topology.
    """
    llm_response = optional_import(
        "google.adk.models.llm_response", "google-adk", "on_block='refuse'"
    )
    genai_types = optional_import("google.genai.types", "google-adk", "on_block='refuse'")
    return llm_response.LlmResponse(
        content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)]),
        custom_metadata={"adapt_agent": {"agent_id": agent_id, "threats": list(threats)}},
    )


# -- the plugin -------------------------------------------------------------------


def governance_plugin(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    tool_policy: ToolPolicy | None = None,
    observer: Any = None,
    on_block: str = "raise",
    on_tool_violation: str = "refuse",
    refusal_text: str = DEFAULT_REFUSAL,
    screen_output: bool = True,
    screen_user_message: bool = True,
    approvals: bool = True,
    buffer_stream: bool = False,
    stream_holdback: int | None = None,
    name: str = DEFAULT_PLUGIN_NAME,
) -> Any:
    """Build an ADK plugin that governs every agent and tool in a run.

    Register it on the ``App`` (``App(plugins=[...])``; ``Runner(plugins=...)``
    is deprecated upstream). See the module docstring for what each hook does.

    Args:
        gate: A pre-built gate; otherwise built from the controls below.
        firewall: Screens the user message, model requests and responses, tool
            arguments and tool results.
        defense: Adversarial analysis of inputs (user message, requests, tool
            arguments and results).
        policy_enforcer: State rules on the user message and model requests;
            tool-scoped rules on tool calls. See "Policy state".
        block_on_violation: ``False`` records without blocking: every decision
            comes back as an allow, and threats are still traced.
        agent_id: Prefix of every label; errors and spans name
            ``"<agent_id>/<adk agent name>"``.
        tool_policy: Name-based allow / deny / approval lists for tool calls.
        observer: An :class:`~adapt_agent.observability.AgentObserver`; one span
            per screened user message, model request/response and tool
            call/result.
        on_block: ``"raise"`` (default) or ``"refuse"``, for the user message
            and model calls.
        on_tool_violation: ``"refuse"`` (default) or ``"raise"``, for tool
            calls and results.
        refusal_text: Body of a refusal when ``on_block="refuse"``.
        screen_output: Also screen model responses.
        screen_user_message: Screen the user message at run level (before it is
            stored in the session).
        approvals: Route "ask" decisions to ADK's confirmation flow. ``False``
            turns every ask into a block.
        buffer_stream: Hold streamed (SSE) output until it has been screened.
        stream_holdback: With ``buffer_stream``, release all but this many
            trailing characters as the stream passes; ``None`` holds the whole
            response.
        name: The plugin name (unique per runner).

    Returns:
        An ``AdaptGovernancePlugin`` (a ``google.adk.plugins.BasePlugin``).

    Raises:
        ValueError: For an unknown mode, or an invalid holdback.
        ImportError: If ``google-adk`` is not installed.
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
    governor = _Governor(
        resolved,
        observer=observer,
        on_block=on_block,
        on_tool_violation=on_tool_violation,
        refusal_text=refusal_text,
        approvals=approvals,
        buffer_stream=buffer_stream,
        stream_holdback=stream_holdback,
        label_per_agent=True,
        tools_governed=True,
    )
    base_plugin = optional_import(
        "google.adk.plugins.base_plugin", "google-adk", "governance_plugin"
    )
    cls = _plugin_class(base_plugin.BasePlugin)
    return cls(
        name=name,
        governor=governor,
        screen_output=screen_output,
        screen_user_message=screen_user_message,
    )


_PLUGIN_CLASSES: dict[type, type] = {}


def _plugin_class(base: type) -> type:
    """Define the plugin class against ADK's ``BasePlugin``, once.

    Defined lazily so this module imports without ADK installed.
    """
    found = _PLUGIN_CLASSES.get(base)
    if found is not None:
        return found

    class AdaptGovernancePlugin(base):
        """ADAPT-Agent governance for every agent and tool of an ADK run."""

        def __init__(
            self,
            *,
            name: str,
            governor: _Governor,
            screen_output: bool,
            screen_user_message: bool,
        ):
            super().__init__(name=name)
            self.governor = governor
            self.gate = governor.gate
            #: Recent :class:`~adapt_agent.core.governance.ToolDecision` objects.
            self.decisions = governor.decisions
            self.screen_output = screen_output
            self.screen_user_message = screen_user_message

        async def on_user_message_callback(
            self, *, invocation_context: Any, user_message: Any
        ) -> Any:
            if not self.screen_user_message:
                return None
            return self.governor.on_user_message(invocation_context, user_message)

        async def before_run_callback(self, *, invocation_context: Any) -> Any:
            return self.governor.before_run(invocation_context)

        async def after_run_callback(self, *, invocation_context: Any) -> None:
            self.governor.after_run(invocation_context)

        async def before_model_callback(self, *, callback_context: Any, llm_request: Any) -> Any:
            return self.governor.before_model(callback_context, llm_request)

        async def after_model_callback(self, *, callback_context: Any, llm_response: Any) -> Any:
            if not self.screen_output:
                return None
            return self.governor.after_model(callback_context, llm_response)

        async def before_tool_callback(
            self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any
        ) -> dict[str, Any] | None:
            return self.governor.before_tool(tool, tool_args, tool_context)

        async def after_tool_callback(
            self,
            *,
            tool: Any,
            tool_args: dict[str, Any],
            tool_context: Any,
            result: dict[str, Any],
        ) -> dict[str, Any] | None:
            return self.governor.after_tool(tool, tool_args, tool_context, result)

    _PLUGIN_CLASSES[base] = AdaptGovernancePlugin
    return AdaptGovernancePlugin


# -- per-agent callbacks ------------------------------------------------------------


def governance_callbacks(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    screen_output: bool = True,
    on_block: str = "raise",
    refusal_text: str = DEFAULT_REFUSAL,
    tool_policy: ToolPolicy | None = None,
    observer: Any = None,
    on_tool_violation: str = "refuse",
    govern_tools: bool = True,
    approvals: bool = True,
    buffer_stream: bool = False,
    stream_holdback: int | None = None,
) -> dict[str, Any]:
    """Build ADK callbacks applying ADAPT-Agent governance to one ``LlmAgent``.

    Prefer :func:`governance_plugin`, which covers the whole tree; use these
    for an agent that needs its own, stricter controls.

    Args:
        gate: A pre-built gate to share across agents; otherwise built from the
            controls below.
        firewall: Screens the outgoing request contents, the model response,
            tool arguments and tool results.
        defense: Adversarial analysis of the request contents and tool traffic.
        policy_enforcer: Evaluated against the request *and the callback's
            session state*, so a rule gating on session data such as
            ``state['trust_score']`` sees it; tool-scoped rules on tool calls.
        block_on_violation: ``False`` scans and records without blocking.
        agent_id: Named in the raised error, and in a refusal's
            ``custom_metadata["adapt_agent"]``, identifying which agent
            in the tree refused.
        screen_output: Also screen the model response.
        on_block: ``"raise"`` (default) or ``"refuse"`` for model calls -- see
            the module docstring.
        refusal_text: Body of the refusal response when ``on_block="refuse"``.
        tool_policy: Name-based allow / deny / approval lists for tool calls.
        observer: An :class:`~adapt_agent.observability.AgentObserver`.
        on_tool_violation: ``"refuse"`` (default) or ``"raise"`` for tool calls
            and results.
        govern_tools: Also return the two tool callbacks.
        approvals: Route "ask" decisions to ADK's confirmation flow; ``False``
            turns every ask into a block.
        buffer_stream: Hold streamed output until screened (see module docs).
        stream_holdback: Trailing characters held back with ``buffer_stream``.

    Returns:
        A dict of ADK callback kwargs -- splat it into ``LlmAgent(...)``. Keys:
        ``before_model_callback``, ``after_model_callback`` (when
        ``screen_output``), ``before_tool_callback`` and
        ``after_tool_callback`` (when ``govern_tools``).

    Raises:
        ValueError: If ``on_block`` or ``on_tool_violation`` is not ``"raise"``
            or ``"refuse"``.
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
    governor = _Governor(
        resolved,
        observer=observer,
        on_block=on_block,
        on_tool_violation=on_tool_violation,
        refusal_text=refusal_text,
        approvals=approvals,
        buffer_stream=buffer_stream,
        stream_holdback=stream_holdback,
        label_per_agent=False,
        tools_governed=govern_tools,
    )

    # Parameter names are ADK's canonical ones: ADK passes them as keywords.
    async def before_model_callback(callback_context: Any, llm_request: Any) -> Any:
        return governor.before_model(callback_context, llm_request)

    callbacks: dict[str, Any] = {"before_model_callback": before_model_callback}

    if screen_output:

        async def after_model_callback(callback_context: Any, llm_response: Any) -> Any:
            return governor.after_model(callback_context, llm_response)

        callbacks["after_model_callback"] = after_model_callback

    if govern_tools:

        async def before_tool_callback(
            tool: Any, args: dict[str, Any], tool_context: Any
        ) -> dict[str, Any] | None:
            return governor.before_tool(tool, args, tool_context)

        async def after_tool_callback(
            tool: Any, args: dict[str, Any], tool_context: Any, tool_response: Any
        ) -> dict[str, Any] | None:
            return governor.after_tool(tool, args, tool_context, tool_response)

        callbacks["before_tool_callback"] = before_tool_callback
        callbacks["after_tool_callback"] = after_tool_callback

    return callbacks


# -- declaring tools from the gate -----------------------------------------------


def gate_confirmation(gate: GovernanceGate, tool_name: str) -> Callable[..., bool]:
    """A ``FunctionTool(require_confirmation=...)`` callable derived from ``gate``.

    For a tool governed *without* the plugin or tool callbacks: ADK then asks a
    human before running any call the gate would not simply allow. A call the
    gate would block also needs confirmation here -- this hook can only pause,
    not refuse, and running it unseen is the one outcome to avoid. With the
    plugin installed it is redundant (the plugin refuses blocks outright and
    asks for asks); combining the two still asks only once.

    ``tool_name`` must be the name the model calls the tool by, since ADK does
    not tell this callable which tool it guards.
    """

    def require_confirmation(**arguments: Any) -> bool:
        arguments.pop("tool_context", None)
        decision = gate.check_tool_call(tool_name, arguments)
        return not decision.allowed

    return require_confirmation


def mcp_toolset_kwargs(
    *,
    gate: GovernanceGate | None = None,
    tool_policy: ToolPolicy | None = None,
    tool_name_prefix: str | None = None,
) -> dict[str, Any]:
    """``McpToolset(...)`` keyword arguments derived from a ``ToolPolicy``.

    Returns ``tool_filter`` (and ``tool_name_prefix`` when given): a predicate
    that hides every tool the policy denies or leaves off its allow-list, so
    the model is never offered them. It matches on the name the plugin will
    see -- ``"<prefix>_<name>"`` when a prefix is set -- so one ``ToolPolicy``
    serves both. Calls to the tools that remain still go through the plugin's
    tool callbacks, which also handle approvals.

    ``require_confirmation`` is deliberately not derived: ADK hands a toolset's
    confirmation callable the call's arguments but not the tool's name, so a
    name-based policy cannot be evaluated there. The plugin's confirmation flow
    covers MCP tools instead.

    Raises:
        ValueError: If neither ``gate`` (with a ``tool_policy``) nor
            ``tool_policy`` is given -- an empty filter would look like a
            control while restricting nothing.
    """
    policy = tool_policy if tool_policy is not None else getattr(gate, "tool_policy", None)
    if policy is None:
        raise ValueError("mcp_toolset_kwargs needs a tool_policy (directly or on the gate)")

    def tool_filter(tool: Any, readonly_context: Any = None) -> bool:
        name = _tool_name(tool)
        exposed = f"{tool_name_prefix}_{name}" if tool_name_prefix else name
        action, _ = policy.verdict(exposed)
        return action != BLOCK

    kwargs: dict[str, Any] = {"tool_filter": tool_filter}
    if tool_name_prefix:
        kwargs["tool_name_prefix"] = tool_name_prefix
    return kwargs


__all__ = [
    "DEFAULT_PLUGIN_NAME",
    "DEFAULT_REFUSAL",
    "WITHHELD_MESSAGE",
    "find_security_error",
    "gate_confirmation",
    "governance_callbacks",
    "governance_plugin",
    "mcp_toolset_kwargs",
    "reraise_security_blocks",
]
