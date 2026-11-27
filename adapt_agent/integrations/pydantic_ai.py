"""Governance as a Pydantic AI capability -- input, tool calls, output, streams.

Pydantic AI's ``Agent`` takes **capabilities**
(:class:`pydantic_ai.capabilities.AbstractCapability`): objects whose hooks run
around every model request, every tool call and every output validation, given
either at construction (``Agent(..., capabilities=[...])``) or per run
(``agent.run(..., capabilities=[...])``). :func:`governance_capability` builds
one that binds every ADAPT-Agent control to those hooks:

.. code-block:: python

    from adapt_agent.core.governance import ToolPolicy
    from adapt_agent.integrations.pydantic_ai import governance_capability

    governance = governance_capability(
        firewall=fw,
        defense=AdversarialDefense(),
        policy_enforcer=enforcer,
        tool_policy=ToolPolicy(deny=["delete_*"], require_approval=["send_email"]),
        agent_id="triage",
    )
    agent = Agent("openai:gpt-5.6-luna", tools=[...], capabilities=[governance])
    # ...or govern an agent you did not construct, for one run:
    result = await other_agent.run(prompt, capabilities=[governance])

What each hook does
-------------------
``before_model_request``
    Screens what is about to reach the model as **input** -- firewall,
    adversarial defense and (state) policy rules: the user prompt, any
    ``message_history`` the caller supplied, and every tool result, retry
    message or externally supplied deferred result. Each message part is
    screened once per run; later steps screen only what is new.
``before_tool_execute``
    ``gate.check_tool_call(name, args, state=...)`` on the validated arguments,
    before the tool function runs: the :class:`~adapt_agent.core.governance.ToolPolicy`
    lists, the firewall/defense over the arguments, tool-scoped policy rules.
    MCP tools (``capabilities.MCP``, ``mcp.MCPToolset``) surface here exactly
    like local function tools.
``after_tool_execute``
    ``gate.check_tool_result(...)``: the indirect-injection seam. A blocked
    result never reaches the next model call -- it is withheld (``"refuse"``) or
    the run aborts (``"raise"``).
``after_model_request`` / ``after_output_validate``
    The firewall over the model's text and over a structured ``output_type``
    field by field.
``get_wrapper_model``
    With ``buffer_streams=True`` (the default), a *streamed* model response is
    held until it is complete and screened before any of it is released -- see
    *Streaming* below.

Tool verdicts
-------------
A **block** is handled per ``on_tool_violation``:

* ``"raise"`` (default) -- the run aborts with
  :class:`~adapt_agent.exceptions.SecurityBlockedError` carrying the threats.
  Pydantic AI propagates an exception raised from a tool hook unchanged (it is
  not turned into a tool failure, and parallel sibling calls do not wrap it in
  an ``ExceptionGroup``), so the abort is real. It is the default because a
  refused call lets the model *keep going* -- free to reach the same end by
  another tool -- and the caller learns nothing unless it reads the transcript.
* ``"refuse"`` -- raises Pydantic AI's ``ToolFailed`` with
  :attr:`ToolDecision.reason <adapt_agent.core.governance.ToolDecision.reason>`
  (tool name and threat labels only, never the content), which the model sees
  as the tool's failed result. The tool does not run; for a blocked *result*,
  the result is discarded and the model sees only the reason.

An **ask** maps to Pydantic AI's deferred-approval protocol: the hook raises
``ApprovalRequired`` before the tool runs, and the run ends returning a
``DeferredToolRequests`` (the agent's ``output_type`` must include it, or a
``HandleDeferredToolCalls`` capability must resolve it inline). The caller
resumes with ``agent.run(message_history=result.all_messages(),
deferred_tool_results=DeferredToolResults(approvals={call_id: True | False}))``.
On resume the call is screened again: an approval satisfies only the *ask*, so
arguments that would be blocked stay blocked, and a denial never executes the
tool. If nothing can pause -- ``DeferredToolRequests`` is not an output type --
the run aborts with ``SecurityBlockedError`` (Pydantic AI's own error chained)
before the tool runs. ``approvals=False`` turns every ask into a block up front
(:meth:`ToolDecision.resolve <adapt_agent.core.governance.ToolDecision.resolve>`).

Policy state
------------
A fail-closed :class:`~adapt_agent.core.policy.PolicyEnforcer` (the default)
treats a rule naming a key the state lacks as violated, so what a rule can rely
on is fixed here. Every hook builds its state from the run's ``deps``:

* a mapping -- its string keys, as-is;
* a dataclass instance -- its fields, shallowly (``getattr``, no deep copy, so
  an HTTP client or connection in ``deps`` is fine);
* a Pydantic model -- its declared ``model_fields``, shallowly;
* anything else (``None``, an arbitrary object) -- nothing.

plus ``messages`` -- the parts being screened at ``before_model_request``, as
``{"role": "user" | "tool", "content": ...}`` dicts (the adapter's shape), empty
at the tool hooks -- and ``context`` (``{}`` unless ``deps`` supplies one).
``policy_state=lambda ctx: {...}`` replaces the ``deps`` extraction entirely.
Tool-scoped rules additionally see ``tool = {"name", "arguments"}``. Policy is
re-evaluated at every model request, so a tool that changes ``deps`` mid-run
(``deps.trust_score -= ...``) is seen by the next request.

Streaming
---------
Without buffering, ``run_stream`` / ``run_stream_events`` / ``iter`` deliver
model text to the consumer as it arrives -- *before* ``after_model_request``
can screen the complete response, so a blocked answer is partly delivered
before the run aborts. ``buffer_streams=True`` (the default) wraps the serving
model so a streamed response is drained, screened (its text and any
structured-output tool arguments) and only then replayed, event for event, to
the consumer. A blocked response is never delivered. The cost is latency: the
consumer receives each model response in one burst at its end instead of token
by token. Non-streamed runs (``run``/``run_sync``) are unaffected.
``buffer_streams=False`` restores live tokens and accepts the gap above.

Run-level coverage
------------------
A capability given to ``agent.run(capabilities=[...])`` governs that run only.
Pydantic AI multi-agent delegation is a tool that calls *another* agent's
``run``; the delegate's run does not inherit the parent's capabilities, so pass
the same capability there too (``delegate.run(..., capabilities=[governance])``)
or construct the delegate with it. The parent's tool hooks still screen the
delegation call's arguments and the delegate's answer as a tool result.

Older seam
----------
:func:`governance_output_validator` / :func:`install_governance` register an
``output_validator`` -- the final output only. Pydantic AI offers no public way
to add a capability to an agent that already exists (``Agent.override`` is a
scoped context manager), so :func:`install_governance` remains output-only;
for an agent you did not build, pass the capability per run instead.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from adapt_agent.core.governance import GovernanceGate, ToolDecision, ToolPolicy
from adapt_agent.exceptions import SecurityBlockedError
from adapt_agent.integrations._common import as_state, build_gate, optional_import, traced

if TYPE_CHECKING:  # pragma: no cover - typing only
    from adapt_agent.adversarial import AdversarialDefense
    from adapt_agent.core.policy import PolicyEnforcer
    from adapt_agent.security.firewall import Firewall

#: ``on_tool_violation`` values.
_TOOL_VIOLATION_MODES = ("raise", "refuse")


def deps_state(deps: Any) -> dict[str, Any]:
    """The policy state a run's ``deps`` contributes -- see the module docstring.

    Shallow on purpose: ``dataclasses.asdict`` deep-copies, which fails on the
    clients and connections ``deps`` conventionally carries, and a rule only
    ever reads top-level keys anyway. Non-string mapping keys are dropped, since
    a rule cannot name them and ``as_state`` takes them as keyword arguments.
    """
    if deps is None:
        return {}
    if isinstance(deps, Mapping):
        items: dict[Any, Any] = dict(deps)
    elif dataclasses.is_dataclass(deps) and not isinstance(deps, type):
        items = {f.name: getattr(deps, f.name) for f in dataclasses.fields(deps)}
    else:
        fields = getattr(type(deps), "model_fields", None)  # a Pydantic v2 model
        if not isinstance(fields, Mapping):
            return {}
        items = {name: getattr(deps, name) for name in fields}
    return {key: value for key, value in items.items() if isinstance(key, str)}


def governance_capability(
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
    buffer_streams: bool = True,
    observer: Any = None,
    policy_state: Callable[[Any], Mapping[str, Any]] | None = None,
) -> Any:
    """Build a :class:`GovernanceCapability` for ``Agent(capabilities=...)`` or ``run(...)``.

    Args:
        gate, firewall, defense, policy_enforcer, block_on_violation, agent_id,
            tool_policy: The controls, as for every integration factory (see
            :func:`~adapt_agent.integrations._common.build_gate`).
        on_tool_violation: ``"raise"`` (default) aborts the run on a blocked
            tool call or result; ``"refuse"`` reports the refusal to the model
            and lets the run continue. See the module docstring for why.
        approvals: Map an ``"ask"`` verdict onto Pydantic AI's deferred
            approval (default). ``False`` treats every ask as a block.
        buffer_streams: Hold streamed model responses until screened (default).
            See *Streaming* in the module docstring for the latency trade-off.
        observer: An :class:`~adapt_agent.observability.AgentObserver`; one span
            per governed model request and per tool call, plus a log entry for
            every tool decision that found something (report-only ones too).
        policy_state: ``ctx -> Mapping`` replacing the ``deps`` extraction for
            policy state.

    Raises:
        ValueError: On an unknown ``on_tool_violation``.
        ImportError: If ``pydantic_ai`` is not installed.
    """
    cls = _capability_class()
    return cls(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
        tool_policy=tool_policy,
        on_tool_violation=on_tool_violation,
        approvals=approvals,
        buffer_streams=buffer_streams,
        observer=observer,
        policy_state=policy_state,
    )


_CAPABILITY_CLASS: Any = None


def __getattr__(name: str) -> Any:
    # `GovernanceCapability` subclasses a Pydantic AI type, so defining it at
    # import would import the SDK -- and this package must import without it.
    if name == "GovernanceCapability":
        return _capability_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _capability_class() -> Any:  # noqa: C901 - one class, its hooks are the seam
    global _CAPABILITY_CLASS
    if _CAPABILITY_CLASS is not None:
        return _CAPABILITY_CLASS

    capabilities = optional_import(
        "pydantic_ai.capabilities", "pydantic-ai", "governance_capability"
    )
    exceptions = optional_import("pydantic_ai.exceptions", "pydantic-ai", "governance_capability")
    messages = optional_import("pydantic_ai.messages", "pydantic-ai", "governance_capability")
    models = optional_import("pydantic_ai.models", "pydantic-ai", "governance_capability")
    wrapper = optional_import("pydantic_ai.models.wrapper", "pydantic-ai", "governance_capability")
    tools = optional_import("pydantic_ai.tools", "pydantic-ai", "governance_capability")

    request_parts = (messages.UserPromptPart, messages.ToolReturnPart, messages.RetryPromptPart)

    class _ScreenedStreamModel(wrapper.WrapperModel):  # type: ignore[name-defined]
        """Drains a streamed response, screens it, then replays it.

        Buffering at the *model* is what makes this sound. Pydantic AI's
        ``stream_text`` reads the accumulated raw response as well as the event
        stream, so holding events back in ``wrap_run_event_stream`` leaves the
        two out of step: held text is both readable early and later delivered
        twice. Here the consumer never sees a live stream at all -- only a
        ``CompletedStreamedResponse`` replaying the captured events, released
        after the screen passed.
        """

        def __init__(self, wrapped: Any, governance: Any) -> None:
            super().__init__(wrapped)
            self._governance = governance

        @asynccontextmanager
        async def request_stream(
            self,
            messages: Any,
            model_settings: Any,
            model_request_parameters: Any,
            run_context: Any = None,
        ) -> AsyncIterator[Any]:
            async with self.wrapped.request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as live:
                events = [event async for event in live]
                response = live.get()
            self._governance._screen_response(response, model_request_parameters)
            yield models.CompletedStreamedResponse(
                response, model_request_parameters=model_request_parameters, replay_events=events
            )

    class GovernanceCapability(capabilities.AbstractCapability):  # type: ignore[name-defined]
        """ADAPT-Agent governance bound to Pydantic AI's capability hooks.

        See :mod:`adapt_agent.integrations.pydantic_ai` for what each hook
        screens, the tool verdict modes, the policy state and streaming.
        """

        def __init__(
            self,
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
            buffer_streams: bool = True,
            observer: Any = None,
            policy_state: Callable[[Any], Mapping[str, Any]] | None = None,
        ) -> None:
            if on_tool_violation not in _TOOL_VIOLATION_MODES:
                raise ValueError(
                    f"on_tool_violation must be one of {_TOOL_VIOLATION_MODES}, "
                    f"got {on_tool_violation!r}"
                )
            self.gate = build_gate(
                gate=gate,
                firewall=firewall,
                defense=defense,
                policy_enforcer=policy_enforcer,
                block_on_violation=block_on_violation,
                agent_id=agent_id,
                tool_policy=tool_policy,
            )
            self.on_tool_violation = on_tool_violation
            self.approvals = approvals
            self.buffer_streams = buffer_streams
            self.observer = observer
            self.policy_state = policy_state
            # Per-run state, replaced by `for_run`. Parts are kept (not just their
            # ids) so a collected part's id can never be reused by a new one.
            self._screened: dict[int, Any] = {}
            self._asked: list[ToolDecision] = []

        # -- per-run isolation ------------------------------------------------

        async def for_run(self, ctx: Any) -> Any:
            # One instance may serve many concurrent runs; what was screened and
            # what is awaiting approval belong to a single run.
            fresh = copy.copy(self)
            fresh._screened = {}
            fresh._asked = []
            return fresh

        # -- state ------------------------------------------------------------

        def _state(self, ctx: Any, texts: list[dict[str, Any]]) -> dict[str, Any]:
            if self.policy_state is not None:
                extra = {
                    key: value
                    for key, value in dict(self.policy_state(ctx)).items()
                    if isinstance(key, str)
                }
            else:
                extra = deps_state(getattr(ctx, "deps", None))
            return as_state(texts, **extra)

        # -- run --------------------------------------------------------------

        async def on_run_error(self, ctx: Any, *, error: BaseException) -> Any:
            # An ask was raised but nothing can pause: Pydantic AI aborts with a
            # configuration error. The tool did not run, so this already failed
            # closed -- re-raise it as the security refusal it is, so the caller
            # can see the threats. (`on_run_error`, not `wrap_run`: Pydantic AI
            # keeps the original error when `wrap_run` raises a different one.)
            if not isinstance(error, exceptions.UserError) or not self._asked:
                raise error
            threats = [t for decision in self._asked for t in decision.threats]
            raise SecurityBlockedError(
                f"{self._asked[0].reason} This run cannot pause for approval: add "
                "DeferredToolRequests to the agent's output_type (or use a "
                "HandleDeferredToolCalls capability), or pass approvals=False.",
                threats,
            ) from error

        # -- model requests ---------------------------------------------------

        async def wrap_model_request(self, ctx: Any, *, request_context: Any, handler: Any) -> Any:
            with traced(self.observer, self.gate.agent_id, "pydantic_ai.model_request"):
                return await handler(request_context)

        async def before_model_request(self, ctx: Any, request_context: Any) -> Any:
            # `{"role", "content"}` dicts -- the shape the adapter hands policy
            # too, so a rule like `state['messages'][0]['content']` reads the same
            # through either binding instead of failing to evaluate (and, under
            # a fail-closed enforcer, blocking every request).
            texts: list[dict[str, Any]] = []
            for message in request_context.messages:
                for part in getattr(message, "parts", ()):
                    if not isinstance(part, request_parts) or id(part) in self._screened:
                        continue
                    self._screened[id(part)] = part
                    content = getattr(part, "content", None)
                    # A RetryPromptPart's non-string content is Pydantic's own
                    # validation report on the model's arguments, not new input.
                    if isinstance(part, messages.RetryPromptPart) and not isinstance(content, str):
                        continue
                    role = "user" if isinstance(part, messages.UserPromptPart) else "tool"
                    texts.append({"role": role, "content": content})
            # Policy runs on every request, even with nothing new to screen: a
            # tool may have changed `deps` since the last one.
            self.gate.review_input(texts, state=self._state(ctx, texts))
            return request_context

        async def after_model_request(
            self, ctx: Any, *, request_context: Any, response: Any
        ) -> Any:
            self.gate.review_output(_response_texts(response))
            return response

        def get_wrapper_model(self, model: Any) -> Any:
            return _ScreenedStreamModel(model, self) if self.buffer_streams else None

        def _screen_response(self, response: Any, parameters: Any) -> None:
            output_tools = {tool.name for tool in getattr(parameters, "output_tools", ())}
            payload: list[Any] = list(_response_texts(response))
            payload.extend(
                part.args
                for part in response.parts
                if isinstance(part, messages.ToolCallPart) and part.tool_name in output_tools
            )
            self.gate.review_output(payload)

        # -- output -----------------------------------------------------------

        async def after_output_validate(self, ctx: Any, *, output_context: Any, output: Any) -> Any:
            # Plain text was screened at `after_model_request`; a deferred-request
            # bundle is not an answer (its calls are screened when they run).
            if not isinstance(output, (str, tools.DeferredToolRequests)):
                self.gate.review_output(output)
            return output

        # -- tools ------------------------------------------------------------

        async def wrap_tool_execute(
            self, ctx: Any, *, call: Any, tool_def: Any, args: Any, handler: Any
        ) -> Any:
            with traced(self.observer, self.gate.agent_id, f"pydantic_ai.tool:{tool_def.name}"):
                return await handler(args)

        async def before_tool_execute(
            self, ctx: Any, *, call: Any, tool_def: Any, args: Any
        ) -> Any:
            decision = self.gate.check_tool_call(tool_def.name, args, state=self._state(ctx, []))
            self._observe(decision)
            action = decision.resolve(approvals_supported=self.approvals)
            if action == "ask":
                # Set only by the caller's DeferredToolResults on resume; the
                # model cannot forge it. It satisfies the ask and nothing else:
                # this call was re-screened above and a block still blocks.
                if getattr(ctx, "tool_call_approved", False):
                    return args
                self._asked.append(decision)
                raise exceptions.ApprovalRequired(
                    metadata={
                        "adapt_agent": {
                            "reason": decision.reason,
                            "threats": list(decision.threats),
                            "agent_id": decision.agent_id,
                        }
                    }
                )
            if action == "block":
                self._refuse(decision)
            return args

        async def after_tool_execute(
            self, ctx: Any, *, call: Any, tool_def: Any, args: Any, result: Any
        ) -> Any:
            decision = self.gate.check_tool_result(
                tool_def.name, result, state=self._state(ctx, [])
            )
            self._observe(decision)
            if decision.blocked:
                self._refuse(decision)
            return result

        def _refuse(self, decision: ToolDecision) -> None:
            if self.on_tool_violation == "refuse":
                raise exceptions.ToolFailed(decision.reason)
            raise SecurityBlockedError(decision.reason, list(decision.threats))

        def _observe(self, decision: ToolDecision) -> None:
            log = getattr(self.observer, "log", None)
            if not decision.threats or not callable(log):
                return
            log(
                "warning",
                decision.reason,
                decision.agent_id,
                {
                    "action": decision.action,
                    "tool": decision.tool_name,
                    "threats": list(decision.threats),
                    "report_only": decision.report_only,
                },
            )

    def _response_texts(response: Any) -> list[str]:
        return [part.content for part in response.parts if isinstance(part, messages.TextPart)]

    GovernanceCapability.__module__ = __name__
    _CAPABILITY_CLASS = GovernanceCapability
    return GovernanceCapability


def governance_output_validator(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    tool_policy: ToolPolicy | None = None,
) -> Callable[[Any], Any]:
    """Build an output validator that screens a Pydantic AI result.

    The final output only. :func:`governance_capability` covers input, tool
    calls, streams and output, and is what new code should use; this remains
    for agents already wired through ``output_validator``.

    Returns:
        A single-argument callable returning the output unchanged, or raising
        :class:`~adapt_agent.exceptions.SecurityBlockedError`. Pydantic AI
        accepts validators with or without a leading ``RunContext``; this one
        takes just the output, which is the form it applies to any single-arg
        callable.

    Register it with :func:`install_governance`, or by hand::

        agent.output_validator(governance_output_validator(firewall=fw))

    Raises:
        ValueError: If a ``policy_enforcer``, ``defense`` or ``tool_policy`` is
            supplied (directly or on a ``gate``). This seam cannot honour them,
            and silently ignoring one would be worse than refusing it.
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
    _reject_unsupported(resolved)

    def adapt_governance_validator(output: Any) -> Any:
        resolved.review_output(output)
        return output

    return adapt_governance_validator


def _reject_unsupported(gate: GovernanceGate) -> None:
    """Refuse the controls an output validator cannot honour.

    An output validator sees only the result, so three of the gate's controls
    are structurally out of reach there:

    * ``policy_enforcer`` gates on agent *state*, which is never in scope;
    * ``defense`` is *input* analysis -- :meth:`GovernanceGate.scan_output` runs
      the firewall only, deliberately, since adversarial-instruction detection
      over an answer flags an agent legitimately quoting an instruction;
    * ``tool_policy`` governs tool calls, which have all run by then.

    Accepting any of them and quietly doing nothing is the failure mode that
    makes a security control dangerous: it looks configured. So this raises and
    names the seam that honours all three, :func:`governance_capability`.
    """
    unsupported = [
        name
        for name, value in (
            ("policy_enforcer", gate.policy_enforcer),
            ("defense", gate.defense),
            ("tool_policy", gate.tool_policy),
        )
        if value is not None
    ]
    if unsupported:
        raise ValueError(
            f"Pydantic AI's output validator cannot honour {', '.join(unsupported)}: "
            "it sees only the final output -- not agent state, input, or tool calls. "
            "Use governance_capability(...) instead, given to Agent(capabilities=[...]) "
            "or agent.run(..., capabilities=[...]); it screens input, tool calls and "
            "output. See adapt_agent.integrations.pydantic_ai."
        )


def install_governance(
    agent: Any,
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    tool_policy: ToolPolicy | None = None,
) -> Any:
    """Register output governance on an existing ``agent`` in place and return it.

    This covers the **final output** only. Pydantic AI's public API attaches a
    capability -- the seam that sees input, tool calls and streams -- at
    construction or per run, never to an agent that already exists, so for full
    coverage use :func:`governance_capability`::

        agent = Agent(model, capabilities=[governance_capability(firewall=fw)])
        result = await existing_agent.run(prompt, capabilities=[governance_capability(...)])

    Raises:
        TypeError: If ``agent`` has no ``output_validator`` (i.e. it is not a
            Pydantic AI agent).
        ValueError: If a ``policy_enforcer``, ``defense`` or ``tool_policy`` is
            supplied -- see :func:`governance_output_validator`.
    """
    register = getattr(agent, "output_validator", None)
    if not callable(register):
        raise TypeError(
            "install_governance expects a Pydantic AI Agent (with an "
            f"`output_validator` method), got {type(agent).__name__}."
        )
    register(
        governance_output_validator(
            gate=gate,
            firewall=firewall,
            defense=defense,
            policy_enforcer=policy_enforcer,
            block_on_violation=block_on_violation,
            agent_id=agent_id,
            tool_policy=tool_policy,
        )
    )
    return agent


# `GovernanceCapability` is reachable as an attribute (built on first access)
# but deliberately not listed: a star-import would then import the SDK.
__all__ = [
    "deps_state",
    "governance_capability",
    "governance_output_validator",
    "install_governance",
]
