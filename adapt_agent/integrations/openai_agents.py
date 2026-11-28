"""Governance bound to the OpenAI Agents SDK's own seams.

The SDK has a seam for every boundary ADAPT-Agent governs, and this module binds
the gate to each of them. Start here:

.. code-block:: python

    from agents import Agent, Runner
    from adapt_agent.core import ToolPolicy
    from adapt_agent.integrations.openai_agents import (
        govern_tools, governance_guardrails, governance_run_hooks,
    )

    agent = Agent(name="triage", tools=[...], handoffs=[billing],
                  **governance_guardrails(firewall=fw, agent_id="triage"))
    govern_tools(agent, firewall=fw, tool_policy=ToolPolicy(require_approval=["refund"]))
    result = await Runner.run(agent, text, hooks=governance_run_hooks(firewall=fw))

==========================  ==================================================
Boundary                    Seam (factory)
==========================  ==================================================
Each tool call / result     ``ToolInputGuardrail`` / ``ToolOutputGuardrail`` on
                            every ``FunctionTool`` and ``MCPServer``
                            (:func:`govern_tools`, :func:`governance_tool_guardrails`)
"Ask" decisions             the SDK's ``needs_approval`` -> ``RunResult.interruptions``
                            -> ``RunState.approve``/``reject`` -> resume
Every model call, every     ``RunHooks.on_llm_start`` / ``on_agent_end`` passed to
agent (handoffs included)   ``Runner.run(hooks=...)`` (:func:`governance_run_hooks`)
Model input, centrally,     ``RunConfig.call_model_input_filter``
with redaction              (:func:`governance_model_input_filter`)
Run input / final answer    ``InputGuardrail`` / ``OutputGuardrail`` on the entry
                            agent (:func:`governance_guardrails`)
One agent's model calls     ``AgentHooks`` (:func:`governance_agent_hooks`)
Streamed output             :func:`govern_stream` over ``Runner.run_streamed``
==========================  ==================================================

**Guardrails do not cover a handoff target.** The SDK runs input guardrails for
the *starting* agent only -- ``run.py`` gates them on ``current_turn == 0``, and
``RunConfig.input_guardrails`` is first-turn-only as well -- so a specialist
reached by a handoff never runs its own. Verified against the installed SDK by
driving a real handoff::

    input guardrails that ran: ['triage']
    handoff happened: True (final agent: specialist)
    specialist's own input guardrail ran: False

:func:`governance_run_hooks` is the run-level answer: one object passed to
``Runner.run(hooks=...)`` fires for every agent the run reaches.

**Streaming.** ``Runner.run_streamed`` hands text deltas to the consumer as the
model produces them, and every post-model check -- output guardrails, hooks --
fires only after the response is complete: a blocked answer has already been
read by then. :func:`govern_stream` is the fix: it holds each model response's
text events back until the whole response has been screened, and releases them
only if it is clean. That trades first-token latency for the guarantee, per
model response; non-text events (tool-call progress, agent changes) still pass
straight through.

**What is not covered.** Tools the SDK executes on OpenAI's servers --
``WebSearchTool``, ``FileSearchTool``, ``CodeInterpreterTool``,
``ImageGenerationTool``, and a ``HostedMCPTool`` without an
``on_approval_request`` callback -- run with no client-side seam before or
after the call. ``ComputerTool`` and ``LocalShellTool`` have no per-call
approval or guardrail seam either. Their *results* still reach the next model
call through ``on_llm_start``, so the hooks screen them there, and the hooks'
``on_tool_start`` applies a :class:`~adapt_agent.core.ToolPolicy` to the local
ones by name; :func:`govern_tools` warns about each such tool it meets.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import warnings
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from adapt_agent.core.governance import ALLOW, ASK, BLOCK, GovernanceGate, ToolDecision
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
    from adapt_agent.core.governance import ToolPolicy
    from adapt_agent.core.policy import PolicyEnforcer
    from adapt_agent.observability import AgentObserver
    from adapt_agent.security.firewall import Firewall


#: Attribute set on every callable this module installs into the SDK, so a
#: second :func:`govern_tools` replaces rather than stacks, and the hooks can
#: tell a tool that is already governed at its own seam from one that is not.
_MARKER = "__adapt_agent_governance__"

#: Name given to the guardrail objects, as the SDK reports them in results.
_GUARDRAIL_NAME = "adapt-agent"

#: ``on_tool_violation`` values.
REFUSE = "refuse"
RAISE = "raise"
_TOOL_VIOLATION_MODES = (REFUSE, RAISE)

#: ``governance_model_input_filter(on_violation=...)`` values.
REDACT = "redact"
_FILTER_VIOLATION_MODES = (RAISE, REDACT)

#: Observer operation labels.
_OP_TOOL_CALL = "openai_agents.tool_call"
_OP_TOOL_RESULT = "openai_agents.tool_result"
_OP_MODEL_INPUT = "openai_agents.model_input"
_OP_MODEL_OUTPUT = "openai_agents.model_output"
_OP_STREAM = "openai_agents.stream_response"

#: Raw stream events that carry model text: held back until screened.
_TEXT_EVENT_TYPES = frozenset(
    {
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_text.annotation.added",
        "response.refusal.delta",
        "response.refusal.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
    }
)
#: Output items whose ``added``/``done`` events carry model text.
_TEXT_ITEM_TYPES = frozenset({"message", "reasoning"})
#: Raw stream events that close one model response.
_RESPONSE_END_TYPES = frozenset({"response.completed", "response.failed", "response.incomplete"})


# -- shared helpers -------------------------------------------------------------


_F = TypeVar("_F", bound=Callable[..., Any])


def _mark(fn: _F) -> _F:
    setattr(fn, _MARKER, True)
    return fn


def _is_ours(obj: Any) -> bool:
    fn = getattr(obj, "guardrail_function", obj)
    return bool(getattr(fn, _MARKER, False))


def _check_tool_mode(on_tool_violation: str) -> None:
    if on_tool_violation not in _TOOL_VIOLATION_MODES:
        raise ValueError(
            f"on_tool_violation must be one of {_TOOL_VIOLATION_MODES!r}, "
            f"not {on_tool_violation!r}"
        )


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _parse_arguments(raw: Any) -> Any:
    """The SDK hands tool arguments over as a JSON string; scan the parsed form.

    Parsed, a rule can read ``tool['arguments']['amount']``. A string that is not
    JSON is scanned as-is rather than dropped: unparseable is not the same as
    harmless.
    """
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    if dataclasses.is_dataclass(raw) and not isinstance(raw, type):
        return dataclasses.asdict(raw)
    return raw


def _tool_state(context: Any) -> dict[str, Any]:
    """Policy state at the tool seam: the run context's keys, and no messages.

    A tool hook sees one call, not the conversation, so ``state['messages']`` is
    empty here; ``state[<key>]`` is whatever the caller passed as
    ``Runner.run(..., context={...})``.
    """
    return as_state([], **context_state(context))


def _info(decision: ToolDecision) -> dict[str, Any]:
    """``output_info`` for a tool guardrail: what a caught tripwire reports."""
    return {
        "agent_id": decision.agent_id,
        "tool": decision.tool_name,
        "action": decision.action,
        "threats": list(decision.threats),
        "report_only": decision.report_only,
    }


def _traced_decision(
    observer: Any, agent_id: str, operation: str, check: Callable[[], ToolDecision]
) -> ToolDecision:
    """Run ``check`` inside an observer span that records a refusal as an error.

    The gate's ``check_*`` methods never raise, so the span would read
    "completed" for a call that was refused. Raising inside the span and
    catching it outside records the verdict without changing the control flow.
    """
    decision: ToolDecision | None = None
    try:
        with traced(observer, agent_id, operation):
            decision = check()
            if not decision.allowed:
                raise SecurityBlockedError(decision.reason, list(decision.threats))
    except SecurityBlockedError:
        if decision is None:  # pragma: no cover - check() never raises
            raise
    return decision


def _call_id(context: Any) -> str | None:
    value = getattr(context, "tool_call_id", None)
    return value if isinstance(value, str) else None


def _call_key(call_id: str, arguments: Any) -> tuple[str, str]:
    """Identify one call by its id *and* its arguments.

    A call id is model-generated and tools are shared across runs, so an id
    alone could pair one call's verdict with another call's arguments.
    """
    try:
        rendered = json.dumps(_parse_arguments(arguments), sort_keys=True, default=repr)
    except (TypeError, ValueError):
        rendered = repr(arguments)
    return call_id, rendered


class _Bounded(OrderedDict[Any, Any]):
    """An insertion-ordered dict that forgets its oldest entries past ``limit``.

    Holds per-call bookkeeping for calls that may never come back (a rejected
    approval is not seen again), so it must not grow for the life of a process.
    """

    def __init__(self, limit: int = 1024) -> None:
        super().__init__()
        self.limit = limit

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        while len(self) > self.limit:
            self.popitem(last=False)


def _approved(context: Any, tool_name: str, call_id: str | None) -> bool:
    """Whether the SDK has a human (or app) approval on record for this call."""
    if call_id is None:
        return False
    is_tool_approved = getattr(context, "is_tool_approved", None)
    if not callable(is_tool_approved):
        return False
    try:
        return is_tool_approved(tool_name, call_id) is True
    except Exception:  # an unknown SDK shape is not an approval
        return False


def _tool_name(tool: Any) -> str:
    name = getattr(tool, "name", None)
    if isinstance(name, str):
        return name
    return tool if isinstance(tool, str) else type(tool).__name__


def _rule_can_ask(condition: str, tool_name: str) -> bool:
    """Whether a ``require_approval`` rule could fire for ``tool_name``.

    A rule that pins ``tool['name']`` to literals can only fire for the tools it
    names; any other rule might fire for any tool. Erring towards "could" only
    costs an approval prompt; erring the other way is safe too, because an ask
    on a tool without the approval callable fails closed to a block.
    """
    if "tool['name']" not in condition and 'tool["name"]' not in condition:
        return True
    return f"'{tool_name}'" in condition or f'"{tool_name}"' in condition


def _gate_can_ask(gate: GovernanceGate, tool_name: str) -> bool:
    """Whether this gate could ever answer "ask" for ``tool_name``.

    Installing a callable ``needs_approval`` is not free in this SDK. For a
    decorated Python tool whose arguments it cannot inspect without running
    user code, the SDK demands manual approval *before* the callable runs -- and
    it treats *every* tool that way when any Pydantic plugin is installed
    (``logfire``'s, pulled in by ``pydantic-ai``, is a common one), because a
    plugin wraps the validator. So the callable goes only on tools the gate can
    actually send for approval.
    """
    policy = gate.tool_policy
    if policy is not None and policy.verdict(tool_name)[0] == ASK:
        return True
    enforcer = gate.policy_enforcer
    if enforcer is None:
        return False
    try:
        rules = enforcer.list_rules()
    except Exception:  # pragma: no cover - a duck-typed enforcer
        return True
    return any(
        rule.get("action") == "require_approval"
        and _rule_can_ask(str(rule.get("condition", "")), tool_name)
        for rule in rules
    )


# -- tool governance ------------------------------------------------------------


class _ToolGovernor:
    """One gate's tool-call governance, expressed in the SDK's tool callbacks.

    Holds the per-call bookkeeping that ties ``needs_approval`` to the input
    guardrail: a call the gate *asked* about must pause for a human, and must
    not be refused by the guardrail while it is waiting to be asked (the SDK can
    run tool input guardrails before approval, via
    ``ToolExecutionConfig(pre_approval_tool_input_guardrails=True)``).
    """

    def __init__(
        self,
        gate: GovernanceGate,
        *,
        observer: Any,
        on_tool_violation: str,
        approvals: bool,
        screen_results: bool,
    ) -> None:
        _check_tool_mode(on_tool_violation)
        self.gate = gate
        self.observer = observer
        self.on_tool_violation = on_tool_violation
        self.approvals = approvals
        self.screen_results = screen_results
        # call -> the decision `needs_approval` reached, reused by the guardrail
        # so one call is evaluated (and its violations recorded) once.
        self._decisions: _Bounded = _Bounded()
        # calls `needs_approval` sent for approval and not yet seen approved.
        self._asked: _Bounded = _Bounded()

    # -- the gate, traced ------------------------------------------------------

    def call_decision(self, tool_name: str, arguments: Any, context: Any) -> ToolDecision:
        return _traced_decision(
            self.observer,
            self.gate.agent_id,
            _OP_TOOL_CALL,
            lambda: self.gate.check_tool_call(
                tool_name, _parse_arguments(arguments), state=_tool_state(context)
            ),
        )

    def result_decision(self, tool_name: str, result: Any, context: Any) -> ToolDecision:
        return _traced_decision(
            self.observer,
            self.gate.agent_id,
            _OP_TOOL_RESULT,
            lambda: self.gate.check_tool_result(tool_name, result, state=_tool_state(context)),
        )

    # -- guardrails ------------------------------------------------------------

    def _refusal(self, agents: Any, decision: ToolDecision, message: str) -> Any:
        output = agents.ToolGuardrailFunctionOutput
        if self.on_tool_violation == RAISE:
            return output.raise_exception(_info(decision))
        return output.reject_content(message, _info(decision))

    def guardrails(self) -> dict[str, list[Any]]:
        agents = optional_import("agents", "openai-agents", "governance_tool_guardrails")

        async def adapt_tool_input(data: Any) -> Any:
            ctx = data.context
            name = ctx.tool_name
            call_id = _call_id(ctx)
            key = _call_key(call_id, ctx.tool_arguments) if call_id is not None else None
            cached = self._decisions.pop(key, None) if key is not None else None
            if cached is not None:
                decision = _traced_decision(
                    self.observer, self.gate.agent_id, _OP_TOOL_CALL, lambda: cached
                )
            else:
                decision = self.call_decision(name, ctx.tool_arguments, ctx)
            action = decision.action
            if action == ASK:
                if _approved(ctx, name, call_id):
                    self._asked.pop(key, None)
                    action = ALLOW
                elif key is not None and key in self._asked:
                    # Pre-approval guardrail pass: `needs_approval` already sent
                    # this call for approval, and refusing it here would replace
                    # the human's decision with ours.
                    action = ALLOW
                else:
                    # An ask with no approval on record and none pending: the
                    # tool would run unseen. Fail closed.
                    action = decision.resolve(approvals_supported=False)
            if action == ALLOW:
                return agents.ToolGuardrailFunctionOutput.allow(_info(decision))
            return self._refusal(agents, decision, decision.reason)

        guardrails: dict[str, list[Any]] = {
            "tool_input_guardrails": [
                agents.ToolInputGuardrail(_mark(adapt_tool_input), name=_GUARDRAIL_NAME)
            ],
            "tool_output_guardrails": [],
        }
        if self.screen_results:

            async def adapt_tool_output(data: Any) -> Any:
                ctx = data.context
                decision = self.result_decision(ctx.tool_name, data.output, ctx)
                if decision.allowed:
                    return agents.ToolGuardrailFunctionOutput.allow(_info(decision))
                # The withheld result is replaced, not merely flagged: the model
                # is told a result was refused, never what it said.
                message = (
                    f"Tool result from {decision.tool_name!r} withheld by ADAPT-Agent: "
                    f"{', '.join(decision.threats)}."
                )
                return self._refusal(agents, decision, message)

            guardrails["tool_output_guardrails"].append(
                agents.ToolOutputGuardrail(_mark(adapt_tool_output), name=_GUARDRAIL_NAME)
            )
        return guardrails

    # -- approvals --------------------------------------------------------------

    def needs_approval(self, tool_name: str, original: Any) -> Callable[..., Awaitable[bool]]:
        """A ``FunctionTool.needs_approval`` that pauses on the gate's "ask".

        Composes with the tool's own setting: approval is needed if *either* the
        tool or the gate wants it. A call the gate blocks is not sent to a
        human -- the input guardrail refuses it -- because approving it would
        change nothing.
        """
        original = getattr(original, "__adapt_agent_original__", original)

        async def needs_approval(run_context: Any, params: Any, call_id: str) -> bool:
            if callable(original):
                own = bool(await _maybe_await(original(run_context, params, call_id)))
            else:
                own = bool(original)
            decision = self.gate.check_tool_call(tool_name, params, state=_tool_state(run_context))
            key = _call_key(call_id, params)
            self._decisions[key] = decision
            if decision.blocked:
                return False  # the guardrail refuses it; a human could change nothing
            if own or (decision.needs_approval and self.approvals):
                self._asked[key] = True
                return True
            return False

        needs_approval.__adapt_agent_original__ = original  # type: ignore[attr-defined]
        return _mark(needs_approval)

    def approval_pair(
        self, tool_name: str, original_needs: Any, original_on_approval: Any
    ) -> tuple[Callable[..., Awaitable[bool]], Callable[..., Awaitable[Any]]]:
        """``needs_approval`` + ``on_approval`` for shell, apply_patch and custom tools.

        Those tools have no guardrail seam, only approvals, so a *block* is
        routed through the approval flow and rejected automatically by
        ``on_approval`` with the gate's reason: the call never runs and no human
        is asked. An "ask" is left for a human (or the app's own
        ``on_approval``). ``on_tool_violation="raise"`` raises from
        ``on_approval``, which aborts the run.
        """
        original_needs = getattr(original_needs, "__adapt_agent_original__", original_needs)
        original_on_approval = getattr(
            original_on_approval, "__adapt_agent_original__", original_on_approval
        )
        decisions: _Bounded = _Bounded()

        async def needs_approval(run_context: Any, payload: Any, call_id: str) -> bool:
            if callable(original_needs):
                own = bool(await _maybe_await(original_needs(run_context, payload, call_id)))
            else:
                own = bool(original_needs)
            decision = self.call_decision(tool_name, payload, run_context)
            decisions[call_id] = decision
            return own or not decision.allowed

        async def on_approval(run_context: Any, approval_item: Any) -> Any:
            call_id = getattr(approval_item, "call_id", None)
            decision = decisions.pop(call_id, None) if isinstance(call_id, str) else None
            if decision is None:
                decision = self.call_decision(
                    tool_name, getattr(approval_item, "arguments", None), run_context
                )
            action = decision.action
            if action == ASK and not self.approvals:
                action = decision.resolve(approvals_supported=False)
            if action == BLOCK:
                if self.on_tool_violation == RAISE:
                    raise SecurityBlockedError(decision.reason, list(decision.threats))
                return {"approve": False, "reason": decision.reason}
            if original_on_approval is not None:
                return await _maybe_await(original_on_approval(run_context, approval_item))
            return {}  # no verdict: the SDK surfaces an interruption for a human

        needs_approval.__adapt_agent_original__ = original_needs  # type: ignore[attr-defined]
        on_approval.__adapt_agent_original__ = original_on_approval  # type: ignore[attr-defined]
        return _mark(needs_approval), _mark(on_approval)

    def hosted_mcp_approval(self, original: Any) -> Callable[..., Awaitable[Any]]:
        """Wrap a ``HostedMCPTool.on_approval_request``: the gate decides first."""
        original = getattr(original, "__adapt_agent_original__", original)

        async def on_approval_request(request: Any) -> Any:
            data = getattr(request, "data", None)
            name = getattr(data, "name", None) or "hosted_mcp"
            decision = self.call_decision(
                name, getattr(data, "arguments", None), getattr(request, "ctx_wrapper", None)
            )
            if not decision.allowed and not (decision.needs_approval and self.approvals):
                if self.on_tool_violation == RAISE:
                    raise SecurityBlockedError(decision.reason, list(decision.threats))
                return {"approve": False, "reason": decision.reason}
            return await _maybe_await(original(request))

        on_approval_request.__adapt_agent_original__ = original  # type: ignore[attr-defined]
        return _mark(on_approval_request)


def _tool_gate(
    *,
    gate: GovernanceGate | None,
    firewall: Firewall | None,
    defense: AdversarialDefense | None,
    policy_enforcer: PolicyEnforcer | None,
    block_on_violation: bool,
    agent_id: str,
    tool_policy: ToolPolicy | None,
) -> GovernanceGate:
    """`build_gate`, refusing a ``tool_policy`` it would silently drop."""
    if gate is not None and tool_policy is not None:
        # `build_gate` keeps a supplied gate's own controls; a tool_policy given
        # alongside it would be accepted and do nothing.
        raise ValueError(
            "Pass tool_policy on the GovernanceGate itself, not alongside gate=: "
            "a supplied gate carries its own controls."
        )
    return build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
        tool_policy=tool_policy,
    )


def _merge(existing: Iterable[Any] | None, ours: list[Any]) -> list[Any]:
    """Our guardrails replace any earlier ADAPT-Agent ones; the app's stay first."""
    return [g for g in (existing or []) if not _is_ours(g)] + ours


def _governor(
    *,
    gate: GovernanceGate | None,
    firewall: Firewall | None,
    defense: AdversarialDefense | None,
    policy_enforcer: PolicyEnforcer | None,
    block_on_violation: bool,
    agent_id: str,
    tool_policy: ToolPolicy | None,
    observer: Any,
    on_tool_violation: str,
    approvals: bool,
    screen_results: bool,
) -> _ToolGovernor:
    return _ToolGovernor(
        _tool_gate(
            gate=gate,
            firewall=firewall,
            defense=defense,
            policy_enforcer=policy_enforcer,
            block_on_violation=block_on_violation,
            agent_id=agent_id,
            tool_policy=tool_policy,
        ),
        observer=observer,
        on_tool_violation=on_tool_violation,
        approvals=approvals,
        screen_results=screen_results,
    )


def governance_tool_guardrails(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    tool_policy: ToolPolicy | None = None,
    observer: AgentObserver | None = None,
    on_tool_violation: str = REFUSE,
    screen_results: bool = True,
) -> dict[str, list[Any]]:
    """Tool guardrails that put every tool call and result through the gate.

    Spread into anything that takes the SDK's tool guardrail lists::

        @function_tool(**governance_tool_guardrails(firewall=fw))
        def fetch(url: str) -> str: ...

        MCPServerStdio(params, **governance_tool_guardrails(firewall=fw))

    Before the tool runs, ``gate.check_tool_call(name, arguments)`` (tool
    policy, firewall and defense over the arguments, tool-scoped policy rules);
    after it returns, ``gate.check_tool_result(name, result)`` -- the
    indirect-injection seam, since whatever a tool fetched reaches the next
    model call as trusted-looking context.

    An "ask" is allowed only for a call the SDK has an approval on record for;
    otherwise it fails closed to a block. These guardrails alone therefore
    never *pause* a run -- use :func:`govern_tools`, which also installs a
    gate-driven ``needs_approval``, or the SDK's own ``needs_approval`` /
    ``require_approval`` so a human is asked first.

    Args:
        gate: A pre-built gate; otherwise built from the controls below.
        firewall: Screens tool arguments and tool results.
        defense: Adversarial analysis of the same.
        policy_enforcer: Its tool-scoped rules (conditions naming ``tool``) are
            evaluated against ``{"name", "arguments"}`` and a state holding the
            run context's keys (``Runner.run(..., context={...})``);
            ``state['messages']`` is empty at this seam.
        block_on_violation: ``False`` is report-only: every call runs and the
            threats are still in the guardrail's ``output_info``.
        agent_id: Named in the refusal and in ``output_info``.
        tool_policy: Name-based allow / deny / approval lists.
        observer: Records one span per tool call and per tool result; a refusal
            is recorded with status ``"error"``.
        on_tool_violation: ``"refuse"`` (default) returns the SDK's own
            ``reject_content`` -- the model is told the call was refused, the
            tool does not run (or its result is withheld) and the run goes on.
            ``"raise"`` returns ``raise_exception``, which aborts the run with
            ``ToolInputGuardrailTripwireTriggered`` /
            ``ToolOutputGuardrailTripwireTriggered``; the threats are on
            ``exc.output.output_info["threats"]``. Refusal is the default
            because it is what the SDK's tool guardrails are built for: the
            model can recover, and parallel tool calls in the same turn are not
            thrown away with the run.
        screen_results: Also attach the output guardrail.

    Returns:
        ``{"tool_input_guardrails": [...], "tool_output_guardrails": [...]}``.

    Raises:
        ImportError: If ``openai-agents`` is not installed.
        ValueError: On an unknown ``on_tool_violation``.
    """
    return _governor(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
        tool_policy=tool_policy,
        observer=observer,
        on_tool_violation=on_tool_violation,
        approvals=False,
        screen_results=screen_results,
    ).guardrails()


def _is_mcp_server(obj: Any) -> bool:
    return (
        hasattr(obj, "tool_input_guardrails")
        and hasattr(obj, "list_tools")
        and hasattr(obj, "call_tool")
    )


def govern_tools(
    target: Any,
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    tool_policy: ToolPolicy | None = None,
    observer: AgentObserver | None = None,
    on_tool_violation: str = REFUSE,
    approvals: bool = True,
    screen_results: bool = True,
) -> Any:
    """Attach tool governance to every tool an agent can reach, in place.

    ``target`` may be an ``Agent``, an ``MCPServer``, a single tool, or a list
    of any of those. From an agent it follows **handoffs** (``Agent`` or
    ``Handoff``), **agents-as-tools** (``Agent.as_tool``) and **MCP servers**
    (``Agent.mcp_servers``), visiting each object once, so a shared tool or a
    cyclic handoff graph is governed exactly once. Calling it again replaces
    the earlier ADAPT-Agent callbacks rather than stacking them.

    Per tool kind:

    * ``FunctionTool`` (``@function_tool``, MCP tools, agents-as-tools): the
      guardrails of :func:`governance_tool_guardrails`, plus -- when
      ``approvals`` and the gate can answer "ask" for that tool -- a
      gate-driven ``needs_approval``, so an "ask" interrupts the run
      (``result.interruptions``) for ``RunState.approve`` / ``reject`` and a
      resume with ``Runner.run(agent, state)``.
    * ``MCPServer``: the guardrails, on its ``tool_input_guardrails`` /
      ``tool_output_guardrails`` (the SDK copies them onto every MCP tool).
      A human approval for an MCP "ask" comes from the server's own
      ``require_approval=``; without one, an ask fails closed to a block.
    * ``ShellTool``, ``ApplyPatchTool``, ``CustomTool``: no guardrail seam, so
      ``needs_approval`` + ``on_approval`` -- a block is auto-rejected with the
      gate's reason (or raises, in ``"raise"`` mode); an ask waits for a human.
    * ``HostedMCPTool``: its ``on_approval_request`` is wrapped so the gate
      decides first. Without one, OpenAI runs the call server-side with no
      client seam.
    * Anything else (hosted search / code / image tools, ``ComputerTool``,
      ``LocalShellTool``) has no per-call seam; a ``RuntimeWarning`` names each
      one, and :func:`governance_run_hooks` covers what it can of them.

    Args:
        target: What to govern (see above).
        approvals: Map "ask" onto the SDK's approval flow. ``False`` turns
            every ask into a block (``ToolDecision.resolve``), for a deployment
            with nobody to approve.
        gate, firewall, defense, policy_enforcer, block_on_violation, agent_id,
        tool_policy, observer, on_tool_violation, screen_results: As for
            :func:`governance_tool_guardrails`.

    Returns:
        ``target``, governed.
    """
    agents = optional_import("agents", "openai-agents", "govern_tools")
    governor = _governor(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
        tool_policy=tool_policy,
        observer=observer,
        on_tool_violation=on_tool_violation,
        approvals=approvals,
        screen_results=screen_results,
    )
    guardrails = governor.guardrails()
    seen: set[int] = set()
    uncovered: list[str] = []

    def govern_tool(tool: Any) -> None:
        if isinstance(tool, agents.FunctionTool):
            tool.tool_input_guardrails = _merge(
                tool.tool_input_guardrails, guardrails["tool_input_guardrails"]
            )
            tool.tool_output_guardrails = _merge(
                tool.tool_output_guardrails, guardrails["tool_output_guardrails"]
            )
            original = getattr(tool.needs_approval, "__adapt_agent_original__", None)
            if original is None:
                original = tool.needs_approval
            if approvals and (original is not False or _gate_can_ask(governor.gate, tool.name)):
                tool.needs_approval = governor.needs_approval(tool.name, original)
            else:
                tool.needs_approval = original
            inner_agent = getattr(tool, "_agent_instance", None)
            if inner_agent is not None:
                visit(inner_agent)
        elif isinstance(tool, (agents.ShellTool, agents.ApplyPatchTool, agents.CustomTool)):
            environment = getattr(tool, "environment", None)
            if (
                isinstance(tool, agents.ShellTool)
                and isinstance(environment, Mapping)
                and environment.get("type", "local") != "local"
            ):
                # A hosted shell runs server-side; the SDK refuses approvals on it.
                uncovered.append(_tool_name(tool))
                return
            tool.needs_approval, tool.on_approval = governor.approval_pair(
                _tool_name(tool), tool.needs_approval, tool.on_approval
            )
        elif isinstance(tool, agents.HostedMCPTool) and tool.on_approval_request is not None:
            tool.on_approval_request = governor.hosted_mcp_approval(tool.on_approval_request)
        else:
            uncovered.append(_tool_name(tool))

    def govern_server(server: Any) -> None:
        server.tool_input_guardrails = _merge(
            server.tool_input_guardrails, guardrails["tool_input_guardrails"]
        )
        server.tool_output_guardrails = _merge(
            server.tool_output_guardrails, guardrails["tool_output_guardrails"]
        )

    def visit(obj: Any) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)
        elif isinstance(obj, agents.Agent):
            for tool in obj.tools:
                visit(tool)
            for server in getattr(obj, "mcp_servers", None) or []:
                visit(server)
            for handoff in obj.handoffs:
                if isinstance(handoff, agents.Handoff):
                    ref = getattr(handoff, "_agent_ref", None)
                    visit(ref() if callable(ref) else None)
                else:
                    visit(handoff)
        elif _is_mcp_server(obj):
            govern_server(obj)
        else:
            govern_tool(obj)

    visit(target)
    if uncovered:
        warnings.warn(
            "ADAPT-Agent cannot govern these tools per call (no client-side tool "
            f"seam in the SDK): {sorted(set(uncovered))}. Pass governance_run_hooks() "
            "to Runner.run(hooks=...) to screen their results and apply the tool "
            "policy to them by name.",
            RuntimeWarning,
            stacklevel=2,
        )
    return target


# -- model-call hooks -----------------------------------------------------------


def _is_governed_tool(tool: Any, *, result: bool) -> bool:
    """Whether ``tool`` is already governed at its own seam by this module."""
    if _is_ours(getattr(tool, "on_approval", None)):
        # Shell / apply_patch / custom: the call is governed by the approval
        # pair; their result has no seam of its own, so the hook still screens it.
        return not result
    attr = "tool_output_guardrails" if result else "tool_input_guardrails"
    return any(_is_ours(g) for g in getattr(tool, attr, None) or [])


class _HookGovernance:
    """The checks shared by :func:`governance_agent_hooks` and :func:`governance_run_hooks`."""

    def __init__(
        self,
        gate: GovernanceGate,
        *,
        observer: Any,
        screen_output: bool,
        govern_tool_calls: bool,
    ) -> None:
        self.gate = gate
        self.observer = observer
        self.screen_output = screen_output
        self.govern_tool_calls = govern_tool_calls

    def model_input(self, context: Any, input_items: Any) -> None:
        # Policy state: the items about to reach the model as `messages`, plus
        # the run context's keys -- never the raw SDK wrapper, which a
        # fail-closed enforcer would read as "every key missing".
        resolved = self.gate
        with traced(self.observer, resolved.agent_id, _OP_MODEL_INPUT):
            resolved.review_input(
                input_items, state=as_state(input_items, **context_state(context))
            )

    def model_output(self, output: Any) -> None:
        resolved = self.gate
        if self.screen_output:
            with traced(self.observer, resolved.agent_id, _OP_MODEL_OUTPUT):
                resolved.review_output(output)

    def tool_start(self, context: Any, tool: Any) -> None:
        """Backstop for tools not governed at their own seam: abort on a block.

        A hook has no refusal channel, so a block here raises, which aborts the
        run before the tool executes (for a function tool the SDK wraps it in
        ``agents.UserError``; see :func:`find_security_block`).
        Function and custom tools arrive with a ``ToolContext`` carrying the
        arguments; computer and shell tools arrive with the run context only,
        so only the tool policy (by name) and tool rules that do not read
        arguments can decide them here.
        """
        if not self.govern_tool_calls or _is_governed_tool(tool, result=False):
            return
        name = _tool_name(tool)
        arguments = getattr(context, "tool_arguments", None)
        decision = _traced_decision(
            self.observer,
            self.gate.agent_id,
            _OP_TOOL_CALL,
            lambda: self.gate.check_tool_call(
                name, _parse_arguments(arguments), state=_tool_state(context)
            ),
        )
        action = decision.action
        if action == ASK:
            approved = _approved(context, name, _call_id(context))
            action = ALLOW if approved else decision.resolve(approvals_supported=False)
        if action == BLOCK:
            raise SecurityBlockedError(decision.reason, list(decision.threats))

    def tool_end(self, context: Any, tool: Any, result: Any) -> None:
        """Abort before a poisoned result from an ungoverned tool reaches the model."""
        if not self.govern_tool_calls or _is_governed_tool(tool, result=True):
            return
        name = _tool_name(tool)
        decision = _traced_decision(
            self.observer,
            self.gate.agent_id,
            _OP_TOOL_RESULT,
            lambda: self.gate.check_tool_result(name, result, state=_tool_state(context)),
        )
        if decision.blocked:
            raise SecurityBlockedError(decision.reason, list(decision.threats))


def _forward_inner(hooks: Any, base: type, inner: Any, governed: set[str]) -> None:
    # Every *other* lifecycle hook belongs to the app. `__getattr__` cannot
    # forward them: the base class defines them, so lookup never misses and the
    # app's would be silently shadowed by the SDK's no-ops -- configured, and
    # doing nothing. Bind them on the instance, where they win.
    for name in dir(base):
        if name.startswith("on_") and name not in governed and hasattr(inner, name):
            setattr(hooks, name, getattr(inner, name))


def governance_agent_hooks(
    *,
    gate: GovernanceGate | None = None,
    firewall: Any = None,
    defense: Any = None,
    policy_enforcer: Any = None,
    agent_id: str = "agent",
    screen_output: bool = True,
    inner: Any = None,
    block_on_violation: bool = True,
    tool_policy: ToolPolicy | None = None,
    observer: AgentObserver | None = None,
    govern_tool_calls: bool = True,
) -> Any:
    """Governance on one agent's ``AgentHooks`` -- prefer :func:`governance_run_hooks`.

    The SDK gates input guardrails on ``current_turn == 0``, so a specialist
    reached by a handoff never runs its own; these hooks run per agent, per
    model call. They cover only the agent they are attached to, though:
    :func:`governance_run_hooks` covers every agent of a run with one object
    and is the better default.

    ``on_llm_start`` fires before every model call and receives the input items
    about to be sent; ``on_end`` fires when the agent finishes and receives its
    answer. Confirmed against the installed SDK, including through a handoff::

        hooks fired: ['triage:on_start', 'triage:on_llm_start(items=1)',
                      'SPECIALIST:on_start', 'SPECIALIST:on_llm_start(items=3)']

    Because it runs per model call, it also screens tool results on their way
    back to the model -- the path that carries whatever a tool fetched.
    ``on_tool_start`` / ``on_tool_end`` add a tool-call backstop for tools not
    governed at their own seam by :func:`govern_tools` (see
    ``govern_tool_calls``).

    A block raises :class:`~adapt_agent.exceptions.SecurityBlockedError`, which
    aborts the run: a hook has no gentler refusal channel.

    Policy state: ``state['messages']`` is the list of input items about to be
    sent, and every key of the run context (``Runner.run(..., context={...})``)
    is merged in. At the tool hooks ``state['messages']`` is empty.

    Args:
        gate: A pre-built gate to share across agents; otherwise built from the
            controls below.
        firewall: Screens the items about to reach the model.
        defense: Adversarial analysis of those items.
        policy_enforcer: Evaluated against the items plus the run context.
        agent_id: Named in the raised error.
        screen_output: Also screen the agent's final answer, on ``on_end``.
        inner: An existing ``AgentHooks`` to delegate to, so an app's own
            lifecycle hooks still run. Governance is applied first.
        block_on_violation: ``False`` records without raising (report-only).
        tool_policy: Applied by the tool backstop.
        observer: One span per governed model call and tool call.
        govern_tool_calls: Check ``on_tool_start`` / ``on_tool_end`` for tools
            that :func:`govern_tools` did not reach (a dynamically added tool,
            an ungoverned MCP server, a computer or shell tool).

    Returns:
        An ``AgentHooks`` instance for ``Agent(hooks=...)``.

    Raises:
        ImportError: If ``openai-agents`` is not installed.
    """
    agents = optional_import("agents", "openai-agents", "governance_agent_hooks")
    governance = _HookGovernance(
        _tool_gate(
            gate=gate,
            firewall=firewall,
            defense=defense,
            policy_enforcer=policy_enforcer,
            block_on_violation=block_on_violation,
            agent_id=agent_id,
            tool_policy=tool_policy,
        ),
        observer=observer,
        screen_output=screen_output,
        govern_tool_calls=govern_tool_calls,
    )

    base: Any = agents.AgentHooks  # resolved at call time; never imported at module load

    class AdaptGovernanceHooks(base):
        async def on_llm_start(
            self, context: Any, agent: Any, system_prompt: Any, input_items: Any
        ) -> None:
            governance.model_input(context, input_items)
            if inner is not None:
                await inner.on_llm_start(context, agent, system_prompt, input_items)

        async def on_end(self, context: Any, agent: Any, output: Any) -> None:
            # The other half of the seam. Guardrails give a handoff target
            # neither input *nor* output screening; covering only the input made
            # this factory the very thing it was added to replace -- a control
            # that looks complete and is half missing.
            governance.model_output(output)
            if inner is not None:
                await inner.on_end(context, agent, output)

        async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
            governance.tool_start(context, tool)
            if inner is not None and hasattr(inner, "on_tool_start"):
                await inner.on_tool_start(context, agent, tool)

        async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
            governance.tool_end(context, tool, result)
            if inner is not None and hasattr(inner, "on_tool_end"):
                await inner.on_tool_end(context, agent, tool, result)

    hooks = AdaptGovernanceHooks()
    if inner is not None:
        _forward_inner(
            hooks, base, inner, {"on_llm_start", "on_end", "on_tool_start", "on_tool_end"}
        )
    return hooks


def governance_run_hooks(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    tool_policy: ToolPolicy | None = None,
    observer: AgentObserver | None = None,
    screen_output: bool = True,
    govern_tool_calls: bool = True,
    inner: Any = None,
) -> Any:
    """Governance on ``RunHooks``: one object covers **every agent in a run**.

    Pass it to ``Runner.run(agent, input, hooks=governance_run_hooks(...))``.
    Unlike guardrails (entry agent, first turn only) and ``AgentHooks`` (one
    agent each), run hooks fire for whichever agent is running -- the entry
    agent and every handoff target. An agent behind ``Agent.as_tool`` runs a
    *nested* run that does not inherit them (verified: the outer hooks saw only
    the outer agent's model calls) -- pass the same object to
    ``as_tool(hooks=...)`` to cover it. A block inside that nested run stops
    the nested run; the SDK then reports the failed tool to the *outer* model
    as a tool error, and the outer run continues.

    * ``on_llm_start`` -- before every model call, the input items about to be
      sent (user input, earlier answers, *tool results*) go through
      ``gate.review_input`` with policy evaluated.
    * ``on_agent_end`` -- each agent's final answer goes through
      ``gate.review_output``.
    * ``on_tool_start`` / ``on_tool_end`` -- a backstop for tools
      :func:`govern_tools` did not reach: a block aborts before the tool runs,
      and a flagged result aborts before it reaches the model.

    A block raises :class:`~adapt_agent.exceptions.SecurityBlockedError` and
    aborts the run. From a *function tool's* hooks the SDK re-raises it wrapped
    in ``agents.UserError`` -- the run still aborts and the tool does not run;
    :func:`find_security_block` recovers ours, threats included. The hooks
    cannot stream-gate: in a streamed run the text has
    already been delivered by ``on_agent_end`` -- wrap the stream with
    :func:`govern_stream`.

    Policy state, per hook: ``state['messages']`` is the input items
    (``on_llm_start``) or empty (tool hooks); every key of the run context
    (``Runner.run(..., context={...})``) is merged in.

    Args:
        gate, firewall, defense, policy_enforcer, block_on_violation, agent_id,
        tool_policy, observer, screen_output, govern_tool_calls: As for
            :func:`governance_agent_hooks`.
        inner: An existing ``RunHooks`` to delegate to after governance.

    Returns:
        A ``RunHooks`` instance.

    Raises:
        ImportError: If ``openai-agents`` is not installed.
    """
    agents = optional_import("agents", "openai-agents", "governance_run_hooks")
    governance = _HookGovernance(
        _tool_gate(
            gate=gate,
            firewall=firewall,
            defense=defense,
            policy_enforcer=policy_enforcer,
            block_on_violation=block_on_violation,
            agent_id=agent_id,
            tool_policy=tool_policy,
        ),
        observer=observer,
        screen_output=screen_output,
        govern_tool_calls=govern_tool_calls,
    )

    base: Any = agents.RunHooks

    class AdaptGovernanceRunHooks(base):
        async def on_llm_start(
            self, context: Any, agent: Any, system_prompt: Any, input_items: Any
        ) -> None:
            governance.model_input(context, input_items)
            if inner is not None:
                await inner.on_llm_start(context, agent, system_prompt, input_items)

        async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
            governance.model_output(output)
            if inner is not None:
                await inner.on_agent_end(context, agent, output)

        async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
            governance.tool_start(context, tool)
            if inner is not None:
                await inner.on_tool_start(context, agent, tool)

        async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
            governance.tool_end(context, tool, result)
            if inner is not None:
                await inner.on_tool_end(context, agent, tool, result)

    hooks = AdaptGovernanceRunHooks()
    if inner is not None:
        _forward_inner(
            hooks, base, inner, {"on_llm_start", "on_agent_end", "on_tool_start", "on_tool_end"}
        )
    return hooks


# -- central model-input filter -----------------------------------------------


def _redacted(item: Any, placeholder: str) -> Any | None:
    """A copy of one input item with its flagged text replaced, or ``None``.

    Only the two shapes whose text is the whole point of the item are redacted:
    a tool result (``function_call_output.output``) and a message
    (``content``). Anything else that was flagged -- a tool call's arguments, a
    reasoning item -- cannot be rewritten without breaking the conversation the
    SDK replays, so the caller fails closed instead.
    """
    if not isinstance(item, Mapping):
        return None
    if item.get("type") == "function_call_output" and "output" in item:
        return {**item, "output": placeholder}
    if "role" in item and "content" in item and item.get("type") in (None, "message"):
        return {**item, "content": placeholder}
    return None


def governance_model_input_filter(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    tool_policy: ToolPolicy | None = None,
    observer: AgentObserver | None = None,
    on_violation: str = RAISE,
    inner: Any = None,
) -> Callable[[Any], Awaitable[Any]]:
    """A ``RunConfig.call_model_input_filter`` that screens -- or redacts -- model input.

    The filter runs on every model call of the run, for every agent, with the
    exact input about to be sent, and may return a modified copy: the one
    central seam that can *withhold* a poisoned item rather than abort::

        Runner.run(agent, text, run_config=RunConfig(
            call_model_input_filter=governance_model_input_filter(firewall=fw,
                                                                  on_violation="redact")))

    Args:
        on_violation: ``"raise"`` (default) raises
            :class:`~adapt_agent.exceptions.SecurityBlockedError`, aborting the
            run. ``"redact"`` replaces each flagged tool result or message with a
            content-free placeholder naming the threats, and the run goes on; a
            flagged item of any other kind (a tool call, a reasoning item) and a
            policy violation cannot be redacted and still raise. The system
            prompt is the developer's and is not screened.
        inner: An existing ``call_model_input_filter`` to run first; its output
            is what gets screened.
        gate, firewall, defense, policy_enforcer, block_on_violation, agent_id,
        observer: As elsewhere. ``tool_policy`` is accepted for a uniform
            signature; this seam sees no tool call, so it never applies it.

    Policy state: ``state['messages']`` is the input items, plus every key of
    the run context.

    Raises:
        ValueError: On an unknown ``on_violation``.
    """
    if on_violation not in _FILTER_VIOLATION_MODES:
        raise ValueError(
            f"on_violation must be one of {_FILTER_VIOLATION_MODES!r}, not {on_violation!r}"
        )
    run_config = optional_import(
        "agents.run_config", "openai-agents", "governance_model_input_filter"
    )
    filter_gate = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=block_on_violation,
        agent_id=agent_id,
        tool_policy=tool_policy,
    )

    async def adapt_model_input_filter(data: Any) -> Any:
        model_data = data.model_data
        if inner is not None:
            model_data = await _maybe_await(inner(data))
        items = list(model_data.input)
        runtime = data.context if isinstance(data.context, Mapping) else {}
        with traced(observer, filter_gate.agent_id, _OP_MODEL_INPUT):
            threats = filter_gate.scan_input(items)
            state = as_state(items, **runtime)
            violations = [f"policy:{v}" for v in filter_gate.policy_violations(state)]
            if not (threats or violations) or not filter_gate.block_on_violation:
                return model_data
            if on_violation == RAISE or violations:
                raise SecurityBlockedError(
                    f"Model input blocked by security controls [{filter_gate.agent_id}]",
                    threats + violations,
                )
            redacted = []
            for item in items:
                found = filter_gate.scan_input(item)
                if not found:
                    redacted.append(item)
                    continue
                replacement = _redacted(
                    item, f"[withheld by ADAPT-Agent: {', '.join(dict.fromkeys(found))}]"
                )
                if replacement is None:
                    raise SecurityBlockedError(
                        f"Model input blocked by security controls [{filter_gate.agent_id}]",
                        found,
                    )
                redacted.append(replacement)
            return run_config.ModelInputData(input=redacted, instructions=model_data.instructions)

    return _mark(adapt_model_input_filter)


# -- streaming --------------------------------------------------------------------


def _raw_type(event: Any) -> str | None:
    if getattr(event, "type", None) != "raw_response_event":
        return None
    kind = getattr(getattr(event, "data", None), "type", None)
    return kind if isinstance(kind, str) else None


def _is_text_event(event: Any, kind: str) -> bool:
    if kind in _TEXT_EVENT_TYPES:
        return True
    if kind in ("response.output_item.added", "response.output_item.done"):
        item = getattr(event.data, "item", None)
        return getattr(item, "type", None) in _TEXT_ITEM_TYPES
    return False


def _event_text(event: Any) -> list[str]:
    data = event.data
    found = []
    for attr in ("delta", "text", "refusal"):
        value = getattr(data, attr, None)
        if isinstance(value, str) and value:
            found.append(value)
    for attr in ("item", "part"):
        value = getattr(data, attr, None)
        if value is not None:
            found.append(value)
    return found


class _HeldResponse:
    """The text events of one model response, held until it has been screened.

    Screening needs the response's text *assembled*: a provider's
    ``...done`` event carries the full text, but a response that fails or is
    cut short never sends one, and then all there is to judge are the deltas.
    Scanned one by one, ``"TOP"`` and ``"SECRET"`` both pass a rule against
    ``TOPSECRET``. So the deltas are joined here, per output item and content
    part, whatever else the stream goes on to say.
    """

    def __init__(self) -> None:
        self.events: list[Any] = []
        self._texts: list[Any] = []
        self._deltas: dict[tuple[Any, ...], list[str]] = {}

    def __bool__(self) -> bool:
        return bool(self.events)

    def add(self, event: Any, kind: str) -> None:
        self.events.append(event)
        self._texts.extend(_event_text(event))
        delta = getattr(event.data, "delta", None)
        if isinstance(delta, str) and delta:
            data = event.data
            part = (
                kind,
                getattr(data, "item_id", None),
                getattr(data, "output_index", None),
                getattr(data, "content_index", None),
            )
            self._deltas.setdefault(part, []).append(delta)

    def screenable(self) -> list[Any]:
        """Everything to screen: each part's deltas joined, then the raw texts."""
        return ["".join(pieces) for pieces in self._deltas.values()] + self._texts


class GovernedStream:
    """A ``RunResultStreaming`` whose text events are released only once screened.

    Iterate it exactly like ``result.stream_events()``. Each model response's
    text-bearing raw events (``response.output_text.delta`` and friends, and the
    message / reasoning output items) are held until that response completes,
    then screened with ``gate.scan_output``; clean, they are released in order,
    followed by the completion event. Flagged, the run is cancelled, nothing of
    that response is delivered, and :class:`SecurityBlockedError` is raised to
    the consumer. Every other event -- tool-call progress, run items, agent
    updates -- passes through as it arrives.

    The latency cost is one model response: text arrives in a burst when the
    response is complete rather than token by token.

    Attributes:
        result: The underlying ``RunResultStreaming`` (``final_output``,
            ``interruptions``, ``to_state()`` once the stream has finished).
        threats: Everything flagged so far -- populated under a report-only gate
            too, where flagged text is still released.
    """

    def __init__(
        self,
        result: Any,
        gate: GovernanceGate,
        *,
        observer: Any = None,
        on_finish: Callable[[], Any] | None = None,
        on_error: Callable[[BaseException], Any] | None = None,
    ) -> None:
        self.result = result
        self.gate = gate
        self.observer = observer
        self.threats: list[str] = []
        self._on_finish = on_finish
        self._on_error = on_error

    def __aiter__(self) -> AsyncIterator[Any]:
        return self.stream_events()

    def _screen(self, texts: list[Any]) -> None:
        with traced(self.observer, self.gate.agent_id, _OP_STREAM):
            found = self.gate.scan_output(texts)
            if not found:
                return
            self.threats.extend(found)
            if self.gate.block_on_violation:
                cancel = getattr(self.result, "cancel", None)
                if callable(cancel):
                    cancel()
                raise SecurityBlockedError(
                    f"Streamed output blocked by security controls [{self.gate.agent_id}]",
                    found,
                )

    async def stream_events(self) -> AsyncIterator[Any]:
        """The governed event stream."""
        held = _HeldResponse()
        try:
            async for event in self.result.stream_events():
                kind = _raw_type(event)
                if kind is not None and _is_text_event(event, kind):
                    held.add(event, kind)
                    continue
                if kind in _RESPONSE_END_TYPES:
                    if held:
                        self._screen(held.screenable())
                        for released in held.events:
                            yield released
                        held = _HeldResponse()
                    yield event
                    continue
                if held and kind is None:
                    # A run item for a response whose end event never came: the
                    # response is over, so screen what it said before moving on.
                    self._screen(held.screenable())
                    for released in held.events:
                        yield released
                    held = _HeldResponse()
                yield event
            if held:
                self._screen(held.screenable())
                for released in held.events:
                    yield released
        except BaseException as exc:
            if isinstance(exc, (GeneratorExit, asyncio.CancelledError)):
                # The consumer stopped reading: stop the run too, or it keeps
                # calling the model (and tools) for an answer nobody will see.
                cancel = getattr(self.result, "cancel", None)
                if callable(cancel):
                    cancel()
            if self._on_error is not None:
                self._on_error(exc)
            raise
        if self._on_finish is not None:
            self._on_finish()


def govern_stream(
    result: Any,
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    block_on_violation: bool = True,
    agent_id: str = "agent",
    tool_policy: ToolPolicy | None = None,
    observer: AgentObserver | None = None,
) -> GovernedStream:
    """Gate a ``Runner.run_streamed`` result so blocked text is never delivered::

        stream = govern_stream(Runner.run_streamed(agent, text), firewall=fw)
        async for event in stream:
            ...                      # SecurityBlockedError if a response is flagged

    See :class:`GovernedStream`. Only the output firewall applies here --
    ``defense``, ``policy_enforcer`` and ``tool_policy`` are accepted for a
    uniform signature and govern nothing on this seam; pair it with
    :func:`governance_run_hooks` and :func:`govern_tools` for the rest.
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
    return GovernedStream(result, resolved, observer=observer)


def find_security_block(exc: BaseException | None) -> SecurityBlockedError | None:
    """The :class:`SecurityBlockedError` behind an exception the SDK raised, if any.

    The SDK wraps an exception raised from a *function tool's* hooks in
    ``agents.UserError("Error running tool ...")``: the run still aborts, but
    the error the caller catches is the SDK's, with ours as its ``__cause__``.
    This walks the cause / context chain so the threats stay reachable::

        try:
            await Runner.run(agent, text, hooks=governance_run_hooks(...))
        except Exception as exc:
            if (blocked := find_security_block(exc)) is not None:
                print(blocked.threats)
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, SecurityBlockedError):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None


# -- run-boundary guardrails ------------------------------------------------------


def governance_guardrails(
    *,
    gate: GovernanceGate | None = None,
    firewall: Firewall | None = None,
    defense: AdversarialDefense | None = None,
    policy_enforcer: PolicyEnforcer | None = None,
    agent_id: str = "agent",
    screen_output: bool = True,
    block_on_violation: bool | None = None,
    tool_policy: ToolPolicy | None = None,
    observer: AgentObserver | None = None,
) -> dict[str, Any]:
    """Build OpenAI Agents input/output guardrails applying ADAPT-Agent governance.

    A guardrail function returns ``GuardrailFunctionOutput(output_info,
    tripwire_triggered)``; a tripped input guardrail makes the SDK raise
    ``InputGuardrailTripwireTriggered`` (``OutputGuardrailTripwireTriggered`` on
    the way out) and abort the run. That is the framework's own refusal path.
    ``output_info`` carries the threat list, so a caught tripwire can report
    *what* matched: ``exc.guardrail_result.output.output_info["threats"]``.

    Args:
        gate: A pre-built gate to share across agents; otherwise built from the
            controls below.
        firewall: Screens the agent input and final output.
        defense: Adversarial analysis of the input.
        policy_enforcer: Evaluated against the input *and the run's runtime
            context* (``Runner.run(..., context=...)``): ``state['messages']``
            is the run input, and every context key is merged in.
        agent_id: Recorded in ``output_info`` so a tripwire names the agent.
        screen_output: Also register an output guardrail.
        block_on_violation: ``False`` is report-only -- the tripwire never
            trips and the threats are still in ``output_info``. Unset, every
            threat trips it, whatever a supplied gate's own setting.
        tool_policy: Accepted for a uniform signature; guardrails see no tool
            call (use :func:`govern_tools`).
        observer: One span per guardrail run.

    Note:
        Input guardrails run for the *starting* agent of a run only. An agent
        reachable by handoff needs :func:`governance_run_hooks`, or its input
        governance silently never runs.

    Returns:
        A dict of ``Agent(...)`` kwargs: ``input_guardrails`` and (when
        ``screen_output``) ``output_guardrails``.

    Raises:
        ImportError: If ``openai-agents`` is not installed -- unlike the other
            integrations this one must build real ``InputGuardrail`` objects.
    """
    agents = optional_import("agents", "openai-agents", "governance_guardrails")

    # Scanning must not raise here: the SDK's tripwire is the refusal mechanism,
    # so the gate reports threats and the guardrail converts them.
    resolved = build_gate(
        gate=gate,
        firewall=firewall,
        defense=defense,
        policy_enforcer=policy_enforcer,
        block_on_violation=False,
        agent_id=agent_id,
        tool_policy=tool_policy,
    )
    trips = block_on_violation is not False

    async def adapt_input_guardrail(context: Any, agent: Any, agent_input: Any) -> Any:
        with traced(observer, resolved.agent_id, _OP_MODEL_INPUT):
            threats = resolved.scan_input(agent_input)
            # The runtime context must be merged in: a rule gating on
            # `state['trust_score']` reads what the caller passed to
            # `Runner.run(..., context=...)`, not the agent input.
            state = as_state(agent_input, **context_state(context))
            threats.extend(f"policy:{n}" for n in resolved.policy_violations(state))
        return agents.GuardrailFunctionOutput(
            output_info={"agent_id": resolved.agent_id, "threats": threats},
            tripwire_triggered=bool(threats) and trips,
        )

    guardrails: dict[str, Any] = {
        "input_guardrails": [agents.InputGuardrail(adapt_input_guardrail, name=_GUARDRAIL_NAME)]
    }

    if screen_output:

        async def adapt_output_guardrail(context: Any, agent: Any, agent_output: Any) -> Any:
            with traced(observer, resolved.agent_id, _OP_MODEL_OUTPUT):
                threats = resolved.scan_output(agent_output)
            return agents.GuardrailFunctionOutput(
                output_info={"agent_id": resolved.agent_id, "threats": threats},
                tripwire_triggered=bool(threats) and trips,
            )

        guardrails["output_guardrails"] = [
            agents.OutputGuardrail(adapt_output_guardrail, name=_GUARDRAIL_NAME)
        ]

    return guardrails


__all__ = [
    "GovernedStream",
    "find_security_block",
    "govern_stream",
    "govern_tools",
    "governance_agent_hooks",
    "governance_guardrails",
    "governance_model_input_filter",
    "governance_run_hooks",
    "governance_tool_guardrails",
]
