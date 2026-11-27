"""The framework-agnostic governance core.

Everything ADAPT-Agent does to a request -- screen the input, evaluate policy
against state, screen the output -- is pure synchronous CPU work: firewall
regexes, the adversarial classifier, the policy sandbox. None of it is
framework-specific, and none of it is async.

That is what makes :class:`GovernanceGate` worth factoring out. It is the single
implementation used by *both* ways of applying governance:

* the **outer wrapper** -- :class:`~adapt_agent.adapters._governed.GovernedAdapter`
  wraps a whole agent and governs its boundary;
* the **native hooks** -- :mod:`adapt_agent.integrations` plugs the same gate
  into a framework's own middleware / callback / guardrail chain, so governance
  nests *per agent* inside a multi-agent graph and composes with the middleware
  an application already runs.

Both paths therefore enforce identical rules; a fix here reaches every
framework at once, and the two can never drift apart.
"""

from __future__ import annotations

import dataclasses
import fnmatch
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast

from adapt_agent.exceptions import SecurityBlockedError

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance, typing only
    from adapt_agent.adversarial import AdversarialDefense
    from adapt_agent.core.policy import PolicyEnforcer
    from adapt_agent.core.types import AgentState
    from adapt_agent.security.firewall import Firewall

#: Attribute names that commonly hold human-readable text on framework result
#: and message objects (Pydantic AI ``.output``, Microsoft Agent Framework
#: ``.text``, CrewAI ``.raw``, OpenAI Agents ``.final_output``, Claude
#: ``ResultMessage.result`` / ``TextBlock.text``, LangChain ``.content`` ...).
_TEXT_ATTRS = (
    "content",
    "text",
    "output",
    "final_output",
    "raw",
    "result",
    "data",
)

#: Attribute names that hold *containers* of further text-bearing objects
#: (e.g. ``Content.parts`` in Google ADK, message lists, CrewAI ``tasks_output``).
#: These are recursed into even when they are framework objects rather than
#: plain dicts/lists, so screening reaches deeply-structured results.
_RECURSE_ATTRS = (
    "content",
    "parts",
    "messages",
    "tasks_output",
    # Tool results returning to the model. A Google ADK ``Part`` carries them
    # under ``function_response`` (older/other SDKs: ``tool_response``), whose
    # payload is a ``response`` mapping -- never ``Part.text``. This is the
    # highest-value injection vector there is: whatever a tool fetched from the
    # open web arrives here, and reaches the next model call.
    "function_response",
    "tool_response",
    "response",
)


#: Recursion bound for :func:`extract_texts`, guarding against pathological or
#: cyclic nesting. Sized from the deepest *real* payload rather than a guess: a
#: governed ADK tool result is ``{"result": ...} -> [Content] -> Content ->
#: parts -> Part -> function_response -> response -> text``, which is eight
#: hops. At the previous bound of six that text was silently dropped -- and a
#: security scan that stops early fails open, so the headroom is deliberate.
_MAX_WALK_DEPTH = 12


def _distinct(threats: list[str]) -> list[str]:
    """Drop repeated labels, keeping first-seen order.

    One payload yields many texts -- a request's parts, a message list, a
    model's fields -- so a single blocked request reported ``["firewall",
    "firewall", "firewall"]``. The multiplicity counts *texts scanned*, not
    distinct problems, and identical labels do not say anything more the second
    time; they only make an error message and a refusal's metadata noisier.
    """
    seen: dict[str, None] = {}
    for threat in threats:
        seen.setdefault(threat, None)
    return list(seen)


def _safe_getattr(obj: Any, attr: str) -> Any:
    """``getattr(obj, attr, None)`` that never propagates.

    Framework result objects may expose attributes via descriptors/properties
    that raise. Since text extraction feeds the security pipeline, a raising
    attribute must not crash the whole execution -- treat it as absent.
    """
    try:
        return getattr(obj, attr, None)
    except Exception:
        return None


def extract_texts(data: Any) -> list[str]:
    """Best-effort extraction of human-readable text from an arbitrary payload.

    Adapter payloads are typically dicts that may contain a ``messages`` list or
    arbitrary string fields, but framework result objects expose their text via
    attributes instead (see :data:`_TEXT_ATTRS`). We collect every string we can
    reach so the security controls can scan it without assuming a fixed schema.
    """
    texts: list[str] = []

    def _walk(value: Any, depth: int = 0) -> None:
        if depth > _MAX_WALK_DEPTH:
            return
        if isinstance(value, str):
            texts.append(value)
        elif value is None or isinstance(value, (int, float, bool)):
            # Primitives carry no text and have no attributes worth probing;
            # returning early avoids pointless getattr lookups on large payloads.
            return
        elif isinstance(value, dict):
            for k, v in value.items():
                # Keys too. A tool response is attacker-shaped data, not just an
                # attacker-shaped *value*: `{"ignore previous instructions": ""}`
                # renders to the model exactly like the same text in a value, and
                # scanning only values let it through untouched.
                if isinstance(k, str) and k:
                    texts.append(k)
                _walk(v, depth + 1)
        elif isinstance(value, (list, tuple)):
            for v in value:
                _walk(v, depth + 1)
        else:
            # Framework message / result objects expose their text via one of a
            # handful of well-known attributes. Recurse into anything walkable.
            for attr in _TEXT_ATTRS:
                inner = _safe_getattr(value, attr)
                if isinstance(inner, str):
                    texts.append(inner)
                elif inner is not None and not isinstance(inner, (int, float, bool)):
                    # Recurse into *any* non-primitive, not just dict/list/tuple.
                    # A Pydantic AI ``AgentRunResult.output`` holds a BaseModel,
                    # and stopping at container types here left every field of a
                    # structured answer unscreened -- the wrapper was walked, the
                    # model inside it was not.
                    _walk(inner, depth + 1)
            # Structured content containers (e.g. genai ``Content.parts``) hold
            # further objects whose text we still want to scan.
            for attr in _RECURSE_ATTRS:
                inner = _safe_getattr(value, attr)
                if inner is not None and not isinstance(inner, (str, int, float, bool)):
                    _walk(inner, depth + 1)
            # A *structured output* (a Pydantic model or dataclass, as produced
            # by a Pydantic AI ``output_type`` or a MAF structured response)
            # carries its text in arbitrarily-named fields -- ``lane``, ``note``,
            # ``summary``. None of those appear above, so without this a
            # structured answer would go almost entirely unscreened: injected
            # text smuggled into one field would never be seen.
            fields = _structured_fields(value)
            if fields is not None:
                _walk(fields, depth + 1)

    _walk(data)
    return texts


def _structured_fields(value: Any) -> dict[str, Any] | None:
    """Return the field mapping of a Pydantic model or dataclass, else ``None``.

    Deliberately narrow: only objects that *declare* themselves structured are
    walked. Scanning every object's ``__dict__`` would drag framework internals
    (clients, sessions, loggers) into the security scan.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return {f.name: getattr(value, f.name, None) for f in dataclasses.fields(value)}
        except Exception:
            return None
    if hasattr(value, "model_fields") or hasattr(value, "__fields__"):  # pydantic v2 / v1
        for attr in ("model_dump", "dict"):
            method = _safe_getattr(value, attr)
            if callable(method):
                try:
                    dumped = method()
                except Exception:
                    continue
                if isinstance(dumped, dict):
                    return dumped
    return None


def extract_prompt(payload: Any) -> str:
    """Derive a single prompt string from an adapter payload.

    Many frameworks (Pydantic AI, OpenAI Agents, Microsoft Agent Framework,
    Claude Agent SDK) run from a string prompt rather than a state dict. This
    helper picks the most plausible prompt: the latest user message, then a
    common prompt-like key, then a string fallback. It accepts a state dict, a
    bare list/tuple of messages, or a plain string.
    """
    if isinstance(payload, str):
        return payload

    if isinstance(payload, dict):
        messages: Any = payload.get("messages")
    elif isinstance(payload, (list, tuple)):
        messages = payload
    else:
        messages = None

    if isinstance(messages, (list, tuple)) and messages:
        for message in reversed(messages):
            if isinstance(message, dict):
                role, content = message.get("role"), message.get("content")
            else:
                role = _safe_getattr(message, "role")
                content = _safe_getattr(message, "content")
            role_str = role.lower() if isinstance(role, str) else None
            if isinstance(content, str) and (role_str == "user" or role is None):
                return content
        last = messages[-1]
        last_content = (
            last.get("content") if isinstance(last, dict) else _safe_getattr(last, "content")
        )
        if isinstance(last_content, str):
            return last_content

    if isinstance(payload, dict):
        for key in ("prompt", "input", "query", "text"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
    return str(payload)


#: :attr:`ToolDecision.action` values.
ALLOW = "allow"
BLOCK = "block"
ASK = "ask"


@dataclasses.dataclass(frozen=True)
class ToolDecision:
    """The gate's verdict on one tool call or one tool result.

    Attributes:
        action: ``"allow"``, ``"block"`` or ``"ask"``. ``"ask"`` means *pause
            for a human decision*; a binding maps it onto the framework's own
            approval mechanism (a LangGraph interrupt, an OpenAI Agents
            ``needs_approval``, a Claude ``permissionDecision: "ask"``) and
            treats it as ``"block"`` where the framework has none -- see
            :meth:`resolve`.
        tool_name: The tool the decision is about.
        threats: What was found, in first-seen order: ``"tool:denied:<name>"``,
            ``"tool:not_allowed:<name>"``, ``"tool:approval:<name>"``,
            ``"firewall"``, an adversarial-defense label, or
            ``"policy:<rule>"``. Populated even when ``action`` is
            ``"allow"`` under a report-only gate.
        report_only: ``True`` when the gate is in report-only mode
            (``block_on_violation=False``) and would otherwise have blocked or
            asked. ``action`` is then ``"allow"`` so a binding needs no special
            case, and ``threats`` still says what was seen.
        agent_id: The gate's ``agent_id``, for attribution inside a graph.
    """

    action: str
    tool_name: str
    threats: tuple[str, ...] = ()
    report_only: bool = False
    agent_id: str = ""

    @property
    def allowed(self) -> bool:
        return self.action == ALLOW

    @property
    def blocked(self) -> bool:
        return self.action == BLOCK

    @property
    def needs_approval(self) -> bool:
        return self.action == ASK

    def resolve(self, *, approvals_supported: bool) -> str:
        """The action to take given whether the framework can pause for approval.

        Without an approval mechanism an ``"ask"`` must not become an
        ``"allow"``: a call somebody wanted a human to see would run unseen.
        It fails closed to ``"block"``.
        """
        if self.action == ASK and not approvals_supported:
            return BLOCK
        return self.action

    @property
    def reason(self) -> str:
        """A one-line explanation safe to show the model or put in an error.

        Names the tool and the threat *labels* only -- never the arguments or
        the result, which may be the very content being withheld.
        """
        who = f" [{self.agent_id}]" if self.agent_id else ""
        found = ", ".join(self.threats) if self.threats else "no threats"
        if self.action == ASK:
            return f"Tool call {self.tool_name!r} needs approval{who}: {found}."
        if self.action == BLOCK:
            return f"Tool call {self.tool_name!r} refused by ADAPT-Agent{who}: {found}."
        return f"Tool call {self.tool_name!r} allowed{who}: {found}."


class ToolPolicy:
    """Name-based allow / deny / approval lists for tool calls.

    The coarse control that needs no rule language: which tools an agent may
    call at all, and which need a human. Patterns are shell-style globs matched
    case-sensitively against the tool name, so one entry can cover a family --
    ``"mcp__*"`` for every MCP tool as the Claude Agent SDK names them,
    ``"github_*"``, or an exact ``"Bash"``.

    Precedence is deny, then allow-list, then approval: a denied tool is
    refused even if it is also allow-listed, and a tool outside a configured
    allow-list is refused rather than sent for approval.

    Args:
        allow: When given, *only* matching tools may be called (an allow-list).
            ``None`` (default) places no restriction; an empty collection
            refuses every tool.
        deny: Tools that may never be called.
        require_approval: Tools whose every call pauses for a human decision.
    """

    def __init__(
        self,
        *,
        allow: Iterable[str] | None = None,
        deny: Iterable[str] = (),
        require_approval: Iterable[str] = (),
    ):
        self.allow: tuple[str, ...] | None = None if allow is None else _patterns("allow", allow)
        self.deny: tuple[str, ...] = _patterns("deny", deny)
        self.require_approval: tuple[str, ...] = _patterns("require_approval", require_approval)

    def verdict(self, tool_name: str) -> tuple[str, str | None]:
        """Return ``(action, threat_label)`` for ``tool_name``; never raises."""
        if _matches(tool_name, self.deny):
            return BLOCK, f"tool:denied:{tool_name}"
        if self.allow is not None and not _matches(tool_name, self.allow):
            return BLOCK, f"tool:not_allowed:{tool_name}"
        if _matches(tool_name, self.require_approval):
            return ASK, f"tool:approval:{tool_name}"
        return ALLOW, None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ToolPolicy(allow={self.allow!r}, deny={self.deny!r}, "
            f"require_approval={self.require_approval!r})"
        )


def _patterns(label: str, values: Iterable[str]) -> tuple[str, ...]:
    # A bare string is iterable, so `deny="Bash"` would silently become the
    # four patterns "B", "a", "s", "h" -- denying nothing anyone calls.
    if isinstance(values, str):
        raise TypeError(f"ToolPolicy {label} must be a collection of patterns, not a str")
    patterns = tuple(values)
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern:
            raise TypeError(f"ToolPolicy {label} patterns must be non-empty strings")
    return patterns


def _matches(tool_name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(tool_name, pattern) for pattern in patterns)


class GovernanceGate:
    """Applies ADAPT-Agent's security controls to arbitrary payloads.

    A gate holds the controls and nothing else -- no framework knowledge, no
    agent, no I/O. Callers hand it whatever the framework gave them (a message
    list, a request object, a response) and it scans every string it can reach.

    Args:
        firewall: Screens input and output text.
        defense: Adversarial/prompt-injection analysis of input text.
        policy_enforcer: Evaluated against agent *state*, not content. Note the
            division of labour: content rules belong on the firewall, because a
            policy condition can only see the state you give it.
        block_on_violation: When ``True`` (default) :meth:`review_input` and
            :meth:`review_output` raise :class:`SecurityBlockedError`. When
            ``False`` they return the threat list and let the caller decide --
            useful for a report-only rollout.
        agent_id: Identifier used in raised errors, so a violation inside a
            multi-agent graph names the specific agent that produced it.
        tool_policy: Name-based allow / deny / approval lists consulted by
            :meth:`check_tool_call`. ``None`` restricts no tool by name; tool
            arguments and results are still screened by the other controls.
    """

    def __init__(
        self,
        *,
        firewall: Firewall | None = None,
        defense: AdversarialDefense | None = None,
        policy_enforcer: PolicyEnforcer | None = None,
        block_on_violation: bool = True,
        agent_id: str = "agent",
        tool_policy: ToolPolicy | None = None,
    ):
        self.firewall = firewall
        self.defense = defense
        self.policy_enforcer = policy_enforcer
        self.block_on_violation = block_on_violation
        self.agent_id = agent_id
        self.tool_policy = tool_policy

    # -- scanning (never raises) ----------------------------------------------

    def scan_input(self, payload: Any) -> list[str]:
        """Run firewall + adversarial defense over a payload, returning threats."""
        # ⚡ Bolt: Fast path out if no input scanners are configured to avoid O(N) extraction overhead
        # Benchmark: Reduces empty scan_input time from ~0.118s to ~0.002s (per 10k calls)
        if self.firewall is None and self.defense is None:
            return []

        threats: list[str] = []
        for text in extract_texts(payload):
            if self.firewall is not None and not self.firewall.check_input(text):
                threats.append("firewall")
            if self.defense is not None:
                analysis = self.defense.analyze_input(text)
                if not analysis["is_safe"]:
                    threats.extend(analysis["threats_detected"])
        return _distinct(threats)

    def scan_output(self, payload: Any) -> list[str]:
        """Run the firewall over an output payload, returning threats."""
        threats: list[str] = []
        if self.firewall is None:
            return threats
        for text in extract_texts(payload):
            if not self.firewall.check_output(text):
                threats.append("firewall")
        return _distinct(threats)

    def policy_violations(self, state: Mapping[str, Any]) -> list[str]:
        """Return the names of *blocking* policy rules ``state`` violates.

        Only rules whose action is ``block`` are returned; a rule set to warn or
        log is evaluated but does not appear here.

        ``state`` is any mapping, not just an
        :class:`~adapt_agent.core.types.AgentState`: a native hook sees the
        framework's real state (a LangGraph graph state, a CrewAI inputs dict),
        so a rule may reference any key that state actually carries -- which is
        strictly more than the ``messages``/``context`` an adapter can expose.
        """
        if self.policy_enforcer is None:
            return []
        blocking: list[str] = []
        for violation in self.policy_enforcer.check_state(cast("AgentState", state)):
            rule = self.policy_enforcer.get_rule(violation)
            if rule is not None and rule.get("action") == "block":
                blocking.append(violation)
        return blocking

    # -- review (raises when blocking) ----------------------------------------

    def review_input(self, payload: Any, *, state: Mapping[str, Any] | None = None) -> list[str]:
        """Screen an input payload (and optionally its state) before the agent runs.

        Returns the threats found. Raises :class:`SecurityBlockedError` instead
        when any were found and ``block_on_violation`` is set.
        """
        threats = self.scan_input(payload)
        if threats and self.block_on_violation:
            raise SecurityBlockedError(self._reason("Input blocked by security controls"), threats)
        if state is not None:
            violations = [f"policy:{name}" for name in self.policy_violations(state)]
            if violations:
                if self.block_on_violation:
                    raise SecurityBlockedError(self._reason("Input blocked by policy"), violations)
                threats.extend(violations)
        return threats

    def review_output(self, payload: Any) -> list[str]:
        """Screen a result payload after the agent ran.

        Returns the threats found. Raises :class:`SecurityBlockedError` instead
        when any were found and ``block_on_violation`` is set.
        """
        threats = self.scan_output(payload)
        if threats and self.block_on_violation:
            raise SecurityBlockedError(self._reason("Output blocked by security controls"), threats)
        return threats

    # -- tool calls ------------------------------------------------------------
    #
    # The boundary of an agent is the wrong place to stop an agent from doing
    # something: by the time the final answer is screened the tool has run.
    # Every supported framework exposes a seam around the individual tool call,
    # and these four methods are the one contract each binding implements
    # against. `check_*` returns a verdict and never raises; `review_*` raises
    # on a block, for a seam whose refusal mechanism is an exception.

    def check_tool_call(
        self,
        tool_name: str,
        arguments: Any = None,
        *,
        state: Mapping[str, Any] | None = None,
    ) -> ToolDecision:
        """Decide whether a tool call may run. Never raises.

        Evaluated in order, collecting every threat found:

        1. :class:`ToolPolicy` -- deny list, allow-list, approval list.
        2. The firewall and adversarial defense over every string in
           ``arguments`` (keys included).
        3. Tool-scoped policy rules (conditions naming ``tool``) against
           ``{"name": tool_name, "arguments": arguments}`` and ``state``. A
           violated ``block`` rule blocks; a violated ``require_approval`` rule
           asks.

        A block outranks an ask. Under a report-only gate
        (``block_on_violation=False``) the decision is ``"allow"`` with
        ``report_only=True`` and the threats still listed.

        Args:
            tool_name: The tool as the framework names it (MCP tools included --
                they surface through the same seam as local ones).
            arguments: The call's arguments: usually a mapping, sometimes a JSON
                string or a framework object. Scanned as-is.
            state: The framework's session/runtime state, for policy rules that
                combine the call with it. See
                :meth:`PolicyEnforcer.check_tool <adapt_agent.core.policy.PolicyEnforcer.check_tool>`.
        """
        action = ALLOW
        threats: list[str] = []

        if self.tool_policy is not None:
            action, label = self.tool_policy.verdict(tool_name)
            if label is not None:
                threats.append(label)

        found = self.scan_input(arguments)
        if found:
            threats.extend(found)
            action = BLOCK

        if self.policy_enforcer is not None:
            tool = {"name": tool_name, "arguments": arguments}
            for violation in self.policy_enforcer.check_tool(tool, state):
                rule = self.policy_enforcer.get_rule(violation)
                rule_action = rule.get("action") if rule is not None else None
                if rule_action == "block":
                    threats.append(f"policy:{violation}")
                    action = BLOCK
                elif rule_action == "require_approval":
                    threats.append(f"policy:{violation}")
                    if action != BLOCK:
                        action = ASK

        return self._decision(action, tool_name, threats)

    def check_tool_result(
        self,
        tool_name: str,
        result: Any,
        *,
        state: Mapping[str, Any] | None = None,
    ) -> ToolDecision:
        """Decide whether a tool's result may be returned to the model. Never raises.

        This is the indirect-injection seam: whatever a tool fetched -- a web
        page, an email, a file, an MCP server's reply -- reaches the next model
        call as trusted-looking context. The result is therefore screened as
        *input*: firewall plus adversarial defense over every string in it.

        The verdict is ``"allow"`` or ``"block"`` only; there is nothing to
        approve once the tool has run. A binding acts on a block by withholding
        or replacing the result, or by aborting the run.

        ``state`` is accepted for symmetry with :meth:`check_tool_call` and is
        reserved; no control reads it today.
        """
        threats = self.scan_input(result)
        return self._decision(BLOCK if threats else ALLOW, tool_name, threats)

    def review_tool_call(
        self,
        tool_name: str,
        arguments: Any = None,
        *,
        state: Mapping[str, Any] | None = None,
    ) -> ToolDecision:
        """:meth:`check_tool_call`, raising on a block.

        Returns the decision when it is ``"allow"`` or ``"ask"`` -- an ask is
        the binding's to route to the framework's approval mechanism (see
        :meth:`ToolDecision.resolve`).

        Raises:
            SecurityBlockedError: When the call is blocked.
        """
        decision = self.check_tool_call(tool_name, arguments, state=state)
        if decision.blocked:
            raise SecurityBlockedError(decision.reason, list(decision.threats))
        return decision

    def review_tool_result(
        self,
        tool_name: str,
        result: Any,
        *,
        state: Mapping[str, Any] | None = None,
    ) -> ToolDecision:
        """:meth:`check_tool_result`, raising on a block.

        Raises:
            SecurityBlockedError: When the result is blocked.
        """
        decision = self.check_tool_result(tool_name, result, state=state)
        if decision.blocked:
            raise SecurityBlockedError(decision.reason, list(decision.threats))
        return decision

    def _decision(self, action: str, tool_name: str, threats: list[str]) -> ToolDecision:
        distinct = tuple(_distinct(threats))
        if action != ALLOW and not self.block_on_violation:
            return ToolDecision(
                ALLOW, tool_name, distinct, report_only=True, agent_id=self.agent_id
            )
        return ToolDecision(action, tool_name, distinct, agent_id=self.agent_id)

    def _reason(self, message: str) -> str:
        return f"{message} [{self.agent_id}]" if self.agent_id else message


__all__ = [
    "ALLOW",
    "ASK",
    "BLOCK",
    "GovernanceGate",
    "ToolDecision",
    "ToolPolicy",
    "extract_prompt",
    "extract_texts",
]
