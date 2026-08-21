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
from collections.abc import Mapping
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
            for v in value.values():
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
    """

    def __init__(
        self,
        *,
        firewall: Firewall | None = None,
        defense: AdversarialDefense | None = None,
        policy_enforcer: PolicyEnforcer | None = None,
        block_on_violation: bool = True,
        agent_id: str = "agent",
    ):
        self.firewall = firewall
        self.defense = defense
        self.policy_enforcer = policy_enforcer
        self.block_on_violation = block_on_violation
        self.agent_id = agent_id

    # -- scanning (never raises) ----------------------------------------------

    def scan_input(self, payload: Any) -> list[str]:
        """Run firewall + adversarial defense over a payload, returning threats."""
        threats: list[str] = []
        for text in extract_texts(payload):
            if self.firewall is not None and not self.firewall.check_input(text):
                threats.append("firewall")
            if self.defense is not None:
                analysis = self.defense.analyze_input(text)
                if not analysis["is_safe"]:
                    threats.extend(analysis["threats_detected"])
        return threats

    def scan_output(self, payload: Any) -> list[str]:
        """Run the firewall over an output payload, returning threats."""
        threats: list[str] = []
        if self.firewall is None:
            return threats
        for text in extract_texts(payload):
            if not self.firewall.check_output(text):
                threats.append("firewall")
        return threats

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

    def _reason(self, message: str) -> str:
        return f"{message} [{self.agent_id}]" if self.agent_id else message


__all__ = ["GovernanceGate", "extract_prompt", "extract_texts"]
