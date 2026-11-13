"""A reference model of the governance pipeline, for the tests to refine against.

This is the pipeline as one would describe it on a whiteboard -- about forty
lines, no framework handling, no regexes, no tracing -- written from the
documented behaviour and sharing **no code** with ``adapt_agent``. The property
tests in ``test_governance_properties.py`` then check that the real
implementation, driven with arbitrary inputs, never disagrees with it.

That is the point of keeping it separate and small: a test that calls the
implementation to compute its own expected value proves nothing, and a model
too large to read at a glance is just a second implementation with its own
bugs. If the two disagree, one of them is wrong, and this one is short enough
to decide which by reading.

Scope (deliberately narrow, so that every claim here is exact):

* payloads and results are JSON-like: ``dict`` / ``list`` / ``str`` / numbers;
* firewall rules are literal substrings plus an optional length cap;
* policy is one rule, "block when ``trust_score`` is below a floor";
* no adversarial defense (it is a heuristic, not a specification) and no
  middleware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SERVED = "served"
BLOCKED_INPUT = "blocked_input"
BLOCKED_POLICY = "blocked_policy"
BLOCKED_OUTPUT = "blocked_output"

#: What ``get_state()`` reports before anything has been served.
EMPTY_STATE: dict[str, Any] = {"messages": [], "context": {}}


@dataclass(frozen=True)
class Controls:
    blocked: tuple[str, ...] = ()  # literal substrings the firewall blocks
    max_length: int | None = None  # per-string length cap
    trust_floor: float | None = None  # policy: block when trust_score < floor
    block: bool = True  # block_on_violation; False = report-only


def texts(value: Any) -> list[str]:
    """Every string in a JSON-like value: string values and dict keys alike."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        found = [k for k in value if isinstance(k, str) and k]
        for item in value.values():
            found.extend(texts(item))
        return found
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in texts(item)]
    return []


def firewall_blocks(controls: Controls, value: Any) -> bool:
    """Whether any string in ``value`` is too long or contains a blocked literal."""
    for text in texts(value):
        if controls.max_length is not None and len(text) > controls.max_length:
            return True
        if any(literal in text for literal in controls.blocked):
            return True
    return False


def policy_blocks(controls: Controls, payload: dict[str, Any]) -> bool:
    """Whether the one policy rule fires. An absent score is *not* a violation."""
    score = payload.get("trust_score")
    if controls.trust_floor is None or isinstance(score, bool):
        return False
    return isinstance(score, (int, float)) and score < controls.trust_floor


def verdict(controls: Controls, payload: dict[str, Any], result: Any) -> str:
    """What happens to one request. ``result`` is what the agent would return."""
    if not controls.block:
        return SERVED  # report-only: everything is recorded, nothing is refused
    if firewall_blocks(controls, payload):
        return BLOCKED_INPUT
    if policy_blocks(controls, payload):
        return BLOCKED_POLICY
    if firewall_blocks(controls, result):
        return BLOCKED_OUTPUT
    return SERVED


def agent_runs(outcome: str) -> bool:
    """The agent is called unless the request was refused before reaching it."""
    return outcome in (SERVED, BLOCKED_OUTPUT)


@dataclass
class Session:
    """The model's whole state across a sequence of requests to one agent."""

    controls: Controls
    state: dict[str, Any] = field(default_factory=lambda: dict(EMPTY_STATE))
    agent_calls: int = 0
    completed_spans: int = 0
    error_spans: int = 0

    def request(self, payload: dict[str, Any], result: dict[str, Any]) -> str:
        outcome = verdict(self.controls, payload, result)
        if agent_runs(outcome):
            self.agent_calls += 1
        if outcome == SERVED:
            # Only a served request becomes the agent's state.
            self.state = {
                "messages": result.get("messages", []),
                "context": {k: v for k, v in result.items() if k != "messages"},
            }
            self.completed_spans += 1
        elif outcome == BLOCKED_OUTPUT:
            self.error_spans += 1
        return outcome
