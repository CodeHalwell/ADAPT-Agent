#!/usr/bin/env python3
"""Mutation check for the security-critical modules.

Coverage says a line *ran* under test; it does not say a test would notice if
the line were wrong. This script answers the second question for the code that
matters most. It re-introduces one security bug at a time (a "mutant") into a
scratch copy of the repository and runs the test suite against it:

* the suite **fails**  -> the mutant is *killed*: some test pins that guarantee;
* the suite **passes** -> the mutant *survived*: the guarantee is untested, and
  a change that removed it -- by a person or by an automated agent -- would
  merge green.

Two kinds of mutant live in :data:`MUTANTS`:

* **Historical** ones re-open a vulnerability that was actually found and fixed
  in this repository. Each cites its entry in ``.jules/sentinel.md``. They are
  the regression contract: a fixed bug stays fixed only while a test fails when
  it comes back.
* **Structural** ones break a guarantee that has not (yet) been broken in
  anger -- the governance gate skipping a scan, a fail-closed path failing open.

The list is hand-written on purpose. A generic mutation tool flips operators
everywhere and reports hundreds of survivors of no security consequence; this
is a short, reviewable statement of *which* bugs the suite promises to catch.

A mutant may name the test files that judge it (:attr:`Mutant.tests`); the
rest run the whole suite. That keeps the check inside CI's time budget without
changing what a kill means -- a selection can only miss a kill, which shows up
as a survivor, never invent one.

When a guard is refactored its mutant's ``old`` text stops matching. That is
reported as **stale** and fails the run, so the list cannot rot silently into
checking nothing. ``tests/test_mutation_check.py`` checks the same thing on
every ordinary test run.

Usage::

    python scripts/mutation_check.py              # run everything
    python scripts/mutation_check.py --list       # show the mutants
    python scripts/mutation_check.py --only fw-   # ids starting with a prefix
    python scripts/mutation_check.py -j 4         # parallel workers

Exit status is non-zero if any mutant survives or is stale, or if the
unmutated suite does not pass (in which case every "kill" would be meaningless).
"""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The fast tests *about this script*; never part of a mutant's verdict.
_SELF_TEST = "tests/test_mutation_check.py"

_FIREWALL = "adapt_agent/security/firewall.py"
_TAINT = "adapt_agent/security/taint_tracker.py"
_GATE = "adapt_agent/core/governance.py"
_POLICY = "adapt_agent/core/policy.py"
_TRUST = "adapt_agent/core/trust.py"
_MEMORY = "adapt_agent/core/memory.py"
_MIDDLEWARE = "adapt_agent/core/middleware.py"
_DEFENSE = "adapt_agent/adversarial/__init__.py"
_TEXT = "adapt_agent/security/_text.py"
_OBSERVER = "adapt_agent/observability/__init__.py"
_OPTIMIZERS = "adapt_agent/optimization/optimizers.py"
_GOVERNED = "adapt_agent/adapters/_governed.py"

#: Stands in for a gate call that has been "forgotten": accepts anything,
#: finds no threats, raises nothing.
_NO_OP = "(lambda *a, **k: [])("

#: Stands in for a forgotten ``check_tool_call`` / ``check_tool_result``: it
#: returns a real "allow" decision, so the binding carries on as if the gate
#: had approved -- exactly what a hook that stopped asking would do.
_ALLOW_TOOL = (
    "(lambda name, *a, **k: __import__('adapt_agent.core.governance', fromlist=['_'])"
    ".ToolDecision('allow', name))("
)


def _tests(*names: str) -> tuple[str, ...]:
    return tuple(f"tests/test_{name}.py" for name in names)


# Test selections (:attr:`Mutant.tests`). Each covers every file that exercises
# the code its mutants touch, not only the one test known to kill them: a
# selection is a budget, not a hint about which assertion matters.
_TESTS_TOOLS = _tests(
    "tool_governance", "policy", "policy_hardening", "governance_properties", "mutant_kills"
)
_TESTS_GATE = _tests(
    "tool_governance",
    "policy",
    "policy_hardening",
    "governance_properties",
    "extract_texts_hardening",
    "governed_adapters",
    "async_governance",
    "rejection_leaves_no_trace",
    "adapters",
    "structured_evals",
    "core",
    "security",
    "integrations",
)
_TESTS_POLICY = _tests(
    "policy",
    "policy_hardening",
    "taint_tracker",
    "trust",
    "security",
    "governance_properties",
    "core",
)
# pytest runs files in the order given and stops at the first failure. The two
# files holding wall-clock "is linear in its length" tests go last: under a
# loaded runner those fail spuriously, and when one is the *first* failure the
# confirmation step spends its attempts deselecting timing flakes (one per
# parametrization) instead of reaching the deterministic test that kills.
_TESTS_DETECTOR = _tests(
    "adversarial_extra",
    "firewall_extra",
    "security",
    "core",
    "governance_properties",
    "mutant_kills",
    "detector_hardening",
    "adversarial",
)
_TESTS_OBSERVER = _tests(
    "observability",
    "observability_extra",
    "observability_hardening",
    "observability_otel",
    "governance_properties",
)
_TESTS_MEMORY = _tests("memory", "core", "governance_properties")
_TESTS_MIDDLEWARE = _tests("middleware", "core", "adapters", "governed_adapters")
_TESTS_OPTIMIZER_GATE = _tests(
    "optimize_optimizers", "optimizer_significance", "cli_optimize", "optimization"
)
_TESTS_OPTIMIZER = _tests(
    "eval_reporting",
    "evaluation",
    "optimization",
    "optimize_evaluation",
    "optimize_retry",
    "optimize_judge",
    "optimize_introspect_claude_agent",
    "optimize_runners",
)


def _binding_tests(framework: str, *extra: str) -> tuple[str, ...]:
    """A binding's SDK-free tests plus the shared adapter suites.

    ``test_real_<framework>`` is left out on purpose. CI's mutation job
    installs no framework, so there it would only skip; anywhere it does run,
    it could credit a kill CI will never see. A binding guarantee observable
    only through the real SDK therefore has no mutant here.
    """
    return _tests(
        f"integrations_{framework}",
        "integrations",
        "governed_adapters",
        "adapters",
        "mutant_kills",
        *extra,
    )


_TESTS_CLAUDE = _binding_tests("claude_agent", "claude_hooks_stubbed")
_TESTS_LANGGRAPH = _binding_tests("langgraph")
_TESTS_CREWAI = _binding_tests("crewai")
_TESTS_OPENAI = _binding_tests("openai_agents", "openai_hooks_stubbed")
_TESTS_ADK = _binding_tests("google_adk")
_TESTS_MAF = _binding_tests("agent_framework")
_BINDING_TESTS = {
    "agent_framework": _TESTS_MAF,
    "claude_agent": _TESTS_CLAUDE,
    "crewai": _TESTS_CREWAI,
    "google_adk": _TESTS_ADK,
    "langgraph": _TESTS_LANGGRAPH,
    "openai_agents": _TESTS_OPENAI,
    "pydantic_ai": _binding_tests("pydantic_ai"),
}


def _hook_mutants() -> tuple[Mutant, ...]:
    """One mutant per place a native integration calls the governance gate.

    Each framework hook is a few lines whose entire job is to make that call.
    A hook that silently stops making it still imports, still registers, and
    still returns the framework's expected shape -- it just governs nothing.
    """
    sites = (
        ("agent-framework", "agent_framework", "resolved.review_input(", "input"),
        ("agent-framework", "agent_framework", "resolved.review_output(", "output"),
        ("claude-agent", "claude_agent", "resolved.review_input(", "input"),
        ("crewai", "crewai", "resolved.review_input(", "input"),
        ("crewai", "crewai", "resolved.review_output(", "output"),
        ("crewai-task", "crewai", "gate.scan_output(", "output"),
        ("google-adk", "google_adk", "resolved.review_input(", "input"),
        ("google-adk", "google_adk", "resolved.review_output(", "output"),
        ("langgraph", "langgraph", "gate.review_input(", "input"),
        ("langgraph", "langgraph", "gate.review_output(", "output"),
        ("openai-guardrail", "openai_agents", "resolved.scan_input(", "input"),
        ("openai-guardrail", "openai_agents", "resolved.scan_output(", "output"),
        ("openai-hooks", "openai_agents", "resolved.review_input(", "input"),
        ("openai-hooks", "openai_agents", "resolved.review_output(", "output"),
        ("pydantic-ai", "pydantic_ai", "resolved.review_output(", "output"),
    )
    return tuple(
        Mutant(
            f"hook-{name}-{direction}",
            f"adapt_agent/integrations/{module}.py",
            call,
            _NO_OP,
            f"the {name} hook screens its {direction}",
            tests=_BINDING_TESTS[module],
        )
        for name, module, call, direction in sites
    )


@dataclass(frozen=True)
class Mutant:
    """One re-introduced bug.

    Attributes:
        id: Short stable identifier (used by ``--only``).
        path: File to mutate, relative to the repository root.
        old: Text that must occur **exactly once** in the file.
        new: What replaces it.
        why: The guarantee this mutant breaks, in a few words.
        history: The ``.jules/sentinel.md`` heading this re-opens, if any.
        also: Further journal headings that record the same fix.
        extra: Further ``(old, new)`` edits in the same file, for a bug that
            cannot be re-introduced by changing one place. Each ``old`` must
            also occur exactly once.
        tests: Test files to run against this mutant instead of the whole
            suite. Empty (the default) runs everything. A narrower selection
            never makes a kill weaker -- a listed test still has to fail, and
            reproducibly -- it can only turn a kill elsewhere into a reported
            survivor, so a list that is too short shows up as a failing run,
            never as a false pass. It exists to keep the full check inside CI's
            time budget.
    """

    id: str
    path: str
    old: str
    new: str
    why: str
    history: str = ""
    also: tuple[str, ...] = ()
    extra: tuple[tuple[str, str], ...] = ()
    tests: tuple[str, ...] = ()

    @property
    def edits(self) -> tuple[tuple[str, str], ...]:
        """Every ``(old, new)`` replacement this mutant makes."""
        return ((self.old, self.new), *self.extra)

    def apply(self, source: str) -> str:
        for old, new in self.edits:
            source = source.replace(old, new)
        return source

    @property
    def reopens(self) -> tuple[str, ...]:
        """Every journal heading this mutant re-opens."""
        return ((self.history,) if self.history else ()) + self.also


MUTANTS: tuple[Mutant, ...] = (
    # ------------------------------------------------------------------ firewall
    Mutant(
        "fw-filter-fail-open",
        _FIREWALL,
        """                logger.error("Error in custom filter: %s", e)
                return False, "custom_filter_error", None""",
        """                logger.error("Error in custom filter: %s", e)
                continue""",
        "a custom filter that raises must block",
        "2026-04-21 - Fail-Closed Firewall Design",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-whitelist-waives-cap",
        _FIREWALL,
        "if self.max_content_length is not None and len(content) > self.max_content_length:",
        "if (not (self.whitelist_mode and any(p.fullmatch(content) for p in self._allowed_patterns))"
        " and self.max_content_length is not None and len(content) > self.max_content_length):",
        "no allowed pattern can waive the length cap",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-length-cap",
        _FIREWALL,
        "len(content) > self.max_content_length:",
        "len(content) > 2 * self.max_content_length:",
        "oversize content is blocked before any regex runs",
        "2026-04-26 - Firewall Resource Exhaustion / Unbounded Growth",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-events-unbounded",
        _FIREWALL,
        "self._security_events: deque[SecurityEvent] = deque(maxlen=max_events)",
        "self._security_events: deque[SecurityEvent] = deque()",
        "the security-event log is bounded",
        "2026-04-26 - Firewall Resource Exhaustion / Unbounded Growth",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-allow-search",
        _FIREWALL,
        """        for pattern in self._allowed:
            try:
                if pattern.fullmatch(content):""",
        """        for pattern in self._allowed:
            try:
                if pattern.search(content):""",
        "an allowed pattern must match the whole content",
        "2024-05-18 - [Firewall Whitelist Bypass via regex search]",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-allow-before-block",
        _FIREWALL,
        "        if self.whitelist_mode:\n            allowed = self._allowed_match(content)",
        "        if True:\n            allowed = self._allowed_match(content)",
        "by default an allowed pattern never overrides a block",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-error-text-leak",
        _FIREWALL,
        '"error": "An error occurred",',
        '"error": str(detail),',
        "exception text is not copied into event metadata",
        "2024-05-24 - [Information Leakage in Event Metadata]",
        extra=(
            (
                'return False, "custom_filter_error", None',
                'return False, "custom_filter_error", str(e)',
            ),
        ),
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-snippet-raw",
        _FIREWALL,
        "self._truncate_escaped(self.sanitize(visible), _SNIPPET_CHARS)",
        "self._truncate_escaped(visible, _SNIPPET_CHARS)",
        "a blocked secret is redacted in the event that records it",
        "2024-05-14 - Exposed sensitive data in firewall events",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-truncate-then-sanitize",
        _FIREWALL,
        "self._truncate_escaped(self.sanitize(visible), _SNIPPET_CHARS)",
        "self.sanitize(self._truncate_escaped(visible, _SNIPPET_CHARS))",
        "sanitise the whole string, then truncate",
        "2024-07-02 - Prevent Partial Secret Leakage via Early Truncation",
        also=("2024-05-25 - Firewall Partial Secret Leakage",),
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-snippet-newlines",
        _TEXT,
        "    if _LOG_UNSAFE_RE.search(bounded) is None:\n        return bounded",
        "    if True:\n        return bounded",
        "newlines are escaped in stored snippets",
        "2024-07-01 - Prevent log poisoning in firewall security events",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-description-newlines",
        _FIREWALL,
        '"description": self._truncate_escaped(description, _MAX_DESCRIPTION_LENGTH),',
        '"description": description,',
        "newlines are escaped in an event's description",
        "2024-11-09 - Log Poisoning in Firewall Security Events",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-description-unbounded",
        _FIREWALL,
        '"description": self._truncate_escaped(description, _MAX_DESCRIPTION_LENGTH),',
        '"description": description[:10000].replace("\\n", "\\\\n").replace("\\r", "\\\\r"),',
        "an escaped description stays within its length bound",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-truncate-overflow",
        _TEXT,
        "        if used + len(piece) > max_length:\n            break",
        "        if False:\n            break",
        "an escaped snippet never exceeds its length bound",
        "2024-10-27 - Fix flawed truncation logic for escapes",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-redos-guard",
        _FIREWALL,
        "        risk = _backtracking_risk(pattern, flags)",
        "        risk = None",
        "obviously catastrophic patterns are rejected",
        "2024-05-23 - ReDoS in Heuristic Regex Guard",
        tests=_TESTS_DETECTOR,
    ),
    # ------------------------------------------------------------- taint tracker
    Mutant(
        "taint-sources-unbounded",
        _TAINT,
        "if len(self._taint_sources) > self.max_tracked_items:",
        "if False:",
        "tracked taint sources are bounded",
        "2026-05-09 - Dictionary Bounding for DoS Prevention",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "taint-data-unbounded",
        _TAINT,
        "if len(self._tainted_data) > self.max_tracked_items:",
        "if False:",
        "tracked tainted items are bounded",
        "2026-05-09 - Dictionary Bounding for DoS Prevention",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "taint-propagation-unbounded",
        _TAINT,
        "deque(maxlen=max_propagations)",
        "deque()",
        "the propagation log is bounded",
        "2026-05-02 - List Bounding for DoS Prevention",
        tests=_TESTS_POLICY,
    ),
    # ------------------------------------------------------------ governance gate
    Mutant(
        "gate-output-unscanned",
        _GATE,
        """        for text in texts:
            if not self.firewall.check_output(text):""",
        """        for text in []:
            if not self.firewall.check_output(text):""",
        "agent output is screened",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gate-defense-ignored",
        _GATE,
        'if not analysis["is_safe"]:',
        "if False:",
        "an adversarial-defense finding blocks the input",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gate-policy-skipped",
        _GATE,
        "        if state is not None:\n            violations = [f",
        "        if False:\n            violations = [f",
        "blocking policy rules are enforced on input",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gate-warn-rules-block",
        _GATE,
        'if rule is not None and rule.get("action") == "block":',
        "if rule is not None:",
        "only action=block rules block",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gate-report-only-ignored",
        _GATE,
        "        if threats and self.block_on_violation:\n            raise SecurityBlockedError("
        'self._reason("Input blocked by security controls"), threats)',
        "        if threats and False:\n            raise SecurityBlockedError("
        'self._reason("Input blocked by security controls"), threats)',
        "a threat in the input raises when blocking is on",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gate-dict-keys",
        _GATE,
        "if isinstance(k, str) and k:",
        "if False:",
        "dict keys are scanned, not only values",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gate-walk-depth",
        _GATE,
        "_MAX_WALK_DEPTH = 20",
        "_MAX_WALK_DEPTH = 6",
        "the scan reaches a real nested tool result",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gate-structured-fields",
        _GATE,
        "            if fields is not None:\n                _each(fields, child)",
        "            if False:\n                _each(fields, child)",
        "fields of a structured output are scanned",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gate-tool-responses",
        _GATE,
        '    "function_response",\n    "tool_response",\n    "response",\n)',
        ")",
        "tool results returning to the model are scanned",
        tests=_TESTS_GATE,
    ),
    # --------------------------------------------------------------------- policy
    Mutant(
        "policy-fail-closed-ignored",
        _POLICY,
        """            return self.fail_closed
        return outcome == "violation"

    def get_rule_stats""",
        """            return False
        return outcome == "violation"

    def get_rule_stats""",
        "fail_closed=True turns an unevaluable rule into a violation",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "policy-default-fail-open",
        _POLICY,
        "def __init__(self, max_violations: int = 1000, fail_closed: bool = True):",
        "def __init__(self, max_violations: int = 1000, fail_closed: bool = False):",
        "an unevaluable rule is a violation unless fail-open is asked for",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "policy-depth-guard",
        _POLICY,
        "if depth > 50:",
        "if depth > 10**9:",
        "condition evaluation depth is bounded",
        "2024-05-25 - AST Evaluation Depth Limit",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "policy-length-guard",
        _POLICY,
        "if len(condition) > 1024:",
        "if len(condition) > 10**9:",
        "condition length is bounded before parsing",
        "2024-05-24 - DoS Vulnerability in AST parsing",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "policy-node-guard",
        _POLICY,
        "if node_count > MAX_CONDITION_NODES:",
        "if node_count > 10**9:",
        "condition node count is bounded",
        "2024-05-24 - DoS Vulnerability in AST parsing",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "policy-allows-calls",
        _POLICY,
        "    ast.Subscript,\n)",
        "    ast.Subscript,\n    ast.Call,\n)",
        "function calls are rejected in conditions",
        "2026-05-05 - AST Whitelisting vs Arbitrary eval()",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "policy-allows-attributes",
        _POLICY,
        "    ast.Subscript,\n)",
        "    ast.Subscript,\n    ast.Attribute,\n)",
        "attribute access is rejected in conditions",
        "2026-05-05 - AST Whitelisting vs Arbitrary eval()",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "policy-allows-pow",
        _POLICY,
        "        ast.Div: operator.truediv,\n",
        "        ast.Div: operator.truediv,\n        ast.Pow: operator.pow,\n",
        "unbounded arithmetic (**) is rejected in conditions",
        "2026-05-05 - AST Whitelisting vs Arbitrary eval()",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "policy-unknown-names",
        _POLICY,
        "if isinstance(node, ast.Name) and node.id not in CONDITION_NAMES:",
        "if False:",
        "a rule naming an unknown variable is rejected at registration",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "policy-violations-unbounded",
        _POLICY,
        "deque(maxlen=max_violations)",
        "deque()",
        "the violation log is bounded",
        "2026-05-02 - List Bounding for DoS Prevention",
        tests=_TESTS_POLICY,
    ),
    # ---------------------------------------------------------------------- trust
    Mutant(
        "trust-self-reported",
        _TRUST,
        """        # if "trust_score" in state:
        #     factors["self_reported"] = state["trust_score"]""",
        """        if "trust_score" in state:
            factors["self_reported"] = state["trust_score"]""",
        "an agent cannot raise its own trust score",
        "2026-05-18 - Privilege Escalation via Unvalidated Trust Input",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "trust-agents-unbounded",
        _TRUST,
        "if len(self._trust_scores) > self.max_agents:",
        "if False:",
        "tracked agents are bounded",
        "2026-05-09 - Dictionary Bounding for DoS Prevention",
        tests=_TESTS_POLICY,
    ),
    # --------------------------------------------------------------------- memory
    Mutant(
        "memory-long-term-unbounded",
        _MEMORY,
        "if len(self._long_term_memory) > self.long_term_capacity:",
        "if False:",
        "long-term memory holds at most its capacity",
        "2025-05-10 - [Memory Exhaustion DoS in MemorySystem]",
        tests=_TESTS_MEMORY,
    ),
    # ----------------------------------------------------------------- middleware
    Mutant(
        "middleware-priority",
        _MIDDLEWARE,
        """        self._pre_middleware.sort(
            key=lambda m: func_priorities.get(m, 0),
            reverse=True,""",
        """        self._pre_middleware.sort(
            key=lambda m: func_priorities.get(m, 0),
            reverse=False,""",
        "higher-priority (security) middleware runs first",
        "2026-05-12 - Middleware Priority Inversion",
        tests=_TESTS_MIDDLEWARE,
    ),
    # --------------------------------------------------------- adversarial defense
    Mutant(
        "defense-custom-patterns",
        _DEFENSE,
        '        if custom is not None:\n            findings.append(_Finding("custom_pattern"',
        '        if False:\n            findings.append(_Finding("custom_pattern"',
        "registered custom patterns are actually matched",
        "2026-05-13 - Broken Custom Attack Pattern Enforcement",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "defense-length-cap",
        _DEFENSE,
        "if self.max_content_length is not None and len(input_text) > self.max_content_length:",
        "if False:",
        "oversize input is refused before analysis",
        "2026-05-23 - AdversarialDefense Resource Exhaustion",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "defense-jailbreak-dropped",
        _DEFENSE,
        '_THREAT_ORDER = ("prompt_injection", "jailbreak", "role_spoofing", "custom_pattern")',
        '_THREAT_ORDER = ("prompt_injection", "role_spoofing", "custom_pattern")',
        "a detected jailbreak marks the input unsafe",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "defense-injection-dropped",
        _DEFENSE,
        '_THREAT_ORDER = ("prompt_injection", "jailbreak", "role_spoofing", "custom_pattern")',
        '_THREAT_ORDER = ("jailbreak", "role_spoofing", "custom_pattern")',
        "a detected injection marks the input unsafe",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "defense-attacks-unbounded",
        _DEFENSE,
        "deque(maxlen=max_attacks)",
        "deque()",
        "the detected-attack log is bounded",
        "2026-05-02 - List Bounding for DoS Prevention",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "defense-content-newlines",
        _DEFENSE,
        '"content": escape_for_log(content, _SNIPPET_LENGTH),',
        '"content": content[:_SNIPPET_LENGTH],',
        "newlines are escaped in a recorded attack's content",
        "2024-06-06 - Log Poisoning Vulnerability in Adversarial Defense",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "defense-indicator-newlines",
        _DEFENSE,
        '"indicator": escape_for_log(indicator, _SNIPPET_LENGTH),',
        '"indicator": indicator[:_SNIPPET_LENGTH],',
        "newlines are escaped in a recorded attack's indicator",
        "2024-11-07 - Log Poisoning in Adversarial Attack Indicators",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "defense-result-newlines",
        _DEFENSE,
        """            "input": escape_for_log(input_text, _SNIPPET_LENGTH),
            "threats_detected": threats,""",
        """            "input": input_text[:_SNIPPET_LENGTH],
            "threats_detected": threats,""",
        "newlines are escaped in the returned analysis snippet",
        "2024-10-27 - Fix log poisoning in AdversarialDefense analysis results",
        tests=_TESTS_DETECTOR,
    ),
    # ------------------------------------------------------------------- observer
    Mutant(
        "observer-logs-unbounded",
        _OBSERVER,
        "deque(maxlen=max_logs)",
        "deque()",
        "the in-memory log is bounded",
        "2026-05-02 - List Bounding for DoS Prevention",
        tests=_TESTS_OBSERVER,
    ),
    Mutant(
        "observer-trace-events-unbounded",
        _OBSERVER,
        'if len(self._traces[trace_id]["events"]) > self.max_events_per_trace:',
        "if False:",
        "events per trace are bounded",
        "2026-05-07 - AgentObserver Nested List and Dictionary Bounding for DoS Prevention",
        tests=_TESTS_OBSERVER,
    ),
    Mutant(
        "observer-metric-names-unbounded",
        _OBSERVER,
        "if len(self._metrics) > self.max_metric_names:",
        "if False:",
        "distinct metric names are bounded",
        "2026-05-07 - AgentObserver Nested List and Dictionary Bounding for DoS Prevention",
        tests=_TESTS_OBSERVER,
    ),
    Mutant(
        "observer-message-length",
        _OBSERVER,
        '"message": self._clean(message, self.max_text_length),',
        '"message": self._clean(message, 10**12),',
        "a logged message is length-bounded",
        "2024-05-29 - [Memory Exhaustion DoS in Logging]",
        tests=_TESTS_OBSERVER,
    ),
    Mutant(
        "observer-message-newlines",
        _OBSERVER,
        '"message": self._clean(message, self.max_text_length),',
        '"message": message[: self.max_text_length],',
        "newlines are escaped in a logged message",
        "2024-06-25 - [Log Poisoning Vulnerability in AgentObserver]",
        tests=_TESTS_OBSERVER,
    ),
    Mutant(
        "observer-description-newlines",
        _OBSERVER,
        '"description": self._clean(description, self.max_text_length),',
        '"description": description[: self.max_text_length],',
        "newlines are escaped in an event description",
        "2024-06-25 - [Log Poisoning Vulnerability in AgentObserver]",
        tests=_TESTS_OBSERVER,
    ),
    # ----------------------------------------------------------- governed adapter
    Mutant(
        "gov-input-not-refused",
        _GOVERNED,
        "        if in_threats and adapter.block_on_violation:",
        "        if False:",
        "the adapter refuses an input its controls flagged",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gov-policy-not-refused",
        _GOVERNED,
        "        if blocking and adapter.block_on_violation:",
        "        if False:",
        "the adapter refuses an input a blocking rule matched",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gov-output-not-refused",
        _GOVERNED,
        "            if out_threats and adapter.block_on_violation:",
        "            if False:",
        "the adapter refuses an output its firewall flagged",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gov-refused-request-becomes-state",
        _GOVERNED,
        "        candidate_state = adapter.extract_state(input_data)",
        "        candidate_state = self._last_state = adapter.extract_state(input_data)",
        "a refused request never becomes the agent's state",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gov-output-block-recorded-as-success",
        _GOVERNED,
        """        except BaseException as exc:  # CancelledError derives from BaseException
            self._trace_error(trace_id, exc)
            raise""",
        """        except BaseException:  # CancelledError derives from BaseException
            if adapter.observer is not None:
                adapter.observer.end_trace(trace_id, status="completed")
            raise""",
        "an output block closes its span as an error, not a success",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "policy-boolop-returns-bool",
        _POLICY,
        """                    if result:
                        return result
                return result""",
        """                    if result:
                        return True
                return False""",
        "`or` yields its operand, as in Python",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "policy-missed-lookup-untracked",
        _POLICY,
        "                context[_MISSED_LOOKUP] = True",
        "                pass",
        "a lookup of an absent key is counted as unresolved",
        tests=_TESTS_POLICY,
    ),
    # ------------------------------------------------------------------ optimizer
    Mutant(
        "opt-proposer-judge-ignored",
        _OPTIMIZERS,
        "return self.proposer_judge if self.proposer_judge is not None else self.judge",
        "return self.judge",
        "a separate proposer judge does the rewriting",
        tests=_TESTS_OPTIMIZER_GATE,
    ),
    Mutant(
        "opt-shared-judge-unflagged",
        _OPTIMIZERS,
        "if not any(same(rewriting, j) for j in scoring):",
        "if True:",
        "a judge that both writes and scores is reported",
        tests=_TESTS_OPTIMIZER_GATE,
    ),
    Mutant(
        "opt-gate-never-rejects",
        _OPTIMIZERS,
        "if not held_out_gain:",
        "if False:",
        "a config that regresses on held-out data is not applied",
        tests=_TESTS_OPTIMIZER_GATE,
    ),
    Mutant(
        "opt-gate-leaves-config-applied",
        _OPTIMIZERS,
        "            # Leave the baseline applied: the agent is exactly as it was given.\n",
        "            target.apply(best_config)\n",
        "a rejected config is removed from the live agent",
        tests=_TESTS_OPTIMIZER_GATE,
    ),
    Mutant(
        "opt-gate-trusts-partial-scores",
        "adapt_agent/optimization/significance.py",
        "        if result.transient or primary in result.transient_metrics:\n"
        "            continue\n",
        "        if result.transient or primary in result.transient_metrics:\n"
        "            pass\n",
        "a throttled held-out row is left out of both sides, not scored as a zero",
        tests=_TESTS_OPTIMIZER_GATE,
    ),
    # ------------------------------------------- governance gate: tool calls
    Mutant(
        "gate-tool-deny-ignored",
        _GATE,
        "        if _matches(tool_name, self.deny):\n            return BLOCK",
        "        if False:\n            return BLOCK",
        "a denied tool is refused",
        tests=_TESTS_TOOLS,
    ),
    Mutant(
        "gate-tool-allowlist-ignored",
        _GATE,
        "        if self.allow is not None and not _matches(tool_name, self.allow):",
        "        if False:",
        "a tool outside a configured allow-list is refused",
        tests=_TESTS_TOOLS,
    ),
    Mutant(
        "gate-tool-approval-list-ignored",
        _GATE,
        "        if _matches(tool_name, self.require_approval):\n" "            return ASK",
        "        if False:\n            return ASK",
        "a tool on the approval list pauses for a human",
        tests=_TESTS_TOOLS,
    ),
    Mutant(
        "gate-tool-arguments-unscanned",
        _GATE,
        "        found = self.scan_input(arguments)",
        "        found = []",
        "a tool call's arguments are screened",
        tests=_TESTS_TOOLS,
    ),
    Mutant(
        "gate-tool-policy-block-ignored",
        _GATE,
        '                if rule_action == "block":\n'
        '                    threats.append(f"policy:{violation}")\n'
        "                    action = BLOCK\n",
        '                if rule_action == "block":\n'
        '                    threats.append(f"policy:{violation}")\n',
        "a violated tool-scoped block rule blocks the call",
        tests=_TESTS_TOOLS,
    ),
    Mutant(
        "gate-tool-policy-approval-ignored",
        _GATE,
        "                    if action != BLOCK:\n" "                        action = ASK",
        "                    if False:\n                        action = ASK",
        "a violated require_approval rule asks",
        tests=_TESTS_TOOLS,
    ),
    Mutant(
        "gate-tool-ask-outranks-block",
        _GATE,
        "                    if action != BLOCK:\n" "                        action = ASK",
        "                    if True:\n                        action = ASK",
        "a block outranks an ask",
        tests=_TESTS_TOOLS,
    ),
    Mutant(
        "gate-tool-result-unscanned",
        _GATE,
        "        threats = self.scan_input(result)\n",
        "        threats = []\n",
        "a tool's result is screened before it reaches the model",
        tests=_TESTS_TOOLS,
    ),
    Mutant(
        "gate-tool-ask-fails-open",
        _GATE,
        "        if self.action == ASK and not approvals_supported:\n" "            return BLOCK",
        "        if self.action == ASK and not approvals_supported:\n" "            return ALLOW",
        "an ask with no approval mechanism fails closed",
        tests=_TESTS_TOOLS,
    ),
    Mutant(
        "gate-tool-report-only-blocks",
        _GATE,
        "        if action != ALLOW and not self.block_on_violation:",
        "        if False:",
        "a report-only gate allows and still lists the threats",
        tests=_TESTS_TOOLS,
    ),
    Mutant(
        "policy-tool-rule-in-check-message",
        _POLICY,
        '        context = {"message": message}\n'
        "\n"
        "        for rule_name, rule in tuple(self._rules.items()):\n"
        "            if _is_tool_rule(rule):\n"
        "                continue",
        '        context = {"message": message}\n'
        "\n"
        "        for rule_name, rule in tuple(self._rules.items()):\n"
        "            if False:\n"
        "                continue",
        "a tool-scoped rule is not evaluated by check_message",
        tests=_TESTS_TOOLS,
    ),
    Mutant(
        "policy-tool-rule-in-check-state",
        _POLICY,
        '        context = {"state": state}\n'
        "\n"
        "        for rule_name, rule in tuple(self._rules.items()):\n"
        "            if _is_tool_rule(rule):\n"
        "                continue",
        '        context = {"state": state}\n'
        "\n"
        "        for rule_name, rule in tuple(self._rules.items()):\n"
        "            if False:\n"
        "                continue",
        "a tool-scoped rule is not evaluated by check_state",
        tests=_TESTS_TOOLS,
    ),
    Mutant(
        "policy-check-tool-runs-every-rule",
        _POLICY,
        "            if not _is_tool_rule(rule):\n                continue",
        "            if False:\n                continue",
        "check_tool evaluates only tool-scoped rules",
        tests=_TESTS_TOOLS,
    ),
    # -------------------------------------- governance gate: text extraction
    Mutant(
        "gate-incomplete-not-reported",
        _GATE,
        "        threats = [UNSCANNABLE] if walk.incomplete else []",
        "        threats = []",
        "a payload the walk could not finish is flagged unscannable",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gate-depth-silent",
        _GATE,
        "        if depth > max_depth:\n"
        '            reasons.setdefault("depth", None)\n'
        "            return",
        "        if depth > max_depth:\n            return",
        "hitting the depth limit makes the walk incomplete",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gate-visited-set-removed",
        _GATE,
        "        if key in seen:\n            return\n",
        "",
        "a cyclic payload is walked once, not forever",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gate-node-budget-removed",
        _GATE,
        "        if len(seen) > max_nodes:\n            raise _BudgetExhausted\n",
        "",
        "the walk stops at its node budget",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gate-texts-not-deduplicated",
        _GATE,
        "        texts = list(dict.fromkeys(walk.texts))",
        "        texts = list(walk.texts)",
        "a repeated string is scanned once",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gate-payload-budget-ignored",
        _GATE,
        "            if sum(len(text) for text in texts) > self.max_payload_chars:",
        "            if False:",
        "a payload over max_payload_chars is refused",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gate-maf-contents",
        _GATE,
        '    "contents",\n',
        "",
        "a Microsoft Agent Framework message's contents are scanned",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gov-middleware-output-unscreened",
        _GOVERNED,
        "                    if rewritten_threats and adapter.block_on_violation:",
        "                    if False:",
        "input rewritten by pre-middleware is screened again",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gov-dual-stream-async-first",
        _GOVERNED,
        "        if isinstance(value, Iterable):\n"
        "            return list(value)\n"
        "        return _run_coro",
        "        return _run_coro",
        "a dual sync/async stream is drained synchronously",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gov-input-refusal-unrecorded",
        _GOVERNED,
        '            self._refused(in_threats, stage="input")\n',
        "",
        "an input refusal is recorded on the observer",
        tests=_TESTS_GATE,
    ),
    Mutant(
        "gov-policy-refusal-unrecorded",
        _GOVERNED,
        '            self._refused(policy_threats, stage="policy")\n',
        "",
        "a policy refusal is recorded on the observer",
        tests=_TESTS_GATE,
    ),
    # -------------------------------------------------- policy, taint, trust
    Mutant(
        "policy-action-unvalidated",
        _POLICY,
        "        if action not in VALID_ACTIONS:\n"
        '            raise ValueError(f"Invalid action {action!r}; expected one of {sorted(VALID_ACTIONS)}")\n'
        "        if severity",
        "        if False:\n"
        '            raise ValueError(f"Invalid action {action!r}; expected one of {sorted(VALID_ACTIONS)}")\n'
        "        if severity",
        "a rule with an unknown action is rejected",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "policy-require-approval-anywhere",
        _POLICY,
        "if action == REQUIRE_APPROVAL and TOOL_SCOPE not in",
        "if False and TOOL_SCOPE not in",
        "require_approval is accepted only on a tool-scoped rule",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "policy-handler-aborts-loop",
        _POLICY,
        "        except Exception:\n            logger.exception(",
        "        except KeyError:\n            logger.exception(",
        "a raising violation handler does not stop the other rules",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "policy-mult-sequences",
        _POLICY,
        "if bin_op_type is ast.Mult and not (",
        "if False and not (",
        "sequence repetition is rejected in conditions",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "policy-nonfinite-compare",
        _POLICY,
        "if (isinstance(left, float) and not math.isfinite(left)) or (",
        "if False and (",
        "a comparison with NaN or inf is unevaluable",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "policy-depth-not-checked-at-registration",
        _POLICY,
        "if _max_eval_depth(tree) > MAX_EVAL_DEPTH:",
        "if False:",
        "an over-deep condition is rejected at registration",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "policy-history-live-state",
        _POLICY,
        '"data": self._summarise(data),',
        '"data": data,',
        "a recorded violation keeps a summary, not the live state",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "taint-evicted-reads-clean",
        _TAINT,
        "self._remember_floor(oldest_data, self._derived_level(oldest_data))",
        "pass",
        "an evicted item keeps its taint level",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "taint-downgrade-allowed",
        _TAINT,
        "if floor is not None and self._LEVEL_PRIORITY[level] < self._LEVEL_PRIORITY[floor]:",
        "if False:",
        "a taint level cannot be lowered",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "taint-empty-mark-counts",
        _TAINT,
        "if not source_ids:\n            return",
        "if False:\n            return",
        "marking with no sources changes nothing",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "trust-nan-delta",
        _TRUST,
        "if not math.isfinite(delta):",
        "if False:",
        "a NaN or inf trust delta is rejected",
        tests=_TESTS_POLICY,
    ),
    Mutant(
        "trust-evicts-distrusted",
        _TRUST,
        "threshold = max(self.distrust_floor, self.initial_trust)",
        "threshold = 0.0",
        "eviction never forgets a distrusted agent",
        tests=_TESTS_POLICY,
    ),
    # ------------------------------------------------- detector and firewall
    Mutant(
        "defense-scanner-dead-alternatives",
        _DEFENSE,
        "live = frozenset(term for term, last in self._last.items() if last > start)",
        "live = frozenset(self._last)",
        "the phrase scanner drops alternatives that ended",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "defense-stack-keeps-descendants",
        _DEFENSE,
        "cut = bisect.bisect_left(removable_sequences, closed_sequence)",
        "cut = len(removable)",
        "closing a scope removes its descendants",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "defense-no-default-cap",
        _DEFENSE,
        "max_content_length: int | None = DEFAULT_MAX_CONTENT_LENGTH,",
        "max_content_length: int | None = None,",
        "the detector has a default input cap",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "defense-invisible-not-stripped",
        _TEXT,
        'return unicodedata.category(char) in ("Cf", "Mn", "Me")',
        "return False",
        "invisible characters are stripped before matching",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "defense-tag-block-not-decoded",
        _DEFENSE,
        "hidden = decode_tag_block(prompt)",
        'hidden = ""',
        "Unicode tag-block smuggling is decoded",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "defense-homoglyphs-not-folded",
        _TEXT,
        "    return word.translate(_HOMOGLYPHS)",
        "    return word",
        "homoglyphs are folded before matching",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-log-escape-newlines-only",
        _TEXT,
        '"[\\x00-\\x1f\\x7f-\\x9f\\u2028\\u2029"',
        '"[\\n\\r"',
        "every control character is escaped in logs",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "defense-no-decoded-payloads",
        _DEFENSE,
        "        views.extend(_decoded_payloads(prompt))",
        "        pass",
        "encoded payloads are decoded and analysed",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "defense-no-rewritten-views",
        _DEFENSE,
        "        views.extend(_rewritten_views(prompt, normalized))",
        "        pass",
        "obfuscated rewrites are analysed",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "defense-role-field-judgement",
        _DEFENSE,
        "                    if key_value or _INLINE_KEY_VALUE_RE.search(tail) is not None:\n"
        "                        continue",
        "                    if False:\n                        continue",
        "a benign role field is not flagged as spoofing",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "defense-weak-role-unjudged",
        _DEFENSE,
        "            elif dialogue or _STRONG_CUE_RE.search(tail) is not None:",
        "            else:",
        "a weak role marker needs a cue to count",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "defense-prefilter-incomplete",
        _DEFENSE,
        '    ("ignore", "disregard", "forget"),\n    ("stop", "quit", "cease"),',
        '    ("ignore", "disregard"),\n    ("stop", "quit", "cease"),',
        "the prefilter admits every override verb",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-normalize-ignored",
        _FIREWALL,
        "            if normalized != content:\n" "                forms.append(normalized)",
        "            if False:\n                forms.append(normalized)",
        "normalize=True also matches the normalised form",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-normalize-replaces-raw",
        _FIREWALL,
        "        forms = [content]",
        "        forms = []",
        "the raw content is always matched",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-normalize-allowlist",
        _FIREWALL,
        "                if pattern.fullmatch(content):\n"
        "                    return pattern.compiled.pattern",
        "                if pattern.fullmatch(normalize_for_matching(content)):\n"
        "                    return pattern.compiled.pattern",
        "an allowed pattern matches the raw content only",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-timeout-fails-open",
        _FIREWALL,
        '                return False, "pattern_timeout", pattern.compiled.pattern',
        "                continue",
        "a pattern that times out blocks",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-blocked-snippet-window",
        _FIREWALL,
        '                    "content_snippet": self._redact_snippet(content)\n'
        "                },",
        '                    "content_snippet": self._truncate_escaped(self.sanitize(content), 100)\n'
        "                },",
        "a blocked event's snippet is windowed around the match",
        tests=_TESTS_DETECTOR,
    ),
    Mutant(
        "fw-redos-alternation",
        _FIREWALL,
        "            _check_branches(body, ignore_case)\n" "            if inner.first",
        "            if inner.first",
        "overlapping alternation under a repeat is rejected",
        tests=_TESTS_DETECTOR,
    ),
    # -------------------------------------------------------------- observer
    Mutant(
        "observer-evicts-live-trace",
        _OBSERVER,
        "                if self._completed:",
        "                if False:",
        "a full trace store evicts completed traces first",
        tests=_TESTS_OBSERVER,
    ),
    Mutant(
        "observer-error-result-uncapped",
        _OBSERVER,
        "        if failed and isinstance(result, str):",
        "        if False:",
        "an error result is capped at max_error_length",
        tests=_TESTS_OBSERVER,
    ),
    Mutant(
        "observer-nonfinite-metric",
        _OBSERVER,
        "        if not math.isfinite(value):\n" '            raise ValueError(f"metric value',
        '        if False:\n            raise ValueError(f"metric value',
        "a NaN or inf metric is refused",
        tests=_TESTS_OBSERVER,
    ),
    Mutant(
        "observer-redactor-fails-open",
        _OBSERVER,
        '                return "[redaction failed]"',
        "                pass",
        "text is withheld when the redactor raises",
        tests=_TESTS_OBSERVER,
    ),
    # ---------------------------------------------------- framework bindings
    # Every guarantee below is pinned by a test that runs without the framework.
    # Some binding seams exist only inside a class built on the real SDK and
    # are therefore pinned by ``tests/test_real_<framework>.py`` alone, which
    # skips in CI's mutation job; they have no mutant here, because it would
    # survive there: the LangGraph callback handler's tool-call check, the
    # Pydantic AI capability's approval check and tool block/withhold, and the
    # CrewAI hooks' tool-call check and refusal.
    Mutant(
        "hook-claude-agent-tool-call",
        "adapt_agent/integrations/claude_agent.py",
        'decision = resolved.check_tool_call(tool_name, hook_input.get("tool_input"), state=policy)',
        "decision = " + _ALLOW_TOOL + 'tool_name, hook_input.get("tool_input"), state=policy)',
        "the Claude PreToolUse hook screens every tool call",
        tests=_TESTS_CLAUDE,
    ),
    Mutant(
        "hook-claude-agent-tool-result",
        "adapt_agent/integrations/claude_agent.py",
        "decision = resolved.check_tool_result(tool_name, response, state=policy)",
        "decision = " + _ALLOW_TOOL + "tool_name, response, state=policy)",
        "the Claude PostToolUse hook screens every tool result",
        tests=_TESTS_CLAUDE,
    ),
    Mutant(
        "hook-claude-agent-error-fails-open",
        "adapt_agent/integrations/claude_agent.py",
        "                return _failure_output(resolved, event, exc, hook_input)",
        "                return {}",
        "a Claude hook that raises answers with a refusal",
        tests=_TESTS_CLAUDE,
    ),
    Mutant(
        "hook-claude-agent-matcher-continues",
        "adapt_agent/integrations/claude_agent.py",
        'on_failure="continue" if event in AUDIT_EVENTS else "block",',
        'on_failure="continue",',
        "a governance matcher tells the CLI to block when the hook fails",
        tests=_TESTS_CLAUDE,
    ),
    Mutant(
        "hook-claude-agent-can-use-tool",
        "adapt_agent/integrations/claude_agent.py",
        "            decision = resolved.check_tool_call(tool_name, input_data, state=policy)",
        "            decision = " + _ALLOW_TOOL + "tool_name, input_data, state=policy)",
        "can_use_tool screens the call before asking the approver",
        tests=_TESTS_CLAUDE,
    ),
    Mutant(
        "hook-langgraph-model-input",
        "adapt_agent/integrations/langgraph.py",
        "threats = self.controls.review_input(",
        "threats = " + _NO_OP,
        "the LangGraph model-call middleware screens the request",
        tests=_TESTS_LANGGRAPH,
    ),
    Mutant(
        "hook-langgraph-model-output",
        "adapt_agent/integrations/langgraph.py",
        "threats = self.controls.review_output(response)",
        "threats = []",
        "the LangGraph model-call middleware screens the response",
        tests=_TESTS_LANGGRAPH,
    ),
    Mutant(
        "hook-langgraph-tool-call",
        "adapt_agent/integrations/langgraph.py",
        'decision = self.gate.check_tool_call(call["name"], call.get("args"), state=state)',
        "decision = " + _ALLOW_TOOL + 'call["name"], call.get("args"), state=state)',
        "the LangGraph tool middleware screens every tool call",
        tests=_TESTS_LANGGRAPH,
    ),
    Mutant(
        "hook-langgraph-tool-result",
        "adapt_agent/integrations/langgraph.py",
        "decision = self.gate.check_tool_result(name, content, state=state)",
        'decision = self.gate.check_tool_result(name, "", state=state)',
        "the LangGraph tool middleware screens every tool result",
        tests=_TESTS_LANGGRAPH,
    ),
    Mutant(
        "adapter-crewai-akickoff",
        "adapt_agent/adapters/crewai/__init__.py",
        'async_run_method_names = ("akickoff", "kickoff_async", "kickoff", "run")',
        'async_run_method_names = ("kickoff_async", "kickoff", "run")',
        "the CrewAI adapter drives native akickoff",
        tests=_TESTS_CREWAI,
    ),
    Mutant(
        "hook-openai-tool-call",
        "adapt_agent/integrations/openai_agents.py",
        "decision = self.call_decision(name, ctx.tool_arguments, ctx)",
        'decision = self.call_decision(name, "{}", ctx)',
        "the OpenAI tool-input guardrail screens the arguments",
        tests=_TESTS_OPENAI,
    ),
    Mutant(
        "hook-openai-tool-result",
        "adapt_agent/integrations/openai_agents.py",
        "decision = self.result_decision(ctx.tool_name, data.output, ctx)",
        'decision = self.result_decision(ctx.tool_name, "", ctx)',
        "the OpenAI tool-output guardrail screens the result",
        tests=_TESTS_OPENAI,
    ),
    Mutant(
        "hook-openai-stream",
        "adapt_agent/integrations/openai_agents.py",
        "found = self.gate.scan_output(texts)",
        "found = []",
        "a governed OpenAI stream screens its text",
        tests=_TESTS_OPENAI,
    ),
    Mutant(
        "hook-openai-model-input",
        "adapt_agent/integrations/openai_agents.py",
        "threats = filter_gate.scan_input(items)",
        "threats = []",
        "the OpenAI model-input filter screens every model call",
        tests=_TESTS_OPENAI,
    ),
    Mutant(
        "adapter-openai-resume-becomes-new-run",
        "adapt_agent/adapters/openai_agents/__init__.py",
        "if _is_run_state(inner) or _is_input_items(inner):",
        "if False:",
        "a RunState resume is passed through, not turned into a new prompt",
        tests=_TESTS_OPENAI,
    ),
    Mutant(
        "hook-google-adk-tool-call",
        "adapt_agent/integrations/google_adk.py",
        "                if decision.blocked:\n"
        "                    raise SecurityBlockedError(decision.reason, list(decision.threats))",
        "                if False:\n"
        "                    raise SecurityBlockedError(decision.reason, list(decision.threats))",
        "a blocked ADK tool call does not run",
        tests=_TESTS_ADK,
    ),
    Mutant(
        "hook-google-adk-tool-result",
        "adapt_agent/integrations/google_adk.py",
        "                if decision.blocked:\n"
        "                    raise SecurityBlockedError(\n"
        '                        f"Tool result from {name!r} withheld',
        "                if False:\n"
        "                    raise SecurityBlockedError(\n"
        '                        f"Tool result from {name!r} withheld',
        "a blocked ADK tool result is replaced",
        tests=_TESTS_ADK,
    ),
    Mutant(
        "hook-google-adk-ask-confirmed",
        "adapt_agent/integrations/google_adk.py",
        'if getattr(confirmation, "confirmed", False) is True:',
        "if True:",
        "an ADK ask runs only on a human confirmation",
        tests=_TESTS_ADK,
    ),
    Mutant(
        "hook-google-adk-stream-aggregate",
        "adapt_agent/integrations/google_adk.py",
        "stream.text += _text_of(",
        "stream.text = _text_of(",
        "an ADK stream is screened as a whole, not chunk by chunk",
        tests=_TESTS_ADK,
    ),
    Mutant(
        "hook-agent-framework-tool-call",
        "adapt_agent/integrations/agent_framework.py",
        "decision = resolved.check_tool_call(name, arguments, state=state)",
        "decision = " + _ALLOW_TOOL + "name, arguments, state=state)",
        "the MAF function middleware screens every tool call",
        tests=_TESTS_MAF,
    ),
    Mutant(
        "hook-agent-framework-tool-result",
        "adapt_agent/integrations/agent_framework.py",
        "if result_decision.blocked:",
        "if False:",
        "a blocked MAF tool result is withheld",
        tests=_TESTS_MAF,
    ),
    Mutant(
        "hook-agent-framework-chat-input",
        "adapt_agent/integrations/agent_framework.py",
        "threats = model_gate.scan_input(_content_payload(screened))",
        "threats = []",
        "the MAF chat middleware screens every model call",
        tests=_TESTS_MAF,
    ),
    Mutant(
        "hook-agent-framework-stream-unbuffered",
        "adapt_agent/integrations/agent_framework.py",
        "context.stream_buffer_updates = True",
        "context.stream_buffer_updates = False",
        "a MAF stream is held until screened",
        tests=_TESTS_MAF,
    ),
    Mutant(
        "hook-agent-framework-output-tool-requests",
        "adapt_agent/integrations/agent_framework.py",
        "        if kind in _TOOL_REQUEST_KINDS:\n            return []",
        "        if False:\n            return []",
        "output screening leaves tool requests to the tool seam",
        tests=_TESTS_MAF,
    ),
    # --------------------------------------------------- optimizer and evals
    Mutant(
        "opt-agent-failure-transient",
        "adapt_agent/optimization/evaluation.py",
        "        return is_infrastructure_error(exc)\n" "\n" "    def _should_retry_agent",
        "        return True\n\n    def _should_retry_agent",
        "an agent's own exception is not excused as transient",
        tests=_TESTS_OPTIMIZER,
    ),
    Mutant(
        "opt-judge-nonfinite",
        "adapt_agent/optimization/judge.py",
        '    if not math.isfinite(value):\n        return None, "non_finite"',
        '    if False:\n        return None, "non_finite"',
        "a NaN or inf judge score is rejected",
        tests=_TESTS_OPTIMIZER,
    ),
    Mutant(
        "opt-claude-tools-widened",
        "adapt_agent/optimization/introspection/claude_agent.py",
        "        if added:\n            raise ValueError(",
        "        if False:\n            raise ValueError(",
        "an optimizer can only narrow allowed_tools",
        tests=_TESTS_OPTIMIZER,
    ),
    Mutant(
        "opt-claude-permission-mode-writable",
        "adapt_agent/optimization/introspection/claude_agent.py",
        "            _read_only(\n" "                obj,\n" '                "permission_mode",',
        "            (lambda o, a, n, k, c, note: bind_attr(o, a, n, k, component=c))(\n"
        "                obj,\n"
        '                "permission_mode",',
        "an optimizer can never write permission_mode",
        tests=_TESTS_OPTIMIZER,
    ),
    Mutant(
        "opt-runner-delegation-skipped",
        "adapt_agent/optimization/runners.py",
        "    delegated = _delegated_runner(agent, output_extractor)\n",
        "    delegated = None\n",
        "framework agents are driven through their own runner, not a probed method",
        tests=_TESTS_OPTIMIZER,
    ),
    *_hook_mutants(),
)


# -- applying mutants -----------------------------------------------------------


def stale_mutants(root: Path = REPO_ROOT) -> list[str]:
    """Describe every mutant whose ``old`` text is not found exactly once.

    Also flags duplicate ids and no-op mutants. Cheap (no test run), so the
    ordinary test suite calls it to keep the list honest between full runs.
    """
    problems: list[str] = []
    seen: set[str] = set()
    sources: dict[str, str] = {}
    for mutant in MUTANTS:
        if mutant.id in seen:
            problems.append(f"{mutant.id}: duplicate id")
        seen.add(mutant.id)
        if mutant.path not in sources:
            path = root / mutant.path
            sources[mutant.path] = path.read_text(encoding="utf-8") if path.is_file() else ""
        for old, new in mutant.edits:
            if old == new:
                problems.append(f"{mutant.id}: old and new are identical")
            count = sources[mutant.path].count(old)
            if count != 1:
                problems.append(
                    f"{mutant.id}: expected its anchor exactly once in {mutant.path}, "
                    f"found {count}: {old.strip().splitlines()[0][:60]!r}"
                )
        # A renamed test file would make pytest exit with a usage error, which
        # must never be mistaken for a kill; and the self-test kills everything.
        for test in mutant.tests:
            if test == _SELF_TEST:
                problems.append(f"{mutant.id}: {_SELF_TEST} cannot judge a mutant")
            elif not (root / test).is_file():
                problems.append(f"{mutant.id}: test file {test} does not exist")
    return problems


_IGNORED = shutil.ignore_patterns(
    ".git",
    "__pycache__",
    "*.pyc",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "build",
    "dist",
    "site",
    "*.egg-info",
    ".coverage",
    "coverage.xml",
)


#: Starts the detail of a run whose test selection was itself broken.
_SELECTION_ERROR = "test selection error"

#: pytest's exit code for "no tests collected". (A missing test file -- exit
#: code 4 -- is caught earlier, by :func:`stale_mutants`; exit code 4 also
#: covers a conftest that fails to import, which a mutant can cause.)
_PYTEST_NO_TESTS = 5


def _run_suite(
    workdir: Path,
    timeout: float,
    *,
    only: str | None = None,
    deselect: tuple[str, ...] = (),
    tests: tuple[str, ...] = (),
) -> tuple[bool, str]:
    """Run pytest in ``workdir``. Returns ``(passed, one-line detail)``.

    ``only`` runs a single test id instead of the suite; ``deselect`` leaves
    test ids out. Both exist for :func:`_check` to confirm a kill. ``tests``
    narrows the run to those files (:attr:`Mutant.tests`).

    A run that collected nothing, or could not start, has not "failed" in any
    sense that says something about the code: its detail starts with
    :data:`_SELECTION_ERROR` so the caller reports it instead of crediting it.
    """
    env = dict(os.environ)
    # The scratch copy must win over any installed (or editable) adapt_agent,
    # and no bytecode may be cached: a mutant the same size as the original,
    # written within the same second, would otherwise run the *stale* .pyc and
    # be reported as a survivor.
    env["PYTHONPATH"] = str(workdir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Selects the derandomized Hypothesis profile (tests/conftest.py): whether a
    # mutant is caught must not depend on which examples happened to be drawn.
    env["CI"] = "1"
    # Wide enough that pytest prints a failing test's full id, which a
    # confirmation run needs verbatim.
    env["COLUMNS"] = "1000"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-x",
        "--no-cov",
        "-p",
        "no:cacheprovider",
        # That file asserts every mutant's anchor is still present in the code.
        # Inside a mutated copy the anchor is, by construction, gone -- so it
        # would "kill" every mutant, including ones no real test notices, and
        # this whole check would pass while proving nothing.
        f"--ignore={_SELF_TEST}",
        *(f"--deselect={test_id}" for test_id in deselect),
    ]
    if only is not None:
        cmd.append(only)
    else:
        cmd.extend(tests)
    try:
        proc = subprocess.run(
            cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        # A mutant that makes the suite hang has certainly been noticed.
        return False, f"timed out after {timeout:.0f}s"
    if proc.returncode == _PYTEST_NO_TESTS:
        tail = (proc.stdout.strip() or proc.stderr.strip()).splitlines()
        return False, f"{_SELECTION_ERROR}: {tail[-1] if tail else proc.returncode}"
    lines = proc.stdout.strip().splitlines()
    failed = next((ln for ln in lines if ln.startswith(("FAILED tests", "ERROR tests"))), "")
    detail = failed or (lines[-1] if lines else proc.stderr.strip()[-200:])
    return proc.returncode == 0, detail


def _failing_test_id(detail: str) -> str | None:
    """The test id from a ``FAILED tests/...::name - reason`` summary line."""
    for prefix in ("FAILED ", "ERROR "):
        if detail.startswith(prefix + "tests"):
            return detail[len(prefix) :].split(" - ", 1)[0].strip()
    return None


KILLED, SURVIVED, STALE, UNCONFIRMED = "killed", "SURVIVED", "STALE", "UNCONFIRMED"


def _confirmed_run(
    workdir: Path, timeout: float, attempts: int = 3, *, tests: tuple[str, ...] = ()
) -> tuple[str, str]:
    """Run the suite against a mutated tree, crediting only reproducible kills.

    The suite stops at its first failure, and under several parallel workers
    that first failure is occasionally a *timing* test that would have failed
    with or without the mutant. Counting that as a kill hides a real survivor
    behind a flake -- CI credited one mutant to an unrelated performance test.

    So a failure is confirmed by re-running that one test alone against the
    same mutated tree. If it fails again the mutant is killed. If it passes it
    was a flake: the suite is run again without that test, so the mutant has to
    be caught by something that actually depends on it.

    Returns ``(status, detail)``. ``UNCONFIRMED`` means every attempt failed on
    a test that then passed alone: nothing reproducible caught the mutant, and
    nothing showed it survives either. It is reported and fails the run --
    never rounded up to a kill.

    ``tests`` narrows every suite run to :attr:`Mutant.tests`. It is passed on
    only when given, so a whole-suite mutant is run exactly as before.
    """
    flaky: list[str] = []
    selection = {"tests": tests} if tests else {}
    for _ in range(attempts):
        passed, detail = _run_suite(workdir, timeout, deselect=tuple(flaky), **selection)
        if passed:
            return SURVIVED, detail
        if detail.startswith(_SELECTION_ERROR):
            return STALE, detail
        test_id = _failing_test_id(detail)
        if test_id is None:
            return KILLED, detail  # a timeout or a collection error: noticed
        if not _run_suite(workdir, timeout, only=test_id)[0]:
            return KILLED, detail
        flaky.append(test_id)
    return UNCONFIRMED, f"{len(flaky)} failures, none reproducible: {', '.join(flaky)}"


@dataclass
class Outcome:
    mutant: Mutant
    status: str  # KILLED | SURVIVED | STALE | UNCONFIRMED
    detail: str


def _check(mutant: Mutant, workdirs: queue.Queue[Path], timeout: float) -> Outcome:
    workdir = workdirs.get()
    try:
        target = workdir / mutant.path
        source = target.read_text(encoding="utf-8")
        if any(source.count(old) != 1 for old, _ in mutant.edits):
            return Outcome(mutant, STALE, "an anchor was not found exactly once")
        target.write_text(mutant.apply(source), encoding="utf-8")
        try:
            status, detail = _confirmed_run(workdir, timeout, tests=mutant.tests)
        finally:
            target.write_text(source, encoding="utf-8")
        return Outcome(mutant, status, detail)
    finally:
        workdirs.put(workdir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--list", action="store_true", help="list mutants and exit")
    parser.add_argument("--only", metavar="PREFIX", help="run only ids starting with PREFIX")
    parser.add_argument(
        "-j", "--jobs", type=int, default=min(4, os.cpu_count() or 1), help="parallel workers"
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="seconds allowed per suite run"
    )
    args = parser.parse_args(argv)

    selected = [m for m in MUTANTS if not args.only or m.id.startswith(args.only)]
    if args.list:
        for m in selected:
            origin = f"  [{m.history}]" if m.history else ""
            print(f"{m.id:34} {m.why}{origin}")
        return 0
    if not selected:
        print(f"No mutant id starts with {args.only!r}", file=sys.stderr)
        return 2

    problems = stale_mutants()
    if problems:
        print("Mutant list is out of date with the code:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    jobs = max(1, min(args.jobs, len(selected)))
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="adapt-mutation-") as tmp:
        workdirs: queue.Queue[Path] = queue.Queue()
        for index in range(jobs):
            workdir = Path(tmp) / f"w{index}"
            shutil.copytree(REPO_ROOT, workdir, ignore=_IGNORED)
            workdirs.put(workdir)

        # Non-vacuity: if the unmutated suite fails, every mutant would be
        # "killed" for reasons that have nothing to do with it.
        baseline = workdirs.get()
        passed, detail = _run_suite(baseline, args.timeout)
        workdirs.put(baseline)
        if not passed:
            print(f"Baseline suite does not pass, so no result would mean anything: {detail}")
            return 1

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            outcomes = list(pool.map(lambda m: _check(m, workdirs, args.timeout), selected))

    for outcome in outcomes:
        print(f"{outcome.status:9} {outcome.mutant.id:34} {outcome.detail[:110]}")
    bad = [o for o in outcomes if o.status != KILLED]
    historical = sum(1 for o in outcomes if o.mutant.history)
    print(
        f"\n{len(outcomes) - len(bad)}/{len(outcomes)} mutants killed "
        f"({historical} re-open a historical vulnerability) "
        f"in {time.perf_counter() - started:.0f}s"
    )
    if bad:
        print("\nNot caught by any test:")
        for outcome in bad:
            origin = f" (re-opens: {outcome.mutant.history})" if outcome.mutant.history else ""
            print(f"  {outcome.mutant.id}: {outcome.mutant.why}{origin}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
