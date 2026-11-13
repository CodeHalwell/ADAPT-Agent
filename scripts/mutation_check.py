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
_OBSERVER = "adapt_agent/observability/__init__.py"
_OPTIMIZERS = "adapt_agent/optimization/optimizers.py"
_GOVERNED = "adapt_agent/adapters/_governed.py"

#: Stands in for a gate call that has been "forgotten": accepts anything,
#: finds no threats, raises nothing.
_NO_OP = "(lambda *a, **k: [])("


def _hook_mutants() -> tuple[Mutant, ...]:
    """One mutant per place a native integration calls the governance gate.

    Each framework hook is a few lines whose entire job is to make that call.
    A hook that silently stops making it still imports, still registers, and
    still returns the framework's expected shape -- it just governs nothing.
    """
    sites = (
        ("agent-framework", "agent_framework", "resolved.review_input(", "input"),
        ("agent-framework", "agent_framework", "resolved.review_output(", "output"),
        ("claude-agent", "claude_agent", "resolved.scan_input(", "input"),
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
    """

    id: str
    path: str
    old: str
    new: str
    why: str
    history: str = ""
    also: tuple[str, ...] = ()
    extra: tuple[tuple[str, str], ...] = ()

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
    ),
    Mutant(
        "fw-whitelist-waives-cap",
        _FIREWALL,
        "if self.max_content_length is not None and len(content) > self.max_content_length:",
        "if (not (self.whitelist_mode and any(p.fullmatch(content) for p in self._allowed_patterns))"
        " and self.max_content_length is not None and len(content) > self.max_content_length):",
        "no allowed pattern can waive the length cap",
    ),
    Mutant(
        "fw-length-cap",
        _FIREWALL,
        "len(content) > self.max_content_length:",
        "len(content) > 2 * self.max_content_length:",
        "oversize content is blocked before any regex runs",
        "2026-04-26 - Firewall Resource Exhaustion / Unbounded Growth",
    ),
    Mutant(
        "fw-events-unbounded",
        _FIREWALL,
        "self._security_events: deque[SecurityEvent] = deque(maxlen=max_events)",
        "self._security_events: deque[SecurityEvent] = deque()",
        "the security-event log is bounded",
        "2026-04-26 - Firewall Resource Exhaustion / Unbounded Growth",
    ),
    Mutant(
        "fw-allow-search",
        _FIREWALL,
        """        if self.whitelist_mode:
            for pattern in self._allowed_patterns:
                if pattern.fullmatch(content):""",
        """        if self.whitelist_mode:
            for pattern in self._allowed_patterns:
                if pattern.search(content):""",
        "an allowed pattern must match the whole content",
        "2024-05-18 - [Firewall Whitelist Bypass via regex search]",
    ),
    Mutant(
        "fw-allow-before-block",
        _FIREWALL,
        "        if self.whitelist_mode:\n            for pattern in self._allowed_patterns:",
        "        if True:\n            for pattern in self._allowed_patterns:",
        "by default an allowed pattern never overrides a block",
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
    ),
    Mutant(
        "fw-snippet-raw",
        _FIREWALL,
        "self._truncate_escaped(self.sanitize(content), 100)",
        "self._truncate_escaped(content, 100)",
        "a blocked secret is redacted in the event that records it",
        "2024-05-14 - Exposed sensitive data in firewall events",
    ),
    Mutant(
        "fw-truncate-then-sanitize",
        _FIREWALL,
        "self._truncate_escaped(self.sanitize(content), 100)",
        "self.sanitize(self._truncate_escaped(content, 100))",
        "sanitise the whole string, then truncate",
        "2024-07-02 - Prevent Partial Secret Leakage via Early Truncation",
        also=("2024-05-25 - Firewall Partial Secret Leakage",),
    ),
    Mutant(
        "fw-snippet-newlines",
        _FIREWALL,
        'return bounded[:best].replace("\\n", "\\\\n").replace("\\r", "\\\\r")',
        "return bounded[:best]",
        "newlines are escaped in stored snippets",
        "2024-07-01 - Prevent log poisoning in firewall security events",
    ),
    Mutant(
        "fw-truncate-overflow",
        _FIREWALL,
        "            if escaped_len <= max_length:\n                best = mid",
        "            if True:\n                best = mid",
        "an escaped snippet never exceeds its length bound",
        "2024-10-27 - Fix flawed truncation logic for escapes",
    ),
    Mutant(
        "fw-redos-guard",
        _FIREWALL,
        '_CATASTROPHIC_RE = re.compile(r"\\([^)(]*[+*]\\)[+*]")',
        '_CATASTROPHIC_RE = re.compile(r"(?!)")',
        "obviously catastrophic patterns are rejected",
        "2024-05-23 - ReDoS in Heuristic Regex Guard",
    ),
    # ------------------------------------------------------------- taint tracker
    Mutant(
        "taint-sources-unbounded",
        _TAINT,
        "if len(self._taint_sources) > self.max_tracked_items:",
        "if False:",
        "tracked taint sources are bounded",
        "2026-05-09 - Dictionary Bounding for DoS Prevention",
    ),
    Mutant(
        "taint-data-unbounded",
        _TAINT,
        "if len(self._tainted_data) > self.max_tracked_items:",
        "if False:",
        "tracked tainted items are bounded",
        "2026-05-09 - Dictionary Bounding for DoS Prevention",
    ),
    Mutant(
        "taint-propagation-unbounded",
        _TAINT,
        "deque(maxlen=max_propagations)",
        "deque()",
        "the propagation log is bounded",
        "2026-05-02 - List Bounding for DoS Prevention",
    ),
    # ------------------------------------------------------------ governance gate
    Mutant(
        "gate-output-unscanned",
        _GATE,
        """        for text in extract_texts(payload):
            if not self.firewall.check_output(text):""",
        """        for text in []:
            if not self.firewall.check_output(text):""",
        "agent output is screened",
    ),
    Mutant(
        "gate-defense-ignored",
        _GATE,
        'if not analysis["is_safe"]:',
        "if False:",
        "an adversarial-defense finding blocks the input",
    ),
    Mutant(
        "gate-policy-skipped",
        _GATE,
        "        if state is not None:\n            violations = [f",
        "        if False:\n            violations = [f",
        "blocking policy rules are enforced on input",
    ),
    Mutant(
        "gate-warn-rules-block",
        _GATE,
        'if rule is not None and rule.get("action") == "block":',
        "if rule is not None:",
        "only action=block rules block",
    ),
    Mutant(
        "gate-report-only-ignored",
        _GATE,
        "        if threats and self.block_on_violation:\n            raise SecurityBlockedError("
        'self._reason("Input blocked by security controls"), threats)',
        "        if threats and False:\n            raise SecurityBlockedError("
        'self._reason("Input blocked by security controls"), threats)',
        "a threat in the input raises when blocking is on",
    ),
    Mutant(
        "gate-dict-keys",
        _GATE,
        "if isinstance(k, str) and k:",
        "if False:",
        "dict keys are scanned, not only values",
    ),
    Mutant(
        "gate-walk-depth",
        _GATE,
        "_MAX_WALK_DEPTH = 12",
        "_MAX_WALK_DEPTH = 6",
        "the scan reaches a real nested tool result",
    ),
    Mutant(
        "gate-structured-fields",
        _GATE,
        "            if fields is not None:\n                _walk(fields, depth + 1)",
        "            if False:\n                _walk(fields, depth + 1)",
        "fields of a structured output are scanned",
    ),
    Mutant(
        "gate-tool-responses",
        _GATE,
        '    "function_response",\n    "tool_response",\n    "response",\n)',
        ")",
        "tool results returning to the model are scanned",
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
    ),
    Mutant(
        "policy-depth-guard",
        _POLICY,
        "if depth > 50:",
        "if depth > 10**9:",
        "condition evaluation depth is bounded",
        "2024-05-25 - AST Evaluation Depth Limit",
    ),
    Mutant(
        "policy-length-guard",
        _POLICY,
        "if len(condition) > 1024:",
        "if len(condition) > 10**9:",
        "condition length is bounded before parsing",
        "2024-05-24 - DoS Vulnerability in AST parsing",
    ),
    Mutant(
        "policy-node-guard",
        _POLICY,
        "if node_count > MAX_CONDITION_NODES:",
        "if node_count > 10**9:",
        "condition node count is bounded",
        "2024-05-24 - DoS Vulnerability in AST parsing",
    ),
    Mutant(
        "policy-allows-calls",
        _POLICY,
        "    ast.Subscript,\n)",
        "    ast.Subscript,\n    ast.Call,\n)",
        "function calls are rejected in conditions",
        "2026-05-05 - AST Whitelisting vs Arbitrary eval()",
    ),
    Mutant(
        "policy-allows-attributes",
        _POLICY,
        "    ast.Subscript,\n)",
        "    ast.Subscript,\n    ast.Attribute,\n)",
        "attribute access is rejected in conditions",
        "2026-05-05 - AST Whitelisting vs Arbitrary eval()",
    ),
    Mutant(
        "policy-allows-pow",
        _POLICY,
        "        ast.Div: operator.truediv,\n",
        "        ast.Div: operator.truediv,\n        ast.Pow: operator.pow,\n",
        "unbounded arithmetic (**) is rejected in conditions",
        "2026-05-05 - AST Whitelisting vs Arbitrary eval()",
    ),
    Mutant(
        "policy-unknown-names",
        _POLICY,
        "if isinstance(node, ast.Name) and node.id not in CONDITION_NAMES:",
        "if False:",
        "a rule naming an unknown variable is rejected at registration",
    ),
    Mutant(
        "policy-violations-unbounded",
        _POLICY,
        "deque(maxlen=max_violations)",
        "deque()",
        "the violation log is bounded",
        "2026-05-02 - List Bounding for DoS Prevention",
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
    ),
    Mutant(
        "trust-agents-unbounded",
        _TRUST,
        "if len(self._trust_scores) > self.max_agents:",
        "if False:",
        "tracked agents are bounded",
        "2026-05-09 - Dictionary Bounding for DoS Prevention",
    ),
    # --------------------------------------------------------------------- memory
    Mutant(
        "memory-long-term-unbounded",
        _MEMORY,
        "if len(self._long_term_memory) > self.long_term_capacity:",
        "if False:",
        "long-term memory holds at most its capacity",
        "2025-05-10 - [Memory Exhaustion DoS in MemorySystem]",
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
    ),
    # --------------------------------------------------------- adversarial defense
    Mutant(
        "defense-custom-patterns",
        _DEFENSE,
        '        if custom is not None:\n            self._record_attack("custom_pattern"',
        '        if False:\n            self._record_attack("custom_pattern"',
        "registered custom patterns are actually matched",
        "2026-05-13 - Broken Custom Attack Pattern Enforcement",
    ),
    Mutant(
        "defense-length-cap",
        _DEFENSE,
        "if self.max_content_length is not None and len(input_text) > self.max_content_length:",
        "if False:",
        "oversize input is refused before analysis",
        "2026-05-23 - AdversarialDefense Resource Exhaustion",
    ),
    Mutant(
        "defense-jailbreak-dropped",
        _DEFENSE,
        "        if jailbreak is not None:",
        "        if False:",
        "a detected jailbreak marks the input unsafe",
    ),
    Mutant(
        "defense-injection-dropped",
        _DEFENSE,
        "        if injection is not None:",
        "        if False:",
        "a detected injection marks the input unsafe",
    ),
    Mutant(
        "defense-attacks-unbounded",
        _DEFENSE,
        "deque(maxlen=max_attacks)",
        "deque()",
        "the detected-attack log is bounded",
        "2026-05-02 - List Bounding for DoS Prevention",
    ),
    Mutant(
        "defense-content-newlines",
        _DEFENSE,
        """            "content": content[:256]
            .replace("\\n", "\\\\n")
            .replace("\\r", "\\\\r")[:100],""",
        """            "content": content[:100],""",
        "newlines are escaped in a recorded attack's content",
        "2024-06-06 - Log Poisoning Vulnerability in Adversarial Defense",
    ),
    Mutant(
        "defense-indicator-newlines",
        _DEFENSE,
        """            "indicator": indicator[:256]
            .replace("\\n", "\\\\n")
            .replace("\\r", "\\\\r")[:100],""",
        """            "indicator": indicator[:100],""",
        "newlines are escaped in a recorded attack's indicator",
        "2024-11-07 - Log Poisoning in Adversarial Attack Indicators",
    ),
    Mutant(
        "defense-result-newlines",
        _DEFENSE,
        """            "input": input_text[:100].replace("\\n", "\\\\n").replace("\\r", "\\\\r"),
            "threats_detected": threats,""",
        """            "input": input_text[:100],
            "threats_detected": threats,""",
        "newlines are escaped in the returned analysis snippet",
        "2024-10-27 - Fix log poisoning in AdversarialDefense analysis results",
    ),
    # ------------------------------------------------------------------- observer
    Mutant(
        "observer-logs-unbounded",
        _OBSERVER,
        "deque(maxlen=max_logs)",
        "deque()",
        "the in-memory log is bounded",
        "2026-05-02 - List Bounding for DoS Prevention",
    ),
    Mutant(
        "observer-trace-events-unbounded",
        _OBSERVER,
        'if len(self._traces[trace_id]["events"]) > self.max_events_per_trace:',
        "if False:",
        "events per trace are bounded",
        "2026-05-07 - AgentObserver Nested List and Dictionary Bounding for DoS Prevention",
    ),
    Mutant(
        "observer-metric-names-unbounded",
        _OBSERVER,
        "if len(self._metrics) > self.max_metric_names:",
        "if False:",
        "distinct metric names are bounded",
        "2026-05-07 - AgentObserver Nested List and Dictionary Bounding for DoS Prevention",
    ),
    Mutant(
        "observer-message-length",
        _OBSERVER,
        '"message": message[:10000].replace',
        '"message": message.replace',
        "a logged message is length-bounded",
        "2024-05-29 - [Memory Exhaustion DoS in Logging]",
    ),
    Mutant(
        "observer-message-newlines",
        _OBSERVER,
        '"message": message[:10000].replace("\\n", "\\\\n").replace("\\r", "\\\\r"),',
        '"message": message[:10000],',
        "newlines are escaped in a logged message",
        "2024-06-25 - [Log Poisoning Vulnerability in AgentObserver]",
    ),
    Mutant(
        "observer-description-newlines",
        _OBSERVER,
        '"description": description[:10000].replace("\\n", "\\\\n").replace("\\r", "\\\\r"),',
        '"description": description[:10000],',
        "newlines are escaped in an event description",
        "2024-06-25 - [Log Poisoning Vulnerability in AgentObserver]",
    ),
    # ----------------------------------------------------------- governed adapter
    Mutant(
        "gov-input-not-refused",
        _GOVERNED,
        "        if in_threats and adapter.block_on_violation:",
        "        if False:",
        "the adapter refuses an input its controls flagged",
    ),
    Mutant(
        "gov-policy-not-refused",
        _GOVERNED,
        "        if blocking and adapter.block_on_violation:",
        "        if False:",
        "the adapter refuses an input a blocking rule matched",
    ),
    Mutant(
        "gov-output-not-refused",
        _GOVERNED,
        "            if out_threats and adapter.block_on_violation:",
        "            if False:",
        "the adapter refuses an output its firewall flagged",
    ),
    Mutant(
        "gov-refused-request-becomes-state",
        _GOVERNED,
        "        candidate_state = adapter.extract_state(input_data)",
        "        candidate_state = self._last_state = adapter.extract_state(input_data)",
        "a refused request never becomes the agent's state",
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
    ),
    Mutant(
        "policy-missed-lookup-untracked",
        _POLICY,
        "                context[_MISSED_LOOKUP] = True",
        "                pass",
        "a lookup of an absent key is counted as unresolved",
    ),
    # ------------------------------------------------------------------ optimizer
    Mutant(
        "opt-proposer-judge-ignored",
        _OPTIMIZERS,
        "return self.proposer_judge if self.proposer_judge is not None else self.judge",
        "return self.judge",
        "a separate proposer judge does the rewriting",
    ),
    Mutant(
        "opt-shared-judge-unflagged",
        _OPTIMIZERS,
        "if not any(same(rewriting, j) for j in scoring):",
        "if True:",
        "a judge that both writes and scores is reported",
    ),
    Mutant(
        "opt-gate-never-rejects",
        _OPTIMIZERS,
        "if comparable and tuned.score < baseline.score - self.validation_tolerance:",
        "if False:",
        "a config that regresses on held-out data is not applied",
    ),
    Mutant(
        "opt-gate-leaves-config-applied",
        _OPTIMIZERS,
        "            # Leave the baseline applied: the agent is exactly as it was given.\n",
        "            target.apply(best_config)\n",
        "a rejected config is removed from the live agent",
    ),
    Mutant(
        "opt-gate-trusts-partial-scores",
        _OPTIMIZERS,
        "if comparable and tuned.score < baseline.score - self.validation_tolerance:",
        "if tuned.score < baseline.score - self.validation_tolerance:",
        "the gate never rejects on an incomplete held-out pass",
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


def _run_suite(workdir: Path, timeout: float) -> tuple[bool, str]:
    """Run pytest in ``workdir``. Returns ``(passed, one-line detail)``."""
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
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        # A mutant that makes the suite hang has certainly been noticed.
        return False, f"timed out after {timeout:.0f}s"
    lines = proc.stdout.strip().splitlines()
    failed = next((ln for ln in lines if ln.startswith(("FAILED tests", "ERROR tests"))), "")
    detail = failed or (lines[-1] if lines else proc.stderr.strip()[-200:])
    return proc.returncode == 0, detail


@dataclass
class Outcome:
    mutant: Mutant
    status: str  # "killed" | "SURVIVED" | "STALE"
    detail: str


def _check(mutant: Mutant, workdirs: queue.Queue[Path], timeout: float) -> Outcome:
    workdir = workdirs.get()
    try:
        target = workdir / mutant.path
        source = target.read_text(encoding="utf-8")
        if any(source.count(old) != 1 for old, _ in mutant.edits):
            return Outcome(mutant, "STALE", "an anchor was not found exactly once")
        target.write_text(mutant.apply(source), encoding="utf-8")
        try:
            passed, detail = _run_suite(workdir, timeout)
        finally:
            target.write_text(source, encoding="utf-8")
        return Outcome(mutant, "SURVIVED" if passed else "killed", detail)
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
    bad = [o for o in outcomes if o.status != "killed"]
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
