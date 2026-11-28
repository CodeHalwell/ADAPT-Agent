"""Every standalone example must actually run, and say what its README says.

``examples/06_optimize_with_golden_dataset.py`` shipped in 0.4.0 exiting 0 and
printing ``improvement=+0.000``: its offline judge stub matched a prompt format
the library had since changed, so the "optimization" example optimized nothing
and nothing noticed. These run each top-level example as a user would and
check the outcome it exists to demonstrate.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted((REPO_ROOT / "examples").glob("[0-9][0-9]_*.py"))


def _run(script: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # Run against this checkout, whatever happens to be installed.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_examples_were_found():
    assert len(EXAMPLES) >= 15


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs_cleanly(script):
    result = _run(script)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip(), "the example printed nothing"
    assert "Traceback" not in result.stderr


def test_every_example_is_listed_in_the_examples_readme():
    readme = (REPO_ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    missing = [p.name for p in EXAMPLES if p.name not in readme]
    assert missing == []


def test_the_optimization_example_actually_optimizes():
    out = _run(REPO_ROOT / "examples" / "06_optimize_with_golden_dataset.py").stdout
    assert "improvement=+1.000" in out
    assert "Answer for France now: Paris" in out
    assert "rejected by the gate: False" in out


def test_the_guardrail_verification_example_shows_each_outcome():
    out = _run(REPO_ROOT / "examples" / "09_verifying_your_guardrails.py").stdout
    assert "'SECRET-[A-Z]{40}': 0" in out  # the pattern that never matches
    assert out.count("\n  rejected ") == 3  # three unevaluable rules refused
    assert "default enforcer : ['low_trust']" in out  # fail-closed
    assert "fail_closed=False: []" in out
    assert "vacuous: ['low_trust', 'free_tier']" in out
    assert "state unchanged by the refusal: True" in out


def _out(name: str) -> str:
    result = _run(REPO_ROOT / "examples" / name)
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


def test_the_adversarial_example_is_honest_about_both_sides():
    out = _out("02_adversarial_defense.py")
    caught, missed = out.split("== 3. What it does NOT catch ==")
    missed = missed.split("== 4.")[0]
    # Everything in section 2 is flagged; everything in section 3 gets through.
    hidden = caught.split("== 2.")[1]
    assert hidden.count("FLAGGED") == 10 and " safe" not in hidden
    assert missed.count(" safe") == 5 and "FLAGGED" not in missed
    assert "a security tutorial    FLAGGED" in out  # the false positive
    assert "near miss              safe" in out
    assert "oversized input -> ['content_too_long']" in out


def test_the_trust_taint_memory_example_shows_each_control_biting():
    out = _out("10_trust_taint_and_memory.py")
    assert "scraper claims 1.00; evaluated at 0.00" in out  # self-report ignored
    assert "policy on scraper    -> ['low_trust']" in out
    assert "policy on researcher -> allowed" in out
    assert "tool_args     level=HIGH" in out
    assert "question -[concatenate]-> prompt" in out and "page -[concatenate]-> prompt" in out
    assert "run the tool call?               False" in out
    assert "run it after validating the args? True" in out
    assert "unknown source -> CRITICAL" in out
    assert "fact-0 (evicted): None" in out
    assert "consolidated 1 item(s)" in out
    assert "long-term holds 2 (capacity 2)" in out


def test_the_middleware_async_rollout_example_shows_each_outcome():
    out = _out("11_middleware_async_and_rollout.py")
    assert "in : mail me at [email] | tag: support-queue" in out
    assert "caller's copy untouched: True" in out
    assert "fail-open  : continued with unsanitised data" in out
    assert "fail-closed: aborted" in out
    assert "out: You said: I am [email] (AI-generated" in out
    assert out.count("completed: done") == 2
    assert "blocked  : ['firewall', 'prompt_injection']" in out
    assert "execute() inside a loop -> Cannot synchronously run" in out
    # Report-only: all four served, and everything that would block is recorded.
    assert "served 4 of 4 requests" in out
    assert "['blocked_input', 'blocked_output']" in out
    assert "adversarial detections          : ['prompt_injection']" in out
    assert "policy rules that fired         : ['low_trust']" in out


def test_the_multi_framework_example_governs_all_seven_adapters_alike():
    out = _out("05_multi_framework_adapters.py")
    adapters = (
        "LangGraph",
        "Pydantic AI",
        "Microsoft Agent Framework",
        "CrewAI",
        "OpenAI Agents SDK",
        "Google ADK",
        "Claude Agent SDK",
    )
    for name in adapters:
        assert f"[{name}] blocked: threats=['firewall', 'prompt_injection']" in out
    served = out.split("=== 1.")[1].split("blocked the same way")[0]
    assert served.count("What is the capital of France?'") == 6  # CrewAI gets template inputs
    # Run-scoped keyword arguments reach the framework; a bare string is accepted.
    assert "on thread user-42" in out
    assert "(deps={'user': 'alice'})" in out
    assert "answered: 'What is 2 + 2?'" in out
    # An input refusal leaves no trace but is recorded as a refusal.
    assert "support stage=input threats=['firewall', 'prompt_injection']" in out
    assert "traces started     : 0" in out
    assert "gate.check_tool_call('delete_records') -> block" in out
    assert "gate.check_tool_call('send_email'    ) -> ask" in out
    assert "without hooks=True -> tool_policy, on_tool_violation and approve" in out


def test_the_native_hooks_example_shows_what_a_wrapper_misses():
    out = _out("12_native_framework_hooks.py")
    assert "outer wrapper       : allowed" in out
    assert "hook before 'writer': BLOCKED ['firewall', 'prompt_injection'] [writer]" in out
    assert "[triage]" in out and "[billing]" in out
    assert "locked account: BLOCKED ['policy:locked'] [support]" in out
    assert "normal account: allowed" in out
    assert "task guardrail : accepted=False" in out
    assert "[adk-agent]" in out and out.count("[maf-agent]") == 2
    assert "[('maf-agent', 'completed'), ('maf-agent', 'error')]" in out
    assert "flagged text in a field : BLOCKED ['firewall'] [pai-agent]" in out
    assert "cannot honour policy_enforcer" in out


def test_the_native_hooks_example_shows_each_tool_verdict():
    # The parts of example 12 that run with no framework installed; a binding
    # that builds SDK objects prints a skip line instead, so it is not pinned.
    out = _out("12_native_framework_hooks.py")
    # The gate's own tool API: allow, block, ask, and an ask that fails closed.
    assert "search                : allow []" in out
    assert "delete_user           : block ['tool:denied:delete_user']" in out
    assert "send_email            : ask   ['tool:approval:send_email']" in out
    assert "refund 900            : ask   ['policy:big_refunds']" in out
    assert "search (bad argument) : block ['firewall', 'prompt_injection']" in out
    assert "fetch result          : block ['firewall', 'prompt_injection']" in out
    assert "ask, no approval path : block" in out
    assert "report-only           : allow report_only=True" in out
    # LangGraph's ToolNode wrapper: an ask with no checkpointer is a block.
    lg = out.split("== 4a.")[1].split("deprecated constructor")[0]
    assert "tool: search      -> allowed" in lg
    assert "tool: delete_user -> BLOCKED ['tool:denied:delete_user']" in lg
    assert "tool: send_email  -> BLOCKED ['tool:approval:send_email']" in lg
    assert "tool result       -> BLOCKED ['firewall', 'prompt_injection']" in lg
    # ADK: refusal as the tool's response, an ask routed to confirmation.
    adk = out.split("== 4c.")[1].split("== 4d.")[0]
    assert "refused by ADAPT-Agent [adk-agent]: tool:denied:delete_user." in adk
    assert "confirmation requested: True" in adk
    assert "withheld by ADAPT-Agent [adk-agent]: firewall, prompt_injection." in adk
    # MAF function middleware: no AgentSession, so the ask is refused.
    maf = out.split("== 4d.")[1].split("== 4e.")[0]
    assert "tool: search      -> ok" in maf
    assert "refused by ADAPT-Agent [maf-tools]: tool:denied:delete_user." in maf
    assert "needs approval [maf-tools]: tool:approval:send_email. (refused)" in maf
    assert "withheld by ADAPT-Agent [maf-tools]: firewall, prompt_injection." in maf
    # Claude's CLI options are derived without the SDK, and never auto-approve.
    assert "'disallowed_tools': ['Bash'], 'tools': ['Read', 'Grep']" in out
    assert "allowed_tools" not in out.replace("disallowed_tools", "")


def test_the_optimizer_strategies_example_reaches_its_stated_results():
    out = _out("13_optimizer_strategies.py")
    lines = {line.split("  ")[1].strip(): line for line in out.splitlines() if " -> " in line}
    for strategy in ("grid", "random", "coordinate ascent", "evolutionary", "default pipeline"):
        assert "0.00 -> 1.00" in lines[strategy], lines[strategy]
    assert "0.00 -> 0.00" in lines["bootstrap few-shot"]  # nothing to bootstrap from
    assert "0.67 -> 1.00" in lines["few-shot, small model"]
    assert "applied 'MEMORISED ANSWERS'" in out  # without held-out data
    assert "rejected=True" in out
    assert "what was refused : {'triage.prompt': 'MEMORISED ANSWERS'}" in out
    assert "reloaded into a fresh agent -> held-out score 1.00" in out
    assert "High token usage" in out


def test_the_end_to_end_example_tells_a_consistent_story():
    out = _out("14_end_to_end_support_desk.py")
    assert "config errors            : none" in out
    assert out.count("'low_trust': 'violation'") == 1
    assert "'long_thread': 'violation'" in out
    assert "[withheld: a knowledge-base page failed screening]" in out
    assert "[refused: prompt_injection, custom_pattern]" in out
    assert "[refused: firewall]" in out
    assert "[refused: policy:low_trust]" in out
    # The innocent caller who hit our poisoned page keeps their trust.
    assert "'bob': 0.6" in out and "'mallory': 0.0" in out
    assert "refunds issued      : none" in out
    assert "refunds for a human : ['alice']" in out
    assert "vacuous rules       : none" in out
    assert "3 completed, 1 error" in out
    assert "['Email me at [email] about returns']" in out
    assert "75% of 4 golden answers correct" in out


def test_the_cli_tour_runs_every_command():
    out = _out("15_cli_tour.py")
    assert "$ adapt-agent validate config.json        (exit 0)" in out
    assert "$ adapt-agent validate bad_config.json        (exit 1)" in out
    assert "Configuration is INVALID (3 error(s))" in out
    assert '"status": "ready"' in out
    assert "installed files: ['.adapt-skill.json', 'SKILL.md', 'references']" in out
    assert "score before: 0.00" in out
    assert "0.00 -> 1.00 on training, 1.00 on held-out, rejected by the gate: False" in out
    assert '"capitals.prompt": "Reply with only the city name."' in out
    assert out.count("(exit 0)") == 8
