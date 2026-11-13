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
    assert len(EXAMPLES) >= 9


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
