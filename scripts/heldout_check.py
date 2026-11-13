#!/usr/bin/env python3
"""Held-out check: does this change still pass the tests it did not write?

A pull request is judged by the tests *in that pull request*. An author --
human or automated -- who is asked to make a failing check pass can do it by
fixing the code or by editing the check, and both produce a green run.
``mutation_check.py`` catches a weakened test for the bugs on its list; this
catches the general case, with no list:

1. **The base branch's tests are run against the pull request's code.** Tests
   the change added are not consulted, and tests it edited or deleted are used
   as they were *before* the change. If they fail, the change altered behaviour
   that an existing test pinned.
2. **No mutant may disappear.** Every id in the base branch's
   ``mutation_check.py`` must still exist in the pull request's.

Neither failure means the change is wrong. Behaviour is allowed to change, and
then the tests that pinned the old behaviour must change with it. It means the
change is an *amendment to the specification* rather than an implementation
detail, and a person should look at exactly that: which previously-passing
tests no longer hold, and why. In CI a maintainer records that decision by
applying the ``spec-amendment`` label, which skips this job.

For that reason CI runs the **base branch's copy of this script**, not the pull
request's: a check that the change under review can rewrite is not held out.

Usage::

    python scripts/heldout_check.py --base PATH_TO_BASE_CHECKOUT [--head PATH]

Locally, a base checkout is one command::

    git worktree add /tmp/base origin/main
    python scripts/heldout_check.py --base /tmp/base
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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
    ".hypothesis",
    "build",
    "dist",
    "site",
    "*.egg-info",
    ".coverage",
    "coverage.xml",
)


_LIST_IDS = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("_mutation_check", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules["_mutation_check"] = module
spec.loader.exec_module(module)
print("\\n".join(m.id for m in module.MUTANTS))
"""


def mutant_ids(script: Path) -> set[str] | None:
    """The mutant ids a ``mutation_check.py`` declares, or ``None`` if absent.

    Loaded in a child process and read off the real ``MUTANTS`` tuple, because
    some ids are built by a helper and no amount of reading the source text
    would find them. A script that cannot be loaded declares no mutants -- so a
    change that breaks it is reported as having removed all of them.
    """
    if not script.is_file():
        return None
    proc = subprocess.run(
        [sys.executable, "-c", _LIST_IDS, str(script)], capture_output=True, text=True
    )
    if proc.returncode != 0:
        return set()
    return set(proc.stdout.split())


def removed_mutants(base: Path, head: Path) -> list[str]:
    before = mutant_ids(base / "scripts" / "mutation_check.py")
    after = mutant_ids(head / "scripts" / "mutation_check.py")
    if before is None:
        return []  # the base branch predates the mutation check
    return sorted(before - (after or set()))


def run_base_tests(base: Path, head: Path, timeout: float) -> tuple[bool, list[str], str]:
    """Run the base branch's ``tests/`` against the head's code.

    Returns ``(passed, failing test ids, summary line)``.
    """
    with tempfile.TemporaryDirectory(prefix="adapt-heldout-") as tmp:
        work = Path(tmp) / "work"
        base_resolved = base.resolve()

        def ignore(directory: str, names: list[str]) -> set[str]:
            skipped = set(_IGNORED(directory, names))
            # A base checkout placed inside the head tree (as CI does) must not
            # be copied into the scratch tree it is being compared against.
            skipped.update(n for n in names if (Path(directory) / n).resolve() == base_resolved)
            return skipped

        shutil.copytree(head, work, ignore=ignore)
        shutil.rmtree(work / "tests", ignore_errors=True)
        shutil.copytree(base / "tests", work / "tests", ignore=_IGNORED)

        env = dict(os.environ)
        env["PYTHONPATH"] = str(work)  # the scratch copy, not an installed package
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["CI"] = "1"  # derandomized property tests: same examples every run
        cmd = [sys.executable, "-m", "pytest", "-q", "--no-cov", "-p", "no:cacheprovider"]
        try:
            proc = subprocess.run(
                cmd, cwd=work, env=env, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return False, [], f"timed out after {timeout:.0f}s"
        lines = proc.stdout.strip().splitlines()
        failing = [
            ln.split(" ", 1)[1].split(" - ", 1)[0]
            for ln in lines
            if ln.startswith(("FAILED tests", "ERROR tests"))
        ]
        summary = lines[-1] if lines else proc.stderr.strip()[-300:]
        return proc.returncode == 0, failing, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True, type=Path, help="checkout of the base branch")
    parser.add_argument(
        "--head",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="checkout of the change under review (default: this repository)",
    )
    parser.add_argument("--timeout", type=float, default=1200.0)
    args = parser.parse_args(argv)

    if not (args.base / "tests").is_dir():
        print(f"{args.base} has no tests/ directory; nothing to hold out.", file=sys.stderr)
        return 2

    problems = False

    removed = removed_mutants(args.base, args.head)
    if removed:
        problems = True
        print("Mutants present on the base branch and missing from this change:")
        for mutant_id in removed:
            print(f"  {mutant_id}")
        print()

    passed, failing, summary = run_base_tests(args.base, args.head, args.timeout)
    print(f"Base-branch tests against this change: {summary}")
    if not passed:
        problems = True
        if failing:
            print("\nTests that passed before this change and do not pass with it:")
            for test_id in failing:
                print(f"  {test_id}")

    if not problems:
        print("\nHeld-out check passed: every pre-existing test still holds.")
        return 0

    print(
        "\nThis is not necessarily a defect. It means the change alters behaviour that an\n"
        "existing test pinned (or drops a mutant), so it amends the specification rather\n"
        "than only the implementation. A maintainer should review exactly the items above\n"
        "and, if they are intended, apply the `spec-amendment` label to the pull request."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
