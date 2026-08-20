#!/usr/bin/env python3
"""Verify every ``uses:`` ref in .github/workflows/ actually resolves.

GitHub resolves action refs while *preparing* a workflow run, before any step
executes, so a typo'd or retired ref fails the whole run with

    Unable to resolve action `owner/repo@ref`, unable to find version `ref`

For release.yml that only happens once a tag is pushed -- the most expensive
place to discover it. This check moves that failure into ordinary CI.

The trap this was written for: ``astral-sh/setup-uv`` published floating major
tags (``v1``..``v7``) and then stopped, so ``@v8``/``@v9``/``@v10`` look
perfectly reasonable but do not exist. Only full ``vX.Y.Z`` tags do.

Run locally with::

    python3 scripts/check_action_refs.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"

# `uses: owner/repo@ref`, optionally with a leading list dash. Local (`./...`)
# and docker (`docker://...`) uses have no remote to check and are skipped by
# the owner/repo shape itself.
USES = re.compile(
    r"^\s*(?:-\s*)?uses:\s*([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)@([A-Za-z0-9._/-]+)",
    re.MULTILINE,
)
SHA = re.compile(r"^[0-9a-f]{40}$")


def remote_refs(repo: str) -> set[str] | None:
    """Short ref names published by ``repo``, or None if the remote is unreachable.

    Unreachable is deliberately not a failure: a network blip must not turn this
    into a flaky gate. A *reachable* remote missing the ref is the real bug.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-remote", "--tags", "--heads", f"https://github.com/{repo}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    refs = set()
    for line in proc.stdout.splitlines():
        _, _, ref = line.partition("refs/")
        if ref and not ref.endswith("^{}"):
            refs.add(ref.split("/", 1)[-1] if "/" in ref else ref)
    return refs or None


def main() -> int:
    pinned: dict[str, set[Path]] = {}
    for workflow in sorted(WORKFLOWS.glob("*.y*ml")):
        for repo, ref in USES.findall(workflow.read_text(encoding="utf-8")):
            pinned.setdefault(f"{repo}@{ref}", set()).add(workflow)

    if not pinned:
        print("No external action refs found -- nothing to check.")
        return 0

    broken, skipped = [], []
    cache: dict[str, set[str] | None] = {}

    for use in sorted(pinned):
        repo, _, ref = use.partition("@")
        if SHA.match(ref):
            print(f"  ok       {use}  (commit pin)")
            continue
        if repo not in cache:
            cache[repo] = remote_refs(repo)
        refs = cache[repo]
        if refs is None:
            skipped.append(use)
            print(f"  skipped  {use}  (remote unreachable)")
        elif ref in refs:
            print(f"  ok       {use}")
        else:
            broken.append(use)
            print(f"  MISSING  {use}")

    if broken:
        print()
        for use in broken:
            where = ", ".join(sorted(p.name for p in pinned[use]))
            print(f"::error::{use} does not resolve (used in {where})")
            repo = use.partition("@")[0]
            newest = sorted(
                (r for r in (cache[repo] or ()) if re.fullmatch(r"v\d+\.\d+\.\d+", r)),
                key=lambda r: [int(n) for n in r.lstrip("v").split(".")],
            )
            if newest:
                print(f"::error::  newest full version published by {repo}: {newest[-1]}")
        print(f"\n{len(broken)} action ref(s) would fail at 'Prepare all required actions'.")
        return 1

    print(f"\nAll {len(pinned) - len(skipped)} checked action ref(s) resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
