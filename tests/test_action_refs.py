"""Tests for scripts/check_action_refs.py.

The script is a CI gate, so the failure mode that matters most is not "it
reports a bad ref wrongly" but "it silently stops seeing any refs at all" --
a gate that always passes is worse than no gate. The first test pins it
against the real workflow files for exactly that reason.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_action_refs.py"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _load_script():
    spec = importlib.util.spec_from_file_location("check_action_refs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script():
    return _load_script()


def _refs_in_workflows(module) -> set[str]:
    found = set()
    for workflow in WORKFLOWS.glob("*.y*ml"):
        text = workflow.read_text(encoding="utf-8")
        found |= {f"{repo}@{ref}" for repo, ref in module.USES.findall(text)}
    return found


def test_parser_still_sees_the_real_workflow_refs(script):
    """Guards against the gate quietly becoming a no-op."""
    found = _refs_in_workflows(script)
    assert found, "no action refs parsed -- the check would pass vacuously"
    # Both `- uses:` (list item) and `uses:` (mapping key) forms must match.
    assert any(ref.startswith("actions/checkout@") for ref in found)
    assert any(ref.startswith("astral-sh/setup-uv@") for ref in found)


def test_setup_uv_is_pinned_to_a_full_version():
    """setup-uv publishes no floating major tags after v7.

    `@v8`/`@v9`/`@v10` look plausible but do not resolve, and the run dies at
    'Prepare all required actions' before any step executes.
    """
    text = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    pins = {line.split("@")[-1].strip() for line in text.splitlines() if "setup-uv@" in line}
    assert pins, "release.yml no longer references setup-uv"
    for pin in pins:
        assert pin.count(".") == 2, (
            f"setup-uv is pinned to '{pin}'; it must be a full vX.Y.Z tag, "
            "because floating major tags are not published past v7"
        )


def test_missing_ref_fails_and_names_the_newest_version(script, monkeypatch, capsys):
    monkeypatch.setattr(script, "remote_refs", lambda repo: {"v7", "v10.0.1", "main"})
    assert script.main() == 1
    out = capsys.readouterr().out
    assert "MISSING" in out
    assert "v10.0.1" in out, "should point at the newest published full version"


def test_all_resolvable_refs_pass(script, monkeypatch, capsys):
    every_ref = _refs_in_workflows(script)
    monkeypatch.setattr(
        script, "remote_refs", lambda repo: {r.partition("@")[2] for r in every_ref}
    )
    assert script.main() == 0
    assert "MISSING" not in capsys.readouterr().out


def test_unreachable_remote_is_skipped_not_failed(script, monkeypatch, capsys):
    """A network blip must not turn the gate flaky."""
    monkeypatch.setattr(script, "remote_refs", lambda repo: None)
    assert script.main() == 0
    assert "skipped" in capsys.readouterr().out


def test_commit_pins_need_no_network(script, monkeypatch, tmp_path, capsys):
    def explode(repo):  # pragma: no cover - must never be called
        raise AssertionError("commit pins should not hit the network")

    monkeypatch.setattr(script, "remote_refs", explode)
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "w.yml").write_text(
        "jobs:\n  a:\n    steps:\n      - uses: actions/checkout@" + "a" * 40 + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(script, "WORKFLOWS", workflows)
    assert script.main() == 0
    assert "commit pin" in capsys.readouterr().out
