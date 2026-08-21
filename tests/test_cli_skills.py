"""Tests for the ``adapt install skill`` / ``adapt skills`` CLI commands."""

import json
from pathlib import Path

import pytest

from adapt_agent.cli import main

SKILL_NAME = "adapt-agent"
SKILL_FILE = "SKILL.md"


def _installed(root: Path) -> Path:
    return root / ".claude" / "skills" / SKILL_NAME


# -- install -------------------------------------------------------------------


def test_install_skill_into_project(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["install", "skill"]) == 0
    out = capsys.readouterr().out
    assert "Installed the 'adapt-agent' skill" in out
    assert (_installed(tmp_path) / SKILL_FILE).is_file()
    assert (_installed(tmp_path) / "references" / "evals.md").is_file()


def test_install_skill_prints_relative_path(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["install", "skill"])
    assert ".claude/skills/adapt-agent" in capsys.readouterr().out


def test_install_skills_plural_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["install", "skills"]) == 0
    assert _installed(tmp_path).is_dir()


def test_install_named_skill(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["install", "skill", SKILL_NAME]) == 0
    assert _installed(tmp_path).is_dir()


def test_install_unknown_skill_returns_one(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["install", "skill", "nope"]) == 1
    assert "Unknown skill" in capsys.readouterr().out


def test_install_unknown_skill_json(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["install", "skill", "nope", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "Unknown skill" in payload["error"]


def test_install_twice_without_force_returns_one(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["install", "skill"]) == 0
    capsys.readouterr()
    assert main(["install", "skill"]) == 1
    assert "--force" in capsys.readouterr().out


def test_install_force_updates(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["install", "skill"])
    (_installed(tmp_path) / SKILL_FILE).write_text("edited locally", encoding="utf-8")
    capsys.readouterr()

    assert main(["install", "skill", "--force"]) == 0
    assert "Updated the 'adapt-agent' skill" in capsys.readouterr().out
    assert (_installed(tmp_path) / SKILL_FILE).read_text(encoding="utf-8").startswith("---")


def test_install_json_output(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["install", "skill", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    entry = payload["installed"][0]
    assert entry["name"] == SKILL_NAME
    assert entry["replaced"] is False
    assert SKILL_FILE in entry["files"]
    assert Path(entry["path"]).is_dir()


def test_install_explicit_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "vendor" / "skills"
    assert main(["install", "skill", "--dir", str(target)]) == 0
    assert (target / SKILL_NAME / SKILL_FILE).is_file()
    assert not (tmp_path / ".claude").exists()  # --dir overrides the default target


def test_install_user_target(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(tmp_path)

    assert main(["install", "skill", "--target", "user"]) == 0
    assert (home / ".claude" / "skills" / SKILL_NAME / SKILL_FILE).is_file()
    assert "will now discover it" in capsys.readouterr().out


def test_install_rejects_unknown_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):  # argparse rejects the choice
        main(["install", "skill", "--target", "cursor"])


def test_install_requires_what_argument():
    with pytest.raises(SystemExit):
        main(["install"])


# -- skills listing --------------------------------------------------------------


def test_skills_lists_bundled_skill(capsys):
    assert main(["skills"]) == 0
    out = capsys.readouterr().out
    assert SKILL_NAME in out
    assert "install skill" in out  # tells the user what to do next


def test_skills_json(capsys):
    assert main(["skills", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    names = [s["name"] for s in payload["skills"]]
    assert SKILL_NAME in names
    entry = next(s for s in payload["skills"] if s["name"] == SKILL_NAME)
    assert entry["description"]
    assert SKILL_FILE in entry["files"]


def test_skills_handles_empty_registry(monkeypatch, capsys):
    import adapt_agent.skills as skills_module

    monkeypatch.setattr(skills_module, "available_skills", lambda: [])
    assert main(["skills"]) == 0
    assert "No skills are bundled" in capsys.readouterr().out


def test_install_handles_empty_registry(tmp_path, monkeypatch, capsys):
    import adapt_agent.skills as skills_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(skills_module, "available_skills", lambda: [])
    assert main(["install", "skill"]) == 0
    assert "No bundled skills" in capsys.readouterr().out


# -- prog name -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv0", "expected"),
    [
        ("/usr/local/bin/adapt", "adapt"),
        ("/usr/local/bin/adapt-agent", "adapt-agent"),
        ("/usr/bin/pytest", "adapt-agent"),
        ("", "adapt-agent"),
    ],
)
def test_prog_name_echoes_invoked_script(argv0, expected, monkeypatch):
    from adapt_agent.cli import _prog_name

    monkeypatch.setattr("sys.argv", [argv0])
    assert _prog_name() == expected


def test_help_uses_prog_name(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["/usr/local/bin/adapt"])
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "usage: adapt " in capsys.readouterr().out
