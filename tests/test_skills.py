"""Tests for ``adapt_agent.skills``: discovery, validation, and installation.

These also act as the quality gate on the skill the library *ships*: its
frontmatter must be valid, its internal links must resolve, and every file must
be covered by the packaging globs so it actually reaches the wheel.
"""

import fnmatch
import re
import sys
from pathlib import Path

import pytest

from adapt_agent.exceptions import SkillError
from adapt_agent.skills import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    SKILL_FILE,
    Skill,
    available_skills,
    default_destination,
    get_skill,
    install_all,
    install_skill,
    parse_frontmatter,
    validate_skill,
)

SKILL_NAME = "adapt-agent"


# -- discovery -----------------------------------------------------------------


def test_bundled_skill_is_discovered():
    names = [s.name for s in available_skills()]
    assert SKILL_NAME in names


def test_available_skills_sorted_and_typed():
    skills = available_skills()
    assert skills == sorted(skills, key=lambda s: s.name)
    assert all(isinstance(s, Skill) for s in skills)


def test_get_skill_returns_the_named_skill():
    skill = get_skill(SKILL_NAME)
    assert skill.name == SKILL_NAME
    assert skill.description


def test_get_skill_unknown_raises_with_available_names():
    with pytest.raises(SkillError, match="Bundled skills"):
        get_skill("no-such-skill")


def test_skill_file_listed_first():
    skill = get_skill(SKILL_NAME)
    assert skill.files[0] == SKILL_FILE
    assert len(skill.files) > 1  # ships supporting references too


def test_skill_read_and_unknown_file():
    skill = get_skill(SKILL_NAME)
    assert skill.read().startswith("---")
    assert "# Evals reference" in skill.read("references/evals.md")
    with pytest.raises(SkillError, match="no file"):
        skill.read("references/nope.md")


def test_skill_to_dict_is_json_friendly():
    payload = get_skill(SKILL_NAME).to_dict()
    assert payload["name"] == SKILL_NAME
    assert isinstance(payload["files"], list)
    assert isinstance(payload["description"], str)


# -- the shipped skill's own quality -------------------------------------------


def test_bundled_skill_validates():
    for skill in available_skills():
        assert validate_skill(skill) == [], f"{skill.name} is not a valid skill"


def test_bundled_skill_frontmatter_fields():
    skill = get_skill(SKILL_NAME)
    frontmatter = parse_frontmatter(skill.read())
    # Only fields in the portable Agent Skills set, so the skill stays valid
    # wherever it is published (Claude Code, claude.ai upload, Skills API).
    portable = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
    assert set(frontmatter) <= portable, f"non-portable frontmatter: {set(frontmatter) - portable}"
    assert frontmatter["name"] == SKILL_NAME  # must match the directory name
    assert len(frontmatter["description"]) <= MAX_DESCRIPTION_LENGTH
    assert len(frontmatter["name"]) <= MAX_NAME_LENGTH


def test_bundled_skill_description_carries_trigger_terms():
    """The description is what an agent matches on: it must name the domain."""
    description = get_skill(SKILL_NAME).description.lower()
    for term in ("eval", "judge", "langgraph", "pydantic ai", "google adk", "guardrail"):
        assert term in description, f"description should mention {term!r}"


def test_bundled_skill_internal_links_resolve():
    """Every relative markdown link in the skill points at a file it ships."""
    skill = get_skill(SKILL_NAME)
    link_re = re.compile(r"\[[^\]]+\]\((?!https?://)([^)#]+)\)")
    for source in skill.files:
        if not source.endswith(".md"):
            continue
        base = source.rsplit("/", 1)[0] if "/" in source else ""
        for target in link_re.findall(skill.read(source)):
            resolved = str(Path(base, target)) if base else target
            resolved = Path(resolved).as_posix()
            assert resolved in skill.files, f"{source} links to missing {target!r}"


def test_bundled_skill_body_is_not_empty_after_frontmatter():
    body = get_skill(SKILL_NAME).read().split("---", 2)[-1]
    assert len(body.strip()) > 500  # real instructions, not a stub


# -- frontmatter parsing --------------------------------------------------------


def test_parse_frontmatter_reads_block():
    parsed = parse_frontmatter("---\nname: demo\ndescription: A demo.\n---\n\n# Body\n")
    assert parsed == {"name": "demo", "description": "A demo."}


def test_parse_frontmatter_without_block_is_empty():
    assert parse_frontmatter("# Just markdown\n") == {}


def test_parse_frontmatter_requires_leading_delimiter():
    assert parse_frontmatter("intro\n---\nname: demo\n---\n") == {}


def test_parse_frontmatter_malformed_yaml_falls_back_to_line_scan():
    # Unbalanced quote: not valid YAML, but the key/value is still recoverable.
    text = '---\nname: demo\ndescription: "unterminated\n---\n\nbody\n'
    parsed = parse_frontmatter(text)
    assert parsed.get("name") == "demo"


def test_parse_frontmatter_non_mapping_falls_back():
    assert parse_frontmatter("---\n- just\n- a list\n---\n") == {}


def test_parse_frontmatter_handles_crlf():
    assert parse_frontmatter("---\r\nname: demo\r\n---\r\nbody\r\n") == {"name": "demo"}


# -- validation ------------------------------------------------------------------


def _skill(name="demo", description="Does a thing.", files=(SKILL_FILE,)):
    return Skill(name=name, description=description, files=tuple(files), metadata={})


def test_validate_flags_missing_skill_file():
    assert "missing SKILL.md" in validate_skill(_skill(files=("other.md",)))


def test_validate_flags_missing_name():
    assert "missing frontmatter name" in validate_skill(_skill(name=""))


def test_validate_flags_non_conventional_name():
    problems = validate_skill(_skill(name="Demo Skill"))
    assert any("lowercase-with-hyphens" in p for p in problems)


def test_validate_flags_long_name():
    problems = validate_skill(_skill(name="a" * (MAX_NAME_LENGTH + 1)))
    assert any("name longer than" in p for p in problems)


def test_validate_flags_missing_and_long_description():
    assert "missing frontmatter description" in validate_skill(_skill(description="   "))
    problems = validate_skill(_skill(description="x" * (MAX_DESCRIPTION_LENGTH + 1)))
    assert any("description longer than" in p for p in problems)


def test_validate_accepts_conventional_names():
    assert validate_skill(_skill(name="a")) == []
    assert validate_skill(_skill(name="adapt-agent-evals")) == []


# -- destinations ------------------------------------------------------------------


def test_default_destination_project_uses_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert default_destination("project") == tmp_path / ".claude" / "skills"


def test_default_destination_project_respects_root(tmp_path):
    assert default_destination("project", root=tmp_path) == tmp_path / ".claude" / "skills"


def test_default_destination_user_uses_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert default_destination("user") == tmp_path / ".claude" / "skills"


def test_default_destination_unknown_target_raises():
    with pytest.raises(SkillError, match="Unknown install target"):
        default_destination("cursor")


# -- installation --------------------------------------------------------------------


def test_install_copies_every_file(tmp_path):
    result = install_skill(SKILL_NAME, tmp_path)
    assert result.path == tmp_path / SKILL_NAME
    assert not result.replaced
    on_disk = sorted(
        p.relative_to(result.path).as_posix() for p in result.path.rglob("*") if p.is_file()
    )
    assert on_disk == sorted(result.skill.files)
    assert (result.path / SKILL_FILE).read_text(encoding="utf-8").startswith("---")


def test_install_defaults_to_project_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = install_skill(SKILL_NAME)
    assert result.path == tmp_path / ".claude" / "skills" / SKILL_NAME
    assert result.path.is_dir()


def test_install_user_target(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    result = install_skill(SKILL_NAME, target="user")
    assert result.path == tmp_path / ".claude" / "skills" / SKILL_NAME


def test_install_refuses_to_clobber_without_force(tmp_path):
    install_skill(SKILL_NAME, tmp_path)
    with pytest.raises(SkillError, match="already exists"):
        install_skill(SKILL_NAME, tmp_path)


def test_install_force_replaces_and_removes_stale_files(tmp_path):
    first = install_skill(SKILL_NAME, tmp_path)
    stale = first.path / "stale.md"
    stale.write_text("left over from an older version", encoding="utf-8")
    (first.path / SKILL_FILE).write_text("locally edited", encoding="utf-8")

    second = install_skill(SKILL_NAME, tmp_path, force=True)
    assert second.replaced is True
    assert not stale.exists()  # replaced, not merged
    assert (second.path / SKILL_FILE).read_text(encoding="utf-8").startswith("---")


def test_install_accepts_a_skill_object(tmp_path):
    skill = get_skill(SKILL_NAME)
    assert skill.install(tmp_path).path == tmp_path / SKILL_NAME


def test_install_unknown_skill_raises(tmp_path):
    with pytest.raises(SkillError, match="Unknown skill"):
        install_skill("nope", tmp_path)


def test_install_all_installs_every_bundled_skill(tmp_path):
    results = install_all(tmp_path)
    assert {r.skill.name for r in results} == {s.name for s in available_skills()}
    for result in results:
        assert (result.path / SKILL_FILE).is_file()


def test_install_result_to_dict(tmp_path):
    payload = install_skill(SKILL_NAME, tmp_path).to_dict()
    assert payload["name"] == SKILL_NAME
    assert payload["replaced"] is False
    assert payload["path"].endswith(SKILL_NAME)


def test_install_reports_os_errors_as_skill_error(tmp_path, monkeypatch):
    import adapt_agent.skills as skills_module

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(skills_module.shutil, "copytree", boom)
    with pytest.raises(SkillError, match="Could not install skill"):
        install_skill(SKILL_NAME, tmp_path)


# -- packaging invariant -----------------------------------------------------------


def test_every_skill_file_is_covered_by_package_data_globs():
    """A skill file not matched by a package-data glob never reaches the wheel."""
    tomllib = pytest.importorskip("tomllib", reason="tomllib requires Python 3.11+")
    root = Path(__file__).resolve().parent.parent
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():  # pragma: no cover - running against an installed copy
        pytest.skip("pyproject.toml not available (installed, not a source checkout)")

    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    patterns = config["tool"]["setuptools"]["package-data"]["adapt_agent"]

    skills_dir = root / "adapt_agent" / "skills"
    for path in skills_dir.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root / "adapt_agent").as_posix()
        if relative.endswith(".py"):
            continue  # modules ship as code, not package data
        assert any(fnmatch.fnmatch(relative, pattern) for pattern in patterns), (
            f"{relative} matches no package-data pattern in pyproject.toml; "
            f"it would be missing from the built wheel"
        )


def test_skills_module_imports_no_heavy_dependencies():
    """Importing the skills registry must not drag in frameworks or SDKs."""
    before = set(sys.modules)
    import adapt_agent.skills  # noqa: F401

    newly_imported = set(sys.modules) - before
    forbidden = ("langgraph", "pydantic_ai", "crewai", "anthropic", "openai", "google")
    assert not [m for m in newly_imported if m.split(".")[0] in forbidden]
