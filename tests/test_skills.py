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


def _skill(name="demo", description="Does a thing.", files=(SKILL_FILE,), declared_name=None):
    return Skill(
        name=name,
        description=description,
        files=tuple(files),
        metadata={},
        declared_name=name if declared_name is None else declared_name,
    )


def test_validate_flags_missing_skill_file():
    assert "missing SKILL.md" in validate_skill(_skill(files=("other.md",)))


def test_validate_flags_missing_directory_name():
    assert "missing skill directory name" in validate_skill(_skill(name=""))


def test_validate_flags_missing_frontmatter_name():
    assert "missing frontmatter name" in validate_skill(_skill(declared_name=""))


def test_validate_flags_frontmatter_directory_mismatch():
    problems = validate_skill(_skill(name="demo", declared_name="something-else"))
    assert any("does not match directory" in p for p in problems)


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


def test_install_restores_the_previous_skill_when_the_swap_fails(tmp_path, monkeypatch):
    """If the final rename fails, the moved-aside install is put back."""
    installed = install_skill(SKILL_NAME, tmp_path)
    (installed.path / SKILL_FILE).write_text("ORIGINAL", encoding="utf-8")

    real_rename = Path.rename
    calls = {"n": 0}

    def flaky_rename(self, target):
        # Let the "move the old install aside" rename through, fail the swap.
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("cross-device link")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky_rename)
    with pytest.raises(SkillError, match="Could not install skill"):
        install_skill(SKILL_NAME, tmp_path, force=True)

    monkeypatch.undo()
    assert (installed.path / SKILL_FILE).read_text(encoding="utf-8") == "ORIGINAL"


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


# -- the skill's code examples must call APIs that exist -------------------------
#
# The skill exists so an agent can copy working code out of it. A recipe naming
# a method that does not exist is therefore a shipped bug, and one that no
# ordinary unit test would notice. This walks every ``obj.method(`` in the
# skill's Python blocks and checks it against the real class.

#: Example variable name -> the first-party class it stands for.
_EXAMPLE_OBJECTS = {
    "firewall": "adapt_agent.security:Firewall",
    "policy": "adapt_agent.core:PolicyEnforcer",
    "defense": "adapt_agent.adversarial:AdversarialDefense",
    "trust": "adapt_agent.core:TrustManager",
    "taint": "adapt_agent.security:TaintTracker",
    "observer": "adapt_agent.observability:AgentObserver",
    "harness": "adapt_agent.evaluation:EvaluationHarness",
    "report": "adapt_agent.evaluation:EvaluationReport",
    "judge": "adapt_agent.evaluation:LLMJudge",
    "target": "adapt_agent.optimization:OptimizableAgent",
    "guarded": "adapt_agent.adapters._governed:_GovernedAgent",
    "result": "adapt_agent.optimization:OptimizationResult",
    "skill": "adapt_agent.skills:Skill",
    "exc": "adapt_agent.exceptions:SecurityBlockedError",
    "source": "adapt_agent.security:TaintSource",
}

#: Names that legitimately are not first-party objects (user code, stdlib,
#: placeholders). Listed explicitly so a new example variable forces a decision
#: rather than silently escaping the check.
_NON_FIRST_PARTY = {
    "cfg",  # a user config dict
    "data",  # a GoldenDataset the user built
    "orchestrator",  # the user's own agent code
    "content",  # the str parameter of a custom firewall filter
    "researcher",  # the user's own sub-agent
    "v",  # a lambda parameter
    "json",  # stdlib
    "config",  # a dict loaded from a JSON config file
    "fw_config",  # the "firewall" sub-dict of that config
}


def _resolve(spec):
    import importlib

    module_name, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def _documented_attributes():
    """Map example variable name -> every attribute/method used on it.

    Covers plain attribute access as well as calls: an example reading a field
    that does not exist (``result.trials``) fails just as hard as one calling a
    method that does not exist.
    """
    used: dict[str, set[str]] = {}
    for skill in available_skills():
        for relative in skill.files:
            if not relative.endswith(".md"):
                continue
            for block in re.findall(r"```python\n(.*?)```", skill.read(relative), re.S):
                for obj, attr in re.findall(
                    r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)", _strip_noise(block)
                ):
                    used.setdefault(obj, set()).add(attr)
    return used


def _strip_noise(block: str) -> str:
    """Drop import lines and string literals before scanning for attributes.

    `from adapt_agent.security import ...` and `"golden.jsonl"` both contain a
    dotted name that is not attribute access on an object.
    """
    lines = [line for line in block.splitlines() if not re.match(r"\s*(from|import)\s", line)]
    text = "\n".join(lines)
    return re.sub(
        r"(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\'|\"[^\"\n]*\"|\'[^\'\n]*\')", '""', text, flags=re.S
    )


def _public_names(cls):
    """Every attribute name valid on ``cls``.

    Three sources, because ``hasattr`` alone sees only the first: class
    attributes and methods; dataclass fields without defaults (annotations
    only, never class attributes); and plain instance attributes assigned as
    ``self.x = ...`` in ``__init__``.
    """
    import inspect

    names = set(dir(cls))
    for klass in getattr(cls, "__mro__", [cls]):
        names.update(getattr(klass, "__annotations__", {}))
        try:
            source = inspect.getsource(klass)
        except (OSError, TypeError):  # pragma: no cover - builtins have no source
            continue
        names.update(re.findall(r"\bself\.([a-z_][a-z0-9_]*)\s*(?::[^=]+)?=", source))
    return names


def test_skill_examples_only_use_attributes_that_exist():
    missing = []
    for obj, attributes in _documented_attributes().items():
        if obj in _NON_FIRST_PARTY:
            continue
        spec = _EXAMPLE_OBJECTS.get(obj)
        if spec is None:
            continue  # covered by the companion test below
        cls = _resolve(spec)
        valid = _public_names(cls)
        for attr in sorted(attributes):
            if attr not in valid:
                missing.append(f"{cls.__name__}.{attr} (documented as {obj}.{attr})")
    assert not missing, "skill documents attributes that do not exist: " + ", ".join(missing)


def test_every_example_object_is_accounted_for():
    """A new example variable must be mapped to a class or explicitly excused."""
    unknown = sorted(
        obj
        for obj in _documented_attributes()
        if obj not in _EXAMPLE_OBJECTS and obj not in _NON_FIRST_PARTY
    )
    assert not unknown, (
        f"unmapped example objects in the skill: {unknown}. Add them to "
        "_EXAMPLE_OBJECTS so their methods get verified, or to _NON_FIRST_PARTY."
    )


# -- installing from a zipped distribution / crash safety -------------------------


def test_install_works_from_a_zip_imported_package(tmp_path):
    """The module claims zip support; prove it on the running interpreter.

    ``importlib.resources.as_file()`` only handles directories from 3.12, so
    materialising the skill directory that way broke zip-imported installs on
    3.10/3.11. Files are copied through the traversable API instead.
    """
    import subprocess
    import zipfile

    root = Path(__file__).resolve().parent.parent
    archive = tmp_path / "pkg.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in (root / "adapt_agent").rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                bundle.write(path, path.relative_to(root).as_posix())

    script = f"""
import sys
sys.path.insert(0, {str(archive)!r})
import adapt_agent.skills as skills
assert type(skills.__loader__).__name__ == "zipimporter", "not imported from the zip"
result = skills.install_skill("adapt-agent", {str(tmp_path / "out")!r})
print(sorted(p.relative_to(result.path).as_posix()
             for p in result.path.rglob("*") if p.is_file()))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=tmp_path
    )
    assert completed.returncode == 0, completed.stderr
    assert "SKILL.md" in completed.stdout
    assert "references/evals.md" in completed.stdout


def test_failed_force_install_preserves_the_existing_skill(tmp_path, monkeypatch):
    """A forced upgrade that dies mid-copy must not destroy what was there."""
    import adapt_agent.skills as skills_module

    installed = install_skill(SKILL_NAME, tmp_path)
    (installed.path / SKILL_FILE).write_text("ORIGINAL", encoding="utf-8")

    def exploding(skill, destination):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / SKILL_FILE).write_bytes(b"partial")
        raise OSError("No space left on device")

    monkeypatch.setattr(skills_module, "_materialize", exploding)
    with pytest.raises(SkillError, match="Could not install skill"):
        install_skill(SKILL_NAME, tmp_path, force=True)

    # The previous installation is still there, and intact.
    assert (installed.path / SKILL_FILE).read_text(encoding="utf-8") == "ORIGINAL"
    # ... and no staging directory was left behind.
    assert [p.name for p in tmp_path.iterdir()] == [SKILL_NAME]


def test_failed_first_install_leaves_no_staging_directory(tmp_path, monkeypatch):
    import adapt_agent.skills as skills_module

    def exploding(skill, destination):
        raise OSError("boom")

    monkeypatch.setattr(skills_module, "_materialize", exploding)
    with pytest.raises(SkillError):
        install_skill(SKILL_NAME, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_materialize_refuses_unsafe_relative_paths(tmp_path):
    import adapt_agent.skills as skills_module

    hostile = Skill(
        name="x",
        description="d",
        files=("../escape.md",),
        metadata={},
        declared_name="x",
    )
    with pytest.raises(SkillError, match="unsafe path"):
        skills_module._materialize(hostile, tmp_path / "dest")


# -- documented configs and constructor arguments must be real --------------------
#
# The attribute audit above catches `obj.does_not_exist`. These catch the other
# way a recipe rots: the attribute exists but the *arguments* or *schema* are
# wrong, so an agent copying the snippet gets an exception anyway.


def _code_blocks(text: str, language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", text, re.DOTALL)


def test_documented_training_configs_parse():
    """Every training-config YAML in the skill must satisfy the real schema."""
    import yaml

    from adapt_agent.optimization.config import parse_training_config

    skill = get_skill(SKILL_NAME)
    checked = 0
    for source in skill.files:
        if not source.endswith(".md"):
            continue
        for block in _code_blocks(skill.read(source), "yaml"):
            raw = yaml.safe_load(block)
            if not isinstance(raw, dict) or "target" not in raw:
                continue  # not a training config
            parse_training_config(raw)  # raises TrainingConfigError if wrong
            checked += 1
    assert checked, "expected at least one training-config example in the skill"


def test_documented_llmjudge_providers_are_registered():
    """`LLMJudge("x")` resolves through the provider registry, not judge aliases.

    Friendly aliases such as "claude" are only understood by ``get_judge()``, so
    a documented ``LLMJudge("claude")`` raises KeyError for anyone copying it.
    """
    from adapt_agent.optimization.providers import available_providers

    registered = set(available_providers())
    skill = get_skill(SKILL_NAME)
    for source in skill.files:
        if not source.endswith(".md"):
            continue
        code = "\n".join(_code_blocks(skill.read(source), "python"))
        for name in re.findall(r'LLMJudge\(\s*"([^"]+)"', code):
            assert name in registered, (
                f"{source} documents LLMJudge({name!r}), which is not a registered "
                f"provider ({sorted(registered)}). Use get_judge({name!r}) instead."
            )


def test_documented_get_judge_providers_resolve():
    """`get_judge("x")` names in the skill must resolve to a real judge."""
    from adapt_agent.optimization.judges import JUDGE_REGISTRY

    skill = get_skill(SKILL_NAME)
    for source in skill.files:
        if not source.endswith(".md"):
            continue
        code = "\n".join(_code_blocks(skill.read(source), "python"))
        for name in re.findall(r'get_judge\(\s*"([^"]+)"', code):
            assert name in JUDGE_REGISTRY, f"{source}: unknown judge provider {name!r}"
