"""Cheap guards on packaging metadata and the docs that describe it.

Each of these drifted silently once: extras floors years stale, an extra that
resolved to a 200-package meta-package, a SECURITY.md that still listed 0.2.x
as supported at 0.5.1, a black target older than the Python floor.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

import adapt_agent

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="tomllib needs Python 3.11+ (dev extras ship no tomli)"
)


@pytest.fixture(scope="module")
def pyproject() -> dict:
    import tomllib

    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _extras(pyproject: dict) -> dict[str, list[str]]:
    return pyproject["project"]["optional-dependencies"]


def _req_name(req: str) -> str:
    return re.split(r"[\s<>=!~\[;]", req.strip(), maxsplit=1)[0].lower().replace("_", "-")


def test_extras_named_in_docs_exist(pyproject) -> None:
    extras = set(_extras(pyproject))
    named: set[str] = set()
    for doc in [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *(ROOT / "docs").glob("*.md")]:
        for group in re.findall(r"adapt-agent\[([^\]]+)\]", doc.read_text(encoding="utf-8")):
            named |= {part.strip() for part in group.split(",")}
    named -= {"extra", "extras"}
    missing = named - extras
    assert not missing, f"docs name extras that pyproject.toml does not define: {sorted(missing)}"


def test_no_extra_depends_on_the_agent_framework_meta_package(pyproject) -> None:
    """`agent-framework` pulls ~200 packages and a pre-release pin uv will not resolve."""
    for extra, reqs in _extras(pyproject).items():
        for req in reqs:
            assert _req_name(req) != "agent-framework", (
                f"extra {extra!r} requires the agent-framework meta-package; "
                "use agent-framework-core"
            )


def test_every_framework_extra_has_a_real_floor(pyproject) -> None:
    """A bare name or a `>=0.0.x`-style floor means the declared minimum is a guess."""
    for extra in (
        "langgraph",
        "microsoft-agent-framework",
        "google-adk",
        "pydantic-ai",
        "crewai",
        "openai-agents",
        "claude-agent",
    ):
        for req in _extras(pyproject)[extra]:
            assert ">=" in req, f"{extra}: {req!r} has no lower bound"
            floor = req.split(">=", 1)[1].split(",", 1)[0].strip()
            assert tuple(int(p) for p in floor.split(".")[:2]) >= (
                0,
                2,
            ), f"{extra}: floor {floor!r} predates the current release line"


def test_all_extra_only_references_defined_extras(pyproject) -> None:
    extras = _extras(pyproject)
    (spec,) = extras["all"]
    inner = re.search(r"\[([^\]]+)\]", spec)
    assert inner is not None
    for name in inner.group(1).split(","):
        assert name.strip() in extras, f"[all] names undefined extra {name!r}"


def test_black_and_ruff_targets_agree_with_requires_python(pyproject) -> None:
    requires = pyproject["project"]["requires-python"]
    floor = re.search(r">=\s*3\.(\d+)", requires)
    assert floor is not None, requires
    minor = int(floor.group(1))

    black_targets = [
        int(t.removeprefix("py3")) for t in pyproject["tool"]["black"]["target-version"]
    ]
    assert min(black_targets) == minor, "black must target exactly the supported floor upward"

    assert pyproject["tool"]["ruff"]["target-version"] == f"py3{minor}"
    assert pyproject["tool"]["mypy"]["python_version"] == f"3.{minor}"

    classifiers = [
        int(m.group(1))
        for c in pyproject["project"]["classifiers"]
        if (m := re.fullmatch(r"Programming Language :: Python :: 3\.(\d+)", c))
    ]
    assert max(black_targets) == max(classifiers), "black targets lag the claimed Pythons"


def test_pre_commit_does_not_override_black_target() -> None:
    text = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "--target-version" not in text, "pyproject.toml is the single source of targets"


def test_security_md_supports_the_current_minor_line() -> None:
    major, minor = adapt_agent.__version__.split(".")[:2]
    text = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    supported = re.findall(r"^\|\s*(\d+\.\d+)\.x\s*\|\s*:white_check_mark:", text, re.MULTILINE)
    assert supported == [
        f"{major}.{minor}"
    ], f"SECURITY.md lists {supported} as supported; the current line is {major}.{minor}.x"
