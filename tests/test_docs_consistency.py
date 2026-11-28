"""Cheap, offline guards that keep the documentation true to the code.

Every check here caught real drift once: snippets importing names that had been
renamed, guides linking to example files that no longer existed, install lines
naming extras ``pyproject.toml`` never defined, and model ids or framework
classes retired upstream still shown as the current way. None of them needs a
framework or a network: a snippet is parsed, not run, and an ``adapt_agent``
name is resolved by importing the (import-light) module and looking it up.
"""

from __future__ import annotations

import ast
import importlib
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Every markdown page a user reads: the docs site, the README, and the
#: per-framework example READMEs (plus the examples index).
DOC_FILES: list[Path] = sorted(
    [
        *(ROOT / "docs").rglob("*.md"),
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "SECURITY.md",
        *(ROOT / "examples").glob("*/README.md"),
        ROOT / "examples" / "README.md",
    ]
)

#: Known failures in files another change is rewriting, keyed by
#: ``(check, relative path)``. Each is ``xfail(strict=False)`` so the suite
#: stays green until that rewrite lands -- then delete the entry.
KNOWN_FAILURES: dict[tuple[str, str], str] = {}

_FENCE = re.compile(r"^([ \t]*)(`{3,})[ \t]*([\w+-]*)[^\n]*\n(.*?)^\1\2[ \t]*$", re.M | re.S)
_LINK = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _param(check: str, path: Path) -> object:
    reason = KNOWN_FAILURES.get((check, _rel(path)))
    marks = [pytest.mark.xfail(reason=reason, strict=False)] if reason else []
    return pytest.param(path, id=_rel(path), marks=marks)


def _params(check: str) -> list[object]:
    return [_param(check, p) for p in DOC_FILES]


def _python_blocks(text: str) -> list[tuple[int, str]]:
    blocks = []
    for m in _FENCE.finditer(text):
        indent, lang, body = m.group(1), m.group(3).lower(), m.group(4)
        if lang not in ("python", "py"):
            continue
        if indent:
            body = "\n".join(
                ln[len(indent) :] if ln.startswith(indent) else ln for ln in body.splitlines()
            )
        blocks.append((text[: m.start()].count("\n") + 1, body))
    return blocks


def _prose(text: str) -> str:
    """The text with fenced code blocks blanked (line numbers preserved)."""
    return _FENCE.sub(lambda m: "\n" * m.group(0).count("\n"), text)


@pytest.mark.parametrize("path", _params("compile"))
def test_python_snippets_compile(path: Path) -> None:
    problems = []
    for line, body in _python_blocks(path.read_text(encoding="utf-8")):
        try:
            ast.parse(body)
        except SyntaxError as exc:
            problems.append(f"line {line}: {exc.msg} (snippet line {exc.lineno})")
    assert not problems, f"{_rel(path)}: python snippets that do not parse: {problems}"


def _resolve(module: str, name: str | None) -> str | None:
    """None if ``module`` (and ``name`` on it) resolve, else what is missing."""
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        missing = getattr(exc, "name", None) or ""
        if not missing.startswith("adapt_agent"):
            return None  # an optional SDK the module needs at import time
        return f"module {module}"
    if name is None or name == "*" or hasattr(mod, name):
        return None
    try:
        importlib.import_module(f"{module}.{name}")
    except ImportError:
        return f"{module}.{name}"
    return None


@pytest.mark.parametrize("path", _params("imports"))
def test_adapt_agent_imports_in_snippets_resolve(path: Path) -> None:
    missing = []
    for line, body in _python_blocks(path.read_text(encoding="utf-8")):
        try:
            tree = ast.parse(body)
        except SyntaxError:
            continue  # reported by test_python_snippets_compile
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("adapt_agent"):
                for alias in node.names:
                    found = _resolve(node.module or "", alias.name)
                    if found:
                        missing.append(f"line {line}: {found}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("adapt_agent"):
                        found = _resolve(alias.name, None)
                        if found:
                            missing.append(f"line {line}: {found}")
    assert not missing, f"{_rel(path)}: snippets import names that do not exist: {missing}"


def _slug(heading: str) -> str:
    """The anchor MkDocs' ``toc`` extension (and GitHub) give a heading."""
    text = re.sub(r"`|\*|_(?=\W)|(?<=\W)_", "", heading.strip()).lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return re.sub(r"\s+", "-", text.strip())


def _anchors(path: Path) -> set[str]:
    prose = _prose(path.read_text(encoding="utf-8"))
    return {_slug(h) for h in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", prose, re.M)}


@pytest.mark.parametrize("path", _params("links"))
def test_relative_links_resolve(path: Path) -> None:
    broken = []
    prose = _prose(path.read_text(encoding="utf-8"))
    for target in _LINK.findall(prose):
        if re.match(r"[a-z][a-z0-9+.-]*:", target):  # http:, https:, mailto:, ...
            continue
        file_part, _, anchor = target.partition("#")
        dest = (path.parent / file_part).resolve() if file_part else path
        if not dest.exists():
            broken.append(target)
            continue
        if anchor and dest.suffix == ".md" and anchor not in _anchors(dest):
            broken.append(f"{target} (no such heading)")
    assert not broken, f"{_rel(path)}: broken relative links: {broken}"


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib needs Python 3.11+")
@pytest.mark.parametrize("path", _params("extras"))
def test_install_extras_exist(path: Path) -> None:
    import tomllib

    extras = set(
        tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
            "optional-dependencies"
        ]
    )
    named: set[str] = set()
    for group in re.findall(r"adapt-agent\[([^\]]+)\]", path.read_text(encoding="utf-8")):
        named |= {part.strip() for part in group.split(",")}
    unknown = named - extras - {"extra", "extras", "<extra>", "..."}
    assert not unknown, f"{_rel(path)} names extras pyproject.toml does not define: {unknown}"


#: Names and ids that are retired upstream or here, shown as if current. Kept
#: short and precise: each pattern matches only the retired *usage*, not a
#: sentence that explains the rename.
RETIRED: dict[str, str] = {
    r"\bChatAgent\(": "MAF's ChatAgent is agent_framework.Agent since 1.0",
    r"\b(?:chat_)?client\.create_agent\((?!\.\.\.\))": "MAF clients expose as_agent()",
    r"\bgemini-2\.0-flash\b|\bgemini-1\.5-": "retired Gemini model id",
    r"\bclaude-opus-4-8\b|\bclaude-3-[0-9a-z-]+": "not a current Claude model id",
    r"\bvalidation_tolerance=": "validation_tolerance is refused (FutureWarning)",
    r"\|\s*plan\s*\|\s*bypassPermissions|Candidates:[^\n]*bypassPermissions": (
        "permission_mode is read-only to the optimizer; bypassPermissions is not a candidate"
    ),
    r"pip install ['\"]?agent-framework(?![-\w])": "use agent-framework-core, not the meta-package",
}


@pytest.mark.parametrize("path", _params("retired"))
def test_no_retired_names(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    hits = []
    for pattern, why in RETIRED.items():
        for m in re.finditer(pattern, text):
            line = text[: m.start()].count("\n") + 1
            hits.append(f"line {line}: {m.group(0)!r} -- {why}")
    assert not hits, f"{_rel(path)}: {hits}"


def test_every_doc_page_is_in_the_mkdocs_nav_once() -> None:
    nav = re.findall(r":\s*([\w/.-]+\.md)\s*$", (ROOT / "mkdocs.yml").read_text(), re.M)
    pages = sorted(_rel(p).removeprefix("docs/") for p in (ROOT / "docs").rglob("*.md"))
    assert sorted(nav) == pages, "mkdocs.yml nav must list every docs/ page exactly once"
