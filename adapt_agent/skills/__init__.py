"""Agent skills bundled with ADAPT-Agent.

An *agent skill* is a folder containing a ``SKILL.md`` file (YAML frontmatter +
markdown instructions) plus optional supporting files. Coding agents such as
Claude Code discover skills from well-known directories -- ``.claude/skills/``
in a project, ``~/.claude/skills/`` for a user -- and load one when its
``description`` matches the task at hand.

ADAPT-Agent ships its own skill *inside the wheel*, so installing the library
also delivers the instructions an agent needs to use it:

.. code-block:: bash

    pip install adapt-agent      # or: uv add adapt-agent
    adapt install skill          # copies the skill into ./.claude/skills/

This module is the programmatic half of that flow:

.. code-block:: python

    from adapt_agent.skills import available_skills, install_skill

    [s.name for s in available_skills()]          # -> ["adapt-agent"]
    result = install_skill("adapt-agent", target="project")
    result.path                                   # -> .../.claude/skills/adapt-agent

Skill files are read through :mod:`importlib.resources`, so everything works
from a wheel, an editable install, or a zipped distribution. Nothing here
imports an agent framework or an LLM SDK.
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Annotation-only: this module has `from __future__ import annotations`, so
    # nothing here is evaluated at runtime. The import is version-gated because
    # `importlib.abc.Traversable` was deprecated in 3.12 and REMOVED in 3.14,
    # while its replacement `importlib.resources.abc` only exists from 3.11.
    if sys.version_info >= (3, 11):
        from importlib.resources.abc import Traversable
    else:
        from importlib.abc import Traversable

from adapt_agent.exceptions import SkillError

#: The entrypoint file every skill directory must contain.
SKILL_FILE = "SKILL.md"

#: Where each install target places skill directories, relative to its root.
#: ``project`` is resolved against the working directory (or an explicit root),
#: ``user`` against the user's home directory. Both are the directories Claude
#: Code discovers skills from; other tools that read ``SKILL.md`` folders can be
#: targeted with an explicit destination instead.
INSTALL_TARGETS: dict[str, tuple[str, ...]] = {
    "project": (".claude", "skills"),
    "user": (".claude", "skills"),
}

#: Conservative portability limits. The Agent Skills format tolerates more, but
#: staying inside these keeps a skill valid everywhere it might be published.
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024

_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\s*(?:\r?\n|\Z)", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    """A skill bundled with the installed ``adapt_agent`` package.

    Args:
        name: The skill identifier -- always its **directory** name. This is
            what locates the packaged files and names the installed directory,
            so it is authoritative even if the frontmatter disagrees.
        description: The frontmatter ``description`` an agent matches against.
        files: Relative paths of every file in the skill, ``SKILL.md`` first.
        metadata: Any remaining frontmatter fields (``license``, ``metadata``…).
        declared_name: The ``name`` the frontmatter claims. Conventionally equal
            to :attr:`name`; :func:`validate_skill` reports a mismatch.
    """

    name: str
    description: str
    files: tuple[str, ...]
    metadata: dict[str, Any]
    declared_name: str = ""

    @property
    def source(self) -> Traversable:
        """The packaged skill directory (a :mod:`importlib.resources` handle)."""
        return _skills_root() / self.name

    def read(self, relative: str = SKILL_FILE) -> str:
        """Return the text of a file inside the skill.

        Args:
            relative: Path relative to the skill directory, e.g.
                ``"references/evals.md"``. Defaults to ``SKILL.md``.
        """
        if relative not in self.files:
            raise SkillError(
                f"Skill {self.name!r} has no file {relative!r}. Available: {list(self.files)}"
            )
        node = self.source
        for part in relative.split("/"):
            node = node / part
        return node.read_text(encoding="utf-8")

    def install(
        self,
        destination: str | Path | None = None,
        *,
        target: str = "project",
        root: str | Path | None = None,
        force: bool = False,
    ) -> InstallResult:
        """Copy this skill into a skills directory. See :func:`install_skill`."""
        return install_skill(self, destination, target=target, root=root, force=force)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly summary (used by the CLI's ``--json`` output)."""
        return {
            "name": self.name,
            "description": self.description,
            "files": list(self.files),
            **({"metadata": self.metadata} if self.metadata else {}),
        }


@dataclass(frozen=True)
class InstallResult:
    """The outcome of installing one skill."""

    skill: Skill
    path: Path
    files: tuple[str, ...]
    #: ``True`` when an existing installation at :attr:`path` was replaced.
    replaced: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.skill.name,
            "path": str(self.path),
            "files": list(self.files),
            "replaced": self.replaced,
        }


# -- discovery -----------------------------------------------------------------


def _skills_root() -> Traversable:
    """The packaged ``adapt_agent/skills`` directory."""
    return resources.files(__name__)


def _iter_skill_dirs() -> Iterator[Traversable]:
    """Yield packaged directories that contain a ``SKILL.md``."""
    for entry in _skills_root().iterdir():
        try:
            if not entry.is_dir():
                continue
            if not (entry / SKILL_FILE).is_file():
                continue
        except (OSError, PermissionError):  # pragma: no cover - unreadable entry
            continue
        yield entry


def available_skills() -> list[Skill]:
    """Return every skill bundled with the installed package, sorted by name."""
    skills = [_load_skill(entry) for entry in _iter_skill_dirs()]
    return sorted(skills, key=lambda s: s.name)


def get_skill(name: str) -> Skill:
    """Return one bundled skill by name.

    Raises:
        SkillError: If no bundled skill has that name.
    """
    for skill in available_skills():
        if skill.name == name:
            return skill
    known = [s.name for s in available_skills()]
    raise SkillError(f"Unknown skill {name!r}. Bundled skills: {known}")


def _load_skill(directory: Traversable) -> Skill:
    """Build a :class:`Skill` from a packaged skill directory."""
    text = (directory / SKILL_FILE).read_text(encoding="utf-8")
    frontmatter = parse_frontmatter(text)
    # The directory name is authoritative: it is what agents use as the skill's
    # identity, and what :attr:`Skill.source` resolves against. Trusting the
    # frontmatter here would let a mismatched `name:` point reads and installs
    # at a package directory that does not exist.
    declared_name = str(frontmatter.pop("name", "") or "")
    description = str(frontmatter.pop("description", "") or "")
    return Skill(
        name=directory.name,
        description=description,
        files=tuple(_relative_files(directory)),
        metadata=frontmatter,
        declared_name=declared_name,
    )


def _relative_files(directory: Traversable, prefix: str = "") -> list[str]:
    """List files under a packaged directory, ``SKILL.md`` first then sorted."""
    found: list[str] = []
    for entry in directory.iterdir():
        relative = f"{prefix}{entry.name}"
        if entry.is_dir():
            found.extend(_relative_files(entry, prefix=f"{relative}/"))
        else:
            found.append(relative)
    found.sort(key=lambda path: (path != SKILL_FILE, path))
    return found


# -- frontmatter ---------------------------------------------------------------


def parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse the leading ``---`` YAML frontmatter block of a SKILL.md.

    Returns an empty dict when the document has no frontmatter. PyYAML (a core
    dependency) does the parsing; if the block is not valid YAML a minimal
    ``key: value`` line scan is used instead, so a slightly malformed skill
    still reports its name and description rather than failing outright.
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return {}
    block = match.group(1)
    try:
        import yaml

        parsed = yaml.safe_load(block)
    except Exception:
        return _scan_key_values(block)
    if not isinstance(parsed, dict):
        return _scan_key_values(block)
    return parsed


def _scan_key_values(block: str) -> dict[str, Any]:
    """Fallback frontmatter parse: top-level ``key: value`` lines only."""
    values: dict[str, Any] = {}
    for line in block.splitlines():
        if not line or line[0].isspace() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        values[key.strip()] = value.strip().strip("\"'")
    return values


def validate_skill(skill: Skill) -> list[str]:
    """Return a list of portability problems with ``skill`` (empty when valid).

    Checks the properties that make a skill discoverable and publishable: a
    ``SKILL.md`` entrypoint; a directory name in the conventional
    lowercase-with-hyphens form and within :data:`MAX_NAME_LENGTH`; a
    frontmatter ``name`` that is present and agrees with that directory; and a
    non-empty ``description`` within :data:`MAX_DESCRIPTION_LENGTH`.
    """
    problems: list[str] = []
    if SKILL_FILE not in skill.files:
        problems.append(f"missing {SKILL_FILE}")
    if not skill.name:
        problems.append("missing skill directory name")
    elif not _NAME_RE.match(skill.name):
        problems.append(f"name {skill.name!r} is not lowercase-with-hyphens")
    elif len(skill.name) > MAX_NAME_LENGTH:
        problems.append(f"name longer than {MAX_NAME_LENGTH} characters")
    if not skill.declared_name:
        problems.append("missing frontmatter name")
    elif skill.declared_name != skill.name:
        problems.append(
            f"frontmatter name {skill.declared_name!r} does not match " f"directory {skill.name!r}"
        )
    if not skill.description.strip():
        problems.append("missing frontmatter description")
    elif len(skill.description) > MAX_DESCRIPTION_LENGTH:
        problems.append(f"description longer than {MAX_DESCRIPTION_LENGTH} characters")
    return problems


# -- installation --------------------------------------------------------------


def default_destination(target: str = "project", root: str | Path | None = None) -> Path:
    """Return the skills directory for an install target.

    Args:
        target: ``"project"`` (``<root>/.claude/skills``, root defaulting to the
            working directory) or ``"user"`` (``~/.claude/skills``).
        root: Base directory for the ``project`` target. Ignored for ``user``.

    Raises:
        SkillError: If ``target`` is not a known install target.
    """
    parts = INSTALL_TARGETS.get(target)
    if parts is None:
        raise SkillError(
            f"Unknown install target {target!r}. Choose from {sorted(INSTALL_TARGETS)}"
        )
    if target == "user":
        base = Path.home()
    else:
        base = Path(root) if root is not None else Path.cwd()
    return base.joinpath(*parts)


def install_skill(
    skill: str | Skill,
    destination: str | Path | None = None,
    *,
    target: str = "project",
    root: str | Path | None = None,
    force: bool = False,
) -> InstallResult:
    """Copy a bundled skill into a skills directory.

    The skill lands in ``<destination>/<skill-name>/``, the layout agents
    discover (its ``SKILL.md`` plus any supporting files).

    Args:
        skill: A skill name or a :class:`Skill` from :func:`available_skills`.
        destination: The skills directory to install into. Defaults to
            :func:`default_destination` for ``target``.
        target: ``"project"`` (default) or ``"user"`` when ``destination`` is
            not given.
        root: Base directory for the ``project`` target (defaults to the
            working directory).
        force: Replace an existing installation. Without it, an existing
            directory raises rather than silently overwriting local edits.

    Returns:
        An :class:`InstallResult` with the installed path and file list.

    Raises:
        SkillError: If the skill is unknown, the destination already contains
            the skill and ``force`` is not set, or the copy fails.
    """
    resolved = skill if isinstance(skill, Skill) else get_skill(skill)
    skills_dir = (
        Path(destination) if destination is not None else default_destination(target, root=root)
    )
    target_dir = skills_dir / resolved.name

    replaced = target_dir.exists()
    if replaced and not force:
        raise SkillError(f"{target_dir} already exists. Pass --force (force=True) to replace it.")

    # Stage the copy in a sibling directory, then swap it into place. Two
    # reasons: a failed copy must never leave the user without the skill they
    # already had, and staging on the same filesystem makes the swap a rename.
    #
    # Destination and staging setup are inside the try as well: a read-only
    # filesystem or a full disk fails here, and this function's contract is to
    # raise SkillError rather than leak a raw OSError at callers.
    staging: Path | None = None
    try:
        skills_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{resolved.name}.", dir=skills_dir))
        staged = staging / resolved.name
        previous = staging / "previous"
        _materialize(resolved, staged)
        if replaced:
            # Move the old install aside rather than deleting it, so a failure
            # during the swap can put it back.
            target_dir.rename(previous)
        try:
            staged.rename(target_dir)
        except OSError:
            if previous.exists() and not target_dir.exists():
                previous.rename(target_dir)
            raise
    except OSError as exc:
        raise SkillError(
            f"Could not install skill {resolved.name!r} to {target_dir}: {exc}"
        ) from exc
    finally:
        # Removes the staged copy on failure, and the superseded install on
        # success -- both live under ``staging``.
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)

    return InstallResult(
        skill=resolved,
        path=target_dir,
        files=resolved.files,
        replaced=replaced,
    )


def _materialize(skill: Skill, destination: Path) -> None:
    """Copy a packaged skill's files to ``destination``.

    Reads through the :mod:`importlib.resources` traversable API one file at a
    time rather than ``as_file()`` on the directory: ``as_file()`` only gained
    directory support in 3.12, so on 3.10/3.11 a zip-imported package raised
    ``IsADirectoryError`` and installing was impossible. Walking the known file
    list works identically from a wheel, an editable checkout, and a zip.
    """
    for relative in skill.files:
        parts = relative.split("/")
        if any(part in ("", ".", "..") for part in parts):  # defensive
            raise SkillError(f"Refusing to install unsafe path {relative!r}")
        node = skill.source
        for part in parts:
            node = node / part
        out = destination.joinpath(*parts)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(node.read_bytes())


def install_all(
    destination: str | Path | None = None,
    *,
    target: str = "project",
    root: str | Path | None = None,
    force: bool = False,
) -> list[InstallResult]:
    """Install every bundled skill. See :func:`install_skill` for the arguments."""
    return [
        install_skill(skill, destination, target=target, root=root, force=force)
        for skill in available_skills()
    ]


__all__ = [
    "INSTALL_TARGETS",
    "MAX_DESCRIPTION_LENGTH",
    "MAX_NAME_LENGTH",
    "SKILL_FILE",
    "InstallResult",
    "Skill",
    "available_skills",
    "default_destination",
    "get_skill",
    "install_all",
    "install_skill",
    "parse_frontmatter",
    "validate_skill",
]
