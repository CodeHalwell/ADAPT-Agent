"""Tests for ``scripts/heldout_check.py`` on two tiny throwaway repositories."""

import importlib.util
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    name = "heldout_check"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "heldout_check.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _repo(root: Path, *, limit: int, test_limit: int, mutants: list[str] | None = None) -> Path:
    """A one-function project: ``allowed(n)`` is ``n <= limit``, with one test."""
    (root / "tests").mkdir(parents=True)
    (root / "guard.py").write_text(f"def allowed(n):\n    return n <= {limit}\n")
    (root / "tests" / "__init__.py").write_text("")
    (root / "tests" / "test_guard.py").write_text(textwrap.dedent(f"""
            from guard import allowed

            def test_limit_is_enforced():
                assert allowed({test_limit}) and not allowed({test_limit + 1})
            """))
    if mutants is not None:
        (root / "scripts").mkdir()
        body = "".join(f"    Mutant({m!r}),\n" for m in mutants)
        (root / "scripts" / "mutation_check.py").write_text(
            "from dataclasses import dataclass\n\n\n@dataclass\nclass Mutant:\n    id: str\n\n\n"
            f"MUTANTS = (\n{body})\n"
        )
    return root


def test_a_change_that_keeps_existing_behaviour_passes(tmp_path):
    base = _repo(tmp_path / "base", limit=10, test_limit=10)
    head = _repo(tmp_path / "head", limit=10, test_limit=10)
    passed, failing, _ = _load().run_base_tests(base, head, timeout=120)
    assert passed and failing == []


def test_a_weakened_test_does_not_hide_a_weakened_guard(tmp_path):
    """The case this script exists for.

    The change loosens the limit from 10 to 1000 *and* edits the test to match,
    so its own test suite is green. The base branch's test still says 10.
    """
    base = _repo(tmp_path / "base", limit=10, test_limit=10)
    head = _repo(tmp_path / "head", limit=1000, test_limit=1000)
    module = _load()

    passed, failing, _ = module.run_base_tests(base, head, timeout=120)
    assert not passed
    assert failing == ["tests/test_guard.py::test_limit_is_enforced"]
    assert module.main(["--base", str(base), "--head", str(head)]) == 1


def test_a_deleted_test_is_still_run(tmp_path):
    base = _repo(tmp_path / "base", limit=10, test_limit=10)
    head = _repo(tmp_path / "head", limit=1000, test_limit=1000)
    (head / "tests" / "test_guard.py").unlink()

    passed, failing, _ = _load().run_base_tests(base, head, timeout=120)
    assert not passed
    assert failing == ["tests/test_guard.py::test_limit_is_enforced"]


def test_a_base_checkout_inside_the_head_tree_is_not_copied_into_it(tmp_path):
    """CI checks the base branch out into a subdirectory of the workspace."""
    head = _repo(tmp_path / "head", limit=10, test_limit=10)
    base = _repo(head / ".base", limit=10, test_limit=10)
    passed, failing, summary = _load().run_base_tests(base, head, timeout=120)
    assert passed, summary
    assert "1 passed" in summary  # the nested copy's test was not collected twice


def test_a_removed_mutant_is_reported(tmp_path):
    base = _repo(tmp_path / "base", limit=10, test_limit=10, mutants=["keep", "drop"])
    head = _repo(tmp_path / "head", limit=10, test_limit=10, mutants=["keep", "new"])
    module = _load()
    assert module.removed_mutants(base, head) == ["drop"]
    assert module.main(["--base", str(base), "--head", str(head)]) == 1


def test_deleting_or_breaking_the_mutant_list_removes_every_mutant(tmp_path):
    base = _repo(tmp_path / "base", limit=10, test_limit=10, mutants=["a", "b"])
    deleted = _repo(tmp_path / "deleted", limit=10, test_limit=10)
    broken = _repo(tmp_path / "broken", limit=10, test_limit=10, mutants=["a", "b"])
    (broken / "scripts" / "mutation_check.py").write_text("raise RuntimeError('nope')\n")
    module = _load()
    assert module.removed_mutants(base, deleted) == ["a", "b"]
    assert module.removed_mutants(base, broken) == ["a", "b"]


def test_a_base_branch_without_a_mutant_list_holds_nothing_out(tmp_path):
    base = _repo(tmp_path / "base", limit=10, test_limit=10)
    head = _repo(tmp_path / "head", limit=10, test_limit=10, mutants=["a"])
    assert _load().removed_mutants(base, head) == []


def test_the_real_mutant_list_can_be_read_including_generated_ids():
    ids = _load().mutant_ids(REPO_ROOT / "scripts" / "mutation_check.py")
    assert "fw-length-cap" in ids
    assert "hook-langgraph-input" in ids  # built by a helper, not a literal
