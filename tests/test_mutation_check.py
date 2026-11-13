"""Keep ``scripts/mutation_check.py`` honest between full runs.

The full check re-runs the suite once per mutant and belongs in its own CI job.
These tests are the cheap part: they make sure the mutant list still *applies*
to the code and still covers the vulnerability journal, so the expensive job
can never quietly degrade into checking nothing.
"""

import importlib.util
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    name = "mutation_check"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "mutation_check.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


#: Journal entries with no mutant, each with the reason. Adding to this set is
#: a reviewed decision; an entry missing from both it and the mutant list fails.
_NO_MUTANT = {
    # The fix bounded the *cost* of escaping (truncate before replace). Undoing
    # it changes run time on huge inputs, not any output a test can assert on.
    # Its observable half -- the length bound -- is ``fw-truncate-overflow``.
    "2024-05-26 - [DoS and Logging Corruption via Naive Escaping]",
}


def _journal_headings() -> set[str]:
    text = (REPO_ROOT / ".jules" / "sentinel.md").read_text(encoding="utf-8")
    return {m.group(1).strip() for m in re.finditer(r"^## (.+)$", text, flags=re.MULTILINE)}


def test_every_mutant_still_applies_to_the_code():
    assert _load().stale_mutants() == []


def test_every_history_reference_names_a_real_journal_entry():
    headings = _journal_headings()
    unknown = {h for m in _load().MUTANTS for h in m.reopens} - headings
    assert unknown == set()


def test_every_journal_entry_has_a_mutant_or_a_stated_reason():
    """A vulnerability fixed without a way to detect its return is half fixed."""
    covered = {h for m in _load().MUTANTS for h in m.reopens}
    headings = _journal_headings()
    assert headings - covered - _NO_MUTANT == set()
    # ...and the exemption list cannot outlive the entries it exempts.
    assert _NO_MUTANT <= headings
    assert not (_NO_MUTANT & covered)


def test_stale_mutants_detects_a_refactored_guard(tmp_path):
    """The staleness check itself must fire -- otherwise it proves nothing."""
    module = _load()
    for mutant in module.MUTANTS:
        target = tmp_path / mutant.path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text((REPO_ROOT / mutant.path).read_text(encoding="utf-8"))
    victim = module.MUTANTS[0]
    path = tmp_path / victim.path
    path.write_text(path.read_text(encoding="utf-8").replace(victim.old, "refactored away"))

    problems = module.stale_mutants(tmp_path)
    assert any(p.startswith(f"{victim.id}:") for p in problems)


def test_this_file_is_excluded_from_mutant_runs():
    """Otherwise the staleness test above would kill every mutant by itself.

    Each mutant removes its own anchor, so ``test_every_mutant_still_applies``
    fails in every mutated copy. Left in, the mutation check would report a
    perfect score no matter how weak the real tests were.
    """
    module = _load()
    assert Path(__file__).resolve() == (REPO_ROOT / module._SELF_TEST).resolve()
    source = (REPO_ROOT / "scripts" / "mutation_check.py").read_text(encoding="utf-8")
    assert 'f"--ignore={_SELF_TEST}"' in source
