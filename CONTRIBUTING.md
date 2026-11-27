# Contributing to ADAPT-Agent

Thanks for your interest in contributing to **ADAPT-Agent** (Adversarial Defense &
Policy Training for LLM Agents). This guide covers how to set up a development
environment, the tooling we use, and the conventions we expect for branches,
commits, and pull requests.

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Development environment

ADAPT-Agent targets supported, currently maintained Python versions. Clone the
repository and install it in editable mode with the development extras:

```bash
git clone https://github.com/CodeHalwell/ADAPT-Agent.git
cd ADAPT-Agent

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install the package with development dependencies
pip install -e ".[dev]"
```

If you are working on a specific framework adapter, install its extra as well, for
example `pip install -e ".[dev,langgraph]"`.

### Testing against a real framework

A plain `.[dev]` install has no agent framework -- and no `pydantic` either, since
the core package does not depend on it -- so framework tests there run against
stubs, which cannot notice when a framework release changes its API. To run the
suite against the real thing, install **one** framework extra per virtualenv:

```bash
python -m venv .venv-crewai && source .venv-crewai/bin/activate
pip install -e ".[dev,crewai]"
pytest -o addopts=""
```

Use a separate environment per framework (`langgraph`, `pydantic-ai`, `crewai`,
`openai-agents`, `claude-agent`, `google-adk`, `microsoft-agent-framework`). They
constrain each other's dependencies: CrewAI pins `openai<3`, which holds the OpenAI
Agents SDK and Pydantic AI back, and a resolver given both quietly picks an old
CrewAI. That is also why `adapt-agent[all]` leaves CrewAI out. The per-framework
tests are written to run against exactly one framework, so a single environment
with several frameworks is not a supported way to run the suite either.

Tests that need a framework (`tests/test_real_<framework>.py`) -- or `pydantic`,
which most frameworks pull in but the dev-only environment does not have -- must
call `pytest.importorskip(...)`, so they skip in the dev-only environment and run
in the matching one. A test module that imports either at module level breaks
collection of the whole main CI job. `tests/conftest.py` sets
`OPENAI_AGENTS_DISABLE_TRACING=1` before any test imports the OpenAI Agents SDK,
so the suite never exports traces over the network; keep tests offline (no API
keys exist in CI).

### Continuous integration

`.github/workflows/ci.yml` runs on every push to `main`, every pull request, a
weekly schedule (Mondays, 05:17 UTC -- so an upstream framework release that
breaks the library is noticed without a commit), and as the gate of a release:

| Job | What it runs |
| --- | --- |
| `lint` | `ruff check .`, `black --check .`, `mypy adapt_agent`, and `scripts/check_action_refs.py` (every workflow action ref resolves). |
| `test` | `pip install -e ".[dev]"` -- **no framework, no pydantic** -- and the whole suite with `--cov-fail-under=85`, on Python 3.10, 3.11, 3.12, 3.13 and 3.14. |
| `frameworks` | One job per framework extra (`langgraph`, `pydantic-ai`, `crewai`, `openai-agents`, `claude-agent`, `google-adk`, `microsoft-agent-framework`), each `.[dev,<extra>]` at its latest release on Python 3.12, running the whole suite so the `test_real_*` files execute. |
| `frameworks-floor` | The same, with each framework pinned to the minimum its extra declares. It is `continue-on-error` (non-blocking) until it has been seen green; if you change a floor in `pyproject.toml`, this job is what validates it. |
| `mutation` | `scripts/mutation_check.py` (below). |
| `held-out` | The base branch's tests against your change (below). |
| `build` | Builds the sdist and wheel, installs the wheel and smoke-tests the import. |

### Pre-commit hooks

We use [pre-commit](https://pre-commit.com/) to run formatting and linting before
each commit. Install the hooks once after cloning:

```bash
pre-commit install
```

The hooks run automatically on `git commit`. To run them against the whole
codebase manually:

```bash
pre-commit run --all-files
```

## Running tests

We use [pytest](https://docs.pytest.org/):

```bash
pytest
```

To check coverage locally:

```bash
pytest --cov=adapt_agent --cov-report=term-missing
```

Pull requests must keep total coverage at **85% or higher** and must pass CI.

### Mutation check

Coverage shows that a line ran under test, not that a test would notice if the
line were wrong. For the security-critical modules CI therefore also runs:

```bash
python scripts/mutation_check.py          # ~2 minutes
python scripts/mutation_check.py --list   # what it checks
```

It re-introduces one security bug at a time into a scratch copy of the repo —
including every historical vulnerability recorded in `.jules/sentinel.md` — and
fails if the test suite still passes with the bug in place.

If you **fix a security bug**, add a mutant that puts it back (and cite the
journal entry in `history=`). A fix is only finished when a test fails without
it. If you **refactor a guard**, its mutant will be reported as *stale*; update
the mutant's `old`/`new` text to match the new code rather than deleting it.
`tests/test_mutation_check.py` enforces both on every ordinary test run.

### Property tests and the reference model

`tests/governance_model.py` is a short, dependency-free model of the governance
pipeline, and `tests/test_governance_properties.py` uses
[Hypothesis](https://hypothesis.readthedocs.io/) to check the implementation
against it on generated inputs. If you change what the pipeline *does*, change
the model too — it is the specification, and it is meant to stay small enough
to read in one sitting. When a property fails, Hypothesis prints a minimal
counter-example; pin it with `@example(...)` once fixed.

### Held-out check

On a pull request CI also runs the **base branch's** tests against your code
(`scripts/heldout_check.py`), and requires every base-branch mutant to still
exist, so a mutant cannot be deleted to make a change pass. It fails when your
change alters behaviour that an existing test pinned — including when you updated that test to match. That is
often exactly right; it means the change amends the specification, and a
maintainer confirms it by applying the `spec-amendment` label. Say in the PR
description which existing tests you changed and why. To run it locally:

```bash
git worktree add /tmp/base origin/main
python scripts/heldout_check.py --base /tmp/base
```

## Linting, formatting, and type checking

Before opening a pull request, make sure the following all pass:

```bash
ruff check adapt_agent tests examples scripts   # Lint
black adapt_agent tests examples scripts        # Format (CI runs black --check)
mypy adapt_agent                                # Static type checking
```

ADAPT-Agent ships a PEP 561 `py.typed` marker, so the public API is expected to be
fully type-annotated. Please add type hints to any new code and keep `mypy` clean.

## What to work on

Contributions are welcome across the library, including:

- **Core** (`adapt_agent/core`) — trust management, policy enforcement, memory,
  middleware.
- **Security** (`adapt_agent/security`) — firewall and taint tracking.
- **Adapters** (`adapt_agent/adapters`) — all adapters share the
  `GovernedAdapter` base; adding support for a new framework is usually a thin
  subclass. New framework integrations are a great place to help.
- **Framework bindings** (`adapt_agent/integrations`) — governance bound to each
  framework's own seams (middleware, hooks, plugins, guardrails), all deciding
  through `GovernanceGate`. A change here needs a `tests/test_real_<framework>.py`
  test run against the real SDK in that framework's own environment.
- **CLI** (`adapt_agent/cli`) — `info`, `validate`, `monitor`, `evaluate`,
  `optimize`, `train`, `install` and `skills` (see [docs/cli.md](docs/cli.md)).
- **Optimization, adversarial defense, evaluation, observability, and patches.**

For security-relevant changes, please also read [SECURITY.md](SECURITY.md). Do not
file vulnerability details in public issues.

## Branch and pull request conventions

1. **Branch off `main`.** Use a short, descriptive branch name with a prefix that
   reflects the change type, for example:
   - `feat/langgraph-streaming`
   - `fix/wheel-packaging`
   - `docs/cli-usage`
   - `test/firewall-coverage`
   - `chore/bump-deps`
2. **Keep PRs focused.** One logical change per pull request makes review faster.
3. **Update the changelog.** Add an entry under `## [Unreleased]` in
   [CHANGELOG.md](CHANGELOG.md) describing user-visible changes.
4. **Fill in the PR template.** Complete the checklist and describe the type of
   change.
5. **Link related issues** using `Fixes #123` / `Closes #123` where applicable.

A pull request is ready to merge when:

- [ ] Tests have been added or updated for the change.
- [ ] Coverage remains at 85% or higher.
- [ ] `ruff check .`, `black .`, and `mypy adapt_agent` all pass.
- [ ] Documentation has been updated where relevant.
- [ ] `CHANGELOG.md` has been updated.
- [ ] CI is green.

Maintainers: see [docs/releasing.md](docs/releasing.md) for how a version tag
publishes to PyPI.

## Commit messages

We recommend [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <short summary>

<optional body explaining what and why>

<optional footer, e.g. "Fixes #123">
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`.

Examples:

```
fix(packaging): include all subpackages in the built wheel
feat(adapters): add LangGraph trust/policy middleware adapter
docs(cli): document validate and monitor commands
```

Write messages in the imperative mood ("add", not "added") and keep the summary
line under ~72 characters.

## Reporting bugs and requesting features

Please use the issue templates:

- [Bug report](.github/ISSUE_TEMPLATE/bug_report.md)
- [Feature request](.github/ISSUE_TEMPLATE/feature_request.md)

Thanks again for contributing!
