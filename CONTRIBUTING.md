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

## Linting, formatting, and type checking

Before opening a pull request, make sure the following all pass:

```bash
ruff check .        # Lint
black .             # Format
mypy adapt_agent    # Static type checking
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
- **CLI** (`adapt_agent/cli`) — `info`, `validate`, and `monitor` commands.
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
