# Pull Request

## Summary

<!-- Describe what this PR does and why. What problem does it solve? -->

Fixes # <!-- issue number, if applicable -->

## Type of change

<!-- Mark all that apply with an "x". -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that changes existing behavior)
- [ ] Documentation update
- [ ] Refactor / internal change (no functional change)
- [ ] Tests / CI
- [ ] Chore (dependencies, packaging, tooling)

## Affected components

<!-- Which parts of ADAPT-Agent does this touch? -->

- [ ] `core` (trust, policy, memory, middleware)
- [ ] `security` (firewall, taint tracker)
- [ ] `adapters` (LangGraph / Semantic Kernel / CrewAI)
- [ ] `cli` (`info` / `validate` / `monitor`)
- [ ] `optimization` / `adversarial` / `evaluation` / `observability` / `patches`
- [ ] Packaging / build / typing
- [ ] Other (describe below)

## Checklist

- [ ] Tests have been added or updated to cover the change.
- [ ] Total test coverage is maintained at **>= 85%**.
- [ ] Documentation has been updated where relevant.
- [ ] Ran `ruff check .` with no errors.
- [ ] Ran `black .` (code is formatted).
- [ ] Ran `mypy adapt_agent` with no errors.
- [ ] Updated `CHANGELOG.md` under the `[Unreleased]` section.
- [ ] CI passes on this PR.

## Additional notes

<!-- Anything reviewers should know: trade-offs, follow-ups, screenshots, etc. -->
