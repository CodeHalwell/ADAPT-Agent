# Changelog

All notable changes to ADAPT-Agent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Full `monitor` command output streaming for live observability sessions (planned).
- Production-ready Semantic Kernel and CrewAI adapters (currently experimental/planned).

## [0.2.0] - 2026-06-26

This release is a productisation pass that turns the 0.1.0 proof of concept into a
package that actually installs and is usable end to end.

### Fixed

- **Critical packaging bug.** The built wheel previously shipped only the top-level
  `adapt_agent/__init__.py` and omitted every subpackage (`core`, `security`,
  `adapters`, `optimization`, `adversarial`, `evaluation`, `observability`,
  `patches`, `cli`). This broke `pip install adapt-agent` for all real usage —
  importing anything beyond `__version__` raised `ImportError`. Packaging is now
  configured to include the complete package tree.

### Added

- **PEP 561 typing marker.** Added a `py.typed` file so that downstream projects
  pick up ADAPT-Agent's inline type hints under `mypy` and other type checkers.
- **Real LangGraph adapter.** `adapt_agent.adapters.langgraph` now provides a
  working integration for wrapping LangGraph agents with ADAPT-Agent's trust,
  policy, and security middleware. The Semantic Kernel
  (`adapt_agent.adapters.semantic_kernel`) and CrewAI
  (`adapt_agent.adapters.crewai`) adapters remain experimental/planned.
- **CLI `validate` command.** `adapt-agent validate <config_file>` validates an
  agent configuration file (trust thresholds, policy rules, firewall settings).
- **CLI `monitor` command.** `adapt-agent monitor --agent-id <id>` starts an
  observability session for a running agent.
- **Expanded public API.** Additional symbols are re-exported from the
  `adapt_agent` top-level package, including core components (`TrustManager`,
  `PolicyEnforcer`, `MemorySystem`, `Middleware`) and security components
  (`Firewall`, `TaintTracker`).
- **CI/CD pipeline** running linting, type checking, and the test suite across
  supported Python versions.
- **Test coverage** raised across the core, security, and adapter modules.
- **Documentation and examples** covering installation, the public API, the CLI,
  and the LangGraph adapter.

### Changed

- Bumped version to `0.2.0`.

## [0.1.0] - 2024-01-01

### Added

- Initial public release.
- Core components: `TrustManager`, `PolicyEnforcer`, `MemorySystem`, and
  `Middleware`.
- Security components: `Firewall` and `TaintTracker`.
- Initial scaffolding for framework adapters (LangGraph, Semantic Kernel, CrewAI),
  optimization, adversarial defense, evaluation, observability, patches, and the
  `adapt-agent` CLI (`info` command).

[Unreleased]: https://github.com/CodeHalwell/ADAPT-Agent/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/CodeHalwell/ADAPT-Agent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/CodeHalwell/ADAPT-Agent/releases/tag/v0.1.0
