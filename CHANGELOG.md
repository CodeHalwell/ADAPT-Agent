# Changelog

All notable changes to ADAPT-Agent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Bundled agent skill** (`adapt_agent/skills/adapt-agent/`): the wheel now
  ships a `SKILL.md` plus reference files (`evals.md`, `guardrails.md`,
  `optimization.md`) that teach a coding agent to use this library. Install it
  with `adapt install skill` (`./.claude/skills/` by default, `--target user`
  for `~/.claude/skills`, `--dir PATH` for anywhere else, `--force` to replace);
  `adapt skills` lists what is bundled. Frontmatter stays within the portable
  Agent Skills field set so the skill remains valid for a claude.ai upload or
  the Skills API.
- **Skills registry API** (`adapt_agent.skills`): `available_skills()`,
  `get_skill()`, `install_skill()` / `install_all()`, `default_destination()`,
  `parse_frontmatter()` and `validate_skill()`, reading files through
  `importlib.resources` so they work from a wheel, an editable checkout, or a
  zipped distribution. Adds `SkillError` to the exception hierarchy. Importing
  the registry pulls in no agent framework or LLM SDK. A skill's **directory
  name is authoritative** — it locates the packaged files and names the
  installed directory — while the frontmatter `name` is kept as
  `Skill.declared_name` and a mismatch between the two is reported by
  `validate_skill()`. Installing copies files through the traversable API and
  stages them in a sibling directory before swapping into place, so it works
  from a zip-imported package on every supported Python and a failed
  `--force` upgrade leaves the existing installation intact. Destination and
  staging setup failures are reported as `SkillError` rather than leaking a raw
  `OSError`, matching the documented contract.
- **`adapt` console script**, a short alias for `adapt-agent` (both map to
  `adapt_agent.cli:main`), so `uv run adapt install skill` works straight after
  `uv add adapt-agent`. Help output echoes whichever name was invoked.
- **One-call framework evals** (`adapt_agent.optimization.evals`, re-exported
  from `adapt_agent.evaluation`): `evaluate_agent(agent, data, metrics=...,
  judge=...)` scores an agent built with any supported framework (LangGraph,
  Microsoft Agent Framework, Google ADK, Pydantic AI, CrewAI, OpenAI Agents
  SDK, Claude Agent SDK) or a plain callable against a golden dataset —
  deterministic checks against specific outputs (text / number / regex / JSON),
  per-row checks, and/or an LLM-as-judge — returning the standard
  `EvaluationReport`. See `docs/evals.md` and `examples/08_agent_evals.py`.
- **Framework-native output extraction**
  (`adapt_agent.optimization.extractors`): `extract_output_text()` unwraps run
  results — Pydantic AI `AgentRunResult`, Microsoft `AgentRunResponse` /
  `ChatMessage`, LangGraph final state, Google ADK / GenAI event streams and
  `Content` parts, CrewAI `CrewOutput`, OpenAI Agents `RunResult`, Claude Agent
  SDK `ResultMessage` / content blocks, conventional mappings/attributes, and
  message/event streams — to final response text so text- and number-level
  metrics compare answers rather than `repr()`s. Structural and
  dependency-free (no framework imports); unrecognised values pass through
  unchanged; extensible via `register_extractor()`. The `EvaluationHarness`
  gains an `output_extractor=` option, and the CLI `evaluate` / `optimize`
  commands gain `--extract-output`.
- **Framework runners** (`adapt_agent.optimization.runners`):
  `framework_runner()` wraps any supported agent as a plain `input -> text`
  callable (auto-adapting plain-string inputs for LangGraph message-state
  graphs via `langgraph_inputs()`), and `adk_runner()` drives a Google ADK
  agent or prebuilt `Runner` synchronously — sessions created per call so eval
  examples stay independent, `google.adk` / `google.genai` imported lazily.
- **Per-row checks metric** (`adapt_agent.optimization.metrics.checks`): each
  dataset row declares how it is scored via `metadata["check"]` — a built-in
  name (`"numeric_close"`), a parameterised form (`{"name": "numeric_close",
  "tolerance": 0.5}`), `"judge"` for LLM-judge rows, or a list combined with
  `min`/`mean` — with undeclared rows falling back to a default (
  `exact_match`). Registered as the `checks` built-in, so `--metric checks`
  works from the CLI (judge-aware when `--judge` is also passed, and routing
  judge calls only to rows that declare a judge check; `--metric judge` grades
  every row explicitly).
- **Declarative YAML/JSON training config** (`adapt_agent.optimization.config`):
  `load_training_config()` / `run_training()` and the `adapt-agent train CONFIG.yaml`
  CLI command wire a whole optimization run (target, dataset, judge, metrics,
  optimizer, explicit parameters) from one file. Temperature/`top_p` bounds that
  exceed a provider's allowable range are clamped with a warning instead of
  crashing; unknown metric/provider/optimizer/kind names raise a clear
  `TrainingConfigError`. See `examples/07_train_from_yaml.py` and
  `examples/train.example.yaml`.
- **Tool & skill optimization**: introspection now discovers `tools`/`skills` as
  searchable knobs (drop-one ablation candidates), a new `ParameterKind.SKILL`,
  and `ToolAblationProposer`. The LLM judge can act as an **adversary**
  (`LLMJudge(adversarial=True)`) and propose *new* tools/skills from observed
  failures (`LLMJudge.suggest_tools`, `LLMJudge.red_team`, `LLMToolProposer`);
  `OptimizationResult.recommendations` surfaces those advisory suggestions.
- `OptimizableAgent.add_tool_parameter()` convenience for declaring tool/skill knobs.

### Packaging

- **Release automation to PyPI on tag, built and published with uv.** Pushing a
  `v*` tag runs a gated pipeline: the tag must match `adapt_agent.__version__`;
  the full lint / type-check / 3.10-3.14 test matrix is re-run on the tagged
  commit (by reusing `ci.yml` as a callable workflow); `uv build --no-sources`
  builds the distributions once, which are then checked with
  `uvx twine check --strict`, asserted to contain the bundled skill, `py.typed`
  and both console scripts, and smoke-tested by installing the wheel into a
  clean `uv venv` and running `adapt install skill`; only then does
  `uv publish` upload them via PyPI Trusted Publishing (OIDC, no stored API
  token, PEP 740 attestations by default) and attach them to a generated
  GitHub Release. `workflow_dispatch` runs the same pipeline against TestPyPI
  for rehearsal, using the `testpypi` publish target declared under
  `[[tool.uv.index]]`. See `docs/releasing.md`.
- Bundled skill files are declared as `package-data` and ship in **both** the
  wheel and the sdist; a test asserts every skill file is covered by a
  `package-data` glob, so a new file can't silently go missing from the wheel.
- Added `MANIFEST.in`, so the sdist also carries `docs/`, `examples/`,
  `CHANGELOG.md` and the other project docs for downstream packagers.
- The distribution version is now single-sourced from `adapt_agent.__version__`
  via `[tool.setuptools.dynamic]`, so the package and the distribution can no
  longer disagree. Both artifacts pass `twine check`.

### Security

- The bundled skill's guardrail recipes previously demonstrated two controls
  that **silently enforce nothing**, both now corrected and covered by tests:
  a policy rule conditioned on `message[...]` never fires under a governed
  adapter (adapters evaluate `check_state()`, so `message` is out of scope and
  the default `fail_closed=False` treats it as no violation), and
  `add_allowed_pattern()` on a default `Firewall` never rejects, because
  allow-list enforcement requires `whitelist_mode=True`. The skill now screens
  content with the firewall, gates state with policy rules, recommends
  `PolicyEnforcer(fail_closed=True)` for security rules, and documents the
  sandbox's real grammar (no function calls, no negative indexes).
- `Firewall`'s class docstring described `whitelist_mode=True` as "a strict
  allowlist". It is not: allowed patterns **exempt** content and never reject
  it, in either mode — content matching no allowed pattern is still allowed
  once the block checks pass, and `whitelist_mode` changes only the precedence
  between allowed and blocked patterns. The docstring now says so and points at
  the working alternative (invert a custom filter). Behaviour is unchanged.
- The skill's example JSON config paired a `message`-scoped policy rule with
  the `fail_closed=True` adapter recipe on the same page, which would have
  refused every request; its rule now gates on state, with content screening
  left to the `firewall` section. Its training-config example bound a
  parameter to a component it never declared — the config parsed but
  `adapt-agent train` would fail at build time.

### Documentation

- **Release guide** (`docs/releasing.md`): the one-time PyPI trusted-publisher
  setup (including the *pending publisher* flow this first release needs), the
  step-by-step release procedure, what the pipeline verifies before publishing,
  and troubleshooting. Linked from the README, `CONTRIBUTING.md` and the nav.
- **Agent skill guide** (`docs/skill.md`): what a skill is, install targets,
  the Python registry API, and the rules for bundling your own. Linked from the
  MkDocs nav, the README, and `docs/cli.md` (which now documents both console
  scripts and the `install` / `skills` commands).
- **Running Evals guide** (`docs/evals.md`): quick start, built-in check
  table, per-row checks, LLM-as-judge, framework-by-framework notes (LangGraph
  / Microsoft Agent Framework / Google ADK / Pydantic AI / CrewAI / OpenAI
  Agents / Claude Agent SDK), output extraction, direct-harness usage, and CLI
  examples. Added `examples/08_agent_evals.py` (offline, no framework or API
  key required) and new API reference entries.
- **Per-framework guides and example ladders** for all seven supported frameworks
  (LangGraph, Microsoft Agent Framework, Google ADK, Pydantic AI, CrewAI, OpenAI
  Agents SDK, Claude Agent SDK). Each framework gains a verbose
  `docs/frameworks/<framework>.md` guide plus an `examples/<framework>/` folder
  with a 3–4 step ladder (basic guarded agent → policy/observability/trust →
  evaluate & optimize → multi-agent system + declarative YAML training), a
  `train.yaml` template, and a README. Added a `docs/frameworks/` index hub, a new
  "Framework Guides" section in the MkDocs nav, and a per-framework table in
  `examples/README.md`. Examples guard their optional-framework import and run
  offline (deterministic judge stubs, no API key) where they exercise ADAPT-Agent.

### Changed / Fixed

- **Judge robustness**: brace-depth JSON parsing, labeled-score extraction with
  out-of-range rejection, `score_is_normalized`, position-swap debiasing in
  `compare`, rubric moved to the provider `system` prompt with fenced untrusted
  content (prompt-injection / reward-hacking hardening), and auth errors are
  re-raised (not silently scored 0.0).
- **Provider sampling safety**: omit `temperature`/`top_p` for models that reject
  them (e.g. Anthropic Opus 4.8/4.7, Fable 5), clamp to each provider's
  `max_temperature`, and retry once without sampling params on a 400.
- **Security defaults**: firewall checks blocked patterns before any allow-list
  short-circuit (allow no longer nullifies block; opt back in with
  `whitelist_mode=True`); adversarial detection normalizes input (NFKC /
  zero-width / whitespace) and records each attack once; `PolicyEnforcer(fail_closed=True)`
  and a node-count cap; `TrustManager` LRU eviction with a distrust floor;
  `TaintTracker` fails closed on evicted sources; `Middleware(fail_closed=True)`
  and duplicate-name rejection.
- Introspection predicate hardening so a Microsoft Agent Framework `ChatAgent` /
  magentic orchestrator is never mis-claimed by another framework's introspector.
- `min_improvement` default raised to `1e-3` (avoid chasing judge noise);
  `EvaluationHarness(failure_threshold=...)`; dataset loaders gain explicit
  `input_key`/`expected_key` overrides and no longer treat an `output` column as
  ground truth.
- `pyyaml` is now a runtime dependency; the "dependency-free" framing is dropped
  in favour of "install what the job needs" (agent-framework SDKs stay lazy).
- **Dataset-driven optimization & evaluation subsystem** (`adapt_agent.optimization`).
  Evaluate any agent against a golden dataset and automatically optimize it --
  tuning prompts, few-shot examples, models, hyperparameters, routing/topology,
  and tool allow-lists -- for a single agent, six specialists, an
  orchestrator + sub-agents, or a workflow, across every supported framework. All
  import-safe and offline-testable (no LLM SDK or framework imported unless used).
  New building blocks:
  - `GoldenDataset` / `Example` -- load from list / JSON / JSONL / CSV, split,
    sample.
  - `Parameter` / `SearchSpace` -- tunable knobs bound to live framework objects.
  - `OptimizableAgent` -- wrap arbitrary agent code as `run()` + a search space,
    via `from_agent` / `from_components` / `from_callable`.
  - `LLMJudge` -- provider-agnostic model-graded scoring **and** prompt
    improvement, used at every stage (`score`/`compare`/`critique`/
    `improve_prompt`/`as_metric`). Provider-specific subclasses `ClaudeJudge`,
    `OpenAIJudge`, `AzureOpenAIJudge`, `GeminiJudge`, `MistralJudge`,
    `CohereJudge`, `GroqJudge`, `TogetherJudge`, `OpenRouterJudge`, `OllamaJudge`,
    `BedrockJudge`, `HuggingFaceJudge` (+ `get_judge`).
  - `ModelProvider` and concrete providers (`anthropic`, `openai`,
    `azure_openai`, `gemini`, `mistral`, `cohere`, `groq`, `together`,
    `openrouter`, `ollama`, `bedrock`, `huggingface`, plus `callable` / `echo`),
    each importing its SDK lazily; `get_provider` / `register_provider`.
  - Metrics: `exact_match`, `contains`, `regex_match`, `token_f1`, `jaccard`,
    `numeric_close`, `json_subset`, `levenshtein_ratio`, and `LLMJudge.as_metric`.
  - `EvaluationHarness` / `EvaluationReport` -- run an agent over a dataset,
    aggregate scores, surface failures.
  - Proposers (`CandidateProposer`, `NumericProposer`, `PromptMutationProposer`,
    `FewShotProposer`, `LLMProposer`) and optimizers (`CoordinateAscentOptimizer`,
    `BootstrapFewShotOptimizer`, `GridSearchOptimizer`, `RandomSearchOptimizer`,
    `EvolutionaryOptimizer`, `PipelineOptimizer`, `make_default_optimizer`).
  - **Deep per-framework introspection** for all seven adapters (LangGraph,
    Microsoft Agent Framework, Google ADK, Pydantic AI, CrewAI, OpenAI Agents,
    Claude Agent SDK): turn a live agent object into bound, tunable parameters
    (`adapt_agent.optimization.introspection`).
  - The evaluation engine is re-exported from `adapt_agent.evaluation` so it is
    discoverable in the "eval" namespace, and key symbols from the package root.
  - **CLI commands** `adapt-agent evaluate` and `adapt-agent optimize` that load
    an agent (``module:attribute``, with ``()`` to call a factory) and a golden
    dataset (``.json``/``.jsonl``/``.csv``), score it with built-in metrics
    and/or an LLM judge (``--judge claude|openai|gemini|...``), and optimize it
    (``--optimizer``, ``--component NAME=module:attr`` for multi-agent systems,
    ``--save-config``, ``--val-data``, ``--json``).
  - Docs (`docs/optimization.md`, expanded `docs/cli.md`) and a runnable, offline
    example (`examples/06_optimize_with_golden_dataset.py`).
- **Shared `GovernedAdapter` base** (`adapt_agent.adapters._governed`) factoring
  out the framework-agnostic governance pipeline (input screening, policy,
  middleware, traced execution, output screening) with transparent handling of
  sync results, async coroutines, and async event streams.
- **Six new framework adapters**, all built on `GovernedAdapter`:
  `MicrosoftAgentFrameworkAdapter`, `GoogleADKAdapter`, `PydanticAIAdapter`,
  `CrewAIAdapter` (now fully implemented), `OpenAIAgentsAdapter`, and
  `ClaudeAgentSDKAdapter`. Each ships with its own optional extra
  (e.g. `adapt-agent[pydantic-ai]`).
- Full `monitor` command output streaming for live observability sessions (planned).

### Changed

- `LangGraphAdapter` now subclasses `GovernedAdapter` (behaviour unchanged).

### Removed

- The placeholder `SemanticKernelAdapter` and the `semantic-kernel` extra have
  been removed.

### Fixed

- `TaintTracker.get_stats` now counts each distinct source once per taint level
  instead of once per (data, source) pairing, so a source tainting many data
  items no longer inflates the `taint_level_distribution`.
- `AgentEvaluator.get_evaluation_results` now returns a shallow copy when called
  without filters, preventing callers from mutating the evaluator's internal
  results buffer.

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
