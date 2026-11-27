# API Reference

A concise reference for every public symbol exported from `adapt_agent`
(`adapt_agent.__all__`), plus the governance core and the evaluation and
optimization modules most code reaches for. Import the top-level names
directly:

```python
from adapt_agent import (
    TrustManager, PolicyEnforcer, MemorySystem, Middleware,
    Firewall, TaintTracker, TaintLevel, TaintSource,
    AdversarialDefense, AgentEvaluator, AgentObserver, AgentOptimizer,
    PatchManager, BaseAdapter,
)
```

`__version__` is also exported (`"0.5.1"` at the time of writing).

---

## Core

### TrustManager

`TrustManager(initial_trust=0.5, min_trust=0.0, max_trust=1.0, max_history=1000, max_agents=1000, distrust_floor=0.25)`

Tracks per-agent trust scores, clamped to `[min_trust, max_trust]`, with bounded
history and a bounded number of tracked agents.

Key methods:

- `get_trust_score(agent_id) -> float`
- `update_trust_score(agent_id, delta, reason="", factors=None) -> float`
- `evaluate_agent_state(agent_id, state) -> TrustScore` — derives a score from
  state factors (e.g. penalizes `policy_violations`). It deliberately ignores any
  agent-self-reported `trust_score` to prevent privilege escalation.
- `is_trusted(agent_id, threshold=0.6) -> bool`
- `get_trust_history(agent_id) -> list[TrustScore]`

### PolicyEnforcer

`PolicyEnforcer(max_violations=1000, fail_closed=True)`

Evaluates declarative rules written in a safe, restricted expression language
against messages and state. See [Policy Enforcement](policy.md).

`fail_closed` decides what a rule that **cannot be evaluated** at run time
means: a violation (`True`, the default since 0.5.0) or no violation (`False`).

Key methods:

- `add_rule(name, description, condition, action="warn", severity="medium")` —
  raises `ValueError` if `condition` exceeds 1024 chars, or could never be
  evaluated (a function call, attribute access, an unsupported operator, or a
  name other than `state` / `message` / `tool`); if `action` is not in
  `VALID_ACTIONS` or `severity` not in `VALID_SEVERITIES`; or if the action is
  `"require_approval"` and the condition does not name `tool`.
- `check_tool(tool, state=None) -> list[str]` — evaluate the **tool-scoped**
  rules (conditions naming `tool`) against `tool = {"name": ..., "arguments": ...}`
  and an optional `state`. Tool-scoped rules are skipped by `check_state` and
  `check_message`. `GovernanceGate.check_tool_call` calls this.
- `evaluate_rules(*, state=None, message=None, tool=None) -> dict[str, str]` —
  dry-run every rule against an example; `"violation"`, `"pass"` or `"error"`
  per rule, with no side effects.
- `get_rule_stats() -> dict[str, dict[str, int]]` — per-rule counts of
  `violation` / `pass` / `error` / `unresolved` outcomes.
- `vacuous_rules() -> list[str]` — rules that have been evaluated and have never
  once seen the data they read.
- `remove_rule(name) -> bool`, `get_rule(name) -> PolicyRule | None`,
  `list_rules() -> list[PolicyRule]`
- `register_handler(action, handler)` — raises `ValueError` for an action not in
  `VALID_ACTIONS`.
- `check_message(message) -> list[str]` (context var: `message`)
- `check_state(state) -> list[str]` (context var: `state`)
- `get_violations(severity=None, limit=None) -> list[dict]`

Module constants in `adapt_agent.core.policy`:
`VALID_ACTIONS = {"warn", "block", "modify", "require_approval"}` and
`VALID_SEVERITIES = {"low", "medium", "high", "critical"}` (frozensets).

### GovernanceGate

`GovernanceGate(*, firewall=None, defense=None, policy_enforcer=None, block_on_violation=True, agent_id="agent", tool_policy=None, max_payload_chars=4_000_000)`

From `adapt_agent.core` (also re-exported by `adapt_agent.integrations`). The
one implementation of the security controls that every adapter and native
binding runs. See [Native framework bindings](integrations.md#the-common-model).

- `scan_input(payload) -> list[str]`, `scan_output(payload) -> list[str]` —
  threats found; never raise.
- `review_input(payload, *, state=None) -> list[str]`,
  `review_output(payload) -> list[str]` — the same, raising
  `SecurityBlockedError` when blocking.
- `policy_violations(state) -> list[str]` — the blocking rules `state` violates.
- `check_tool_call(tool_name, arguments=None, *, state=None) -> ToolDecision` —
  tool policy, then firewall + defense over the arguments, then tool-scoped
  policy rules. Never raises.
- `check_tool_result(tool_name, result, *, state=None) -> ToolDecision` — the
  result screened as input (the indirect-injection seam); `"allow"` or
  `"block"` only. Never raises.
- `review_tool_call(...)`, `review_tool_result(...)` — the same, raising
  `SecurityBlockedError` on a block (an `"ask"` is returned).

`max_payload_chars` bounds the distinct text of one payload: over it, nothing is
scanned and the threat is `"payload_too_large"` (`None` disables the bound). A
payload the walk cannot read in full — too deep, too many objects, or holding a
one-shot stream — yields `"unscannable"`. Both are blocking threats like any
other. Constants: `DEFAULT_MAX_PAYLOAD_CHARS`, `UNSCANNABLE`,
`PAYLOAD_TOO_LARGE`, and the actions `ALLOW`, `BLOCK`, `ASK`
(`adapt_agent.core.governance`).

### ToolDecision

`ToolDecision(action, tool_name, threats=(), report_only=False, agent_id="")`
(frozen dataclass, from `adapt_agent.core`)

- `action` — `"allow"`, `"block"` or `"ask"` (pause for a human).
- `threats` — e.g. `"tool:denied:<name>"`, `"tool:not_allowed:<name>"`,
  `"tool:approval:<name>"`, `"firewall"`, `"prompt_injection"`, `"policy:<rule>"`.
- `report_only` — the gate is report-only and would otherwise have blocked or
  asked; `action` is then `"allow"`.
- `.allowed`, `.blocked`, `.needs_approval`
- `.reason` — one line naming the tool and the threat labels, never the
  arguments or result; safe to show the model.
- `.resolve(*, approvals_supported) -> str` — the action to take; an `"ask"`
  becomes `"block"` when the framework cannot pause for approval.

### ToolPolicy

`ToolPolicy(*, allow=None, deny=(), require_approval=())` (from `adapt_agent.core`)

Shell-style glob lists over tool names, matched case-sensitively. Precedence:
deny, then allow-list (`None` = no restriction, `[]` = refuse every tool), then
approval. A bare string instead of a collection raises `TypeError`.
`verdict(tool_name) -> (action, threat_label | None)`.

### walk_texts / TextExtraction

`walk_texts(data) -> TextExtraction` (`adapt_agent.core.governance`)

Collects every string reachable in a payload — mappings (keys too), sized
collections, `bytes` (decoded as UTF-8), the declared fields of dataclasses,
Pydantic, attrs and msgspec models, `__slots__`, `to_dict()`, and well-known
text attributes — walking each object once. Never raises and never consumes a
one-shot iterator. `TextExtraction(texts, incomplete)` reports what was missed:
`incomplete` holds `"depth"`, `"budget"`, `"stream"` or `"error"`, and
`.complete` is `True` when it is empty. `extract_texts(data)` returns the texts
alone.

### MemorySystem

`MemorySystem(short_term_capacity=100, long_term_capacity=10000)`

Bounded short-term and long-term key/value memory. Long-term eviction drops the
least-accessed item.

Key methods:

- `store_short_term(key, value, metadata=None)`
- `store_long_term(key, value, metadata=None)`
- `retrieve(key, from_long_term=False) -> Any | None`
- `search(query, from_long_term=False, limit=10) -> list[dict]` (substring match)
- `consolidate() -> int` — promotes frequently accessed short-term items
- `clear_short_term()`, `clear_long_term()`, `get_stats() -> dict`

### Middleware

`Middleware()`

Composable pre/post processing pipeline over `dict` payloads. Middleware are
sorted by `priority` (higher runs first). A middleware that raises is logged and
skipped (the pipeline continues).

Key methods:

- `add_pre_middleware(middleware, name=None, priority=0)`
- `add_post_middleware(middleware, name=None, priority=0)`
- `remove_middleware(name) -> bool`
- `process_input(data) -> dict`, `process_output(data) -> dict`
- `wrap_function(func) -> Callable`
- `list_middleware() -> list[dict]`

---

## Security

### Firewall

`Firewall(max_content_length=None, max_events=1000, whitelist_mode=False, normalize=False)`

Input/output screening with a `search` blocklist, fail-closed custom filters, a
length cap that no allowed pattern can waive, a `fullmatch` allowlist, and
bounded events. Block-first by default; `whitelist_mode=True` lets a matching
allowed pattern skip the blocklist and filters; `normalize=True` also matches
patterns against a normalized view of the text. See
[Security Model](security.md).

Key methods:

- `add_blocked_pattern(pattern, flags=0)`, `add_allowed_pattern(pattern, flags=0)`
- `add_custom_filter(filter_func)` — `filter_func(content) -> bool` (True blocks)
- `check_input(content) -> bool`, `check_output(content) -> bool`
  (True = allowed)
- `check_message(message) -> bool`
- `sanitize(content, replacement="[REDACTED]") -> str`
- `dry_run(content) -> dict` — `{"allowed", "reason", "pattern"}`: the decision
  `check_input` would make, recording nothing.
- `get_pattern_hits() -> dict[str, int]` — how many times each blocked pattern
  has blocked content.
- `get_security_events(severity=None, limit=None) -> list[SecurityEvent]`
- `get_stats() -> dict`

### TaintTracker

`TaintTracker(max_propagations=1000, max_tracked_items=1000, unknown_source_level=TaintLevel.CRITICAL)`

Tracks taint across data flow using the `TaintLevel` lattice, with bounded
sources, tainted-data map, and propagation history.

Key methods:

- `register_source(source_id, source_type, level=TaintLevel.MEDIUM, metadata=None) -> TaintSource`
- `mark_tainted(data_id, source_ids)`
- `is_tainted(data_id) -> bool`
- `get_taint_level(data_id) -> TaintLevel` (highest among sources)
- `get_taint_sources(data_id) -> list[TaintSource]`
- `propagate_taint(from_data_id, to_data_id, operation="unknown")`
- `sanitize(data_id)` — clears taint
- `reset()` — forget every source, taint record and saturation state
- `get_taint_flow(data_id) -> list[dict]`, `get_stats() -> dict`

### TaintLevel

`Enum` with members `UNTAINTED`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` (string
values match the member names lowercased).

### TaintSource

`TaintSource(source_id, source_type, level, metadata=None)`

Value object describing a taint origin. Attributes: `source_id`, `source_type`,
`level` (a `TaintLevel`), `metadata`, `timestamp`.

---

## Adversarial / Evaluation / Observability / Optimization

### AdversarialDefense

`AdversarialDefense(max_attacks=1000, max_content_length=100_000)`

Detects prompt injection, jailbreaks, and custom patterns; bounds the attack
store and sanitizes recorded content against log poisoning. See
[Security Model](security.md).

Key methods:

- `detect_prompt_injection(prompt, prompt_lower=None) -> bool`
- `detect_jailbreak(prompt, prompt_lower=None) -> bool`
- `detect_custom_pattern(prompt, prompt_lower=None) -> bool`
- `detect_role_spoofing(prompt) -> bool` — forged role turns and tags
- `analyze_input(input_text) -> dict` (`is_safe`, `threats_detected`, `details`,
  ...); input over `max_content_length` is reported as `content_too_long`
- `add_attack_pattern(pattern)`
- `get_detected_attacks(attack_type=None, limit=None) -> list[dict]`

### AgentEvaluator

`AgentEvaluator(max_results=1000)`

Scores responses with custom metrics and aggregates them. See
[Observability](observability.md#agentevaluator).

Key methods:

- `register_metric(name, metric_func)` — `metric_func(output, expected) -> float`
- `evaluate_response(agent_id, input_data, output_data, expected_output=None) -> dict`
- `compute_aggregate_metrics(agent_id=None) -> dict[str, float]`
- `get_evaluation_results(agent_id=None, limit=None) -> list[dict]`

### evaluate_agent and the eval toolkit

`evaluate_agent(agent, data, *, metrics=None, judge=None, judge_criteria=None, judge_rubric=None, primary_metric=None, input_adapter="auto", output_extractor="auto", input_key=None, expected_key=None, capture_output=True, max_results=10_000, failure_threshold=1.0, concurrency=1, conversation_input="messages") -> EvaluationReport`

One-call golden-dataset evals for any supported framework agent (or callable):
loads the dataset, discovers the framework run method, adapts inputs
(LangGraph), unwraps framework-native outputs, and scores with deterministic
checks, the per-row `checks` dispatcher, and/or an LLM-as-judge. See
[Running Evals](evals.md).

Companion utilities (importable from `adapt_agent.evaluation` and
`adapt_agent.optimization`):

- `extract_output_text(value)` — best-effort, dependency-free unwrapping of
  framework results (`AgentRunResult`, `AgentRunResponse`, LangGraph state,
  ADK events, `CrewOutput`, `RunResult`, message streams) to final text;
  unrecognised values pass through unchanged.
- `register_extractor(name, predicate, unwrap)` / `available_extractors()` —
  extend extraction for custom result types.
- `framework_runner(agent, *, input_adapter="auto", output_extractor=...)` —
  wrap any supported agent as a plain `input -> text` callable.
- `langgraph_inputs(value)` — the string → message-state input adapter.
- `adk_runner(target, *, app_name=..., user_id=..., message_factory=None, output_extractor=...)`
  — drive a Google ADK agent (or prebuilt `Runner`) synchronously with a fresh
  session per call.
- `checks(default="exact_match", judge=None, aggregate="min")` — the per-row
  check dispatcher metric (rows declare `{"check": ...}` in metadata).
- `EvaluationHarness(metrics, ..., output_extractor=None)` — the underlying
  measurement engine; see [Optimization & Evaluation](optimization.md).

### Trajectory metrics

`adapt_agent.optimization.trajectory` (also importable from
`adapt_agent.evaluation`) scores what an agent *did*: the tool calls it made, in
order, with their arguments. See [Running Evals](evals.md#trajectory-tool-call-evals).

- Metric factories: `tool_called(name=None, *, min_times=1)`,
  `tool_call_args_match(expected_tools=None, *, check="exact_match")`,
  `tool_sequence(expected_tools=None, *, mode="in_order", check_args=False, check="exact_match")`,
  `no_forbidden_tools(forbidden=None)`,
  `tool_call_budget(max_calls=None, *, max_steps=None)`. Expectations come from
  the factory arguments or the row (`expected_tools`, `forbidden_tools`,
  `max_tool_calls`, `max_steps`); a row declaring none is an error for that row,
  never a silent pass.
- `extract_trajectory(value) -> Trajectory` — recover the calls from a
  framework-native result; `Trajectory(tool_calls, steps)`,
  `ToolCall(name, arguments, result, id)`.

### Reporting: exporting and comparing runs

`adapt_agent.optimization.reporting` (also importable from
`adapt_agent.evaluation`) — what `adapt evaluate --output`, `--junit` and
`--baseline` are built on. See [Running Evals](evals.md#exporting-results-and-comparing-runs).

- `results_document(report, *, max_field_chars=4000, metadata=None) -> dict`,
  `to_json(report, path=None, ...) -> str` — the full run as JSON.
- `to_junit(report, path=None, *, thresholds=None, ...) -> str` — JUnit XML, one
  test case per row × metric.
- `load_results(path) -> dict`;
  `compare_runs(current, baseline, *, tolerance=0.0) -> RunComparison` —
  per-metric deltas and the rows that regressed or improved.

### AgentObserver

`AgentObserver(max_logs=1000, max_traces=1000, max_metrics=1000, max_events_per_trace=1000, max_metric_names=1000, *, max_text_length=10000, max_error_length=200, redactor=None, otel=False)`

Records traces, events, logs, metrics, refusals and tool verdicts with bounded
storage and log-poisoning defenses. `otel=True` (or a `Tracer`, or an
`OTelBridge`) mirrors them into OpenTelemetry spans; it needs the `otel` extra.
See [Observability](observability.md#agentobserver).

Key methods:

- `start_trace(trace_id, agent_id, operation, metadata=None, *, parent_id=None) -> dict`
- `end_trace(trace_id, status="completed", result=None, *, threats=None, error=None, report_only=False)`
- `record_refusal(agent_id, operation, threats=None, *, stage="input", reason=None, error=None, report_only=False, parent_id=None, metadata=None) -> dict`
  — a request refused before any trace was opened; read them back with
  `get_refusals(agent_id=None, limit=None)`.
- `tool_span(agent_id, tool_name, *, parent_id=None, tool_call_id=None, metadata=None)`
  — context manager timing one tool call; `span.record(decision, phase)`.
- `record_tool_decision(agent_id, decision, *, phase="call", parent_id=None, tool_call_id=None, metadata=None) -> dict`
  — one `ToolDecision` as a zero-length tool span; read them back with
  `get_tool_decisions(agent_id=None, tool_name=None, limit=None)`.
- `log_event(trace_id, event_type, description, metadata=None)`
- `log(level, message, agent_id=None, metadata=None)`
- `record_metric(metric_name, value)` — rejects NaN and infinity
- `get_traces(agent_id=None, status=None, limit=None) -> list[dict]`
- `get_logs(level=None, agent_id=None, limit=None) -> list[dict]`
- `get_metric_stats(metric_name) -> dict[str, float]`, `get_stats() -> dict[str, int]`

Getters return snapshots: mutating them does not change what is stored.

### Optimizers

`GridSearchOptimizer`, `RandomSearchOptimizer`, `CoordinateAscentOptimizer`,
`BootstrapFewShotOptimizer`, `EvolutionaryOptimizer`, `PipelineOptimizer` and
`make_default_optimizer(harness, ...)`. See
[Optimization & Evaluation](optimization.md).

Shared constructor options (keyword-only): `max_evals=50`, `seed=0`,
`min_improvement=1e-3`, `judge=None`, `suggest_tools=False`,
`cache_evaluations=True`, `verbose=False`, and:

- `validation_gate=True` — with a held-out set (`val_dataset`, or the automatic
  split), apply the winner only if it shows a **gain** over the untuned agent
  there.
- `alpha=0.05` — significance level of the paired sign-flip test every
  acceptance rests on.
- `repeats=None`, `detect_nondeterminism=True` — measure a non-deterministic
  target several times per comparison (`DEFAULT_NONDETERMINISTIC_REPEATS = 3`
  once noise is detected).
- `holdout_fraction=0.25`, `min_holdout_rows=10` — the automatic held-out split.
- `allow_small_sample=True` — decide comparisons too small to be significant on
  the point estimate (reported `significant=False`); `False` accepts nothing
  unproven.
- `proposer_judge=None` — a separate judge for critique and prompt rewriting, so
  the model that writes a candidate is not the one that scores it.

`validation_tolerance` is no longer used; passing it emits a `FutureWarning`.

`optimize(agent, dataset, *, val_dataset=None, ...) -> OptimizationResult`

`OptimizationResult` reports the verdict's evidence: `significant`, `delta`,
`p_value`, `ci_low`, `ci_high`, `n_rows`, `repeats`, `alpha`,
`significance_basis`, `nondeterministic`, `untuned_parameters`, and the
validation-gate fields `validation_baseline_score`, `validation_rejected`,
`rejected_config`, `rejected_reason`. `improved` is `True` only for a
significant gain. See [Optimization](optimization.md#what-improved-means).

### Significance testing

`adapt_agent.optimization.significance` — the paired test behind every
acceptance. `compare_rows(incumbent, challenger, *, alpha=0.05, min_effect=0.0, exclude=(), repeats=1, interval=False, seed=0) -> PairedComparison`
compares per-row scores with an exact sign-flip test (a seeded Monte Carlo or a
normal approximation takes over for long runs of continuous deltas).
`PairedComparison` carries `delta`, `p_value`, `n_rows`, `n_better`, `n_worse`,
`ci_low`, `ci_high`; `min_rows_for_significance(alpha)` is the smallest row
count that could ever be significant (5 at `alpha=0.05`). `PairedComparison`
and `compare_rows` are also importable from `adapt_agent.optimization`.

### Model providers

`get_provider(name, **kwargs) -> ModelProvider` (from `adapt_agent`, or
`adapt_agent.optimization.providers`). `DEFAULT_MODELS` maps each provider name
to the model used when none is given (`"anthropic": "claude-opus-5-5"`, ...). A
`ModelProvider(model, *, temperature=0.0, max_tokens=None, system=None)` offers:

- `complete(prompt, **overrides) -> str`, and `acomplete(...)`, its async twin.
- `complete_json(prompt, schema, **overrides) -> str` — output constrained to a
  JSON schema; raises `StructuredOutputUnsupported` (a `NotImplementedError`)
  before any request when the provider or its SDK cannot do it.
- `last_usage` / `usage` — `TokenUsage(input_tokens, output_tokens, calls)` for
  the latest call and the running total.

A call that succeeded but produced no usable answer raises a
`ProviderResponseError` (attributes `kind`, `provider`, `model`, `stop_reason`,
`partial_text`) subclass: `TruncatedResponseError`, `RefusedResponseError` or
`EmptyResponseError`. Messages never quote the model's text.

### AgentOptimizer

`AgentOptimizer(max_metrics=1000, max_suggestions=1000)`

Records performance samples and suggests improvements (slow execution / high
token usage). See [Observability](observability.md#agentoptimizer).

Key methods:

- `analyze_performance(agent_id, execution_time, token_usage=None, success=True) -> dict`
- `suggest_optimizations(agent_id) -> list[dict]`

---

## Patches & Adapters

### PatchManager

`PatchManager()`

Registers and applies framework patches, tracking which have been applied
(idempotent re-apply; a raising patch returns `False`).

Key methods:

- `register_patch(patch_id, framework, description, patch_func, version_requirement=None)`
- `apply_patch(patch_id, target) -> bool`
- `list_patches(framework=None) -> list[dict]`
- `is_applied(patch_id) -> bool`

### BaseAdapter

`BaseAdapter(config=None)` (abstract)

The contract all framework adapters implement. See
[Framework Adapters](adapters.md).

Abstract methods: `wrap_agent(agent) -> Agent`,
`extract_state(agent) -> AgentState`,
`inject_middleware(agent, middleware) -> Any`.
Concrete helpers: `validate_agent(agent) -> bool`,
`get_framework_name() -> str`.

All concrete adapters subclass `GovernedAdapter`, which implements the shared
governance pipeline. Its keyword-only constructor takes `firewall`, `defense`,
`policy_enforcer`, `observer`, `middleware`, `agent_id`, `block_on_violation`,
`tool_policy` and `max_payload_chars`, and exposes them as one `gate`
(`GovernanceGate`). A wrapped agent's `execute(input_data, **run_kwargs)` /
`aexecute(...)` forward the keyword arguments to the framework's run method.
Available from `adapt_agent.adapters`: `LangGraphAdapter`,
`MicrosoftAgentFrameworkAdapter`, `GoogleADKAdapter`, `PydanticAIAdapter`,
`CrewAIAdapter`, `OpenAIAgentsAdapter`, `ClaudeAgentSDKAdapter`.

---

## Types

Exported from `adapt_agent` (defined in `adapt_agent.core.types`):

| Type | Kind | Shape |
|------|------|-------|
| `Agent` | Protocol | `execute(input_data) -> dict`, `get_state() -> AgentState` |
| `Adapter` | Protocol | `wrap_agent(agent) -> Agent`, `extract_state(agent) -> AgentState` |
| `AgentMessage` | TypedDict | `role: str`, `content: str`, `metadata?: dict` |
| `AgentState` | TypedDict | `messages: list[AgentMessage]`, `context: dict`, `trust_score?: float`, `policy_violations?: list[str]` |
| `TrustScore` | TypedDict | `score: float`, `confidence: float`, `factors: dict[str, float]`, `timestamp: str` |
| `PolicyRule` | TypedDict | `name`, `description`, `condition`, `action`, `severity` (all `str`) |
| `SecurityEvent` | TypedDict | `event_type`, `severity`, `description`, `timestamp` (`str`), `metadata: dict` |

---

## Exceptions

From `adapt_agent.exceptions`:

- `AdaptError` — base class for all ADAPT-Agent errors.
- `SecurityBlockedError(reason, threats=None)` — raised when a control blocks
  input/output. Attributes: `reason: str`, `threats: list[str]`.
- `AdapterError` — base class for adapter errors (e.g. wrapping a non-graph).
- `MissingDependencyError(package, extra)` — raised when an optional framework
  dependency is required but not installed.
- `IncompleteEvaluationError` — an evaluation could not score every row and the
  caller needs all of them (an optimizer's baseline that stays incomplete).
- `SkillError` — installing or reading a bundled agent skill failed.

`ProviderResponseError` and its subclasses live in
`adapt_agent.optimization.providers` (see [Model providers](#model-providers)).
