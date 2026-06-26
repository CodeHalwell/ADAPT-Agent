# API Reference

A concise reference for every public symbol exported from `adapt_agent`
(`adapt_agent.__all__`). Import any of these from the top level:

```python
from adapt_agent import (
    TrustManager, PolicyEnforcer, MemorySystem, Middleware,
    Firewall, TaintTracker, TaintLevel, TaintSource,
    AdversarialDefense, AgentEvaluator, AgentObserver, AgentOptimizer,
    PatchManager, BaseAdapter,
)
```

`__version__` is also exported (currently `"0.2.0"`).

---

## Core

### TrustManager

`TrustManager(initial_trust=0.5, min_trust=0.0, max_trust=1.0, max_history=1000, max_agents=1000)`

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

`PolicyEnforcer(max_violations=1000)`

Evaluates declarative rules written in a safe, restricted expression language
against messages and state. See [Policy Enforcement](policy.md).

Key methods:

- `add_rule(name, description, condition, action="warn", severity="medium")` —
  raises `ValueError` if `condition` exceeds 1024 chars.
- `remove_rule(name) -> bool`, `get_rule(name) -> PolicyRule | None`,
  `list_rules() -> list[PolicyRule]`
- `register_handler(action, handler)`
- `check_message(message) -> list[str]` (context var: `message`)
- `check_state(state) -> list[str]` (context var: `state`)
- `get_violations(severity=None, limit=None) -> list[dict]`

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

`Firewall(max_content_length=None, max_events=1000)`

Input/output screening with a `fullmatch` whitelist, a `search` blocklist,
fail-closed custom filters, a length cap, and bounded events. See
[Security Model](security.md).

Key methods:

- `add_blocked_pattern(pattern, flags=0)`, `add_allowed_pattern(pattern, flags=0)`
- `add_custom_filter(filter_func)` — `filter_func(content) -> bool` (True blocks)
- `check_input(content) -> bool`, `check_output(content) -> bool`
  (True = allowed)
- `check_message(message) -> bool`
- `sanitize(content, replacement="[REDACTED]") -> str`
- `get_security_events(severity=None, limit=None) -> list[SecurityEvent]`
- `get_stats() -> dict`

### TaintTracker

`TaintTracker(max_propagations=1000, max_tracked_items=1000)`

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

`AdversarialDefense(max_attacks=1000, max_content_length=None)`

Detects prompt injection, jailbreaks, and custom patterns; bounds the attack
store and sanitizes recorded content against log poisoning. See
[Security Model](security.md).

Key methods:

- `detect_prompt_injection(prompt, prompt_lower=None) -> bool`
- `detect_jailbreak(prompt, prompt_lower=None) -> bool`
- `detect_custom_pattern(prompt, prompt_lower=None) -> bool`
- `analyze_input(input_text) -> dict` (`is_safe`, `threats_detected`, ...)
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

### AgentObserver

`AgentObserver(max_logs=1000, max_traces=1000, max_metrics=1000, max_events_per_trace=1000, max_metric_names=1000)`

Records traces, events, logs, and metrics with bounded storage and log-poisoning
defenses. See [Observability](observability.md#agentobserver).

Key methods:

- `start_trace(trace_id, agent_id, operation, metadata=None) -> dict`
- `end_trace(trace_id, status="completed", result=None)`
- `log_event(trace_id, event_type, description, metadata=None)`
- `log(level, message, agent_id=None, metadata=None)`
- `record_metric(metric_name, value)`
- `get_traces(agent_id=None, status=None, limit=None) -> list[dict]`
- `get_logs(level=None, agent_id=None, limit=None) -> list[dict]`
- `get_metric_stats(metric_name) -> dict[str, float]`

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
governance pipeline. Available from `adapt_agent.adapters`: `LangGraphAdapter`,
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
