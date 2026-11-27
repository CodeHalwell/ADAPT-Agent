---
name: adapt-agent
description: Evaluate, guard, and optimize LLM agents built with LangGraph, Microsoft Agent Framework, Google ADK, Pydantic AI, CrewAI, the OpenAI Agents SDK, or the Claude Agent SDK, using the ADAPT-Agent Python library. Use when writing or running agent evals against a golden dataset (text, numeric, regex, JSON, tool-call trajectory, multi-turn, or LLM-as-judge scoring) and gating CI on them, when adding guardrails to an agent (governing tool calls with allow/deny/approval policy, prompt-injection firewall, policy rules, trust and taint tracking, OpenTelemetry tracing), or when automatically tuning an agent's prompts, models, hyperparameters, or tools against a dataset.
license: MIT
---

# ADAPT-Agent

ADAPT-Agent adds three capabilities to an LLM agent you already have, without
rewriting it: **guardrails** (govern what goes in, what comes out, **and which
tools the agent may call**), **evals** (score it against a golden dataset, gate CI
on the result), and **optimization** (tune its prompts, models and tools
automatically, and know whether the gain was real).

Every framework object is duck-typed — the library never imports LangGraph,
Pydantic AI, Google ADK, and so on unless your own code already did.

## Install

```bash
pip install adapt-agent          # or: uv add adapt-agent
```

The core install is dependency-light. Framework extras (`adapt-agent[langgraph]`,
`[microsoft-agent-framework]`, `[google-adk]`, `[pydantic-ai]`, `[crewai]`,
`[openai-agents]`, `[claude-agent]`) install the framework itself — you need one
only if your agent already uses it. `[otel]` adds the OpenTelemetry bridge.
**Install one framework per environment**: `[all]` omits CrewAI (it pins
`openai<3`), so give CrewAI its own environment.

## Pick the task

| The user wants to … | Use | Depth |
| --- | --- | --- |
| Stop an agent calling a tool, require approval for one, or screen tool results | a native binding + `ToolPolicy` — below | [references/guardrails.md](references/guardrails.md) |
| Block prompt injection, enforce policy, trace runs | firewall / adapter / binding — below | [references/guardrails.md](references/guardrails.md) |
| Score an agent against expected outputs, or the tools it called | `evaluate_agent(...)` — below | [references/evals.md](references/evals.md) |
| Fail CI when an agent regresses | `adapt evaluate --fail-under ...` — below | [references/evals.md](references/evals.md) |
| Grade open-ended answers | LLM-as-judge — below | [references/evals.md](references/evals.md) |
| Improve prompts/models/tools automatically | an optimizer — below | [references/optimization.md](references/optimization.md) |

Read the reference file for the task at hand before writing non-trivial code;
each carries the full API surface, per-framework notes and gotchas:

- [references/guardrails.md](references/guardrails.md) — which control to reach for,
  the gate's tool API (`check_tool_call` / `check_tool_result`, `ToolPolicy`,
  `ToolDecision`, tool-scoped rules, approvals), **the current native binding for
  each framework with a minimal snippet**, refuse-vs-raise, streaming, report-only
  rollout, adapters (`aexecute`), `Firewall` (`normalize=`), `PolicyEnforcer`
  (action validation, condition language), `AdversarialDefense` (threat labels and
  limits), trust/taint, and `AgentObserver` with its OpenTelemetry bridge.
- [references/evals.md](references/evals.md) — `evaluate_agent` parameters, every
  built-in check and **what changed in their semantics**, trajectory and multi-turn
  evals, completeness, CI gating flags and exit codes, result export and baseline
  comparison, judges and typed errors, per-framework notes.
- [references/optimization.md](references/optimization.md) — `OptimizableAgent`,
  what gets introspected per framework, the optimizer strategies, **what
  "improved" means** (paired significance test, repeats, held-out gain,
  small-sample verdict), the validation gate, and the declarative
  `adapt-agent train` config.

## Adding guardrails

**Which control, in one rule: put it where it can see what it must stop.** An
adapter wraps the *boundary* of a run (the user's input, the final answer); by the
time it sees the answer, every tool has already run. To stop or approve a tool
call, or to catch an injection that arrived inside a tool result, bind the gate to
the framework's own tool seam. The two compose — use the binding for the inside
and an adapter at the edge.

| Need | Control |
| --- | --- |
| Deny / allow-list / require approval for tools by name | `ToolPolicy(allow=, deny=, require_approval=)` (globs: `"mcp__*"`) |
| Gate a tool on its arguments or session state | a **tool-scoped** policy rule (condition names `tool`) with `action="block"` or `"require_approval"` |
| Screen tool arguments and tool results for injection | automatic in every binding (`gate.check_tool_call` / `check_tool_result`) |
| Block content patterns | `Firewall` (+ `normalize=True` for new deployments) |
| Detect injection / jailbreak / role-spoofing phrasing | `AdversarialDefense` |
| Gate on a score or flag in state | `PolicyEnforcer` rule on `state` |
| Trace and audit | `AgentObserver(otel=True)` and `observer=` on any binding |

The one contract every binding is built on:

```python
from adapt_agent import AdversarialDefense, Firewall
from adapt_agent.core import GovernanceGate, ToolPolicy

gate = GovernanceGate(
    firewall=Firewall(),
    defense=AdversarialDefense(),
    tool_policy=ToolPolicy(deny=["delete_*"], require_approval=["send_email"]),
    agent_id="support",
)
decision = gate.check_tool_call("send_email", {"to": "a@example.com"})   # never raises
decision.action        # "allow" | "block" | "ask"  (an ask never becomes an allow)
decision.reason        # one content-free line, safe to show the model
gate.check_tool_result("search", fetched_page)   # screened as input: the indirect-injection seam
```

Each framework has a native binding in `adapt_agent.integrations.<framework>`;
all take `firewall=`, `defense=`, `policy_enforcer=` (or a shared `gate=`),
`tool_policy=`, `block_on_violation=`, `agent_id=` and `observer=`:

| Framework | Binding | Attach to |
| --- | --- | --- |
| LangGraph / LangChain v1 | `langgraph.governance_middleware()` | `create_agent(middleware=[...])` |
| Pydantic AI | `pydantic_ai.governance_capability()` | `Agent(capabilities=[...])` or `agent.run(capabilities=[...])` |
| CrewAI | `crewai.governance_hooks()` | `with governance_hooks(...):` around `kickoff()` |
| OpenAI Agents SDK | `openai_agents.govern_tools()` + `governance_run_hooks()` | the agent graph, and `Runner.run(hooks=...)` |
| Claude Agent SDK | `claude_agent.governance_hooks()` + `native_tool_options()` | `ClaudeAgentOptions(hooks=..., ...)` |
| Google ADK | `google_adk.governance_plugin()` | `App(plugins=[...])` |
| Microsoft Agent Framework | `agent_framework.governance_bundle()` | `Agent(middleware=[...])` |

A minimal one (Pydantic AI; snippets for the others are in the guardrails
reference):

```python
from pydantic_ai import Agent
from adapt_agent.integrations.pydantic_ai import governance_capability

agent = Agent(model, tools=[send_email, search], capabilities=[
    governance_capability(firewall=fw, tool_policy=ToolPolicy(deny=["delete_*"]),
                          on_tool_violation="refuse", agent_id="support-bot"),
])
```

Things to decide, and get right, because each fails quietly:

* **Refuse or raise.** *Refuse* hands the model a content-free reason and lets the
  run continue; *raise* aborts with `SecurityBlockedError` (`.reason`, `.threats`).
  Defaults differ per framework (see the reference); a blocked model input/output
  always raises.
* **Approvals.** `require_approval` pauses the call through the framework's own
  mechanism and a human can approve an ask, **never a block**. Where a framework
  cannot pause, the ask becomes a block.
* **Roll out report-only first.** `block_on_violation=False` lets everything through
  and records what would have been blocked (`decision.report_only`). The detector
  is heuristic, English-first, and flags text that *quotes* attacks.
* **Streaming.** A token already delivered cannot be recalled; each binding has a
  buffering option (trading time-to-first-token) that holds a response until it
  is screened.
* **Policy state.** The enforcer is **fail-closed** by default: a rule it cannot
  evaluate is a violation, so a `state['trust_score']` rule blocks everything if
  the binding's state has no such key. `add_rule` rejects an unknown `action`
  (`warn`, `block`, `modify`, `require_approval`) or `severity`. Keep `message`
  rules and `state` rules on separate enforcers; tool rules name `tool`.
* **The Claude Agent SDK binding is unverified against the real CLI**, and
  LangGraph MCP tools are untested; CrewAI A2A delegation is not hooked.

### Wrapping the boundary

An adapter screens a run's input and output and traces it. Policy rules gate on
agent *state*; content screening belongs to the firewall:

```python
from adapt_agent.adapters import LangGraphAdapter   # …or the adapter for your framework
from adapt_agent.security import Firewall
from adapt_agent.core import PolicyEnforcer
from adapt_agent.exceptions import SecurityBlockedError

# Content screening belongs to the firewall: it scans the whole input.
firewall = Firewall(max_content_length=10_000)
firewall.add_blocked_pattern(r"(?i)ignore previous instructions")
firewall.add_blocked_pattern(r"(?i)\bpassword\b")

# Policy rules gate on agent *state*. Under an adapter only `state` is in
# scope -- a rule written against `message` can never be evaluated.
policy = PolicyEnforcer()  # fail-closed by default
policy.add_rule(
    name="low_trust",
    description="Block callers whose trust score is too low",
    condition="state['trust_score'] < 0.5",
    action="block",       # warn | block | modify (| require_approval on tool rules)
    severity="high",
)

guarded = LangGraphAdapter(firewall=firewall, policy_enforcer=policy).wrap_agent(compiled_graph)

try:
    # `trust_score` must be present in the state, or the rule cannot be
    # evaluated -- and by default (fail-closed) that counts as a violation.
    result = guarded.execute(
        {"messages": [{"role": "user", "content": "Hello"}], "trust_score": 0.9}
    )
except SecurityBlockedError as exc:
    print(exc.reason, exc.threats)
```

Things that fail silently here:

* **An adapter does not govern tool calls**, and its `tool_policy=` does nothing on
  its own: the tools have run by the time it screens the answer. Add the native
  binding above for tool calls and keep the adapter for the boundary.
* **The enforcer is fail-closed**: a rule referencing state the adapter does not
  populate blocks *everything*. An adapter's `extract_state()` yields `messages` and
  `context`; anything else (like `trust_score`) must be in the input you pass to
  `execute()`. Extra keyword arguments (`config=`, `deps=`) reach the framework's
  run call.
* Conditions are evaluated in a sandbox (no `eval`): no function calls, attribute
  access or slices; `*` is numeric-only; NaN comparisons are errors.
* **In an async app, `await guarded.aexecute(...)`**: `execute` cannot drive an
  async framework inside a running loop, and a worker thread severs tracing
  context. Adapters exist for every framework and share this constructor; see the
  reference.

## Running evals

The one call that covers most eval work:

```python
from adapt_agent.evaluation import evaluate_agent

report = evaluate_agent(
    agent,                       # any framework agent, or a plain callable
    [
        # exact text match (the default check)
        {"input": "What is the capital of France?", "expected": "Paris"},
        # the answer number: "The answer is 42." passes against 42
        {"input": "What is 6 * 7?", "expected": 42, "check": "numeric_close"},
        # open-ended: graded by the LLM judge against per-row criteria
        {"input": "Summarise our refund policy", "check": "judge",
         "criteria": "Mentions the 30-day window and the exceptions."},
    ],
    judge="claude",              # only needed for judge rows; omit otherwise
)

report.score            # headline score in [0, 1]
report.aggregate        # {"checks": 0.83, ...} per-metric means
report.completeness()   # rows scored / errored / dropped as transient / metric errors
report.failures()       # the rows that fell short — inspect .inputs/.output/.expected
```

`data` may also be a `.jsonl` / `.json` / `.csv` path or a `GoldenDataset`.
Column names are auto-detected (`input`/`question`/`prompt`, and
`expected`/`answer`/`label`/`gold`); pass `input_key=` / `expected_key=` to be
explicit. Give rows an `id`.

`evaluate_agent` unwraps framework-native results to final response text (a Pydantic
AI `AgentRunResult`, a LangGraph state dict, a Microsoft `AgentResponse`, a Google
ADK event stream) and adapts inputs where a framework needs it (a string becomes
`{"messages": [...]}` for a LangGraph message-state graph).

### Checks

Each dataset row may declare how it is scored with a `check` field; rows without
one fall back to `exact_match`.

| Check | Passes when … |
| --- | --- |
| `exact_match` | output equals expected after NFKC / case / whitespace normalisation; punctuation stripped at token **edges** only (`-5` ≠ `5`) |
| `contains` | expected appears on **word boundaries** |
| `regex_match` | output matches the expected value as a regex (empty pattern = row error) |
| `numeric_close` | the *answer* number (after the last `answer`/`result`/`=` marker, else the last number) is within tolerance |
| `token_f1`, `jaccard`, `levenshtein_ratio` | continuous similarity scores |
| `json_subset` | every key/value of the expected dict appears in the output, recursively and type-strict |
| `tool_called`, `tool_call_args_match`, `tool_sequence`, `no_forbidden_tools`, `tool_call_budget` | **trajectory**: what tools the agent called, in what order, with what arguments |
| `judge` | the LLM judge grades the row |

**These defaults changed** (the old ones scored wrong answers as right): if a
golden score was pinned on the old semantics it will move, and each old behaviour
is one option away (`exact_match(punctuation="all")`, `contains(word_boundary=False)`,
`numeric_close(which="first")`, `json_subset(strict_types=False)`). Rows can
parameterise a check — `{"check": {"name": "numeric_close", "tolerance": 0.5}}` —
or combine several — `{"check": ["contains", "numeric_close"]}` (all must pass).
A metric that raises or returns NaN is a **row error**, never a pass. Multi-turn
rows use `"turns": [...]`.

**Structured output?** Score it per field rather than as one blended number:

```python
from adapt_agent.evaluation import field_metrics

report = evaluate_agent(agent, data, metrics=field_metrics(["lane", "matter", "action", "pack"]))
report.aggregate    # {"lane": 0.94, "matter": 0.90, "action": 0.63, "pack": 0.0}
```

That `pack: 0.0` is a column the agent never gets right; averaged into a single
score it would read as a mild dip. `evaluate_agent` notices the metrics are
structural and keeps the payload (parsing a JSON answer, including a fenced one)
instead of flattening it to text.

### LLM-as-judge

`judge=` accepts a provider name (`"claude"`, `"openai"`, `"gemini"`, …), an
`LLMJudge`, or any `prompt -> text` callable. Pin the model for a reproducible
eval — defaults move between releases:

```python
from adapt_agent.optimization.judges import ClaudeJudge

report = evaluate_agent(agent, data, judge=ClaudeJudge(model="claude-opus-5-5"))
```

Judges are prompt-injection hardened (agent output is fenced as data) and **fail
loudly**: a score outside the scale, NaN, or a truncated / refused / empty response
is an error on that row (counted in `report.n_metric_errors`), a misconfigured
provider raises, and only a throttled call is retried and then dropped. With the
default per-row `checks`, only rows declaring `{"check": "judge"}` spend judge calls.

### Gating CI

```bash
adapt evaluate "myapp.agents:build()" --data evals/golden.jsonl \
    --metric checks --extract-output --concurrency 8 \
    --fail-under 0.9 --fail-on-error --fail-on-incomplete \
    --baseline evals/baseline.json --fail-on-regression \
    --output eval-results.json --junit eval-junit.xml
```

Exit codes: `0` passed, `1` could not run, `2` usage, `3` `--fail-under` missed,
`4` rows errored (**always** when every row errored), `5` rows dropped as transient
(**always** when every row was), `6` regressed against the baseline. **Without a
gate flag an eval scoring 0% still exits 0**, so always pass `--fail-under`. An
exception raised by your agent is a failed row, not a dropped one.

## Optimizing an agent

Turn an agent into a search space and improve it against the same golden data:

```python
from adapt_agent.optimization import (
    OptimizableAgent, EvaluationHarness, CoordinateAscentOptimizer,
    GoldenDataset, exact_match,
)
from adapt_agent.optimization.judges import get_judge

data = GoldenDataset.from_jsonl("golden.jsonl")
train, holdout = data.split(0.8, seed=0)

target = OptimizableAgent.from_agent(agent)        # discovers prompts/models/tools
judge = get_judge("claude")                        # provider alias -> ClaudeJudge
harness = EvaluationHarness([exact_match(), judge.as_metric("quality")],
                            primary_metric="quality")

result = CoordinateAscentOptimizer(harness, judge=judge).optimize(
    target, train, val_dataset=holdout)
result.improved          # True only if the gain was statistically significant
result.p_value, result.delta, result.validation_rejected
result.to_config("specialists/.config/tuned.yaml")   # reviewable file, not just live state
```

**`improved` is a significance verdict, not `best_score > baseline_score`.** A
candidate is accepted only when it wins an exact paired test (`alpha`, default
`0.05`), a non-deterministic target is measured `repeats` times, and the final
verdict needs a gain on held-out data (`val_dataset`, or an automatic split from
about 38 rows). On fewer than five rows the best guess is still applied but
`improved` is `False`: say so rather than reporting a win. A winner that shows no
held-out gain is **not applied** (`validation_rejected`, `rejected_config`).
`to_config` writes `{component: {parameter: value}}` YAML for an
**optimize → diff → review → commit** loop — machine-rewritten prompts should not
reach production unread. `adapt-agent optimize` takes `--alpha`, `--repeats`,
`--holdout-fraction`, `--val-data`, `--save-config`.
`references/optimization.md` covers multi-agent systems, the other strategies and
the declarative `adapt-agent train config.yaml` flow.

## Gotchas

- **Score the answer, not the wrapper.** Outside `evaluate_agent` (driving
  `EvaluationHarness` directly), pass `output_extractor=extract_output_text` or the
  metric sees a `repr()`.
- **Google ADK needs a runner.** ADK agents execute inside a `Runner` with a
  session; use `adk_runner(agent_or_runner)` and pass that to `evaluate_agent`.
- **Async frameworks are driven synchronously by `execute`/`evaluate`.** Inside an
  already-running event loop use `await guarded.aexecute(...)` and `await
  harness.aevaluate(agent, data, concurrency=8)` — not a worker thread.
- **Concurrency.** `concurrency=` on `evaluate_agent` / the CLI; on the optimizer
  path set it on `EvaluationHarness(..., concurrency=8)`, since an optimizer calls
  `harness.evaluate(target, dataset)` with no kwargs.
- **A throttled example is not a bad answer.** 429s / 5xx / provider timeouts are
  retried, then counted in `report.n_transient_errors` and left out of the score
  (so `report.is_complete` is `False`). **Your agent's own exceptions are failures**
  unless they show infrastructure provenance. Gate on `--fail-on-incomplete`.
- **Default model IDs changed** (`claude-opus-5-5`, `gpt-5.6-luna`,
  `gemini-3.5-flash`) and SDK clients run with `max_retries=0`; Azure needs an
  explicit `api_version`. Pass `model=` to pin.
- **A structured (non-text) output survives extraction unchanged**, so score it
  with `json_subset` or `field_metrics`, not `exact_match`.
- **Unlabeled rows are fine** for judge-graded evals; `expected` is optional.
- **A config file or target string is code**: `module:attribute` is imported and a
  trailing `()` is called. Run only ones you would run as a script.
- **Failure inspection beats a single number**: `report.failures()` returns the rows
  to look at, and `failure_threshold=` sets the cutoff for continuous metrics (the
  default `1.0` counts anything short of perfect as a failure).
