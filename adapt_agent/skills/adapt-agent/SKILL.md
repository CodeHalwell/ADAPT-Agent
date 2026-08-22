---
name: adapt-agent
description: Evaluate, guard, and optimize LLM agents built with LangGraph, Microsoft Agent Framework, Google ADK, Pydantic AI, CrewAI, the OpenAI Agents SDK, or the Claude Agent SDK, using the ADAPT-Agent Python library. Use when writing or running agent evals against a golden dataset (exact text match, numeric tolerance, regex, JSON, or LLM-as-judge scoring), when adding guardrails to an agent (prompt-injection firewall, policy rules, trust scoring, taint tracking), or when automatically tuning an agent's prompts, models, hyperparameters, or tools against a dataset.
license: MIT
---

# ADAPT-Agent

ADAPT-Agent adds three capabilities to an LLM agent you already have, without
rewriting it: **evals** (score it against a golden dataset), **guardrails**
(screen its inputs and outputs), and **optimization** (tune its prompts, models
and tools automatically).

Every framework object is duck-typed — the library never imports LangGraph,
Pydantic AI, Google ADK, and so on unless your own code already did.

## Install

```bash
pip install adapt-agent          # or: uv add adapt-agent
```

The core install is dependency-light. Framework extras (`adapt-agent[langgraph]`,
`[microsoft-agent-framework]`, `[google-adk]`, `[pydantic-ai]`, `[crewai]`,
`[openai-agents]`, `[claude-agent]`) only install the framework itself — you
need one only if your agent already uses it.

## Pick the task

| The user wants to … | Use | Depth |
| --- | --- | --- |
| Score an agent against expected outputs | `evaluate_agent(...)` — below | [references/evals.md](references/evals.md) |
| Check text/number answers, or grade open-ended ones | checks + LLM-as-judge — below | [references/evals.md](references/evals.md) |
| Block prompt injection, enforce policy, trace runs | a governed adapter — below | [references/guardrails.md](references/guardrails.md) |
| Govern one agent *inside* a multi-agent graph | native hooks — below | [references/guardrails.md](references/guardrails.md) |
| Improve prompts/models/tools automatically | an optimizer — below | [references/optimization.md](references/optimization.md) |

Read the reference file for the task at hand before writing non-trivial code;
each one carries the full API surface, per-framework notes, and gotchas:

- [references/evals.md](references/evals.md) — `evaluate_agent` parameters, every
  built-in check, per-row check specs, judges and providers, driving the
  `EvaluationHarness` directly, and a per-framework table (LangGraph input
  adaptation, Google ADK runners, Pydantic AI structured output, …).
- [references/guardrails.md](references/guardrails.md) — the adapter matrix,
  the async `aexecute` path, the native-hook integrations (middleware /
  callbacks / guardrails per framework), the `Firewall` / `TrustManager` /
  `TaintTracker` APIs (`adapt_agent.security`), `PolicyEnforcer`
  (`adapt_agent.core`) and `AdversarialDefense` (`adapt_agent.adversarial`, not
  `security`), the governance pipeline order, and config-file schema.
- [references/optimization.md](references/optimization.md) — `OptimizableAgent`,
  what gets introspected per framework, the optimizer strategies, proposers, and
  the declarative `adapt-agent train` config.

## Running evals

The one call that covers most eval work:

```python
from adapt_agent.evaluation import evaluate_agent

report = evaluate_agent(
    agent,                       # any framework agent, or a plain callable
    [
        # exact text match (the default check)
        {"input": "What is the capital of France?", "expected": "Paris"},
        # number match: "The answer is 42." passes against 42
        {"input": "What is 6 * 7?", "expected": 42, "check": "numeric_close"},
        # open-ended: graded by the LLM judge against per-row criteria
        {"input": "Summarise our refund policy", "check": "judge",
         "criteria": "Mentions the 30-day window and the exceptions."},
    ],
    judge="claude",              # only needed for judge rows; omit otherwise
)

report.score        # headline score in [0, 1]
report.aggregate    # {"checks": 0.83, ...} per-metric means
report.failures()   # the rows that fell short — inspect .inputs/.output/.expected
```

`data` may also be a `.jsonl` / `.json` / `.csv` path or a `GoldenDataset`.
Column names are auto-detected (`input`/`question`/`prompt`, and
`expected`/`answer`/`label`/`gold`); pass `input_key=` / `expected_key=` to be
explicit.

`evaluate_agent` handles the two things that otherwise break naive evals:

1. **Framework-native results are unwrapped** to final response text — a
   Pydantic AI `AgentRunResult`, a LangGraph state dict, a Microsoft
   `AgentRunResponse`, a Google ADK event stream — so a text or number check
   compares the answer, not a `repr()`.
2. **Inputs are adapted** where a framework needs it (a plain string becomes
   `{"messages": [...]}` for a LangGraph message-state graph).

### Checks

Each dataset row may declare how it is scored with a `check` field; rows
without one fall back to `exact_match`.

| Check | Passes when … |
| --- | --- |
| `exact_match` | output equals expected (case/whitespace/punctuation normalised) |
| `contains` | expected appears inside the output |
| `regex_match` | output matches the expected value as a regex |
| `numeric_close` | the first number in the output is within tolerance of expected's |
| `token_f1`, `jaccard`, `levenshtein_ratio` | continuous similarity scores |
| `json_subset` | every key/value of the expected dict appears in the output dict |
| `judge` | the LLM judge grades the row |

Rows can parameterise a check — `{"check": {"name": "numeric_close", "tolerance": 0.5}}` —
or combine several — `{"check": ["contains", "numeric_close"]}` (all must pass).

**Structured output?** Score it per field rather than as one blended number:

```python
from adapt_agent.evaluation import field_metrics

report = evaluate_agent(agent, data, metrics=field_metrics(["lane", "matter", "action", "pack"]))
report.aggregate    # {"lane": 0.94, "matter": 0.90, "action": 0.63, "pack": 0.0}
```

That `pack: 0.0` is a column the agent never gets right; averaged into a single
score it would read as a mild dip. `evaluate_agent` notices the metrics are
structural and switches to `extract_output_payload`, which strips the framework
envelope but keeps the payload (parsing a JSON answer, including a fenced one).
A deterministic field check is cheaper and more correct here than a judge.

To score every row the same way instead, pass `metrics=`:

```python
evaluate_agent(agent, data, metrics=["exact_match", "token_f1"])
evaluate_agent(agent, data, metrics="judge", judge=my_judge)   # judge every row
```

### LLM-as-judge

`judge=` accepts a provider name (`"claude"`, `"openai"`, `"gemini"`,
`"mistral"`, `"ollama"`, …), an `LLMJudge` instance, or any
`prompt -> text` callable. Judges are prompt-injection hardened (agent output
is fenced as data) and fail closed — an unreachable judge scores `0.0` rather
than inflating results.

Judge economy matters: with the default per-row `checks`, only rows declaring
`{"check": "judge"}` spend judge calls. Passing `metrics=["checks", "judge"]`
grades every row as well.

### From the CLI

```bash
adapt-agent evaluate myapp.agents:agent --data golden.jsonl \
    --metric checks --judge claude --extract-output
```

The target is `module:attribute`; append `()` to call a factory.

## Adding guardrails

Wrap an existing agent so a firewall, policy engine, adversarial detector, and
tracer run on every execution:

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
# scope -- a rule written against `message` silently never fires.
policy = PolicyEnforcer(fail_closed=True)
policy.add_rule(
    name="low_trust",
    description="Block callers whose trust score is too low",
    condition="state['trust_score'] < 0.5",
    action="block",
    severity="high",
)

guarded = LangGraphAdapter(firewall=firewall, policy_enforcer=policy).wrap_agent(compiled_graph)

try:
    # `trust_score` must be present in the state, or the rule cannot be
    # evaluated -- and with fail_closed=True that counts as a violation.
    result = guarded.execute(
        {"messages": [{"role": "user", "content": "Hello"}], "trust_score": 0.9}
    )
except SecurityBlockedError as exc:
    print(exc.reason, exc.threats)
```

Two things worth getting right, because both fail silently:

* **Use the firewall for content, policy rules for state.** An adapter's
  `execute()` evaluates policy conditions with `check_state()`, so only `state`
  is in scope; a condition referencing `message` is unevaluable and, by
  default, treated as *no violation* while the agent runs on.
* **`fail_closed=True`** makes such a rule block instead of passing quietly —
  but it cuts both ways: a rule referencing state the adapter does not populate
  now blocks *everything*. An adapter's `extract_state()` yields `messages` and
  `context`; anything else (like `trust_score`) has to be in the input you pass
  to `execute()`.

Policy conditions are evaluated in a sandbox (no `eval`), which also rules out
function calls and negative indexes. Adapters exist for every supported
framework and share this constructor — see
[references/guardrails.md](references/guardrails.md).

### In an async app, await `aexecute`

`execute` drives an async framework by running its coroutine to completion,
which is impossible inside a running event loop — so in an async web handler it
raises. Use the async twin, which applies identical governance:

```python
result = await guarded.aexecute({"messages": [{"role": "user", "content": "Hi"}]})
```

Prefer this over pushing `execute` onto a worker thread: a thread serialises
concurrent requests and severs `contextvars`, losing the OpenTelemetry span
parentage. Pydantic AI, the Claude Agent SDK and Microsoft Agent Framework are
all async-native, so this is their normal path.

### Governing one agent inside a graph

Wrapping a multi-agent workflow governs only its outer boundary. To give each
specialist its own rules, plug into the framework's own middleware/callback
chain instead — same controls, same `GovernanceGate`:

```python
from adapt_agent.integrations.agent_framework import governance_middleware

agent = chat_client.create_agent(
    instructions="...",
    middleware=[usage_middleware("nos"),                          # the app's own
                governance_middleware(firewall=fw, agent_id="nos")],
)
```

There is a factory per framework (`agent_framework`, `google_adk`,
`openai_agents`, `claude_agent`, `langgraph`, `crewai`, `pydantic_ai`) in
`adapt_agent.integrations`; `agent_id` names the refusing agent in the error.
Keep `wrap_agent` where a framework has no hook concept.

## Optimizing an agent

Turn an agent into a search space and improve it against the same golden data:

```python
from adapt_agent.optimization import (
    OptimizableAgent, EvaluationHarness, CoordinateAscentOptimizer,
    GoldenDataset, exact_match,
)
from adapt_agent.optimization.judges import get_judge

target = OptimizableAgent.from_agent(agent)        # discovers prompts/models/tools
judge = get_judge("claude")                        # provider alias -> ClaudeJudge
harness = EvaluationHarness([exact_match(), judge.as_metric("quality")],
                            primary_metric="quality")

result = CoordinateAscentOptimizer(harness, judge=judge).optimize(target, dataset)
result.improvement       # baseline -> best; the winner is applied in place
result.to_config("specialists/.config/tuned.yaml")   # ...and to a reviewable file
```

`best_config` is applied to live objects and dies with the process. `to_config`
writes it as `{component: {parameter: value}}` YAML, so the loop becomes
**optimize → diff → review → commit** and the app keeps loading prompts from
version control. `load_tuned_config(path)` flattens it back for `target.apply`.
Machine-rewritten prompts should not reach production without a human reading
the diff.

The judge both scores candidates and rewrites prompts from observed failures.
`references/optimization.md` covers multi-agent systems, the other search
strategies, and the declarative `adapt-agent train config.yaml` flow.

## Gotchas

- **Score the answer, not the wrapper.** Outside `evaluate_agent` (e.g. driving
  `EvaluationHarness` directly), pass
  `output_extractor=extract_output_text` or the metric sees a `repr()`.
- **Google ADK needs a runner.** ADK agents execute inside a `Runner` with a
  session; use `adk_runner(agent_or_runner)` and pass that to `evaluate_agent`.
- **Async frameworks are driven synchronously by `execute`** (coroutines
  awaited, event streams drained). Inside an already-running event loop use
  `await guarded.aexecute(...)`, or `await harness.aevaluate(agent, data,
  concurrency=8)` for evals — not a worker thread, which serialises requests and
  drops tracing context.
- **A serial eval is the reason nobody re-runs one.** A coordinate-ascent sweep
  is `max_evals x len(dataset)` LLM round-trips. Pass `concurrency=` to
  `evaluate_agent`/`evaluate` (threads, for sync agents) or use `aevaluate`
  (no threads, for async ones); ordering and per-example error handling are
  unchanged. On the optimizer path pass it to `EvaluationHarness(...,
  concurrency=8)` instead — an `Optimizer` calls `harness.evaluate(target,
  dataset)` with no kwargs, so a per-call argument never reaches it.
- **A throttled example is not a bad answer.** Transient failures (429, 5xx,
  timeouts) are retried with backoff, and one that outlives its retries is
  counted in `report.n_transient_errors` and excluded from the score. Scoring it
  zero would bias systematically, not randomly: the candidate evaluated while
  the provider was busiest would score lowest, and the optimizer would select
  for luck. Genuine agent errors are never retried and still score zero.
- **A structured (non-text) output survives extraction unchanged**, so score it
  with `json_subset` or a custom callable rather than `exact_match`.
- **Unlabeled rows are fine** for judge-graded evals; `expected` is optional.
- **Failure inspection beats a single number**: `report.failures()` returns the
  rows to look at, and `failure_threshold=` sets the cutoff for continuous
  metrics (the default `1.0` counts anything short of perfect as a failure).
