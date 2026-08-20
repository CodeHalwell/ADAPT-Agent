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
| Improve prompts/models/tools automatically | an optimizer — below | [references/optimization.md](references/optimization.md) |

Read the reference file for the task at hand before writing non-trivial code;
each one carries the full API surface, per-framework notes, and gotchas:

- [references/evals.md](references/evals.md) — `evaluate_agent` parameters, every
  built-in check, per-row check specs, judges and providers, driving the
  `EvaluationHarness` directly, and a per-framework table (LangGraph input
  adaptation, Google ADK runners, Pydantic AI structured output, …).
- [references/guardrails.md](references/guardrails.md) — the adapter matrix,
  `Firewall` / `PolicyEnforcer` / `AdversarialDefense` / `TrustManager` /
  `TaintTracker` APIs, the governance pipeline order, and config-file schema.
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

firewall = Firewall(max_content_length=10_000)
firewall.add_blocked_pattern(r"(?i)ignore previous instructions")

policy = PolicyEnforcer()
policy.add_rule(
    name="no_secrets",
    description="Block messages that mention a password",
    condition="'password' in message['content']",
    action="block",
    severity="high",
)

guarded = LangGraphAdapter(firewall=firewall, policy_enforcer=policy).wrap_agent(compiled_graph)

try:
    result = guarded.execute({"messages": [{"role": "user", "content": "Hello"}]})
except SecurityBlockedError as exc:
    print(exc.reason, exc.threats)
```

Policy conditions are evaluated in a sandbox (no `eval`). Adapters exist for
every supported framework and share this constructor —
see `references/guardrails.md`.

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
```

The judge both scores candidates and rewrites prompts from observed failures.
`references/optimization.md` covers multi-agent systems, the other search
strategies, and the declarative `adapt-agent train config.yaml` flow.

## Gotchas

- **Score the answer, not the wrapper.** Outside `evaluate_agent` (e.g. driving
  `EvaluationHarness` directly), pass
  `output_extractor=extract_output_text` or the metric sees a `repr()`.
- **Google ADK needs a runner.** ADK agents execute inside a `Runner` with a
  session; use `adk_runner(agent_or_runner)` and pass that to `evaluate_agent`.
- **Async frameworks are driven synchronously** (coroutines awaited, event
  streams drained). Inside an already-running event loop — a notebook, an async
  web handler — run the eval in a worker thread instead.
- **A structured (non-text) output survives extraction unchanged**, so score it
  with `json_subset` or a custom callable rather than `exact_match`.
- **Unlabeled rows are fine** for judge-graded evals; `expected` is optional.
- **Failure inspection beats a single number**: `report.failures()` returns the
  rows to look at, and `failure_threshold=` sets the cutoff for continuous
  metrics (the default `1.0` counts anything short of perfect as a failure).
