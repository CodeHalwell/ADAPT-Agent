# Command-Line Interface

Installing ADAPT-Agent provides two console scripts — `adapt-agent` and the
shorter `adapt` — which are the same program (`adapt_agent.cli:main`). Either
name accepts every command below, plus a global `--version`. Under uv, prefix
with `uv run`.

```bash
adapt --version
adapt                  # no command prints help
uv run adapt install skill
```

---

## `adapt install skill`

Installs the [agent skill](skill.md) bundled inside the wheel into a skills
directory, so a coding agent can pick it up.

```bash
adapt install skill                        # -> ./.claude/skills/adapt-agent
adapt install skill adapt-agent            # a specific bundled skill
adapt install skill --target user          # -> ~/.claude/skills
adapt install skill --dir path/to/skills   # an explicit directory
adapt install skill --force                # replace an existing installation
adapt install skill --json
```

Exits non-zero if the skill is unknown, or if it is already installed and
`--force` was not given.

## `adapt skills`

Lists the skills bundled with this installation.

```bash
adapt skills
adapt skills --json
```

---

## `adapt-agent info`

Prints library information and a summary of features.

```bash
adapt-agent info
```

```text
ADAPT-Agent v0.2.0
Adversarial Defense & Policy Training for LLM Agents
...
Features:
  - Trust management and policy enforcement
  - Security firewall and taint tracking
  - Adversarial defense (prompt injection / jailbreak detection)
  - Adapters: LangGraph, Microsoft Agent Framework, Google ADK,
    Pydantic AI, CrewAI, OpenAI Agents SDK, Claude Agent SDK
  - Performance optimization, evaluation and observability
```

---

## `adapt-agent validate`

Validates a JSON configuration file. Checks that policy-rule conditions parse,
regex patterns compile, and action/severity values are valid. The
`require_approval` action is accepted on **tool-scoped** rules only -- rules
whose condition references `tool` -- because only a tool call can be paused for
a human; on any other rule it is reported as an error.

```bash
adapt-agent validate config.json
adapt-agent validate config.json --json   # machine-readable output
```

Exit code is `0` when valid and `1` when there are errors (or the file cannot be
read/parsed). Human-readable output lists each error; `--json` emits
`{"valid": <bool>, "errors": [...]}`.

```text
Configuration is valid. (2 policy rule(s))
```

---

## `adapt-agent monitor`

Initializes the observability/security stack for an agent and prints a readiness
snapshot. A config file is optional; when provided it is validated first (an
invalid config aborts with exit code `1`).

```bash
adapt-agent monitor --agent-id agent-007
adapt-agent monitor --agent-id agent-007 --config config.json
adapt-agent monitor --agent-id agent-007 --config config.json --json
```

JSON output shape:

```json
{
  "status": "ready",
  "agent_id": "agent-007",
  "version": "0.2.0",
  "controls": {
    "policy_rules": 2,
    "firewall_blocked_patterns": 1,
    "firewall_allowed_patterns": 1,
    "adversarial_patterns": 1
  },
  "timestamp": "2026-06-26T00:00:00+00:00"
}
```

---

## `adapt-agent evaluate`

Evaluates an agent against a golden dataset and prints the metric scores.

The `target` is your agent, given as `module:attribute` — append `()` to call a
zero-argument factory. It may resolve to an `OptimizableAgent`, a framework
object (CrewAI `Crew`, Pydantic AI `Agent`, …), or a plain runner callable. The
current working directory is added to `sys.path`, so a target in your project
(e.g. `src/agents.py`) is importable as `src.agents:build_agent()`.

```bash
# A built-in metric:
adapt-agent evaluate myapp.agents:agent --data golden.jsonl --metric exact_match

# Score with an LLM-as-judge (provider-agnostic) plus a built-in metric:
adapt-agent evaluate "myapp.agents:build()" --data golden.jsonl \
    --metric token_f1 --judge claude --judge-model claude-opus-5-5 --primary judge

# Machine-readable:
adapt-agent evaluate myapp.agents:agent --data golden.jsonl --metric exact_match --json
```

Dataset files are dispatched by extension: `.json`, `.jsonl`, `.csv`. Metrics
(`--metric`, repeatable) are any built-in: `exact_match`, `contains`,
`regex_match`, `token_f1`, `jaccard`, `numeric_close`, `json_subset`,
`levenshtein_ratio`, `checks`, and the trajectory metrics `tool_called`,
`tool_call_args_match`, `tool_sequence`, `no_forbidden_tools`,
`tool_call_budget` (rows declare `expected_tools` etc. -- see
[Running Evals](evals.md#trajectory-tool-call-evals)). Rows may also be
multi-turn conversations (`"turns": [...]`). Adding `--judge PROVIDER` appends
an LLM-judge metric named `judge` (providers include `claude`, `openai`,
`gemini`, `mistral`, `cohere`, `groq`, `together`, `ollama`, `bedrock`,
`huggingface`). `--primary` selects the headline metric (defaults to the
first). `--concurrency N` runs N rows at once (thread pool).

Every run prints what the score covers -- rows total, scored, errored (scored
0), dropped as transient, and with a metric error -- and any aggregate computed
over a subset of the rows is labelled `PARTIAL` next to the number:

```text
Evaluated 'myapp.agents:agent' over 3 example(s):
  rows: 3 total, 3 scored (2 errored, scored 0), 0 dropped as transient, 0 with a metric error
  primary metric: exact_match = 0.3333  (over 3/3 rows)
    - exact_match: 0.3333  (over 3/3 rows)
  errors: 2   avg latency: 0.0003s   p50: 0.0002s   p95: 0.0009s
    ! row 1 [ToolTimeout]: agent exceeded its step budget
  GATE FAILED (exit 3): exact_match = 0.3333 < 0.9000 (--fail-under)
```

An exception raised by your agent counts as that row failing. Only an error
that shows it came from a provider SDK, an HTTP client, or carries an HTTP
status is treated as a transient infrastructure fault and dropped from the
score ([details](evals.md#what-counts-as-transient)).

### Gating CI

| Option | Effect |
|--------|--------|
| `--fail-under [METRIC=]SCORE` | Exit `3` if the metric's aggregate is below `SCORE`. A bare number gates every metric; `METRIC=SCORE` gates one and overrides a bare value. Repeatable. An unknown metric is an error. |
| `--fail-on-error` | Exit `4` if any row errored (the agent raised, or a metric failed). |
| `--fail-on-incomplete` | Exit `5` if any row was dropped as transient, i.e. the score covers part of the dataset. |
| `--baseline RESULTS.json` | Compare with a previous `--output` file: per-metric deltas, rows that regressed / improved / appeared / disappeared. |
| `--fail-on-regression` | With `--baseline`: exit `6` if a metric's aggregate or any row's score dropped by more than `--regression-tolerance` (default `0`). |
| `--output PATH` | Write full per-row results as JSON (the `--baseline` input). |
| `--junit PATH` | Write JUnit XML: one testcase per row × metric; failures carry expected / actual / input (truncated; the document is size-capped). |

Exit codes:

| Exit | Meaning |
|------|---------|
| `0` | Ran; every gate passed. |
| `1` | Could not run: the target, dataset or judge failed to load, bad option values, an empty dataset. Printed as `ERROR: ...` (or `{"status": "error", ...}` with `--json`). |
| `2` | Command-line usage error (argparse). |
| `3` | `--fail-under` missed. |
| `4` | Rows errored with `--fail-on-error` -- **always** when every row errored. |
| `5` | Rows dropped as transient with `--fail-on-incomplete` -- **always** when every row was. |
| `6` | Regressed against `--baseline` with `--fail-on-regression`. |

When several gates fail, the most severe code wins (`4`, `5`, `3`, `6`) and each
failure is printed. With `--json`, stdout carries the summary, `completeness`,
the per-row `rows`, a `gates` block (`passed`, `exit_code`, `failures`,
`thresholds`) and, with `--baseline`, the comparison.

Copy-paste CI step:

```yaml
- name: Agent evals
  run: |
    adapt evaluate "myapp.agents:build()" --data evals/golden.jsonl \
      --metric checks --extract-output --concurrency 8 \
      --fail-under 0.9 --fail-on-error --fail-on-incomplete \
      --baseline evals/baseline.json --fail-on-regression \
      --output eval-results.json --junit eval-junit.xml
```

Refresh the baseline by committing a new `--output` file from a run you accept.
Give dataset rows an `id` so rows are matched by identity, not position.

---

## `adapt-agent optimize`

Optimizes an agent against a golden dataset — tuning prompts, few-shot examples,
models, hyperparameters, routing, and tools — then applies the best
configuration to the live agent and reports the improvement.

```bash
# Single agent, default "do everything" pipeline:
adapt-agent optimize "myapp.agents:build()" --data golden.jsonl \
    --metric exact_match --judge openai --optimizer default --max-evals 60

# A multi-agent system: the target is the entrypoint, components are tunable:
adapt-agent optimize myapp.app:orchestrate \
    --component researcher=myapp.agents:researcher \
    --component writer=myapp.agents:writer \
    --data golden.jsonl --metric token_f1 --judge claude \
    --optimizer coordinate_ascent --val-data holdout.jsonl \
    --save-config best_config.json
```

Shares all of `evaluate`'s options, plus:

| Option | Description |
|--------|-------------|
| `--optimizer` | `default` (pipeline), `coordinate_ascent`, `bootstrap_few_shot`, `grid`, `random`, `evolutionary`. |
| `--max-evals` | Evaluation budget (default 60). |
| `--seed` | RNG seed for reproducible search (default 0). |
| `--component NAME=module:attr` | Register a framework component to introspect for tunable knobs (repeatable). With components, `target`/`--runner` is the system entrypoint. |
| `--runner module:attr` | Explicit runner callable driving the whole system. |
| `--val-data FILE` | Held-out dataset; its score is reported as `validation`, and a tuned config that scores worse on it than the untuned agent is not applied. |
| `--save-config FILE` | Write the winning configuration where `load_tuned_config` reads it back: `.json` as flat `{"component.knob": value}`, `.yaml`/`.yml` nested by component via `OptimizationResult.to_config`. Values that cannot round-trip -- live tool lists, callables, SDK objects -- are **not** written (a stand-in string would break the agent when re-applied); they are named under `_provenance.unexportable` (JSON) or a comment header (YAML) and on the console. |
| `--concurrency N` | Rows evaluated at once per trial. |
| `--verbose` | Log each trial. |

`--json` emits the `OptimizationResult` summary (`improved`, `baseline_score`,
`best_score`, `improvement`, `validation_score`, `n_evals`, `best_config`).

!!! warning "The target executes: configs and targets are code"
    `module:attribute` **imports** that module -- running its top-level code --
    and a trailing `()` **calls** the object. The current working directory is
    appended to `sys.path`, so a module in the directory you run from is
    importable (appended, not prepended, so it cannot shadow an installed
    package). The same holds for `--component`, `--runner` and every `target:`
    entry of an `adapt train` config. **Only run targets and configs you would
    run as a script**; a config file from an untrusted source is untrusted
    code. An LLM judge or a live framework agent will make real model/API calls
    (and may require credentials), so start with a small dataset.

---

## `adapt-agent train`

Runs a full optimization pass from a YAML/JSON training config (see
`examples/train.example.yaml` and `adapt_agent.optimization.config`).

```bash
adapt-agent train train.yaml
adapt-agent train train.yaml --save-config tuned.yaml   # or .json
adapt-agent train train.yaml --json
```

`--save-config` writes exactly as `optimize --save-config` does. The config's
`judge:` section accepts only `provider`, `model`, `adversarial`, `scale`,
`pass_threshold`, `score_is_normalized`, `criteria`, `metric_name`, `rubric`,
`max_failures`, `on_error`, `temperature`, `max_tokens` and `system`; anything
else -- notably `base_url` and `api_key` -- is refused, so a config cannot send
graded outputs to another host or carry a credential (use the provider's
environment variable). `metrics:` entries are names or mappings for metrics
that need options, e.g. `{name: field_match, field: lane}`.

---

## Configuration file schema

The config root must be a JSON **object**. All top-level keys are optional.

| Key | Type | Description |
|-----|------|-------------|
| `policy_rules` | array of objects | Each: `name` (required), `condition` (required string, ≤ 1024 chars, must parse), `description`, `action` (`warn`/`block`/`modify`, or `require_approval` on a rule whose condition references `tool`; default `warn`), `severity` (`low`/`medium`/`high`/`critical`, default `medium`). |
| `firewall` | object | `blocked_patterns` (array of valid regex strings), `allowed_patterns` (array of valid regex strings), `max_content_length` (positive integer). |
| `adversarial` | object | `attack_patterns` (array of strings). |

### Complete example

```json
{
  "policy_rules": [
    {
      "name": "no_passwords",
      "description": "Block messages mentioning a password",
      "condition": "'password' in message['content']",
      "action": "block",
      "severity": "high"
    },
    {
      "name": "non_empty_context",
      "description": "Warn when the agent state carries context",
      "condition": "state['context'] != {}",
      "action": "warn",
      "severity": "medium"
    }
  ],
  "firewall": {
    "blocked_patterns": [
      "(?i)ignore previous instructions",
      "\\b\\d{3}-\\d{2}-\\d{4}\\b"
    ],
    "allowed_patterns": [
      "[A-Za-z0-9 ?.!,]+"
    ],
    "max_content_length": 10000
  },
  "adversarial": {
    "attack_patterns": [
      "leak the system prompt"
    ]
  }
}
```

!!! note "Condition validation"
    `validate` applies the same check as `PolicyEnforcer.add_rule`: the
    condition must parse, be within the length limit, use only the supported
    safe nodes (comparisons, boolean ops, arithmetic, subscripts, literals — no
    function calls or attribute access), and reference only `state` or
    `message`. A config that validates is one whose rules will register. It
    cannot check that the keys a rule reads exist in your data; see
    [Policy Enforcement](policy.md) for `evaluate_rules`.
