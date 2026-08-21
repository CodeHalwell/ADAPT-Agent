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
regex patterns compile, and action/severity values are valid.

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
    --metric token_f1 --judge claude --judge-model claude-opus-4-8 --primary judge

# Machine-readable:
adapt-agent evaluate myapp.agents:agent --data golden.jsonl --metric exact_match --json
```

Dataset files are dispatched by extension: `.json`, `.jsonl`, `.csv`. Metrics
(`--metric`, repeatable) are any built-in: `exact_match`, `contains`,
`regex_match`, `token_f1`, `jaccard`, `numeric_close`, `json_subset`,
`levenshtein_ratio`. Adding `--judge PROVIDER` appends an LLM-judge metric named
`judge` (providers include `claude`, `openai`, `gemini`, `mistral`, `cohere`,
`groq`, `together`, `ollama`, `bedrock`, `huggingface`). `--primary` selects the
headline metric (defaults to the first). Exit code is `0` on success, `1` on any
error (printed as text, or `{"status": "error", ...}` with `--json`).

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
| `--val-data FILE` | Held-out dataset; its score is reported as `validation`. |
| `--save-config FILE` | Write the winning configuration to a JSON file. |
| `--verbose` | Log each trial. |

`--json` emits the `OptimizationResult` summary (`improved`, `baseline_score`,
`best_score`, `improvement`, `validation_score`, `n_evals`, `best_config`).

!!! note "The target executes"
    `evaluate`/`optimize` import and run your agent code, which executes it.
    Point them at your own modules. An LLM judge or a live framework agent will
    make real model/API calls (and may require credentials), so start with a
    small dataset.

---

## Configuration file schema

The config root must be a JSON **object**. All top-level keys are optional.

| Key | Type | Description |
|-----|------|-------------|
| `policy_rules` | array of objects | Each: `name` (required), `condition` (required string, ≤ 1024 chars, must parse), `description`, `action` (`warn`/`block`/`modify`, default `warn`), `severity` (`low`/`medium`/`high`/`critical`, default `medium`). |
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
    `validate` only checks that a condition *parses* as a Python expression and
    is within the length limit. It does not check that the condition uses only
    the supported safe nodes — that restriction is enforced at evaluation time by
    the [`PolicyEnforcer`](policy.md). Keep conditions to comparisons, boolean
    ops, arithmetic, subscripts, and literals (no function calls or attribute
    access).
