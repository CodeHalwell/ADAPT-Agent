# Command-Line Interface

Installing ADAPT-Agent provides the `adapt-agent` console script
(`adapt_agent.cli:main`). It exposes three commands plus a global `--version`.

```bash
adapt-agent --version
adapt-agent            # no command prints help
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
