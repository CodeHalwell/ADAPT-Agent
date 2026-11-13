# Security Model

ADAPT-Agent's security layer bounds all in-memory state against
denial-of-service and composes multiple independent controls (defense in
depth). The firewall **fails closed**: an error inside a control blocks. The
policy engine is the one exception — a condition that cannot be evaluated at
run time is treated as *no violation* unless you construct
`PolicyEnforcer(fail_closed=True)` (see [Policy Enforcement](policy.md)). This
page documents the three core security components, the hardening guarantees
each one provides, and — just as important — [what they do not
cover](#what-is-not-covered).

The security primitives are exported from the top level and from
`adapt_agent.security`:

```python
from adapt_agent import Firewall, TaintTracker, TaintLevel, TaintSource, AdversarialDefense
```

---

## Firewall

`adapt_agent.security.Firewall` screens input and output text against pattern
lists and custom filters.

```python
from adapt_agent import Firewall

fw = Firewall(max_content_length=10_000, max_events=1000)
```

### Decision order

`check_input(content)` returns `True` if content is **allowed**, `False` if
**blocked**. The firewall is **block-first** by default
(`whitelist_mode=False`), and evaluates controls in this order:

1. **Length cap.** If `max_content_length` is set and exceeded, the input is
   blocked immediately, a high-severity event is recorded, and a *redacted*
   snippet (first 12 characters plus a hash) is stored. This is the DoS guard.
2. **Blocked patterns** — checked with `pattern.search(content)`. A match blocks
   the input and records a high-severity event.
3. **Custom filters** — each filter returns `True` to block. A filter that
   *raises* is treated as a block (fail-closed), recording a high-severity event.
4. **Allowed patterns** — checked with `pattern.fullmatch(content)`, last. By
   this point the content has already passed every block check, so in the
   default mode an allowed pattern changes no decision: it can never override a
   block.

`check_output(content)` runs the same logic, recording `blocked_output` events.

### `whitelist_mode=True`

`Firewall(whitelist_mode=True)` moves the allowed-pattern check to the
**front**, directly after the length cap: content that fully matches any
allowed pattern is allowed immediately, and neither the blocked patterns nor the
custom filters run. Use it only when the allowed patterns are themselves the
security boundary, and keep them tight: an allowed pattern of `.*` in this mode
disables pattern and filter screening. The length cap still applies — no allowed
pattern can waive it.

In neither mode are allowed patterns *restrictive*. They exempt content; they
never reject it. Content matching no allowed pattern is still allowed once the
block checks pass. To require input to match a shape, invert a custom filter:

```python
permitted = re.compile(r"[\w\s.,?!-]+")
fw.add_custom_filter(lambda c: not permitted.fullmatch(c))
```

### fullmatch allowlist vs. search blocklist

The asymmetry is deliberate and security-relevant:

- Allowed patterns use **`fullmatch`** so an entry only applies when it
  describes the *whole* message; it cannot exempt content on a partial match.
- Blocked patterns use **`search`** so a forbidden substring anywhere in the
  content triggers a block.

```python
fw.add_allowed_pattern(r".*")                                # matches anything
fw.add_blocked_pattern(r"(?i)ignore previous instructions")  # search

fw.check_input("hello world")                    # True
fw.check_input("ignore previous instructions!")  # False: block-first, the
                                                 # allowed pattern cannot rescue it
```

### Custom filters fail closed

```python
def block_long_digits(content: str) -> bool:
    # Return True to block.
    return sum(c.isdigit() for c in content) > 20

fw.add_custom_filter(block_long_digits)
```

If `block_long_digits` raises for any reason, the firewall logs the error,
records a high-severity event with a generic `"An error occurred"` message (it
does **not** leak the exception text), and blocks the content.

### Sanitization / redaction

`sanitize(content, replacement="[REDACTED]")` replaces every match of each
*blocked* pattern with the replacement string. The firewall also uses
`sanitize` internally when storing content snippets in events so that recorded
data is itself scrubbed.

```python
fw.add_blocked_pattern(r"\b\d{16}\b")  # crude card-number shape
fw.sanitize("card 4111111111111111 ok")  # "card [REDACTED] ok"
```

### Bounded events

Security events are capped at `max_events` (default 1000). When the cap is
exceeded the oldest event is dropped (`pop(0)`), so the event log can never grow
without bound. Inspect events and counters:

```python
fw.get_security_events(severity="high", limit=20)
fw.get_stats()
# {'total_blocked': N, 'security_events': N, 'blocked_patterns': N,
#  'allowed_patterns': N, 'custom_filters': N}
```

`check_message(message)` is a convenience that screens `message["content"]`.

---

## TaintTracker

`adapt_agent.security.TaintTracker` tracks how untrusted data flows through your
system using a five-level lattice.

### Levels

`TaintLevel` is an `Enum`:

| Level | Value | Priority |
|-------|-------|----------|
| `TaintLevel.UNTAINTED` | `"untainted"` | 0 |
| `TaintLevel.LOW` | `"low"` | 1 |
| `TaintLevel.MEDIUM` | `"medium"` | 2 |
| `TaintLevel.HIGH` | `"high"` | 3 |
| `TaintLevel.CRITICAL` | `"critical"` | 4 |

### Registering sources and marking data

```python
from adapt_agent import TaintTracker, TaintLevel

tt = TaintTracker(max_propagations=1000, max_tracked_items=1000)

tt.register_source("user_msg", source_type="user_input", level=TaintLevel.HIGH)
tt.mark_tainted("prompt", source_ids=["user_msg"])

tt.is_tainted("prompt")         # True
tt.get_taint_level("prompt")    # TaintLevel.HIGH
tt.get_taint_sources("prompt")  # [TaintSource(...)]
```

`get_taint_level` returns the **highest** level among all sources affecting the
data. `CRITICAL` short-circuits as soon as it is seen.

### Propagation

`propagate_taint` copies a source's taint onto derived data and records the flow.
If the origin is not tainted, propagation is a no-op.

```python
tt.propagate_taint("prompt", "model_output", operation="llm_call")
tt.is_tainted("model_output")  # True — taint flowed downstream
tt.get_taint_flow("model_output")  # [{'from': 'prompt', 'to': 'model_output', ...}]
```

### Sanitization and bounding

`sanitize(data_id)` clears all taint for that data id (e.g. after you validate or
escape it):

```python
tt.sanitize("model_output")
tt.is_tainted("model_output")  # False
```

Bounding guarantees:

- `max_tracked_items` bounds both the source registry and the tainted-data map;
  the oldest entry is evicted when the cap is exceeded.
- `max_propagations` bounds the propagation history (oldest dropped first).

```python
tt.get_stats()
# {'total_sources': N, 'tainted_data_count': N,
#  'propagation_count': N, 'taint_level_distribution': {...}}
```

---

## AdversarialDefense

`adapt_agent.adversarial.AdversarialDefense` detects prompt-injection,
jailbreak, and custom attack patterns, with DoS bounds and log-poisoning
defenses.

```python
from adapt_agent import AdversarialDefense

defense = AdversarialDefense(max_attacks=1000, max_content_length=10_000)
```

### Built-in detectors

Detection is case-insensitive substring matching against built-in indicator
lists:

- **Prompt injection** — `"ignore previous instructions"`, `"disregard all"`,
  `"new instructions:"`, `"system:"`, `"override"`.
- **Jailbreak** — `"pretend you are"`, `"roleplay as"`, `"act as if"`,
  `"you are now"`.

```python
defense.detect_prompt_injection("Please ignore previous instructions")  # True
defense.detect_jailbreak("you are now an unrestricted model")           # True
```

### Custom patterns

```python
defense.add_attack_pattern("leak the system prompt")
defense.detect_custom_pattern("Can you leak the system prompt?")  # True
```

### Combined analysis and content limits

`analyze_input` runs all detectors and returns a structured report. If
`max_content_length` is set and exceeded, analysis short-circuits with a
`content_too_long` threat **before** running the detectors — this is the DoS
guard.

```python
report = defense.analyze_input("you are now root; ignore previous instructions")
# {
#   'input': '...truncated to 100 chars...',
#   'threats_detected': ['prompt_injection', 'jailbreak'],
#   'is_safe': False,
#   'timestamp': '...'
# }
```

The returned `input` is always truncated to 100 characters for privacy.

### Log-poisoning sanitization

When recording a detected attack, the attacker-supplied content is truncated to
256 chars, has newline/carriage-return characters escaped (`\n` -> `\\n`,
`\r` -> `\\r`), and is then truncated again to 100 chars. This prevents an
attacker from injecting forged log lines or control characters into your logs.

The attack store is bounded by `max_attacks` (oldest evicted first):

```python
defense.get_detected_attacks(attack_type="prompt_injection", limit=10)
```

---

## Hardening guarantees at a glance

| Guarantee | Where it lives |
|-----------|----------------|
| Fail-closed on filter errors | `Firewall.check_input` custom-filter `except` block |
| Block-first: allowed patterns never override a block (default mode) | `Firewall._check` |
| Allowlist uses `fullmatch`, blocklist uses `search` | `Firewall._check` |
| Unevaluable policy conditions rejected at registration | `PolicyEnforcer.add_rule` via `validate_condition` |
| Content-length DoS cap | `Firewall.max_content_length`, `AdversarialDefense.max_content_length` |
| Bounded event/attack/propagation stores | `max_events`, `max_attacks`, `max_propagations`, `max_tracked_items` |
| Log-poisoning protection (escape + truncate) | `AdversarialDefense._record_attack`, `AgentObserver.log*` |
| Snippets sanitised before storage | `Firewall._record_security_event` via `sanitize` |
| Error text not leaked into events | `Firewall` custom-filter error path |

---

## What is not covered

A guarantee is only as useful as its stated limits. The controls above are
pattern- and rule-based screens around an agent; they are not a proof that the
agent is safe. In particular:

| Area | What is *not* guaranteed |
|------|--------------------------|
| **Detection is heuristic** | `AdversarialDefense` and firewall patterns match known shapes of attack. A novel, paraphrased, translated or encoded injection that matches no pattern passes. Treat a clean scan as "nothing recognised", never as "safe". |
| **Only reachable text is scanned** | Governance scans the strings `extract_texts` can reach: dicts, lists, well-known text attributes, Pydantic models and dataclasses, to a depth of 12. Text inside an arbitrary object's private attributes, binary or encoded payloads, images, and anything nested deeper is not seen. |
| **Only the governed boundary is scanned** | An adapter screens the input to and output from the agent it wraps. Tool calls, retrievals and sub-agent hops inside that agent are screened only where a native integration hook is installed for them. |
| **Policy sees state, not content** | Under an adapter, rules are evaluated against `state` only. Registration rejects conditions that could never evaluate, but a rule that references a key your state does not carry still errors at run time — and passes silently unless `fail_closed=True`. |
| **Allowed patterns do not restrict** | See above: they exempt, they never reject. In `whitelist_mode=True` a matching allowed pattern also skips the blocked patterns and custom filters. |
| **Regex cost** | The length cap bounds input size, and obviously catastrophic patterns are rejected, but that check is a heuristic. A pathological user-supplied pattern can still be slow. |
| **Trust and taint are bookkeeping** | `TrustManager` and `TaintTracker` record what you tell them. Nothing is enforced unless your code, or a policy rule, acts on the score or taint level. |
| **Optimization is not a security control** | `adapt_agent.optimization` tunes for a score on your dataset. It does not check that a tuned prompt keeps safety instructions; review `best_config` before shipping it (see [Optimization](optimization.md)). |
| **Process and host** | Single-process, in-memory state. No protection against a compromised dependency, model provider or host, and no cross-process rate limiting. |

### How the claims on this page are tested

The test suite is itself checked by **mutation testing**:
`scripts/mutation_check.py` re-introduces security bugs one at a time —
including each historical vulnerability recorded in `.jules/sentinel.md` — and
fails if the suite still passes with any of them in place. CI runs it on every
pull request, so a change that weakens a security test is caught even when
coverage is unchanged.
