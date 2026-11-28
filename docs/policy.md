# Policy Enforcement

`adapt_agent.core.PolicyEnforcer` evaluates declarative rules against agent
messages and state. Conditions are written in a **safe, restricted expression
language** that is parsed to an AST and evaluated by a hand-written interpreter —
it never calls `eval()` and never executes arbitrary Python.

```python
from adapt_agent import PolicyEnforcer

policy = PolicyEnforcer(max_violations=1000)
```

---

## Adding rules

```python
policy.add_rule(
    name="no_passwords",
    description="Block messages that mention a password",
    condition="'password' in message['content']",
    action="block",      # one of: warn (default), block, modify, require_approval
    severity="high",     # one of: low, medium, high, critical
)
```

Both sets are closed and spelled exactly (`VALID_ACTIONS` /
`VALID_SEVERITIES` in the `adapt_agent.core.policy` module). `add_rule` raises
`ValueError`, naming the valid values, for anything else -- `"Block"`, `"deny"`
and `"blok"` used to be accepted and then enforced nothing, because the gate
compares `action == "block"`. `register_handler` validates its action the same
way, since a handler for an unknown action could never run.

`add_rule` enforces a **1024-character limit** on the condition string and raises
`ValueError` if exceeded (CPU-exhaustion DoS protection — long strings are
expensive to parse into ASTs), and a limit of **200 AST nodes** on the parsed
condition:

```python
policy.add_rule("big", "too long", condition="x" * 2000)
# ValueError: Condition length 2000 exceeds maximum allowed length of 1024
```

Other rule management:

```python
policy.get_rule("no_passwords")     # -> PolicyRule | None
policy.list_rules()                 # -> list[PolicyRule]
policy.remove_rule("no_passwords")  # -> bool
```

---

## Checking messages and state

`check_message` evaluates every rule with the variable `message` in scope.
`check_state` evaluates every rule with the variable `state` in scope. Both
return the list of **violated** rule names (a rule "fires" when its condition
evaluates truthy), record the violation, and run any registered handler.

```python
# message context: {"message": <the AgentMessage>}
policy.check_message({"role": "user", "content": "my password is hunter2"})
# -> ['no_passwords']

# state context: {"state": <the AgentState>} -- on its own enforcer; see
# "One enforcer per scope" below.
policy = PolicyEnforcer()
policy.add_rule(
    name="context_present",
    description="Fire when context has any keys",
    condition="state['context'] != {}",
    action="warn",
)
```

!!! warning
    **Function calls are not part of the expression language** — there is no
    `len(...)` or `str(...)`. `add_rule` rejects a condition it could never
    evaluate with a `ValueError`, so the mistake surfaces when the rule is
    registered rather than as a rule that silently never fires:

    ```python
    policy.add_rule("long", "too long", condition="len(state['context']) > 0")
    # ValueError: Unsupported construct in condition: Call (...)
    policy.add_rule("typo", "typo", condition="stat['trust_score'] < 0.5")
    # ValueError: Unknown name 'stat' in condition; only ['message', 'state', 'tool'] are available
    ```

---

## The safe condition language

Conditions are parsed with `ast.parse(condition, mode="eval")` (cached) and
walked by `PolicyEnforcer._eval_node`. Only the following node types are
supported:

### Supported

| Category | Supported forms |
|----------|-----------------|
| Literals | numbers, strings, `True`/`False`/`None` (`ast.Constant`) |
| Names | variables in the evaluation context (`message`, `state`, and `tool` for [tool-scoped rules](#tool-scoped-rules-and-check_tool)) |
| Collections | list `[...]`, tuple `(...)`, set `{...}`, dict `{k: v}` |
| Comparisons | `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in` (chained allowed) |
| Boolean ops | `and`, `or` |
| Arithmetic | `+`, `-`, `/` (binary); `*` between **numbers** only; negative numeric literals |
| Subscripts | `obj[key]` / `obj[index]` (single index only) |

Chained comparisons work as in Python (`0 < x < 10`). Subscript failures
(`KeyError`/`IndexError`/`TypeError`) return `None` rather than raising.

`*` refuses sequences (`'a' * 400000000`, `[0] * n`): repetition allocates
without bound, so a condition could exhaust memory on one check. It is an
evaluation error, so the usual fail-closed rule applies. `**` is not available
at all.

A **non-finite number is never compared**: `NaN > 100` is `False`, which would
make a threshold rule a quiet pass for exactly the value that should worry you,
so a comparison with `NaN` or an infinity is an evaluation error (fail-closed
under the default).

### Not supported (rejected)

- **Function calls** — `len(...)`, `str(...)`, `any(...)`, etc.
- **Attribute access** — `message.content`, `obj.attr`.
- **Slices** — `seq[1:2]` (`"Slices are not supported in conditions"`).
- **Unary `not` / `~` / unary minus on an expression / bitwise / power / modulo**
  and any other operator not listed. (A negative *literal* such as `-1` or
  `-0.5` is supported, including as an index: `state['messages'][-1]`.)
- **Unknown names** — any name other than `message`, `state` and `tool`
  (`"Unknown name 'x' in condition; ..."`).

Every item above is detectable from the condition text alone, so `add_rule`
(and `adapt validate`) rejects it with a `ValueError` at registration. What
registration **cannot** see is your data: `state['n'] > 5` is well-formed but
cannot be evaluated when `n` is missing. That case is decided at run time by
`fail_closed`:

| | A rule that cannot be evaluated is… |
| --- | --- |
| `PolicyEnforcer()` — **default, fail-closed** | logged and treated as a **violation**. A `block` rule refuses the request. |
| `PolicyEnforcer(fail_closed=False)` | logged and treated as *no violation*. The rule silently does nothing. |

The default is fail-closed because a control that could not reach a verdict
should not report that all is well. The cost is that it is loud: a `block` rule
that reads a key your state never carries refuses **every** request until you
supply the key or fix the rule. That is the intended signal — the alternative
was a rule that looked installed and never fired.

!!! warning "One enforcer per scope"
    `check_state` puts only `state` in scope and `check_message` only
    `message`. A rule naming the other one cannot be evaluated, so under the
    default it is reported as violated on every call. Keep `message` rules and
    `state` rules on **separate** `PolicyEnforcer` instances. The governance
    gate (every adapter and framework binding) calls only `check_state` and
    `check_tool`, so the enforcer you give it should hold `state` rules and
    [tool-scoped rules](#tool-scoped-rules-and-check_tool) only.

!!! note "Changed in 0.5.0"
    The default was `fail_closed=False`. Pass it explicitly to keep the old
    behaviour while you check your rules with `vacuous_rules()` (below).

### Checking a rule is not vacuous

A rule that can never fire is worse than no rule, because it looks installed.
`evaluate_rules` dry-runs every rule against an example — nothing is recorded
and no handler runs — and reports `"violation"`, `"pass"` or `"error"` per rule:

```python
policy.evaluate_rules(state={"messages": [], "context": {}, "trust_score": 0.2})
# {'low_trust': 'violation'}
policy.evaluate_rules(state={"messages": [], "context": {}})
# {'low_trust': 'error'}   <- trust_score missing: a violation by default (fail-closed)
```

For each blocking rule, assert in your own tests that one realistic example
gives `"violation"` and another gives `"pass"`.

### Finding a rule that has never seen its data

`evaluate_rules` checks the examples you thought of. For the traffic you
actually get, the enforcer counts outcomes per rule:

```python
policy.get_rule_stats()
# {'low_trust': {'violation': 3, 'pass': 41, 'error': 0, 'unresolved': 0},
#  'free_tier': {'violation': 0, 'pass': 44, 'error': 0, 'unresolved': 44}}

policy.vacuous_rules()
# ['free_tier']
```

`unresolved` counts the evaluations in which the rule could not see everything
it asked for: it errored, **or** it looked up a key that was not there. The
second case matters because it does not error. A failed `state[...]` lookup
evaluates to `None`, so `state['tier'] == 'free'` is a quiet `"pass"` when
`tier` is absent — which is the right answer for one request and the wrong one
if `tier` is *never* there because of a typo, or because the adapter does not
populate it.

`vacuous_rules()` lists the rules for which **every** evaluation so far was
unresolved. Check it after driving a representative request through your agent
in a test, or periodically in production. A rule that has not been evaluated at
all is not listed.

[`examples/09_verifying_your_guardrails.py`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/examples/09_verifying_your_guardrails.py)
runs all of this end to end, offline.

### `and` / `or`

These behave as in Python: they yield one of their operands, not a bool. So
`(state['override'] or state['default']) > 5` compares the value, and a
comparison Python would refuse (`0 < None`) is an evaluation error here too
rather than a silent verdict.

### Depth limit

Evaluation is bounded to a recursion depth of **50**. A condition deeper than
that could only ever error, so `add_rule` / `validate_condition` refuse it at
registration; the run-time check (`"Maximum evaluation depth exceeded"`) remains
as a backstop.

### Examples that work

```python
"'password' in message['content']"
"message['role'] == 'system'"
"message['content'] in ['', 'ping']"
"state['trust_score'] < 0.3"
"'admin' in state['context'] and state['context']['admin'] == True"
"message['content'] == 'a' + 'b'"
```

### Examples that are rejected by `add_rule`

```python
"len(message['content']) > 100"   # function call
"message.content == 'x'"          # attribute access
"message['content'][0:5] == 'oops'"  # slice
```

---

## Actions and handlers

`add_rule(action=...)` records the action on the rule. The enforcer ships with no
default behavior for actions beyond recording the violation — you attach behavior
by registering a handler per action via `register_handler`. When a rule fires,
its action's handler (if registered) is invoked with the rule dict.

```python
def on_block(rule):
    print(f"BLOCKED by rule {rule['name']} (severity={rule['severity']})")

policy.register_handler("block", on_block)

policy.check_message({"role": "user", "content": "password: hunter2"})
# prints: BLOCKED by rule no_passwords (severity=high)
```

The governance gate behind every adapter and framework binding inspects rule
actions directly: a fired state rule whose `action == "block"` refuses the
request (a `SecurityBlockedError`, reported with the threat label
`policy:<rule name>`) when `block_on_violation=True`; `warn` and `modify` rules
are recorded but do not block. See [Framework Adapters](adapters.md).

**A handler cannot weaken a verdict.** The violation is counted and recorded
*before* the handler runs. If the handler raises, the failure is logged
(`logger.exception`), the remaining rules are still evaluated, and the violation
stands -- so a buggy handler cannot hide a later `block` rule or turn a
violation into a pass. The one exception is `SecurityBlockedError`: a handler
may raise that on purpose to refuse the request, and it propagates.

---

## Tool-scoped rules and `check_tool`

A rule whose condition names `tool` is **tool-scoped**. It is evaluated only by
`check_tool`, and skipped by `check_state` / `check_message`, so tool rules can
share an enforcer with ordinary ones without tripping the fail-closed "name not
in scope" violation. `tool` is `{"name": str, "arguments": ...}`, and `state` is
also in scope when the caller passes one:

```python
policy.add_rule(
    "no_wire_when_untrusted", "Wire transfers need a trusted session",
    "tool['name'] == 'wire_transfer' and state['trust_score'] < 0.8", action="block",
)
policy.add_rule(
    "approve_large_refunds", "A human signs off refunds over 500",
    "tool['name'] == 'refund' and tool['arguments']['amount'] > 500",
    action="require_approval",
)
violated = policy.check_tool({"name": "refund", "arguments": {"amount": 900}}, state=session)
```

`check_tool` returns the violated rule names; look each up with `get_rule` to act
on its `action`. `require_approval` asks for a human decision instead of
refusing: a framework that can pause a tool call surfaces it, and
`ToolDecision.resolve(approvals_supported=False)` turns it into a block. It is
**only valid on a tool-scoped rule** -- a message or state that already arrived
cannot be paused, so `add_rule` raises `ValueError` rather than install a rule
nothing could act on. Everything above applies unchanged: an unevaluable tool
rule (say it reads `state[...]` and the caller passed no state) is a violation
under the default fail-closed setting.

---

## Inspecting violations

Violations are recorded with the rule name, type (`"message"`, `"state"` or
`"tool"`), severity, timestamp, and a **bounded summary** of the offending data.
The store is bounded by `max_violations` (oldest dropped first).

The summary is not the object. It holds the type, the size, up to 20 key names
(each cut to 64 characters) and the `role` / `name` string if there is one --
never the values. The live state keeps changing after a violation, and keeping a
reference to it would both show later conversation turns under an earlier
violation and pin arbitrary user content in memory.

```python
policy.get_violations()                              # all
policy.get_violations(severity="high")               # filter by severity
policy.get_violations(severity="high", limit=10)     # most recent 10 high
```

---

## Concurrency

`check_state`, `check_message` and `check_tool` evaluate a snapshot of the rule
set, so rules may be added or removed from another thread while checks run, and
the per-rule counters (`get_rule_stats`, `vacuous_rules`) are updated under a
lock.

---

## Trust and taint guarantees

These are the companions the policy engine is normally run with.

**`TrustManager`**

- `update_trust_score` rejects a non-finite or non-numeric delta with
  `ValueError` (`min(1.0, nan)` is `1.0`: a NaN delta would pin any agent at full
  trust).
- Eviction cannot launder distrust. An agent scoring below
  `max(distrust_floor, initial_trust)` is never evicted while another entry could
  be, because eviction resets an agent to `initial_trust` -- a trust *increase*.
  Churning throwaway ids therefore cannot flush a distrusted agent. The memory
  cap is still a hard bound: if every slot holds a distrusted agent, the
  least-distrusted one is dropped and a warning is logged.

**`TaintTracker`** (fail-closed, and levels only go up)

- A tainted item evicted from the bounded data map is not forgotten: its level is
  kept in a second bounded map, so it still reads as tainted at that level, and
  taint propagated from it carries the same floor.
- If that map overflows too, the tracker is *saturated*
  (`get_stats()["saturated"]`): any id it has no record of reads as tainted at the
  highest level it had to forget. This is the trade-off -- a flood of junk ids can
  degrade a long-lived tracker to "unknown means tainted", an availability cost,
  but can no longer make real taint read clean. `reset()` clears it.
- `mark_tainted(x, [])` is a no-op, so empty marks cannot push real taint out.
- `register_source` raises `ValueError` if the id is already registered at a
  higher level (or was evicted while referenced at a higher level): re-registering
  at `LOW` would silently downgrade everything it had tainted.

`MemorySystem` and `PatchManager` take a lock around their shared state; a
re-registered patch id is no longer reported as already applied.
