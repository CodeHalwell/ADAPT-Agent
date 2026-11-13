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
    action="block",      # one of: warn (default), block, modify
    severity="high",     # one of: low, medium, high, critical
)
```

`add_rule` enforces a **1024-character limit** on the condition string and raises
`ValueError` if exceeded (CPU-exhaustion DoS protection — long strings are
expensive to parse into ASTs):

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
    # ValueError: Unknown name 'stat' in condition; only ['message', 'state'] are available
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
| Names | variables in the evaluation context (`message`, `state`) |
| Collections | list `[...]`, tuple `(...)`, set `{...}`, dict `{k: v}` |
| Comparisons | `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in` (chained allowed) |
| Boolean ops | `and`, `or` |
| Arithmetic | `+`, `-`, `*`, `/` (binary) |
| Subscripts | `obj[key]` / `obj[index]` (single index only) |

Chained comparisons work as in Python (`0 < x < 10`). Subscript failures
(`KeyError`/`IndexError`/`TypeError`) return `None` rather than raising.

### Not supported (rejected)

- **Function calls** — `len(...)`, `str(...)`, `any(...)`, etc.
- **Attribute access** — `message.content`, `obj.attr`.
- **Slices** — `seq[1:2]` raises `"Slices are not supported"`.
- **Unary not / bitwise / power / modulo** and any other operator not listed.
- **Unknown variables** — referencing a name not in the context raises
  `"Unknown variable: ..."`.

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
    `state` rules on **separate** `PolicyEnforcer` instances. A governed
    adapter only ever calls `check_state`, so the enforcer you give it should
    hold `state` rules only.

!!! note "Changed after 0.4.0"
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

### `and` / `or`

These behave as in Python: they yield one of their operands, not a bool. So
`(state['override'] or state['default']) > 5` compares the value, and a
comparison Python would refuse (`0 < None`) is an evaluation error here too
rather than a silent verdict.

### Depth limit

Evaluation is bounded to a recursion depth of **50**; deeper expressions raise
`"Maximum evaluation depth exceeded (DoS protection)"`, also caught and treated
as no violation.

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

The LangGraph adapter inspects rule actions directly: a fired rule whose
`action == "block"` causes a `SecurityBlockedError` when
`block_on_violation=True`. See [Framework Adapters](adapters.md).

---

## Inspecting violations

Violations are recorded with the rule name, type (`"message"` or `"state"`),
severity, timestamp, and the offending data. The store is bounded by
`max_violations` (oldest dropped first).

```python
policy.get_violations()                              # all
policy.get_violations(severity="high")               # filter by severity
policy.get_violations(severity="high", limit=10)     # most recent 10 high
```
