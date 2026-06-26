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

# state context: {"state": <the AgentState>}
policy.add_rule(
    name="empty_context_ok",
    description="Fire when context has any keys",
    condition="len(state['context']) > 0",  # NOTE: len() is NOT supported — see below
    action="warn",
)
```

!!! warning
    The two `len(...)` examples on this page are intentionally used to illustrate
    a common mistake: **function calls are not part of the expression language.**
    A condition that uses an unsupported construct raises during evaluation, is
    caught, logged, and treated as "no violation" (returns `False`). Rewrite such
    rules using only the supported nodes listed below.

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

Any unsupported node raises `ValueError("Unsupported AST node: ...")`, which is
caught by `_evaluate_condition`, logged, and yields `False` (no violation).

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

### Examples that are rejected (fire-as-False)

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
