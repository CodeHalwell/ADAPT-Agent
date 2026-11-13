"""Policy enforcement for LLM agents."""

import ast
import logging
import operator
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, cast

from adapt_agent.core.types import AgentMessage, AgentState, PolicyRule

logger = logging.getLogger(__name__)

# Maximum number of AST nodes allowed in a single condition. This bounds CPU on
# pathological-but-shallow conditions (e.g. a flat 250-element literal) that the
# depth cap alone would not catch.
MAX_CONDITION_NODES = 200


class ConditionEvaluationError(Exception):
    """Raised when a policy condition cannot be evaluated.

    This is deliberately distinct from a condition simply evaluating to ``False``
    so the enforcer can tell "the rule did not match" apart from "the rule could
    not be checked" and apply its fail-open/fail-closed policy accordingly.
    """


def _count_nodes(tree: ast.AST) -> int:
    """Count the total number of AST nodes in ``tree``."""
    return sum(1 for _ in ast.walk(tree))


@lru_cache(maxsize=1024)
def _parse_condition(condition: str) -> ast.Expression:
    """Cache AST parsing for policy conditions.

    Raises:
        ValueError: if the parsed condition exceeds ``MAX_CONDITION_NODES`` nodes.
    """
    tree = ast.parse(condition, mode="eval")
    node_count = _count_nodes(tree)
    if node_count > MAX_CONDITION_NODES:
        raise ValueError(
            f"Condition node count {node_count} exceeds maximum allowed "
            f"{MAX_CONDITION_NODES} (DoS protection)"
        )
    return tree


#: Names a condition may reference. ``check_state`` binds ``state`` and
#: ``check_message`` binds ``message``; nothing else is ever in scope.
CONDITION_NAMES = frozenset({"state", "message"})

_SUPPORTED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.Compare,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.BinOp,
    ast.Subscript,
)


def validate_condition(condition: str) -> None:
    """Reject a condition the evaluator could never evaluate.

    A rule whose condition always errors is *vacuous*: under the default
    fail-closed setting it fires on everything, and under ``fail_closed=False``
    it never fires. Neither is what its author meant, and both used to be
    discovered only from a warning in the logs. Everything checked here is
    decidable from the text alone, so it is checked once, at registration.

    What this cannot see is the *data*: ``state["n"] > 5`` is well-formed but
    still errors when ``n`` is absent. Use
    :meth:`PolicyEnforcer.evaluate_rules` with a representative state for that.

    Raises:
        ValueError: on a syntax error, an oversized condition, an unsupported
            construct (calls, attribute access, ``not``, slices, ...), an
            unsupported operator, or a name other than ``state``/``message``.
    """
    try:
        tree = _parse_condition(condition)
    except SyntaxError as exc:
        raise ValueError(f"Condition is not a valid expression: {exc.msg}") from exc
    for node in ast.walk(tree):
        if isinstance(node, _SUPPORTED_NODES):
            if isinstance(node, ast.Name) and node.id not in CONDITION_NAMES:
                raise ValueError(
                    f"Unknown name {node.id!r} in condition; only "
                    f"{sorted(CONDITION_NAMES)} are available"
                )
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
                raise ValueError("Slices are not supported in conditions")
        elif isinstance(node, ast.cmpop):
            if type(node) not in PolicyEnforcer._OPERATORS:
                raise ValueError(f"Unsupported comparison in condition: {type(node).__name__}")
        elif isinstance(node, ast.operator):
            if type(node) not in PolicyEnforcer._BINOPS:
                raise ValueError(f"Unsupported operator in condition: {type(node).__name__}")
        else:
            raise ValueError(
                f"Unsupported construct in condition: {type(node).__name__} "
                "(conditions allow literals, state[...]/message[...] lookups, "
                "comparisons, and/or, and + - * /)"
            )


#: Context key under which a failed ``state[...]`` lookup is noted during one
#: rule's evaluation. An object, not a string, so no condition can name it.
_MISSED_LOOKUP: Any = object()


class PolicyEnforcer:
    """Enforces policies and rules on LLM agent behavior.

    The PolicyEnforcer validates agent actions, messages, and state changes
    against defined policies and can take corrective actions when violations occur.
    """

    _OPERATORS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
    }

    _BINOPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def __init__(self, max_violations: int = 1000, fail_closed: bool = True):
        """Initialize the PolicyEnforcer.

        Args:
            max_violations: Maximum number of violations to store in memory.
            fail_closed: How to treat a condition that cannot be evaluated
                at run time -- it compares against a key the state does not
                carry, names ``message`` where only ``state`` is in scope, hits
                an arithmetic error, ...
                When ``True`` (default) the error is logged and treated as a
                VIOLATION, so a ``block`` rule still fires on malformed or
                unexpected input (fail-closed): a control that could not reach
                a verdict does not report that all is well.
                When ``False`` the error is logged and treated as "no
                violation" (fail-open) -- the default up to 0.4.0, and the one
                under which a mistyped or mis-scoped rule silently does
                nothing. In either case the error is always logged.

                Because an out-of-scope name is an evaluation error, keep
                ``message`` rules and ``state`` rules on **separate**
                enforcers: under fail-closed, ``check_state`` reports every
                ``message`` rule as violated, and vice versa.
        """
        self.max_violations = max_violations
        self.fail_closed = fail_closed
        self._rules: dict[str, PolicyRule] = {}
        # ⚡ Bolt: Replace O(N) list.pop(0) ring buffer with O(1) collections.deque
        self._violations: deque[dict[str, Any]] = deque(maxlen=max_violations)
        # Per-rule outcome counts, so a rule that has only ever *errored* can be
        # found. Bounded by the number of rules (entries die with their rule).
        self._rule_stats: dict[str, dict[str, int]] = {}
        self._rule_handlers: dict[str, Callable] = {}

    def add_rule(
        self,
        name: str,
        description: str,
        condition: str,
        action: str = "warn",
        severity: str = "medium",
    ) -> None:
        """Add a policy rule.

        Args:
            name: Unique name for the rule
            description: Human-readable description
            condition: Condition expression to evaluate
            action: Action to take on violation (warn, block, modify)
            severity: Severity level (low, medium, high, critical)
        """
        # SECURITY: Prevent CPU exhaustion DoS attacks by limiting the length of conditions parsed into ASTs
        if len(condition) > 1024:
            raise ValueError(
                f"Condition length {len(condition)} exceeds maximum allowed length of 1024"
            )

        # SECURITY: Reject, up front, any condition the evaluator could never
        # evaluate -- broken syntax, too many AST nodes (CPU bound), an
        # unsupported construct such as a call or attribute access, or a name
        # other than ``state``/``message``. Such a rule is vacuous: it would
        # never fire (fail-open) or fire on everything (fail-closed).
        validate_condition(condition)

        rule: PolicyRule = {
            "name": name,
            "description": description,
            "condition": condition,
            "action": action,
            "severity": severity,
        }
        self._rules[name] = rule
        # A re-registered rule is a new rule: old counts would describe the
        # condition it replaced.
        self._rule_stats[name] = {"violation": 0, "pass": 0, "error": 0, "unresolved": 0}

    def remove_rule(self, name: str) -> bool:
        """Remove a policy rule.

        Args:
            name: Name of the rule to remove

        Returns:
            True if rule was removed, False if not found
        """
        if name in self._rules:
            del self._rules[name]
            self._rule_stats.pop(name, None)
            return True
        return False

    def get_rule(self, name: str) -> PolicyRule | None:
        """Get a policy rule by name.

        Args:
            name: Name of the rule

        Returns:
            PolicyRule if found, None otherwise
        """
        return self._rules.get(name)

    def list_rules(self) -> list[PolicyRule]:
        """List all policy rules.

        Returns:
            List of all registered policy rules
        """
        return list(self._rules.values())

    def register_handler(self, action: str, handler: Callable) -> None:
        """Register a handler for a policy action.

        Args:
            action: Action type (e.g., 'warn', 'block')
            handler: Callable to handle the action
        """
        self._rule_handlers[action] = handler

    def check_message(self, message: AgentMessage) -> list[str]:
        """Check a message against all policy rules.

        Args:
            message: Message to check

        Returns:
            List of violated rule names
        """
        violations = []
        # ⚡ Bolt: Hoist context instantiation outside the loop to avoid redundant object creation
        context = {"message": message}

        for rule_name, rule in self._rules.items():
            if self._check_rule(rule_name, rule, context):
                violations.append(rule_name)
                self._record_violation(rule_name, "message", message)
                self._handle_violation(rule)

        return violations

    def check_state(self, state: AgentState) -> list[str]:
        """Check agent state against all policy rules.

        Args:
            state: Agent state to check

        Returns:
            List of violated rule names
        """
        violations = []
        # ⚡ Bolt: Hoist context instantiation outside the loop to avoid redundant object creation
        context = {"state": state}

        for rule_name, rule in self._rules.items():
            if self._check_rule(rule_name, rule, context):
                violations.append(rule_name)
                self._record_violation(rule_name, "state", state)
                self._handle_violation(rule)

        return violations

    def evaluate_rules(
        self,
        *,
        state: AgentState | None = None,
        message: AgentMessage | None = None,
    ) -> dict[str, str]:
        """Dry-run every rule against an example, without side effects.

        Returns ``{rule_name: outcome}`` where outcome is ``"violation"``,
        ``"pass"`` or ``"error"`` (the condition could not be evaluated against
        this example -- typically a missing key). Nothing is recorded and no
        handler runs, so this is safe to call from a test or a startup check.

        Use it to show a rule is not vacuous: a rule you expect to block should
        report ``"violation"`` for at least one example and ``"pass"`` for at
        least one other. A rule that reports ``"error"`` for a realistic
        example is one that blocks that example under the default
        fail-closed setting, and silently does nothing under
        ``fail_closed=False``.
        """
        context: dict[str, Any] = {}
        if state is not None:
            context["state"] = state
        if message is not None:
            context["message"] = message
        outcomes: dict[str, str] = {}
        for rule_name, rule in self._rules.items():
            try:
                tree = _parse_condition(rule["condition"])
                fired = bool(self._eval_node(tree.body, context))
            except Exception:
                outcomes[rule_name] = "error"
            else:
                outcomes[rule_name] = "violation" if fired else "pass"
        return outcomes

    def get_violations(
        self,
        severity: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get recorded policy violations.

        Args:
            severity: Filter by severity level
            limit: Maximum number of violations to return

        Returns:
            List of violation records
        """
        violations = self._violations

        if severity:
            # ⚡ Bolt: Fast path for finding limited recent violations with a specific severity
            if limit:
                results = []
                for v in reversed(violations):
                    if v["severity"] == severity:
                        results.append(v)
                        if len(results) >= limit:
                            break
                results.reverse()
                return results
            return [v for v in violations if v["severity"] == severity]

        if limit:
            return list(violations)[-limit:]

        return list(violations)

    def _evaluate_condition(
        self,
        condition: str,
        context: dict[str, Any],
        rule_name: str | None = None,
    ) -> bool:
        """Evaluate a policy condition.

        Distinguishes a condition that evaluates to ``False`` (no violation) from
        a condition that cannot be evaluated at all (unknown variable, unsupported
        node, arithmetic error, ...). An evaluation error is always logged as a
        warning; it is then treated as a violation when ``self.fail_closed`` is set
        (the default) and as "no violation" otherwise (fail-open).

        Args:
            condition: Condition expression
            context: Context for evaluation
            rule_name: Name of the owning rule, for clearer log messages

        Returns:
            True if condition is met (violation), False otherwise. On an
            evaluation error returns ``self.fail_closed``.
        """
        outcome = self._outcome(condition, context, rule_name)
        if outcome == "error":
            return self.fail_closed
        return outcome == "violation"

    def _outcome(
        self,
        condition: str,
        context: dict[str, Any],
        rule_name: str | None = None,
    ) -> str:
        """Evaluate a condition to ``"violation"``, ``"pass"`` or ``"error"``.

        The three-way answer :meth:`_evaluate_condition` collapses to a bool.
        Kept separate so the enforcer can count *errors* per rule: under
        fail-open an error and a pass are the same ``False`` to the caller, and
        a rule that only ever errors is indistinguishable from one that is
        simply never violated.
        """
        try:
            tree = _parse_condition(condition)
            return "violation" if self._eval_node(tree.body, context) else "pass"
        except Exception as e:
            label = f" for rule '{rule_name}'" if rule_name else ""
            logger.warning(
                "Error evaluating policy condition%s '%s': %s — treating as %s",
                label,
                condition,
                e,
                "VIOLATION (fail_closed)" if self.fail_closed else "no violation (fail_open)",
            )
            return "error"

    def _check_rule(self, rule_name: str, rule: PolicyRule, context: dict[str, Any]) -> bool:
        """Evaluate one rule for real: count the outcome, return "violated?"."""
        context.pop(_MISSED_LOOKUP, None)
        outcome = self._outcome(rule["condition"], context, rule_name)
        missed = context.pop(_MISSED_LOOKUP, False)
        stats = self._rule_stats.setdefault(
            rule_name, {"violation": 0, "pass": 0, "error": 0, "unresolved": 0}
        )
        stats[outcome] += 1
        if outcome == "error" or missed:
            stats["unresolved"] += 1
        if outcome == "error":
            return self.fail_closed
        return outcome == "violation"

    def get_rule_stats(self) -> dict[str, dict[str, int]]:
        """Per-rule counts of ``violation`` / ``pass`` / ``error`` outcomes.

        ``unresolved`` additionally counts the evaluations -- of any outcome --
        in which the rule could not see everything it asked for: it errored, or
        a ``state[...]`` lookup found no such key (which evaluates to ``None``
        rather than erroring, so ``state['tier'] == 'free'`` is a quiet
        ``pass`` when ``tier`` is absent).

        Counts cover :meth:`check_state` and :meth:`check_message` calls since
        the rule was registered (:meth:`evaluate_rules` is a dry run and is not
        counted).
        """
        return {name: dict(stats) for name, stats in self._rule_stats.items()}

    def vacuous_rules(self) -> list[str]:
        """Rules that have been evaluated and have **never once** seen their data.

        In every evaluation so far such a rule either errored or looked up a
        key that was not there -- typically because it reads a key the state
        never carries (a typo, or data the adapter does not populate), or names
        ``message`` where only ``state`` is ever in scope (as under an
        adapter). It has decided nothing on evidence since it was installed:
        it has blocked everything it errored on, or under
        ``fail_closed=False`` silently passed.

        A rule that has not been evaluated at all is not listed: there is no
        evidence either way. Check this after real traffic, or in a test after
        driving a representative request through the agent.
        """
        return [
            name
            for name, stats in self._rule_stats.items()
            if stats["unresolved"] > 0
            and stats["unresolved"] == stats["violation"] + stats["pass"] + stats["error"]
        ]

    def _eval_node(self, node: ast.AST, context: dict[str, Any], depth: int = 0) -> Any:
        if depth > 50:
            raise ValueError("Maximum evaluation depth exceeded (DoS protection)")
        # NOTE: ``type(node) is X`` dispatch is intentionally used instead of
        # ``isinstance`` for speed in this hot path. mypy cannot narrow on
        # identity checks, so each branch uses ``cast`` (a zero-cost, runtime
        # no-op) to access the concrete node's attributes.
        node_type = type(node)
        if node_type is ast.Constant:
            return cast(ast.Constant, node).value
        elif node_type is ast.Name:
            name_node = cast(ast.Name, node)
            if name_node.id in context:
                return context[name_node.id]
            raise ValueError(f"Unknown variable: {name_node.id}")
        elif node_type is ast.List:
            return [self._eval_node(elt, context, depth + 1) for elt in cast(ast.List, node).elts]
        elif node_type is ast.Tuple:
            return tuple(
                self._eval_node(elt, context, depth + 1) for elt in cast(ast.Tuple, node).elts
            )
        elif node_type is ast.Set:
            return {self._eval_node(elt, context, depth + 1) for elt in cast(ast.Set, node).elts}
        elif node_type is ast.Dict:
            dict_node = cast(ast.Dict, node)
            return {
                self._eval_node(k, context, depth + 1): self._eval_node(v, context, depth + 1)
                for k, v in zip(dict_node.keys, dict_node.values, strict=False)
                if k is not None
            }
        elif node_type is ast.Compare:
            cmp_node = cast(ast.Compare, node)
            left = self._eval_node(cmp_node.left, context, depth + 1)
            for op, comp in zip(cmp_node.ops, cmp_node.comparators, strict=False):
                right = self._eval_node(comp, context, depth + 1)
                op_type = type(op)
                if op_type not in self._OPERATORS:
                    raise ValueError(f"Unsupported operator: {op_type}")
                if not self._OPERATORS[op_type](left, right):
                    return False
                left = right
            return True
        elif node_type is ast.BoolOp:
            bool_node = cast(ast.BoolOp, node)
            bool_op_type = type(bool_node.op)
            # Python semantics: `and`/`or` yield one of their *operands*, not a
            # bool. Returning True/False here made `(0 or state['x']) < 1` mean
            # something different in a rule than in Python -- and, worse, made
            # a comparison Python refuses (`0 < None`) quietly succeed.
            result: Any = None
            if bool_op_type is ast.And:
                for value in bool_node.values:
                    result = self._eval_node(value, context, depth + 1)
                    if not result:
                        return result
                return result
            elif bool_op_type is ast.Or:
                for value in bool_node.values:
                    result = self._eval_node(value, context, depth + 1)
                    if result:
                        return result
                return result
            raise ValueError(f"Unsupported boolean operator: {bool_op_type}")
        elif node_type is ast.BinOp:
            bin_node = cast(ast.BinOp, node)
            left = self._eval_node(bin_node.left, context, depth + 1)
            right = self._eval_node(bin_node.right, context, depth + 1)
            bin_op_type = type(bin_node.op)
            if bin_op_type not in self._BINOPS:
                raise ValueError(f"Unsupported binary operator: {bin_op_type}")
            return self._BINOPS[bin_op_type](left, right)
        elif node_type is ast.Subscript:
            sub_node = cast(ast.Subscript, node)
            val = self._eval_node(sub_node.value, context, depth + 1)
            slice_type = type(sub_node.slice)
            if slice_type is ast.Slice:
                raise ValueError("Slices are not supported")
            slice_val = self._eval_node(sub_node.slice, context, depth + 1)
            try:
                return val[slice_val]
            except (KeyError, IndexError, TypeError):
                # Deliberately not an error: `state['flag'] == True` should be
                # a quiet "no" when the flag is absent. But it is noted, so a
                # rule whose lookups have *never* resolved can be reported.
                context[_MISSED_LOOKUP] = True
                return None
        else:
            raise ValueError(f"Unsupported AST node: {node_type}")

    def _record_violation(
        self,
        rule_name: str,
        violation_type: str,
        data: Any,
    ) -> None:
        """Record a policy violation.

        Args:
            rule_name: Name of violated rule
            violation_type: Type of violation
            data: Associated data
        """
        rule = self._rules[rule_name]
        violation = {
            "rule_name": rule_name,
            "violation_type": violation_type,
            "severity": rule["severity"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        self._violations.append(violation)

    def _handle_violation(self, rule: PolicyRule) -> None:
        """Handle a policy violation.

        Args:
            rule: The violated rule
        """
        action = rule["action"]
        if action in self._rule_handlers:
            self._rule_handlers[action](rule)
