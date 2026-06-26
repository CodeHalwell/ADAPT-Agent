"""Policy enforcement for LLM agents."""

import ast
import logging
import operator
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

    def __init__(self, max_violations: int = 1000, fail_closed: bool = False):
        """Initialize the PolicyEnforcer.

        Args:
            max_violations: Maximum number of violations to store in memory.
            fail_closed: How to treat a condition that cannot be evaluated
                (unknown variable, unsupported node, arithmetic error, ...).
                When ``False`` (default) such an error is logged and treated as
                "no violation" (fail-open) — preserving historical behaviour.
                When ``True`` the error is logged and treated as a VIOLATION so
                that ``block`` rules still fire on malformed or edge-case input
                (fail-closed). In either case the error is always logged.
        """
        self.max_violations = max_violations
        self.fail_closed = fail_closed
        self._rules: dict[str, PolicyRule] = {}
        self._violations: list[dict[str, Any]] = []
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

        # SECURITY: Bound CPU on pathological-but-shallow conditions (e.g. a flat
        # 250-element literal) by rejecting conditions with too many AST nodes at
        # registration time. ``_parse_condition`` raises ValueError when the node
        # count exceeds ``MAX_CONDITION_NODES``; it also surfaces syntax errors so
        # rules with broken syntax are rejected up front rather than silently.
        _parse_condition(condition)

        rule: PolicyRule = {
            "name": name,
            "description": description,
            "condition": condition,
            "action": action,
            "severity": severity,
        }
        self._rules[name] = rule

    def remove_rule(self, name: str) -> bool:
        """Remove a policy rule.

        Args:
            name: Name of the rule to remove

        Returns:
            True if rule was removed, False if not found
        """
        if name in self._rules:
            del self._rules[name]
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
            # Simple condition checking (can be extended with more sophisticated evaluation)
            if self._evaluate_condition(rule["condition"], context, rule_name):
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
            if self._evaluate_condition(rule["condition"], context, rule_name):
                violations.append(rule_name)
                self._record_violation(rule_name, "state", state)
                self._handle_violation(rule)

        return violations

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
            return violations[-limit:]

        return violations

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
        and as "no violation" otherwise (fail-open, the historical behaviour).

        Args:
            condition: Condition expression
            context: Context for evaluation
            rule_name: Name of the owning rule, for clearer log messages

        Returns:
            True if condition is met (violation), False otherwise. On an
            evaluation error returns ``self.fail_closed``.
        """
        try:
            tree = _parse_condition(condition)
            return bool(self._eval_node(tree.body, context))
        except Exception as e:
            label = f" for rule '{rule_name}'" if rule_name else ""
            logger.warning(
                "Error evaluating policy condition%s '%s': %s — treating as %s",
                label,
                condition,
                e,
                "VIOLATION (fail_closed)" if self.fail_closed else "no violation (fail_open)",
            )
            return self.fail_closed

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
            if bool_op_type is ast.And:
                for value in bool_node.values:
                    if not self._eval_node(value, context, depth + 1):
                        return False
                return True
            elif bool_op_type is ast.Or:
                for value in bool_node.values:
                    if self._eval_node(value, context, depth + 1):
                        return True
                return False
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

        # SECURITY: Prevent unbounded memory growth
        if len(self._violations) > self.max_violations:
            self._violations.pop(0)

    def _handle_violation(self, rule: PolicyRule) -> None:
        """Handle a policy violation.

        Args:
            rule: The violated rule
        """
        action = rule["action"]
        if action in self._rule_handlers:
            self._rule_handlers[action](rule)
