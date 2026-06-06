"""Policy enforcement for LLM agents."""

import ast
import logging
import operator
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Callable, Optional

from adapt_agent.core.types import AgentMessage, AgentState, PolicyRule

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1024)
def _parse_condition(condition: str) -> ast.AST:
    """Cache AST parsing for policy conditions."""
    return ast.parse(condition, mode="eval")


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

    def __init__(self, max_violations: int = 1000):
        """Initialize the PolicyEnforcer.

        Args:
            max_violations: Maximum number of violations to store in memory.
        """
        self.max_violations = max_violations
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

    def get_rule(self, name: str) -> Optional[PolicyRule]:
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
            if self._evaluate_condition(rule["condition"], context):
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
            if self._evaluate_condition(rule["condition"], context):
                violations.append(rule_name)
                self._record_violation(rule_name, "state", state)
                self._handle_violation(rule)

        return violations

    def get_violations(
        self,
        severity: Optional[str] = None,
        limit: Optional[int] = None,
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

    def _evaluate_condition(self, condition: str, context: dict[str, Any]) -> bool:
        """Evaluate a policy condition.

        Args:
            condition: Condition expression
            context: Context for evaluation

        Returns:
            True if condition is met (violation), False otherwise
        """
        try:
            tree = _parse_condition(condition)
            return bool(self._eval_node(tree.body, context))
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False

    def _eval_node(self, node: ast.AST, context: dict[str, Any], depth: int = 0) -> Any:
        if depth > 50:
            raise ValueError("Maximum evaluation depth exceeded (DoS protection)")
        node_type = type(node)
        if node_type is ast.Constant:
            return node.value
        elif node_type is ast.Name:
            if node.id in context:
                return context[node.id]
            raise ValueError(f"Unknown variable: {node.id}")
        elif node_type is ast.List:
            return [self._eval_node(elt, context, depth + 1) for elt in node.elts]
        elif node_type is ast.Tuple:
            return tuple(self._eval_node(elt, context, depth + 1) for elt in node.elts)
        elif node_type is ast.Set:
            return {self._eval_node(elt, context, depth + 1) for elt in node.elts}
        elif node_type is ast.Dict:
            return {
                self._eval_node(k, context, depth + 1): self._eval_node(v, context, depth + 1)
                for k, v in zip(node.keys, node.values)
            }
        elif node_type is ast.Compare:
            left = self._eval_node(node.left, context, depth + 1)
            for op, comp in zip(node.ops, node.comparators):
                right = self._eval_node(comp, context, depth + 1)
                op_type = type(op)
                if op_type not in self._OPERATORS:
                    raise ValueError(f"Unsupported operator: {op_type}")
                if not self._OPERATORS[op_type](left, right):
                    return False
                left = right
            return True
        elif node_type is ast.BoolOp:
            op_type = type(node.op)
            if op_type is ast.And:
                for value in node.values:
                    if not self._eval_node(value, context, depth + 1):
                        return False
                return True
            elif op_type is ast.Or:
                for value in node.values:
                    if self._eval_node(value, context, depth + 1):
                        return True
                return False
            raise ValueError(f"Unsupported boolean operator: {op_type}")
        elif node_type is ast.BinOp:
            left = self._eval_node(node.left, context, depth + 1)
            right = self._eval_node(node.right, context, depth + 1)
            op_type = type(node.op)
            if op_type not in self._BINOPS:
                raise ValueError(f"Unsupported binary operator: {op_type}")
            return self._BINOPS[op_type](left, right)
        elif node_type is ast.Subscript:
            val = self._eval_node(node.value, context, depth + 1)
            slice_type = type(node.slice)
            if slice_type is ast.Slice:
                raise ValueError("Slices are not supported")
            slice_val = self._eval_node(node.slice, context, depth + 1)
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
