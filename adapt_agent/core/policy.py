"""Policy enforcement for LLM agents."""

from datetime import datetime, timezone
from typing import Any, Callable, Optional

from adapt_agent.core.types import AgentMessage, AgentState, PolicyRule


class PolicyEnforcer:
    """Enforces policies and rules on LLM agent behavior.

    The PolicyEnforcer validates agent actions, messages, and state changes
    against defined policies and can take corrective actions when violations occur.
    """

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

        for rule_name, rule in self._rules.items():
            # Simple condition checking (can be extended with more sophisticated evaluation)
            if self._evaluate_condition(rule["condition"], {"message": message}):
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

        for rule_name, rule in self._rules.items():
            if self._evaluate_condition(rule["condition"], {"state": state}):
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
                return list(reversed(results))
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
        # Placeholder implementation - real implementation would use
        # a safe expression evaluator or DSL
        # For now, always return False (no violations)
        return False

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
