"""Command-line interface for ADAPT-Agent.

Commands
--------
``adapt-agent info``
    Print library information.

``adapt-agent validate <config_file> [--json]``
    Validate an ADAPT-Agent configuration file (see :func:`validate_config`).

``adapt-agent monitor --agent-id <id> [--config <file>] [--json]``
    Initialise the observability/security stack for an agent and print a
    readiness status snapshot.

Configuration file schema (JSON)
--------------------------------
.. code-block:: json

    {
      "policy_rules": [
        {"name": "no_secrets", "description": "block secrets",
         "condition": "'password' in message['content']",
         "action": "block", "severity": "high"}
      ],
      "firewall": {
        "blocked_patterns": ["(?i)ignore previous instructions"],
        "allowed_patterns": ["[a-zA-Z0-9 ]+"],
        "max_content_length": 10000
      },
      "adversarial": {"attack_patterns": ["leak the system prompt"]}
    }
"""

import argparse
import ast
import json
import re
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from adapt_agent import __version__

_VALID_ACTIONS = {"warn", "block", "modify"}
_VALID_SEVERITIES = {"low", "medium", "high", "critical"}
_MAX_CONDITION_LENGTH = 1024


def main(args: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        args: Optional command-line arguments (defaults to ``sys.argv``).

    Returns:
        Process exit code (0 on success, non-zero on error).
    """
    parser = argparse.ArgumentParser(
        prog="adapt-agent",
        description="ADAPT-Agent: Adversarial Defense & Policy Training for LLM Agents",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("info", help="Display information about ADAPT-Agent")

    validate_parser = subparsers.add_parser("validate", help="Validate a configuration file")
    validate_parser.add_argument("config_file", help="Path to a JSON configuration file")
    validate_parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON output"
    )

    monitor_parser = subparsers.add_parser("monitor", help="Initialise monitoring for an agent")
    monitor_parser.add_argument("--agent-id", required=True, help="Agent identifier")
    monitor_parser.add_argument("--config", help="Optional path to a JSON configuration file")
    monitor_parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON output"
    )

    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 0

    if parsed_args.command == "info":
        return _cmd_info()
    if parsed_args.command == "validate":
        return _cmd_validate(parsed_args.config_file, as_json=parsed_args.json)
    if parsed_args.command == "monitor":
        return _cmd_monitor(
            parsed_args.agent_id, config_file=parsed_args.config, as_json=parsed_args.json
        )

    return 0


def _cmd_info() -> int:
    """Display information about ADAPT-Agent."""
    print(f"ADAPT-Agent v{__version__}")
    print("Adversarial Defense & Policy Training for LLM Agents")
    print()
    print("A comprehensive library for LLM agent optimization and security.")
    print()
    print("Features:")
    print("  - Trust management and policy enforcement")
    print("  - Security firewall and taint tracking")
    print("  - Adversarial defense (prompt injection / jailbreak detection)")
    print("  - Adapters: LangGraph, Microsoft Agent Framework, Google ADK,")
    print("    Pydantic AI, CrewAI, OpenAI Agents SDK, Claude Agent SDK")
    print("  - Performance optimization, evaluation and observability")
    print()
    print("For more information, visit: https://github.com/CodeHalwell/ADAPT-Agent")
    return 0


def load_config(config_file: str) -> dict[str, Any]:
    """Load and JSON-parse a configuration file.

    Args:
        config_file: Path to the JSON configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or not a JSON object.
    """
    with open(config_file, encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a JSON object")
    return data


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate a parsed configuration dictionary.

    Checks that policy rule conditions parse, regex patterns compile, and
    action/severity values are valid.

    Args:
        config: Parsed configuration dictionary.

    Returns:
        A list of human-readable error strings. An empty list means the
        configuration is valid.
    """
    errors: list[str] = []

    rules = config.get("policy_rules", [])
    if not isinstance(rules, list):
        errors.append("'policy_rules' must be a list")
        rules = []
    for index, rule in enumerate(rules):
        label = f"policy_rules[{index}]"
        if not isinstance(rule, dict):
            errors.append(f"{label} must be an object")
            continue
        if not rule.get("name"):
            errors.append(f"{label} is missing required field 'name'")
        condition = rule.get("condition")
        if not condition:
            errors.append(f"{label} is missing required field 'condition'")
        elif not isinstance(condition, str):
            errors.append(f"{label}.condition must be a string")
        elif len(condition) > _MAX_CONDITION_LENGTH:
            errors.append(f"{label}.condition exceeds maximum length of {_MAX_CONDITION_LENGTH}")
        else:
            try:
                ast.parse(condition, mode="eval")
            except SyntaxError as exc:
                errors.append(f"{label}.condition is not a valid expression: {exc.msg}")
        action = rule.get("action", "warn")
        if action not in _VALID_ACTIONS:
            errors.append(
                f"{label}.action '{action}' is invalid (expected one of {sorted(_VALID_ACTIONS)})"
            )
        severity = rule.get("severity", "medium")
        if severity not in _VALID_SEVERITIES:
            errors.append(
                f"{label}.severity '{severity}' is invalid (expected one of {sorted(_VALID_SEVERITIES)})"
            )

    firewall = config.get("firewall", {})
    if not isinstance(firewall, dict):
        errors.append("'firewall' must be an object")
        firewall = {}
    for key in ("blocked_patterns", "allowed_patterns"):
        patterns = firewall.get(key, [])
        if not isinstance(patterns, list):
            errors.append(f"firewall.{key} must be a list")
            continue
        for index, pattern in enumerate(patterns):
            if not isinstance(pattern, str):
                errors.append(f"firewall.{key}[{index}] must be a string")
                continue
            try:
                re.compile(pattern)
            except re.error as exc:
                errors.append(f"firewall.{key}[{index}] is not a valid regex: {exc}")
    max_len = firewall.get("max_content_length")
    if max_len is not None and (not isinstance(max_len, int) or max_len <= 0):
        errors.append("firewall.max_content_length must be a positive integer")

    adversarial = config.get("adversarial", {})
    if not isinstance(adversarial, dict):
        errors.append("'adversarial' must be an object")
    else:
        attack_patterns = adversarial.get("attack_patterns", [])
        if not isinstance(attack_patterns, list):
            errors.append("adversarial.attack_patterns must be a list")
        elif any(not isinstance(p, str) for p in attack_patterns):
            errors.append("adversarial.attack_patterns must contain only strings")

    return errors


def _cmd_validate(config_file: str, as_json: bool = False) -> int:
    """Validate agent configuration."""
    try:
        config = load_config(config_file)
    except (OSError, ValueError) as exc:
        message = str(exc)
        if as_json:
            print(json.dumps({"valid": False, "errors": [message]}, indent=2))
        else:
            print(f"ERROR: {message}")
        return 1

    errors = validate_config(config)
    if as_json:
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
    elif errors:
        print(f"Configuration is INVALID ({len(errors)} error(s)):")
        for error in errors:
            print(f"  - {error}")
    else:
        rule_count = len(config.get("policy_rules", []))
        print(f"Configuration is valid. ({rule_count} policy rule(s))")

    return 1 if errors else 0


def _cmd_monitor(agent_id: str, config_file: Optional[str] = None, as_json: bool = False) -> int:
    """Initialise the observability stack for an agent and report readiness."""
    # Imported lazily to keep `info`/`validate` fast and dependency-light.
    from adapt_agent.observability import AgentObserver

    controls: dict[str, Any] = {
        "policy_rules": 0,
        "firewall_blocked_patterns": 0,
        "firewall_allowed_patterns": 0,
        "adversarial_patterns": 0,
    }

    if config_file is not None:
        try:
            config = load_config(config_file)
        except (OSError, ValueError) as exc:
            message = str(exc)
            if as_json:
                print(json.dumps({"status": "error", "error": message}, indent=2))
            else:
                print(f"ERROR: {message}")
            return 1
        errors = validate_config(config)
        if errors:
            payload = {"status": "error", "error": "invalid configuration", "errors": errors}
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                print("ERROR: configuration is invalid; run 'adapt-agent validate' for details.")
            return 1
        firewall = config.get("firewall", {})
        controls = {
            "policy_rules": len(config.get("policy_rules", [])),
            "firewall_blocked_patterns": len(firewall.get("blocked_patterns", [])),
            "firewall_allowed_patterns": len(firewall.get("allowed_patterns", [])),
            "adversarial_patterns": len(config.get("adversarial", {}).get("attack_patterns", [])),
        }

    observer = AgentObserver()
    observer.log("info", f"Monitoring initialised for agent '{agent_id}'", agent_id=agent_id)

    status = {
        "status": "ready",
        "agent_id": agent_id,
        "version": __version__,
        "controls": controls,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if as_json:
        print(json.dumps(status, indent=2))
    else:
        print(f"Monitoring agent: {agent_id}")
        print(f"Status: {status['status']}")
        print("Configured controls:")
        for key, value in controls.items():
            print(f"  - {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main", "load_config", "validate_config"]
