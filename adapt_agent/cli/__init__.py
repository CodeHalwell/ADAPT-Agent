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

``adapt-agent evaluate <target> --data <file> [--metric NAME ...] [--judge PROVIDER]``
    Evaluate an agent against a golden dataset and print the scores. ``<target>``
    is ``module:attribute`` (append ``()`` to call a factory).

``adapt-agent optimize <target> --data <file> [--optimizer S] [--judge PROVIDER] ...``
    Optimize an agent against a golden dataset (prompts, few-shot, models,
    hyperparameters, routing, tools) and apply the best configuration. Register
    multi-agent components with ``--component NAME=module:attr`` (repeatable) and
    the system entrypoint as the ``<target>`` (or ``--runner``).

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

# Static name lists for argparse help/choices, kept here so building the parser
# never imports the (heavier) optimization subsystem for `info`/`validate`.
_BUILTIN_METRIC_NAMES = (
    "exact_match",
    "contains",
    "regex_match",
    "token_f1",
    "jaccard",
    "numeric_close",
    "json_subset",
    "levenshtein_ratio",
)
_OPTIMIZER_CHOICES = (
    "default",
    "coordinate_ascent",
    "bootstrap_few_shot",
    "grid",
    "random",
    "evolutionary",
)


def _builtin_metric_names() -> list[str]:
    return list(_BUILTIN_METRIC_NAMES)


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

    # `evaluate` and `optimize` share the same target/dataset/metric/judge options.
    for name, help_text in (
        ("evaluate", "Evaluate an agent against a golden dataset"),
        ("optimize", "Optimize an agent against a golden dataset"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument(
            "target",
            help="Agent to load as 'module:attribute' (append '()' to call a factory). "
            "May be an OptimizableAgent, a framework object, or a runner callable.",
        )
        sub.add_argument(
            "--data", required=True, help="Golden dataset file (.json / .jsonl / .csv)"
        )
        sub.add_argument(
            "--component",
            action="append",
            default=[],
            metavar="NAME=module:attr",
            help="Register a named framework component to introspect (repeatable).",
        )
        sub.add_argument(
            "--runner",
            help="Runner callable as 'module:attribute' (append '()' to call a factory).",
        )
        sub.add_argument(
            "--metric",
            action="append",
            default=[],
            metavar="NAME",
            help=f"Built-in metric to apply (repeatable): {sorted(_builtin_metric_names())}.",
        )
        sub.add_argument("--judge", help="LLM-judge provider (e.g. claude, openai, gemini).")
        sub.add_argument("--judge-model", help="Model id for the judge provider.")
        sub.add_argument("--primary", help="Name of the primary (headline) metric.")
        sub.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
        if name == "optimize":
            sub.add_argument(
                "--optimizer",
                default="default",
                choices=sorted(_OPTIMIZER_CHOICES),
                help="Search strategy (default: 'default' pipeline).",
            )
            sub.add_argument("--max-evals", type=int, default=60, help="Evaluation budget.")
            sub.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
            sub.add_argument("--val-data", help="Optional held-out dataset for validation.")
            sub.add_argument(
                "--save-config", help="Write the winning configuration to this JSON file."
            )
            sub.add_argument("--verbose", action="store_true", help="Log each trial.")

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
    if parsed_args.command == "evaluate":
        return _cmd_evaluate(parsed_args)
    if parsed_args.command == "optimize":
        return _cmd_optimize(parsed_args)

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


# -- optimize / evaluate ------------------------------------------------------


def _load_object(spec: str) -> Any:
    """Import and return the object named by ``module:attribute``.

    A trailing ``()`` calls the resolved object (a zero-arg factory). The current
    working directory is added to ``sys.path`` so a user's project is importable.

    Raises:
        ValueError: If ``spec`` is not of the form ``module:attribute``.
        ImportError / AttributeError: If the module or attribute cannot be found.
    """
    import importlib
    import os

    call = spec.endswith("()")
    if call:
        spec = spec[:-2]
    module_name, sep, attr_path = spec.partition(":")
    if not sep or not module_name or not attr_path:
        raise ValueError(f"Expected 'module:attribute', got {spec!r}")

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    obj: Any = importlib.import_module(module_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj() if call else obj


def _load_dataset(path: str) -> Any:
    """Load a golden dataset, dispatching by file extension."""
    from adapt_agent.optimization import GoldenDataset

    lower = path.lower()
    if lower.endswith(".jsonl"):
        return GoldenDataset.from_jsonl(path)
    if lower.endswith(".json"):
        return GoldenDataset.from_json(path)
    if lower.endswith(".csv"):
        return GoldenDataset.from_csv(path)
    raise ValueError(f"Unsupported dataset extension for {path!r} (use .json/.jsonl/.csv)")


def _build_components(specs: list[str]) -> dict[str, Any]:
    """Parse repeated ``NAME=module:attr`` component specs into live objects."""
    components: dict[str, Any] = {}
    for item in specs:
        name, sep, spec = item.partition("=")
        name = name.strip()
        if not sep or not name or not spec:
            raise ValueError(f"--component expects NAME=module:attr, got {item!r}")
        components[name] = _load_object(spec)
    return components


def _build_judge(provider: Optional[str], model: Optional[str]) -> Any:
    """Construct an LLM judge for a provider name, or ``None`` when unset."""
    if not provider:
        return None
    from adapt_agent.optimization.judges import get_judge

    return get_judge(provider, model=model)


def _build_metrics(names: list[str], judge: Any, primary: Optional[str]) -> tuple[list[Any], str]:
    """Build the metric list (built-ins + optional judge) and the primary name."""
    from adapt_agent.optimization import get_metric

    metrics: list[Any] = [get_metric(name) for name in names]
    if judge is not None:
        metrics.append(judge.as_metric("judge"))
    if not metrics:
        raise ValueError(
            "No metrics specified. Pass --metric NAME (one or more) and/or --judge PROVIDER."
        )
    return metrics, (primary or metrics[0].name)


def _build_target(args: Any) -> Any:
    """Resolve the optimization target from the parsed CLI args."""
    from adapt_agent.optimization import OptimizableAgent, wrap
    from adapt_agent.optimization.evaluation import resolve_runner

    target_obj = _load_object(args.target)
    components = _build_components(args.component)
    explicit_runner = _load_object(args.runner) if args.runner else None

    if components:
        # With components, the positional target (or --runner) is the entrypoint
        # that drives the whole system; the components supply the tunable knobs.
        runner = resolve_runner(explicit_runner if explicit_runner is not None else target_obj)
        return OptimizableAgent.from_components(components, runner=runner, name=args.target)
    return wrap(target_obj, runner=explicit_runner, name=args.target)


def _build_optimizer(
    name: str, harness: Any, judge: Any, max_evals: int, seed: int, verbose: bool
) -> Any:
    """Instantiate the requested optimizer strategy."""
    from adapt_agent.optimization import (
        BootstrapFewShotOptimizer,
        CoordinateAscentOptimizer,
        EvolutionaryOptimizer,
        GridSearchOptimizer,
        RandomSearchOptimizer,
        make_default_optimizer,
    )

    if name == "default":
        return make_default_optimizer(
            harness, judge=judge, max_evals=max_evals, seed=seed, verbose=verbose
        )
    classes = {
        "coordinate_ascent": CoordinateAscentOptimizer,
        "bootstrap_few_shot": BootstrapFewShotOptimizer,
        "grid": GridSearchOptimizer,
        "random": RandomSearchOptimizer,
        "evolutionary": EvolutionaryOptimizer,
    }
    return classes[name](harness, max_evals=max_evals, seed=seed, judge=judge, verbose=verbose)


def _fail(exc: Exception, as_json: bool) -> int:
    """Print an error (text or JSON) and return exit code 1."""
    message = f"{type(exc).__name__}: {exc}"
    if as_json:
        print(json.dumps({"status": "error", "error": message}, indent=2))
    else:
        print(f"ERROR: {message}")
    return 1


def _cmd_evaluate(args: Any) -> int:
    """Evaluate an agent against a golden dataset."""
    from adapt_agent.optimization import EvaluationHarness

    try:
        target = _build_target(args)
        dataset = _load_dataset(args.data)
        judge = _build_judge(args.judge, args.judge_model)
        metrics, primary = _build_metrics(args.metric, judge, args.primary)
        harness = EvaluationHarness(metrics, primary_metric=primary)
        report = harness.evaluate(target, dataset)
    except Exception as exc:
        return _fail(exc, args.json)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(f"Evaluated '{args.target}' over {report.n} example(s):")
        print(f"  primary metric: {report.primary_metric} = {report.score:.4f}")
        for metric_name, value in sorted(report.aggregate.items()):
            print(f"    - {metric_name}: {value:.4f}")
        print(f"  errors: {report.n_errors}   avg latency: {report.avg_latency:.4f}s")
    return 0


def _cmd_optimize(args: Any) -> int:
    """Optimize an agent against a golden dataset."""
    from adapt_agent.optimization import EvaluationHarness

    try:
        target = _build_target(args)
        dataset = _load_dataset(args.data)
        val_dataset = _load_dataset(args.val_data) if args.val_data else None
        judge = _build_judge(args.judge, args.judge_model)
        metrics, primary = _build_metrics(args.metric, judge, args.primary)
        harness = EvaluationHarness(metrics, primary_metric=primary)
        optimizer = _build_optimizer(
            args.optimizer, harness, judge, args.max_evals, args.seed, args.verbose
        )
        result = optimizer.optimize(target, dataset, val_dataset=val_dataset)
    except Exception as exc:
        return _fail(exc, args.json)

    if args.save_config:
        try:
            with open(args.save_config, "w", encoding="utf-8") as handle:
                json.dump(result.best_config, handle, indent=2, default=str)
        except OSError as exc:
            return _fail(exc, args.json)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(f"Optimized '{args.target}' with strategy '{args.optimizer}':")
        print(f"  baseline   : {result.baseline_score:.4f}")
        print(f"  best       : {result.best_score:.4f}  (improvement {result.improvement:+.4f})")
        if result.validation_score is not None:
            print(f"  validation : {result.validation_score:.4f}")
        print(f"  evaluations: {result.n_evals}")
        if result.best_config:
            print("  best config:")
            for key, value in result.best_config.items():
                preview = repr(value)
                if len(preview) > 80:
                    preview = preview[:77] + "..."
                print(f"    - {key} = {preview}")
        else:
            print("  no improving configuration found (agent left unchanged).")
        if args.save_config:
            print(f"  saved best config to: {args.save_config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main", "load_config", "validate_config"]
