"""Command-line interface for ADAPT-Agent."""

import argparse
import sys
from typing import List, Optional


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        args: Optional command-line arguments
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        prog="adapt-agent",
        description="ADAPT-Agent: Adversarial Defense & Policy Training for LLM Agents",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Display information about ADAPT-Agent",
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate agent configuration",
    )
    validate_parser.add_argument(
        "config_file",
        help="Path to configuration file",
    )
    
    # Monitor command
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Start monitoring agent",
    )
    monitor_parser.add_argument(
        "--agent-id",
        required=True,
        help="Agent identifier",
    )
    
    parsed_args = parser.parse_args(args)
    
    if not parsed_args.command:
        parser.print_help()
        return 0
    
    if parsed_args.command == "info":
        return _cmd_info()
    elif parsed_args.command == "validate":
        return _cmd_validate(parsed_args.config_file)
    elif parsed_args.command == "monitor":
        return _cmd_monitor(parsed_args.agent_id)
    
    return 0


def _cmd_info() -> int:
    """Display information about ADAPT-Agent."""
    print("ADAPT-Agent v0.1.0")
    print("Adversarial Defense & Policy Training for LLM Agents")
    print()
    print("A comprehensive library for LLM agent optimization and security.")
    print()
    print("Features:")
    print("  - Trust management and policy enforcement")
    print("  - Security firewall and taint tracking")
    print("  - Multi-framework support (LangGraph, Semantic Kernel, CrewAI)")
    print("  - Adversarial defense")
    print("  - Performance optimization")
    print("  - Comprehensive evaluation and observability")
    print()
    print("For more information, visit: https://github.com/CodeHalwell/ADAPT-Agent")
    return 0


def _cmd_validate(config_file: str) -> int:
    """Validate agent configuration."""
    print(f"Validating configuration: {config_file}")
    print("Validation not yet implemented.")
    return 0


def _cmd_monitor(agent_id: str) -> int:
    """Monitor agent."""
    print(f"Monitoring agent: {agent_id}")
    print("Monitoring not yet implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
