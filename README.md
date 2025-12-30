# ADAPT-Agent

**A**dversarial **D**efense &amp; **P**olicy **T**raining for LLM **Agent**s

A comprehensive Python library for LLM agent optimization and security, providing tools for trust management, policy enforcement, adversarial defense, and multi-framework integration.

## Features

- **Core Components**: Trust management, policy enforcement, memory systems, and middleware
- **Multi-Framework Support**: Adapters for LangGraph, Semantic Kernel, and CrewAI
- **Security**: Built-in firewall and taint tracking for LLM security
- **Optimization**: Tools for optimizing agent performance
- **Adversarial Defense**: Protection against adversarial attacks
- **Evaluation**: Comprehensive evaluation frameworks
- **Observability**: Monitoring and debugging tools
- **Patches**: Framework-specific patches and improvements
- **CLI**: Command-line interface for common tasks

## Installation

```bash
pip install adapt-agent
```

### Optional Dependencies

Install with support for specific frameworks:

```bash
# For LangGraph support
pip install adapt-agent[langgraph]

# For Semantic Kernel support
pip install adapt-agent[semantic-kernel]

# For CrewAI support
pip install adapt-agent[crewai]

# Install all optional dependencies
pip install adapt-agent[all]
```

## Quick Start

```python
from adapt_agent.core import TrustManager, PolicyEnforcer
from adapt_agent.security import Firewall

# Initialize core components
trust_manager = TrustManager()
policy_enforcer = PolicyEnforcer()
firewall = Firewall()

# Use with your LLM agents...
```

## Project Structure

```
adapt_agent/
├── core/              # Core functionality (types, trust, policy, memory, middleware)
├── adapters/          # Framework adapters (LangGraph, Semantic Kernel, CrewAI)
├── security/          # Security components (firewall, taint tracking)
├── optimization/      # Optimization tools
├── adversarial/       # Adversarial defense
├── evaluation/        # Evaluation frameworks
├── observability/     # Monitoring and debugging
├── patches/           # Framework patches
└── cli/               # Command-line interface
```

## Development

```bash
# Clone the repository
git clone https://github.com/CodeHalwell/ADAPT-Agent.git
cd ADAPT-Agent

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
ruff check .
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
