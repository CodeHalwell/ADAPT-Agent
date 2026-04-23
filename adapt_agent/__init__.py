"""
ADAPT-Agent: Adversarial Defense & Policy Training for LLM Agents

A comprehensive library for LLM agent optimization and security.
"""

__version__ = "0.1.0"

# Core exports
from adapt_agent.core import (
    TrustManager,
    PolicyEnforcer,
    MemorySystem,
    Middleware,
)

# Security exports
from adapt_agent.security import (
    Firewall,
    TaintTracker,
)

__all__ = [
    "__version__",
    "TrustManager",
    "PolicyEnforcer",
    "MemorySystem",
    "Middleware",
    "Firewall",
    "TaintTracker",
]
