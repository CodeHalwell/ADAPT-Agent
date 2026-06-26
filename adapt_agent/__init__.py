"""
ADAPT-Agent: Adversarial Defense & Policy Training for LLM Agents

A comprehensive library for LLM agent optimization and security.
"""

__version__ = "0.2.0"

# Core exports
# Framework adapters
from adapt_agent.adapters import BaseAdapter

# Adversarial defense
from adapt_agent.adversarial import AdversarialDefense
from adapt_agent.core import (
    MemorySystem,
    Middleware,
    PolicyEnforcer,
    TrustManager,
)
from adapt_agent.core.types import (
    Adapter,
    Agent,
    AgentMessage,
    AgentState,
    PolicyRule,
    SecurityEvent,
    TrustScore,
)

# Evaluation, observability, optimization
from adapt_agent.evaluation import AgentEvaluator
from adapt_agent.observability import AgentObserver
from adapt_agent.optimization import (
    AgentOptimizer,
    CoordinateAscentOptimizer,
    EvaluationHarness,
    Example,
    GoldenDataset,
    LLMJudge,
    ModelProvider,
    OptimizableAgent,
    Parameter,
    ParameterKind,
    get_provider,
    make_default_optimizer,
)

# Patches
from adapt_agent.patches import PatchManager

# Security exports
from adapt_agent.security import (
    Firewall,
    TaintLevel,
    TaintSource,
    TaintTracker,
)

__all__ = [
    "__version__",
    # Core
    "TrustManager",
    "PolicyEnforcer",
    "MemorySystem",
    "Middleware",
    # Types
    "Agent",
    "Adapter",
    "AgentMessage",
    "AgentState",
    "TrustScore",
    "PolicyRule",
    "SecurityEvent",
    # Security
    "Firewall",
    "TaintTracker",
    "TaintLevel",
    "TaintSource",
    # Adversarial / evaluation / observability / optimization
    "AdversarialDefense",
    "AgentEvaluator",
    "AgentObserver",
    "AgentOptimizer",
    # Dataset-driven optimization & evaluation
    "GoldenDataset",
    "Example",
    "EvaluationHarness",
    "LLMJudge",
    "ModelProvider",
    "get_provider",
    "OptimizableAgent",
    "Parameter",
    "ParameterKind",
    "CoordinateAscentOptimizer",
    "make_default_optimizer",
    # Patches & adapters
    "PatchManager",
    "BaseAdapter",
]
