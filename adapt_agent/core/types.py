"""Core type definitions for ADAPT-Agent."""

from typing import Any, Protocol, TypedDict

from typing_extensions import NotRequired


class AgentMessage(TypedDict):
    """Represents a message in an agent conversation."""

    role: str
    content: str
    metadata: NotRequired[dict[str, Any]]


class AgentState(TypedDict):
    """Represents the state of an agent."""

    messages: list[AgentMessage]
    context: dict[str, Any]
    trust_score: NotRequired[float]
    policy_violations: NotRequired[list[str]]


class TrustScore(TypedDict):
    """Represents a trust score calculation."""

    score: float
    confidence: float
    factors: dict[str, float]
    timestamp: str


class PolicyRule(TypedDict):
    """Represents a policy rule."""

    name: str
    description: str
    condition: str
    action: str
    severity: str


class SecurityEvent(TypedDict):
    """Represents a security event."""

    event_type: str
    severity: str
    description: str
    timestamp: str
    metadata: dict[str, Any]


class Agent(Protocol):
    """Protocol for agent implementations."""

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent with given input."""
        ...

    def get_state(self) -> AgentState:
        """Get the current agent state."""
        ...


class Adapter(Protocol):
    """Protocol for framework adapters."""

    def wrap_agent(self, agent: Any) -> Agent:
        """Wrap a framework-specific agent."""
        ...

    def extract_state(self, agent: Any) -> AgentState:
        """Extract state from a framework-specific agent."""
        ...


# Type aliases
MessageList = list[AgentMessage]
ContextDict = dict[str, Any]
TrustScoreValue = float
PolicyRuleList = list[PolicyRule]
