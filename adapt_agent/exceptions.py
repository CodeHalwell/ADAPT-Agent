"""Exception hierarchy for ADAPT-Agent.

All exceptions raised intentionally by ADAPT-Agent derive from :class:`AdaptError`,
so callers can catch the whole family with a single ``except AdaptError``.
"""


class AdaptError(Exception):
    """Base class for all ADAPT-Agent errors."""


class SecurityBlockedError(AdaptError):
    """Raised when a security control blocks an agent input or output.

    Attributes:
        reason: Human-readable reason the content was blocked.
        threats: List of threat identifiers that triggered the block.
    """

    def __init__(self, reason: str, threats: "list[str] | None" = None):
        self.reason = reason
        self.threats = threats or []
        super().__init__(reason)


class AdapterError(AdaptError):
    """Base class for framework-adapter errors."""


class MissingDependencyError(AdapterError):
    """Raised when an optional framework dependency is required but not installed."""

    def __init__(self, package: str, extra: str):
        self.package = package
        self.extra = extra
        super().__init__(
            f"The '{package}' package is required for this adapter. "
            f"Install it with: pip install adapt-agent[{extra}]"
        )


__all__ = [
    "AdaptError",
    "SecurityBlockedError",
    "AdapterError",
    "MissingDependencyError",
]
