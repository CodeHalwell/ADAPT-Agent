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


class SkillError(AdaptError):
    """Raised for problems installing or reading a bundled agent skill."""


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
    "SkillError",
    "AdapterError",
    "MissingDependencyError",
]


class IncompleteEvaluationError(AdaptError):
    """An evaluation could not score every row, and the caller needs all of them.

    Raised when an optimizer's *baseline* stays incomplete after a re-run.
    Every candidate is compared against the baseline, so a score computed over
    a subset of the dataset makes the whole search meaningless -- and it fails
    quietly: the inflated baseline is unbeatable, so the run ends reporting the
    starting configuration, which is indistinguishable from "nothing improved".
    Better to stop and say so.
    """
