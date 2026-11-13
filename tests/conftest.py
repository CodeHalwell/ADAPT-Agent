"""Shared pytest configuration."""

import os

try:
    from hypothesis import HealthCheck, settings
except ImportError:  # pragma: no cover - hypothesis is a dev dependency
    pass
else:
    # No per-example deadline: several properties drive an event loop per
    # example, and a slow CI runner should not turn that into a flaky failure.
    settings.register_profile(
        "default",
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    # Under CI the same examples every run: a property that fails must fail for
    # the change that broke it, not at random on someone else's pull request.
    settings.register_profile("ci", parent=settings.get_profile("default"), derandomize=True)
    settings.load_profile("ci" if os.environ.get("CI") else "default")
