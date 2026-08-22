"""Tests for the exception hierarchy."""

from __future__ import annotations

import inspect

import adapt_agent.exceptions as exceptions


def test_every_exception_is_exported() -> None:
    """`__all__` is the supported export surface, so it has to be complete.

    `IncompleteEvaluationError` was the one of six that was missed, so
    `from adapt_agent.exceptions import *` could not reach the exception the
    optimizer deliberately raises -- a caller on that surface had no way to
    catch it specifically. Asserted as a rule over the module rather than as
    one more name in a list, because a list is what went stale.
    """
    defined = {
        name
        for name, obj in vars(exceptions).items()
        if inspect.isclass(obj)
        and issubclass(obj, BaseException)
        and obj.__module__ == exceptions.__name__
    }
    assert defined - set(exceptions.__all__) == set(), "an exception is missing from __all__"
    assert set(exceptions.__all__) - defined == set(), "__all__ names something undefined"


def test_the_incomplete_evaluation_error_is_reachable_by_star_import() -> None:
    namespace: dict = {}
    exec("from adapt_agent.exceptions import *", namespace)  # noqa: S102
    assert "IncompleteEvaluationError" in namespace
    assert issubclass(namespace["IncompleteEvaluationError"], namespace["AdaptError"])
