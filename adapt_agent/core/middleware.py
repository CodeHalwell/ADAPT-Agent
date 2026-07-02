"""Middleware system for LLM agents."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


MiddlewareFunc = Callable[[dict[str, Any]], dict[str, Any]]


class Middleware:
    """Middleware system for intercepting and modifying agent interactions.

    Provides a composable middleware pipeline for pre-processing and
    post-processing agent inputs and outputs.

    Failure policy is explicit. By default the pipeline is *fail-open*: a
    middleware that raises is logged and skipped, and the (unmodified) data
    flows on to the next middleware. This is convenient but dangerous when a
    middleware is a security control (e.g. a sanitizer): if it crashes, raw
    data would silently pass through. Set ``fail_closed=True`` to make a
    raising middleware abort the pipeline by re-raising, so a sanitizer failure
    stops the request instead of leaking unsanitized data.
    """

    def __init__(self, fail_closed: bool = False):
        """Initialize the Middleware system.

        Args:
            fail_closed: When True, a middleware that raises aborts the whole
                pipeline (the exception is re-raised) instead of being logged
                and skipped. When False (default), the error is logged and the
                pipeline continues with the unmodified data (fail-open).
        """
        self.fail_closed = fail_closed
        self._pre_middleware: list[MiddlewareFunc] = []
        self._post_middleware: list[MiddlewareFunc] = []
        self._middleware_metadata: dict[str, dict[str, Any]] = {}

    def add_pre_middleware(
        self,
        middleware: MiddlewareFunc,
        name: str | None = None,
        priority: int = 0,
    ) -> None:
        """Add middleware to run before agent execution.

        Args:
            middleware: Middleware function
            name: Optional name for the middleware
            priority: Priority (higher runs first)

        Raises:
            ValueError: If a middleware with the same name is already registered.
        """
        resolved_name = name or middleware.__name__
        self._reject_duplicate_name(resolved_name)
        self._middleware_metadata[resolved_name] = {
            "type": "pre",
            "priority": priority,
            "function": middleware,
        }

        self._pre_middleware.append(middleware)

        # Create mapping of function objects to priority
        func_priorities = {
            m["function"]: m["priority"]
            for m in self._middleware_metadata.values()
            if m["type"] == "pre"
        }

        # Sort by priority
        self._pre_middleware.sort(
            key=lambda m: func_priorities.get(m, 0),
            reverse=True,
        )

    def add_post_middleware(
        self,
        middleware: MiddlewareFunc,
        name: str | None = None,
        priority: int = 0,
    ) -> None:
        """Add middleware to run after agent execution.

        Args:
            middleware: Middleware function
            name: Optional name for the middleware
            priority: Priority (higher runs first)

        Raises:
            ValueError: If a middleware with the same name is already registered.
        """
        resolved_name = name or middleware.__name__
        self._reject_duplicate_name(resolved_name)
        self._middleware_metadata[resolved_name] = {
            "type": "post",
            "priority": priority,
            "function": middleware,
        }

        self._post_middleware.append(middleware)

        # Create mapping of function objects to priority
        func_priorities = {
            m["function"]: m["priority"]
            for m in self._middleware_metadata.values()
            if m["type"] == "post"
        }

        # Sort by priority
        self._post_middleware.sort(
            key=lambda m: func_priorities.get(m, 0),
            reverse=True,
        )

    def _reject_duplicate_name(self, name: str) -> None:
        """Reject a duplicate middleware name.

        A single ``_middleware_metadata`` entry is keyed by name, so allowing
        two middlewares to share a name corrupts the registry: both functions
        get appended to the pipeline but only one metadata entry exists, and
        ``remove_middleware`` would then orphan the other. Refuse up front.

        Args:
            name: Resolved middleware name.

        Raises:
            ValueError: If the name is already registered.
        """
        if name in self._middleware_metadata:
            raise ValueError(
                f"Middleware named {name!r} is already registered; " "names must be unique"
            )

    def remove_middleware(self, name: str) -> bool:
        """Remove a middleware by name.

        Args:
            name: Name of the middleware to remove

        Returns:
            True if removed, False if not found
        """
        if name not in self._middleware_metadata:
            return False

        metadata = self._middleware_metadata[name]
        middleware_func = metadata["function"]

        if metadata["type"] == "pre":
            self._pre_middleware.remove(middleware_func)
        else:
            self._post_middleware.remove(middleware_func)

        del self._middleware_metadata[name]
        return True

    def process_input(self, data: dict[str, Any], copy: bool = True) -> dict[str, Any]:
        """Process data through pre-middleware pipeline.

        Args:
            data: Input data to process
            copy: Whether to copy the data before processing

        Returns:
            Processed data

        Raises:
            Exception: If ``fail_closed`` is True and a middleware raises, the
                original exception is propagated (the pipeline aborts).
        """
        if not self._pre_middleware:
            return data

        result = data.copy() if copy else data

        for middleware in self._pre_middleware:
            try:
                result = middleware(result)
            except Exception as e:
                if self.fail_closed:
                    logger.error(
                        "Aborting pre-middleware pipeline: %s raised %s " "(fail_closed=True)",
                        middleware.__name__,
                        e,
                    )
                    raise
                # Fail-open: log clearly and continue with unmodified data.
                logger.error(
                    "Error in pre-middleware %s: %s (fail_closed=False, "
                    "passing data through unmodified)",
                    middleware.__name__,
                    e,
                )

        return result

    def process_output(self, data: dict[str, Any], copy: bool = True) -> dict[str, Any]:
        """Process data through post-middleware pipeline.

        Args:
            data: Output data to process
            copy: Whether to copy the data before processing

        Returns:
            Processed data

        Raises:
            Exception: If ``fail_closed`` is True and a middleware raises, the
                original exception is propagated (the pipeline aborts).
        """
        if not self._post_middleware:
            return data

        result = data.copy() if copy else data

        for middleware in self._post_middleware:
            try:
                result = middleware(result)
            except Exception as e:
                if self.fail_closed:
                    logger.error(
                        "Aborting post-middleware pipeline: %s raised %s " "(fail_closed=True)",
                        middleware.__name__,
                        e,
                    )
                    raise
                # Fail-open: log clearly and continue with unmodified data.
                logger.error(
                    "Error in post-middleware %s: %s (fail_closed=False, "
                    "passing data through unmodified)",
                    middleware.__name__,
                    e,
                )

        return result

    def wrap_function(self, func: Callable) -> Callable:
        """Wrap a function with middleware processing.

        Args:
            func: Function to wrap

        Returns:
            Wrapped function
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self._pre_middleware and not self._post_middleware:
                return func(*args, **kwargs)

            # Convert args/kwargs to dict for middleware
            input_data = {"args": args, "kwargs": kwargs}

            # Pre-process
            # ⚡ Bolt: pass copy=False since input_data is a fresh dict to avoid redundant allocations
            processed_input = self.process_input(input_data, copy=False)

            # Execute function
            result = func(
                *processed_input["args"],
                **processed_input["kwargs"],
            )

            # Post-process
            output_data = {"result": result}
            # ⚡ Bolt: pass copy=False since output_data is a fresh dict to avoid redundant allocations
            processed_output = self.process_output(output_data, copy=False)

            return processed_output["result"]

        return wrapper

    def list_middleware(self) -> list[dict[str, Any]]:
        """List all registered middleware.

        Returns:
            List of middleware metadata
        """
        return [{"name": name, **metadata} for name, metadata in self._middleware_metadata.items()]
