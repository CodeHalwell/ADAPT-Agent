"""CrewAI adapter for ADAPT-Agent.

`CrewAI <https://docs.crewai.com>`_ orchestrates a ``Crew`` of agents that you
run with ``crew.kickoff(inputs=...)`` (sync) or ``crew.kickoff_async(...)``. The
result is a ``CrewOutput`` whose final text is on ``.raw``.

This adapter wraps a ``Crew`` (or anything exposing a callable ``kickoff`` /
``kickoff_async`` / ``run``) and applies ADAPT-Agent's governance pipeline. The
payload's ``context`` (its non-``messages`` keys) is forwarded to ``kickoff`` as
the ``inputs`` mapping of template variables when the runner accepts it.
Importing this module never imports ``crewai``.

Example
-------
>>> from crewai import Crew                      # doctest: +SKIP
>>> crew = Crew(agents=[...], tasks=[...])        # doctest: +SKIP
>>> from adapt_agent.adapters import CrewAIAdapter
>>> from adapt_agent.security import Firewall
>>> adapter = CrewAIAdapter(firewall=Firewall())
>>> guarded = adapter.wrap_agent(crew)            # doctest: +SKIP
>>> guarded.execute({"messages": [{"role": "user", "content": "hi"}], "topic": "AI"})  # doctest: +SKIP
"""

import inspect
from typing import Any

from adapt_agent.adapters._governed import GovernedAdapter, Runner


class CrewAIAdapter(GovernedAdapter):
    """Adapter for integrating ADAPT-Agent with CrewAI crews.

    Wrap a ``Crew`` (or any object exposing a callable ``kickoff`` /
    ``kickoff_async`` / ``run``). See
    :class:`~adapt_agent.adapters._governed.GovernedAdapter` for the full
    constructor signature.
    """

    framework_name = "CrewAI"
    run_method_names = ("kickoff", "kickoff_async", "run")
    operation = "crewai.kickoff"

    def _prepare_input(self, payload: dict[str, Any]) -> Any:
        # CrewAI's ``inputs`` are template variables, not a chat transcript.
        return {k: v for k, v in payload.items() if k != "messages"}

    def _call_runner(self, runner: Runner, payload: dict[str, Any]) -> Any:
        inputs = self._prepare_input(payload)
        # Prefer the idiomatic ``kickoff(inputs=...)`` call when supported.
        if self._accepts_kwarg(runner, "inputs"):
            return runner(inputs=inputs)
        return runner(inputs)

    @staticmethod
    def _accepts_kwarg(runner: Runner, name: str) -> bool:
        try:
            signature = inspect.signature(runner)
        except (TypeError, ValueError):
            return False
        for param in signature.parameters.values():
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                return True
            if param.name == name and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                return True
        return False


__all__ = ["CrewAIAdapter"]
