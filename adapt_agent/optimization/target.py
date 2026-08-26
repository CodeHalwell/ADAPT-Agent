"""The optimization target: wrap arbitrary agent code as a tunable unit.

:class:`OptimizableAgent` is the bridge between *your* agent -- however it is
built -- and the optimizers. It separates two concerns:

* **How to run it.** A single ``runner`` callable ``input -> output`` that drives
  the whole system. This is opaque to the optimizer, so it works identically for
  a single "mega" agent, six specialist agents, an orchestrator delegating to
  sub-agents, or a multi-step workflow, even when the code is spread across many
  files. As long as the runner closes over the live component objects, mutating a
  parameter changes what the next run does.

* **What to tune.** A set of :class:`~adapt_agent.optimization.parameters.Parameter`
  objects, discovered automatically by introspecting the framework
  ``components`` you register (see
  :mod:`adapt_agent.optimization.introspection`) and/or declared explicitly for
  knobs no framework exposes (routing thresholds, few-shot blocks, custom flags).

Example -- an orchestrator with two specialist sub-agents::

    from adapt_agent.optimization import OptimizableAgent

    target = OptimizableAgent.from_components(
        components={"researcher": researcher_agent, "writer": writer_agent},
        runner=lambda q: orchestrator.handle(q),   # uses the live sub-agents
        name="research-writer",
    )
    target.parameters            # -> researcher.system_prompt, writer.model, ...
    out = target.run("Summarise the news")
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from adapt_agent.optimization.evaluation import _RUN_METHOD_NAMES, resolve_runner
from adapt_agent.optimization.introspection import introspect_components
from adapt_agent.optimization.parameters import Parameter, ParameterKind, SearchSpace


class OptimizableAgent:
    """An agent (any architecture) exposed as a runnable + a tunable search space.

    Prefer the ``from_*`` constructors; the initializer is the low-level form.

    Args:
        runner: Callable mapping an input to an output. Drives the whole system.
        parameters: Explicit parameters to expose (merged with introspected ones).
        components: ``{name: framework_object}`` to introspect for parameters.
        name: Human-readable identifier used in reports.
        exclude: Introspected parameter names to drop before ``parameters`` is
            merged in. For when a framework upgrade starts introspecting a knob
            you already hand-bound *under a different name* onto the same
            underlying storage (e.g. your own parameter reads/writes
            ``options.default_options["instructions"]`` as
            ``"agent.system_prompt"``, and introspection now also discovers it
            as ``"agent.instructions"``): same storage, different names, so the
            duplicate-name check below never catches it, and the optimizer
            silently gets two knobs for one prompt -- doubling that part of the
            search and letting whichever knob is tried last overwrite the
            other's candidate. Naming the newly-introspected one here removes
            it from the space entirely, leaving your hand-bound parameter as
            the sole knob for that storage.
        replace: When ``True``, a declared parameter in ``parameters`` whose
            name collides with an introspected one *replaces* it instead of
            raising. Default ``False`` keeps the collision a hard error, since
            an unintended same-name clash (a typo, two components introspecting
            to the same name) is far more often a real mistake than an
            intentional override.
    """

    def __init__(
        self,
        runner: Callable[[Any], Any],
        *,
        parameters: list[Parameter] | None = None,
        components: dict[str, Any] | None = None,
        name: str = "agent",
        exclude: Iterable[str] | None = None,
        replace: bool = False,
    ):
        if not callable(runner):
            raise TypeError("OptimizableAgent runner must be callable")
        self._runner = runner
        self.name = name
        self.components: dict[str, Any] = dict(components or {})

        space = SearchSpace()
        # Introspected parameters first (stable, framework-derived names) ...
        for param in introspect_components(self.components):
            space.add(param)
        # ... minus any the caller wants dropped before anything else sees them
        # (a name absent from the space -- introspection changed, or never
        # found it -- is not an error: excluding a knob that turned out not to
        # exist is a no-op, not a mistake).
        for excluded_name in exclude or ():
            space.remove(excluded_name)
        # ... then explicit declarations (allowed to add knobs frameworks miss,
        # and -- with replace=True -- to override how an introspected one is
        # read/written without renaming it).
        for param in parameters or []:
            if param.name in space and not replace:
                raise ValueError(
                    f"Declared parameter {param.name!r} collides with an "
                    f"introspected parameter; rename it to disambiguate, pass "
                    f"replace=True to override it, or exclude={{{param.name!r}}} "
                    f"to drop the introspected one first."
                )
            space.add(param, replace=replace)
        self._space = space

    # -- constructors ----------------------------------------------------------

    @classmethod
    def from_components(
        cls,
        components: dict[str, Any],
        *,
        runner: Callable[[Any], Any] | None = None,
        parameters: list[Parameter] | None = None,
        name: str = "agent",
        exclude: Iterable[str] | None = None,
        replace: bool = False,
    ) -> OptimizableAgent:
        """Build from named framework objects.

        If ``runner`` is omitted and exactly one component is directly runnable
        (callable, has ``run``, or has ``execute``), it becomes the runner.
        See the class docstring for ``exclude``/``replace``.
        """
        if runner is None:
            runner = cls._infer_runner(components)
        return cls(
            runner,
            parameters=parameters,
            components=components,
            name=name,
            exclude=exclude,
            replace=replace,
        )

    @classmethod
    def from_agent(
        cls,
        agent: Any,
        *,
        runner: Callable[[Any], Any] | None = None,
        component_name: str = "agent",
        parameters: list[Parameter] | None = None,
        name: str = "agent",
        exclude: Iterable[str] | None = None,
        replace: bool = False,
    ) -> OptimizableAgent:
        """Build from a single framework object (the common single-agent case).

        When no ``runner`` is given, a framework-native run method is resolved
        (``run_sync``/``invoke``/``kickoff``/``run``, a bare callable, a
        governed ``execute``). An object with none of those -- an OpenAI Agents
        ``Agent`` (the SDK drives it via ``Runner.run_sync``), a Claude Agent
        SDK options object (driven via ``query``), a bare Google ADK agent
        (driven inside a ``Runner``) -- falls back to
        :func:`~adapt_agent.optimization.runners.framework_runner`, which knows
        each framework's driving machinery and unwraps results to final
        response text. See the class docstring for ``exclude``/``replace``.
        """
        run = runner if runner is not None else _default_agent_runner(agent)
        return cls(
            run,
            parameters=parameters,
            components={component_name: agent},
            name=name,
            exclude=exclude,
            replace=replace,
        )

    @classmethod
    def from_callable(
        cls,
        runner: Callable[[Any], Any],
        *,
        parameters: list[Parameter] | None = None,
        components: dict[str, Any] | None = None,
        name: str = "agent",
        exclude: Iterable[str] | None = None,
        replace: bool = False,
    ) -> OptimizableAgent:
        """Build from a plain runner plus optional components/parameters."""
        return cls(
            runner,
            parameters=parameters,
            components=components,
            name=name,
            exclude=exclude,
            replace=replace,
        )

    @staticmethod
    def _infer_runner(components: dict[str, Any]) -> Callable[[Any], Any]:
        runnable = [obj for obj in components.values() if _is_runnable(obj)]
        if len(runnable) == 1:
            return resolve_runner(runnable[0])
        raise ValueError(
            "Could not infer a runner: supply `runner=` explicitly. A runner is "
            "only inferred when exactly one component is directly runnable "
            f"(found {len(runnable)})."
        )

    # -- running ---------------------------------------------------------------

    def run(self, input_data: Any) -> Any:
        """Execute the wrapped system on a single input."""
        return self._runner(input_data)

    __call__ = run

    # -- parameters ------------------------------------------------------------

    @property
    def search_space(self) -> SearchSpace:
        return self._space

    @property
    def parameters(self) -> list[Parameter]:
        return list(self._space)

    def parameters_of_kind(self, kind: ParameterKind) -> list[Parameter]:
        return self._space.of_kind(kind)

    def add_parameter(self, parameter: Parameter, *, replace: bool = False) -> None:
        """Declare an extra parameter after construction.

        Raises :class:`ValueError` if ``parameter.name`` is already taken
        (introspected or previously added) unless ``replace=True``, which
        drops the existing one first -- see the class docstring's ``replace``
        for when that is the right call. To remove an introspected knob
        entirely with no replacement, use ``search_space.remove(name)``.
        """
        self._space.add(parameter, replace=replace)

    def add_tool_parameter(
        self,
        name: str,
        *,
        kind: ParameterKind = ParameterKind.TOOL,
        getter: Callable[[], Any],
        setter: Callable[[Any], None],
        candidates: list[Any] | None = None,
        candidate_tools: list[Any] | None = None,
        component: str | None = None,
    ) -> Parameter:
        """Register a tool/skill selection knob as a real search space.

        This is the one-call convenience for making the set of tools (or
        higher-level skills/plugins) an agent has access to *optimizable*. The
        ``getter``/``setter`` bind to wherever the live tool list lives (e.g. an
        agent's ``tools`` attribute), so the optimizer can swap the active set in
        place between runs.

        The candidate search space is resolved in priority order:

        1. Explicit ``candidates`` (a list of tool-list values) if supplied.
        2. Otherwise, if ``candidate_tools`` (the full pool of available tool
           objects/names) is given, *drop-one ablation* subsets are derived via
           :func:`adapt_agent.optimization.introspection.tool_subset_candidates`
           -- the full set first, then each subset missing one tool. This makes
           tool/skill *selection* a genuine search space rather than a fixed list.

        Args:
            name: Unique parameter name, conventionally ``"<component>.tools"``.
            kind: :attr:`ParameterKind.TOOL` (default) or
                :attr:`ParameterKind.SKILL` for higher-level named skills.
            getter: Zero-arg callable returning the live tool/skill collection.
            setter: One-arg callable writing a new collection onto the component.
            candidates: Optional explicit list of candidate collections.
            candidate_tools: Optional pool of available tools to derive drop-one
                ablation subsets from when ``candidates`` is not supplied.
            component: Optional owning sub-agent / node name.

        Returns:
            The :class:`Parameter` that was added to the search space.
        """
        if candidates is None and candidate_tools is not None:
            tools = list(candidate_tools)
            try:
                from adapt_agent.optimization.introspection import (
                    tool_subset_candidates,
                )

                candidates = tool_subset_candidates(tools)
            except ImportError:
                # Helper not available yet; fall back to the full set as the
                # single candidate so the parameter is still well-formed.
                candidates = [tools]
            if not candidates:
                # ``tool_subset_candidates`` returns [] for <2 tools; keep the
                # current set as the sole candidate.
                candidates = [tools]

        param = Parameter(
            name=name,
            kind=kind,
            value=getter(),
            candidates=candidates,
            getter=getter,
            setter=setter,
            component=component,
        )
        self._space.add(param)
        return param

    def snapshot(self) -> dict[str, Any]:
        """Capture current values of all parameters (for restore)."""
        return self._space.snapshot()

    def apply(self, config: dict[str, Any]) -> None:
        """Apply a candidate configuration to the live components."""
        self._space.apply(config)

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore a previously captured snapshot."""
        self._space.restore(snapshot)

    def describe(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the target and its search space."""
        params = []
        for p in self._space:
            params.append(
                {
                    "name": p.name,
                    "kind": p.kind.value,
                    "component": p.component,
                    "optimizable": p.optimizable,
                    "n_candidates": len(p.candidates) if p.candidates else None,
                    "bounds": p.bounds,
                }
            )
        return {
            "name": self.name,
            "components": list(self.components),
            "n_parameters": len(self._space),
            "n_optimizable": len(self._space.optimizable()),
            "parameters": params,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"OptimizableAgent(name={self.name!r}, components={list(self.components)}, "
            f"params={len(self._space)}, optimizable={len(self._space.optimizable())})"
        )


def _default_agent_runner(agent: Any) -> Callable[[Any], Any]:
    """Resolve how to drive ``agent`` when the caller supplied no runner.

    A framework-native run method (or bare callable / governed ``execute``)
    wins, preserving framework-native outputs exactly as before. Only when
    nothing is directly runnable does this fall back to
    :func:`~adapt_agent.optimization.runners.framework_runner`, which delegates
    by detected framework (OpenAI Agents ``Runner``, Claude Agent SDK
    ``query``, a Google ADK ``Runner``) and extracts final response text --
    previously these agents raised ``TypeError`` here and forced every caller
    to hand-write the driving lambda the SDK documents anyway.
    """
    try:
        return resolve_runner(agent)
    except TypeError:
        # Imported lazily to keep this module import-light; framework_runner
        # re-raises the same TypeError when the object is not a recognisable
        # framework either, so the error surface is unchanged for junk inputs.
        from adapt_agent.optimization.runners import framework_runner

        return framework_runner(agent)


def _is_runnable(obj: Any) -> bool:
    if callable(obj):
        return True
    # Recognize the same entrypoints as ``resolve_runner`` (framework-native run
    # methods + the governed ``execute``) so a single framework component can be
    # inferred as the runner.
    return any(callable(getattr(obj, name, None)) for name in (*_RUN_METHOD_NAMES, "execute"))


def wrap(
    target: OptimizableAgent | Any,
    *,
    runner: Callable[[Any], Any] | None = None,
    components: dict[str, Any] | None = None,
    parameters: list[Parameter] | None = None,
    name: str = "agent",
    exclude: Iterable[str] | None = None,
    replace: bool = False,
) -> OptimizableAgent:
    """Coerce a variety of inputs into an :class:`OptimizableAgent`.

    Accepts an existing :class:`OptimizableAgent` (returned as-is -- ``exclude``/
    ``replace`` are then ignored, since its search space is already built; pass
    them to the ``from_*`` constructor that built it instead), a plain runner
    callable, or a single framework object. This is the convenience entry
    optimizers use so users can pass whatever they have. See
    :class:`OptimizableAgent` for what ``exclude``/``replace`` do.
    """
    if isinstance(target, OptimizableAgent):
        return target
    if components is not None:
        return OptimizableAgent.from_components(
            components,
            runner=runner or (target if callable(target) else None),
            parameters=parameters,
            name=name,
            exclude=exclude,
            replace=replace,
        )
    if callable(target) and runner is None:
        return OptimizableAgent.from_callable(
            target, parameters=parameters, name=name, exclude=exclude, replace=replace
        )
    return OptimizableAgent.from_agent(
        target, runner=runner, parameters=parameters, name=name, exclude=exclude, replace=replace
    )


__all__ = ["OptimizableAgent", "wrap"]
