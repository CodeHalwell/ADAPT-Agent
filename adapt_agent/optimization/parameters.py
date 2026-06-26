"""Tunable parameters and search spaces for agent optimization.

The optimization subsystem treats *any* agent -- a single "mega" agent, six
specialist agents, an orchestrator with sub-agents, or a multi-step workflow --
as a flat collection of :class:`Parameter` objects. A parameter is one knob that
can be read and written on a live framework object (a prompt, a model name, a
temperature, a few-shot example block, a routing threshold, a tool allow-list).

This module is deliberately framework-agnostic and dependency-free. Framework
*introspection* (turning a CrewAI ``Crew`` or a Pydantic AI ``Agent`` into a list
of parameters) lives in :mod:`adapt_agent.optimization.introspection`; here we
only define the data model and the helpers an optimizer needs to enumerate,
sample, apply, snapshot, and restore parameter values.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ParameterKind(str, Enum):
    """The category of an optimizable knob.

    The string values are stable identifiers usable in configs and reports.
    """

    PROMPT = "prompt"  # system prompts, instructions, role/goal/backstory
    FEW_SHOT = "few_shot"  # in-context example blocks bootstrapped from data
    MODEL = "model"  # model / deployment identifier
    HYPERPARAM = "hyperparam"  # temperature, top_p, max_tokens, ...
    ROUTING = "routing"  # orchestrator routing / handoff / topology knobs
    TOOL = "tool"  # tool / skill allow-lists and selection


@dataclass
class Parameter:
    """A single tunable knob bound to a live agent component.

    A parameter is the unit the optimizer manipulates. It carries both a
    *search space* (discrete ``candidates`` and/or numeric ``bounds``) and the
    *binding* (``getter`` / ``setter``) that lets an optimizer read the current
    value off a framework object and write a new one back in place.

    Args:
        name: Unique, stable identifier, conventionally ``"<component>.<knob>"``
            (e.g. ``"researcher.system_prompt"``). Used as the key in candidate
            configurations and in optimization reports.
        kind: The :class:`ParameterKind` category.
        value: The current value. When a ``getter`` is supplied this is treated
            as a cached default; :meth:`read` always prefers the live getter.
        candidates: Optional explicit list of discrete candidate values forming
            the search space for this parameter.
        bounds: Optional ``(low, high)`` numeric range (inclusive) for continuous
            or integer hyperparameters.
        step: Optional step size used when gridding a numeric ``bounds`` range.
        getter: Optional zero-arg callable returning the live value.
        setter: Optional one-arg callable that writes a value onto the live
            component. A parameter without a setter is read-only and is skipped
            by optimizers (it still appears in reports for transparency).
        component: Optional name of the sub-agent / node this knob belongs to.
        mutable: When ``False`` the optimizer never changes this parameter even
            if a setter exists (useful for pinning a known-good value).
        metadata: Free-form annotations (framework name, source attribute, ...).
    """

    name: str
    kind: ParameterKind
    value: Any = None
    candidates: list[Any] | None = None
    bounds: tuple[float, float] | None = None
    step: float | None = None
    getter: Callable[[], Any] | None = None
    setter: Callable[[Any], None] | None = None
    component: str | None = None
    mutable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Parameter.name must be a non-empty string")
        if not isinstance(self.kind, ParameterKind):
            # Accept the raw string value for ergonomics.
            self.kind = ParameterKind(self.kind)
        if self.bounds is not None:
            low, high = self.bounds
            if low > high:
                raise ValueError(f"Parameter {self.name!r} bounds low > high: {self.bounds!r}")

    @property
    def optimizable(self) -> bool:
        """``True`` if an optimizer is allowed to change this parameter."""
        return self.mutable and self.setter is not None

    def read(self) -> Any:
        """Return the live value (via ``getter``) or the cached ``value``."""
        if self.getter is not None:
            try:
                return self.getter()
            except Exception:
                # A flaky getter must never crash an optimization run; fall back
                # to the last known value.
                return self.value
        return self.value

    def write(self, new_value: Any) -> None:
        """Write ``new_value`` onto the live component and update the cache.

        Raises:
            ValueError: If the parameter has no setter (it is read-only).
        """
        if self.setter is None:
            raise ValueError(f"Parameter {self.name!r} is read-only (no setter)")
        self.setter(new_value)
        self.value = new_value

    def enumerate_candidates(self, *, numeric_points: int = 5) -> list[Any]:
        """Return the discrete search space for this parameter.

        Explicit ``candidates`` take priority. Otherwise, if numeric ``bounds``
        are present, a grid is produced: stepped by ``step`` when given, else
        ``numeric_points`` evenly spaced points. Falls back to the single current
        value when nothing else is known.
        """
        if self.candidates:
            return list(self.candidates)
        if self.bounds is not None:
            low, high = self.bounds
            if self.step:
                points: list[Any] = []
                current = low
                # Guard against zero/negative step producing an infinite loop.
                step = abs(self.step) or (high - low or 1.0)
                while current <= high + 1e-9:
                    points.append(self._coerce_numeric(current))
                    current += step
                return points or [self._coerce_numeric(low)]
            if numeric_points <= 1 or high == low:
                return [self._coerce_numeric(low)]
            span = high - low
            return [
                self._coerce_numeric(low + span * i / (numeric_points - 1))
                for i in range(numeric_points)
            ]
        return [self.read()]

    def sample(self, rng: random.Random) -> Any:
        """Draw a single random value from this parameter's search space."""
        if self.candidates:
            return rng.choice(self.candidates)
        if self.bounds is not None:
            low, high = self.bounds
            raw = rng.uniform(low, high)
            return self._coerce_numeric(raw)
        return self.read()

    def _coerce_numeric(self, raw: float) -> Any:
        """Coerce a grid/sample point to int only for genuinely integer bounds.

        The distinction is by *type*, not value: ``bounds=(1, 50)`` (int literals,
        e.g. ``max_tokens``) yields ints, while ``bounds=(0.0, 2.0)`` (float
        literals, e.g. ``temperature``) stays float -- even though both endpoints
        are whole numbers. This keeps continuous hyperparameters continuous.
        """
        if self.bounds is not None and all(
            isinstance(b, int) and not isinstance(b, bool) for b in self.bounds
        ):
            # Only collapse to int when the step (if any) is also integral.
            if self.step is None or (
                isinstance(self.step, int) and not isinstance(self.step, bool)
            ):
                return int(round(raw))
        return round(raw, 6)


class SearchSpace:
    """An ordered, name-indexed collection of :class:`Parameter` objects.

    Wraps the parameters discovered for an agent and provides the bulk
    operations optimizers rely on: snapshot/restore of live values, applying a
    candidate configuration, filtering by kind/component, and enumerating or
    sampling whole-space configurations.
    """

    def __init__(self, parameters: Iterable[Parameter] = ()):
        self._params: dict[str, Parameter] = {}
        for param in parameters:
            self.add(param)

    def add(self, parameter: Parameter) -> None:
        """Add a parameter, rejecting duplicate names."""
        if parameter.name in self._params:
            raise ValueError(f"Duplicate parameter name: {parameter.name!r}")
        self._params[parameter.name] = parameter

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self._params.values())

    def __len__(self) -> int:
        return len(self._params)

    def __getitem__(self, name: str) -> Parameter:
        return self._params[name]

    def __contains__(self, name: object) -> bool:
        return name in self._params

    @property
    def names(self) -> list[str]:
        return list(self._params)

    def of_kind(self, kind: ParameterKind) -> list[Parameter]:
        """Return parameters of a given :class:`ParameterKind`."""
        return [p for p in self._params.values() if p.kind is kind]

    def of_component(self, component: str) -> list[Parameter]:
        """Return parameters belonging to a named component / sub-agent."""
        return [p for p in self._params.values() if p.component == component]

    def optimizable(self) -> list[Parameter]:
        """Return only the parameters an optimizer may change."""
        return [p for p in self._params.values() if p.optimizable]

    def snapshot(self) -> dict[str, Any]:
        """Capture the current live value of every parameter."""
        return {name: param.read() for name, param in self._params.items()}

    def apply(self, config: dict[str, Any]) -> None:
        """Write a (partial) configuration onto the live components.

        Unknown keys are ignored so a configuration produced for one search
        space can be applied to a compatible subset without error. Read-only
        parameters in ``config`` are silently skipped.
        """
        for name, value in config.items():
            param = self._params.get(name)
            if param is not None and param.optimizable:
                param.write(value)

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore values captured by :meth:`snapshot` (best-effort)."""
        for name, value in snapshot.items():
            param = self._params.get(name)
            if param is not None and param.setter is not None:
                try:
                    param.write(value)
                except Exception:
                    # Restoration must never raise mid-cleanup.
                    param.value = value

    def grid(self, *, numeric_points: int = 5, max_configs: int = 256) -> list[dict[str, Any]]:
        """Enumerate the Cartesian product of all optimizable candidate sets.

        The product is bounded by ``max_configs``; once the running product would
        exceed the cap, remaining parameters are pinned to their current value so
        the returned list never explodes combinatorially.
        """
        configs: list[dict[str, Any]] = [{}]
        for param in self.optimizable():
            options = param.enumerate_candidates(numeric_points=numeric_points)
            if len(options) <= 1:
                continue
            if len(configs) * len(options) > max_configs:
                # Budget exhausted: keep this knob at its current value.
                continue
            configs = [dict(cfg, **{param.name: opt}) for cfg in configs for opt in options]
        return configs

    def sample_config(self, rng: random.Random) -> dict[str, Any]:
        """Draw a random whole-space configuration of optimizable parameters."""
        return {param.name: param.sample(rng) for param in self.optimizable()}


__all__ = ["ParameterKind", "Parameter", "SearchSpace"]
