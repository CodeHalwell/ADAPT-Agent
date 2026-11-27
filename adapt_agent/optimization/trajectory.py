"""Trajectory metrics: score what an agent *did*, not only what it said.

An agent that answers "It is sunny in Paris" without calling the weather tool
has guessed; one that calls ``delete_db`` on the way to a correct answer has
failed. Text metrics cannot tell either from a good run. These metrics read the
:class:`~adapt_agent.optimization.extractors.Trajectory` -- the tool calls, in
order, with their arguments -- recovered from the framework-native result by
:func:`~adapt_agent.optimization.extractors.extract_trajectory`.

A row declares its expectations in the dataset, next to ``input`` / ``expected``
-- the same place a per-row ``check`` lives::

    {"input": "Weather in Paris?", "expected": "sunny",
     "expected_tools": [{"name": "get_weather", "args": {"city": "Paris"}}],
     "forbidden_tools": ["delete_*"], "max_tool_calls": 3}

Expectations a metric reads from the row:

* ``expected_tools`` -- a list of tool names or ``{"name": ..., "args": {...}}``
  mappings (``arguments`` is accepted for ``args``).
* ``forbidden_tools`` -- tool names or glob patterns that must not be called.
* ``max_tool_calls`` / ``max_steps`` -- budgets.

Every metric also takes its expectations as factory arguments, which win over
the row. A row that declares nothing a metric needs is an *error* for that row
(recorded by the harness), never a silent pass: a trajectory check with no
expectation would otherwise report 1.0 for every run.

How the harness feeds them: an
:class:`~adapt_agent.optimization.evaluation.EvaluationHarness` keeps the *raw*
result of each run (before ``output_extractor`` flattens it to text) and these
metrics recover the trajectory from it, so they compose with text metrics and
with the per-row :func:`~adapt_agent.optimization.metrics.checks` dispatcher
(``"check": "tool_sequence"``). Called directly, they accept a raw framework
result, a :class:`Trajectory`, or the custom-agent dict form
``{"output": ..., "tool_calls": [...]}``.
"""

from __future__ import annotations

import contextvars
import fnmatch
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from .extractors import ToolCall, Trajectory, extract_trajectory
from .metrics import Metric, get_metric, metric_factory, register_metric

#: The raw (pre-extraction) agent result for the row being scored. Set by the
#: harness around scoring; a ContextVar so concurrent rows -- threads, and
#: ``asyncio.to_thread`` workers, which copy the context -- each see their own.
_RAW_OUTPUT: contextvars.ContextVar[Any] = contextvars.ContextVar("adapt_eval_raw_output")
_TRAJECTORY_CACHE: contextvars.ContextVar[dict[int, Trajectory]] = contextvars.ContextVar(
    "adapt_eval_trajectory_cache"
)

_SEQUENCE_MODES = ("exact", "in_order", "any_order", "no_extra")


@contextmanager
def scoring_raw_output(raw: Any) -> Iterator[None]:
    """Make ``raw`` the result trajectory metrics read while scoring one row."""
    token = _RAW_OUTPUT.set(raw)
    cache_token = _TRAJECTORY_CACHE.set({})
    try:
        yield
    finally:
        _RAW_OUTPUT.reset(token)
        _TRAJECTORY_CACHE.reset(cache_token)


def trajectory_of(output: Any) -> Trajectory:
    """The trajectory for ``output``: the row's raw result when scoring in a harness.

    A :class:`Trajectory` passed in is used as-is. Inside a harness the raw
    result wins over ``output``, because ``output`` has usually been flattened
    to text by the time a metric sees it.
    """
    if isinstance(output, Trajectory):
        return output
    raw = _RAW_OUTPUT.get(_MISSING)
    source = output if raw is _MISSING else raw
    cache = _TRAJECTORY_CACHE.get(None)
    if cache is not None:
        cached = cache.get(id(source))
        if cached is not None:
            return cached
    trajectory = extract_trajectory(source)
    if cache is not None:
        cache[id(source)] = trajectory
    return trajectory


_MISSING = object()


# -- reading expectations ------------------------------------------------------


def _meta(example: Any) -> Mapping[str, Any]:
    meta = getattr(example, "metadata", None)
    return meta if isinstance(meta, Mapping) else {}


def _expected_tools(example: Any, override: Any, metric: str) -> list[dict[str, Any]]:
    spec = override if override is not None else _meta(example).get("expected_tools")
    if spec is None:
        raise ValueError(
            f"{metric}: this row declares no 'expected_tools' and none were passed to the "
            "metric; add expected_tools to the row, or apply the metric only to rows that "
            "declare it (per-row 'check')"
        )
    if isinstance(spec, (str, Mapping)):
        spec = [spec]
    if not isinstance(spec, Sequence):
        raise ValueError(f"{metric}: expected_tools must be a list, got {type(spec).__name__}")
    out: list[dict[str, Any]] = []
    for item in spec:
        if isinstance(item, str):
            out.append({"name": item, "args": None})
        elif isinstance(item, ToolCall):
            out.append({"name": item.name, "args": dict(item.arguments)})
        elif isinstance(item, Mapping) and isinstance(item.get("name"), str):
            args = item.get("args", item.get("arguments"))
            if args is not None and not isinstance(args, Mapping):
                raise ValueError(f"{metric}: args for {item['name']!r} must be a mapping")
            out.append({"name": item["name"], "args": None if args is None else dict(args)})
        else:
            raise ValueError(f"{metric}: cannot read an expected tool from {item!r}")
    return out


# -- argument matching ---------------------------------------------------------


def _arg_matches(expected: Any, actual: Any, default_check: Metric) -> bool:
    """Whether one argument value satisfies its expectation.

    ``expected`` may be a plain value -- strings are compared with the metric's
    text check (``exact_match`` by default, so case and edge punctuation do not
    matter), other values by equality with ``True != 1`` -- or a matcher
    mapping naming any built-in text metric::

        {"$check": "contains", "value": "weather"}
        {"$check": "numeric_close", "value": 42, "tolerance": 0.5}
        {"$check": "regex_match", "value": "^paris$"}
    """
    if isinstance(expected, Mapping) and "$check" in expected:
        params = dict(expected)
        name = params.pop("$check")
        value = params.pop("value", None)
        factory = metric_factory(name)
        if factory is None:
            raise ValueError(f"unknown argument check {name!r}")
        matcher = factory(**params)
        return matcher(_text_or_value(actual), value) >= 1.0
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(
            k in actual and _arg_matches(v, actual[k], default_check) for k, v in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return False
        return all(_arg_matches(e, a, default_check) for e, a in zip(expected, actual, strict=True))
    if isinstance(expected, str):
        return isinstance(actual, (str, int, float)) and default_check(str(actual), expected) >= 1.0
    if isinstance(expected, bool) != isinstance(actual, bool):
        return False
    return bool(expected == actual)


def _text_or_value(value: Any) -> Any:
    return value if isinstance(value, (str, int, float)) else str(value)


def _args_subset(
    expected: Mapping[str, Any] | None, actual: Mapping[str, Any], check: Metric
) -> bool:
    if not expected:
        return True
    return all(k in actual and _arg_matches(v, actual[k], check) for k, v in expected.items())


# -- metrics -------------------------------------------------------------------


def tool_called(name: str | None = None, *, min_times: int = 1) -> Metric:
    """1.0 iff the agent called the tool(s).

    With ``name``, that one tool must be called at least ``min_times`` times.
    Without it, every tool in the row's ``expected_tools`` must be called (any
    order; extra calls allowed) and the score is the fraction that were.
    """

    def _fn(output: Any, expected: Any, example: Any = None) -> float:
        names = trajectory_of(output).tool_names
        if name is not None:
            return 1.0 if names.count(name) >= min_times else 0.0
        wanted = [t["name"] for t in _expected_tools(example, None, "tool_called")]
        if not wanted:
            return 1.0
        return sum(1 for w in wanted if w in names) / len(wanted)

    return Metric("tool_called", _fn, needs_example=True)


def tool_call_args_match(
    expected_tools: Sequence[Any] | None = None, *, check: str = "exact_match"
) -> Metric:
    """Fraction of expected calls made with (at least) the expected arguments.

    Each expected ``{"name": ..., "args": {...}}`` is matched against the
    agent's calls to that tool: the expected arguments must be a *subset* of
    some call's arguments (extra arguments are fine), each value compared with
    the ``check`` text metric for strings, equality otherwise, or an explicit
    ``{"$check": "...", "value": ...}`` matcher. Each actual call satisfies at
    most one expectation, so two expected identical calls need two real ones.
    """
    text_check = get_metric(check)

    def _fn(output: Any, expected: Any, example: Any = None) -> float:
        wanted = _expected_tools(example, expected_tools, "tool_call_args_match")
        if not wanted:
            return 1.0
        calls = list(trajectory_of(output).tool_calls)
        used: set[int] = set()
        matched = 0
        for spec in wanted:
            for i, call in enumerate(calls):
                if i in used or call.name != spec["name"]:
                    continue
                if _args_subset(spec["args"], call.arguments, text_check):
                    used.add(i)
                    matched += 1
                    break
        return matched / len(wanted)

    return Metric("tool_call_args_match", _fn, needs_example=True)


def tool_sequence(
    expected_tools: Sequence[Any] | None = None,
    *,
    mode: str = "in_order",
    check_args: bool = False,
    check: str = "exact_match",
) -> Metric:
    """1.0 iff the agent's tool calls match the expected sequence under ``mode``.

    Modes:

    * ``"exact"`` -- the calls are exactly the expected list, in order, nothing
      else.
    * ``"in_order"`` (default) -- the expected calls appear in this order;
      other calls may be interleaved.
    * ``"any_order"`` -- every expected call appears (as a multiset); order and
      extra calls do not matter.
    * ``"no_extra"`` -- every expected call appears, in any order, and no call
      outside the expected set was made.

    With ``check_args=True`` a call only counts as an expected one when its
    arguments satisfy the expectation's ``args`` (as in
    :func:`tool_call_args_match`); otherwise only names are compared.
    """
    if mode not in _SEQUENCE_MODES:
        raise ValueError(f"mode must be one of {_SEQUENCE_MODES}, got {mode!r}")
    text_check = get_metric(check)

    def _fn(output: Any, expected: Any, example: Any = None) -> float:
        wanted = _expected_tools(example, expected_tools, "tool_sequence")
        calls = list(trajectory_of(output).tool_calls)

        def ok(spec: dict[str, Any], call: ToolCall) -> bool:
            if call.name != spec["name"]:
                return False
            return not check_args or _args_subset(spec["args"], call.arguments, text_check)

        if mode == "exact":
            same = len(calls) == len(wanted) and all(
                ok(s, c) for s, c in zip(wanted, calls, strict=True)
            )
            return 1.0 if same else 0.0
        if mode == "in_order":
            position = 0
            for call in calls:
                if position < len(wanted) and ok(wanted[position], call):
                    position += 1
            return 1.0 if position == len(wanted) else 0.0
        used: set[int] = set()
        for spec in wanted:
            hit = next((i for i, c in enumerate(calls) if i not in used and ok(spec, c)), None)
            if hit is None:
                return 0.0
            used.add(hit)
        if mode == "no_extra":
            allowed = {s["name"] for s in wanted}
            if any(c.name not in allowed for c in calls):
                return 0.0
        return 1.0

    return Metric("tool_sequence", _fn, needs_example=True)


def no_forbidden_tools(forbidden: Sequence[str] | None = None) -> Metric:
    """1.0 iff no forbidden tool was called.

    ``forbidden`` (or the row's ``forbidden_tools``) lists names or glob
    patterns -- ``"delete_*"`` forbids every tool starting with ``delete_``.
    """

    def _fn(output: Any, expected: Any, example: Any = None) -> float:
        patterns = forbidden if forbidden is not None else _meta(example).get("forbidden_tools")
        if patterns is None:
            raise ValueError(
                "no_forbidden_tools: this row declares no 'forbidden_tools' and none were "
                "passed to the metric"
            )
        if isinstance(patterns, str):
            patterns = [patterns]
        names = trajectory_of(output).tool_names
        hit = any(fnmatch.fnmatchcase(n, p) for n in names for p in patterns)
        return 0.0 if hit else 1.0

    return Metric("no_forbidden_tools", _fn, needs_example=True)


def tool_call_budget(max_calls: int | None = None, *, max_steps: int | None = None) -> Metric:
    """1.0 iff the run stayed within its tool-call and/or step budget.

    ``max_calls`` / ``max_steps`` (or the row's ``max_tool_calls`` /
    ``max_steps``). A step is one model response; when the result does not
    expose its step count the row is an error rather than a guess.
    """

    def _fn(output: Any, expected: Any, example: Any = None) -> float:
        meta = _meta(example)
        calls_cap = max_calls if max_calls is not None else meta.get("max_tool_calls")
        steps_cap = max_steps if max_steps is not None else meta.get("max_steps")
        if calls_cap is None and steps_cap is None:
            raise ValueError(
                "tool_call_budget: no budget -- pass max_calls/max_steps or declare "
                "max_tool_calls/max_steps on the row"
            )
        trajectory = trajectory_of(output)
        if calls_cap is not None and len(trajectory.tool_calls) > int(calls_cap):
            return 0.0
        if steps_cap is not None:
            if trajectory.steps is None:
                raise ValueError(
                    "tool_call_budget: this result does not expose a step count; budget "
                    "tool calls with max_tool_calls instead"
                )
            if trajectory.steps > int(steps_cap):
                return 0.0
        return 1.0

    return Metric("tool_call_budget", _fn, needs_example=True)


#: Trajectory metrics by name, for datasets, configs and the CLI.
TRAJECTORY_METRICS: dict[str, Callable[..., Metric]] = {
    "tool_called": tool_called,
    "tool_call_args_match": tool_call_args_match,
    "tool_sequence": tool_sequence,
    "no_forbidden_tools": no_forbidden_tools,
    "tool_call_budget": tool_call_budget,
}

# Registered by name so a row can declare `"check": "tool_sequence"` (or
# `{"name": "tool_sequence", "mode": "exact"}`) and `get_metric` finds them.
for _name, _factory in TRAJECTORY_METRICS.items():
    register_metric(_name, _factory)


__all__ = [
    "TRAJECTORY_METRICS",
    "ToolCall",
    "Trajectory",
    "extract_trajectory",
    "no_forbidden_tools",
    "scoring_raw_output",
    "tool_call_args_match",
    "tool_call_budget",
    "tool_called",
    "tool_sequence",
    "trajectory_of",
]
