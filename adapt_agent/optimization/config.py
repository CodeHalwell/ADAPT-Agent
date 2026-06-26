"""Declarative YAML/JSON training configuration for agent optimization.

This module turns a single config file into a fully wired optimization run, so a
user can "train" their agent system -- tune prompts, few-shot blocks, models,
hyperparameters, routing knobs, and **tool/skill allow-lists** -- and harden it
against unwanted behaviour, without writing glue code.

A config file describes five things:

* ``target`` -- how to run the system (an entrypoint ``module:attribute``) and,
  optionally, the named ``components`` (sub-agents) to introspect for tunable knobs.
* ``dataset`` -- the golden data (path + format + optional column overrides).
* ``judge`` -- the provider-agnostic LLM-as-judge, used both to score outputs and,
  when ``adversarial: true``, to act as a harsh critic/adversary that drives prompt
  rewrites and proposes new tools/skills.
* ``metrics`` / ``primary_metric`` -- how runs are scored.
* ``optimizer`` -- the search strategy and budget.
* ``parameters`` -- optional explicit knobs the framework does not expose
  (routing thresholds, magentic round limits, a model candidate pool, ...). Each
  binds to a live attribute on a resolved component via ``attr`` / ``attr_path``.

Validation is friendly and **fail-soft where it should be**: a temperature bound
that exceeds the provider's allowable range is clamped with a warning rather than
crashing the run; unknown metric/provider/optimizer names raise a clear
:class:`TrainingConfigError`.

Example (YAML)::

    target:
      entrypoint: "myapp.app:run"          # callable input -> output
      components:                          # introspected for tunable knobs
        manager:    "myapp.agents:manager"
        researcher: "myapp.agents:researcher"
    dataset:
      path: "golden.jsonl"
      format: jsonl
    judge:
      provider: anthropic
      model: claude-opus-4-8
      adversarial: true                    # the judge IS the adversary
    metrics: [exact_match, token_f1]
    optimizer:
      type: default
      max_evals: 60
      suggest_tools: true                  # let the judge propose new tools/skills
    parameters:
      - name: workflow.max_round_count
        kind: routing
        component: workflow
        attr: max_round_count
        bounds: [4, 12]
        step: 1
      - name: manager.temperature
        kind: hyperparam
        component: manager
        attr_path: chat_client.temperature
        bounds: [0.0, 1.5]                 # clamped to the provider max if needed

Then::

    from adapt_agent.optimization.config import run_training
    result = run_training("train.yaml")
    print(result)                 # baseline vs best, applied in place
    for tip in result.recommendations:
        print(tip)                # judge-proposed new tools/skills, etc.
"""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapt_agent.optimization.dataset import GoldenDataset
from adapt_agent.optimization.evaluation import EvaluationHarness
from adapt_agent.optimization.metrics import BUILTIN_METRICS, Metric, get_metric
from adapt_agent.optimization.optimizers import (
    BootstrapFewShotOptimizer,
    CoordinateAscentOptimizer,
    EvolutionaryOptimizer,
    GridSearchOptimizer,
    OptimizationResult,
    Optimizer,
    RandomSearchOptimizer,
    make_default_optimizer,
)
from adapt_agent.optimization.parameters import Parameter, ParameterKind
from adapt_agent.optimization.target import OptimizableAgent

logger = logging.getLogger(__name__)

#: Hard ceiling for any sampling temperature when the provider's max is unknown.
_DEFAULT_MAX_TEMPERATURE = 2.0

_OPTIMIZER_TYPES: dict[str, Any] = {
    "default": "default",  # special-cased -> make_default_optimizer
    "coordinate_ascent": CoordinateAscentOptimizer,
    "grid": GridSearchOptimizer,
    "random": RandomSearchOptimizer,
    "evolutionary": EvolutionaryOptimizer,
    "bootstrap_few_shot": BootstrapFewShotOptimizer,
}


class TrainingConfigError(ValueError):
    """Raised when a training configuration is invalid or cannot be wired up."""


# --------------------------------------------------------------------------- #
# Config dataclasses                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class ParameterSpec:
    """A single explicit tunable knob declared in the config.

    Binds to ``getattr(component, attr)`` (or a dotted ``attr_path``) on a resolved
    component object, so the optimizer can read and write it in place.
    """

    name: str
    kind: str
    component: str | None = None
    attr: str | None = None
    attr_path: str | None = None
    bounds: tuple[float, float] | None = None
    candidates: list[Any] | None = None
    step: float | None = None
    max_temperature: float | None = None


@dataclass
class TargetSpec:
    entrypoint: str | None = None
    runner: str | None = None
    components: dict[str, str] = field(default_factory=dict)
    name: str = "agent"


@dataclass
class DatasetSpec:
    path: str
    format: str | None = None
    input_key: str | None = None
    expected_key: str | None = None
    val_path: str | None = None


@dataclass
class JudgeSpec:
    provider: str
    model: str | None = None
    adversarial: bool = False
    scale: int = 10
    pass_threshold: float = 0.6
    score_is_normalized: bool = False
    criteria: str | None = None
    metric_name: str = "quality"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerSpec:
    type: str = "default"
    max_evals: int = 60
    min_improvement: float | None = None
    seed: int | None = 0
    kinds: list[str] | None = None
    suggest_tools: bool = True
    verbose: bool = False


@dataclass
class TrainingConfig:
    """A fully-parsed training configuration (see module docstring)."""

    target: TargetSpec
    dataset: DatasetSpec
    judge: JudgeSpec | None = None
    metrics: list[str] = field(default_factory=lambda: ["exact_match"])
    primary_metric: str | None = None
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)
    parameters: list[ParameterSpec] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Loading & parsing                                                           #
# --------------------------------------------------------------------------- #


def load_training_config(path: str | Path) -> TrainingConfig:
    """Parse a YAML (or JSON) training config file into a :class:`TrainingConfig`.

    Raises:
        TrainingConfigError: if the file is missing, unparseable, or invalid.
    """
    p = Path(path)
    if not p.exists():
        raise TrainingConfigError(f"Config file not found: {p}")
    raw = _load_mapping(p)
    return parse_training_config(raw)


def _load_mapping(p: Path) -> dict[str, Any]:
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - pyyaml is a core dep
            raise TrainingConfigError(
                "PyYAML is required to read YAML configs. Install adapt-agent "
                "(which depends on pyyaml) or use a .json config."
            ) from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise TrainingConfigError(f"Invalid YAML in {p}: {exc}") from exc
    elif suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TrainingConfigError(f"Invalid JSON in {p}: {exc}") from exc
    else:
        # Best effort: try YAML (it is a superset of JSON).
        try:
            import yaml

            data = yaml.safe_load(text)
        except Exception as exc:  # noqa: BLE001
            raise TrainingConfigError(
                f"Unknown config format for {p!r} (use .yaml/.yml/.json): {exc}"
            ) from exc
    if not isinstance(data, dict):
        raise TrainingConfigError(f"Config root must be a mapping, got {type(data).__name__}")
    return data


def parse_training_config(raw: dict[str, Any]) -> TrainingConfig:
    """Validate and structure a raw config mapping."""
    target_raw = raw.get("target")
    if not isinstance(target_raw, dict):
        raise TrainingConfigError("Config must contain a 'target' mapping")
    target = TargetSpec(
        entrypoint=target_raw.get("entrypoint"),
        runner=target_raw.get("runner"),
        components=dict(target_raw.get("components") or {}),
        name=target_raw.get("name", "agent"),
    )
    if not target.entrypoint and not target.runner and not target.components:
        raise TrainingConfigError(
            "target needs at least an 'entrypoint', a 'runner', or 'components'"
        )

    ds_raw = raw.get("dataset")
    if not isinstance(ds_raw, dict) or not ds_raw.get("path"):
        raise TrainingConfigError("Config must contain a 'dataset' mapping with a 'path'")
    dataset = DatasetSpec(
        path=ds_raw["path"],
        format=ds_raw.get("format"),
        input_key=ds_raw.get("input_key"),
        expected_key=ds_raw.get("expected_key"),
        val_path=ds_raw.get("val_path"),
    )

    judge = None
    judge_raw = raw.get("judge")
    if isinstance(judge_raw, dict):
        if not judge_raw.get("provider"):
            raise TrainingConfigError("judge requires a 'provider'")
        known = {
            "provider",
            "model",
            "adversarial",
            "scale",
            "pass_threshold",
            "score_is_normalized",
            "criteria",
            "metric_name",
        }
        judge = JudgeSpec(
            provider=judge_raw["provider"],
            model=judge_raw.get("model"),
            adversarial=bool(judge_raw.get("adversarial", False)),
            scale=int(judge_raw.get("scale", 10)),
            pass_threshold=float(judge_raw.get("pass_threshold", 0.6)),
            score_is_normalized=bool(judge_raw.get("score_is_normalized", False)),
            criteria=judge_raw.get("criteria"),
            metric_name=judge_raw.get("metric_name", "quality"),
            extra={k: v for k, v in judge_raw.items() if k not in known},
        )

    metrics = raw.get("metrics") or ["exact_match"]
    if isinstance(metrics, str):
        metrics = [metrics]
    if not isinstance(metrics, list):
        raise TrainingConfigError("'metrics' must be a list of metric names")
    for m in metrics:
        if m not in BUILTIN_METRICS:
            raise TrainingConfigError(f"Unknown metric {m!r}. Available: {sorted(BUILTIN_METRICS)}")

    opt_raw = raw.get("optimizer") or {}
    if not isinstance(opt_raw, dict):
        raise TrainingConfigError("'optimizer' must be a mapping")
    opt_type = str(opt_raw.get("type", "default")).lower()
    if opt_type not in _OPTIMIZER_TYPES:
        raise TrainingConfigError(
            f"Unknown optimizer type {opt_type!r}. Available: {sorted(_OPTIMIZER_TYPES)}"
        )
    optimizer = OptimizerSpec(
        type=opt_type,
        max_evals=int(opt_raw.get("max_evals", 60)),
        min_improvement=(
            float(opt_raw["min_improvement"]) if "min_improvement" in opt_raw else None
        ),
        seed=opt_raw.get("seed", 0),
        kinds=list(opt_raw["kinds"]) if opt_raw.get("kinds") else None,
        suggest_tools=bool(opt_raw.get("suggest_tools", True)),
        verbose=bool(opt_raw.get("verbose", False)),
    )
    if optimizer.kinds:
        for k in optimizer.kinds:
            _coerce_kind(k)  # validates

    parameters = [_parse_parameter_spec(p) for p in (raw.get("parameters") or [])]

    return TrainingConfig(
        target=target,
        dataset=dataset,
        judge=judge,
        metrics=metrics,
        primary_metric=raw.get("primary_metric"),
        optimizer=optimizer,
        parameters=parameters,
    )


def _parse_parameter_spec(raw: Any) -> ParameterSpec:
    if not isinstance(raw, dict) or "name" not in raw or "kind" not in raw:
        raise TrainingConfigError("each parameter needs at least 'name' and 'kind'")
    _coerce_kind(raw["kind"])  # validates
    bounds = raw.get("bounds")
    if bounds is not None:
        if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
            raise TrainingConfigError(f"parameter {raw['name']!r} bounds must be [low, high]")
        bounds = (bounds[0], bounds[1])
    return ParameterSpec(
        name=raw["name"],
        kind=raw["kind"],
        component=raw.get("component"),
        attr=raw.get("attr"),
        attr_path=raw.get("attr_path"),
        bounds=bounds,
        candidates=raw.get("candidates"),
        step=raw.get("step"),
        max_temperature=raw.get("max_temperature"),
    )


def _coerce_kind(kind: str) -> ParameterKind:
    try:
        return ParameterKind(kind)
    except ValueError as exc:
        valid = [k.value for k in ParameterKind]
        raise TrainingConfigError(f"Unknown parameter kind {kind!r}. Valid: {valid}") from exc


# --------------------------------------------------------------------------- #
# Object resolution                                                           #
# --------------------------------------------------------------------------- #


def _resolve_object(spec: str) -> Any:
    """Resolve a ``"module:attribute"`` spec, optionally calling a ``...()`` factory."""
    call = spec.endswith("()")
    ref = spec[:-2] if call else spec
    if ":" not in ref:
        raise TrainingConfigError(f"Object spec must be 'module:attribute', got {spec!r}")
    module_name, _, attr_path = ref.partition(":")
    try:
        obj: Any = importlib.import_module(module_name)
    except ImportError as exc:
        raise TrainingConfigError(f"Could not import module {module_name!r}: {exc}") from exc
    for part in attr_path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise TrainingConfigError(
                f"{module_name!r} has no attribute path {attr_path!r}"
            ) from exc
    if call:
        obj = obj()
    return obj


def _attr_getter_setter(
    obj: Any, attr_path: str
) -> tuple[Callable[[], Any], Callable[[Any], None]]:
    """Build live getter/setter for ``obj.<dotted.attr.path>``."""
    parts = attr_path.split(".")
    parent_path, leaf = parts[:-1], parts[-1]

    def _parent() -> Any:
        cur = obj
        for p in parent_path:
            cur = getattr(cur, p)
        return cur

    def _getter() -> Any:
        return getattr(_parent(), leaf, None)

    def _setter(value: Any) -> None:
        setattr(_parent(), leaf, value)

    return _getter, _setter


# --------------------------------------------------------------------------- #
# Building live objects                                                       #
# --------------------------------------------------------------------------- #


def build_dataset(spec: DatasetSpec) -> GoldenDataset:
    return _load_dataset(spec.path, spec.format, spec.input_key, spec.expected_key)


def _load_dataset(
    path: str, fmt: str | None, input_key: str | None, expected_key: str | None
) -> GoldenDataset:
    p = Path(path)
    if not p.exists():
        raise TrainingConfigError(f"Dataset file not found: {p}")
    fmt = (fmt or p.suffix.lstrip(".")).lower()
    kw = {"input_key": input_key, "expected_key": expected_key}
    try:
        if fmt in ("jsonl", "ndjson"):
            return GoldenDataset.from_jsonl(str(p), **kw)
        if fmt == "json":
            return GoldenDataset.from_json(str(p), **kw)
        if fmt == "csv":
            return GoldenDataset.from_csv(str(p), **kw)
    except Exception as exc:  # noqa: BLE001
        raise TrainingConfigError(f"Failed to load dataset {p} as {fmt!r}: {exc}") from exc
    raise TrainingConfigError(f"Unknown dataset format {fmt!r} (use jsonl/json/csv)")


def build_judge(spec: JudgeSpec | None) -> Any:
    """Build an LLM judge from a :class:`JudgeSpec` (or ``None``)."""
    if spec is None:
        return None
    from adapt_agent.optimization.judges import JUDGE_REGISTRY, get_judge

    kw: dict[str, Any] = {
        "adversarial": spec.adversarial,
        "scale": spec.scale,
        "pass_threshold": spec.pass_threshold,
        "score_is_normalized": spec.score_is_normalized,
        **spec.extra,
    }
    provider = spec.provider.lower()
    try:
        if provider in JUDGE_REGISTRY:
            return get_judge(provider, model=spec.model, **kw)
        # Fall back to a raw provider (e.g. "echo"/"callable") wrapped in LLMJudge.
        from adapt_agent.optimization.judge import LLMJudge
        from adapt_agent.optimization.providers import get_provider

        prov_kw = {"model": spec.model} if spec.model is not None else {}
        return LLMJudge(get_provider(provider, **prov_kw), **kw)
    except KeyError as exc:
        raise TrainingConfigError(f"Unknown judge provider {spec.provider!r}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise TrainingConfigError(f"Failed to build judge: {exc}") from exc


def build_harness(config: TrainingConfig, judge: Any) -> EvaluationHarness:
    metrics: list[Metric] = [get_metric(name) for name in config.metrics]
    judge_metric_name: str | None = None
    if judge is not None and config.judge is not None:
        judge_metric_name = config.judge.metric_name
        metrics.append(judge.as_metric(judge_metric_name, criteria=config.judge.criteria))
    primary = config.primary_metric
    if primary is None:
        primary = judge_metric_name if judge_metric_name is not None else config.metrics[0]
    names = [m.name for m in metrics]
    if primary not in names:
        raise TrainingConfigError(
            f"primary_metric {primary!r} is not among configured metrics {names}"
        )
    return EvaluationHarness(metrics, primary_metric=primary)


def build_target(config: TrainingConfig) -> OptimizableAgent:
    target_spec = config.target
    components = {name: _resolve_object(ref) for name, ref in target_spec.components.items()}

    runner = None
    if target_spec.runner:
        runner = _resolve_object(target_spec.runner)
    elif target_spec.entrypoint:
        runner = _resolve_object(target_spec.entrypoint)

    if components:
        agent = OptimizableAgent.from_components(
            components=components, runner=runner, name=target_spec.name
        )
    elif runner is not None:
        agent = OptimizableAgent.from_callable(runner, name=target_spec.name)
    else:  # pragma: no cover - guarded in parse
        raise TrainingConfigError("target needs an entrypoint/runner or components")

    for pspec in config.parameters:
        agent.add_parameter(_build_parameter(pspec, components))
    return agent


def _build_parameter(spec: ParameterSpec, components: dict[str, Any]) -> Parameter:
    kind = _coerce_kind(spec.kind)
    bounds = _validate_bounds(spec)

    if spec.candidates is not None and not (spec.attr or spec.attr_path):
        raise TrainingConfigError(
            f"parameter {spec.name!r} with candidates still needs an attr/attr_path to bind to"
        )

    attr_path = spec.attr_path or spec.attr
    if not attr_path:
        raise TrainingConfigError(f"parameter {spec.name!r} needs an 'attr' or 'attr_path'")

    if spec.component is None:
        raise TrainingConfigError(f"parameter {spec.name!r} needs a 'component'")
    obj = components.get(spec.component)
    if obj is None:
        raise TrainingConfigError(
            f"parameter {spec.name!r} references unknown component {spec.component!r}; "
            f"declare it under target.components"
        )

    getter, setter = _attr_getter_setter(obj, attr_path)
    return Parameter(
        name=spec.name,
        kind=kind,
        value=getter(),
        candidates=spec.candidates,
        bounds=bounds,
        step=spec.step,
        getter=getter,
        setter=setter,
        component=spec.component,
    )


def _validate_bounds(spec: ParameterSpec) -> tuple[float, float] | None:
    """Validate/clamp numeric bounds; temperature ranges are clamped, not fatal."""
    if spec.bounds is None:
        return None
    low, high = spec.bounds
    if low > high:
        raise TrainingConfigError(f"parameter {spec.name!r} has bounds low > high: {spec.bounds!r}")
    is_temperature = "temperature" in (spec.attr_path or spec.attr or spec.name or "").lower()
    if is_temperature:
        max_temp = (
            spec.max_temperature if spec.max_temperature is not None else _DEFAULT_MAX_TEMPERATURE
        )
        clamped_high = min(high, max_temp)
        clamped_low = max(low, 0.0)
        if (clamped_low, clamped_high) != (low, high):
            logger.warning(
                "parameter %s temperature bounds %s exceed the allowable range "
                "[0, %s]; clamping to (%s, %s).",
                spec.name,
                spec.bounds,
                max_temp,
                clamped_low,
                clamped_high,
            )
        return (clamped_low, clamped_high)
    return (low, high)


# --------------------------------------------------------------------------- #
# Optimizer assembly + run                                                    #
# --------------------------------------------------------------------------- #


def build_optimizer(config: TrainingConfig, harness: EvaluationHarness, judge: Any) -> Optimizer:
    spec = config.optimizer
    common: dict[str, Any] = {
        "max_evals": spec.max_evals,
        "seed": spec.seed,
        "verbose": spec.verbose,
    }
    if spec.min_improvement is not None:
        common["min_improvement"] = spec.min_improvement

    if spec.type == "default":
        return make_default_optimizer(
            harness,
            judge=judge,
            max_evals=spec.max_evals,
            seed=spec.seed,
            verbose=spec.verbose,
            min_improvement=spec.min_improvement,
        )

    cls = _OPTIMIZER_TYPES[spec.type]
    kwargs: dict[str, Any] = dict(common)
    kwargs["judge"] = judge
    # Strategies that support these extras.
    if spec.type in ("coordinate_ascent", "bootstrap_few_shot", "evolutionary"):
        kwargs["suggest_tools"] = spec.suggest_tools and judge is not None
    if spec.type == "coordinate_ascent" and spec.kinds:
        kwargs["kinds"] = tuple(_coerce_kind(k) for k in spec.kinds)
    optimizer: Optimizer = cls(harness, **kwargs)
    return optimizer


def run_training(config: TrainingConfig | str | Path) -> OptimizationResult:
    """Run an end-to-end training/optimization pass from a config (or config path).

    Returns the :class:`OptimizationResult` with the best configuration already
    applied in place to the live agent, plus any advisory ``recommendations`` the
    judge proposed (e.g. new tools/skills to add).
    """
    if not isinstance(config, TrainingConfig):
        config = load_training_config(config)

    judge = build_judge(config.judge)
    harness = build_harness(config, judge)
    target = build_target(config)
    optimizer = build_optimizer(config, harness, judge)

    val_dataset = None
    if config.dataset.val_path:
        val_dataset = _load_dataset(
            config.dataset.val_path,
            config.dataset.format,
            config.dataset.input_key,
            config.dataset.expected_key,
        )
    dataset = build_dataset(config.dataset)
    if not dataset:
        raise TrainingConfigError("dataset is empty")

    return optimizer.optimize(target, dataset, val_dataset=val_dataset)


__all__ = [
    "TrainingConfig",
    "TargetSpec",
    "DatasetSpec",
    "JudgeSpec",
    "OptimizerSpec",
    "ParameterSpec",
    "TrainingConfigError",
    "load_training_config",
    "parse_training_config",
    "build_dataset",
    "build_judge",
    "build_harness",
    "build_target",
    "build_optimizer",
    "run_training",
]
