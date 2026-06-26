"""Tests for the declarative YAML/JSON training configuration."""

from __future__ import annotations

import sys
import textwrap

import pytest

from adapt_agent.optimization.config import (
    DatasetSpec,
    ParameterSpec,
    TrainingConfigError,
    _build_parameter,
    _resolve_object,
    _validate_bounds,
    build_dataset,
    load_training_config,
    parse_training_config,
    run_training,
)


def _write(path, text):
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
# parsing & validation                                                        #
# --------------------------------------------------------------------------- #


def test_parse_minimal_config():
    cfg = parse_training_config(
        {
            "target": {"entrypoint": "builtins:str"},
            "dataset": {"path": "golden.jsonl"},
        }
    )
    assert cfg.target.entrypoint == "builtins:str"
    assert cfg.dataset.path == "golden.jsonl"
    assert cfg.metrics == ["exact_match"]
    assert cfg.optimizer.type == "default"


def test_parse_requires_target():
    with pytest.raises(TrainingConfigError):
        parse_training_config({"dataset": {"path": "x.jsonl"}})


def test_parse_requires_dataset_path():
    with pytest.raises(TrainingConfigError):
        parse_training_config({"target": {"entrypoint": "m:a"}, "dataset": {}})


def test_parse_rejects_unknown_metric():
    with pytest.raises(TrainingConfigError):
        parse_training_config(
            {"target": {"entrypoint": "m:a"}, "dataset": {"path": "x"}, "metrics": ["nope"]}
        )


def test_parse_rejects_unknown_optimizer():
    with pytest.raises(TrainingConfigError):
        parse_training_config(
            {
                "target": {"entrypoint": "m:a"},
                "dataset": {"path": "x"},
                "optimizer": {"type": "bogus"},
            }
        )


def test_parse_rejects_unknown_kind():
    with pytest.raises(TrainingConfigError):
        parse_training_config(
            {
                "target": {"entrypoint": "m:a"},
                "dataset": {"path": "x"},
                "parameters": [{"name": "p", "kind": "not_a_kind", "component": "c", "attr": "x"}],
            }
        )


def test_judge_spec_parsed():
    cfg = parse_training_config(
        {
            "target": {"entrypoint": "m:a"},
            "dataset": {"path": "x"},
            "judge": {"provider": "anthropic", "model": "claude-opus-4-8", "adversarial": True},
        }
    )
    assert cfg.judge is not None
    assert cfg.judge.adversarial is True
    assert cfg.judge.provider == "anthropic"


def test_load_yaml_file(tmp_path):
    path = _write(
        tmp_path / "train.yaml",
        """
        target:
          entrypoint: "builtins:str"
        dataset:
          path: golden.jsonl
        metrics: [exact_match, token_f1]
        optimizer:
          type: coordinate_ascent
          max_evals: 5
        """,
    )
    cfg = load_training_config(path)
    assert cfg.metrics == ["exact_match", "token_f1"]
    assert cfg.optimizer.type == "coordinate_ascent"
    assert cfg.optimizer.max_evals == 5


def test_load_missing_file():
    with pytest.raises(TrainingConfigError):
        load_training_config("/no/such/file.yaml")


# --------------------------------------------------------------------------- #
# temperature clamping (the headline error-handling requirement)              #
# --------------------------------------------------------------------------- #


def test_temperature_bounds_clamped_with_default_max(caplog):
    spec = ParameterSpec(
        name="manager.temperature",
        kind="hyperparam",
        component="manager",
        attr_path="chat_client.temperature",
        bounds=(0.0, 5.0),
    )
    bounds = _validate_bounds(spec)
    assert bounds == (0.0, 2.0)  # clamped to the default allowable range


def test_temperature_bounds_clamped_to_provider_max():
    spec = ParameterSpec(
        name="t",
        kind="hyperparam",
        attr="temperature",
        bounds=(-1.0, 1.8),
        max_temperature=1.0,
    )
    assert _validate_bounds(spec) == (0.0, 1.0)


def test_non_temperature_bounds_not_clamped():
    spec = ParameterSpec(
        name="max_round_count", kind="routing", attr="max_round_count", bounds=(4, 12)
    )
    assert _validate_bounds(spec) == (4, 12)


def test_bounds_low_gt_high_rejected():
    spec = ParameterSpec(name="t", kind="hyperparam", attr="temperature", bounds=(2.0, 1.0))
    with pytest.raises(TrainingConfigError):
        _validate_bounds(spec)


# --------------------------------------------------------------------------- #
# object resolution & parameter binding                                       #
# --------------------------------------------------------------------------- #


def test_resolve_object_and_factory():
    assert _resolve_object("builtins:str") is str
    assert _resolve_object("builtins:list()") == []


def test_resolve_object_bad_spec():
    with pytest.raises(TrainingConfigError):
        _resolve_object("no_colon_here")


def test_build_parameter_binds_live_attr():
    class Comp:
        def __init__(self):
            self.prompt = "old"

    comp = Comp()
    spec = ParameterSpec(
        name="c.prompt",
        kind="prompt",
        component="c",
        attr="prompt",
        candidates=["a", "b"],
    )
    param = _build_parameter(spec, {"c": comp})
    assert param.read() == "old"
    param.write("new")
    assert comp.prompt == "new"
    assert param.candidates == ["a", "b"]


def test_build_parameter_unknown_component():
    spec = ParameterSpec(name="p", kind="prompt", component="missing", attr="x")
    with pytest.raises(TrainingConfigError):
        _build_parameter(spec, {})


# --------------------------------------------------------------------------- #
# end-to-end run (offline, no judge)                                          #
# --------------------------------------------------------------------------- #


def test_run_training_end_to_end(tmp_path, monkeypatch):
    # A tiny project module the config will import.
    module = _write(
        tmp_path / "mini_app.py",
        """
        SHARED = {"prefix": ""}

        class Cfg:
            prefix = ""

        cfg = Cfg()

        def run(q):
            return f"{cfg.prefix}{q}"
        """,
    )
    assert module
    monkeypatch.syspath_prepend(str(tmp_path))

    data = _write(
        tmp_path / "golden.jsonl",
        """
        {"input": "France", "expected": "ANSWER:France"}
        {"input": "Japan", "expected": "ANSWER:Japan"}
        """,
    )

    cfg = parse_training_config(
        {
            "target": {
                "entrypoint": "mini_app:run",
                "components": {"cfg": "mini_app:cfg"},
            },
            "dataset": {"path": data, "format": "jsonl"},
            "metrics": ["exact_match"],
            "optimizer": {"type": "coordinate_ascent", "max_evals": 10, "seed": 0},
            "parameters": [
                {
                    "name": "cfg.prefix",
                    "kind": "prompt",
                    "component": "cfg",
                    "attr": "prefix",
                    "candidates": ["", "ANSWER:"],
                }
            ],
        }
    )
    result = run_training(cfg)
    # The "ANSWER:" prefix candidate makes every output match -> best score 1.0.
    assert result.best_score >= result.baseline_score
    assert result.best_score == pytest.approx(1.0)
    assert result.best_config.get("cfg.prefix") == "ANSWER:"


def test_build_dataset_missing_file():
    with pytest.raises(TrainingConfigError):
        build_dataset(DatasetSpec(path="/no/such/data.jsonl"))


@pytest.fixture(autouse=True)
def _cleanup_mini_app():
    yield
    sys.modules.pop("mini_app", None)
