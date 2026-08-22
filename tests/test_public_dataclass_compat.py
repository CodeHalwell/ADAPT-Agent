"""The public dataclasses keep their established positional signatures.

Inserting a field into the middle of a public dataclass does not raise -- it
silently rebinds every positional argument after it. Two of these shipped in
this release before review caught them:

    EvaluationReport(aggregate, metric, results, 0, 4.0, 0.75)
        -> n_transient_errors=4.0, n_evaluated=0.75, total_latency=0.0

    OptimizationResult(..., validation_score, ["tune the prompt"])
        -> validation_complete=["tune the prompt"], recommendations=[]

Both went unnoticed because every call site in the repo uses keywords, so the
suite could not see it. These assertions pin the *shape* callers depend on,
independently of how the library happens to construct them.
"""

from __future__ import annotations

import dataclasses

import pytest

from adapt_agent.optimization.evaluation import EvaluationReport, ExampleResult
from adapt_agent.optimization.optimizers import OptimizationResult, Trial

#: The field order each public dataclass had at the 0.3.0 release. A new field
#: may be *appended* freely; anything that changes this prefix is a silent
#: breaking change for positional callers.
ESTABLISHED_PREFIXES = {
    EvaluationReport: [
        "aggregate",
        "primary_metric",
        "results",
        "n_errors",
        "total_latency",
        "failure_threshold",
    ],
    ExampleResult: [
        "index",
        "inputs",
        "output",
        "expected",
        "scores",
        "latency",
        "error",
    ],
    OptimizationResult: [
        "best_config",
        "best_score",
        "baseline_score",
        "baseline_config",
        "history",
        "best_report",
        "validation_score",
        "recommendations",
    ],
    Trial: ["config", "score", "strategy", "accepted", "metrics"],
}


@pytest.mark.parametrize(
    ("cls", "prefix"), ESTABLISHED_PREFIXES.items(), ids=lambda v: getattr(v, "__name__", "")
)
def test_new_fields_are_appended_not_inserted(cls: type, prefix: list[str]) -> None:
    current = [f.name for f in dataclasses.fields(cls)]
    assert current[: len(prefix)] == prefix, (
        f"{cls.__name__} changed its established positional signature; append new "
        f"fields instead of inserting them"
    )


def test_the_previously_valid_evaluation_report_call_still_means_what_it_did() -> None:
    report = EvaluationReport({"m": 1.0}, "m", [], 0, 4.0, 0.75)

    assert report.total_latency == 4.0
    assert report.failure_threshold == 0.75
    assert report.n_transient_errors == 0
    assert report.is_complete is True


def test_the_previously_valid_optimization_result_call_still_means_what_it_did() -> None:
    result = OptimizationResult({}, 1.0, 0.0, {}, [], None, 0.5, ["tune the prompt"])

    assert result.recommendations == ["tune the prompt"]
    assert result.validation_complete is True
    assert result.to_dict()["recommendations"] == ["tune the prompt"]
