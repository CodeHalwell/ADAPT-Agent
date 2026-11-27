"""Exporting and comparing evaluation runs.

An :class:`~adapt_agent.optimization.evaluation.EvaluationReport` lives in
memory. To gate CI on it, archive it, or ask "did this change make the agent
worse?", it has to become a file and two files have to be comparable. This
module is that layer, and the ``adapt evaluate`` flags ``--output``,
``--junit`` and ``--baseline`` are thin wrappers over it:

* :func:`results_document` / :func:`to_json` -- the full run as JSON: summary,
  completeness, latency percentiles, usage, and one record per row (inputs,
  expected, output, every score, errors, per-turn detail).
* :func:`to_junit` -- JUnit XML, one ``<testcase>`` per row x metric, so any CI
  system renders failing rows natively. Failures carry the expected and actual
  values, truncated, and the whole document is size-capped.
* :func:`compare_runs` -- per-metric deltas between two runs and the rows that
  regressed or improved, matched by row ``id`` when the dataset has one, else
  by position.

Everything here is pure standard library.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapt_agent.optimization.evaluation import EvaluationReport, ExampleResult

#: Version of the results-document layout written by :func:`results_document`.
RESULTS_SCHEMA_VERSION = 1

#: Longest rendering of one value (input / expected / output) in an export.
DEFAULT_MAX_FIELD_CHARS = 4_000
#: Size ceiling for a JUnit document; detail is dropped past it, never rows.
DEFAULT_JUNIT_MAX_BYTES = 5_000_000

# XML 1.0 cannot carry most C0 control characters at all, escaped or not.
_XML_ILLEGAL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _jsonable(value: Any, limit: int) -> Any:
    """A JSON-safe, size-bounded rendering of ``value``.

    Plain JSON values are kept as values so a baseline comparison reads them
    back unchanged; anything else (framework objects) becomes a truncated
    ``repr``-ish string rather than failing the export.
    """
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _truncate(value, limit)
    try:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return _truncate(str(value), limit)
    if len(text) <= limit:
        return json.loads(text)
    return _truncate(text, limit)


def _row_key(row: Mapping[str, Any]) -> str:
    row_id = row.get("id")
    return f"id:{row_id}" if row_id not in (None, "") else f"index:{row.get('index')}"


def _row_record(result: ExampleResult, limit: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "index": result.index,
        "id": result.row_id,
        "inputs": _jsonable(result.inputs, limit),
        "expected": _jsonable(result.expected, limit),
        "output": _jsonable(result.output, limit),
        "scores": dict(result.scores),
        "error": result.error,
        "error_type": result.error_type,
        "transient": result.transient,
        "transient_metrics": list(result.transient_metrics),
        "metric_errors": dict(result.metric_errors),
        "latency": result.latency,
        "attempts": result.attempts,
        "usage": dict(result.usage) if result.usage else None,
    }
    if result.turns is not None:
        record["turns"] = [
            {
                **{k: v for k, v in turn.items() if k not in ("input", "output", "expected")},
                "input": _jsonable(turn.get("input"), limit),
                "expected": _jsonable(turn.get("expected"), limit),
                "output": _jsonable(turn.get("output"), limit),
            }
            for turn in result.turns
        ]
    return record


def results_document(
    report: EvaluationReport,
    *,
    max_field_chars: int = DEFAULT_MAX_FIELD_CHARS,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The full run as a JSON-safe mapping (what :func:`to_json` writes).

    ``summary`` is :meth:`EvaluationReport.to_dict` plus completeness, latency
    percentiles, usage and metric-error counts; ``rows`` holds every *stored*
    per-row record (``max_results`` bounds that; ``summary.n_evaluated`` is
    always the true row count).
    """
    summary = report.to_dict()
    summary.update(
        {
            "completeness": report.completeness(),
            "latency_p50": report.latency_p50,
            "latency_p95": report.latency_p95,
            "usage": dict(report.usage),
            "n_metric_errors": report.n_metric_errors,
            "n_agent_errors": report.n_agent_errors,
            "transient_by_metric": dict(report.transient_by_metric),
        }
    )
    return {
        "schema": "adapt-agent/eval-results",
        "schema_version": RESULTS_SCHEMA_VERSION,
        "metadata": dict(metadata or {}),
        "summary": summary,
        "rows": [_row_record(r, max_field_chars) for r in report.results],
    }


def to_json(
    report: EvaluationReport,
    path: str | Path | None = None,
    *,
    max_field_chars: int = DEFAULT_MAX_FIELD_CHARS,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Serialise :func:`results_document` to JSON, writing ``path`` when given."""
    text = json.dumps(
        results_document(report, max_field_chars=max_field_chars, metadata=metadata),
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    return text


def load_results(path: str | Path) -> dict[str, Any]:
    """Read a results document written by :func:`to_json`.

    Raises:
        ValueError: If the file is not a results document.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "summary" not in data or "rows" not in data:
        raise ValueError(
            f"{path} is not an evaluation results file (write one with "
            "`adapt evaluate --output` or reporting.to_json)"
        )
    return data


# -- JUnit ---------------------------------------------------------------------


def _xml_text(value: Any, limit: int) -> str:
    rendered = value if isinstance(value, str) else json.dumps(_jsonable(value, limit), default=str)
    return _XML_ILLEGAL.sub("�", _truncate(str(rendered), limit))


def to_junit(
    report: EvaluationReport,
    path: str | Path | None = None,
    *,
    thresholds: float | Mapping[str, float] | None = None,
    suite_name: str = "adapt-agent-eval",
    max_field_chars: int = 1_000,
    max_bytes: int = DEFAULT_JUNIT_MAX_BYTES,
) -> str:
    """Render the run as JUnit XML: one ``<testcase>`` per row x metric.

    A case *fails* when its score is below its threshold -- ``thresholds`` as
    one number for every metric or a per-metric mapping, defaulting to the
    report's ``failure_threshold`` -- and carries the expected and actual
    values in the failure body. A row the agent raised on is an ``<error>``;
    a row dropped as a transient infrastructure fault is ``<skipped>`` (it
    was not measured). Stored rows only (``max_results`` bounds them).

    Once the document approaches ``max_bytes``, failure bodies are reduced to
    their one-line message; cases are never dropped, so the pass/fail counts
    stay exact.
    """

    def threshold_for(metric: str) -> float:
        if isinstance(thresholds, Mapping):
            return float(thresholds.get(metric, report.failure_threshold))
        if thresholds is not None:
            return float(thresholds)
        return report.failure_threshold

    suite = ET.Element("testsuite", name=suite_name)
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    budget = max_bytes
    for result in report.results:
        label = f"row {result.index}" + (f" [{result.row_id}]" if result.row_id else "")
        for metric, score in sorted(result.scores.items()):
            counts["tests"] += 1
            case = ET.SubElement(
                suite,
                "testcase",
                classname=f"{suite_name}.{metric}",
                name=label,
                time=f"{result.latency:.6f}",
            )
            detail_allowed = budget > 0
            if metric in result.transient_metrics:
                counts["skipped"] += 1
                ET.SubElement(
                    case, "skipped", message="dropped as a transient infrastructure fault"
                )
                continue
            if result.error is not None and not result.transient:
                counts["errors"] += 1
                node = ET.SubElement(
                    case,
                    "error",
                    message=_xml_text(result.error, 300),
                    type=result.error_type or "AgentError",
                )
                if detail_allowed:
                    node.text = _xml_text(result.inputs, max_field_chars)
                    budget -= len(node.text or "") + 200
                continue
            cutoff = threshold_for(metric)
            metric_error = result.metric_errors.get(metric)
            if metric_error is not None or score < cutoff:
                counts["failures"] += 1
                message = (
                    f"{metric} errored: {metric_error}"
                    if metric_error is not None
                    else f"{metric} = {score:.4f} < {cutoff:.4f}"
                )
                node = ET.SubElement(case, "failure", message=_xml_text(message, 300))
                if detail_allowed:
                    node.text = (
                        f"expected: {_xml_text(result.expected, max_field_chars)}\n"
                        f"actual:   {_xml_text(result.output, max_field_chars)}\n"
                        f"input:    {_xml_text(result.inputs, max_field_chars)}"
                    )
                    budget -= len(node.text) + 200
    for key, value in counts.items():
        suite.set(key, str(value))
    if budget <= 0:
        suite.set("truncated", "true")
    root = ET.Element("testsuites")
    root.append(suite)
    text = ET.tostring(root, encoding="unicode")
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + text
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(xml + "\n", encoding="utf-8")
    return xml


# -- comparison ----------------------------------------------------------------


@dataclass
class RunComparison:
    """The difference between a run and a baseline run.

    ``metrics`` maps each metric present in both runs to ``{"baseline",
    "current", "delta"}``. ``regressed_rows`` / ``improved_rows`` list
    ``{"key", "metric", "baseline", "current"}`` for rows whose score moved by
    more than ``tolerance``; ``new_rows`` / ``missing_rows`` are row keys
    present in only one run. ``warnings`` flags anything that makes the
    comparison less than like-for-like (a partial run on either side, a
    different primary metric, rows matched by position).
    """

    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    regressed_rows: list[dict[str, Any]] = field(default_factory=list)
    improved_rows: list[dict[str, Any]] = field(default_factory=list)
    new_rows: list[str] = field(default_factory=list)
    missing_rows: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tolerance: float = 0.0

    @property
    def regressed_metrics(self) -> list[str]:
        """Metrics whose aggregate dropped by more than ``tolerance``."""
        return sorted(m for m, d in self.metrics.items() if d["delta"] < -self.tolerance)

    @property
    def has_regression(self) -> bool:
        """A metric's aggregate dropped, or a row's score did."""
        return bool(self.regressed_metrics or self.regressed_rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": {k: dict(v) for k, v in self.metrics.items()},
            "regressed_metrics": self.regressed_metrics,
            "regressed_rows": list(self.regressed_rows),
            "improved_rows": list(self.improved_rows),
            "new_rows": list(self.new_rows),
            "missing_rows": list(self.missing_rows),
            "warnings": list(self.warnings),
            "tolerance": self.tolerance,
            "has_regression": self.has_regression,
        }


def _as_document(run: EvaluationReport | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(run, EvaluationReport):
        return results_document(run)
    if "summary" not in run or "rows" not in run:
        raise ValueError("expected an EvaluationReport or a results document (see to_json)")
    return run


def compare_runs(
    current: EvaluationReport | Mapping[str, Any],
    baseline: EvaluationReport | Mapping[str, Any],
    *,
    tolerance: float = 0.0,
) -> RunComparison:
    """Compare ``current`` with ``baseline``: aggregate deltas and moved rows.

    Either side may be a live report or a results document read back with
    :func:`load_results`. Rows are matched by their dataset ``id`` when both
    runs carry one (stable across reordering), otherwise by position -- which
    is flagged in ``warnings``, because inserting a row then shifts every match.
    A row excluded as transient on either side is not compared for the metric
    it lost.
    """
    cur, base = _as_document(current), _as_document(baseline)
    out = RunComparison(tolerance=tolerance)
    cur_agg = cur["summary"].get("aggregate", {})
    base_agg = base["summary"].get("aggregate", {})
    for metric in sorted(set(cur_agg) & set(base_agg)):
        b, c = float(base_agg[metric]), float(cur_agg[metric])
        out.metrics[metric] = {"baseline": b, "current": c, "delta": c - b}
    only = sorted(set(cur_agg) ^ set(base_agg))
    if only:
        out.warnings.append(f"metrics present in only one run were not compared: {only}")
    if cur["summary"].get("primary_metric") != base["summary"].get("primary_metric"):
        out.warnings.append("the two runs have different primary metrics")
    for label, doc in (("current", cur), ("baseline", base)):
        if not doc["summary"].get("is_complete", True):
            out.warnings.append(
                f"the {label} run is partial (rows dropped as transient); its aggregates "
                "cover a subset of the dataset"
            )

    cur_rows = {_row_key(r): r for r in cur["rows"]}
    base_rows = {_row_key(r): r for r in base["rows"]}
    if any(k.startswith("index:") for k in (*cur_rows, *base_rows)):
        out.warnings.append(
            "rows without an 'id' were matched by position; give dataset rows an 'id' "
            "so a reordered or extended dataset still compares like-for-like"
        )
    out.new_rows = sorted(set(cur_rows) - set(base_rows))
    out.missing_rows = sorted(set(base_rows) - set(cur_rows))
    for key in sorted(set(cur_rows) & set(base_rows)):
        c_row, b_row = cur_rows[key], base_rows[key]
        skip = set(c_row.get("transient_metrics") or ()) | set(b_row.get("transient_metrics") or ())
        for metric in sorted(set(c_row.get("scores", {})) & set(b_row.get("scores", {}))):
            if metric in skip:
                continue
            b, c = float(b_row["scores"][metric]), float(c_row["scores"][metric])
            entry = {"key": key, "metric": metric, "baseline": b, "current": c}
            if c < b - tolerance:
                out.regressed_rows.append(entry)
            elif c > b + tolerance:
                out.improved_rows.append(entry)
    return out


__all__ = [
    "RESULTS_SCHEMA_VERSION",
    "RunComparison",
    "compare_runs",
    "load_results",
    "results_document",
    "to_json",
    "to_junit",
]
