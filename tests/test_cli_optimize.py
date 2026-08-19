"""Tests for the ``adapt-agent evaluate`` / ``optimize`` CLI commands."""

import json

import pytest

from adapt_agent.cli import main

# A tiny self-contained agent module written to disk and imported by the CLI.
_AGENT_MODULE = """
STATE = {"prompt": "Answer the question."}
_CAP = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome"}


def run(q):
    country = q.replace("What is the capital of", "").strip(" ?")
    if "ONLY" in STATE["prompt"]:
        return _CAP.get(country, "unknown")
    return "some big city"


from adapt_agent.optimization import OptimizableAgent, Parameter, ParameterKind


def build():
    p = Parameter(
        "agent.prompt",
        ParameterKind.PROMPT,
        value=STATE["prompt"],
        candidates=["Answer the question.", "Answer with ONLY the capital city name."],
        getter=lambda: STATE["prompt"],
        setter=lambda v: STATE.__setitem__("prompt", v),
        component="agent",
    )
    return OptimizableAgent.from_callable(run, parameters=[p], name="capital")
"""

_DATA = [
    {"input": "What is the capital of France?", "expected": "Paris"},
    {"input": "What is the capital of Japan?", "expected": "Tokyo"},
    {"input": "What is the capital of Italy?", "expected": "Rome"},
]


@pytest.fixture()
def agent_env(tmp_path, monkeypatch):
    """Write the agent module + dataset and make the module importable."""
    module_name = f"_cli_agent_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module_name}.py").write_text(_AGENT_MODULE, encoding="utf-8")
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("\n".join(json.dumps(r) for r in _DATA), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield module_name, str(data_path), tmp_path
    # Avoid leaking the temp module into other tests.
    import sys

    sys.modules.pop(module_name, None)


def test_evaluate_text_output(agent_env, capsys):
    module_name, data, _ = agent_env
    code = main(["evaluate", f"{module_name}:build()", "--data", data, "--metric", "exact_match"])
    assert code == 0
    out = capsys.readouterr().out
    assert "exact_match" in out
    assert "3 example" in out


def test_evaluate_json_output(agent_env, capsys):
    module_name, data, _ = agent_env
    code = main(
        ["evaluate", f"{module_name}:build()", "--data", data, "--metric", "exact_match", "--json"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["primary_metric"] == "exact_match"
    assert payload["n"] == 3


def test_optimize_grid_improves_and_saves(agent_env, capsys):
    module_name, data, tmp_path = agent_env
    save = tmp_path / "best.json"
    code = main(
        [
            "optimize",
            f"{module_name}:build()",
            "--data",
            data,
            "--metric",
            "exact_match",
            "--optimizer",
            "grid",
            "--save-config",
            str(save),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "improvement +1.0000" in out
    saved = json.loads(save.read_text(encoding="utf-8"))
    assert saved["agent.prompt"] == "Answer with ONLY the capital city name."


def test_optimize_json_output(agent_env, capsys):
    module_name, data, _ = agent_env
    code = main(
        [
            "optimize",
            f"{module_name}:build()",
            "--data",
            data,
            "--metric",
            "exact_match",
            "--optimizer",
            "coordinate_ascent",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["improved"] is True
    assert payload["best_score"] == 1.0


def test_evaluate_unknown_metric_fails(agent_env, capsys):
    module_name, data, _ = agent_env
    code = main(["evaluate", f"{module_name}:build()", "--data", data, "--metric", "nope"])
    assert code == 1
    assert "ERROR" in capsys.readouterr().out


def test_no_metric_and_no_judge_fails(agent_env, capsys):
    module_name, data, _ = agent_env
    code = main(["evaluate", f"{module_name}:build()", "--data", data])
    assert code == 1
    assert "No metrics" in capsys.readouterr().out


def test_bad_target_spec_fails(agent_env, capsys):
    _, data, _ = agent_env
    code = main(["evaluate", "not_a_valid_spec", "--data", data, "--metric", "exact_match"])
    assert code == 1
    out = capsys.readouterr().out
    assert "ERROR" in out


def test_unsupported_dataset_extension_fails(agent_env, capsys, tmp_path):
    module_name, _, _ = agent_env
    bad = tmp_path / "data.txt"
    bad.write_text("nope", encoding="utf-8")
    code = main(
        ["evaluate", f"{module_name}:build()", "--data", str(bad), "--metric", "exact_match"]
    )
    assert code == 1
    assert "Unsupported dataset extension" in capsys.readouterr().out


def test_optimize_json_error_output(agent_env, capsys):
    _, data, _ = agent_env
    code = main(["optimize", "bad:spec:here", "--data", data, "--metric", "exact_match", "--json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"


def test_components_multi_agent(tmp_path, monkeypatch, capsys):
    """The --component / target-as-runner path wires a multi-agent system."""
    module_name = f"_cli_multi_{tmp_path.name.replace('-', '_')}"
    module_src = """
from adapt_agent.optimization import Parameter, ParameterKind

ROUTER = {"prompt": "route"}
_CAP = {"France": "Paris", "Japan": "Tokyo"}


class Specialist:
    def __init__(self):
        self.prompt = "Answer the question."

    def answer(self, country):
        if "ONLY" in self.prompt:
            return _CAP.get(country, "unknown")
        return "a city"


specialist = Specialist()
# Make the specialist introspectable by the generic registry would require a
# framework shape; instead expose a declared parameter via a wrapper component.


def orchestrate(q):
    country = q.replace("What is the capital of", "").strip(" ?")
    return specialist.answer(country)
"""
    (tmp_path / f"{module_name}.py").write_text(module_src, encoding="utf-8")
    data = tmp_path / "d.jsonl"
    data.write_text(
        json.dumps({"input": "What is the capital of France?", "expected": "Paris"}),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    # target is the runner; component supplies an object (no introspectable params
    # here, so optimization is a no-op but the wiring must succeed).
    code = main(
        [
            "evaluate",
            f"{module_name}:orchestrate",
            "--component",
            f"spec={module_name}:specialist",
            "--data",
            str(data),
            "--metric",
            "exact_match",
        ]
    )
    assert code == 0
    import sys

    sys.modules.pop(module_name, None)


# -- --extract-output / --metric checks ----------------------------------------

_WRAPPED_AGENT_MODULE = '''
class _RunResult:
    """Pydantic-AI-shaped wrapper around the answer text."""

    def __init__(self, output):
        self.output = output

    def all_messages(self):
        return []


_CAP = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome"}


def run(q):
    country = q.replace("What is the capital of", "").strip(" ?")
    return _RunResult(_CAP.get(country, "unknown"))
'''


@pytest.fixture()
def wrapped_agent_env(tmp_path, monkeypatch):
    """An agent returning framework-native (wrapped) outputs plus a dataset."""
    module_name = f"_cli_wrapped_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module_name}.py").write_text(_WRAPPED_AGENT_MODULE, encoding="utf-8")
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("\n".join(json.dumps(r) for r in _DATA), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield module_name, str(data_path)
    import sys

    sys.modules.pop(module_name, None)


def test_evaluate_wrapped_output_scores_zero_without_flag(wrapped_agent_env, capsys):
    module_name, data = wrapped_agent_env
    code = main(
        ["evaluate", f"{module_name}:run", "--data", data, "--metric", "exact_match", "--json"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["aggregate"]["exact_match"] == 0.0


def test_evaluate_extract_output_flag_unwraps(wrapped_agent_env, capsys):
    module_name, data = wrapped_agent_env
    code = main(
        [
            "evaluate",
            f"{module_name}:run",
            "--data",
            data,
            "--metric",
            "exact_match",
            "--extract-output",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["aggregate"]["exact_match"] == 1.0


def test_evaluate_checks_metric_with_per_row_declarations(tmp_path, monkeypatch, capsys):
    module_name = f"_cli_checks_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module_name}.py").write_text(
        "def run(q):\n    return 'Paris' if 'France' in q else 'roughly 42'\n",
        encoding="utf-8",
    )
    rows = [
        {"input": "What is the capital of France?", "expected": "Paris"},
        {"input": "What is 6*7?", "expected": "42", "check": "numeric_close"},
    ]
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        code = main(
            [
                "evaluate",
                f"{module_name}:run",
                "--data",
                str(data_path),
                "--metric",
                "checks",
                "--json",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["aggregate"]["checks"] == 1.0
    finally:
        import sys

        sys.modules.pop(module_name, None)


def test_evaluate_checks_with_judge_does_not_grade_every_row(tmp_path, monkeypatch, capsys):
    """--metric checks --judge X must not auto-append a judge over every row."""
    module_name = f"_cli_checksjudge_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module_name}.py").write_text(
        "def run(q):\n    return 'Paris'\n", encoding="utf-8"
    )
    # No row declares a judge check, so no judge call is ever made.
    rows = [{"input": "capital of France?", "expected": "Paris"}]
    data_path = tmp_path / "data.jsonl"
    data_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        code = main(
            [
                "evaluate",
                f"{module_name}:run",
                "--data",
                str(data_path),
                "--metric",
                "checks",
                "--judge",
                "claude",
                "--json",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        # The checks dispatcher routes judging per-row; no extra "judge" metric.
        assert set(payload["aggregate"]) == {"checks"}
        assert payload["aggregate"]["checks"] == 1.0
    finally:
        import sys

        sys.modules.pop(module_name, None)


def test_evaluate_metric_judge_requires_judge_flag(agent_env, capsys):
    module_name, data, _ = agent_env
    code = main(["evaluate", f"{module_name}:build()", "--data", data, "--metric", "judge"])
    assert code == 1
    assert "--judge" in capsys.readouterr().out
