"""``adapt evaluate`` as a CI gate, tuned-config persistence, and config trust checks."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET

import pytest

from adapt_agent import cli
from adapt_agent.cli import main

_AGENTS = """
class ToolTimeout(Exception):
    pass


def perfect(q):
    return "a" + q[1:]


def bad(q):
    return "wrong"


def crashing(q):
    raise RuntimeError("boom")


def slow(q):
    if q != "q0":
        raise ToolTimeout("agent exceeded its step budget")
    return "a0"


def half(q):
    return "a" + q[1:] if q in ("q0", "q1") else "wrong"


class _Provider(Exception):
    status_code = 503


def throttled(q):
    if q == "q0":
        return "a0"
    raise RuntimeError("agent call failed") from _Provider("busy")


def tooly(q):
    return {"output": "a" + q[1:], "tool_calls": [{"name": "search", "args": {"q": q}}]}


def search(x):
    return x


def noisy(x):
    return x


class Comp:
    tools = [search, noisy]
    prompt = "Answer."


comp = Comp()


def run(q):
    return "a" + q[1:] if noisy not in comp.tools and comp.prompt == "ONLY" else "x"


def run_prompt(q):
    return "a" + q[1:] if comp.prompt == "ONLY" else "x"


def build():
    from adapt_agent.optimization import OptimizableAgent, Parameter, ParameterKind

    agent = OptimizableAgent.from_callable(run, name="t")
    agent.add_parameter(
        Parameter(
            "comp.tools",
            ParameterKind.TOOL,
            candidates=[[search, noisy], [search]],
            getter=lambda: comp.tools,
            setter=lambda v: setattr(comp, "tools", v),
            component="comp",
        )
    )
    agent.add_parameter(
        Parameter(
            "comp.prompt",
            ParameterKind.PROMPT,
            value="Answer.",
            candidates=["Answer.", "ONLY"],
            getter=lambda: comp.prompt,
            setter=lambda v: setattr(comp, "prompt", v),
            component="comp",
        )
    )
    return agent
"""


@pytest.fixture()
def env(tmp_path, monkeypatch):
    module = f"_gate_agents_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module}.py").write_text(_AGENTS, encoding="utf-8")
    rows = [{"input": f"q{i}", "expected": f"a{i}", "id": f"r{i}"} for i in range(3)]
    data = tmp_path / "d.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    yield module, str(data), tmp_path
    sys.modules.pop(module, None)


def _evaluate(module: str, agent: str, data: str, *extra: str) -> int:
    return main(
        ["evaluate", f"{module}:{agent}", "--data", data, "--metric", "exact_match", *extra]
    )


def test_no_gate_flags_keeps_exit_zero_for_a_completed_run(env, capsys):
    module, data, _ = env
    assert _evaluate(module, "bad", data) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "3 total, 3 scored" in out
    assert "(over 3/3 rows)" in out


def test_fail_under_bare_and_per_metric(env, capsys):
    module, data, _ = env
    assert _evaluate(module, "half", data, "--fail-under", "0.9") == cli.EXIT_THRESHOLD
    assert "GATE FAILED (exit 3)" in capsys.readouterr().out
    assert _evaluate(module, "half", data, "--fail-under", "exact_match=0.6") == cli.EXIT_OK
    # A per-metric value overrides a bare one regardless of order.
    assert (
        _evaluate(module, "half", data, "--fail-under", "exact_match=0.5", "--fail-under", "0.9")
        == cli.EXIT_OK
    )


@pytest.mark.parametrize("spec", ["nope=0.5", "1.5", "abc"])
def test_fail_under_rejects_bad_values_before_running(env, capsys, spec):
    module, data, _ = env
    assert _evaluate(module, "perfect", data, "--fail-under", spec) == cli.EXIT_COULD_NOT_RUN
    assert "ERROR" in capsys.readouterr().out


def test_every_row_erroring_fails_even_without_flags(env, capsys):
    module, data, _ = env
    assert _evaluate(module, "crashing", data) == cli.EXIT_ERRORS
    assert "every row errored" in capsys.readouterr().out


def test_an_agent_timeout_is_a_failure_not_a_perfect_score(env, capsys):
    module, data, _ = env
    code = _evaluate(module, "slow", data, "--fail-on-error")
    out = capsys.readouterr().out
    assert code == cli.EXIT_ERRORS
    assert "exact_match = 0.3333" in out
    assert "2 errored" in out
    assert "ToolTimeout" in out


def test_fail_on_incomplete_and_partial_label(env, capsys):
    module, data, _ = env
    assert _evaluate(module, "throttled", data) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "PARTIAL: over 1 of 3 rows; 2 dropped as transient" in out
    assert _evaluate(module, "throttled", data, "--fail-on-incomplete") == cli.EXIT_INCOMPLETE


def test_could_not_run_is_exit_one(env, capsys):
    module, data, tmp_path = env
    assert _evaluate(module, "missing_attr", data) == cli.EXIT_COULD_NOT_RUN
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert _evaluate(module, "perfect", str(empty)) == cli.EXIT_COULD_NOT_RUN


def test_json_output_has_rows_completeness_and_gates(env, capsys):
    module, data, _ = env
    code = _evaluate(module, "slow", data, "--json", "--fail-under", "0.5")
    payload = json.loads(capsys.readouterr().out)
    assert code == cli.EXIT_THRESHOLD
    assert payload["gates"]["exit_code"] == 3 and not payload["gates"]["passed"]
    assert payload["completeness"]["rows_errored"] == 2
    assert [r["id"] for r in payload["rows"]] == ["r0", "r1", "r2"]
    assert payload["primary_metric"] == "exact_match"


def test_output_junit_and_baseline_regression(env, capsys):
    module, data, tmp_path = env
    results = tmp_path / "base.json"
    junit = tmp_path / "junit.xml"
    assert _evaluate(module, "perfect", data, "--output", str(results), "--junit", str(junit)) == 0
    suite = ET.parse(junit).getroot().find("testsuite")
    assert suite is not None and suite.get("tests") == "3" and suite.get("failures") == "0"
    capsys.readouterr()

    code = _evaluate(module, "half", data, "--baseline", str(results), "--fail-on-regression")
    out = capsys.readouterr().out
    assert code == cli.EXIT_REGRESSION
    assert "1.0000 -> 0.6667" in out
    assert "rows regressed: 1" in out
    assert _evaluate(module, "half", data, "--baseline", str(results)) == cli.EXIT_OK
    assert (
        _evaluate(
            module,
            "half",
            data,
            "--baseline",
            str(results),
            "--fail-on-regression",
            "--regression-tolerance",
            "1",
        )
        == cli.EXIT_OK
    )


def test_fail_on_regression_requires_baseline(env, capsys):
    module, data, _ = env
    assert _evaluate(module, "perfect", data, "--fail-on-regression") == cli.EXIT_COULD_NOT_RUN


def test_concurrency_and_trajectory_metrics_from_the_cli(env, tmp_path, capsys):
    module, _, tmp = env
    rows = [
        {"input": f"q{i}", "expected": f"a{i}", "expected_tools": [{"name": "search"}]}
        for i in range(6)
    ]
    data = tmp / "t.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    code = main(
        [
            "evaluate",
            f"{module}:tooly",
            "--data",
            str(data),
            "--metric",
            "exact_match",
            "--metric",
            "tool_called",
            "--extract-output",
            "--concurrency",
            "3",
            "--fail-under",
            "1.0",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0, payload
    assert payload["aggregate"] == {"exact_match": 1.0, "tool_called": 1.0}


# -- tuned-config persistence --------------------------------------------------------


def test_optimize_save_config_never_writes_function_reprs(env, capsys):
    from adapt_agent.optimization import load_tuned_config

    module, data, tmp_path = env
    path = tmp_path / "best.json"
    code = main(
        [
            "optimize",
            f"{module}:build()",
            "--data",
            data,
            "--metric",
            "exact_match",
            "--optimizer",
            "grid",
            "--save-config",
            str(path),
        ]
    )
    assert code == 0
    assert "not saved" in capsys.readouterr().out
    text = path.read_text()
    assert "<function" not in text
    saved = json.loads(text)
    assert saved["comp.prompt"] == "ONLY"
    assert "comp.tools" in saved["_provenance"]["unexportable"]
    assert load_tuned_config(path) == {"comp.prompt": "ONLY"}


def test_optimize_save_config_yaml_uses_to_config(env, capsys):
    from adapt_agent.optimization import load_tuned_config

    module, data, tmp_path = env
    path = tmp_path / "best.yaml"
    args = ["optimize", f"{module}:build()", "--data", data, "--metric", "exact_match"]
    assert main([*args, "--optimizer", "grid", "--save-config", str(path)]) == 0
    assert load_tuned_config(path) == {"comp.prompt": "ONLY"}
    assert "<function" not in path.read_text().split("# Not exported")[0]


def test_train_save_config_round_trips(env, capsys):
    from adapt_agent.optimization import load_tuned_config

    module, data, tmp_path = env
    config = tmp_path / "train.yaml"
    config.write_text(
        f"""
target:
  entrypoint: "{module}:run_prompt"
  components: {{comp: "{module}:comp"}}
dataset: {{path: {data}}}
metrics: [exact_match]
optimizer: {{type: grid, max_evals: 10}}
parameters:
  - {{name: comp.prompt, kind: prompt, component: comp, attr: prompt,
      candidates: ["Answer.", "ONLY"]}}
""",
        encoding="utf-8",
    )
    saved = tmp_path / "tuned.json"
    code = main(["train", str(config), "--save-config", str(saved)])
    assert code == 0, capsys.readouterr().out
    assert load_tuned_config(saved)["comp.prompt"] == "ONLY"


# -- config trust boundary -----------------------------------------------------------


def _training(tmp_path, judge: str = "", metrics: str = "[exact_match]") -> str:
    data = tmp_path / "d.jsonl"
    data.write_text('{"input": "q", "expected": "a"}\n', encoding="utf-8")
    path = tmp_path / "t.yaml"
    path.write_text(
        f'target: {{entrypoint: "json:dumps"}}\ndataset: {{path: {data}}}\n'
        f"metrics: {metrics}\n{judge}",
        encoding="utf-8",
    )
    return str(path)


@pytest.mark.parametrize("key", ["base_url", "api_key", "client", "azure_endpoint"])
def test_judge_section_refuses_endpoint_and_credential_keys(tmp_path, key):
    from adapt_agent.optimization.config import TrainingConfigError, load_training_config

    path = _training(tmp_path, f"judge: {{provider: echo, {key}: 'http://evil'}}\n")
    with pytest.raises(TrainingConfigError, match="unsupported key"):
        load_training_config(path)


def test_judge_section_still_accepts_documented_options(tmp_path):
    from adapt_agent.optimization.config import load_training_config

    path = _training(tmp_path, "judge: {provider: echo, temperature: 0.2, rubric: 'r'}\n")
    assert load_training_config(path).judge is not None


def test_field_match_by_name_gets_a_config_error_not_a_type_error(tmp_path):
    from adapt_agent.optimization.config import TrainingConfigError, load_training_config

    with pytest.raises(TrainingConfigError, match="needs options"):
        load_training_config(_training(tmp_path, metrics="[field_match]"))


def test_field_match_mapping_spec_builds(tmp_path):
    from adapt_agent.optimization.config import build_harness, load_training_config

    config = load_training_config(
        _training(tmp_path, metrics="[{name: field_match, field: lane}, exact_match]")
    )
    harness = build_harness(config, None)
    assert [m.name for m in harness.metrics] == ["lane", "exact_match"]


def test_default_anthropic_model_falls_back_without_the_provider_constant():
    from adapt_agent.optimization import config, providers

    expected = getattr(providers, "DEFAULT_ANTHROPIC_MODEL", "claude-opus-5-5")
    assert config.DEFAULT_ANTHROPIC_MODEL == expected


# -- validate: require_approval -------------------------------------------------------


def _validate(tmp_path, rule: dict) -> list[str]:
    return cli.validate_config({"policy_rules": [rule]})


def test_validate_accepts_require_approval_on_tool_rules(tmp_path):
    rule = {
        "name": "approve_deletes",
        "condition": "tool['name'] == 'delete_db'",
        "action": "require_approval",
    }
    assert _validate(tmp_path, rule) == []


def test_validate_rejects_require_approval_outside_tool_rules(tmp_path):
    rule = {
        "name": "x",
        "condition": "'secret' in message['content']",
        "action": "require_approval",
    }
    errors = _validate(tmp_path, rule)
    assert errors and "tool-scoped" in errors[0]


def test_a_metric_error_shows_in_completeness_and_fails_the_error_gate(env, capsys):
    module, data, _ = env
    args = ["evaluate", f"{module}:perfect", "--data", data, "--metric", "tool_called"]
    assert main(args) == cli.EXIT_OK
    assert "3 with a metric error" in capsys.readouterr().out
    assert main([*args, "--fail-on-error"]) == cli.EXIT_ERRORS
