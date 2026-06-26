"""Tests for the ADAPT-Agent command-line interface (``adapt_agent.cli``)."""

import json

import pytest

from adapt_agent import __version__
from adapt_agent.cli import load_config, main, validate_config


def _write_json(tmp_path, name, data):
    """Helper: write ``data`` as JSON to ``tmp_path/name`` and return the path string."""
    path = tmp_path / name
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def _valid_config():
    return {
        "policy_rules": [
            {
                "name": "no_secrets",
                "description": "block secrets",
                "condition": "'password' in message['content']",
                "action": "block",
                "severity": "high",
            }
        ],
        "firewall": {
            "blocked_patterns": ["(?i)ignore previous instructions"],
            "allowed_patterns": ["[a-zA-Z0-9 ]+"],
            "max_content_length": 10000,
        },
        "adversarial": {"attack_patterns": ["leak the system prompt"]},
    }


# ---------------------------------------------------------------------------
# main() top-level behaviour
# ---------------------------------------------------------------------------
def test_main_no_args_prints_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "usage" in out.lower()
    assert "adapt-agent" in out


def test_main_info_returns_zero_and_prints_version(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert __version__ in out
    assert f"ADAPT-Agent v{__version__}" in out


def test_main_version_action_exits():
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0


def test_version_output(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    out = capsys.readouterr().out
    assert __version__ in out


# ---------------------------------------------------------------------------
# validate_config: valid configuration
# ---------------------------------------------------------------------------
def test_validate_config_valid_returns_empty():
    assert validate_config(_valid_config()) == []


def test_validate_config_empty_dict_is_valid():
    # All sections are optional; an empty config should produce no errors.
    assert validate_config({}) == []


# ---------------------------------------------------------------------------
# validate_config: error cases
# ---------------------------------------------------------------------------
def test_validate_config_missing_rule_name():
    config = _valid_config()
    del config["policy_rules"][0]["name"]
    errors = validate_config(config)
    assert any("name" in e for e in errors)


def test_validate_config_missing_condition():
    config = _valid_config()
    del config["policy_rules"][0]["condition"]
    errors = validate_config(config)
    assert any("condition" in e and "missing" in e for e in errors)


def test_validate_config_blank_condition():
    config = _valid_config()
    config["policy_rules"][0]["condition"] = ""
    errors = validate_config(config)
    assert any("condition" in e and "missing" in e for e in errors)


def test_validate_config_non_string_condition():
    config = _valid_config()
    config["policy_rules"][0]["condition"] = 123
    errors = validate_config(config)
    assert any("condition must be a string" in e for e in errors)


def test_validate_config_condition_too_long():
    config = _valid_config()
    # A long but syntactically valid string literal expression.
    config["policy_rules"][0]["condition"] = "'" + ("a" * 1100) + "'"
    errors = validate_config(config)
    assert any("exceeds maximum length" in e for e in errors)


def test_validate_config_condition_syntax_error():
    config = _valid_config()
    config["policy_rules"][0]["condition"] = "this is not valid python !!!"
    errors = validate_config(config)
    assert any("not a valid expression" in e for e in errors)


def test_validate_config_invalid_action():
    config = _valid_config()
    config["policy_rules"][0]["action"] = "explode"
    errors = validate_config(config)
    assert any("action" in e and "invalid" in e for e in errors)


def test_validate_config_invalid_severity():
    config = _valid_config()
    config["policy_rules"][0]["severity"] = "apocalyptic"
    errors = validate_config(config)
    assert any("severity" in e and "invalid" in e for e in errors)


def test_validate_config_invalid_regex_blocked_patterns():
    config = _valid_config()
    config["firewall"]["blocked_patterns"] = ["[unclosed"]
    errors = validate_config(config)
    assert any("blocked_patterns" in e and "regex" in e for e in errors)


def test_validate_config_non_list_policy_rules():
    config = _valid_config()
    config["policy_rules"] = {"not": "a list"}
    errors = validate_config(config)
    assert any("'policy_rules' must be a list" in e for e in errors)


def test_validate_config_non_int_max_content_length():
    config = _valid_config()
    config["firewall"]["max_content_length"] = "lots"
    errors = validate_config(config)
    assert any("max_content_length must be a positive integer" in e for e in errors)


def test_validate_config_negative_max_content_length():
    config = _valid_config()
    config["firewall"]["max_content_length"] = -5
    errors = validate_config(config)
    assert any("max_content_length must be a positive integer" in e for e in errors)


def test_validate_config_non_string_attack_patterns():
    config = _valid_config()
    config["adversarial"]["attack_patterns"] = ["ok", 42]
    errors = validate_config(config)
    assert any("attack_patterns must contain only strings" in e for e in errors)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------
def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "does_not_exist.json"))


def test_load_config_invalid_json_raises(tmp_path):
    path = _write_json(tmp_path, "bad.json", "{not valid json")
    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_array_root_raises(tmp_path):
    path = _write_json(tmp_path, "array.json", [1, 2, 3])
    with pytest.raises(ValueError):
        load_config(path)


def test_load_config_valid_returns_dict(tmp_path):
    path = _write_json(tmp_path, "good.json", _valid_config())
    config = load_config(path)
    assert isinstance(config, dict)
    assert "policy_rules" in config


# ---------------------------------------------------------------------------
# main validate command
# ---------------------------------------------------------------------------
def test_main_validate_valid_file_returns_zero(tmp_path, capsys):
    path = _write_json(tmp_path, "valid.json", _valid_config())
    assert main(["validate", path]) == 0
    out = capsys.readouterr().out
    assert "valid" in out.lower()


def test_main_validate_invalid_file_returns_one(tmp_path, capsys):
    config = _valid_config()
    config["policy_rules"][0]["action"] = "explode"
    path = _write_json(tmp_path, "invalid.json", config)
    assert main(["validate", path]) == 1
    out = capsys.readouterr().out
    assert "INVALID" in out


def test_main_validate_json_valid(tmp_path, capsys):
    path = _write_json(tmp_path, "valid.json", _valid_config())
    assert main(["validate", path, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["errors"] == []


def test_main_validate_json_invalid(tmp_path, capsys):
    config = _valid_config()
    config["policy_rules"][0]["severity"] = "nope"
    path = _write_json(tmp_path, "invalid.json", config)
    assert main(["validate", path, "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert isinstance(payload["errors"], list)
    assert payload["errors"]


def test_main_validate_missing_file_returns_one(tmp_path, capsys):
    missing = str(tmp_path / "nope.json")
    assert main(["validate", missing]) == 1
    out = capsys.readouterr().out
    assert "ERROR" in out


def test_main_validate_missing_file_json(tmp_path, capsys):
    missing = str(tmp_path / "nope.json")
    assert main(["validate", missing, "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert payload["errors"]


# ---------------------------------------------------------------------------
# main monitor command
# ---------------------------------------------------------------------------
def test_main_monitor_returns_zero_and_prints_status(capsys):
    assert main(["monitor", "--agent-id", "a1"]) == 0
    out = capsys.readouterr().out
    assert "a1" in out
    assert "ready" in out.lower()


def test_main_monitor_json(capsys):
    assert main(["monitor", "--agent-id", "a1", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ready"
    assert payload["agent_id"] == "a1"
    assert payload["version"] == __version__
    assert "controls" in payload
    assert "timestamp" in payload


def test_main_monitor_with_valid_config_reflects_counts(tmp_path, capsys):
    path = _write_json(tmp_path, "valid.json", _valid_config())
    assert main(["monitor", "--agent-id", "a1", "--config", path, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    controls = payload["controls"]
    assert controls["policy_rules"] == 1
    assert controls["firewall_blocked_patterns"] == 1
    assert controls["firewall_allowed_patterns"] == 1
    assert controls["adversarial_patterns"] == 1


def test_main_monitor_with_invalid_config_returns_one(tmp_path, capsys):
    config = _valid_config()
    config["policy_rules"][0]["action"] = "explode"
    path = _write_json(tmp_path, "invalid.json", config)
    assert main(["monitor", "--agent-id", "a1", "--config", path]) == 1
    out = capsys.readouterr().out
    assert "ERROR" in out


def test_main_monitor_with_missing_config_returns_one(tmp_path):
    missing = str(tmp_path / "nope.json")
    assert main(["monitor", "--agent-id", "a1", "--config", missing]) == 1


def test_main_monitor_requires_agent_id():
    with pytest.raises(SystemExit):
        main(["monitor"])
