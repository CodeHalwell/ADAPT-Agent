"""Tests for structured-output scoring and the exportable optimizer config."""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from adapt_agent.optimization.evals import evaluate_agent
from adapt_agent.optimization.extractors import extract_output_payload, extract_output_text
from adapt_agent.optimization.metrics import (
    checks,
    exact_match,
    field_match,
    field_metrics,
    json_subset,
)
from adapt_agent.optimization.optimizers import OptimizationResult, load_tuned_config


class Envelope:
    """A recognised framework wrapper (MAF ``AgentRunResponse``-shaped)."""

    def __init__(self, text):
        self.text = text
        self.messages = []


class Model:
    def model_dump(self):
        return {"lane": "NOS", "matter": "M1"}


# -- extract_output_payload ----------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (Envelope('{"lane": "NOS"}'), {"lane": "NOS"}),
        (Envelope('```json\n{"lane": "NOS"}\n```'), {"lane": "NOS"}),
        (Envelope('```\n{"lane": "NOS"}\n```'), {"lane": "NOS"}),
        (Envelope("not json at all"), "not json at all"),
        (Model(), {"lane": "NOS", "matter": "M1"}),
        ({"lane": "NOS"}, {"lane": "NOS"}),
        ('{"lane": "NOS"}', {"lane": "NOS"}),
        ('[{"lane": "NOS"}]', [{"lane": "NOS"}]),
        (None, ""),
        ("plain text", "plain text"),
    ],
)
def test_extract_output_payload_shapes(value, expected):
    assert extract_output_payload(value) == expected


def test_payload_keeps_structure_that_text_extraction_flattens():
    """The gap this closes: a recognised envelope collapses to text."""
    envelope = Envelope('{"lane": "NOS"}')
    assert extract_output_text(envelope) == '{"lane": "NOS"}'  # a string
    assert extract_output_payload(envelope) == {"lane": "NOS"}  # the payload


def test_payload_handles_a_dataclass_output():
    @dataclasses.dataclass
    class Triage:
        lane: str

    # A dataclass is not a mapping and has no model_dump; it survives unchanged
    # rather than being mangled.
    assert extract_output_payload(Triage(lane="NOS")).lane == "NOS"


# -- field_match ---------------------------------------------------------------


def test_field_match_scores_one_column():
    out = {"lane": "NOS", "matter": "M1"}
    exp = {"lane": "NOS", "matter": "M2"}
    assert field_match("lane")(out, exp) == 1.0
    assert field_match("matter")(out, exp) == 0.0


def test_field_match_reports_under_the_field_name():
    assert field_match("lane").name == "lane"
    assert field_match("lane", name="lane_accuracy").name == "lane_accuracy"


def test_field_match_uses_the_named_inner_check():
    out, exp = {"total": "about 42.01"}, {"total": 42}
    assert field_match("total", check="numeric_close", tolerance=0.05)(out, exp) == 1.0
    assert field_match("total", check="numeric_close")(out, exp) == 0.0


def test_field_match_missing_field_is_a_failure_by_default():
    assert field_match("pack")({"lane": "NOS"}, {"lane": "NOS"}) == 0.0
    assert field_match("pack", missing=0.5)({"lane": "NOS"}, {"lane": "NOS"}) == 0.5


def test_field_match_parses_a_json_string_side():
    assert field_match("lane")('{"lane": "NOS"}', {"lane": "NOS"}) == 1.0


def test_field_match_rejects_bad_configuration():
    with pytest.raises(ValueError, match="Unknown check"):
        field_match("lane", check="nope")
    with pytest.raises(ValueError, match="cannot nest inside itself"):
        field_match("lane", check="field_match")


def test_field_metrics_builds_one_per_field():
    metrics = field_metrics(["lane", "matter", "action"])
    assert [m.name for m in metrics] == ["lane", "matter", "action"]


def test_field_match_is_reachable_as_a_per_row_check():
    """A dataset row may declare ``{"check": {"name": "field_match", ...}}``."""
    metric = checks()
    example = type("Ex", (), {"metadata": {"check": {"name": "field_match", "field": "lane"}}})()
    assert metric({"lane": "NOS"}, {"lane": "NOS"}, example) == 1.0
    assert metric({"lane": "OTHER"}, {"lane": "NOS"}, example) == 0.0


# -- automatic extractor selection --------------------------------------------


class JsonAgent:
    def run(self, _):
        return Envelope('{"lane": "NOS", "matter": "M1", "action": "file", "pack": "none"}')


ROWS = [{"input": "e", "expected": {"lane": "NOS", "matter": "M1", "action": "file", "pack": "P1"}}]


def test_all_structural_metrics_select_payload_extraction():
    report = evaluate_agent(JsonAgent(), ROWS * 4, metrics=field_metrics(["lane", "pack"]))
    assert report.aggregate == {"lane": 1.0, "pack": 0.0}


def test_json_subset_alone_also_selects_payload_extraction():
    report = evaluate_agent(JsonAgent(), ROWS, metrics=json_subset())
    assert report.score == 0.75  # three of four keys match


def test_mixed_metrics_keep_text_extraction():
    """A dict would score 0.0 against exact_match, so text must win."""
    report = evaluate_agent(JsonAgent(), ROWS, metrics=[exact_match(), field_match("lane")])
    assert report.aggregate["exact_match"] == 0.0
    assert report.aggregate["lane"] == 1.0  # field_match parses the JSON string


def test_explicit_extractor_overrides_the_automatic_choice():
    report = evaluate_agent(
        JsonAgent(), ROWS, metrics=json_subset(), output_extractor=extract_output_text
    )
    assert report.score == 0.75  # json_subset parses a JSON string too


def test_renaming_a_structural_metric_keeps_it_structural():
    report = evaluate_agent(JsonAgent(), ROWS, metrics={"lane_acc": field_match("lane")})
    assert report.aggregate["lane_acc"] == 1.0


def test_evaluate_agent_accepts_concurrency():
    report = evaluate_agent(JsonAgent(), ROWS * 6, metrics=field_metrics(["lane"]), concurrency=3)
    assert report.n == 6
    assert report.aggregate["lane"] == 1.0


# -- OptimizationResult.to_config ---------------------------------------------


def _result() -> OptimizationResult:
    return OptimizationResult(
        best_config={
            "researcher.system_prompt": "Be careful.\nBe brief.",
            "researcher.temperature": 0.2,
            "writer.model": "gpt-4o",
        },
        best_score=0.91,
        baseline_score=0.63,
    )


def test_to_config_nests_by_component():
    assert _result().to_config() == {
        "researcher": {"system_prompt": "Be careful.\nBe brief.", "temperature": 0.2},
        "writer": {"model": "gpt-4o"},
    }


def test_to_config_writes_reviewable_yaml(tmp_path):
    path = tmp_path / "nested" / "tuned.yaml"
    _result().to_config(path)
    text = path.read_text(encoding="utf-8")
    assert "baseline=0.6300 best=0.9100 improvement=+0.2800" in text
    assert "Review this diff before committing" in text
    assert yaml.safe_load(text)["writer"]["model"] == "gpt-4o"


def test_to_config_header_can_be_suppressed(tmp_path):
    path = tmp_path / "tuned.yaml"
    _result().to_config(path, header=False)
    assert not path.read_text(encoding="utf-8").startswith("#")


def test_to_config_round_trips_through_load_tuned_config(tmp_path):
    path = tmp_path / "tuned.yaml"
    result = _result()
    result.to_config(path)
    assert load_tuned_config(path) == result.best_config


def test_to_config_places_bare_names_under_a_default_component():
    result = OptimizationResult(
        best_config={"temperature": 0.5}, best_score=1.0, baseline_score=0.0
    )
    assert result.to_config() == {"agent": {"temperature": 0.5}}
    assert result.to_config(default_component="root") == {"root": {"temperature": 0.5}}


def test_load_tuned_config_rejects_a_non_mapping(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain a mapping"):
        load_tuned_config(path)


def test_load_tuned_config_tolerates_a_hand_flattened_entry(tmp_path):
    path = tmp_path / "mixed.yaml"
    path.write_text("researcher:\n  temperature: 0.2\nbare_knob: 7\n", encoding="utf-8")
    assert load_tuned_config(path) == {"researcher.temperature": 0.2, "bare_knob": 7}


def test_evaluate_agent_rejects_an_unknown_extractor_string():
    with pytest.raises(ValueError, match="must be a callable, None, or 'auto'"):
        evaluate_agent(lambda _: "a", ROWS, output_extractor="bogus")


def test_to_config_writes_json_when_the_extension_says_so(tmp_path):
    import json

    path = tmp_path / "tuned.json"
    _result().to_config(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["writer"]["model"] == "gpt-4o"
    assert loaded["_provenance"]["best_score"] == 0.91


def test_provenance_records_what_the_run_bought(tmp_path):
    result = OptimizationResult(
        best_config={"a.b": 1},
        best_score=0.9,
        baseline_score=0.6,
        validation_score=0.85,
        history=[object(), object()],
    )
    yaml_path = tmp_path / "t.yaml"
    result.to_config(yaml_path)
    header = yaml_path.read_text(encoding="utf-8").splitlines()[1]
    assert "baseline=0.6000" in header and "best=0.9000" in header
    assert "validation=0.85" in header and "over 2 evals" in header


def test_json_round_trip_skips_the_provenance_block(tmp_path):
    path = tmp_path / "tuned.json"
    result = _result()
    result.to_config(path)
    assert load_tuned_config(path) == result.best_config


def test_saved_config_round_trips_through_apply_and_reproduces_the_score(tmp_path):
    """The acceptance criterion: reload the file and get the tuned agent back."""
    from adapt_agent.optimization.parameters import Parameter, ParameterKind
    from adapt_agent.optimization.target import OptimizableAgent

    class Component:
        def __init__(self):
            self.system_prompt = "baseline prompt"

    component = Component()
    target = OptimizableAgent(
        lambda x: component.system_prompt,  # runner: irrelevant to apply()
        name="app",
        components={"researcher": component},
        parameters=[
            Parameter(
                name="researcher.system_prompt",
                kind=ParameterKind.PROMPT,
                value="baseline prompt",
                candidates=["baseline prompt", "tuned prompt"],
                getter=lambda c=component: c.system_prompt,
                setter=lambda v, c=component: setattr(c, "system_prompt", v),
                component="researcher",
            )
        ],
    )
    result = OptimizationResult(
        best_config={"researcher.system_prompt": "tuned prompt"},
        best_score=0.91,
        baseline_score=0.63,
    )
    path = tmp_path / "tuned.yaml"
    result.to_config(path)

    component.system_prompt = "baseline prompt"  # simulate a fresh process
    target.apply(load_tuned_config(path))
    assert component.system_prompt == "tuned prompt"
