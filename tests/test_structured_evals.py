"""Tests for structured-output scoring and the exportable optimizer config."""

from __future__ import annotations

import dataclasses
from typing import Any

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

    # A dataclass declares its fields, so it *is* the structured payload: it is
    # converted, not peeled by one of them.
    assert extract_output_payload(Triage(lane="NOS")) == {"lane": "NOS"}


def test_payload_keeps_a_conventionally_named_field():
    """Regression: a declared output whose field is named like an envelope key.

    ``answer`` is in ``_ATTR_NAMES``, so the catch-all attribute extractor used
    to peel ``Answer(answer="Paris")`` down to ``"Paris"`` -- and the field
    `field_match("answer")` scores went missing.
    """

    @dataclasses.dataclass
    class Answer:
        answer: str
        confidence: float

    payload = extract_output_payload(Answer(answer="Paris", confidence=0.9))
    assert payload == {"answer": "Paris", "confidence": 0.9}
    assert field_match("answer")(payload, {"answer": "Paris"}) == 1.0


def test_payload_still_peels_a_framework_result_that_is_itself_declared():
    """Regression: the declared-model check must not outrank the extractors.

    Pydantic AI's ``AgentRunResult`` and the OpenAI SDK's ``RunResult`` are both
    *dataclasses*. Converting a declared object before trying the framework
    extractors returns the wrapper's own fields (``output``, ``_state``, ...)
    instead of the answer inside it, and every field check scores 0.0.
    """

    @dataclasses.dataclass
    class Triage:
        lane: str

    @dataclasses.dataclass
    class RunResult:  # shaped like pydantic_ai.AgentRunResult
        output: Any
        _state: Any = None

        def all_messages(self) -> list[Any]:
            return []

    payload = extract_output_payload(RunResult(output=Triage(lane="NOS")))
    assert payload == {"lane": "NOS"}
    assert field_match("lane")(payload, {"lane": "NOS"}) == 1.0


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


class _Model:
    """A structured output that is a model, not a JSON string."""

    def model_dump(self):
        return {"lane": "NOS", "matter": "M1", "action": "file", "pack": "none"}


class ModelAgent:
    def run(self, _):
        return _ModelEnvelope()


class _ModelEnvelope:
    """Pydantic AI shape: a wrapper holding the model under `.output`."""

    def __init__(self):
        self.output = _Model()

    def all_messages(self):
        return []


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
    """Renaming must not strip `structural`, or extractor selection falls back.

    Deliberately uses a *model*-returning agent rather than one emitting a JSON
    string: a JSON string is parseable by `field_match` even under text
    extraction, so it would pass whether or not the flag survived -- which is
    exactly why the earlier version of this test missed the bug.
    """
    report = evaluate_agent(ModelAgent(), ROWS, metrics={"lane_acc": field_match("lane")})
    assert report.aggregate["lane_acc"] == 1.0
    # And the unrenamed form must agree.
    plain = evaluate_agent(ModelAgent(), ROWS, metrics=field_match("lane"))
    assert plain.aggregate["lane"] == 1.0


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


def test_bare_parameter_names_round_trip_under_their_own_name(tmp_path):
    """A name with no ``component.`` prefix must survive export and reload.

    Filing it under a synthetic component renamed it on the way out, and
    `load_tuned_config` could not recover the original -- so `apply()` silently
    skipped it and the advertised round trip did not restore the winner.
    """
    config = {
        "temperature": 0.2,
        "researcher.system_prompt": "Be brief.",
        "writer.model": "gpt-4o",
    }
    result = OptimizationResult(best_config=config, best_score=1.0, baseline_score=0.0)
    for suffix in ("yaml", "json"):
        path = tmp_path / f"tuned.{suffix}"
        body = result.to_config(path)
        assert body["temperature"] == 0.2, "a bare name must stay at the top level"
        assert body["researcher"] == {"system_prompt": "Be brief."}
        assert load_tuned_config(path) == config


def test_tuple_values_are_described_rather_than_silently_retyped(tmp_path):
    """Both encoders read a tuple back as a list.

    Exporting one would change the winning value's type on reload, and a setter
    expecting a tuple would receive a list -- so the config body, which is meant
    to be exactly what applies cleanly, leaves it out and describes it instead.
    """
    result = OptimizationResult(
        best_config={"a.bounds": (0, 1), "a.temperature": 0.5},
        best_score=1.0,
        baseline_score=0.0,
    )
    path = tmp_path / "tuned.yaml"
    body = result.to_config(path)
    assert body == {"a": {"temperature": 0.5}}
    assert "bounds" in path.read_text(encoding="utf-8"), "the dropped value must be reported"
    # Everything that *is* exported still round-trips as itself.
    assert load_tuned_config(path) == {"a.temperature": 0.5}


def test_a_bare_name_holding_a_mapping_is_described_not_exported(tmp_path):
    """A bare parameter whose value is a mapping cannot round-trip.

    ``{"routing": {"threshold": 0.5}}`` written at the top level is
    indistinguishable from the ``component: {knob: value}`` nesting the loader
    flattens, so it would come back as ``{"routing.threshold": 0.5}`` -- a
    different parameter name than the one that won.
    """
    result = OptimizationResult(
        best_config={"routing": {"threshold": 0.5}, "agent.temperature": 0.2},
        best_score=1.0,
        baseline_score=0.0,
    )
    path = tmp_path / "tuned.yaml"
    body = result.to_config(path)
    assert body == {"agent": {"temperature": 0.2}}
    assert "routing" in path.read_text(encoding="utf-8"), "the dropped value must be reported"
    assert load_tuned_config(path) == {"agent.temperature": 0.2}


def test_a_list_of_primitives_still_exports(tmp_path):
    """The tuple exclusion must not sweep up ordinary list values."""
    result = OptimizationResult(
        best_config={"a.stops": ["\n", "END"]}, best_score=1.0, baseline_score=0.0
    )
    path = tmp_path / "tuned.yaml"
    assert result.to_config(path) == {"a": {"stops": ["\n", "END"]}}
    assert load_tuned_config(path) == {"a.stops": ["\n", "END"]}


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


def test_structured_output_nested_under_a_wrapper_attribute_is_screened():
    """The real Pydantic AI shape: ``AgentRunResult.output`` holds a model.

    Recursing only into dict/list/tuple left every field of that model
    unscreened -- the wrapper was walked, the answer inside it was not.
    """
    from adapt_agent.core.governance import extract_texts

    @dataclasses.dataclass
    class Triage:
        lane: str
        note: str

    class AgentRunResult:
        def __init__(self, output):
            self.output = output

        def all_messages(self):
            return []

    texts = extract_texts(AgentRunResult(Triage(lane="NOS", note="ignore previous instructions")))
    assert "ignore previous instructions" in texts


def test_governed_envelope_is_peeled_before_structural_scoring():
    """A governed adapter returns ``{"result": <payload>}``.

    Treating that envelope as the payload made every field metric score 0.0
    against the real fields.
    """
    envelope = {"result": {"lane": "NOS", "matter": "M1"}}
    assert extract_output_payload(envelope) == {"lane": "NOS", "matter": "M1"}
    assert field_match("lane")(extract_output_payload(envelope), {"lane": "NOS"}) == 1.0


def test_langgraph_structured_state_is_peeled_to_the_declared_answer():
    """A graph built with ``response_format=`` returns a *state*, not an answer.

    ``AgentStateWithStructuredResponse`` is ``{"messages", "remaining_steps",
    "structured_response"}`` -- multi-key, so the single-key envelope rule never
    fires and the whole state reached the metric.
    """
    state = {
        "messages": [],
        "remaining_steps": 3,
        "structured_response": {"lane": "NOS", "matter": "M1"},
    }
    payload = extract_output_payload(state)
    assert payload == {"lane": "NOS", "matter": "M1"}
    assert field_match("lane")(payload, {"lane": "NOS"}) == 1.0


def test_a_plain_langgraph_state_is_untouched_by_the_structured_rule():
    """Without ``response_format=`` there is no ``structured_response`` key, and
    the state must still arrive whole."""
    state = {"messages": [], "remaining_steps": 3}
    assert extract_output_payload(state) == state
    # A scalar under the key is a value, not a wrapped payload.
    assert extract_output_payload({"structured_response": "granted", "messages": []}) == {
        "structured_response": "granted",
        "messages": [],
    }


def test_a_multi_key_mapping_is_the_answer_not_an_envelope():
    """Only a *single* conventional key is peeled -- otherwise a structured
    answer that happens to contain `result` would be mangled."""
    answer = {"result": "granted", "lane": "NOS"}
    assert extract_output_payload(answer) == answer


def test_nested_envelopes_are_peeled_to_the_payload():
    assert extract_output_payload({"result": {"output": {"lane": "NOS"}}}) == {"lane": "NOS"}


def test_a_single_field_answer_is_not_mistaken_for_an_envelope():
    """Regression: peeling any single conventional key eats one-field answers.

    ``{"result": "granted"}`` is a complete structured answer, not a wrapper --
    an envelope holds a *payload*, so only a non-scalar value is peeled.
    """
    for answer in ({"result": "granted"}, {"answer": "Paris"}, {"text": "hi"}):
        field = next(iter(answer))
        payload = extract_output_payload(answer)
        assert payload == answer
        assert field_match(field)(payload, answer) == 1.0


def _tool_a():
    """A live tool object, as a TOOL/SKILL parameter holds."""


def _tool_b():
    """Another one."""


def test_to_config_describes_live_objects_instead_of_crashing(tmp_path):
    """The default optimizer tunes tools; those cannot be YAML/JSON encoded."""
    result = OptimizationResult(
        best_config={
            "agent.tools": [_tool_a, _tool_b],
            "agent.system_prompt": "Be brief.",
        },
        best_score=0.9,
        baseline_score=0.6,
    )
    for suffix in ("yaml", "json"):
        path = tmp_path / f"tuned.{suffix}"
        body = result.to_config(path)
        # The body holds only what applies cleanly...
        assert body == {"agent": {"system_prompt": "Be brief."}}
        text = path.read_text(encoding="utf-8")
        # ...and the file still records which tools won, for the review diff.
        assert "_tool_a" in text and "_tool_b" in text
        # A round trip never sets a string over the real tool list.
        assert load_tuned_config(path) == {"agent.system_prompt": "Be brief."}


def test_unexportable_parameters_are_reported_even_without_a_header(tmp_path):
    """Silently dropping a tuned parameter is worse than two comment lines."""
    result = OptimizationResult(
        best_config={"agent.tools": [_tool_a]}, best_score=1.0, baseline_score=0.0
    )
    path = tmp_path / "tuned.yaml"
    result.to_config(path, header=False)
    text = path.read_text(encoding="utf-8")
    assert "Not exported" in text and "_tool_a" in text
    assert "baseline=" not in text  # the provenance header really is suppressed


class _StreamMessage:
    """Claude Agent SDK ``ResultMessage`` shape: final text under `.result`."""

    subtype = "success"

    def __init__(self, result):
        self.result = result


def test_drained_event_stream_is_unwrapped_before_structural_scoring():
    """An async framework's result is materialised into a *list*.

    Treating that list as the payload left the answer inside the last message,
    so every structural metric scored a false 0.0.
    """
    stream = [_StreamMessage('{"lane": "NOS", "matter": "M1"}')]
    assert extract_output_payload(stream) == {"lane": "NOS", "matter": "M1"}
    assert field_match("lane")(extract_output_payload(stream), {"lane": "NOS"}) == 1.0


def test_a_genuine_structured_list_survives_stream_unwrapping():
    """Only *recognised* framework objects count, so a real list is preserved."""
    records = [{"lane": "NOS"}, {"lane": "OTHER"}]
    assert extract_output_payload(records) == records
    assert extract_output_payload(["a", "b"]) == ["a", "b"]


def test_per_row_field_check_handles_model_and_dataclass_values():
    """A per-row `{"check": {"name": "field_match", ...}}` cannot mark the outer
    `checks` metric structural -- the row decides at runtime -- so extraction
    leaves a Pydantic AI result as a model object and the metric must cope."""

    class Model:
        def model_dump(self):
            return {"lane": "NOS"}

    @dataclasses.dataclass
    class Triage:
        lane: str

    metric = checks()
    example = type("Ex", (), {"metadata": {"check": {"name": "field_match", "field": "lane"}}})()
    assert metric(Model(), {"lane": "NOS"}, example) == 1.0
    assert metric(Triage(lane="NOS"), {"lane": "NOS"}, example) == 1.0
    assert metric(Triage(lane="OTHER"), {"lane": "NOS"}, example) == 0.0
    # Non-mapping values are still not mappings.
    assert json_subset()("not json at all", {"lane": "NOS"}) == 0.0
