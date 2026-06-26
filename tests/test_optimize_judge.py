"""Offline, deterministic tests for adapt_agent.optimization.judge.

No third-party SDK is imported and no network call is made. The judge is
driven entirely by canned completion functions / CallableProvider.
"""

import pytest

from adapt_agent.optimization.judge import (
    _DEFAULT_RUBRIC,
    JudgeVerdict,
    LLMJudge,
    _criteria_block,
    _extract_json,
    _first_number,
    _normalize_score,
    _reference_block,
    _render_failures,
    _stringify,
    _strip_fences,
)
from adapt_agent.optimization.providers import CallableProvider


def const(text):
    """Return a completion fn that always yields ``text`` (ignoring the prompt)."""
    return lambda prompt: text


# -- JudgeVerdict dataclass ----------------------------------------------------


def test_judge_verdict_defaults():
    v = JudgeVerdict(0.5, True)
    assert v.score == 0.5
    assert v.passed is True
    assert v.reasoning == ""
    assert v.raw == ""
    assert v.metadata == {}


# -- score() parsing -----------------------------------------------------------


def test_score_parses_json_object_normalized():
    judge = LLMJudge(const('{"score": 8, "pass": true, "reasoning": "good"}'))
    v = judge.score("in", "out")
    assert v.score == pytest.approx(0.8)
    assert v.passed is True
    assert v.reasoning == "good"
    assert v.raw == '{"score": 8, "pass": true, "reasoning": "good"}'


def test_score_on_zero_to_one_float_taken_as_is():
    judge = LLMJudge(const('{"score": 0.42}'))
    v = judge.score("in", "out")
    assert v.score == pytest.approx(0.42)
    # 0.42 < 0.6 default threshold and no explicit pass -> False
    assert v.passed is False


def test_score_bare_integer_fallback_when_not_json():
    # Unparseable as JSON, but a number is present -> heuristic fallback.
    judge = LLMJudge(const("I would rate this a 7 out of 10."))
    v = judge.score("in", "out")
    assert v.score == pytest.approx(0.7)
    # No "pass" field -> derived from threshold; 0.7 >= 0.6
    assert v.passed is True


def test_score_unparseable_no_number_returns_on_error():
    judge = LLMJudge(const("no numbers here at all"), on_error=0.0)
    v = judge.score("in", "out")
    assert v.score == 0.0
    assert v.passed is False
    assert v.reasoning == "unparseable"
    assert v.raw == "no numbers here at all"


def test_score_pass_threshold_derived_when_no_pass_field():
    judge = LLMJudge(const('{"score": 6}'), pass_threshold=0.6)
    assert judge.score("i", "o").passed is True
    judge2 = LLMJudge(const('{"score": 5}'), pass_threshold=0.6)
    assert judge2.score("i", "o").passed is False


def test_score_explicit_pass_field_overrides_threshold():
    # Score is low but pass=true explicitly -> passed True.
    judge = LLMJudge(const('{"score": 1, "pass": true}'))
    assert judge.score("i", "o").passed is True
    # High score but pass=false explicitly -> passed False.
    judge2 = LLMJudge(const('{"score": 10, "pass": false}'))
    assert judge2.score("i", "o").passed is False


def test_score_metadata_captures_extra_fields():
    judge = LLMJudge(const('{"score": 9, "pass": true, "reasoning": "ok", "subscores": {"a": 1}}'))
    v = judge.score("i", "o")
    assert v.metadata == {"subscores": {"a": 1}}


def test_score_non_bool_pass_field_falls_back_to_threshold():
    # pass is a non-bool truthy value -> ignored, threshold used instead.
    judge = LLMJudge(const('{"score": 9, "pass": "yes"}'), pass_threshold=0.6)
    v = judge.score("i", "o")
    assert v.passed is True  # 0.9 >= 0.6


def test_score_provider_returns_none_raw():
    # A completion fn returning None -> _complete yields None -> on_error path.
    judge = LLMJudge(lambda p: None, on_error=0.0, pass_threshold=0.6)
    v = judge.score("i", "o")
    assert v.score == 0.0
    assert v.passed is False  # on_error (0.0) < threshold
    assert v.raw == ""


def test_score_provider_none_raw_passes_when_on_error_high():
    judge = LLMJudge(lambda p: None, on_error=0.9, pass_threshold=0.6)
    v = judge.score("i", "o")
    assert v.score == 0.9
    assert v.passed is True


def test_score_provider_raises_swallowed_to_on_error():
    def boom(prompt):
        raise RuntimeError("provider exploded")

    judge = LLMJudge(boom, on_error=0.0)
    v = judge.score("i", "o")
    assert v.score == 0.0
    assert v.raw == ""


def test_score_via_callable_provider():
    provider = CallableProvider(const('{"score": 10, "pass": true}'))
    judge = LLMJudge(provider)
    assert judge.score("i", "o").score == pytest.approx(1.0)


def test_score_custom_scale():
    judge = LLMJudge(const('{"score": 50}'), scale=100)
    assert judge.score("i", "o").score == pytest.approx(0.5)


# -- compare() -----------------------------------------------------------------


def test_compare_winner_a():
    judge = LLMJudge(const('{"winner": "A", "reasoning": "x"}'))
    assert judge.compare("i", "a", "b") == "A"


def test_compare_winner_b():
    judge = LLMJudge(const('{"winner": "B"}'))
    assert judge.compare("i", "a", "b") == "B"


def test_compare_tie_explicit():
    judge = LLMJudge(const('{"winner": "tie"}'))
    assert judge.compare("i", "a", "b") == "tie"


def test_compare_fallback_scan_starts_with_a():
    judge = LLMJudge(const("A is clearly better"))
    assert judge.compare("i", "a", "b") == "A"


def test_compare_fallback_scan_starts_with_b():
    judge = LLMJudge(const("B wins this round"))
    assert judge.compare("i", "a", "b") == "B"


def test_compare_fallback_quoted_a():
    # Does not start with A/B, but contains "A".
    judge = LLMJudge(const('the better one is "A" overall'))
    assert judge.compare("i", "a", "b") == "A"


def test_compare_fallback_default_tie():
    judge = LLMJudge(const("cannot decide really"))
    assert judge.compare("i", "a", "b") == "tie"


def test_compare_none_returns_tie():
    judge = LLMJudge(lambda p: None)
    assert judge.compare("i", "a", "b") == "tie"


# -- critique() ----------------------------------------------------------------


def test_critique_returns_stripped_text():
    judge = LLMJudge(const("  needs more detail  \n"))
    assert judge.critique("i", "o") == "needs more detail"


def test_critique_none_returns_empty_string():
    judge = LLMJudge(lambda p: None)
    assert judge.critique("i", "o") == ""


def test_critique_with_expected_and_criteria():
    captured = {}

    def fn(prompt):
        captured["prompt"] = prompt
        return "feedback"

    judge = LLMJudge(fn)
    judge.critique("the input", "the output", "the expected", criteria="be terse")
    assert "REFERENCE ANSWER" in captured["prompt"]
    assert "TASK-SPECIFIC CRITERIA" in captured["prompt"]
    assert "be terse" in captured["prompt"]


# -- improve_prompt() ----------------------------------------------------------


def test_improve_prompt_returns_cleaned_text():
    judge = LLMJudge(const("A better instruction."))
    out = judge.improve_prompt("old instruction", [{"input": "x", "output": "y"}])
    assert out == "A better instruction."


def test_improve_prompt_strips_fences():
    judge = LLMJudge(const("```\nFenced instruction\n```"))
    assert judge.improve_prompt("old", []) == "Fenced instruction"


def test_improve_prompt_strips_language_fences():
    judge = LLMJudge(const("```markdown\nSome text\n```"))
    assert judge.improve_prompt("old", []) == "Some text"


def test_improve_prompt_none_when_empty_current():
    judge = LLMJudge(const("anything"))
    assert judge.improve_prompt("", []) is None


def test_improve_prompt_none_when_completion_empty():
    judge = LLMJudge(const(""))
    assert judge.improve_prompt("old", []) is None


def test_improve_prompt_none_when_unchanged():
    judge = LLMJudge(const("  same instruction  "))
    # Cleaned == current.strip() -> degenerate -> None
    assert judge.improve_prompt("same instruction", []) is None


def test_improve_prompt_none_when_completion_none():
    judge = LLMJudge(lambda p: None)
    assert judge.improve_prompt("old", []) is None


def test_improve_prompt_truncates_failures_to_max():
    captured = {}

    def fn(prompt):
        captured["prompt"] = prompt
        return "new prompt"

    judge = LLMJudge(fn, max_failures=2)
    failures = [
        {"input": f"in{i}", "output": "o", "expected": "e", "critique": "c"} for i in range(5)
    ]
    judge.improve_prompt("old", failures)
    # Only the first 2 failures should be rendered.
    assert "in0" in captured["prompt"]
    assert "in1" in captured["prompt"]
    assert "in2" not in captured["prompt"]


# -- as_metric() ---------------------------------------------------------------


class _Example:
    def __init__(self, inputs, metadata=None):
        self.inputs = inputs
        self.metadata = metadata or {}


def test_as_metric_basic():
    judge = LLMJudge(const('{"score": 8}'))
    metric = judge.as_metric("judge")
    assert metric.name == "judge"
    assert metric.needs_example is True
    # Metric clamps; 0.8 in range.
    assert metric("out", "exp", _Example("the input")) == pytest.approx(0.8)


def test_as_metric_reads_example_inputs_and_criteria():
    captured = {}

    def fn(prompt):
        captured["prompt"] = prompt
        return '{"score": 5}'

    judge = LLMJudge(fn)
    metric = judge.as_metric()
    ex = _Example("MY_INPUT", metadata={"criteria": "MY_CRITERIA"})
    metric("out", "exp", ex)
    assert "MY_INPUT" in captured["prompt"]
    assert "MY_CRITERIA" in captured["prompt"]


def test_as_metric_none_example_uses_empty_input():
    captured = {}

    def fn(prompt):
        captured["prompt"] = prompt
        return '{"score": 3}'

    judge = LLMJudge(fn)
    metric = judge.as_metric(criteria="default_crit")
    # needs_example=True but call without example -> input_data stays "".
    result = metric("out", "exp", None)
    assert result == pytest.approx(0.3)
    assert "default_crit" in captured["prompt"]


def test_as_metric_example_metadata_none_falls_back_to_default_criteria():
    captured = {}

    def fn(prompt):
        captured["prompt"] = prompt
        return '{"score": 4}'

    judge = LLMJudge(fn)
    metric = judge.as_metric(criteria="fallback_crit")

    class _Ex:
        inputs = "i"
        metadata = None  # `or {}` path

    metric("out", "exp", _Ex())
    assert "fallback_crit" in captured["prompt"]


# -- internals: helper functions ----------------------------------------------


def test_stringify_str_passthrough():
    assert _stringify("hello") == "hello"


def test_stringify_dict_json():
    assert _stringify({"a": 1}) == '{"a": 1}'


def test_stringify_non_serializable_uses_default_str():
    class Weird:
        def __repr__(self):
            return "WEIRD"

    # json.dumps with default=str succeeds -> quoted string.
    assert _stringify(Weird()) == '"WEIRD"'


def test_reference_block_none_empty():
    assert _reference_block(None) == ""


def test_reference_block_present():
    block = _reference_block("answer")
    assert "REFERENCE ANSWER" in block
    assert "answer" in block


def test_criteria_block_empty_and_present():
    assert _criteria_block(None) == ""
    assert _criteria_block("") == ""
    assert "TASK-SPECIFIC CRITERIA" in _criteria_block("c")


def test_extract_json_whole_string():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_embedded():
    assert _extract_json('prefix {"a": 1} suffix') == {"a": 1}


def test_extract_json_with_fences():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_non_dict_returns_none():
    assert _extract_json("[1, 2, 3]") is None


def test_extract_json_no_object_returns_none():
    assert _extract_json("no json here") is None


def test_extract_json_malformed_brace_span_returns_none():
    assert _extract_json("{not valid json}") is None


def test_strip_fences_plain_passthrough():
    assert _strip_fences("plain text") == "plain text"


def test_strip_fences_with_language():
    assert _strip_fences("```python\ncode\n```") == "code"


def test_strip_fences_open_without_close():
    assert _strip_fences("```\ncode here") == "code here"


def test_first_number_int():
    assert _first_number("rating: 7") == 7.0


def test_first_number_float_and_negative():
    assert _first_number("score -3.5 here") == -3.5


def test_first_number_none():
    assert _first_number("no digits") is None


def test_normalize_score_bool_true_false():
    assert _normalize_score(True, 10, 0.0) == 1.0
    assert _normalize_score(False, 10, 0.0) == 0.0


def test_normalize_score_int_on_scale():
    assert _normalize_score(8, 10, 0.0) == pytest.approx(0.8)


def test_normalize_score_already_normalized():
    assert _normalize_score(0.5, 10, 0.0) == 0.5


def test_normalize_score_clamps_over_scale():
    assert _normalize_score(20, 10, 0.0) == 1.0


def test_normalize_score_negative_clamped_to_zero():
    assert _normalize_score(-5, 10, 0.0) == 0.0


def test_normalize_score_string_number():
    assert _normalize_score("7", 10, 0.0) == pytest.approx(0.7)


def test_normalize_score_string_no_number_returns_default():
    assert _normalize_score("none", 10, 0.25) == 0.25


def test_normalize_score_other_type_returns_default():
    assert _normalize_score(None, 10, 0.33) == 0.33
    assert _normalize_score([1], 10, 0.33) == 0.33


def test_normalize_score_scale_one_does_not_take_as_is():
    # scale == 1 disables the [0,1] as-is branch; value/scale used.
    assert _normalize_score(0.5, 1, 0.0) == 0.5  # 0.5/1 = 0.5 anyway
    assert _normalize_score(1, 1, 0.0) == 1.0


def test_render_failures_empty():
    assert _render_failures([]) == "(no specific failures captured)"


def test_render_failures_full():
    out = _render_failures([{"input": "i", "output": "o", "expected": "e", "critique": "bad"}])
    assert "Failure 1" in out
    assert "INPUT: i" in out
    assert "PRODUCED: o" in out
    assert "EXPECTED: e" in out
    assert "CRITIQUE: bad" in out


def test_render_failures_omits_missing_optional():
    out = _render_failures([{"input": "i", "output": "o"}])
    assert "EXPECTED" not in out
    assert "CRITIQUE" not in out


# -- constructor coercion ------------------------------------------------------


def test_constructor_accepts_provider_name_string():
    judge = LLMJudge("echo")
    # echo provider echoes the prompt back; not JSON -> fallback heuristic
    # (the score prompt contains digits like 0-10), so just verify it runs.
    assert isinstance(judge.score("i", "o"), JudgeVerdict)


def test_constructor_accepts_callable():
    judge = LLMJudge(const('{"score": 5}'))
    assert callable(judge.complete)


def test_constructor_min_scale_and_max_failures():
    judge = LLMJudge(const("x"), scale=0, max_failures=0)
    assert judge.scale == 1
    assert judge.max_failures == 1


def test_default_rubric_used():
    judge = LLMJudge(const('{"score": 5}'))
    assert judge.rubric == _DEFAULT_RUBRIC
