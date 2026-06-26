"""Tests for adapt_agent.optimization.metrics."""

import pytest

from adapt_agent.optimization import metrics as M
from adapt_agent.optimization.metrics import (
    Metric,
    coerce_metric,
    contains,
    exact_match,
    get_metric,
    jaccard,
    json_subset,
    levenshtein_ratio,
    normalize_text,
    numeric_close,
    regex_match,
    token_f1,
)


def approx(x):
    return pytest.approx(x, abs=1e-9)


# -- normalize_text -----------------------------------------------------------


def test_normalize_text_lower_strip_punct_whitespace():
    assert normalize_text("  Hello,  WORLD!! ") == "hello world"


def test_normalize_text_no_lower():
    assert normalize_text("Hello World", lower=False) == "Hello World"


def test_normalize_text_no_strip_punct():
    assert normalize_text("a, b.", strip_punct=False) == "a, b."


def test_normalize_text_non_string_inputs():
    assert normalize_text(None) == ""
    assert normalize_text(123) == "123"


# -- Metric wrapper / clamping ------------------------------------------------


def test_metric_call_without_example():
    m = Metric("two", lambda o, e: 2.0)  # out of range -> clamped to 1.0
    assert m("x", "y") == 1.0


def test_metric_clamps_below_zero():
    m = Metric("neg", lambda o, e: -5.0)
    assert m("x", "y") == 0.0


def test_metric_clamp_non_numeric_returns_zero():
    m = Metric("bad", lambda o, e: "not-a-number")
    assert m("x", "y") == 0.0


def test_metric_needs_example_dispatch():
    captured = {}

    def fn(output, expected, example):
        captured["example"] = example
        return 1.0

    m = Metric("ex", fn, needs_example=True)
    assert m("o", "e", example="THE-EXAMPLE") == 1.0
    assert captured["example"] == "THE-EXAMPLE"


def test_metric_not_needs_example_ignores_third_arg():
    def fn(output, expected):
        return 0.5

    m = Metric("noex", fn, needs_example=False)
    assert m("o", "e", example="ignored") == 0.5


# -- coerce_metric ------------------------------------------------------------


def test_coerce_metric_passthrough():
    m = exact_match()
    assert coerce_metric(m) is m


def test_coerce_metric_wraps_named_callable():
    def my_metric(o, e):
        return 1.0

    m = coerce_metric(my_metric)
    assert isinstance(m, Metric)
    assert m.name == "my_metric"


def test_coerce_metric_default_name_for_anonymous():
    m = coerce_metric(lambda o, e: 1.0, default_name="anon")
    # lambda has __name__ == "<lambda>"; truthy so it's used.
    assert m.name == "<lambda>"


def test_coerce_metric_rejects_non_callable():
    with pytest.raises(TypeError):
        coerce_metric(42)  # type: ignore[arg-type]


# -- exact_match --------------------------------------------------------------


def test_exact_match_normalized():
    m = exact_match()
    assert m("Hello, World!", "hello world") == 1.0
    assert m("foo", "bar") == 0.0


def test_exact_match_no_normalize():
    m = exact_match(normalize=False)
    assert m("abc", "abc") == 1.0
    assert m("Abc", "abc") == 0.0


# -- contains -----------------------------------------------------------------


def test_contains_normalized():
    m = contains()
    assert m("The answer is Paris.", "paris") == 1.0
    assert m("The answer is London.", "paris") == 0.0


def test_contains_empty_expected_is_zero():
    m = contains()
    assert m("anything", "") == 0.0


def test_contains_no_normalize():
    m = contains(normalize=False)
    assert m("hello world", "lo wo") == 1.0
    assert m("hello world", "LO WO") == 0.0
    assert m("x", "") == 0.0


# -- regex_match --------------------------------------------------------------


def test_regex_match_explicit_pattern_ignores_expected():
    m = regex_match(r"\d{3}")
    assert m("code 123 here", "anything") == 1.0
    assert m("no digits", "123") == 0.0


def test_regex_match_explicit_pattern_case_insensitive_default():
    m = regex_match(r"hello")
    assert m("HELLO there", None) == 1.0


def test_regex_match_per_example_pattern():
    m = regex_match()  # no explicit pattern -> uses expected as regex
    assert m("the value is 42", r"\d+") == 1.0
    assert m("no number", r"\d+") == 0.0


def test_regex_match_custom_flags():
    m = regex_match(r"hello", flags=0)  # case-sensitive
    assert m("HELLO", None) == 0.0
    assert m("hello", None) == 1.0


# -- token_f1 -----------------------------------------------------------------


def test_token_f1_both_empty():
    assert token_f1()("", "") == 1.0


def test_token_f1_one_empty():
    assert token_f1()("hello", "") == 0.0
    assert token_f1()("", "hello") == 0.0


def test_token_f1_identical():
    assert token_f1()("the cat sat", "the cat sat") == 1.0


def test_token_f1_partial_overlap():
    # pred = a b c (3), gold = b c d (3), overlap = 2
    # precision = 2/3, recall = 2/3, f1 = 2/3
    assert token_f1()("a b c", "b c d") == approx(2 / 3)


def test_token_f1_no_overlap():
    assert token_f1()("a b", "c d") == 0.0


def test_token_f1_repeated_tokens_capped_by_gold_counts():
    # pred has "cat cat", gold has single "cat"; overlap counts only 1.
    # pred = cat cat (2), gold = cat (1), overlap = 1
    # precision = 1/2, recall = 1/1, f1 = 2*0.5*1/(1.5) = 2/3
    assert token_f1()("cat cat", "cat") == approx(2 / 3)


# -- jaccard ------------------------------------------------------------------


def test_jaccard_both_empty():
    assert jaccard()("", "") == 1.0


def test_jaccard_identical():
    assert jaccard()("a b c", "c b a") == 1.0


def test_jaccard_partial():
    # a={a,b,c}, b={b,c,d}; intersection=2, union=4 -> 0.5
    assert jaccard()("a b c", "b c d") == approx(0.5)


def test_jaccard_disjoint():
    assert jaccard()("a b", "c d") == 0.0


def test_jaccard_one_empty():
    # a={}, b={x}; not (both empty); union={x}; intersection 0 -> 0.0
    assert jaccard()("", "x") == 0.0


# -- numeric_close ------------------------------------------------------------


def test_numeric_close_absolute_within_tolerance():
    m = numeric_close(tolerance=0.5)
    assert m("3.2", "3.0") == 1.0
    assert m("4.0", "3.0") == 0.0


def test_numeric_close_default_tolerance():
    m = numeric_close()
    assert m("2.0", "2.0") == 1.0
    assert m("2.0", "2.1") == 0.0


def test_numeric_close_relative():
    m = numeric_close(tolerance=0.1, relative=True)
    # bound = 0.1 * |1000| = 100; diff 50 <= 100 -> match
    assert m("1050", "1000") == 1.0
    # diff 150 > 100 -> no match
    assert m("1150", "1000") == 0.0


def test_numeric_close_extracts_from_text():
    m = numeric_close(tolerance=0.001)
    assert m("the result is 42 units", "42") == 1.0


def test_numeric_close_handles_commas():
    m = numeric_close(tolerance=0.5)
    assert m("1,234", "1234") == 1.0


def test_numeric_close_no_number_is_zero():
    m = numeric_close()
    assert m("no number here", "5") == 0.0
    assert m("5", "no number here") == 0.0


def test_numeric_close_bool_not_treated_as_number():
    m = numeric_close()
    # booleans excluded from numeric extraction -> 0.0
    assert m(True, 1) == 0.0


def test_numeric_close_accepts_native_numbers():
    m = numeric_close(tolerance=0.5)
    assert m(3.2, 3.0) == 1.0


# -- json_subset --------------------------------------------------------------


def test_json_subset_dict_full_match():
    m = json_subset()
    assert m({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2}) == 1.0


def test_json_subset_partial_match():
    m = json_subset()
    # expected has 2 keys, only 1 matches -> 0.5
    assert m({"a": 1, "b": 99}, {"a": 1, "b": 2}) == approx(0.5)


def test_json_subset_json_string_inputs():
    m = json_subset()
    assert m('{"a": 1, "b": 2}', '{"a": 1}') == 1.0


def test_json_subset_invalid_inputs_zero():
    m = json_subset()
    assert m("not json", {"a": 1}) == 0.0
    assert m({"a": 1}, "not json") == 0.0
    assert m([1, 2], {"a": 1}) == 0.0  # list not a mapping


def test_json_subset_empty_expected_zero():
    m = json_subset()
    assert m({"a": 1}, {}) == 0.0


def test_json_subset_json_string_non_dict_zero():
    m = json_subset()
    assert m("[1, 2, 3]", {"a": 1}) == 0.0  # valid JSON but not a dict


# -- levenshtein_ratio --------------------------------------------------------


def test_levenshtein_identical():
    assert levenshtein_ratio()("hello", "hello") == 1.0


def test_levenshtein_both_empty():
    assert levenshtein_ratio()("", "") == 1.0


def test_levenshtein_one_empty():
    assert levenshtein_ratio()("", "abc") == 0.0


def test_levenshtein_typo():
    # "kitten" vs "sitting": distance 3, longest 7 -> 1 - 3/7
    score = levenshtein_ratio()("kitten", "sitting")
    assert score == approx(1 - 3 / 7)


def test_levenshtein_normalizes_before_comparing():
    # normalization makes these identical.
    assert levenshtein_ratio()("Hello, World!", "hello world") == 1.0


# -- low-level helpers (via behaviour) ----------------------------------------


def test_extract_number_negative_and_decimal():
    assert M._extract_number("-3.5") == approx(-3.5)
    assert M._extract_number("value: 7") == approx(7.0)
    assert M._extract_number("none") is None


def test_levenshtein_swap_branch():
    # exercise the len(a) < len(b) swap branch.
    assert M._levenshtein("ab", "abcd") == 2
    assert M._levenshtein("abcd", "ab") == 2


def test_levenshtein_helper_empty_args():
    assert M._levenshtein("", "") == 0
    assert M._levenshtein("", "abc") == 3
    assert M._levenshtein("abc", "") == 3


# -- registry / get_metric ----------------------------------------------------


def test_get_metric_returns_each_builtin():
    for name in M.BUILTIN_METRICS:
        m = get_metric(name)
        assert isinstance(m, Metric)


def test_get_metric_names_match_factory_names():
    # All built-ins are zero-arg factories returning a usable Metric.
    m = get_metric("exact_match")
    assert m.name == "exact_match"
    assert m("a", "a") == 1.0


def test_get_metric_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_metric("does_not_exist")


def test_builtin_metrics_keys_complete():
    expected = {
        "exact_match",
        "contains",
        "regex_match",
        "token_f1",
        "jaccard",
        "numeric_close",
        "json_subset",
        "levenshtein_ratio",
    }
    assert set(M.BUILTIN_METRICS) == expected
