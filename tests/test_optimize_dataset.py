"""Tests for adapt_agent.optimization.dataset."""

import json

import pytest

from adapt_agent.optimization.dataset import Example, GoldenDataset

# -- Example.from_record ------------------------------------------------------


def test_example_defaults():
    ex = Example(inputs="hi")
    assert ex.inputs == "hi"
    assert ex.expected is None
    assert ex.metadata == {}


def test_from_record_key_detection_priority():
    rec = {"question": "Q", "answer": "A"}
    ex = Example.from_record(rec)
    assert ex.inputs == "Q"
    assert ex.expected == "A"
    assert ex.metadata == {}


def test_from_record_input_key_order():
    # "inputs" wins over "question" when both present.
    rec = {"inputs": "first", "question": "second"}
    ex = Example.from_record(rec)
    assert ex.inputs == "first"
    # "question" was not popped, so it survives as metadata.
    assert ex.metadata == {"question": "second"}


def test_from_record_expected_key_order():
    rec = {"input": "i", "expected": "e", "answer": "a"}
    ex = Example.from_record(rec)
    assert ex.expected == "e"
    assert ex.metadata == {"answer": "a"}


def test_from_record_leftover_becomes_metadata():
    rec = {"prompt": "p", "gold": "g", "difficulty": "hard", "id": 7}
    ex = Example.from_record(rec)
    assert ex.inputs == "p"
    assert ex.expected == "g"
    assert ex.metadata == {"difficulty": "hard", "id": 7}


def test_from_record_explicit_metadata_merged_with_leftovers():
    rec = {"input": "i", "output": "o", "metadata": {"a": 1}, "extra": 2}
    ex = Example.from_record(rec)
    assert ex.metadata == {"a": 1, "extra": 2}


def test_from_record_non_dict_metadata_ignored():
    rec = {"input": "i", "metadata": "not-a-dict", "extra": 9}
    ex = Example.from_record(rec)
    # Non-dict metadata is discarded; only leftovers remain.
    assert ex.metadata == {"extra": 9}


def test_from_record_does_not_mutate_input():
    rec = {"input": "i", "expected": "e"}
    Example.from_record(rec)
    assert rec == {"input": "i", "expected": "e"}


def test_from_record_missing_keys_give_none():
    ex = Example.from_record({"foo": "bar"})
    assert ex.inputs is None
    assert ex.expected is None
    assert ex.metadata == {"foo": "bar"}


# -- GoldenDataset construction / from_list -----------------------------------


def test_init_rejects_non_example():
    with pytest.raises(TypeError):
        GoldenDataset([Example("ok"), {"not": "example"}])  # type: ignore[list-item]


def test_from_list_mixed_example_and_dict():
    ds = GoldenDataset.from_list(
        [Example(inputs="a", expected="A"), {"input": "b", "expected": "B"}]
    )
    assert len(ds) == 2
    assert ds[0].inputs == "a"
    assert ds[1].inputs == "b"
    assert ds[1].expected == "B"


def test_from_list_rejects_bad_type():
    with pytest.raises(TypeError):
        GoldenDataset.from_list([42])


# -- from_json ----------------------------------------------------------------


def test_from_json_list(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps([{"input": "x", "expected": "y"}]), encoding="utf-8")
    ds = GoldenDataset.from_json(path)
    assert len(ds) == 1
    assert ds[0].inputs == "x"
    assert ds[0].expected == "y"


def test_from_json_examples_wrapper(tmp_path):
    path = tmp_path / "d.json"
    payload = {"examples": [{"question": "q", "answer": "a"}]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    ds = GoldenDataset.from_json(path)
    assert len(ds) == 1
    assert ds[0].inputs == "q"
    assert ds[0].expected == "a"


def test_from_json_invalid_shape_raises(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"nope": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        GoldenDataset.from_json(path)


def test_from_json_accepts_string_path(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps([{"input": "x"}]), encoding="utf-8")
    ds = GoldenDataset.from_json(str(path))
    assert len(ds) == 1


# -- from_jsonl ---------------------------------------------------------------


def test_from_jsonl(tmp_path):
    path = tmp_path / "d.jsonl"
    lines = [
        json.dumps({"input": "a", "expected": "A"}),
        "",  # blank line skipped
        "   ",  # whitespace-only skipped
        json.dumps({"input": "b", "expected": "B", "tag": "t"}),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    ds = GoldenDataset.from_jsonl(path)
    assert len(ds) == 2
    assert ds[0].inputs == "a"
    assert ds[1].metadata == {"tag": "t"}


# -- from_csv -----------------------------------------------------------------


def test_from_csv(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text(
        "question,answer,difficulty\n" "What is 2+2?,4,easy\n" "Capital of France?,Paris,medium\n",
        encoding="utf-8",
    )
    ds = GoldenDataset.from_csv(path)
    assert len(ds) == 2
    assert ds[0].inputs == "What is 2+2?"
    assert ds[0].expected == "4"
    assert ds[0].metadata == {"difficulty": "easy"}
    assert ds[1].expected == "Paris"


# -- sequence protocol --------------------------------------------------------


def test_len_and_bool():
    assert len(GoldenDataset()) == 0
    assert bool(GoldenDataset()) is False
    ds = GoldenDataset([Example("a")])
    assert len(ds) == 1
    assert bool(ds) is True


def test_iter():
    ds = GoldenDataset([Example("a"), Example("b")])
    assert [ex.inputs for ex in ds] == ["a", "b"]


def test_getitem_int():
    ds = GoldenDataset([Example("a"), Example("b")])
    assert ds[1].inputs == "b"


def test_getitem_slice_returns_goldendataset():
    ds = GoldenDataset([Example(str(i)) for i in range(5)])
    sub = ds[1:3]
    assert isinstance(sub, GoldenDataset)
    assert [ex.inputs for ex in sub] == ["1", "2"]


def test_examples_property_is_copy():
    ds = GoldenDataset([Example("a")])
    copy = ds.examples
    copy.append(Example("b"))
    assert len(ds) == 1


# -- transforms ---------------------------------------------------------------


def test_shuffled_deterministic_and_non_mutating():
    ds = GoldenDataset([Example(str(i)) for i in range(10)])
    a = ds.shuffled(seed=42)
    b = ds.shuffled(seed=42)
    assert [e.inputs for e in a] == [e.inputs for e in b]
    # Original unchanged.
    assert [e.inputs for e in ds] == [str(i) for i in range(10)]
    # Different seed -> (very likely) different order.
    c = ds.shuffled(seed=1)
    assert [e.inputs for e in a] != [e.inputs for e in c]


def test_split_fractions():
    ds = GoldenDataset([Example(str(i)) for i in range(10)])
    train, holdout = ds.split(0.7, seed=0)
    assert len(train) == 7
    assert len(holdout) == 3
    assert isinstance(train, GoldenDataset)
    assert isinstance(holdout, GoldenDataset)


def test_split_rounding():
    ds = GoldenDataset([Example(str(i)) for i in range(10)])
    train, holdout = ds.split(0.75, seed=0)
    # round(10 * 0.75) == 8 in banker's rounding? 7.5 -> 8.
    assert len(train) == 8
    assert len(holdout) == 2


def test_split_deterministic_with_seed():
    ds = GoldenDataset([Example(str(i)) for i in range(20)])
    t1, h1 = ds.split(0.5, seed=5)
    t2, h2 = ds.split(0.5, seed=5)
    assert [e.inputs for e in t1] == [e.inputs for e in t2]
    assert [e.inputs for e in h1] == [e.inputs for e in h2]


def test_split_no_shuffle_preserves_order():
    ds = GoldenDataset([Example(str(i)) for i in range(10)])
    train, holdout = ds.split(0.6, shuffle=False)
    assert [e.inputs for e in train] == [str(i) for i in range(6)]
    assert [e.inputs for e in holdout] == [str(i) for i in range(6, 10)]


def test_split_invalid_fraction_raises():
    ds = GoldenDataset([Example("a")])
    with pytest.raises(ValueError):
        ds.split(1.5)
    with pytest.raises(ValueError):
        ds.split(-0.1)


def test_split_boundary_fractions():
    ds = GoldenDataset([Example(str(i)) for i in range(4)])
    train, holdout = ds.split(0.0, shuffle=False)
    assert len(train) == 0
    assert len(holdout) == 4
    train, holdout = ds.split(1.0, shuffle=False)
    assert len(train) == 4
    assert len(holdout) == 0


def test_sample_subset_deterministic():
    ds = GoldenDataset([Example(str(i)) for i in range(10)])
    s1 = ds.sample(3, seed=7)
    s2 = ds.sample(3, seed=7)
    assert len(s1) == 3
    assert [e.inputs for e in s1] == [e.inputs for e in s2]


def test_sample_n_at_least_len_returns_all():
    ds = GoldenDataset([Example("a"), Example("b")])
    s = ds.sample(5, seed=0)
    assert len(s) == 2
    assert isinstance(s, GoldenDataset)


def test_filter():
    ds = GoldenDataset([Example(i) for i in range(6)])
    even = ds.filter(lambda ex: ex.inputs % 2 == 0)
    assert isinstance(even, GoldenDataset)
    assert [e.inputs for e in even] == [0, 2, 4]


# -- to_records round-trip ----------------------------------------------------


def test_to_records_includes_metadata_only_when_present():
    ds = GoldenDataset(
        [
            Example(inputs="a", expected="A", metadata={"k": 1}),
            Example(inputs="b", expected="B"),
        ]
    )
    recs = ds.to_records()
    assert recs[0] == {"inputs": "a", "expected": "A", "metadata": {"k": 1}}
    assert recs[1] == {"inputs": "b", "expected": "B"}
    assert "metadata" not in recs[1]


def test_to_records_round_trip_through_from_list():
    ds = GoldenDataset(
        [
            Example(inputs="a", expected="A", metadata={"k": 1}),
            Example(inputs="b", expected=None),
        ]
    )
    recs = ds.to_records()
    ds2 = GoldenDataset.from_list(recs)
    assert len(ds2) == 2
    assert ds2[0].inputs == "a"
    assert ds2[0].expected == "A"
    assert ds2[0].metadata == {"k": 1}
    assert ds2[1].inputs == "b"
    assert ds2[1].expected is None


def test_to_records_metadata_is_copied():
    ex = Example(inputs="a", metadata={"k": 1})
    ds = GoldenDataset([ex])
    rec = ds.to_records()[0]
    rec["metadata"]["k"] = 999
    assert ex.metadata == {"k": 1}
