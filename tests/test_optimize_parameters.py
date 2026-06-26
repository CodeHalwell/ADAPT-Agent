"""Tests for adapt_agent.optimization.parameters."""

import random

import pytest

from adapt_agent.optimization.parameters import (
    Parameter,
    ParameterKind,
    SearchSpace,
)

# -- helpers ------------------------------------------------------------------


class Cell:
    """A tiny live component with a readable/writable attribute."""

    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, v):
        self.value = v


def make_param(name="p", **kw):
    kw.setdefault("kind", ParameterKind.HYPERPARAM)
    return Parameter(name=name, **kw)


# -- Parameter construction / validation --------------------------------------


def test_post_init_accepts_raw_string_kind():
    p = Parameter(name="x", kind="prompt")
    assert p.kind is ParameterKind.PROMPT


def test_post_init_rejects_empty_name():
    with pytest.raises(ValueError):
        Parameter(name="", kind=ParameterKind.PROMPT)


def test_post_init_rejects_non_string_name():
    with pytest.raises(ValueError):
        Parameter(name=None, kind=ParameterKind.PROMPT)  # type: ignore[arg-type]


def test_post_init_rejects_invalid_kind_string():
    with pytest.raises(ValueError):
        Parameter(name="x", kind="not-a-kind")


def test_post_init_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        Parameter(name="x", kind=ParameterKind.HYPERPARAM, bounds=(1.0, 0.0))


def test_post_init_allows_equal_bounds():
    p = Parameter(name="x", kind=ParameterKind.HYPERPARAM, bounds=(0.5, 0.5))
    assert p.bounds == (0.5, 0.5)


# -- optimizable property -----------------------------------------------------


def test_optimizable_true_when_mutable_and_setter():
    p = make_param(setter=lambda v: None)
    assert p.optimizable is True


def test_optimizable_false_without_setter():
    p = make_param()
    assert p.optimizable is False


def test_optimizable_false_when_not_mutable():
    p = make_param(setter=lambda v: None, mutable=False)
    assert p.optimizable is False


# -- read ---------------------------------------------------------------------


def test_read_prefers_getter():
    cell = Cell("live")
    p = make_param(value="cached", getter=cell.get)
    assert p.read() == "live"


def test_read_falls_back_to_cached_value_without_getter():
    p = make_param(value="cached")
    assert p.read() == "cached"


def test_read_failing_getter_falls_back_to_value():
    def boom():
        raise RuntimeError("flaky")

    p = make_param(value="cached", getter=boom)
    assert p.read() == "cached"


# -- write --------------------------------------------------------------------


def test_write_sets_value_and_updates_cache():
    cell = Cell()
    p = make_param(setter=cell.set)
    p.write(42)
    assert cell.value == 42
    assert p.value == 42


def test_write_read_only_raises():
    p = make_param()
    with pytest.raises(ValueError, match="read-only"):
        p.write(1)


# -- enumerate_candidates -----------------------------------------------------


def test_enumerate_explicit_candidates():
    p = make_param(candidates=["a", "b", "c"])
    out = p.enumerate_candidates()
    assert out == ["a", "b", "c"]
    # Returns a fresh list (not the same object).
    assert out is not p.candidates


def test_enumerate_numeric_bounds_no_step_default_points():
    # bounds (0.5, 2.5) are non-integral, so points stay floats.
    p = make_param(bounds=(0.5, 2.5))
    out = p.enumerate_candidates(numeric_points=5)
    assert out == [0.5, 1.0, 1.5, 2.0, 2.5]


def test_enumerate_numeric_bounds_custom_points():
    p = make_param(bounds=(0.5, 2.5))
    out = p.enumerate_candidates(numeric_points=3)
    assert out == [0.5, 1.5, 2.5]


def test_enumerate_float_bounds_stay_float():
    # (0.0, 1.0) are FLOAT literals (e.g. top_p) -> stays continuous, not collapsed.
    p = make_param(bounds=(0.0, 1.0))
    out = p.enumerate_candidates(numeric_points=5)
    assert out == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert all(isinstance(x, float) for x in out)


def test_enumerate_int_bounds_collapse_to_int():
    # (0, 4) are genuine int literals (e.g. max_tokens) -> coerced to int.
    p = make_param(bounds=(0, 4))
    out = p.enumerate_candidates(numeric_points=5)
    assert out == [0, 1, 2, 3, 4]
    assert all(isinstance(x, int) for x in out)


def test_enumerate_numeric_points_one_returns_low():
    p = make_param(bounds=(0.2, 0.9))
    assert p.enumerate_candidates(numeric_points=1) == [0.2]


def test_enumerate_numeric_equal_bounds_returns_single():
    p = make_param(bounds=(0.5, 0.5))
    assert p.enumerate_candidates(numeric_points=5) == [0.5]


def test_enumerate_numeric_with_step():
    p = make_param(bounds=(0.0, 1.0), step=0.5)
    out = p.enumerate_candidates()
    assert out == [0.0, 0.5, 1.0]


def test_enumerate_numeric_with_zero_step_does_not_hang():
    # step of 0 is falsy -> `if self.step:` False -> numeric_points path.
    # (0.0, 2.0) are float literals -> values stay float (continuous).
    p = make_param(bounds=(0.0, 2.0), step=0)
    out = p.enumerate_candidates(numeric_points=5)
    assert out == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert all(isinstance(x, float) for x in out)


def test_enumerate_integer_bounds_coerced_to_int_with_step():
    p = make_param(bounds=(1, 4), step=1)
    out = p.enumerate_candidates()
    assert out == [1, 2, 3, 4]
    assert all(isinstance(x, int) for x in out)


def test_enumerate_integer_bounds_coerced_without_step():
    p = make_param(bounds=(0, 10))
    out = p.enumerate_candidates(numeric_points=3)
    assert out == [0, 5, 10]
    assert all(isinstance(x, int) for x in out)


def test_enumerate_integer_bounds_with_non_integer_step_no_int_collapse():
    # A non-integral step disables the int-collapse: values pass through
    # round(raw, 6). The starting `low` (an int) stays int; stepped values
    # become floats.
    p = make_param(bounds=(0, 1), step=0.5)
    out = p.enumerate_candidates()
    assert out == [0, 0.5, 1.0]
    assert isinstance(out[1], float) and isinstance(out[2], float)


def test_enumerate_falls_back_to_read():
    p = make_param(value="only")
    assert p.enumerate_candidates() == ["only"]


# -- sample -------------------------------------------------------------------


def test_sample_candidates_deterministic():
    p = make_param(candidates=[10, 20, 30, 40])
    a = [p.sample(random.Random(7)) for _ in range(3)]
    b = [p.sample(random.Random(7)) for _ in range(3)]
    assert a == b
    assert all(x in [10, 20, 30, 40] for x in a)


def test_sample_numeric_bounds_deterministic_and_in_range():
    p = make_param(bounds=(0.0, 1.0))
    v1 = p.sample(random.Random(123))
    v2 = p.sample(random.Random(123))
    assert v1 == v2
    assert 0.0 <= v1 <= 1.0


def test_sample_integer_bounds_returns_int():
    p = make_param(bounds=(0, 100))
    v = p.sample(random.Random(1))
    assert isinstance(v, int)
    assert 0 <= v <= 100


def test_sample_no_space_returns_read():
    p = make_param(value="x")
    assert p.sample(random.Random(0)) == "x"


# -- SearchSpace --------------------------------------------------------------


def test_searchspace_add_and_membership():
    s = SearchSpace()
    p = make_param("a")
    s.add(p)
    assert "a" in s
    assert len(s) == 1
    assert s["a"] is p
    assert s.names == ["a"]


def test_searchspace_init_from_iterable():
    s = SearchSpace([make_param("a"), make_param("b")])
    assert len(s) == 2
    assert [p.name for p in s] == ["a", "b"]


def test_searchspace_rejects_duplicate():
    s = SearchSpace([make_param("a")])
    with pytest.raises(ValueError, match="Duplicate"):
        s.add(make_param("a"))


def test_searchspace_contains_non_string():
    s = SearchSpace([make_param("a")])
    assert (123 in s) is False


def test_of_kind_filter():
    s = SearchSpace(
        [
            make_param("a", kind=ParameterKind.PROMPT),
            make_param("b", kind=ParameterKind.MODEL),
            make_param("c", kind=ParameterKind.PROMPT),
        ]
    )
    names = [p.name for p in s.of_kind(ParameterKind.PROMPT)]
    assert names == ["a", "c"]


def test_of_component_filter():
    s = SearchSpace(
        [
            make_param("a", component="researcher"),
            make_param("b", component="writer"),
            make_param("c", component="researcher"),
        ]
    )
    names = [p.name for p in s.of_component("researcher")]
    assert names == ["a", "c"]


def test_optimizable_filter():
    s = SearchSpace(
        [
            make_param("a", setter=lambda v: None),
            make_param("b"),  # no setter
            make_param("c", setter=lambda v: None, mutable=False),
        ]
    )
    assert [p.name for p in s.optimizable()] == ["a"]


# -- snapshot / apply / restore -----------------------------------------------


def test_snapshot_apply_restore_round_trip():
    c1, c2 = Cell("p1-orig"), Cell(0.5)
    s = SearchSpace(
        [
            make_param("p1", kind=ParameterKind.PROMPT, getter=c1.get, setter=c1.set),
            make_param("p2", getter=c2.get, setter=c2.set, bounds=(0.0, 1.0)),
        ]
    )
    snap = s.snapshot()
    assert snap == {"p1": "p1-orig", "p2": 0.5}

    s.apply({"p1": "p1-new", "p2": 0.9})
    assert c1.value == "p1-new"
    assert c2.value == 0.9

    s.restore(snap)
    assert c1.value == "p1-orig"
    assert c2.value == 0.5


def test_apply_ignores_unknown_and_readonly_keys():
    cell = Cell("orig")
    s = SearchSpace(
        [
            make_param("known", getter=cell.get, setter=cell.set),
            make_param("ro", value="ro"),  # read-only, has no setter
        ]
    )
    s.apply({"known": "new", "unknown": "x", "ro": "changed"})
    assert cell.value == "new"
    assert s["ro"].value == "ro"


def test_restore_skips_params_without_setter():
    cell = Cell("a")
    s = SearchSpace(
        [
            make_param("has_setter", getter=cell.get, setter=cell.set),
            make_param("no_setter", value="keep"),
        ]
    )
    # no_setter present in snapshot but has no setter -> silently skipped.
    s.restore({"has_setter": "z", "no_setter": "ignored", "absent": "x"})
    assert cell.value == "z"
    assert s["no_setter"].value == "keep"


def test_restore_failing_setter_falls_back_to_value_assignment():
    def boom(_):
        raise RuntimeError("cannot write")

    p = make_param("p", setter=boom, value="orig")
    s = SearchSpace([p])
    s.restore({"p": "fallback"})
    # write() raised, restore caught it and assigned .value directly.
    assert p.value == "fallback"


# -- grid ---------------------------------------------------------------------


def test_grid_cartesian_product():
    s = SearchSpace(
        [
            make_param("a", candidates=[1, 2], setter=lambda v: None),
            make_param("b", candidates=["x", "y"], setter=lambda v: None),
        ]
    )
    grid = s.grid()
    assert len(grid) == 4
    assert {"a": 1, "b": "x"} in grid
    assert {"a": 2, "b": "y"} in grid


def test_grid_skips_single_option_params():
    s = SearchSpace(
        [
            make_param("a", candidates=[1, 2], setter=lambda v: None),
            make_param("b", candidates=["only"], setter=lambda v: None),
        ]
    )
    grid = s.grid()
    assert len(grid) == 2
    assert all("b" not in cfg for cfg in grid)


def test_grid_ignores_non_optimizable():
    s = SearchSpace(
        [
            make_param("a", candidates=[1, 2], setter=lambda v: None),
            make_param("b", candidates=[3, 4]),  # no setter -> not optimizable
        ]
    )
    grid = s.grid()
    assert len(grid) == 2
    assert all("b" not in cfg for cfg in grid)


def test_grid_bounded_by_max_configs():
    # Two params with 3 candidates each = 9 combos; cap at 4 keeps only first.
    s = SearchSpace(
        [
            make_param("a", candidates=[1, 2, 3], setter=lambda v: None),
            make_param("b", candidates=[4, 5, 6], setter=lambda v: None),
        ]
    )
    grid = s.grid(max_configs=4)
    # First param expands to 3; second would push to 9 > 4 so it's pinned.
    assert len(grid) == 3
    assert all(set(cfg) == {"a"} for cfg in grid)


def test_grid_empty_when_no_optimizable():
    s = SearchSpace([make_param("a", value=1)])
    assert s.grid() == [{}]


# -- sample_config ------------------------------------------------------------


def test_sample_config_deterministic():
    s = SearchSpace(
        [
            make_param("a", candidates=[1, 2, 3], setter=lambda v: None),
            make_param("b", bounds=(0.0, 1.0), setter=lambda v: None),
            make_param("c", value="fixed"),  # not optimizable, excluded
        ]
    )
    cfg1 = s.sample_config(random.Random(99))
    cfg2 = s.sample_config(random.Random(99))
    assert cfg1 == cfg2
    assert set(cfg1) == {"a", "b"}
    assert cfg1["a"] in [1, 2, 3]
    assert 0.0 <= cfg1["b"] <= 1.0
