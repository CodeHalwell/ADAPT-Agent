"""Tests for the Middleware system."""

import pytest

from adapt_agent.core import Middleware


def test_pre_middleware_ordering_by_priority():
    """Higher-priority pre-middleware should run first."""
    mw = Middleware()
    calls = []

    def low(data):
        calls.append("low")
        return data

    def high(data):
        calls.append("high")
        return data

    mw.add_pre_middleware(low, name="low", priority=1)
    mw.add_pre_middleware(high, name="high", priority=10)

    mw.process_input({})
    assert calls == ["high", "low"]


def test_post_middleware_ordering_by_priority():
    """Higher-priority post-middleware should run first."""
    mw = Middleware()
    calls = []

    def low(data):
        calls.append("low")
        return data

    def high(data):
        calls.append("high")
        return data

    mw.add_post_middleware(low, name="low", priority=1)
    mw.add_post_middleware(high, name="high", priority=10)

    mw.process_output({})
    assert calls == ["high", "low"]


def test_process_input_applies_transform():
    """process_input should apply middleware transforms."""
    mw = Middleware()

    def add_flag(data):
        data["flag"] = True
        return data

    mw.add_pre_middleware(add_flag, name="add_flag")
    result = mw.process_input({"x": 1})
    assert result["flag"] is True
    assert result["x"] == 1


def test_process_input_no_middleware_returns_same_object():
    """With no pre-middleware, the same object is returned (fast path)."""
    mw = Middleware()
    data = {"x": 1}
    assert mw.process_input(data) is data


def test_process_output_applies_transform():
    """process_output should apply middleware transforms."""
    mw = Middleware()

    def upper(data):
        data["result"] = str(data["result"]).upper()
        return data

    mw.add_post_middleware(upper, name="upper")
    result = mw.process_output({"result": "hello"})
    assert result["result"] == "HELLO"


def test_middleware_exception_is_caught_and_pipeline_continues(caplog):
    """A raising middleware is caught/logged and the pipeline continues."""
    mw = Middleware()
    calls = []

    def boom(data):
        raise RuntimeError("boom")

    def after(data):
        calls.append("after")
        data["after"] = True
        return data

    mw.add_pre_middleware(boom, name="boom", priority=10)
    mw.add_pre_middleware(after, name="after", priority=1)

    with caplog.at_level("ERROR"):
        result = mw.process_input({})

    assert calls == ["after"]
    assert result["after"] is True
    assert any("boom" in rec.message for rec in caplog.records)


def test_fail_closed_aborts_pre_pipeline_on_error():
    """With fail_closed=True, a raising pre-middleware aborts (re-raises)."""
    mw = Middleware(fail_closed=True)
    calls = []

    def boom(data):
        raise RuntimeError("sanitizer failed")

    def after(data):
        calls.append("after")
        data["after"] = True
        return data

    mw.add_pre_middleware(boom, name="boom", priority=10)
    mw.add_pre_middleware(after, name="after", priority=1)

    with pytest.raises(RuntimeError, match="sanitizer failed"):
        mw.process_input({})

    # Pipeline aborted: the downstream middleware never ran.
    assert calls == []


def test_fail_closed_aborts_post_pipeline_on_error():
    """With fail_closed=True, a raising post-middleware aborts (re-raises)."""
    mw = Middleware(fail_closed=True)

    def boom(data):
        raise ValueError("kaboom")

    mw.add_post_middleware(boom, name="boom")
    with pytest.raises(ValueError, match="kaboom"):
        mw.process_output({"result": 5})


def test_fail_open_is_default_and_continues(caplog):
    """Default (fail_closed=False) logs the error and passes data through."""
    mw = Middleware()
    assert mw.fail_closed is False

    def boom(data):
        raise RuntimeError("boom")

    mw.add_pre_middleware(boom, name="boom")
    with caplog.at_level("ERROR"):
        result = mw.process_input({"x": 1})
    # Unmodified data flows through and the error is logged clearly.
    assert result["x"] == 1
    assert any("boom" in rec.message for rec in caplog.records)


def test_duplicate_pre_middleware_name_rejected():
    """Registering two pre-middleware with the same name raises."""
    mw = Middleware()

    def a(data):
        return data

    def b(data):
        return data

    mw.add_pre_middleware(a, name="dup")
    with pytest.raises(ValueError, match="already registered"):
        mw.add_pre_middleware(b, name="dup")

    # Registry not corrupted: still exactly one entry, and remove works once.
    assert len(mw.list_middleware()) == 1
    assert mw.remove_middleware("dup") is True
    assert mw.remove_middleware("dup") is False


def test_duplicate_post_middleware_name_rejected():
    """Registering two post-middleware with the same name raises."""
    mw = Middleware()

    def a(data):
        return data

    def b(data):
        return data

    mw.add_post_middleware(a, name="dup")
    with pytest.raises(ValueError, match="already registered"):
        mw.add_post_middleware(b, name="dup")


def test_duplicate_name_across_pre_and_post_rejected():
    """A pre and a post middleware cannot share the same name."""
    mw = Middleware()

    def a(data):
        return data

    def b(data):
        return data

    mw.add_pre_middleware(a, name="dup")
    with pytest.raises(ValueError, match="already registered"):
        mw.add_post_middleware(b, name="dup")


def test_process_output_exception_is_caught(caplog):
    """A raising post-middleware is caught and logged."""
    mw = Middleware()

    def boom(data):
        raise ValueError("kaboom")

    mw.add_post_middleware(boom, name="boom")
    with caplog.at_level("ERROR"):
        result = mw.process_output({"result": 5})
    assert result["result"] == 5
    assert any("boom" in rec.message for rec in caplog.records)


def test_remove_middleware_true_and_false():
    """remove_middleware returns True when found, False otherwise."""
    mw = Middleware()

    def m(data):
        return data

    mw.add_pre_middleware(m, name="m")
    assert mw.remove_middleware("m") is True
    assert mw.remove_middleware("m") is False
    assert mw.remove_middleware("never") is False


def test_remove_post_middleware():
    """Removing a post middleware works and clears it from the pipeline."""
    mw = Middleware()

    def m(data):
        data["touched"] = True
        return data

    mw.add_post_middleware(m, name="m")
    assert mw.remove_middleware("m") is True
    # No longer applied
    result = mw.process_output({"result": 1})
    assert "touched" not in result


def test_list_middleware_reflects_registrations():
    """list_middleware reflects current registrations."""
    mw = Middleware()
    assert mw.list_middleware() == []

    def pre(data):
        return data

    def post(data):
        return data

    mw.add_pre_middleware(pre, name="pre", priority=5)
    mw.add_post_middleware(post, name="post", priority=2)

    listing = mw.list_middleware()
    names = {entry["name"] for entry in listing}
    assert names == {"pre", "post"}
    by_name = {entry["name"]: entry for entry in listing}
    assert by_name["pre"]["type"] == "pre"
    assert by_name["pre"]["priority"] == 5
    assert by_name["post"]["type"] == "post"


def test_wrap_function_runs_pre_and_post():
    """wrap_function wraps a callable and runs pre/post middleware."""
    mw = Middleware()

    def double_arg(data):
        args = data["args"]
        data["args"] = (args[0] * 2,)
        return data

    def add_ten(data):
        data["result"] = data["result"] + 10
        return data

    mw.add_pre_middleware(double_arg, name="double_arg")
    mw.add_post_middleware(add_ten, name="add_ten")

    def fn(x):
        return x + 1

    wrapped = mw.wrap_function(fn)
    # pre doubles arg: 5 -> 10, fn -> 11, post adds 10 -> 21
    assert wrapped(5) == 21


def test_wrap_function_fast_path_no_middleware():
    """With no middleware, wrap_function calls the function directly."""
    mw = Middleware()

    def fn(x, y=0):
        return x + y

    wrapped = mw.wrap_function(fn)
    assert wrapped(3, y=4) == 7
    # __name__ preserved via functools.wraps
    assert wrapped.__name__ == "fn"
