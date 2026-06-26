"""Offline, deterministic tests for ``adapt_agent.optimization.introspection``."""

import pytest

import adapt_agent.optimization.introspection as intro
from adapt_agent.optimization.introspection import (
    available,
    bind_attr,
    bind_item,
    bind_mapping_key,
    detect,
    introspect,
    introspect_components,
    register,
)
from adapt_agent.optimization.parameters import Parameter, ParameterKind


@pytest.fixture
def clean_registry():
    """Snapshot the global registry and restore it after the test.

    Also forces ``_loaded`` so ``_ensure_loaded`` is a no-op during the test
    (the real framework modules already register themselves) and restores it.
    """
    saved_registry = list(intro._REGISTRY)
    saved_names = set(intro._REGISTERED_NAMES)
    saved_loaded = intro._loaded
    intro._loaded = True  # prevent lazy reload from re-populating mid-test
    try:
        yield
    finally:
        intro._REGISTRY = saved_registry
        intro._REGISTERED_NAMES = saved_names
        intro._loaded = saved_loaded


# -- register / detect / introspect routing -----------------------------------


def _make_dummy_introspector(marker):
    def predicate(obj):
        return getattr(obj, "_dummy_marker", None) == marker

    def introspector(obj):
        return [Parameter(name="knob", kind=ParameterKind.PROMPT, value="v")]

    return predicate, introspector


class _Dummy:
    def __init__(self, marker="m"):
        self._dummy_marker = marker


def test_register_and_detect(clean_registry):
    pred, ins = _make_dummy_introspector("m")
    register("dummy", pred, ins)
    assert "dummy" in available()
    assert detect(_Dummy("m")) == "dummy"


def test_detect_non_matching_returns_none(clean_registry):
    intro._REGISTRY = []
    intro._REGISTERED_NAMES = set()
    pred, ins = _make_dummy_introspector("m")
    register("dummy", pred, ins)
    assert detect(object()) is None


def test_introspect_routes_to_registered(clean_registry):
    intro._REGISTRY = []
    intro._REGISTERED_NAMES = set()
    pred, ins = _make_dummy_introspector("m")
    register("dummy", pred, ins)
    params = introspect(_Dummy("m"))
    assert [p.name for p in params] == ["knob"]


def test_introspect_non_matching_returns_empty(clean_registry):
    intro._REGISTRY = []
    intro._REGISTERED_NAMES = set()
    pred, ins = _make_dummy_introspector("m")
    register("dummy", pred, ins)
    assert introspect(object()) == []


def test_register_replaces_existing_name(clean_registry):
    intro._REGISTRY = []
    intro._REGISTERED_NAMES = set()
    register("dummy", lambda o: True, lambda o: [Parameter("a", ParameterKind.PROMPT)])
    register("dummy", lambda o: True, lambda o: [Parameter("b", ParameterKind.PROMPT)])
    # Only one entry, and it is the second registration.
    assert [n for n, _, _ in intro._REGISTRY].count("dummy") == 1
    assert [p.name for p in introspect(object())] == ["b"]


def test_predicate_raising_is_swallowed(clean_registry):
    intro._REGISTRY = []
    intro._REGISTERED_NAMES = set()

    def boom_pred(obj):
        raise RuntimeError("predicate boom")

    register("boomer", boom_pred, lambda o: [Parameter("x", ParameterKind.PROMPT)])
    register(
        "good",
        lambda o: True,
        lambda o: [Parameter("y", ParameterKind.PROMPT)],
    )
    # detect skips the raising predicate and finds the good one.
    assert detect(object()) == "good"
    # introspect skips the raising one too.
    assert [p.name for p in introspect(object())] == ["y"]


def test_introspector_raising_is_swallowed(clean_registry):
    intro._REGISTRY = []
    intro._REGISTERED_NAMES = set()

    def boom_ins(obj):
        raise RuntimeError("introspector boom")

    register("boomer", lambda o: True, boom_ins)
    register("good", lambda o: True, lambda o: [Parameter("z", ParameterKind.PROMPT)])
    assert [p.name for p in introspect(object())] == ["z"]


def test_introspect_skips_empty_result_tries_next(clean_registry):
    intro._REGISTRY = []
    intro._REGISTERED_NAMES = set()
    register("empty", lambda o: True, lambda o: [])
    register("full", lambda o: True, lambda o: [Parameter("p", ParameterKind.PROMPT)])
    assert [p.name for p in introspect(object())] == ["p"]


# -- component prefixing ------------------------------------------------------


def test_introspect_component_prefixing(clean_registry):
    intro._REGISTRY = []
    intro._REGISTERED_NAMES = set()
    register(
        "dummy",
        lambda o: True,
        lambda o: [Parameter("knob", ParameterKind.PROMPT, value="v")],
    )
    params = introspect(object(), component="comp")
    assert params[0].name == "comp.knob"
    assert params[0].component == "comp"


def test_introspect_component_not_double_prefixed(clean_registry):
    intro._REGISTRY = []
    intro._REGISTERED_NAMES = set()
    register(
        "dummy",
        lambda o: True,
        lambda o: [Parameter("comp.knob", ParameterKind.PROMPT, value="v")],
    )
    params = introspect(object(), component="comp")
    # Already prefixed -> not doubled.
    assert params[0].name == "comp.knob"


def test_introspect_components_namespacing_and_dedupe(clean_registry):
    intro._REGISTRY = []
    intro._REGISTERED_NAMES = set()
    # Every object yields a param named "knob"; namespacing keeps them distinct.
    register(
        "dummy",
        lambda o: True,
        lambda o: [Parameter("knob", ParameterKind.PROMPT, value="v")],
    )
    params = introspect_components({"a": object(), "b": object()})
    names = [p.name for p in params]
    assert names == ["a.knob", "b.knob"]


def test_introspect_components_dedupes_duplicate_names(clean_registry):
    intro._REGISTRY = []
    intro._REGISTERED_NAMES = set()
    # Introspector returns two params with the SAME name under one component;
    # after prefixing they collide and the duplicate is dropped.
    register(
        "dummy",
        lambda o: True,
        lambda o: [
            Parameter("knob", ParameterKind.PROMPT, value="1"),
            Parameter("knob", ParameterKind.PROMPT, value="2"),
        ],
    )
    params = introspect_components({"a": object()})
    assert [p.name for p in params] == ["a.knob"]


def test_introspect_components_unrecognized_contributes_nothing(clean_registry):
    intro._REGISTRY = []
    intro._REGISTERED_NAMES = set()
    register("dummy", lambda o: False, lambda o: [Parameter("k", ParameterKind.PROMPT)])
    assert introspect_components({"a": object()}) == []


# -- bind_attr ----------------------------------------------------------------


def test_bind_attr_missing_returns_none():
    class Obj:
        pass

    assert bind_attr(Obj(), "nope", "n", ParameterKind.PROMPT) is None


def test_bind_attr_present_getter_setter_roundtrip():
    class Obj:
        def __init__(self):
            self.prompt = "hello"

    obj = Obj()
    param = bind_attr(obj, "prompt", "obj.prompt", ParameterKind.PROMPT, candidates=["hello", "hi"])
    assert param is not None
    assert param.value == "hello"
    assert param.read() == "hello"
    # Writing mutates the live object.
    param.write("hi")
    assert obj.prompt == "hi"
    assert param.read() == "hi"
    assert param.metadata["source"] == "attr:prompt"
    assert param.candidates == ["hello", "hi"]


def test_bind_attr_extra_metadata_merged():
    class Obj:
        x = 1

    param = bind_attr(Obj(), "x", "n", ParameterKind.HYPERPARAM, metadata={"extra": "yes"})
    assert param.metadata["extra"] == "yes"
    assert param.metadata["source"] == "attr:x"


def test_bind_attr_present_but_none_value():
    class Obj:
        attr = None

    param = bind_attr(Obj(), "attr", "n", ParameterKind.PROMPT)
    # hasattr is True even when the value is None.
    assert param is not None
    assert param.read() is None


# -- bind_item ----------------------------------------------------------------


def test_bind_item_dict_roundtrip():
    d = {"k": "v"}
    param = bind_item(d, "k", "n", ParameterKind.PROMPT)
    assert param is not None
    assert param.read() == "v"
    param.write("v2")
    assert d["k"] == "v2"
    assert param.metadata["source"] == "item:k"


def test_bind_item_list_roundtrip():
    lst = ["a", "b", "c"]
    param = bind_item(lst, 1, "n", ParameterKind.PROMPT)
    assert param.read() == "b"
    param.write("B")
    assert lst[1] == "B"


def test_bind_item_missing_dict_key_returns_none():
    assert bind_item({}, "missing", "n", ParameterKind.PROMPT) is None


def test_bind_item_bad_list_index_returns_none():
    assert bind_item([], 5, "n", ParameterKind.PROMPT) is None


def test_bind_item_bad_key_type_returns_none():
    # Indexing a list with a string raises TypeError -> None.
    assert bind_item([1, 2], "x", "n", ParameterKind.PROMPT) is None


def test_bind_item_getter_after_key_removed_returns_none():
    d = {"k": "v"}
    param = bind_item(d, "k", "n", ParameterKind.PROMPT)
    del d["k"]
    assert param.read() is None


# -- bind_mapping_key ---------------------------------------------------------


def test_bind_mapping_key_present():
    class Obj:
        def __init__(self):
            self.settings = {"temperature": 0.7}

    obj = Obj()
    param = bind_mapping_key(obj, "settings", "temperature", "obj.temp", ParameterKind.HYPERPARAM)
    assert param is not None
    assert param.read() == 0.7
    param.write(0.2)
    assert obj.settings["temperature"] == 0.2
    assert param.metadata["source"] == "settings[temperature]"


def test_bind_mapping_key_missing_attr_returns_none():
    class Obj:
        pass

    assert bind_mapping_key(Obj(), "settings", "k", "n", ParameterKind.HYPERPARAM) is None


def test_bind_mapping_key_attr_not_dict_returns_none():
    class Obj:
        settings = "not a dict"

    assert bind_mapping_key(Obj(), "settings", "k", "n", ParameterKind.HYPERPARAM) is None


def test_bind_mapping_key_missing_key_returns_none():
    class Obj:
        def __init__(self):
            self.settings = {"other": 1}

    assert bind_mapping_key(Obj(), "settings", "k", "n", ParameterKind.HYPERPARAM) is None


# -- available() loads real framework modules ---------------------------------


def test_ensure_loaded_swallows_bad_module(monkeypatch):
    # A malformed/missing optional module must not break loading the others.
    saved_registry = list(intro._REGISTRY)
    saved_names = set(intro._REGISTERED_NAMES)
    monkeypatch.setattr(
        intro,
        "_FRAMEWORK_MODULES",
        ("adapt_agent.optimization.introspection.__does_not_exist__",),
    )
    monkeypatch.setattr(intro, "_loaded", False)
    try:
        # Should not raise despite the unimportable module name.
        names = available()
        assert isinstance(names, list)
    finally:
        intro._REGISTRY = saved_registry
        intro._REGISTERED_NAMES = saved_names
        intro._loaded = True


def test_available_includes_real_frameworks():
    # No clean_registry fixture: this exercises the real lazy-load path.
    names = available()
    # The package ships several framework introspectors; at least one should
    # have registered on lazy load.
    assert isinstance(names, list)
    assert "crewai" in names
