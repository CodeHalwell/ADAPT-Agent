"""Tests for the PatchManager."""

from adapt_agent.patches import PatchManager


def test_register_and_apply_patch_calls_func_and_sets_applied():
    manager = PatchManager()
    calls = []

    def patch_func(target):
        calls.append(target)

    manager.register_patch(
        patch_id="p1",
        framework="langchain",
        description="A test patch",
        patch_func=patch_func,
    )

    target = object()
    assert manager.is_applied("p1") is False

    result = manager.apply_patch("p1", target)

    assert result is True
    assert calls == [target]
    assert manager.is_applied("p1") is True


def test_apply_patch_unknown_id_returns_false():
    manager = PatchManager()

    assert manager.apply_patch("does-not-exist", object()) is False
    assert manager.is_applied("does-not-exist") is False


def test_apply_patch_second_time_does_not_rerun_func():
    manager = PatchManager()
    calls = []

    manager.register_patch(
        patch_id="p1",
        framework="langchain",
        description="A test patch",
        patch_func=lambda target: calls.append(target),
    )

    target = object()
    assert manager.apply_patch("p1", target) is True
    assert manager.apply_patch("p1", target) is True

    # The patch function ran only once.
    assert len(calls) == 1
    assert manager.is_applied("p1") is True


def test_apply_patch_returns_false_when_func_raises():
    manager = PatchManager()

    def boom(target):
        raise RuntimeError("patch failure")

    manager.register_patch(
        patch_id="bad",
        framework="autogen",
        description="A failing patch",
        patch_func=boom,
    )

    result = manager.apply_patch("bad", object())

    assert result is False
    assert manager.is_applied("bad") is False


def test_list_patches_metadata_applied_flag_and_framework_filter():
    manager = PatchManager()

    manager.register_patch(
        patch_id="p1",
        framework="langchain",
        description="First patch",
        patch_func=lambda target: None,
    )
    manager.register_patch(
        patch_id="p2",
        framework="autogen",
        description="Second patch",
        patch_func=lambda target: None,
    )

    # All patches, none applied yet.
    all_patches = manager.list_patches()
    assert len(all_patches) == 2
    by_id = {p["patch_id"]: p for p in all_patches}
    assert by_id["p1"]["framework"] == "langchain"
    assert by_id["p1"]["description"] == "First patch"
    assert by_id["p1"]["applied"] is False
    assert by_id["p2"]["applied"] is False

    # Apply one and confirm the flag updates.
    assert manager.apply_patch("p1", object()) is True
    by_id = {p["patch_id"]: p for p in manager.list_patches()}
    assert by_id["p1"]["applied"] is True
    assert by_id["p2"]["applied"] is False

    # Filter by framework.
    langchain_patches = manager.list_patches(framework="langchain")
    assert len(langchain_patches) == 1
    assert langchain_patches[0]["patch_id"] == "p1"

    autogen_patches = manager.list_patches(framework="autogen")
    assert len(autogen_patches) == 1
    assert autogen_patches[0]["patch_id"] == "p2"

    assert manager.list_patches(framework="nonexistent") == []
