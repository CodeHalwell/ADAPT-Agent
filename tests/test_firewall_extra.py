"""Additional tests for the Firewall, covering uncovered branches."""

import re

from adapt_agent.security import Firewall


def test_add_blocked_pattern_blocks_and_records_redacted_event():
    """A matching blocked pattern blocks input and records a sanitized snippet."""
    firewall = Firewall()
    firewall.add_blocked_pattern(r"SECRET-\d+")

    secret = "this contains SECRET-12345 inside"
    assert firewall.check_input(secret) is False

    events = firewall.get_security_events()
    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "blocked_input"
    assert event["severity"] == "high"
    assert "blocked pattern" in event["description"].lower()

    # The recorded snippet must be redacted: no raw secret leaked.
    snippet = event["metadata"]["content_snippet"]
    assert "SECRET-12345" not in snippet
    assert "[REDACTED]" in snippet


def test_block_first_allowed_pattern_cannot_override_block():
    """Default mode is block-first: an allowed fullmatch cannot nullify a block."""
    firewall = Firewall()
    firewall.add_blocked_pattern(r"password")
    firewall.add_allowed_pattern(r"password")

    # Block-first default: the blocked pattern wins even on an exact allowed
    # fullmatch. A broad allow pattern must never silently nullify the blocklist.
    assert firewall.check_input("password") is False
    assert firewall.check_input("my password is here") is False

    # The block was recorded as a security event.
    events = firewall.get_security_events()
    assert len(events) == 2
    assert all(e["event_type"] == "blocked_input" for e in events)


def test_allowed_pattern_whitelists_clean_content_in_default_mode():
    """In block-first mode an allowed fullmatch still whitelists content that
    passes every block check (it just cannot override a block)."""
    firewall = Firewall()
    firewall.add_blocked_pattern(r"danger")
    firewall.add_allowed_pattern(r"hello world")

    # Fullmatch of an allowed pattern, no block match -> allowed.
    assert firewall.check_input("hello world") is True


def test_whitelist_mode_preserves_allow_short_circuit():
    """whitelist_mode=True restores the legacy allow-fullmatch short-circuit."""
    firewall = Firewall(whitelist_mode=True)
    firewall.add_blocked_pattern(r"password")
    firewall.add_allowed_pattern(r"password")

    # Strict allowlist: an exact allowed fullmatch short-circuits to allowed,
    # before block checks run.
    assert firewall.check_input("password") is True

    # Merely containing the allowed substring is NOT a fullmatch, so it remains
    # subject to block rules and is blocked.
    assert firewall.check_input("my password is here") is False


def test_custom_filter_returning_true_blocks():
    """A custom filter returning True blocks the input (medium severity)."""
    firewall = Firewall()
    firewall.add_custom_filter(lambda content: "block-me" in content)

    assert firewall.check_input("please block-me now") is False
    assert firewall.check_input("totally fine") is True

    events = firewall.get_security_events()
    assert len(events) == 1
    assert events[0]["severity"] == "medium"
    assert events[0]["description"] == "Content blocked by custom filter"


def test_custom_filter_raising_fails_closed_with_generic_error():
    """A raising custom filter fails closed and stores a generic error message."""
    firewall = Firewall()

    def boom(content: str) -> bool:
        raise RuntimeError("super sensitive internal detail")

    firewall.add_custom_filter(boom)

    assert firewall.check_input("anything") is False

    events = firewall.get_security_events()
    assert len(events) == 1
    event = events[0]
    assert event["severity"] == "high"
    # The error must be generic and must not leak the raised exception text.
    assert event["metadata"]["error"] == "An error occurred"
    assert "super sensitive internal detail" not in str(event)


def test_max_content_length_blocks_oversize_input():
    """Input exceeding max_content_length is blocked with a high-severity event."""
    firewall = Firewall(max_content_length=10)

    assert firewall.check_input("short") is True
    assert firewall.check_input("this is way too long") is False

    events = firewall.get_security_events()
    assert len(events) == 1
    event = events[0]
    assert event["severity"] == "high"
    assert "exceeds maximum" in event["description"]
    assert event["metadata"]["length"] == len("this is way too long")

    stats = firewall.get_stats()
    assert stats["total_blocked"] == 1


def test_sanitize_redacts_blocked_patterns():
    """sanitize replaces every blocked pattern occurrence."""
    firewall = Firewall()
    firewall.add_blocked_pattern(r"\bAPIKEY-\w+\b")

    content = "key1 APIKEY-abc and key2 APIKEY-def"
    sanitized = firewall.sanitize(content)
    assert "APIKEY-abc" not in sanitized
    assert "APIKEY-def" not in sanitized
    assert sanitized.count("[REDACTED]") == 2

    # Custom replacement string is honoured.
    assert firewall.sanitize("APIKEY-xyz", replacement="<X>") == "<X>"


def test_check_output_applies_block_logic_with_output_event_type():
    """check_output applies the same blocking logic but labels events as output."""
    firewall = Firewall()
    firewall.add_blocked_pattern(r"forbidden")

    assert firewall.check_output("forbidden content") is False
    assert firewall.check_output("fine content") is True

    events = firewall.get_security_events()
    assert len(events) == 1
    # Output checks must be labeled distinctly, not as blocked_input.
    assert events[0]["event_type"] == "blocked_output"


def test_add_pattern_rejects_catastrophic_regex():
    """Obviously catastrophic (nested-quantifier) patterns are rejected."""
    import pytest

    firewall = Firewall()
    with pytest.raises(ValueError):
        firewall.add_blocked_pattern(r"(a+)+")
    with pytest.raises(ValueError):
        firewall.add_allowed_pattern(r"(a*)*")


def test_redacted_snippet_does_not_leak_raw_content():
    """Length-cap events store a hashed/masked snippet, not raw content."""
    firewall = Firewall(max_content_length=10)
    secret = "TOPSECRETpayloadvalue123"
    assert firewall.check_input(secret) is False

    snippet = firewall.get_security_events()[0]["metadata"]["content_snippet"]
    assert secret not in snippet
    assert "sha256:" in snippet


def test_check_message_delegates_to_check_input():
    """check_message checks the message content field."""
    firewall = Firewall()
    firewall.add_blocked_pattern(r"forbidden")

    blocked_message = {"role": "user", "content": "forbidden text"}
    allowed_message = {"role": "user", "content": "hello"}

    assert firewall.check_message(blocked_message) is False
    assert firewall.check_message(allowed_message) is True


def test_get_security_events_severity_filter_and_limit():
    """get_security_events filters by severity and applies a limit."""
    firewall = Firewall()
    firewall.add_blocked_pattern(r"high")  # high severity
    firewall.add_custom_filter(lambda c: "med" in c)  # medium severity

    firewall.check_input("high one")
    firewall.check_input("med one")
    firewall.check_input("high two")
    firewall.check_input("med two")

    all_events = firewall.get_security_events()
    assert len(all_events) == 4

    high_events = firewall.get_security_events(severity="high")
    assert len(high_events) == 2
    assert all(e["severity"] == "high" for e in high_events)

    # Severity + limit returns the most recent matching events.
    high_limited = firewall.get_security_events(severity="high", limit=1)
    assert len(high_limited) == 1
    assert high_limited[0]["severity"] == "high"

    # Limit alone returns most recent events.
    limited = firewall.get_security_events(limit=2)
    assert len(limited) == 2
    assert limited == all_events[-2:]


def test_max_events_bounding():
    """The security event store is bounded by max_events."""
    firewall = Firewall(max_events=3)
    firewall.add_blocked_pattern(r"bad")

    for i in range(10):
        firewall.check_input(f"bad {i}")

    events = firewall.get_security_events()
    assert len(events) == 3
    # Oldest events were evicted; the newest survive. The blocked pattern
    # itself is redacted in the snippet, but the trailing index remains.
    assert events[-1]["metadata"]["content_snippet"] == "[REDACTED] 9"


def test_get_stats_counts():
    """get_stats reports counts for patterns, filters, and blocks."""
    firewall = Firewall()
    firewall.add_blocked_pattern(r"a")
    firewall.add_blocked_pattern(r"b")
    firewall.add_allowed_pattern(r"zzz")
    firewall.add_custom_filter(lambda c: False)

    firewall.check_input("contains a")

    stats = firewall.get_stats()
    assert stats["blocked_patterns"] == 2
    assert stats["allowed_patterns"] == 1
    assert stats["custom_filters"] == 1
    assert stats["total_blocked"] == 1
    assert stats["security_events"] == 1


def test_blocked_pattern_with_ignorecase_flag():
    """Flags passed to add_blocked_pattern are applied to the compiled pattern."""
    firewall = Firewall()
    firewall.add_blocked_pattern(r"secret", flags=re.IGNORECASE)

    assert firewall.check_input("SECRET value") is False
