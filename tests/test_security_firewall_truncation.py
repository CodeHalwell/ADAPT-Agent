import pytest
from adapt_agent.security.firewall import Firewall

def test_truncate_escaped_legitimate_backslash():
    text = "ab\\n"
    truncated = Firewall._truncate_escaped(text, 3)
    assert truncated == "ab\\"

def test_truncate_escaped_newline():
    text = "a\nb"
    truncated = Firewall._truncate_escaped(text, 3)
    assert truncated == "a\\n"

def test_truncate_escaped_long():
    text = "a\n" * 10
    truncated = Firewall._truncate_escaped(text, 5)
    assert truncated == "a\\na"

def test_truncate_escaped_ending_with_backslash_then_newline():
    text = "a\\\n"
    truncated = Firewall._truncate_escaped(text, 3)
    assert truncated == "a\\"
