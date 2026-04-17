import pytest
from adapt_agent.security.firewall import Firewall

def test_firewall_bypass():
    firewall = Firewall()
    firewall.add_blocked_pattern(r"malicious")
    firewall.add_allowed_pattern(r"safe")

    assert firewall.check_input("safe malicious") == False
    assert firewall.check_output("safe malicious") == False

def test_firewall_fail_closed():
    firewall = Firewall()
    def broken_filter(x):
        raise ValueError("broken")
    firewall.add_custom_filter(broken_filter)

    assert firewall.check_input("test") == False
    assert firewall.check_output("test") == False

def test_firewall_strict_whitelist():
    firewall = Firewall()
    firewall.add_allowed_pattern(r"only_this")

    assert firewall.check_input("something else") == False
    assert firewall.check_output("something else") == False

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", "tests/test_firewall_bypass.py"]))
