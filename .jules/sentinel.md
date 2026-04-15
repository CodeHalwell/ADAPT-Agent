## 2025-04-15 - [Critical Firewall Bypass and Fail-Open Vulnerabilities]
**Vulnerability:**
1. Allowlist bypassing: The `check_input` logic in the firewall previously evaluated allowed patterns before blocked patterns, which allowed an attacker to bypass all security blocks if their input matched any allowed pattern.
2. Fail-open filters: The custom filters feature allowed execution to continue even when a custom filter function raised an exception (failing open), ignoring the potential security risk and potentially exposing sensitive information or vulnerabilities in the filter itself.
3. Loose Whitelist: `_allowed_patterns` wasn't enforced as a strict whitelist, meaning that if patterns were provided, a message that didn't match any of them could still slip through and return True.

**Learning:** When developing input-validation and access-control firewalls, checking rules in an incorrect order can negate the effectiveness of blocklists entirely. Additionally, exceptions in security controls must follow a 'fail-closed' paradigm to prevent exploitation during error states. A whitelist should act strictly as a whitelist.

**Prevention:** Always evaluate block rules (denylists) before allow rules (allowlists). Implement strict fail-closed exception handling for custom security filters. If a whitelist is defined, explicitly deny anything that does not match it.
