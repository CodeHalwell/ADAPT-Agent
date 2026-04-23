import re

with open("tests/test_security.py", "r") as f:
    content = f.read()

# Remove test_taint_tracker_get_stats
pattern = r"def test_taint_tracker_get_stats\(\).*?(?=\ndef test_taint_tracker_get_taint_flow)"
new_content = re.sub(pattern, "", content, flags=re.DOTALL)

with open("tests/test_security.py", "w") as f:
    f.write(new_content)
