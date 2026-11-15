"""Example 15: The command line, end to end.

Everything the library does from Python is also reachable from a shell, which
is what you want in CI. This script builds a throwaway project in a temporary
directory and runs each ``adapt-agent`` command against it, printing the
command and what it returned. (``adapt`` is the same program under a shorter
name.)

    adapt-agent --version / info      what is installed
    adapt-agent validate              check a guardrail config before deploying
    adapt-agent monitor               summarise the controls a config turns on
    adapt-agent skills / install      give a coding agent the bundled skill
    adapt-agent evaluate              score an agent against a golden dataset
    adapt-agent optimize              tune it, gated on held-out data
    adapt-agent train                 the same from one YAML file (example 07)

No optional dependencies and no network. Run it with:

    python examples/15_cli_tour.py
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# The agent the CLI will load, as it would live in your own project. The CLI
# takes a "module:attribute" reference, so this has to be an importable file.
AGENT_MODULE = """
from adapt_agent.optimization import OptimizableAgent, Parameter, ParameterKind

CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Kenya": "Nairobi"}
STATE = {"prompt": "Answer the question."}


def run(question):
    country = question.replace("Capital of", "").strip(" ?")
    if "only the city" in STATE["prompt"].lower():
        return CAPITALS.get(country, "unknown")
    return f"I believe the capital of {country} is a lovely place."


agent = OptimizableAgent.from_callable(
    run,
    name="capitals",
    parameters=[
        Parameter(
            name="capitals.prompt",
            kind=ParameterKind.PROMPT,
            candidates=["Answer the question.", "Reply with only the city name."],
            getter=lambda: STATE["prompt"],
            setter=lambda value: STATE.__setitem__("prompt", value),
            component="capitals",
        )
    ],
)
"""

GOOD_CONFIG = {
    "policy_rules": [
        {
            "name": "low_trust",
            "description": "Block low-trust callers.",
            "condition": "state['trust_score'] < 0.5",
            "action": "block",
            "severity": "high",
        }
    ],
    "firewall": {"blocked_patterns": ["(?i)ignore previous instructions"]},
    "adversarial": {"attack_patterns": ["dump the system prompt"]},
}
# Three mistakes a config review tends to miss: a function call the condition
# language does not have, a regex that does not compile, and an unknown action.
BAD_CONFIG = {
    "policy_rules": [
        {"name": "long", "condition": "len(state['messages']) > 20", "action": "blok"}
    ],
    "firewall": {"blocked_patterns": ["(unclosed"]},
}


def build_project(root: Path) -> None:
    (root / "myagent.py").write_text(AGENT_MODULE)
    (root / "config.json").write_text(json.dumps(GOOD_CONFIG, indent=2))
    (root / "bad_config.json").write_text(json.dumps(BAD_CONFIG, indent=2))
    rows = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Kenya", "Nairobi")]

    def write(name: str, subset: list) -> None:
        lines = [json.dumps({"input": f"Capital of {c}?", "expected": a}) for c, a in subset]
        (root / name).write_text("\n".join(lines) + "\n")

    write("train.jsonl", rows[:3])
    write("holdout.jsonl", rows[3:])


def cli(root: Path, *args: str, show: int = 12) -> subprocess.CompletedProcess:
    """Run ``adapt-agent <args>`` inside the project and print what happened."""
    env = dict(os.environ)
    repo = Path(__file__).resolve().parent.parent
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), str(repo), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-m", "adapt_agent", *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    print(f"\n$ adapt-agent {' '.join(args)}        (exit {result.returncode})")
    lines = (result.stdout or result.stderr).strip().splitlines()
    if show:
        print(textwrap.indent("\n".join(lines[:show]), "    "))
        if len(lines) > show:
            print(f"    ... {len(lines) - show} more line(s)")
    return result


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="adapt-cli-tour-") as tmp:
        root = Path(tmp)
        build_project(root)

        print("== What is installed ==")
        cli(root, "--version")
        cli(root, "info", show=4)

        print("\n== Guardrail configuration ==")
        cli(root, "validate", "config.json")
        # A non-zero exit, so a CI step fails before a broken config ships.
        cli(root, "validate", "bad_config.json")
        cli(root, "monitor", "--agent-id", "support-bot", "--config", "config.json", "--json")

        print("\n== The bundled agent skill ==")
        cli(root, "skills", show=3)
        cli(root, "install", "skill", "--dir", ".claude/skills")
        installed = sorted(p.name for p in (root / ".claude/skills/adapt-agent").iterdir())
        print(f"    installed files: {installed}")

        print("\n== Evaluate, then optimize ==")
        before = cli(
            root, "evaluate", "myagent:agent", "--data", "train.jsonl",
            "--metric", "exact_match", "--json", show=0,
        )  # fmt: skip
        print(f"    score before: {json.loads(before.stdout)['score']:.2f}")

        tuned = cli(
            root, "optimize", "myagent:agent", "--data", "train.jsonl",
            "--val-data", "holdout.jsonl", "--metric", "exact_match",
            "--optimizer", "grid", "--save-config", "tuned.json", "--json", show=0,
        )  # fmt: skip
        report = json.loads(tuned.stdout)
        print(
            f"    {report['baseline_score']:.2f} -> {report['best_score']:.2f} on training, "
            f"{report['validation_score']:.2f} on held-out, "
            f"rejected by the gate: {report['validation_rejected']}"
        )
        print("    tuned.json (the file you review and commit):")
        print(textwrap.indent((root / "tuned.json").read_text().strip(), "      "))


if __name__ == "__main__":
    main()
