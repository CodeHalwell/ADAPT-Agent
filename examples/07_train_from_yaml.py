"""Example 07: "Train" an agent from a declarative config (offline, no network).

This mirrors example 06 but drives the whole optimization run from a single
:class:`~adapt_agent.optimization.config.TrainingConfig` -- the same structure you
would normally write in a ``train.yaml`` file and run with::

    adapt-agent train train.yaml

Here we build the config in-process and point it at agent code defined in this
very module (``__main__``), so the example runs with no API key and no network.
Swap the ``judge`` block for a real provider (``anthropic`` / ``openai`` / ...)
and point ``dataset.path`` at your golden data to train for real. The judge can be
flipped into an *adversary* with ``adversarial: true`` -- it then grades like a
harsh critic and proposes new tools/skills via ``optimizer.suggest_tools``.

Run it with:

    python examples/07_train_from_yaml.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from adapt_agent.optimization.config import parse_training_config, run_training

# --- agent code "spread across the project" -------------------------------- #
# A trivial orchestrator whose behaviour depends on a tunable prefix knob.


class Cfg:
    """Holds the live, tunable knob the optimizer will rewrite in place."""

    prefix = ""


cfg = Cfg()


def run(question: str) -> str:
    """The system entrypoint (could route to many sub-agents)."""
    return f"{cfg.prefix}{question}"


def main() -> None:
    # 1. A golden dataset on disk (any of jsonl/json/csv).
    rows = [
        {"input": "France", "expected": "ANSWER:France"},
        {"input": "Japan", "expected": "ANSWER:Japan"},
        {"input": "Italy", "expected": "ANSWER:Italy"},
    ]
    tmp = Path(tempfile.mkdtemp())
    data_path = tmp / "golden.jsonl"
    data_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    # 2. The training config (this is exactly what a YAML file encodes).
    config = parse_training_config(
        {
            "target": {
                "entrypoint": "__main__:run",  # callable input -> output
                "components": {"cfg": "__main__:cfg"},  # live knobs live here
            },
            "dataset": {"path": str(data_path), "format": "jsonl"},
            "metrics": ["exact_match"],
            "optimizer": {"type": "coordinate_ascent", "max_evals": 10, "seed": 0},
            # No framework exposes this knob, so we declare it explicitly. A
            # temperature bound that exceeds the provider's range would be clamped
            # with a warning rather than crashing the run.
            "parameters": [
                {
                    "name": "cfg.prefix",
                    "kind": "prompt",
                    "component": "cfg",
                    "attr": "prefix",
                    "candidates": ["", "ANSWER:"],
                }
            ],
        }
    )

    print("Baseline answer for France:", run("France"))
    result = run_training(config)

    print("\nResult:", result)
    print("Best config:", result.best_config)
    print("Final prefix applied in place:", repr(cfg.prefix))
    print("Answer for France now:", run("France"))
    if result.recommendations:
        print("\nJudge recommendations (advisory):")
        for tip in result.recommendations:
            print("  -", tip)


if __name__ == "__main__":
    main()
