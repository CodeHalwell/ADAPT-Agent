"""Example 04 (Claude Agent SDK): a multi-agent system, governed and trained.

The Claude Agent SDK has no built-in multi-agent primitive -- you compose one by
running several configured ``query(prompt, options=...)`` passes in sequence and
wiring their outputs together. Here we build a classic **researcher -> writer ->
reviewer** pipeline, each "agent" being its own ``ClaudeAgentOptions`` object,
and an ``orchestrator`` function that drives all three as one unit.

That whole orchestrator can be:

1. **Guarded as ONE unit** -- wrap the orchestrator callable with
   ``ClaudeAgentSDKAdapter`` so a single ``execute`` screens input/output of the
   entire pipeline (shown briefly at the end).
2. **Optimized as ONE system** -- register all three options objects as
   ``components`` of an ``OptimizableAgent`` and let ``make_default_optimizer``
   tune every agent's prompt, model, hyperparameters and *tool/skill allow-list*
   (drop-one ablation), while the judge proposes brand-new tools/skills on
   ``result.recommendations``.

A parallel **YAML config path** (``run_training("claude_agent.train.yaml")``)
does the same thing declaratively; we exercise an in-memory equivalent here so
the example runs with NO API key and NO network (deterministic judge stub).

Run it with:

    python examples/claude_agent/04_multi_agent_and_training.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    import claude_agent_sdk  # noqa: F401
except ImportError:
    claude_agent_sdk = None  # type: ignore[assignment]

from adapt_agent import Firewall
from adapt_agent.adapters import ClaudeAgentSDKAdapter
from adapt_agent.optimization import (
    EvaluationHarness,
    GoldenDataset,
    LLMJudge,
    OptimizableAgent,
    ParameterKind,
    make_default_optimizer,
    token_f1,
)
from adapt_agent.optimization.config import parse_training_config, run_training

_OPUS = "claude-opus-4-8"
_HAIKU = "claude-haiku-4-5"


@dataclass
class FakeOptions:
    """Stands in for ``claude_agent_sdk.ClaudeAgentOptions`` (one per sub-agent)."""

    system_prompt: str = "Help."
    model: str = _HAIKU
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    max_turns: int = 1
    permission_mode: str = "default"


# --- Three configured agents (each a ClaudeAgentOptions-shaped object) ------ #
RESEARCHER = FakeOptions(
    system_prompt="Find facts.",
    # A distracting tool the optimizer can learn to drop via ablation.
    allowed_tools=["web_search", "scratchpad"],
)
WRITER = FakeOptions(system_prompt="Write something.")
REVIEWER = FakeOptions(system_prompt="Check it.")

_FACTS = {
    "France": "Paris is the capital of France.",
    "Japan": "Tokyo is the capital of Japan.",
    "Italy": "Rome is the capital of Italy.",
}


def _ask(options: FakeOptions, role: str, payload: str) -> str:
    """Deterministic stand-in for one ``query(prompt, options=options)`` pass.

    Each sub-agent does its best work only under a *precise* system prompt; the
    "scratchpad" tool is pure noise that hurts the researcher until dropped.
    Real usage replaces this with an actual SDK call per agent.
    """
    precise = any(k in options.system_prompt.lower() for k in ("only", "concise", "exactly"))
    if role == "researcher":
        country = payload
        fact = _FACTS.get(country, f"{country} is a country.")
        noisy = "scratchpad" in options.allowed_tools
        return fact + (" (unverified notes attached)" if noisy and not precise else "")
    if role == "writer":
        # Turn the research note into a one-line answer.
        first = payload.split(".")[0]
        return first.split(" is ")[0] if precise else f"In summary, {first.lower()}."
    # reviewer: pass the writer's text through, trimming noise when precise.
    return payload.strip() if precise else payload


def orchestrator(prompt: str) -> str:
    """Drive researcher -> writer -> reviewer as one governed/optimizable unit.

    The parameter is named ``prompt`` so the same callable works both as the
    OptimizableAgent runner (called positionally) and as a wrapped agent for
    ``ClaudeAgentSDKAdapter`` (which calls ``runner(prompt=...)``, mirroring the
    real SDK's ``query(prompt=...)``).
    """
    country = prompt.replace("What is the capital of", "").strip(" ?")
    research = _ask(RESEARCHER, "researcher", country)
    draft = _ask(WRITER, "writer", research)
    final = _ask(REVIEWER, "reviewer", draft)
    return final


def deterministic_judge_stub(prompt: str, system: str | None = None) -> str:
    """Offline judge: rewrites prompts on request, else grades the response.

    Branches on the ``system`` instruction (rewrite vs. grade) exactly like the
    real ``LLMJudge`` prompt shapes. In *adversarial* mode the same callable is
    asked to propose NEW tools/skills -- we answer that too, so
    ``result.recommendations`` is populated offline.
    """
    sys = system or ""
    if "Rewrite the instruction" in sys:
        return "Respond with ONLY the exact capital city name, concise, nothing else."
    if "propose" in sys.lower() and "tools" in sys.lower():
        # Shape expected by the judge's suggest_tools parser.
        return (
            '{"tools": [{"name": "fact_verifier", "description": "Cross-checks a '
            'claimed fact against a trusted source", "rationale": "reduces '
            'unverified notes"}]}'
        )
    answer = ""
    if "<response>" in prompt and "</response>" in prompt:
        answer = prompt.split("<response>", 1)[1].split("</response>", 1)[0].strip()
    # Reward a clean single-word capital.
    score = 9 if answer and " " not in answer else 3
    return f'{{"score": {score}, "pass": {str(score >= 6).lower()}, "reasoning": "auto"}}'


def build_dataset() -> GoldenDataset:
    return GoldenDataset.from_list(
        [
            {"input": "What is the capital of France?", "expected": "Paris"},
            {"input": "What is the capital of Japan?", "expected": "Tokyo"},
            {"input": "What is the capital of Italy?", "expected": "Rome"},
        ]
    )


def optimize_in_code() -> None:
    """Path A: optimize the whole multi-agent system via the Python API."""
    print("================ Path A: optimize via the Python API ================")
    data = build_dataset()

    # Register all three agents as components so EVERY agent's knobs are tuned.
    # `runner=orchestrator` drives the whole pipeline for each evaluation.
    agent = OptimizableAgent.from_components(
        components={"researcher": RESEARCHER, "writer": WRITER, "reviewer": REVIEWER},
        runner=orchestrator,
        name="claude-research-team",
    )

    # Introspection ALREADY made the researcher's `allowed_tools` a drop-one
    # ablation search space, so we don't redeclare it (that would bind the same
    # list twice). To show declaring a knob the framework does NOT expose, add a
    # higher-level SKILL allow-list we maintain ourselves -- candidates are the
    # full set first, then each drop-one subset.
    skills = {"researcher": ["web_search", "summarize"]}
    agent.add_tool_parameter(
        "researcher.skills",
        kind=ParameterKind.SKILL,
        getter=lambda: skills["researcher"],
        setter=lambda v: skills.__setitem__("researcher", v),
        candidate_tools=["web_search", "summarize"],
    )

    print("Parameters in the search space:")
    for p in agent.parameters:
        print(f"  - {p.name:26} kind={p.kind.value}")

    # Adversarial judge: grades like a harsh critic AND proposes new tools.
    judge = LLMJudge(deterministic_judge_stub, adversarial=True)
    harness = EvaluationHarness(
        metrics=[token_f1(), judge.as_metric("quality")],
        primary_metric="quality",
    )

    print("\nBaseline:", harness.evaluate(agent, data))
    print("Baseline answer for France:", orchestrator("What is the capital of France?"))

    # The full pipeline: few-shot -> prompts -> models/hparams -> tools/skills,
    # with judge-driven new-tool suggestions enabled by `suggest_tools=True`.
    optimizer = make_default_optimizer(
        harness, judge=judge, max_evals=40, seed=0, suggest_tools=True
    )
    result = optimizer.optimize(agent, data)

    print("\nBest:", harness.evaluate(agent, data))
    print("Improvement (primary):", result.improvement)
    print("Best config:")
    for name, value in result.best_config.items():
        print(f"  {name} = {value!r}")
    print("Answer for France now:", orchestrator("What is the capital of France?"))

    if result.recommendations:
        print("\nJudge recommendations (advisory new tools/skills):")
        for tip in result.recommendations:
            print("  -", tip)


def train_from_config() -> None:
    """Path B: the SAME optimization driven by a declarative training config.

    This mirrors ``run_training("claude_agent.train.yaml")`` but builds the config
    in-process and points it at this module (``__main__``) with a temp dataset, so
    it runs offline with no API key. The bundled ``claude_agent.train.yaml`` is the
    real-world version (it names a file dataset and the ``anthropic`` judge
    provider); to run it for real::

        from adapt_agent.optimization.config import run_training
        result = run_training("examples/claude_agent/claude_agent.train.yaml")

    Here we keep the judge offline by selecting metric-driven optimization
    (``primary_metric: token_f1``) and tuning the writer prompt over explicit
    candidates -- a deterministic signal that needs no model.
    """
    import json
    import tempfile
    from pathlib import Path

    print("\n============ Path B: optimize via a training config ============")
    # Reset the writer prompt so we can watch the config path tune it from scratch.
    WRITER.system_prompt = "Write something."

    # The config dataset is a file (any of jsonl/json/csv); write a temp one.
    tmp = Path(tempfile.mkdtemp())
    data_path = tmp / "golden.jsonl"
    data_path.write_text("\n".join(json.dumps(r) for r in _DATASET_ROWS), encoding="utf-8")

    config = parse_training_config(
        {
            "target": {
                "entrypoint": "__main__:orchestrator",  # callable input -> output
                "components": {  # the live sub-agents whose knobs get introspected
                    "researcher": "__main__:RESEARCHER",
                    "writer": "__main__:WRITER",
                    "reviewer": "__main__:REVIEWER",
                },
            },
            "dataset": {"path": str(data_path), "format": "jsonl"},
            # Judge omitted here so the run stays fully offline; token_f1 drives
            # the search. The bundled YAML shows the adversarial-judge variant.
            "metrics": ["token_f1", "exact_match"],
            "primary_metric": "token_f1",
            "optimizer": {"type": "default", "max_evals": 40, "seed": 0},
            # A knob no framework auto-exposes: the writer's prompt as candidates.
            "parameters": [
                {
                    "name": "writer.system_prompt",
                    "kind": "prompt",
                    "component": "writer",
                    "attr": "system_prompt",
                    "candidates": [
                        "Write something.",
                        "Respond with ONLY the exact capital city name, concise.",
                    ],
                }
            ],
        }
    )

    print("Baseline answer for France:", orchestrator("What is the capital of France?"))
    result = run_training(config)
    print("Improvement (primary):", result.improvement)
    print("Final writer prompt applied in place:", repr(WRITER.system_prompt))
    print("Answer for France now:", orchestrator("What is the capital of France?"))


def guard_the_whole_system() -> None:
    """Bonus: guard the entire orchestrator as a single governed unit."""
    print("\n============ Bonus: guard the orchestrator as ONE unit ============")
    firewall = Firewall(max_content_length=10_000)
    firewall.add_blocked_pattern(r"(?i)ignore[\w ]*?instructions")
    adapter = ClaudeAgentSDKAdapter(firewall=firewall, agent_id="claude-team")
    # The orchestrator takes a prompt string; the adapter derives it from the
    # payload's latest user message and screens the pipeline's output.
    guarded = adapter.wrap_agent(orchestrator)
    out = guarded.execute(
        {"messages": [{"role": "user", "content": "What is the capital of Italy?"}]}
    )
    print("Guarded pipeline output:", out)


# Inline dataset rows reused by the config path (same as build_dataset()).
_DATASET_ROWS = [
    {"input": "What is the capital of France?", "expected": "Paris"},
    {"input": "What is the capital of Japan?", "expected": "Tokyo"},
    {"input": "What is the capital of Italy?", "expected": "Rome"},
]


def main() -> None:
    optimize_in_code()
    train_from_config()
    guard_the_whole_system()


if __name__ == "__main__":
    main()
