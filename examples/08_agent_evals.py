"""Example 08: One-call agent evals across frameworks (offline).

The eval loop everybody needs: *"here is my agent, here is a golden dataset of
inputs and expected outputs -- score it."* ADAPT-Agent's ``evaluate_agent`` runs
that loop against **any** supported framework:

* deterministic checks -- exact/contained text, regex, token-F1, a **number
  within tolerance**, JSON subset, edit distance;
* **per-row checks** -- each dataset row can declare *how* it is scored via a
  ``check`` field (``"exact_match"`` here, ``"numeric_close"`` there, an
  LLM-judge elsewhere);
* **LLM-as-judge** -- model-graded scoring for open-ended rows, provider-
  agnostic (Claude / OpenAI / Gemini / ... or any callable);
* **framework-native output extraction** -- a Pydantic AI ``AgentRunResult``, a
  LangGraph state dict, a Microsoft Agent Framework ``AgentRunResponse``, or a
  Google ADK event stream is unwrapped to the final response text
  automatically, so metrics compare answers, not ``repr()``s.

This example runs with **no framework installed and no API key**: each agent is
a small stub with the exact shape (methods + result objects) of the real
framework, which is all the duck-typed eval stack looks at. Swap in your real
agent object and the same calls work unchanged:

    from adapt_agent.evaluation import evaluate_agent
    report = evaluate_agent(my_agent, "golden.jsonl", judge="claude")

Run it with:

    python examples/08_agent_evals.py
"""

from __future__ import annotations

import json

from adapt_agent.evaluation import adk_runner, evaluate_agent

# --------------------------------------------------------------------------- #
# A tiny "model" shared by every stub agent below.
# --------------------------------------------------------------------------- #

_ANSWERS = {
    "What is the capital of France?": "Paris",
    "What is 6 * 7?": "The answer is 42.",
    "Give me one word for a happy feeling.": "joy",
}


def _answer(question: str) -> str:
    return _ANSWERS.get(question, "I am not sure.")


# --------------------------------------------------------------------------- #
# Framework-shaped stub agents. Each mirrors the *shape* of the real framework
# object (same run method, same result type), which is all the eval stack sees.
# --------------------------------------------------------------------------- #


class PydanticAIAgent:
    """Like ``pydantic_ai.Agent``: ``run_sync(q)`` returns an ``AgentRunResult``."""

    class AgentRunResult:
        def __init__(self, output: str):
            self.output = output

        def all_messages(self):
            return []

    def run_sync(self, question: str):
        return self.AgentRunResult(_answer(question))


class MicrosoftChatAgent:
    """Like ``agent_framework.ChatAgent``: async ``run(q)`` -> ``AgentRunResponse``."""

    class ChatMessage:
        def __init__(self, text: str):
            self.role, self.text, self.contents = "assistant", text, []

    class AgentRunResponse:
        def __init__(self, text: str):
            self.messages = [MicrosoftChatAgent.ChatMessage(text)]
            self.text = text

    async def run(self, question: str):
        return self.AgentRunResponse(_answer(question))


class LangGraphGraph:
    """Like a compiled LangGraph graph: ``invoke(state) -> state``.

    ``evaluate_agent`` detects the LangGraph shape and wraps each plain-string
    dataset input into ``{"messages": [{"role": "user", "content": ...}]}``
    automatically, then reads the final message back out of the state.
    """

    def __init__(self):
        self.nodes = {}

    def invoke(self, state: dict) -> dict:
        question = state["messages"][-1]["content"]
        reply = {"role": "assistant", "content": _answer(question)}
        return {"messages": [*state["messages"], reply]}


class ADKRunner:
    """Like ``google.adk.Runner``: ``run(user_id=, session_id=, new_message=)``
    yields events whose ``content.parts[*].text`` carry the response."""

    class _Part:
        def __init__(self, text=None):
            self.text = text

    class _Content:
        def __init__(self, parts):
            self.parts, self.role = parts, "model"

    class _Event:
        def __init__(self, text):
            self.content = ADKRunner._Content([ADKRunner._Part(text)])

    class _Sessions:
        def create_session(self, *, app_name, user_id, session_id):
            return object()

    def __init__(self):
        self.app_name = "eval-app"
        self.session_service = self._Sessions()

    def run(self, *, user_id, session_id, new_message):
        yield self._Event(None)  # e.g. a tool-call event with no text
        yield self._Event(_answer(new_message))


# --------------------------------------------------------------------------- #
# The golden dataset: each row can declare HOW it is checked.
# --------------------------------------------------------------------------- #

DATASET = [
    # Exact text match (the default check when a row declares none).
    {"input": "What is the capital of France?", "expected": "Paris"},
    # Number match: passes although the agent replies "The answer is 42."
    {"input": "What is 6 * 7?", "expected": 42, "check": "numeric_close"},
    # Parameterised check: any number within +/- 0.5 of 42 would also pass.
    {
        "input": "What is 6 * 7?",
        "expected": 42,
        "check": {"name": "numeric_close", "tolerance": 0.5},
    },
    # Open-ended row: graded by the LLM judge against per-row criteria.
    {
        "input": "Give me one word for a happy feeling.",
        "check": "judge",
        "criteria": "Response must be a single positive-emotion word.",
    },
]


def offline_judge(prompt: str, system: str | None = None) -> str:
    """A deterministic stand-in for a real LLM judge (no network, no key).

    Real usage: pass ``judge="claude"`` / ``judge="openai"`` / ... or an
    ``LLMJudge`` instance instead of this function.
    """
    response = ""
    if "<response>" in prompt:
        response = prompt.split("<response>")[1].split("</response>")[0].strip()
    score = 9 if (response and len(response.split()) == 1) else 2
    return json.dumps({"score": score, "pass": score >= 6, "reasoning": "offline stub"})


# --------------------------------------------------------------------------- #
# Run the same eval against every framework.
# --------------------------------------------------------------------------- #


def main() -> None:
    agents = {
        "Pydantic AI": PydanticAIAgent(),
        "Microsoft Agent Framework": MicrosoftChatAgent(),
        "LangGraph": LangGraphGraph(),
        # Google ADK agents run inside a Runner; adk_runner() handles sessions
        # and message packing. (message_factory keeps this stub offline -- with
        # the real SDK installed you simply omit it.)
        "Google ADK": adk_runner(ADKRunner(), message_factory=lambda s: s),
    }

    for name, agent in agents.items():
        report = evaluate_agent(agent, DATASET, judge=offline_judge, metrics="checks")
        print(f"{name:28} -> {report}")

    # Dig into one report: aggregate scores, per-example results, failures.
    # failure_threshold=0.6 treats a judge grade >= 0.6 as passing (the default
    # of 1.0 would flag every imperfect judge score as a failure).
    report = evaluate_agent(
        PydanticAIAgent(),
        DATASET,
        judge=offline_judge,
        metrics="checks",
        failure_threshold=0.6,
    )
    print("\nAggregate:", report.aggregate)
    for result in report.results:
        print(f"  #{result.index}: scores={result.scores} output={result.output!r}")
    print("Failures:", [f.inputs for f in report.failures()] or "none")

    # Judge-only evals (no reference answers needed): grade every row.
    judged = evaluate_agent(
        PydanticAIAgent(),
        [{"input": "Give me one word for a happy feeling."}],
        judge=offline_judge,
    )
    print("\nJudge-only:", judged)

    # The same eval is available from the CLI:
    #   adapt-agent evaluate myapp.agents:agent --data golden.jsonl \
    #       --metric checks --judge claude --extract-output


if __name__ == "__main__":
    main()
