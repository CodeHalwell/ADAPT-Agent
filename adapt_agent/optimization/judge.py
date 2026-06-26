"""LLM-as-judge: model-graded evaluation and prompt improvement.

The judge is the connective tissue of the optimization subsystem. It is used at
*every* stage:

* **Evaluation** -- :meth:`LLMJudge.score` grades an agent output against a
  reference and/or a rubric, producing a normalized ``[0, 1]`` score that the
  :class:`~adapt_agent.optimization.evaluation.EvaluationHarness` can treat like
  any other metric (see :meth:`LLMJudge.as_metric`).
* **Selection** -- :meth:`LLMJudge.compare` does pairwise preference judging,
  which optimizers use to choose between two candidate outputs without a
  reference answer.
* **Improvement** -- :meth:`LLMJudge.critique` explains *why* an output is weak,
  and :meth:`LLMJudge.improve_prompt` rewrites an instruction given a batch of
  judged failures. The LLM proposer
  (:mod:`adapt_agent.optimization.proposers`) is built on these.

To keep ``adapt_agent`` dependency-free and offline-testable, the judge never
imports an LLM SDK. You supply a ``complete`` callable -- ``Callable[[str], str]``
-- that maps a prompt to a completion. Wrap whatever provider you like (Anthropic,
OpenAI, a local model, or a deterministic stub in tests).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from adapt_agent.optimization.providers import ModelProvider

#: A provider-agnostic text completion function: prompt in, completion out.
CompletionFn = Callable[[str], str]


@dataclass
class JudgeVerdict:
    """The result of a single grading call.

    Args:
        score: Quality in ``[0, 1]`` (the raw model rating, normalized).
        passed: Whether the output met the pass threshold.
        reasoning: The judge's natural-language justification.
        raw: The raw completion text (for auditing / debugging).
        metadata: Extra parsed fields (e.g. per-criterion sub-scores).
    """

    score: float
    passed: bool
    reasoning: str = ""
    raw: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


_DEFAULT_RUBRIC = (
    "Grade how well the RESPONSE answers the INPUT. Reward correctness, "
    "completeness, faithfulness to any reference, and clarity. Penalize "
    "factual errors, hallucination, and irrelevance."
)

_SCORE_PROMPT = """\
You are a meticulous evaluation judge. Grade a single agent response.

INPUT:
{input}

RESPONSE:
{output}
{reference_block}{criteria_block}
{rubric}

Return ONLY a JSON object on a single line with this exact shape:
{{"score": <integer 0-10>, "pass": <true|false>, "reasoning": "<one short sentence>"}}
Where 0 means completely wrong/unusable and 10 means an ideal response.
"""

_COMPARE_PROMPT = """\
You are a meticulous evaluation judge performing a pairwise comparison.

INPUT:
{input}

RESPONSE A:
{output_a}

RESPONSE B:
{output_b}
{criteria_block}
Decide which response is better at addressing the INPUT. If they are equally
good, answer "tie".

Return ONLY a JSON object on a single line:
{{"winner": "A"|"B"|"tie", "reasoning": "<one short sentence>"}}
"""

_CRITIQUE_PROMPT = """\
You are an expert reviewer improving an AI agent.

INPUT:
{input}

AGENT RESPONSE:
{output}
{reference_block}{criteria_block}
Explain concisely (2-4 sentences) what is wrong or missing in the response and
what the agent should do differently next time. Be specific and actionable.
"""

_IMPROVE_PROMPT = """\
You are a prompt engineer improving the instruction given to an AI agent.

CURRENT INSTRUCTION:
\"\"\"
{current}
\"\"\"
{criteria_block}
The agent produced these FAILURES on a golden dataset (input, what it produced,
the expected answer, and a reviewer critique):

{failures}

Rewrite the instruction so the agent avoids these failures while staying general
(do NOT hard-code answers to these specific inputs). Keep it clear and concise.

Return ONLY the rewritten instruction text, with no preamble, quotes, or
markdown fences.
"""


class LLMJudge:
    """Model-graded evaluation backed by a pluggable completion function.

    Args:
        complete: How the judge reaches a model. Accepts a
            :class:`~adapt_agent.optimization.providers.ModelProvider`, a bare
            ``Callable[[str], str]``, or a registered provider name (e.g.
            ``"anthropic"``). This is the provider-agnostic seam: swap vendors
            without touching the judge.
        rubric: Default grading rubric used by :meth:`score` when an example
            does not provide its own criteria.
        pass_threshold: Normalized score (``[0, 1]``) at or above which an output
            is considered a pass.
        scale: The maximum integer the model is asked to rate on (default 10);
            scores are divided by this to normalize to ``[0, 1]``.
        max_failures: Maximum number of failures embedded in an
            :meth:`improve_prompt` meta-prompt (keeps prompts bounded).
        on_error: Score returned when a grading call or parse fails. Defaults to
            ``0.0`` so a broken judge fails closed rather than inflating results.
    """

    def __init__(
        self,
        complete: ModelProvider | CompletionFn | str,
        *,
        rubric: str = _DEFAULT_RUBRIC,
        pass_threshold: float = 0.6,
        scale: int = 10,
        max_failures: int = 8,
        on_error: float = 0.0,
    ):
        # Coerce providers / names / callables to a single completion callable.
        from adapt_agent.optimization.providers import as_provider

        if isinstance(complete, str) or not callable(complete):
            self.complete: CompletionFn = as_provider(complete)
        else:
            self.complete = complete
        self.rubric = rubric
        self.pass_threshold = pass_threshold
        self.scale = max(1, int(scale))
        self.max_failures = max(1, int(max_failures))
        self.on_error = on_error

    # -- grading ---------------------------------------------------------------

    def score(
        self,
        input_data: Any,
        output: Any,
        expected: Any = None,
        *,
        criteria: str | None = None,
        rubric: str | None = None,
    ) -> JudgeVerdict:
        """Grade a single ``output`` and return a :class:`JudgeVerdict`."""
        prompt = _SCORE_PROMPT.format(
            input=_stringify(input_data),
            output=_stringify(output),
            reference_block=_reference_block(expected),
            criteria_block=_criteria_block(criteria),
            rubric=rubric or self.rubric,
        )
        raw = self._complete(prompt)
        if raw is None:
            return JudgeVerdict(self.on_error, self.on_error >= self.pass_threshold, raw="")
        parsed = _extract_json(raw)
        if parsed is None:
            # Last-ditch heuristic: find the first integer in the text.
            number = _first_number(raw)
            if number is None:
                return JudgeVerdict(self.on_error, False, reasoning="unparseable", raw=raw)
            parsed = {"score": number}
        score = _normalize_score(parsed.get("score"), self.scale, self.on_error)
        passed = parsed.get("pass")
        if not isinstance(passed, bool):
            passed = score >= self.pass_threshold
        return JudgeVerdict(
            score=score,
            passed=passed,
            reasoning=str(parsed.get("reasoning", "")),
            raw=raw,
            metadata={k: v for k, v in parsed.items() if k not in ("score", "pass", "reasoning")},
        )

    def compare(
        self,
        input_data: Any,
        output_a: Any,
        output_b: Any,
        *,
        criteria: str | None = None,
    ) -> str:
        """Pairwise preference. Returns ``"A"``, ``"B"`` or ``"tie"``."""
        prompt = _COMPARE_PROMPT.format(
            input=_stringify(input_data),
            output_a=_stringify(output_a),
            output_b=_stringify(output_b),
            criteria_block=_criteria_block(criteria),
        )
        raw = self._complete(prompt)
        if raw is None:
            return "tie"
        parsed = _extract_json(raw) or {}
        winner = str(parsed.get("winner", "")).strip().upper()
        if winner in ("A", "B"):
            return winner
        # Fall back to scanning the raw text.
        text = raw.strip().upper()
        if text.startswith("A") or '"A"' in text:
            return "A"
        if text.startswith("B") or '"B"' in text:
            return "B"
        return "tie"

    def critique(
        self,
        input_data: Any,
        output: Any,
        expected: Any = None,
        *,
        criteria: str | None = None,
    ) -> str:
        """Return actionable natural-language feedback on an output."""
        prompt = _CRITIQUE_PROMPT.format(
            input=_stringify(input_data),
            output=_stringify(output),
            reference_block=_reference_block(expected),
            criteria_block=_criteria_block(criteria),
        )
        return (self._complete(prompt) or "").strip()

    def improve_prompt(
        self,
        current: str,
        failures: list[dict[str, Any]],
        *,
        criteria: str | None = None,
    ) -> str | None:
        """Propose an improved instruction from a batch of judged failures.

        Args:
            current: The instruction being optimized.
            failures: Records with keys ``input``, ``output``, ``expected`` and
                optionally ``critique``. Truncated to ``max_failures``.
            criteria: Optional task-level grading criteria to honor.

        Returns:
            The rewritten instruction, or ``None`` if the completion failed or
            returned something unusable (so callers can keep the current value).
        """
        if not current:
            return None
        rendered = _render_failures(failures[: self.max_failures])
        prompt = _IMPROVE_PROMPT.format(
            current=current,
            criteria_block=_criteria_block(criteria),
            failures=rendered,
        )
        proposal = self._complete(prompt)
        if not proposal:
            return None
        cleaned = _strip_fences(proposal).strip()
        # Reject degenerate proposals (empty or echoing the meta-prompt).
        if not cleaned or cleaned == current.strip():
            return None
        return cleaned

    # -- metric adapter --------------------------------------------------------

    def as_metric(
        self,
        name: str = "llm_judge",
        *,
        criteria: str | None = None,
        rubric: str | None = None,
    ):
        """Return a :class:`~adapt_agent.optimization.metrics.Metric` wrapping the judge.

        The returned metric needs the full :class:`~adapt_agent.optimization.dataset.Example`
        (for the input and any per-example ``criteria`` in metadata) and is
        therefore evaluated by the harness with example context.
        """
        # Imported lazily to avoid a circular import at module load time.
        from adapt_agent.optimization.metrics import Metric

        def _fn(output: Any, expected: Any, example: Any = None) -> float:
            ex_criteria = criteria
            input_data: Any = ""
            if example is not None:
                input_data = getattr(example, "inputs", "")
                meta = getattr(example, "metadata", {}) or {}
                ex_criteria = meta.get("criteria", criteria)
            return self.score(
                input_data, output, expected, criteria=ex_criteria, rubric=rubric
            ).score

        return Metric(name, _fn, needs_example=True)

    # -- internals -------------------------------------------------------------

    def _complete(self, prompt: str) -> str | None:
        """Call the completion function, swallowing provider errors."""
        try:
            result = self.complete(prompt)
        except Exception:
            return None
        return result if isinstance(result, str) else (str(result) if result is not None else None)


# -- parsing / formatting helpers ---------------------------------------------


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _reference_block(expected: Any) -> str:
    if expected is None:
        return ""
    return f"\nREFERENCE ANSWER:\n{_stringify(expected)}\n"


def _criteria_block(criteria: str | None) -> str:
    if not criteria:
        return ""
    return f"\nTASK-SPECIFIC CRITERIA:\n{criteria}\n"


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object from a completion."""
    text = _strip_fences(text)
    # Fast path: the whole string is JSON.
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Otherwise locate the first balanced-looking {...} span.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` / ``` ... ``` markdown fences if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9]*\s*", "", stripped)
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _first_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _normalize_score(raw: Any, scale: int, default: float) -> float:
    if isinstance(raw, bool):  # bool is an int subclass; treat as pass/fail
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        num = _first_number(raw)
        if num is None:
            return default
        value = num
    else:
        return default
    # Values already in [0, 1] are taken as-is; larger values are on the scale.
    if 0.0 <= value <= 1.0 and scale > 1:
        return value
    return max(0.0, min(1.0, value / scale))


def _render_failures(failures: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for i, f in enumerate(failures, 1):
        parts = [f"--- Failure {i} ---", f"INPUT: {_stringify(f.get('input'))}"]
        parts.append(f"PRODUCED: {_stringify(f.get('output'))}")
        if f.get("expected") is not None:
            parts.append(f"EXPECTED: {_stringify(f.get('expected'))}")
        if f.get("critique"):
            parts.append(f"CRITIQUE: {f['critique']}")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks) if blocks else "(no specific failures captured)"


__all__ = ["CompletionFn", "JudgeVerdict", "LLMJudge"]
