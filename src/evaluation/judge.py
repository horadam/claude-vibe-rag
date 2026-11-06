"""Model-graded checks: is the answer correct, and is it grounded in the context?

Keyword checks can't tell a right answer from a lucky one, or notice an
invented detail. A separate judge model grades two things independently:

- correct: does the answer satisfy the human-written rubric?
- grounded: is every factual claim supported by the retrieved context?

An answer can be correct but ungrounded (the model knew it from training) or
grounded but incorrect (faithful to the wrong chunks). Retrieval and prompt
changes move these two numbers differently, so they are reported separately.
"""

import logging
from dataclasses import dataclass
from functools import cache
from typing import Literal

import anthropic

from src.config import JUDGE_MODEL, require_env
from src.evaluation.dataset import EvalCase
from src.generate import build_context_block
from src.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

Verdict = Literal["PASS", "FAIL", "ERROR"]

JUDGE_SYSTEM_PROMPT = """\
You are a strict, impartial evaluator for an HR handbook assistant. You will \
receive the employee's QUESTION, a RUBRIC written by a human, the CONTEXT \
excerpts the assistant was given, and the assistant's ANSWER.

Grade two criteria independently:

1. correct — PASS only if the answer satisfies every requirement in the rubric. \
Missing a required fact or distinction is a FAIL.
2. grounded — PASS only if every factual claim in the answer is supported by the \
CONTEXT. Any number, rule or detail that is not in the context is a FAIL, even \
if it happens to be true. A refusal that makes no factual claims is grounded.

Record your grades with the record_grade tool. Keep each reason to one sentence.\
"""

_GRADE_TOOL = {
    "name": "record_grade",
    "description": "Record the grade for both criteria.",
    "input_schema": {
        "type": "object",
        "properties": {
            "correct": {"type": "string", "enum": ["PASS", "FAIL"]},
            "correct_reason": {"type": "string"},
            "grounded": {"type": "string", "enum": ["PASS", "FAIL"]},
            "grounded_reason": {"type": "string"},
        },
        "required": ["correct", "correct_reason", "grounded", "grounded_reason"],
    },
}


@dataclass
class JudgeResult:
    """Both verdicts, with reasons. ERROR means the judge itself failed."""

    correct: Verdict
    correct_reason: str
    grounded: Verdict
    grounded_reason: str
    model: str


@cache
def _client() -> anthropic.Anthropic:
    """Return the shared Anthropic client, created on first use."""
    return anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))


def build_judge_message(case: EvalCase, answer: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble the judge's user message from the case, context and answer."""
    return (
        f"QUESTION: {case.question}\n\n"
        f"RUBRIC: {case.rubric}\n\n"
        f"CONTEXT:\n{build_context_block(chunks) or '(no context retrieved)'}\n\n"
        f"ANSWER: {answer}"
    )


def parse_grade(tool_input: object, model: str) -> JudgeResult:
    """Turn the judge's tool input into a JudgeResult; malformed input is an ERROR.

    A judge failure is reported as ERROR, not FAIL, so a broken judge can't
    quietly lower the pass rate.
    """
    if not isinstance(tool_input, dict):
        return _error(f"judge returned {type(tool_input).__name__}, expected an object", model)
    verdicts: dict[str, Verdict] = {}
    for key in ("correct", "grounded"):
        value = tool_input.get(key)
        if value not in ("PASS", "FAIL"):
            return _error(f"judge returned invalid {key!r}: {value!r}", model)
        verdicts[key] = value
    return JudgeResult(
        correct=verdicts["correct"],
        correct_reason=str(tool_input.get("correct_reason", "")),
        grounded=verdicts["grounded"],
        grounded_reason=str(tool_input.get("grounded_reason", "")),
        model=model,
    )


def _error(reason: str, model: str) -> JudgeResult:
    """Build a JudgeResult that marks both criteria as ERROR."""
    logger.error("judge error: %s", reason)
    return JudgeResult("ERROR", reason, "ERROR", reason, model)


def judge(case: EvalCase, answer: str, chunks: list[RetrievedChunk]) -> JudgeResult:
    """Ask the judge model to grade one answer on correctness and groundedness."""
    message = _client().messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        system=JUDGE_SYSTEM_PROMPT,
        tools=[_GRADE_TOOL],
        # Forcing the tool call makes the grade structured data, not prose to parse.
        tool_choice={"type": "tool", "name": "record_grade"},
        messages=[{"role": "user", "content": build_judge_message(case, answer, chunks)}],
    )
    tool_uses = [block for block in message.content if block.type == "tool_use"]
    if not tool_uses:
        return _error("judge did not call record_grade", message.model)
    return parse_grade(tool_uses[0].input, message.model)
