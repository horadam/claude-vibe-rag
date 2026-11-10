"""Run eval cases through the RAG pipeline in-process and score each one.

The pipeline stages are passed in as functions, so tests can swap in fakes
and a retrieval-only run can skip generation entirely.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from src.evaluation.dataset import EvalCase
from src.evaluation.judge import JudgeResult
from src.evaluation.scoring import (
    KeywordResult,
    RetrievalScores,
    is_refusal,
    keyword_check,
    score_retrieval,
)
from src.generate import REFUSAL_MESSAGE, GenerationResult
from src.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

RetrieveFn = Callable[[str, int], list[RetrievedChunk]]
GenerateFn = Callable[[str, list[RetrievedChunk]], GenerationResult]
JudgeFn = Callable[[EvalCase, str, list[RetrievedChunk]], JudgeResult]


@dataclass
class RetrievedRef:
    """What was retrieved for a case, without the chunk text."""

    source: str
    chunk_index: int
    distance: float


@dataclass
class CaseResult:
    """Everything measured for one case. Serialised as-is into the run record."""

    id: str
    category: str
    question: str
    expects_refusal: bool
    expected_sources: list[str] = field(default_factory=list)
    retrieved: list[RetrievedRef] = field(default_factory=list)
    retrieval: RetrievalScores | None = None
    answer: str | None = None
    refused: bool | None = None
    keyword: KeywordResult | None = None
    judge: JudgeResult | None = None
    retrieve_ms: float | None = None
    generate_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def refusal_ok(self) -> bool | None:
        """Did the model refuse exactly when it should have? None if not generated."""
        if self.refused is None:
            return None
        return self.refused == self.expects_refusal

    @property
    def passed(self) -> bool | None:
        """Deterministic pass: keywords present and refusal behaviour correct.

        None for retrieval-only runs, where there is no answer to check.
        """
        if self.error:
            return False
        if self.keyword is None:
            return None
        return self.keyword.passed and bool(self.refusal_ok)


def _elapsed_ms(start: float) -> float:
    """Milliseconds since start, rounded for readable reports."""
    return round((time.perf_counter() - start) * 1000, 1)


def run_case(
    case: EvalCase,
    retrieve_fn: RetrieveFn,
    generate_fn: GenerateFn | None,
    judge_fn: JudgeFn | None,
    top_k: int,
) -> CaseResult:
    """Retrieve, optionally generate and judge, and score a single case."""
    result = CaseResult(
        id=case.id,
        category=case.category,
        question=case.question,
        expects_refusal=case.expects_refusal,
        expected_sources=list(case.expected_sources),
    )

    start = time.perf_counter()
    chunks = retrieve_fn(case.question, top_k)
    result.retrieve_ms = _elapsed_ms(start)
    result.retrieved = [RetrievedRef(c.source, c.chunk_index, round(c.distance, 4)) for c in chunks]
    result.retrieval = score_retrieval([c.source for c in chunks], case.expected_sources)

    if generate_fn is None:
        return result

    start = time.perf_counter()
    generation = generate_fn(case.question, chunks)
    result.generate_ms = _elapsed_ms(start)
    result.answer = generation.answer
    result.input_tokens = generation.input_tokens
    result.output_tokens = generation.output_tokens
    result.refused = is_refusal(generation.answer, REFUSAL_MESSAGE)
    result.keyword = keyword_check(generation.answer, case.must_contain, case.must_not_contain)

    if judge_fn is not None:
        result.judge = judge_fn(case, generation.answer, chunks)
    return result


def run_eval(
    cases: list[EvalCase],
    retrieve_fn: RetrieveFn,
    generate_fn: GenerateFn | None,
    judge_fn: JudgeFn | None,
    top_k: int,
    on_result: Callable[[CaseResult], None] | None = None,
) -> list[CaseResult]:
    """Run every case and return their results in order."""
    results: list[CaseResult] = []
    for case in cases:
        try:
            result = run_case(case, retrieve_fn, generate_fn, judge_fn, top_k)
        except Exception as exc:
            # Deliberately not re-raised: one failing API call shouldn't throw
            # away the rest of a paid run. The error is logged, stored on the
            # case, counted in the summary and makes the CLI exit non-zero.
            logger.exception("case %s failed", case.id)
            result = CaseResult(
                id=case.id,
                category=case.category,
                question=case.question,
                expects_refusal=case.expects_refusal,
                expected_sources=list(case.expected_sources),
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results
