"""Deterministic scoring: keyword checks, refusal detection and retrieval metrics.

None of this calls a model, so it is cheap, repeatable and unit-tested.
"""

import re
from dataclasses import dataclass

from src.evaluation.dataset import Phrase

# Typographic characters a model may emit that should match their plain form.
_CHAR_MAP = str.maketrans(
    {
        "‘": "'",  # left single quote
        "’": "'",  # right single quote / apostrophe
        "“": '"',
        "”": '"',
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen (GPT writes "company-issued" with it)
        "\u2012": "-",  # figure dash
        "–": "-",  # en dash
        "—": "-",  # em dash
        " ": " ",  # non-breaking space
        "\u2212": "-",  # minus sign
        "\u202f": " ",  # narrow non-breaking space
    }
)
_MARKDOWN_EMPHASIS = re.compile(r"[*_`]+")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase and strip formatting so equivalent answers compare equal.

    "**1.5 days**" and "1.5 days", or "don’t" and "don't", should not decide
    whether a case passes.
    """
    text = text.translate(_CHAR_MAP)
    text = _MARKDOWN_EMPHASIS.sub("", text)
    return _WHITESPACE.sub(" ", text).strip().lower()


@dataclass
class KeywordResult:
    """Outcome of the must_contain / must_not_contain checks."""

    passed: bool
    failures: list[str]


def keyword_check(
    answer: str, must_contain: tuple[Phrase, ...], must_not_contain: tuple[str, ...]
) -> KeywordResult:
    """Check required phrases (any alternative counts) and forbidden phrases."""
    text = normalize(answer)
    failures: list[str] = []
    for entry in must_contain:
        options = entry if isinstance(entry, list) else [entry]
        if not any(normalize(option) in text for option in options):
            label = " | ".join(options)
            failures.append(f"missing: {label!r}")
    for phrase in must_not_contain:
        if normalize(phrase) in text:
            failures.append(f"found forbidden: {phrase!r}")
    return KeywordResult(passed=not failures, failures=failures)


def is_refusal(answer: str, refusal_message: str) -> bool:
    """True when the answer contains the system prompt's refusal sentence."""
    return normalize(refusal_message).rstrip(".") in normalize(answer)


@dataclass
class RetrievalScores:
    """Source-level retrieval quality for one question.

    - hit: at least one expected source was retrieved.
    - recall: share of expected sources that appear in the top k.
    - reciprocal_rank: 1 / rank of the first chunk from an expected source
      (1.0 = first result, 0.5 = second, 0 = none). Averaged, this is MRR.
    - precision: share of retrieved chunks that come from an expected source.
      Low precision means the prompt is padded with unrelated text.
    """

    hit: bool
    recall: float
    reciprocal_rank: float
    precision: float
    missing_sources: list[str]


def score_retrieval(
    retrieved_sources: list[str], expected_sources: tuple[str, ...]
) -> RetrievalScores | None:
    """Score retrieved chunk sources (in rank order) against the labelled ones.

    Returns None for refusal cases, which have no expected sources to find.
    """
    if not expected_sources:
        return None
    expected = set(expected_sources)
    found = expected & set(retrieved_sources)

    reciprocal_rank = 0.0
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected:
            reciprocal_rank = 1.0 / rank
            break

    relevant_chunks = sum(1 for s in retrieved_sources if s in expected)
    precision = relevant_chunks / len(retrieved_sources) if retrieved_sources else 0.0

    return RetrievalScores(
        hit=bool(found),
        recall=len(found) / len(expected),
        reciprocal_rank=reciprocal_rank,
        precision=precision,
        missing_sources=sorted(expected - found),
    )
