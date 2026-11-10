"""Load and validate the eval question set."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from src.config import CORPUS_DIR

# Categories where the right behaviour is to refuse with the refusal message.
REFUSAL_CATEGORIES: frozenset[str] = frozenset({"out_of_scope", "false_premise"})
CATEGORIES: frozenset[str] = frozenset(
    {"single_doc", "multi_doc", "ambiguous"} | REFUSAL_CATEGORIES
)

# A must_contain entry is either one phrase, or a list of acceptable
# alternatives ("2 business days" or "two business days").
Phrase = str | list[str]


@dataclass(frozen=True)
class EvalCase:
    """One eval question with everything needed to score it."""

    id: str
    question: str
    category: str
    rubric: str
    expected_sources: tuple[str, ...]
    must_contain: tuple[Phrase, ...]
    must_not_contain: tuple[str, ...]

    @property
    def expects_refusal(self) -> bool:
        """True when the handbook has no answer and the model should refuse."""
        return self.category in REFUSAL_CATEGORIES


class DatasetError(ValueError):
    """The question file is malformed or references missing documents."""


def _parse_case(raw: dict, known_sources: set[str]) -> EvalCase:
    """Validate one raw JSON object and convert it into an EvalCase."""
    case_id = raw.get("id", "<missing id>")
    for field in ("id", "question", "category", "rubric"):
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            raise DatasetError(f"{case_id}: '{field}' must be a non-empty string")

    category = raw["category"]
    if category not in CATEGORIES:
        raise DatasetError(f"{case_id}: unknown category {category!r}")

    sources = tuple(raw.get("expected_sources", []))
    missing = [s for s in sources if s not in known_sources]
    if missing:
        raise DatasetError(f"{case_id}: expected_sources not in corpus: {missing}")
    if category in REFUSAL_CATEGORIES and sources:
        raise DatasetError(f"{case_id}: refusal cases must not list expected_sources")
    if category not in REFUSAL_CATEGORIES and not sources:
        raise DatasetError(f"{case_id}: answerable cases need expected_sources")

    must_contain: list[Phrase] = []
    for entry in raw.get("must_contain", []):
        if isinstance(entry, list):
            if not entry or not all(isinstance(x, str) for x in entry):
                raise DatasetError(f"{case_id}: alternatives must be non-empty string lists")
            must_contain.append(list(entry))
        elif isinstance(entry, str):
            must_contain.append(entry)
        else:
            raise DatasetError(f"{case_id}: must_contain entries are strings or lists")

    return EvalCase(
        id=raw["id"],
        question=raw["question"],
        category=category,
        rubric=raw["rubric"],
        expected_sources=sources,
        must_contain=tuple(must_contain),
        must_not_contain=tuple(raw.get("must_not_contain", [])),
    )


def load_cases(path: Path, corpus_dir: Path = CORPUS_DIR) -> list[EvalCase]:
    """Read the question file and return validated cases, failing on any problem."""
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise DatasetError(f"{path} must contain a JSON list")

    known_sources = {p.name for p in corpus_dir.glob("*.md")}
    cases = [_parse_case(raw, known_sources) for raw in raw_cases]

    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise DatasetError(f"duplicate case id {case.id!r}")
        seen.add(case.id)
    return cases


def dataset_fingerprint(path: Path) -> str:
    """Short hash of the question file, so a run records which version it used."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
