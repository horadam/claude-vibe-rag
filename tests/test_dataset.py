"""Tests for loading and validating the eval question set."""

import json
import tempfile
import unittest
from pathlib import Path

from src.config import EVAL_QUESTIONS_FILE
from src.evaluation.dataset import DatasetError, load_cases


def _write_cases(cases: list[dict]) -> Path:
    """Write cases to a temporary JSON file and return its path."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(cases, handle)
    handle.close()
    return Path(handle.name)


def _case(**overrides: object) -> dict:
    """A valid answerable case, with fields overridden as needed."""
    case = {
        "id": "c1",
        "question": "How much PTO?",
        "category": "single_doc",
        "rubric": "States 1.5 days.",
        "expected_sources": ["pto_policy.md"],
        "must_contain": ["1.5 days"],
    }
    case.update(overrides)
    return case


class RealDatasetTest(unittest.TestCase):
    def test_question_file_is_valid(self) -> None:
        cases = load_cases(EVAL_QUESTIONS_FILE)
        self.assertGreater(len(cases), 0)
        categories = {c.category for c in cases}
        for required in ("single_doc", "multi_doc", "ambiguous", "out_of_scope"):
            self.assertIn(required, categories)

    def test_multi_doc_cases_need_two_sources(self) -> None:
        for case in load_cases(EVAL_QUESTIONS_FILE):
            if case.category == "multi_doc":
                self.assertGreaterEqual(len(case.expected_sources), 2, case.id)


class ValidationTest(unittest.TestCase):
    def test_unknown_source_is_rejected(self) -> None:
        path = _write_cases([_case(expected_sources=["nope.md"])])
        with self.assertRaisesRegex(DatasetError, "not in corpus"):
            load_cases(path)

    def test_answerable_case_needs_sources(self) -> None:
        path = _write_cases([_case(expected_sources=[])])
        with self.assertRaisesRegex(DatasetError, "need expected_sources"):
            load_cases(path)

    def test_refusal_case_must_not_list_sources(self) -> None:
        path = _write_cases([_case(category="out_of_scope")])
        with self.assertRaisesRegex(DatasetError, "must not list"):
            load_cases(path)

    def test_duplicate_ids_are_rejected(self) -> None:
        path = _write_cases([_case(), _case()])
        with self.assertRaisesRegex(DatasetError, "duplicate"):
            load_cases(path)

    def test_unknown_category_is_rejected(self) -> None:
        path = _write_cases([_case(category="trivia")])
        with self.assertRaisesRegex(DatasetError, "unknown category"):
            load_cases(path)

    def test_alternatives_are_kept_as_lists(self) -> None:
        path = _write_cases([_case(must_contain=[["2 days", "two days"], "pto"])])
        case = load_cases(path)[0]
        self.assertEqual(case.must_contain, (["2 days", "two days"], "pto"))
        self.assertFalse(case.expects_refusal)


if __name__ == "__main__":
    unittest.main()
