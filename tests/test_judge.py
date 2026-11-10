"""Tests for parsing the judge's structured grade."""

import unittest

from src.evaluation.dataset import EvalCase
from src.evaluation.judge import build_judge_message, parse_grade
from src.retrieve import RetrievedChunk


class ParseGradeTest(unittest.TestCase):
    def test_valid_grade(self) -> None:
        result = parse_grade(
            {"correct": "PASS", "correct_reason": "ok", "grounded": "FAIL", "grounded_reason": "invented"},
            "judge-model",
        )
        self.assertEqual((result.correct, result.grounded), ("PASS", "FAIL"))
        self.assertEqual(result.grounded_reason, "invented")

    def test_invalid_verdict_is_error_not_fail(self) -> None:
        result = parse_grade({"correct": "MAYBE", "grounded": "PASS"}, "judge-model")
        self.assertEqual((result.correct, result.grounded), ("ERROR", "ERROR"))

    def test_non_object_is_error(self) -> None:
        self.assertEqual(parse_grade("PASS", "judge-model").correct, "ERROR")


class JudgeMessageTest(unittest.TestCase):
    def test_message_includes_context_and_rubric(self) -> None:
        case = EvalCase("c", "Q?", "single_doc", "Must say X.", ("a.md",), (), ())
        chunk = RetrievedChunk("a__0", "Policy text", "a.md", 0, "A", 0.1)
        message = build_judge_message(case, "Answer", [chunk])
        self.assertIn("RUBRIC: Must say X.", message)
        self.assertIn("[source: a.md]\nPolicy text", message)
        self.assertIn("ANSWER: Answer", message)


if __name__ == "__main__":
    unittest.main()
