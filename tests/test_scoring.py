"""Tests for deterministic scoring: normalisation, keywords, refusals, retrieval."""

import unittest

from src.evaluation.scoring import is_refusal, keyword_check, normalize, score_retrieval
from src.generate import REFUSAL_MESSAGE


class NormalizeTest(unittest.TestCase):
    def test_curly_quotes_and_markdown(self) -> None:
        self.assertEqual(normalize("I **don’t**  have\nit"), "i don't have it")

    def test_dashes(self) -> None:
        self.assertEqual(normalize("2–10 days"), "2-10 days")


class KeywordTest(unittest.TestCase):
    def test_any_alternative_passes(self) -> None:
        result = keyword_check("Give two business days notice.", (["2 business days", "two business days"],), ())
        self.assertTrue(result.passed)

    def test_missing_phrase_is_reported(self) -> None:
        result = keyword_check("Eight weeks.", ("2 business days",), ())
        self.assertFalse(result.passed)
        self.assertEqual(result.failures, ["missing: '2 business days'"])

    def test_forbidden_phrase_fails(self) -> None:
        result = keyword_check("Up to 26 weeks.", (), ("26 weeks",))
        self.assertFalse(result.passed)

    def test_bold_markdown_still_matches(self) -> None:
        self.assertTrue(keyword_check("You accrue **1.5 days** a month.", ("1.5 days",), ()).passed)


class RefusalTest(unittest.TestCase):
    def test_exact_refusal(self) -> None:
        self.assertTrue(is_refusal(REFUSAL_MESSAGE, REFUSAL_MESSAGE))

    def test_curly_apostrophe_refusal(self) -> None:
        self.assertTrue(is_refusal("I don’t have that information in the handbook", REFUSAL_MESSAGE))

    def test_answer_is_not_refusal(self) -> None:
        self.assertFalse(is_refusal("You accrue 1.5 days per month.", REFUSAL_MESSAGE))


class RetrievalScoreTest(unittest.TestCase):
    def test_refusal_case_has_no_retrieval_score(self) -> None:
        self.assertIsNone(score_retrieval(["a.md"], ()))

    def test_first_rank_hit(self) -> None:
        s = score_retrieval(["a.md", "b.md", "a.md", "c.md"], ("a.md",))
        self.assertTrue(s.hit)
        self.assertEqual(s.recall, 1.0)
        self.assertEqual(s.reciprocal_rank, 1.0)
        self.assertEqual(s.precision, 0.5)

    def test_partial_recall_and_later_rank(self) -> None:
        s = score_retrieval(["x.md", "y.md", "b.md"], ("a.md", "b.md"))
        self.assertEqual(s.recall, 0.5)
        self.assertAlmostEqual(s.reciprocal_rank, 1 / 3)
        self.assertEqual(s.missing_sources, ["a.md"])

    def test_complete_miss(self) -> None:
        s = score_retrieval(["x.md"], ("a.md",))
        self.assertFalse(s.hit)
        self.assertEqual((s.recall, s.reciprocal_rank, s.precision), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
