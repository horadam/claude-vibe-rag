"""Tests for the eval runner and report, using fake pipeline stages."""

import unittest

from src.evaluation.dataset import EvalCase
from src.evaluation.judge import JudgeResult
from src.evaluation.report import build_run_record, compare_runs, percentile, render_markdown
from src.evaluation.runner import run_eval
from src.generate import REFUSAL_MESSAGE, GenerationResult
from src.retrieve import RetrievedChunk

ANSWERABLE = EvalCase("pto", "How much PTO?", "single_doc", "1.5 days", ("pto.md",), ("1.5 days",), ())
REFUSAL = EvalCase("ticker", "Ticker?", "out_of_scope", "Refuse.", (), (), ())

ANSWERS = {
    "How much PTO?": "You accrue 1.5 days per month (pto.md).",
    "Ticker?": REFUSAL_MESSAGE,
}


def fake_retrieve(question: str, top_k: int) -> list[RetrievedChunk]:
    """Always return two chunks: the PTO policy first."""
    return [
        RetrievedChunk("pto__0", "PTO text", "pto.md", 0, "PTO", 0.2),
        RetrievedChunk("hr__0", "Other text", "other.md", 0, "Other", 0.5),
    ][:top_k]


def fake_generate(question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
    """Return a canned answer with token counts."""
    return GenerationResult(ANSWERS[question], [], "fake-model", 100, 20)


def fake_judge(case: EvalCase, answer: str, chunks: list[RetrievedChunk]) -> JudgeResult:
    """Pass everything."""
    return JudgeResult("PASS", "ok", "PASS", "ok", "fake-judge")


def failing_retrieve(question: str, top_k: int) -> list[RetrievedChunk]:
    """Simulate an API failure."""
    raise RuntimeError("embedding API down")


CONFIG = {
    "mode": "judged", "top_k": 2, "generation_model": "fake-model", "judge_model": "fake-judge",
    "embedding_model": "fake-embed", "chunk_target_tokens": 500, "chunk_overlap_tokens": 50,
    "dataset_sha": "abc", "filters": {},
}


class RunnerTest(unittest.TestCase):
    def test_full_run_scores_everything(self) -> None:
        results = run_eval([ANSWERABLE, REFUSAL], fake_retrieve, fake_generate, fake_judge, 2)
        pto, ticker = results
        self.assertTrue(pto.passed)
        self.assertEqual(pto.retrieval.reciprocal_rank, 1.0)
        self.assertEqual(pto.retrieval.precision, 0.5)
        self.assertFalse(pto.refused)
        self.assertTrue(ticker.refused)
        self.assertTrue(ticker.passed)
        self.assertIsNone(ticker.retrieval)
        self.assertEqual(pto.input_tokens, 100)

    def test_retrieval_only_has_no_answer_verdict(self) -> None:
        [result] = run_eval([ANSWERABLE], fake_retrieve, None, None, 2)
        self.assertIsNone(result.answer)
        self.assertIsNone(result.passed)
        self.assertEqual(result.retrieval.recall, 1.0)

    def test_false_refusal_fails(self) -> None:
        def refuse(question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
            return GenerationResult(REFUSAL_MESSAGE, [])

        [result] = run_eval([ANSWERABLE], fake_retrieve, refuse, None, 2)
        self.assertFalse(result.refusal_ok)
        self.assertFalse(result.passed)

    def test_error_is_recorded_and_run_continues(self) -> None:
        with self.assertLogs("src.evaluation.runner", level="ERROR") as logs:
            results = run_eval([ANSWERABLE, REFUSAL], failing_retrieve, fake_generate, None, 2)
        self.assertEqual(len(logs.records), 2)
        self.assertEqual(len(results), 2)
        self.assertIn("embedding API down", results[0].error)
        self.assertFalse(results[0].passed)


class ReportTest(unittest.TestCase):
    def test_percentile_nearest_rank(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2)
        self.assertEqual(percentile([1, 2, 3, 4], 95), 4)
        self.assertIsNone(percentile([], 50))

    def test_summary_and_markdown(self) -> None:
        results = run_eval([ANSWERABLE, REFUSAL], fake_retrieve, fake_generate, fake_judge, 2)
        record = build_run_record(results, CONFIG, "test")
        summary = record["summary"]
        self.assertEqual(summary["answers"], {"passed": 2, "total": 2})
        self.assertEqual(summary["refusal"]["correct_refusals"], 1)
        self.assertEqual(summary["refusal"]["false_refusals"], 0)
        self.assertEqual(summary["retrieval"]["mrr"], 1.0)
        self.assertEqual(summary["tokens"], {"input": 200, "output": 40})
        markdown = render_markdown(record)
        self.assertIn("| Judge: grounded | 2 / 2 (100%) |", markdown)
        self.assertIn("None.", markdown)

    def test_compare_reports_flipped_case(self) -> None:
        good = build_run_record(
            run_eval([ANSWERABLE], fake_retrieve, fake_generate, None, 2), CONFIG, None
        )

        def wrong(question: str, chunks: list[RetrievedChunk]) -> GenerationResult:
            return GenerationResult("You accrue 2 days.", [])

        bad = build_run_record(run_eval([ANSWERABLE], fake_retrieve, wrong, None, 2), CONFIG, None)
        lines = compare_runs(good, bad)
        self.assertIn("  pto: PASS → FAIL", lines)
        self.assertTrue(any(line.startswith("  - answer checks") for line in lines))


if __name__ == "__main__":
    unittest.main()
