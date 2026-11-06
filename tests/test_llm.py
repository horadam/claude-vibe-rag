"""Tests for the provider adapter, the eval key check and per-request keys on /ask."""

import json
import logging
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from eval_cli import required_keys
from src import llm
from src.evaluation import judge as judge_module
from src.evaluation.dataset import EvalCase
from src.generate import GenerationResult
from src.retrieve import RetrievedChunk

FAKE_OPENAI_KEY = "sk-test-not-a-real-key-123"


def _openai_response(content: str | None = None, tool_args: str | None = None) -> SimpleNamespace:
    """Build an object shaped like an OpenAI chat completion."""
    tool_calls = None
    if tool_args is not None:
        tool_calls = [SimpleNamespace(function=SimpleNamespace(name="record_grade", arguments=tool_args))]
    message = SimpleNamespace(content=content, refusal=None, tool_calls=tool_calls)
    return SimpleNamespace(
        model="gpt-test-2026",
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30),
    )


class _FakeOpenAI:
    """Records the request and returns a canned response."""

    def __init__(self, response: SimpleNamespace) -> None:
        self.request: dict = {}

        def create(**kwargs: object) -> SimpleNamespace:
            self.request = kwargs
            return response

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


class OpenAICompletionTest(unittest.TestCase):
    def test_maps_text_model_and_tokens(self) -> None:
        fake = _FakeOpenAI(_openai_response(content="Answer (pto_policy.md)"))
        with patch.object(llm, "openai_client", return_value=fake):
            result = llm.complete("openai", "gpt-x", "SYS", "USER", max_tokens=100)
        self.assertEqual(result, llm.Completion("Answer (pto_policy.md)", "gpt-test-2026", 120, 30))
        self.assertEqual(fake.request["messages"][0], {"role": "system", "content": "SYS"})
        self.assertGreater(fake.request["max_completion_tokens"], 100)

    def test_empty_answer_raises(self) -> None:
        fake = _FakeOpenAI(_openai_response(content=None))
        with patch.object(llm, "openai_client", return_value=fake):
            with self.assertRaises(llm.ProviderOutputError):
                llm.complete("openai", "gpt-x", "SYS", "USER", max_tokens=100)

    def test_unknown_provider(self) -> None:
        with self.assertRaises(ValueError):
            llm.complete("gemini", "m", "s", "u", max_tokens=10)


class OpenAIJudgeTest(unittest.TestCase):
    CASE = EvalCase("c", "Q?", "single_doc", "Must say X.", ("a.md",), (), ())
    CHUNKS = [RetrievedChunk("a__0", "Policy text", "a.md", 0, "A", 0.1)]

    def _judge_with(self, tool_args: str | None) -> judge_module.JudgeResult:
        fake = _FakeOpenAI(_openai_response(tool_args=tool_args))
        with (
            patch.object(llm, "openai_client", return_value=fake),
            patch.object(judge_module, "JUDGE_PROVIDER", "openai"),
        ):
            result = judge_module.judge(self.CASE, "Answer", self.CHUNKS)
        self.assertEqual(fake.request["tool_choice"], {"type": "function", "function": {"name": "record_grade"}})
        return result

    def test_json_arguments_are_decoded(self) -> None:
        args = {"correct": "PASS", "correct_reason": "ok", "grounded": "PASS", "grounded_reason": "ok"}
        result = self._judge_with(json.dumps(args))
        self.assertEqual((result.correct, result.grounded, result.model), ("PASS", "PASS", "gpt-test-2026"))

    def test_invalid_json_is_error(self) -> None:
        with self.assertLogs("src.evaluation.judge", level="ERROR"):
            result = self._judge_with("{not json")
        self.assertEqual((result.correct, result.grounded), ("ERROR", "ERROR"))

    def test_no_tool_call_is_error(self) -> None:
        with self.assertLogs("src.evaluation.judge", level="ERROR"):
            result = self._judge_with(None)
        self.assertEqual(result.correct, "ERROR")


class RequiredKeysTest(unittest.TestCase):
    def test_retrieval_only_needs_openai(self) -> None:
        self.assertEqual(required_keys(True, False, "anthropic", "anthropic"), ["OPENAI_API_KEY"])

    def test_openai_everywhere_needs_one_key(self) -> None:
        self.assertEqual(required_keys(False, True, "openai", "openai"), ["OPENAI_API_KEY"])

    def test_mixed_providers(self) -> None:
        self.assertEqual(
            required_keys(False, True, "openai", "anthropic"),
            ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        )


class AskKeysTest(unittest.TestCase):
    def setUp(self) -> None:
        from src import server

        self.server = server
        self.client = TestClient(server.app)
        self.calls: dict = {}

        def fake_retrieve(question: str, api_key: str | None = None) -> list[RetrievedChunk]:
            self.calls["retrieve_key"] = api_key
            return [RetrievedChunk("a__0", "Policy text", "a.md", 0, "A", 0.1)]

        def fake_generate(question: str, chunks: list, provider: str, api_key: str | None) -> GenerationResult:
            self.calls.update(provider=provider, generate_key=api_key)
            return GenerationResult("Answer (a.md)", [], model="m", input_tokens=1, output_tokens=1)

        self.patches = [
            patch.object(server, "retrieve", fake_retrieve),
            patch.object(server, "generate", fake_generate),
            patch.dict(os.environ, {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()

    def test_header_key_is_used_and_never_echoed_or_logged(self) -> None:
        with self.assertLogs(level=logging.DEBUG) as logs:
            res = self.client.post(
                "/ask",
                json={"question": "PTO?", "provider": "openai"},
                headers={"X-OpenAI-Key": FAKE_OPENAI_KEY},
            )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.calls, {"retrieve_key": FAKE_OPENAI_KEY, "provider": "openai", "generate_key": FAKE_OPENAI_KEY})
        self.assertNotIn(FAKE_OPENAI_KEY, res.text)
        self.assertNotIn(FAKE_OPENAI_KEY, "\n".join(logs.output))

    def test_missing_key_is_503_with_fix(self) -> None:
        res = self.client.post("/ask", json={"question": "PTO?", "provider": "openai"})
        self.assertEqual(res.status_code, 503)
        self.assertIn("Model settings", res.json()["detail"])

    def test_anthropic_answers_still_need_openai_for_search(self) -> None:
        res = self.client.post(
            "/ask",
            json={"question": "PTO?", "provider": "anthropic"},
            headers={"X-Anthropic-Key": "sk-ant-test"},
        )
        self.assertEqual(res.status_code, 503)
        self.assertIn("OpenAI", res.json()["detail"])

    def test_env_key_is_the_fallback(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "from-env"}):
            res = self.client.post("/ask", json={"question": "PTO?", "provider": "openai"})
        self.assertEqual(res.status_code, 200)
        self.assertIsNone(self.calls["generate_key"])

    def test_provider_error_masks_key(self) -> None:
        from src.server import redact_keys

        message = "Incorrect API key provided: sk-proj-abc1****wxyz. Find your key at platform."
        self.assertEqual(
            redact_keys(message),
            "Incorrect API key provided: sk-…[redacted]. Find your key at platform.",
        )

    def test_config_reports_booleans_only(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": FAKE_OPENAI_KEY}):
            res = self.client.get("/api/config")
        self.assertEqual(res.json()["server_keys"], {"anthropic": False, "openai": True})
        self.assertNotIn(FAKE_OPENAI_KEY, res.text)


if __name__ == "__main__":
    unittest.main()
