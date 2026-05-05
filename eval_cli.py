"""Eval harness: runs all questions through /ask and reports pass rates."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import httpx
import anthropic
from src.config import ANTHROPIC_API_KEY, GENERATION_MODEL

API_URL = "http://localhost:8000/ask"
QUESTIONS_FILE = Path("evals/questions.json")
TIMEOUT = 60.0

GRADER_PROMPT = """\
You are an impartial evaluator for an HR chatbot. You will be given a question, \
an evaluation rubric, and the chatbot's answer. Judge whether the answer satisfies \
the rubric.

Respond with JSON only, no markdown fences:
{"result": "PASS" or "FAIL", "reason": "<one sentence>"}

Be strict: if the rubric requires specific facts or distinctions and the answer \
omits them, return FAIL.\
"""

_anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def keyword_check(answer: str, must_contain: list[str], must_not_contain: list[str]) -> tuple[bool, list[str]]:
    """Return (passed, list_of_failures) for keyword containment checks."""
    lower = answer.lower()
    failures: list[str] = []
    for phrase in must_contain:
        if phrase.lower() not in lower:
            failures.append(f"missing: {phrase!r}")
    for phrase in must_not_contain:
        if phrase.lower() in lower:
            failures.append(f"found forbidden: {phrase!r}")
    return len(failures) == 0, failures


def model_grade(question: str, rubric: str, answer: str) -> tuple[bool, str]:
    """Ask Claude to judge the answer against the rubric. Returns (passed, reason)."""
    user_message = f"QUESTION: {question}\n\nRUBRIC: {rubric}\n\nANSWER: {answer}"
    message = _anthropic.messages.create(
        model=GENERATION_MODEL,
        max_tokens=256,
        system=GRADER_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = message.content[0].text.strip()
    try:
        parsed = json.loads(raw)
        passed = parsed["result"].upper() == "PASS"
        reason = parsed.get("reason", "")
    except Exception:
        passed = False
        reason = f"grader returned unparseable response: {raw[:100]}"
    return passed, reason


def main() -> None:
    args = sys.argv[1:]
    use_model_grader = "--model-grade" in args

    questions = json.loads(QUESTIONS_FILE.read_text())

    kw_passed = kw_failed = 0
    mg_passed = mg_failed = 0
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"kw_pass": 0, "kw_fail": 0, "mg_pass": 0, "mg_fail": 0})

    mode = "keyword + model-graded" if use_model_grader else "keyword"
    print(f"Running {len(questions)} questions against {API_URL} [{mode}]\n")
    print("=" * 70)

    for q in questions:
        qid = q["id"]
        category = q["category"]

        try:
            response = httpx.post(
                API_URL,
                json={"question": q["question"]},
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            answer = response.json()["answer"]
        except Exception as exc:
            print(f"[ERROR] {qid}: {exc}")
            kw_failed += 1
            by_category[category]["kw_fail"] += 1
            continue

        kw_ok, kw_failures = keyword_check(answer, q["must_contain"], q["must_not_contain"])

        kw_label = "PASS" if kw_ok else "FAIL"
        if kw_ok:
            kw_passed += 1
            by_category[category]["kw_pass"] += 1
        else:
            kw_failed += 1
            by_category[category]["kw_fail"] += 1

        if use_model_grader:
            mg_ok, mg_reason = model_grade(q["question"], q["rubric"], answer)
            mg_label = "PASS" if mg_ok else "FAIL"
            if mg_ok:
                mg_passed += 1
                by_category[category]["mg_pass"] += 1
            else:
                mg_failed += 1
                by_category[category]["mg_fail"] += 1
            print(f"[KW:{kw_label} MG:{mg_label}]  {qid} ({category})")
        else:
            print(f"[{kw_label}]  {qid} ({category})")

        if not kw_ok:
            for f in kw_failures:
                print(f"        keyword: {f}")

        if use_model_grader and not mg_ok:
            print(f"        model:   {mg_reason}")

        if not kw_ok or (use_model_grader and not mg_ok):
            print(f"        answer:  {answer[:300]}")

    total = kw_passed + kw_failed
    print("=" * 70)
    print(f"\nKeyword:  {kw_passed}/{total} passed ({100 * kw_passed // total}%)")
    if use_model_grader:
        mg_total = mg_passed + mg_failed
        print(f"Model:    {mg_passed}/{mg_total} passed ({100 * mg_passed // mg_total}%)")

    print("\nBy category:")
    for cat, counts in sorted(by_category.items()):
        kw_total = counts["kw_pass"] + counts["kw_fail"]
        row = f"  {cat:<22} keyword {counts['kw_pass']}/{kw_total}"
        if use_model_grader:
            mg_total = counts["mg_pass"] + counts["mg_fail"]
            row += f"   model {counts['mg_pass']}/{mg_total}"
        print(row)

    if kw_failed or (use_model_grader and mg_failed):
        sys.exit(1)


if __name__ == "__main__":
    main()
