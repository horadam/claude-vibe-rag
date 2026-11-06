"""Eval harness CLI: run the question set through the pipeline and report.

Runs in-process (no server needed). Every run is saved to evals/runs/ and a
full, unfiltered run also rewrites evals/results.md.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    EMBEDDING_MODEL,
    EVAL_QUESTIONS_FILE,
    EVAL_RESULTS_FILE,
    EVAL_RUNS_DIR,
    GENERATION_MODEL,
    JUDGE_MODEL,
    TOP_K,
    require_env,
)
from src.evaluation.dataset import dataset_fingerprint, load_cases
from src.evaluation.report import (
    build_run_record,
    compare_runs,
    render_markdown,
    save_run,
)
from src.evaluation.runner import CaseResult, run_eval


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Define and parse the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", "--model-grade", action="store_true", dest="judge",
                        help="also grade correctness and groundedness with the judge model")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="score retrieval only; no generation (embedding calls only)")
    parser.add_argument("--k", type=int, default=TOP_K, help=f"chunks to retrieve (default {TOP_K})")
    parser.add_argument("--ids", help="comma-separated case ids to run")
    parser.add_argument("--category", help="run only this category")
    parser.add_argument("--label", help="short name added to the run id, e.g. k3")
    parser.add_argument("--compare", metavar="RUN",
                        help="run file to diff against, or 'latest' for the previous run")
    parser.add_argument("--no-save", action="store_true", help="don't write run files")
    args = parser.parse_args(argv)
    if args.judge and args.retrieval_only:
        parser.error("--judge needs generated answers; drop --retrieval-only")
    return args


def _verdict(result: CaseResult) -> str:
    """Short status label for the progress line."""
    if result.error:
        return "ERROR"
    if result.passed is None:
        return "RETR " if result.retrieval is None or result.retrieval.recall == 1 else "MISS "
    return "PASS " if result.passed else "FAIL "


def print_case(result: CaseResult) -> None:
    """Print one progress line, plus details for anything that went wrong."""
    parts = [f"[{_verdict(result)}] {result.id} ({result.category})"]
    if result.retrieval:
        r = result.retrieval
        parts.append(f"recall {r.recall:.2f} rr {r.reciprocal_rank:.2f}")
    if result.judge:
        parts.append(f"judge {result.judge.correct}/{result.judge.grounded}")
    if result.generate_ms is not None:
        parts.append(f"{(result.retrieve_ms + result.generate_ms) / 1000:.1f}s")
    print("  ".join(parts))

    if result.error:
        print(f"        error:   {result.error}")
        return
    if result.retrieval and result.retrieval.missing_sources:
        print(f"        missing: {', '.join(result.retrieval.missing_sources)}")
    if result.keyword and not result.keyword.passed:
        for failure in result.keyword.failures:
            print(f"        keyword: {failure}")
    if result.refusal_ok is False:
        print(f"        refusal: {'refused an answerable question' if result.refused else 'should have refused'}")
    if result.judge and result.judge.correct != "PASS":
        print(f"        correct: {result.judge.correct_reason}")
    if result.judge and result.judge.grounded != "PASS":
        print(f"        grounded:{result.judge.grounded_reason}")
    if result.answer and (result.passed is False or (result.judge and result.judge.correct != "PASS")):
        print(f"        answer:  {result.answer[:300]}")


def _latest_run(runs_dir: Path) -> Path | None:
    """Most recent saved run file, by name (run ids sort by time)."""
    runs = sorted(runs_dir.glob("*.json"))
    return runs[-1] if runs else None


def main(argv: list[str]) -> int:
    """Run the eval and return the process exit code."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s — %(message)s")
    args = parse_args(argv)

    cases = load_cases(EVAL_QUESTIONS_FILE)
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        unknown = wanted - {c.id for c in cases}
        if unknown:
            print(f"Unknown case ids: {', '.join(sorted(unknown))}")
            return 2
        cases = [c for c in cases if c.id in wanted]
    if args.category:
        cases = [c for c in cases if c.category == args.category]
    if not cases:
        print("No cases selected.")
        return 2
    filtered = bool(args.ids or args.category)

    baseline_path: Path | None = None
    if args.compare == "latest":
        baseline_path = _latest_run(EVAL_RUNS_DIR)
        if baseline_path is None:
            print("No previous run to compare with.")
    elif args.compare:
        baseline_path = Path(args.compare)

    # Fail once, up front, instead of once per case.
    needed = ["OPENAI_API_KEY"] + ([] if args.retrieval_only else ["ANTHROPIC_API_KEY"])
    try:
        for name in needed:
            require_env(name)
    except RuntimeError as exc:
        print(exc)
        return 2

    # Imported here so --help and dataset errors don't need the pipeline.
    from src.retrieve import retrieve
    generate_fn = judge_fn = None
    if not args.retrieval_only:
        from src.generate import generate
        generate_fn = generate
    if args.judge:
        from src.evaluation.judge import judge
        judge_fn = judge

    mode = "retrieval-only" if args.retrieval_only else ("judged" if args.judge else "answers")
    print(f"Running {len(cases)} cases [{mode}, k={args.k}]\n" + "=" * 70)
    results = run_eval(cases, retrieve, generate_fn, judge_fn, args.k, on_result=print_case)
    print("=" * 70)

    config = {
        "mode": mode,
        "top_k": args.k,
        "generation_model": None if args.retrieval_only else GENERATION_MODEL,
        "judge_model": JUDGE_MODEL if args.judge else None,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_target_tokens": CHUNK_TARGET_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "dataset_sha": dataset_fingerprint(EVAL_QUESTIONS_FILE),
        "filters": {"ids": args.ids, "category": args.category},
    }
    record = build_run_record(results, config, args.label)

    # The summary printed here is the same markdown written to results.md.
    print(render_markdown(record))

    if baseline_path is not None:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        print("\n".join(compare_runs(baseline, record)) + "\n")

    if not args.no_save:
        run_path = save_run(record, EVAL_RUNS_DIR)
        print(f"Saved run to {run_path}")
        if not filtered and not args.retrieval_only:
            EVAL_RESULTS_FILE.write_text(render_markdown(record), encoding="utf-8")
            print(f"Updated {EVAL_RESULTS_FILE}")

    failed = any(r.error or r.passed is False for r in results)
    failed |= any(
        r.judge is not None and (r.judge.correct != "PASS" or r.judge.grounded != "PASS")
        for r in results
    )
    if args.retrieval_only:
        failed |= any(r.retrieval and r.retrieval.recall < 1 for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
