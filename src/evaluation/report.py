"""Aggregate case results into a summary, a saved run record and a markdown report."""

import json
import logging
import math
import subprocess
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.runner import CaseResult

logger = logging.getLogger(__name__)


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[index]


def _mean(values: list[float]) -> float | None:
    """Arithmetic mean rounded to 3 places; None for an empty list."""
    return round(sum(values) / len(values), 3) if values else None


def _retrieval_summary(results: list[CaseResult]) -> dict:
    """Average the retrieval scores over cases that have expected sources."""
    scored = [r.retrieval for r in results if r.retrieval is not None]
    return {
        "cases": len(scored),
        "hit_rate": _mean([1.0 if s.hit else 0.0 for s in scored]),
        "recall": _mean([s.recall for s in scored]),
        "mrr": _mean([s.reciprocal_rank for s in scored]),
        "precision": _mean([s.precision for s in scored]),
    }


def summarize(results: list[CaseResult]) -> dict:
    """Compute every headline metric for a run."""
    generated = [r for r in results if r.answer is not None]
    judged = [r for r in results if r.judge is not None]
    should_refuse = [r for r in generated if r.expects_refusal]
    answerable = [r for r in generated if not r.expects_refusal]

    by_category: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)

    categories = {}
    for category, items in sorted(by_category.items()):
        recalls = [r.retrieval.recall for r in items if r.retrieval is not None]
        categories[category] = {
            "cases": len(items),
            "passed": sum(1 for r in items if r.passed),
            "judge_correct": sum(1 for r in items if r.judge and r.judge.correct == "PASS"),
            "judge_grounded": sum(1 for r in items if r.judge and r.judge.grounded == "PASS"),
            "judged": sum(1 for r in items if r.judge),
            "recall": _mean(recalls),
        }

    retrieve_ms = [r.retrieve_ms for r in results if r.retrieve_ms is not None]
    generate_ms = [r.generate_ms for r in results if r.generate_ms is not None]

    return {
        "cases": len(results),
        "errors": sum(1 for r in results if r.error),
        "answers": {
            "passed": sum(1 for r in generated if r.passed),
            "total": len(generated),
        },
        "retrieval": _retrieval_summary(results),
        "refusal": {
            "correct_refusals": sum(1 for r in should_refuse if r.refused),
            "should_refuse": len(should_refuse),
            "false_refusals": sum(1 for r in answerable if r.refused),
            "answerable": len(answerable),
        },
        "judge": {
            "correct": sum(1 for r in judged if r.judge.correct == "PASS"),
            "grounded": sum(1 for r in judged if r.judge.grounded == "PASS"),
            "errors": sum(1 for r in judged if r.judge.correct == "ERROR"),
            "total": len(judged),
        },
        "latency_ms": {
            "retrieve_p50": percentile(retrieve_ms, 50),
            "retrieve_p95": percentile(retrieve_ms, 95),
            "generate_p50": percentile(generate_ms, 50),
            "generate_p95": percentile(generate_ms, 95),
        },
        "tokens": {
            "input": sum(r.input_tokens for r in results),
            "output": sum(r.output_tokens for r in results),
        },
        "by_category": categories,
    }


def git_state() -> dict:
    """Current commit and whether the tree has uncommitted changes.

    Returns nulls outside a git checkout: the run is still valid, just less
    traceable, so this is logged rather than raised.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("could not read git state: %s", exc)
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def case_to_dict(result: CaseResult) -> dict:
    """Serialise a case, including the derived pass flags."""
    data = asdict(result)
    data["passed"] = result.passed
    data["refusal_ok"] = result.refusal_ok
    return data


def build_run_record(results: list[CaseResult], config: dict, label: str | None) -> dict:
    """Bundle config, summary and per-case detail into one reproducible record."""
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + (f"-{label}" if label else "")
    return {
        "run_id": run_id,
        "created_at": now.isoformat(timespec="seconds"),
        "label": label,
        "config": config | {"git": git_state()},
        "summary": summarize(results),
        "cases": [case_to_dict(r) for r in results],
    }


def save_run(record: dict, runs_dir: Path) -> Path:
    """Write the run record as JSON and return its path."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{record['run_id']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def _ratio(passed: int, total: int) -> str:
    """Format "passed / total (pct%)", or an em dash when nothing was measured."""
    if not total:
        return "—"
    return f"{passed} / {total} ({100 * passed / total:.0f}%)"


def _num(value: float | None, digits: int = 2) -> str:
    """Format a metric, or an em dash when it wasn't measured."""
    return "—" if value is None else f"{value:.{digits}f}"


def _ms(value: float | None) -> str:
    """Format milliseconds as seconds for readability."""
    return "—" if value is None else f"{value / 1000:.2f} s"


def headline_metrics(summary: dict) -> dict[str, float | None]:
    """The comparable numbers of a run as fractions (0–1), for diffing runs."""

    def frac(passed: int, total: int) -> float | None:
        return round(passed / total, 3) if total else None

    s = summary
    return {
        "answer checks": frac(s["answers"]["passed"], s["answers"]["total"]),
        "judge: correct": frac(s["judge"]["correct"], s["judge"]["total"]),
        "judge: grounded": frac(s["judge"]["grounded"], s["judge"]["total"]),
        "retrieval hit rate": s["retrieval"]["hit_rate"],
        "retrieval recall": s["retrieval"]["recall"],
        "retrieval MRR": s["retrieval"]["mrr"],
        "context precision": s["retrieval"]["precision"],
        "correct refusals": frac(s["refusal"]["correct_refusals"], s["refusal"]["should_refuse"]),
        "false refusals": frac(s["refusal"]["false_refusals"], s["refusal"]["answerable"]),
    }


def _failure_lines(record: dict) -> list[str]:
    """One bullet per case that failed any check."""
    lines: list[str] = []
    for case in record["cases"]:
        problems: list[str] = []
        if case["error"]:
            problems.append(f"error: {case['error']}")
        if case["keyword"] and not case["keyword"]["passed"]:
            problems.extend(case["keyword"]["failures"])
        if case["refusal_ok"] is False:
            problems.append("refused" if case["refused"] else "did not refuse")
        if case["retrieval"] and case["retrieval"]["missing_sources"]:
            problems.append(f"not retrieved: {', '.join(case['retrieval']['missing_sources'])}")
        judge = case["judge"]
        if judge and judge["correct"] != "PASS":
            problems.append(f"judge correct {judge['correct']}: {judge['correct_reason']}")
        if judge and judge["grounded"] != "PASS":
            problems.append(f"judge grounded {judge['grounded']}: {judge['grounded_reason']}")
        if problems:
            lines.append(f"- `{case['id']}` ({case['category']}): " + "; ".join(problems))
    return lines


def render_markdown(record: dict) -> str:
    """Render a run record as the human-readable evals/results.md."""
    s = record["summary"]
    c = record["config"]
    git = c["git"]
    commit = git["commit"] or "unknown"
    if git["dirty"]:
        commit += " (uncommitted changes)"
    k = c["top_k"]

    lines = [
        "# Eval results",
        "",
        f"Run `{record['run_id']}` · {record['created_at']} · commit {commit}",
        "",
        f"Generator `{c['generation_model']}` · judge `{c['judge_model'] or 'not run'}` · "
        f"embeddings `{c['embedding_model']}` · top-k {k} · "
        f"chunks ~{c['chunk_target_tokens']} tokens · dataset `{c['dataset_sha']}` · "
        f"mode {c['mode']}",
        "",
        "Regenerated by `uv run python eval_cli.py`. Full per-case detail is in "
        f"`evals/runs/{record['run_id']}.json` on the machine that ran it.",
        "",
        "## Headline",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Cases | {s['cases']} ({s['errors']} errors) |",
        f"| Answer checks (keywords + refusal) | {_ratio(s['answers']['passed'], s['answers']['total'])} |",
        f"| Judge: correct | {_ratio(s['judge']['correct'], s['judge']['total'])} |",
        f"| Judge: grounded | {_ratio(s['judge']['grounded'], s['judge']['total'])} |",
        f"| Retrieval hit rate @{k} | {_num(s['retrieval']['hit_rate'])} |",
        f"| Retrieval recall @{k} | {_num(s['retrieval']['recall'])} |",
        f"| Retrieval MRR | {_num(s['retrieval']['mrr'])} |",
        f"| Context precision @{k} | {_num(s['retrieval']['precision'])} |",
        f"| Correct refusals | {_ratio(s['refusal']['correct_refusals'], s['refusal']['should_refuse'])} |",
        f"| False refusals | {_ratio(s['refusal']['false_refusals'], s['refusal']['answerable'])} |",
        f"| Retrieve latency p50 / p95 | {_ms(s['latency_ms']['retrieve_p50'])} / {_ms(s['latency_ms']['retrieve_p95'])} |",
        f"| Generate latency p50 / p95 | {_ms(s['latency_ms']['generate_p50'])} / {_ms(s['latency_ms']['generate_p95'])} |",
        f"| Tokens in / out | {s['tokens']['input']:,} / {s['tokens']['output']:,} |",
        "",
        "## By category",
        "",
        "| Category | Cases | Answer checks | Judge correct | Judge grounded | Recall |",
        "|---|---|---|---|---|---|",
    ]
    for category, row in s["by_category"].items():
        judged = row["judged"]
        lines.append(
            f"| {category} | {row['cases']} | {row['passed']} | "
            f"{row['judge_correct'] if judged else '—'} | "
            f"{row['judge_grounded'] if judged else '—'} | {_num(row['recall'])} |"
        )

    failures = _failure_lines(record)
    lines += ["", "## Failures", ""]
    lines += failures or ["None."]
    return "\n".join(lines) + "\n"


def compare_runs(baseline: dict, current: dict) -> list[str]:
    """Describe how current differs from baseline: metric deltas and flipped cases."""
    lines = [f"Compared with {baseline['run_id']}:"]
    before = headline_metrics(baseline["summary"])
    after = headline_metrics(current["summary"])
    for name, new in after.items():
        old = before.get(name)
        if old is None or new is None:
            continue
        delta = new - old
        marker = "  " if abs(delta) < 1e-9 else ("+ " if delta > 0 else "- ")
        lines.append(f"  {marker}{name:<20} {old:.2f} → {new:.2f}")

    old_cases = {c["id"]: c for c in baseline["cases"]}
    for case in current["cases"]:
        old = old_cases.get(case["id"])
        if old is None:
            lines.append(f"  new case: {case['id']}")
            continue
        if old["passed"] is not None and case["passed"] is not None and old["passed"] != case["passed"]:
            lines.append(f"  {case['id']}: {'FAIL → PASS' if case['passed'] else 'PASS → FAIL'}")
        old_hit = old["retrieval"]["hit"] if old["retrieval"] else None
        new_hit = case["retrieval"]["hit"] if case["retrieval"] else None
        if old_hit is not None and new_hit is not None and old_hit != new_hit:
            lines.append(f"  {case['id']}: retrieval {'miss → hit' if new_hit else 'hit → miss'}")
    return lines


def list_runs(runs_dir: Path) -> list[dict]:
    """Saved runs, newest first, each with its config and headline metrics."""
    runs: list[dict] = []
    for path in sorted(runs_dir.glob("*.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            config = record["config"]
            runs.append(
                {
                    "run_id": record["run_id"],
                    "created_at": record["created_at"],
                    "label": record["label"],
                    "mode": config["mode"],
                    "top_k": config["top_k"],
                    "generation_model": config["generation_model"],
                    "judge_model": config["judge_model"],
                    "filtered": any(config.get("filters", {}).values()),
                    "cases": record["summary"]["cases"],
                    "headline": headline_metrics(record["summary"]),
                }
            )
        except (json.JSONDecodeError, KeyError) as exc:
            # One unreadable file shouldn't hide every other run; it's logged
            # and left out of the list.
            logger.warning("skipping unreadable run file %s: %r", path.name, exc)
    return runs


def load_run(runs_dir: Path, run_id: str) -> dict | None:
    """Load one run by id; None if no such run file exists."""
    known = {p.stem: p for p in runs_dir.glob("*.json")}
    path = known.get(run_id)
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))
