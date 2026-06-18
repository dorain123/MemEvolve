"""Summarize formal xBench memory-system runs without exposing task text."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROVIDERS = ["lightweight_memory", "agent_kb", "dynamic_cheatsheet"]


@dataclass(frozen=True)
class TaskResult:
    provider: str
    task_id: int
    score: int
    status: str
    extracted_answer: str
    elapsed_time: float
    total_tokens: int
    api_calls: int


def read_results(path: Path, provider: str) -> list[TaskResult]:
    rows: list[TaskResult] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path}:{line_no} is not valid JSONL: {exc}") from exc

        metrics = row.get("metrics") or {}
        rows.append(
            TaskResult(
                provider=provider,
                task_id=int(row["task_id"]),
                score=int(row.get("score", 0)),
                status=str(row.get("status", "")),
                extracted_answer=str(row.get("extracted_answer") or ""),
                elapsed_time=float(metrics.get("elapsed_time", 0)),
                total_tokens=int(metrics.get("total_tokens", 0)),
                api_calls=int(metrics.get("api_calls", 0)),
            )
        )
    return sorted(rows, key=lambda row: row.task_id)


def validate_provider_rows(rows: list[TaskResult], expected_ids: set[int]) -> list[str]:
    seen = [row.task_id for row in rows]
    seen_set = set(seen)
    failures: list[str] = []
    missing = sorted(expected_ids - seen_set)
    duplicates = sorted(task_id for task_id in seen_set if seen.count(task_id) > 1)
    unexpected = sorted(seen_set - expected_ids)
    bad_status = [row.task_id for row in rows if row.task_id in expected_ids and row.status != "success"]
    if missing:
        failures.append(f"missing={missing}")
    if duplicates:
        failures.append(f"duplicates={duplicates}")
    if unexpected:
        failures.append(f"unexpected={unexpected}")
    if bad_status:
        failures.append(f"non_success={bad_status}")
    return failures


def summarize_provider(rows: list[TaskResult]) -> dict[str, float | int | str]:
    total = len(rows)
    correct = sum(row.score == 1 for row in rows)
    total_tokens = sum(row.total_tokens for row in rows)
    total_time = sum(row.elapsed_time for row in rows)
    total_api_calls = sum(row.api_calls for row in rows)
    return {
        "tasks": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / total if total else 0,
        "total_time_min": total_time / 60,
        "avg_time_sec": total_time / total if total else 0,
        "api_calls": total_api_calls,
        "wrong_ids": ", ".join(str(row.task_id) for row in rows if row.score == 0) or "-",
    }


def build_markdown(all_rows: dict[str, list[TaskResult]], expected_ids: set[int]) -> str:
    providers = list(all_rows)
    lines: list[str] = [
        "# xBench Memory Results Summary",
        "",
        "This summary intentionally omits decrypted benchmark prompts and golden answers.",
        "",
        "## Integrity",
        "",
        "| provider | rows | unique_ids | missing | duplicate | status_counts |",
        "|---|---:|---:|---|---|---|",
    ]

    for provider, rows in all_rows.items():
        ids = [row.task_id for row in rows]
        counts = Counter(row.status for row in rows)
        missing = sorted(expected_ids - set(ids))
        duplicates = sorted(task_id for task_id in set(ids) if ids.count(task_id) > 1)
        lines.append(
            f"| {provider} | {len(rows)} | {len(set(ids))} | {missing or '-'} | "
            f"{duplicates or '-'} | {dict(counts)} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| provider | correct/tasks | accuracy | total_tokens | avg_tokens | total_time_min | avg_time_sec | api_calls | wrong_ids |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for provider, rows in all_rows.items():
        summary = summarize_provider(rows)
        lines.append(
            f"| {provider} | {summary['correct']}/{summary['tasks']} | "
            f"{summary['accuracy']:.1%} | {summary['total_tokens']:,} | "
            f"{summary['avg_tokens']:,.0f} | {summary['total_time_min']:.1f} | "
            f"{summary['avg_time_sec']:.1f} | {summary['api_calls']} | {summary['wrong_ids']} |"
        )

    lines.extend(
        [
            "",
            "## Per Task Scores",
            "",
            f"| task_id | {' | '.join(providers)} |",
            f"|---:|{'|'.join('---:' for _ in providers)}|",
        ]
    )
    rows_by_provider = {
        provider: {row.task_id: row for row in rows} for provider, rows in all_rows.items()
    }
    for task_id in sorted(expected_ids):
        values = []
        for provider in providers:
            row = rows_by_provider[provider].get(task_id)
            values.append(str(row.score) if row else "NA")
        lines.append(f"| {task_id} | {' | '.join(values)} |")

    lines.extend(
        [
            "",
            "## Evaluation Flags",
            "",
            "- Empty extracted answers on failed rows should be manually reviewed; they can indicate judge extraction failures rather than agent failures.",
            "- High-token failures should be inspected for retrieval loops or overly broad memory retrieval.",
        ]
    )
    flagged = []
    for provider, rows in all_rows.items():
        for row in rows:
            if row.score == 0 and not row.extracted_answer:
                flagged.append(f"{provider} task {row.task_id}: score=0 with empty extracted_answer")
    if flagged:
        lines.extend(f"- {item}" for item in flagged)
    else:
        lines.append("- No empty-answer failed rows detected.")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="+", default=DEFAULT_PROVIDERS)
    parser.add_argument("--output-prefix", default="formal")
    parser.add_argument("--task-ids", default="1-20")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "xbench_output" / "formal_summary" / "results_summary.md",
    )
    args = parser.parse_args()

    start, end = (int(part) for part in args.task_ids.split("-", 1))
    expected_ids = set(range(start, end + 1))
    all_rows: dict[str, list[TaskResult]] = {}
    validation_failures: list[str] = []

    for provider in args.providers:
        path = ROOT / "xbench_output" / f"{args.output_prefix}_{provider}.jsonl"
        rows = read_results(path, provider)
        failures = validate_provider_rows(rows, expected_ids)
        if failures:
            validation_failures.append(f"{provider}: {', '.join(failures)}")
        all_rows[provider] = rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_markdown(all_rows, expected_ids), encoding="utf-8")
    print(f"Wrote {args.out}")

    if validation_failures:
        print("Validation failures:")
        for failure in validation_failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
