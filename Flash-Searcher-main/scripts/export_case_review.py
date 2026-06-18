"""Export side-by-side case reviews for selected xBench task ids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROVIDERS = ["lightweight_memory", "agent_kb", "dynamic_cheatsheet"]
DEFAULT_TASK_IDS = ["5", "10", "16", "17", "19", "20"]


def load_case(provider: str, task_id: str, output_prefix: str) -> dict | None:
    path = ROOT / "xbench_output" / f"{output_prefix}_{provider}.jsonl"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("task_id")) == task_id:
            return row
    return None


def clipped(value: object, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n...[truncated]..."


def build_case_markdown(task_id: str, providers: list[str], output_prefix: str) -> str:
    lines = [f"# xBench Case Review: Task {task_id}", ""]
    for provider in providers:
        row = load_case(provider, task_id, output_prefix)
        lines.extend([f"## {provider}", ""])
        if row is None:
            lines.extend(["Result not found.", ""])
            continue

        metrics = row.get("metrics") or {}
        lines.extend(
            [
                f"- score: `{row.get('score')}`",
                f"- status: `{row.get('status')}`",
                f"- extracted_answer: `{row.get('extracted_answer')}`",
                f"- golden_answer: `{row.get('golden_answer')}`",
                f"- elapsed_time_sec: `{metrics.get('elapsed_time', 0):.1f}`",
                f"- total_tokens: `{metrics.get('total_tokens', 0)}`",
                f"- api_calls: `{metrics.get('api_calls', 0)}`",
                "",
                "### Question",
                "",
                clipped(row.get("question"), 2000),
                "",
                "### Agent Result",
                "",
                clipped(row.get("agent_result"), 5000),
                "",
                "### Trajectory Excerpt",
                "",
                clipped(row.get("agent_trajectory"), 9000),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-ids", nargs="+", default=DEFAULT_TASK_IDS)
    parser.add_argument("--providers", nargs="+", default=DEFAULT_PROVIDERS)
    parser.add_argument("--output-prefix", default="formal")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "xbench_output" / "case_reviews")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for task_id in args.task_ids:
        out_path = args.out_dir / f"case_task_{task_id}.md"
        out_path.write_text(
            build_case_markdown(task_id, args.providers, args.output_prefix),
            encoding="utf-8",
        )
        print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
