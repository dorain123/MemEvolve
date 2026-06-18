"""Run the task-1 xBench memory-system comparison.

This helper keeps the original Flash-Searcher entrypoint unchanged while making
the experiment reproducible on this workstation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = ROOT.parents[1] / "SearchSwarm" / "harness" / ".env"

PROVIDER_STORAGE_DIRS = {
    "agent_kb": [ROOT / "storage" / "agent_kb"],
    "lightweight_memory": [ROOT / "storage" / "lightweight_memory"],
    "dynamic_cheatsheet": [ROOT / "storage" / "dynamic_cheatsheet"],
}


def ensure_inside_root(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(ROOT.resolve())
    return resolved


def remove_path(path: Path) -> None:
    resolved = ensure_inside_root(path)
    if not resolved.exists():
        return
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink()


def load_eval_env(env_file: Path) -> dict[str, str]:
    values = dotenv_values(env_file)
    required = ["API_KEY", "API_BASE_URL", "SERPER_API_KEY"]
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise RuntimeError(f"Missing required values in {env_file}: {', '.join(missing)}")

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = values["API_KEY"]
    env["OPENAI_API_BASE"] = values["API_BASE_URL"]
    env["OPENAI_BASE_URL"] = values["API_BASE_URL"]
    env["DEFAULT_MODEL"] = "deepseek-v4-flash"
    env["DEFAULT_JUDGE_MODEL"] = "deepseek-v4-flash"
    env["SERPER_API_KEY"] = values["SERPER_API_KEY"]
    env["WEB_ACCESS_PROVIDER"] = "crawl4ai"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def parse_task_indices(task_indices: str) -> list[int]:
    ids: list[int] = []
    for part in task_indices.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def format_task_indices(task_ids: list[int]) -> str:
    return ",".join(str(task_id) for task_id in sorted(set(task_ids)))


def read_completed_task_ids(outfile: Path) -> set[int]:
    if not outfile.exists():
        return set()

    completed: set[int] = set()
    for line_no, line in enumerate(outfile.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL in {outfile} at line {line_no}: {exc}") from exc
        if row.get("status") == "success" and row.get("task_id") is not None:
            completed.add(int(row["task_id"]))
    return completed


def validate_results(outfile: Path, expected_ids: set[int]) -> list[str]:
    if not outfile.exists():
        return [f"Missing output file: {outfile}"]

    seen: list[int] = []
    all_ids: list[int] = []
    failures: list[str] = []
    for line_no, line in enumerate(outfile.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(f"Line {line_no} is not valid JSON: {exc}")
            continue
        try:
            task_id = int(row.get("task_id"))
        except (TypeError, ValueError):
            failures.append(f"Line {line_no} has invalid task_id: {row.get('task_id')!r}")
            continue
        all_ids.append(task_id)
        if task_id in expected_ids:
            seen.append(task_id)
            if row.get("status") != "success":
                failures.append(f"Task {task_id} status is {row.get('status')!r}")

    seen_set = set(seen)
    missing = sorted(expected_ids - seen_set)
    duplicates = sorted(task_id for task_id in seen_set if seen.count(task_id) > 1)
    unexpected = sorted(set(all_ids) - expected_ids)
    if missing:
        failures.append(f"Missing task ids: {missing}")
    if duplicates:
        failures.append(f"Duplicate task ids: {duplicates}")
    if unexpected:
        failures.append(f"Unexpected task ids: {unexpected}")
    return failures


def run_provider(provider: str, args: argparse.Namespace, env: dict[str, str]) -> int:
    outfile = ROOT / "xbench_output" / f"{args.output_prefix}_{provider}.jsonl"
    task_dir = ROOT / "xbench_output" / f"{args.output_prefix}_{provider}_tasks"
    expected_ids = set(parse_task_indices(args.task_indices))

    if args.fresh:
        for path in PROVIDER_STORAGE_DIRS.get(provider, []):
            remove_path(path)
        remove_path(outfile)
        remove_path(task_dir)

    run_ids = sorted(expected_ids)
    if args.resume:
        completed_ids = read_completed_task_ids(outfile)
        run_ids = sorted(expected_ids - completed_ids)
        if completed_ids:
            print(
                f"{provider}: found {len(completed_ids & expected_ids)} completed tasks, "
                f"{len(run_ids)} remaining.",
                flush=True,
            )

    if not run_ids:
        failures = validate_results(outfile, expected_ids)
        if failures:
            print(f"{provider}: validation failed after resume skip: {failures}", flush=True)
            return 1
        print(f"{provider}: all requested tasks already complete.", flush=True)
        return 0

    cmd = [
        sys.executable,
        "run_flash_searcher_mm_xbench.py",
        "--infile",
        "./data/xbench/DeepSearch.csv",
        "--outfile",
        str(outfile),
        "--memory_provider",
        provider,
        "--task_indices",
        format_task_indices(run_ids),
        "--max_steps",
        str(args.max_steps),
        "--concurrency",
        str(args.concurrency),
        "--judge_model",
        "deepseek-v4-flash",
        "--direct_output_dir",
        str(task_dir),
    ]

    print(f"\n=== Running {provider} ===", flush=True)
    print(" ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
    print(f"=== {provider} exited with {completed.returncode} ===", flush=True)
    if completed.returncode != 0:
        return completed.returncode

    failures = validate_results(outfile, expected_ids)
    if failures:
        print(f"{provider}: validation failed: {failures}", flush=True)
        return 1
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["lightweight_memory", "agent_kb", "dynamic_cheatsheet"],
    )
    parser.add_argument("--task-indices", default="1-20")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--output-prefix", default="formal")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Remove provider storage and prior output before running the requested tasks.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Run all requested task ids even when the output file already contains completed rows.",
    )
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    env = load_eval_env(args.env_file)

    failures = []
    for provider in args.providers:
        code = run_provider(provider, args, env)
        if code != 0:
            failures.append((provider, code))

    if failures:
        print(f"Failures: {failures}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
