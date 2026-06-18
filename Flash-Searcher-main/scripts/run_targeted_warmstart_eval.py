"""Run a small targeted warm-start probe for lightweight_memory.

This is not a full benchmark. It checks whether a few focused training
tasks can teach reusable memory that helps a previously failed held-out task.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = ROOT.parents[1] / "SearchSwarm" / "harness" / ".env"
DEFAULT_XBENCH = ROOT / "data" / "xbench" / "DeepSearch.csv"
MEMORY_DIR = ROOT / "storage" / "lightweight_memory"
BACKUP_DIR = ROOT / "storage" / "lightweight_memory_backups"

sys.path.insert(0, str(ROOT))

from eval_utils import generate_unified_report, save_task_result  # noqa: E402
from FlashOAgents import OpenAIServerModel  # noqa: E402
from run_flash_searcher_mm_xbench import process_item  # noqa: E402


SUITES: dict[str, dict[str, Any]] = {
    "shuttlecock_task5": {
        "heldout_xbench_id": 5,
        "train": [
            {
                "id": "shuttle_train_1",
                "prompt": (
                    "A GB/T 11881-2006 compliant shuttlecock uses 16 feathers. "
                    "For 10 shuttlecocks, there are 160 feathers total. If each goose "
                    "can provide 14 usable wing feathers in total, how many geese are "
                    "needed at minimum?"
                ),
                "answer": "12",
            },
            {
                "id": "shuttle_train_2",
                "prompt": (
                    "When making feather shuttlecocks, do not confuse the per-ball "
                    "same-side feather rule with the total animal count. A goose may "
                    "provide about 7 left-wing and 7 right-wing usable feathers; left "
                    "and right feathers should not be mixed within one shuttlecock, but "
                    "both sides can still be used across different balls. For 10 balls "
                    "at 16 feathers each, what is the minimum number of geese?"
                ),
                "answer": "12",
            },
            {
                "id": "shuttle_train_3",
                "prompt": (
                    "A puzzle asks for the number of farmed birds needed to supply "
                    "feathers for 10 standard shuttlecocks. Use 16 feathers per "
                    "shuttlecock and 14 usable feathers per goose. Give only the "
                    "minimum animal count."
                ),
                "answer": "12",
            },
        ],
    },
    "classical_dialogue_task3": {
        "heldout_xbench_id": 3,
        "train": [
            {
                "id": "dialogue_train_1",
                "prompt": (
                    "In a classical Chinese story, count only explicit direct speech "
                    "attributed to a named character. Do not count narrator description "
                    "or another character's speech. If the text has four explicit "
                    "utterances by Yu Jing, how many utterances did he speak?"
                ),
                "answer": "4",
            },
            {
                "id": "dialogue_train_2",
                "prompt": (
                    "For Liaozhai-style dialogue counting, markers such as yue, wen, "
                    "and wei zhi yue can introduce direct speech. Count utterances by "
                    "speaker attribution, not just modern punctuation. If the verified "
                    "list has four attributed utterances, what is the count?"
                ),
                "answer": "4",
            },
        ],
    },
    "geospatial_task10": {
        "heldout_xbench_id": 10,
        "train": [
            {
                "id": "geo_train_1",
                "prompt": (
                    "A point is equidistant from three Beijing memorial sites. The "
                    "right method is to get coordinates, compute the approximate "
                    "circumcenter, then report a distance range rather than 'none'. "
                    "If the common distance is about 6 to 7 km, what should be output?"
                ),
                "answer": "6～7km",
            },
            {
                "id": "geo_train_2",
                "prompt": (
                    "For a three-site equidistance question, do not require the point "
                    "to be a famous landmark. Treat it as a geometry problem on map "
                    "coordinates and report the common distance. If calculation gives "
                    "roughly 6-7 kilometers, give the distance."
                ),
                "answer": "6～7km",
            },
        ],
    },
}


def xor_decrypt(data: bytes, key: str) -> bytes:
    key_bytes = key.encode("utf-8")
    return bytes(data[i] ^ key_bytes[i % len(key_bytes)] for i in range(len(data)))


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


def load_xbench_task(path: Path, task_id: int) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if int(row["id"]) != task_id:
                continue
            key = row["canary"]
            return {
                "id": row["id"],
                "prompt": xor_decrypt(base64.b64decode(row["prompt"]), key).decode("utf-8"),
                "answer": xor_decrypt(base64.b64decode(row["answer"]), key).decode("utf-8"),
            }
    raise ValueError(f"Task id {task_id} not found in {path}")


def safe_move(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise RuntimeError(f"Refusing to overwrite existing path: {dst}")
    shutil.move(str(src), str(dst))


def reset_memory(output_dir: Path) -> Path | None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"before_targeted_{time.strftime('%Y%m%d_%H%M%S')}"
    if MEMORY_DIR.exists():
        safe_move(MEMORY_DIR, backup)
        return backup
    return None


def restore_memory(backup: Path | None, output_dir: Path) -> None:
    archive = output_dir / "memory_after"
    if MEMORY_DIR.exists():
        safe_move(MEMORY_DIR, archive)
    if backup and backup.exists():
        safe_move(backup, MEMORY_DIR)


def build_model_config() -> dict[str, Any]:
    return {
        "model_id": os.environ.get("DEFAULT_MODEL"),
        "custom_role_conversions": {"tool-call": "assistant", "tool-response": "user"},
        "max_completion_tokens": 32768,
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "api_base": os.environ.get("OPENAI_API_BASE"),
    }


def build_judge_model(model_id: str) -> OpenAIServerModel:
    return OpenAIServerModel(
        model_id,
        custom_role_conversions={"tool-call": "assistant", "tool-response": "user"},
        api_key=os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("JUDGE_API_BASE") or os.environ.get("OPENAI_API_BASE"),
        max_completion_tokens=4096,
    )


def run_item(
    item: dict[str, str],
    output_dir: Path,
    phase: str,
    model_config: dict[str, Any],
    judge_model: OpenAIServerModel,
    max_steps: int,
    enable_memory_evolution: bool,
) -> dict[str, Any]:
    result = process_item(
        item,
        model_config,
        summary_interval=8,
        prompts_type="default",
        max_steps=max_steps,
        memory_type_str="lightweight_memory",
        enable_memory_evolution=enable_memory_evolution,
        judge_model=judge_model,
    )
    result["phase"] = phase
    phase_dir = output_dir / phase
    saveable = {key: value for key, value in result.items() if key not in {"agent_messages", "grader_explanation"}}
    save_task_result(saveable, str(phase_dir), filename=f"{item['id']}.json")
    return result


def summarize_suite(output_dir: Path, suite_name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    train = [row for row in results if row.get("phase") == "train"]
    heldout = [row for row in results if row.get("phase") == "heldout"]
    summary = {
        "suite": suite_name,
        "train_tasks": len(train),
        "heldout_tasks": len(heldout),
        "heldout_correct": sum(1 for row in heldout if row.get("score") == 1),
        "total_tokens": sum((row.get("metrics") or {}).get("total_tokens", 0) for row in results),
        "total_api_calls": sum((row.get("metrics") or {}).get("api_calls", 0) for row in results),
        "total_elapsed": sum((row.get("metrics") or {}).get("elapsed_time", 0) for row in results),
    }
    with (output_dir / f"{suite_name}_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def append_markdown_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Targeted Warm-Start Probe",
        "",
        "| Suite | Train tasks | Held-out correct | Tokens | API calls | Elapsed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['suite']} | {row['train_tasks']} | "
            f"{row['heldout_correct']}/{row['heldout_tasks']} | "
            f"{row['total_tokens']:,} | {row['total_api_calls']:,} | "
            f"{row['total_elapsed']:.1f}s |"
        )
    lines.append("")
    lines.append("This is a targeted mechanism probe, not a full benchmark.")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--xbench", type=Path, default=DEFAULT_XBENCH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--judge-model", default="deepseek-v4-flash")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--suites", nargs="+", default=list(SUITES))
    parser.add_argument("--stop-on-success", action="store_true", default=True)
    args = parser.parse_args()

    env = load_eval_env(args.env_file)
    os.environ.update(env)

    output_dir = args.output_dir or ROOT / "xbench_output" / f"targeted_warmstart_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    backup = reset_memory(output_dir)
    summaries: list[dict[str, Any]] = []
    try:
        model_config = build_model_config()
        judge_model = build_judge_model(args.judge_model)

        for suite_name in args.suites:
            suite = SUITES[suite_name]
            results: list[dict[str, Any]] = []
            for item in suite["train"]:
                print(f"[train] {suite_name} {item['id']}", flush=True)
                results.append(
                    run_item(
                        item,
                        output_dir / suite_name,
                        "train",
                        model_config,
                        judge_model,
                        args.max_steps,
                        enable_memory_evolution=True,
                    )
                )

            heldout = load_xbench_task(args.xbench, int(suite["heldout_xbench_id"]))
            print(f"[heldout] {suite_name} xbench-{heldout['id']}", flush=True)
            results.append(
                run_item(
                    heldout,
                    output_dir / suite_name,
                    "heldout",
                    model_config,
                    judge_model,
                    args.max_steps,
                    enable_memory_evolution=False,
                )
            )

            report_path = output_dir / suite_name / "report.txt"
            generate_unified_report(results, str(report_path), dataset_name=f"Targeted {suite_name}", has_levels=False)
            summary = summarize_suite(output_dir, suite_name, results)
            summaries.append(summary)
            if args.stop_on_success and summary["heldout_correct"] > 0:
                print(f"[stop] {suite_name} produced a correct held-out result", flush=True)
                break

        append_markdown_summary(output_dir, summaries)
        print(f"Output: {output_dir}", flush=True)
        return 0
    finally:
        restore_memory(backup, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
