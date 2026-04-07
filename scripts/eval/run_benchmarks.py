"""Run lm-evaluation-harness benchmarks on a model.

Runs each task individually and saves results after each task so resumed
execution only pays for missing work.

Canonical usage:
    python scripts/eval/run_benchmarks.py \
        --model-id results/my_run/final_model \
        --suite-label zero_shot \
        --trust-remote-code

Legacy custom output usage:
    python scripts/eval/run_benchmarks.py \
        --model-id Qwen/Qwen3.5-4B-Base \
        --output-dir results/benchmarks/qwen35-4b-base \
        --trust-remote-code
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    SRC_ROOT = REPO_ROOT / "src"
    for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

from mrp.eval_artifacts import (
    atomic_write_json,
    prepare_eval_paths,
    write_eval_manifest,
)


TASKS = [
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "piqa",
    "lambada_openai",
    "truthfulqa_mc1",
    "truthfulqa_mc2",
]

METRIC_PRIORITY = ["acc_norm,none", "acc,none", "mc2,none", "mc1,none"]


def parse_tasks(raw: str | None) -> list[str]:
    if raw is None:
        return list(TASKS)
    return [task.strip() for task in raw.split(",") if task.strip()]


def get_primary_score(task_data: dict[str, Any]) -> tuple[str | None, float | None]:
    for key in METRIC_PRIORITY:
        if key in task_data:
            return key.split(",")[0], task_data[key]
    return None, None


def resolve_output_dir(
    *,
    model_id: str,
    output_dir: str | None,
    run_root: str | None,
    suite_label: str,
) -> tuple[Path, Any | None]:
    if output_dir:
        return Path(output_dir).expanduser().resolve(), None
    paths = prepare_eval_paths(
        suite_id="benchmarks",
        model_ref=model_id,
        run_root=run_root,
        group=suite_label,
    )
    return paths.suite_dir, paths


def _load_existing_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_resume_files(
    *,
    summary_path: Path,
    results_path: Path,
    summary: dict[str, Any],
    all_results: dict[str, Any],
) -> None:
    atomic_write_json(results_path, all_results)
    atomic_write_json(summary_path, summary)


def run_benchmarks(
    *,
    model_id: str,
    tokenizer_id: str | None = None,
    output_dir: str | None = None,
    run_root: str | None = None,
    suite_label: str = "zero_shot",
    tasks: list[str] | None = None,
    batch_size: str = "auto",
    trust_remote_code: bool = False,
    skip_generate: bool = False,
) -> dict[str, Any]:
    selected_tasks = tasks or list(TASKS)
    resolved_output_dir, canonical_paths = resolve_output_dir(
        model_id=model_id,
        output_dir=output_dir,
        run_root=run_root,
        suite_label=suite_label,
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = resolved_output_dir / "benchmark_summary.json"
    results_path = resolved_output_dir / "benchmark_results.json"
    metadata_path = resolved_output_dir / "benchmark_metadata.json"
    summary = _load_existing_json(summary_path)
    all_results = _load_existing_json(results_path)

    if canonical_paths is not None:
        write_eval_manifest(
            paths=canonical_paths,
            model_ref=model_id,
            tokenizer_ref=tokenizer_id or model_id,
            checkpoint_ref=model_id,
            dataset={"suite": suite_label, "framework": "lm_eval"},
            command=" ".join(["python", "scripts/eval/run_benchmarks.py"]),
            status="running",
            source_artifacts=[],
            artifacts=[summary_path, results_path, metadata_path],
            metadata={"tasks": selected_tasks},
        )

    tasks_to_run = []
    for task in selected_tasks:
        if task in summary:
            if summary[task].get("metric") == "error":
                print(
                    "RETRY %s (previous run failed: %s)"
                    % (task, summary[task].get("error", "unknown"))
                )
                del summary[task]
                if task in all_results:
                    del all_results[task]
            else:
                print("SKIP %s (already done: %.4f)" % (task, summary[task]["score"]))
                continue
        if skip_generate and task == "mmlu_pro":
            print("SKIP %s (--skip-generate)" % task)
            continue
        tasks_to_run.append(task)

    if not tasks_to_run:
        print("All tasks complete!")
        return {
            "output_dir": str(resolved_output_dir),
            "tasks_completed": len(summary),
            "tasks_total": len(selected_tasks),
            "suite_label": suite_label,
        }

    import torch
    from mrp.model_loading import load_text_model
    from transformers import AutoTokenizer

    print("\nLoading model: %s" % model_id)
    t0 = time.time()
    loaded = load_text_model(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype="auto",
    )
    model = loaded.model
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("  On GPU: %s" % torch.cuda.get_device_name(0))
    print("  Loaded in %.1fs" % (time.time() - t0))

    tokenizer_name = tokenizer_id or model_id
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
    )

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    for index, task in enumerate(tasks_to_run):
        print("\n=== [%d/%d] Running: %s ===" % (index + 1, len(tasks_to_run), task))
        task_t0 = time.time()
        try:
            result = evaluator.simple_evaluate(
                model=lm,
                tasks=[task],
                batch_size=batch_size,
            )
            task_data = result["results"].get(task, {})
            all_results[task] = task_data

            metric, score = get_primary_score(task_data)
            if metric and score is not None:
                summary[task] = {"metric": metric, "score": round(score, 4)}
                print("  %s = %.4f (%.0fs)" % (metric, score, time.time() - task_t0))
            else:
                print("  No primary metric found (%.0fs)" % (time.time() - task_t0))
        except Exception as exc:
            print("  FAILED: %s (%.0fs)" % (str(exc)[:200], time.time() - task_t0))
            summary[task] = {"metric": "error", "score": 0, "error": str(exc)[:200]}

        _write_resume_files(
            summary_path=summary_path,
            results_path=results_path,
            summary=summary,
            all_results=all_results,
        )
        print("  Saved. (%d/%d tasks complete)" % (len(summary), len(selected_tasks)))

    metadata = {
        "schema_version": 1,
        "suite_id": "benchmarks",
        "suite_label": suite_label,
        "model_id": model_id,
        "tokenizer_id": tokenizer_name,
        "batch_size": batch_size,
        "trust_remote_code": trust_remote_code,
        "tasks": selected_tasks,
        "elapsed_s": round(time.time() - t0, 1),
        "output_dir": str(resolved_output_dir),
    }
    atomic_write_json(metadata_path, metadata)

    if canonical_paths is not None:
        write_eval_manifest(
            paths=canonical_paths,
            model_ref=model_id,
            tokenizer_ref=tokenizer_name,
            checkpoint_ref=model_id,
            dataset={"suite": suite_label, "framework": "lm_eval"},
            command=" ".join(["python", "scripts/eval/run_benchmarks.py"]),
            status="completed",
            source_artifacts=[],
            artifacts=[summary_path, results_path, metadata_path],
            metadata={"tasks": selected_tasks, "completed_tasks": len(summary)},
        )

    elapsed = time.time() - t0
    print("\n=== All Benchmarks Complete ===")
    print("%-20s %10s" % ("Task", "Score"))
    print("-" * 32)
    for task in selected_tasks:
        if task not in summary:
            continue
        entry = summary[task]
        if entry["metric"] != "error":
            print("%-20s %10.4f" % (task, entry["score"]))
        else:
            print("%-20s %10s" % (task, "FAILED"))

    print("\nTotal time: %.0f min" % (elapsed / 60))
    print("Results saved to %s" % resolved_output_dir)
    return {
        "output_dir": str(resolved_output_dir),
        "suite_label": suite_label,
        "tasks_completed": len(summary),
        "tasks_total": len(selected_tasks),
        "elapsed_s": round(elapsed, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmarks via lm-eval (incremental)")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--suite-label", default="zero_shot")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task list. Defaults to the canonical seven-task suite.",
    )
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Deprecated. No generation tasks in current suite.",
    )
    return parser


def main() -> dict[str, Any]:
    args = build_parser().parse_args()
    return run_benchmarks(
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
        output_dir=args.output_dir,
        run_root=args.run_root,
        suite_label=args.suite_label,
        tasks=parse_tasks(args.tasks),
        batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
        skip_generate=args.skip_generate,
    )


if __name__ == "__main__":
    main()
