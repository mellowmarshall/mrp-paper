from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    SRC_ROOT = REPO_ROOT / "src"
    for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

from mrp.eval_artifacts import prepare_eval_paths, resolve_run_root, write_eval_json
from mrp.eval_suites import (
    DEFAULT_FEWSHOT_TASKS,
    compute_calibration_eval,
    compute_fewshot_eval,
    compute_layer_prediction_quality,
    compute_perplexity_eval,
    compute_speed_eval,
    compute_tokenization_eval,
    load_eval_context,
)
from mrp.quick_eval import quick_eval
from scripts.eval.run_benchmarks import parse_tasks, run_benchmarks


SUITE_FILENAMES = {
    "perplexity": "perplexity_eval.json",
    "calibration": "calibration_eval.json",
    "tokenization": "tokenization_eval.json",
    "layer_prediction_quality": "layer_prediction_quality.json",
    "speed": "speed_eval.json",
    "fewshot": "fewshot_eval.json",
}


def _parse_suites(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _suite_exists(
    *,
    suite_id: str,
    filename: str,
    model_id: str,
    run_root: str | None,
    group: str | None = None,
) -> Path:
    paths = prepare_eval_paths(
        suite_id=suite_id,
        model_ref=model_id,
        run_root=run_root,
        group=group,
    )
    return paths.artifact_path(filename)


def _maybe_skip(path: Path, *, force: bool, skip_existing: bool) -> bool:
    if force:
        return False
    return skip_existing and path.exists()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run canonical eval suites into <run>/evals/")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument(
        "--suites",
        default="quick_eval,perplexity,calibration,tokenization",
        help="Comma-separated suites. Options: quick_eval, benchmarks, fewshot, perplexity, calibration, tokenization, layer_prediction_quality, speed",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-sequences", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--benchmark-suite-label", default="zero_shot")
    parser.add_argument(
        "--benchmark-tasks",
        default=None,
        help="Optional comma-separated benchmark task override for the benchmarks suite.",
    )
    parser.add_argument("--fewshot-tasks", default=",".join(DEFAULT_FEWSHOT_TASKS))
    parser.add_argument("--fewshot-shots", default="0,5")
    parser.add_argument("--eval-limit", type=float, default=None)
    return parser


def main() -> dict[str, Any]:
    args = build_parser().parse_args()
    suites = _parse_suites(args.suites)
    resolved_run_root = resolve_run_root(args.run_root or args.model_id)
    results: dict[str, Any] = {
        "run_root": str(resolved_run_root),
        "suites_requested": suites,
        "completed": {},
        "skipped": {},
    }

    needs_context = any(
        suite in {
            "perplexity",
            "calibration",
            "tokenization",
            "layer_prediction_quality",
            "speed",
        }
        for suite in suites
    )
    context: dict[str, Any] | None = None
    if needs_context:
        context = load_eval_context(
            model_id=args.model_id,
            tokenizer_id=args.tokenizer_id,
            device=args.device,
            n_sequences=args.n_sequences,
            max_length=args.max_length,
            trust_remote_code=args.trust_remote_code,
        )

    if "quick_eval" in suites:
        target = _suite_exists(
            suite_id="quick_eval",
            filename="quick_eval.json",
            model_id=args.model_id,
            run_root=str(resolved_run_root),
        )
        if _maybe_skip(target, force=args.force, skip_existing=args.skip_existing):
            results["skipped"]["quick_eval"] = str(target)
        else:
            quick_eval(
                checkpoints=[args.model_id],
                n_sequences=args.n_sequences,
                max_length=args.max_length,
                device=args.device,
                tokenizer_id=args.tokenizer_id or args.model_id,
                json_output=True,
                run_root=str(resolved_run_root),
            )
            results["completed"]["quick_eval"] = str(target)

    if "benchmarks" in suites:
        benchmark_dir = prepare_eval_paths(
            suite_id="benchmarks",
            model_ref=args.model_id,
            run_root=str(resolved_run_root),
            group=args.benchmark_suite_label,
        ).suite_dir
        if _maybe_skip(
            benchmark_dir / "benchmark_summary.json",
            force=args.force,
            skip_existing=args.skip_existing,
        ):
            results["skipped"]["benchmarks"] = str(benchmark_dir)
        else:
            run_benchmarks(
                model_id=args.model_id,
                tokenizer_id=args.tokenizer_id,
                output_dir=None,
                run_root=str(resolved_run_root),
                suite_label=args.benchmark_suite_label,
                tasks=parse_tasks(args.benchmark_tasks),
                batch_size=args.batch_size,
                trust_remote_code=args.trust_remote_code,
            )
            results["completed"]["benchmarks"] = str(benchmark_dir)

    if "fewshot" in suites:
        target = _suite_exists(
            suite_id="fewshot",
            filename="fewshot_eval.json",
            model_id=args.model_id,
            run_root=str(resolved_run_root),
        )
        if _maybe_skip(target, force=args.force, skip_existing=args.skip_existing):
            results["skipped"]["fewshot"] = str(target)
        else:
            payload = compute_fewshot_eval(
                model_id=args.model_id,
                tokenizer_id=args.tokenizer_id,
                tasks=_parse_suites(args.fewshot_tasks),
                shots=[int(value) for value in _parse_suites(args.fewshot_shots)],
                batch_size=args.batch_size,
                device=args.device,
                limit=args.eval_limit,
                trust_remote_code=args.trust_remote_code,
            )
            paths = prepare_eval_paths(
                suite_id="fewshot",
                model_ref=args.model_id,
                run_root=str(resolved_run_root),
            )
            out_path = write_eval_json(
                paths=paths,
                filename="fewshot_eval.json",
                payload=payload,
                model_ref=args.model_id,
                tokenizer_ref=args.tokenizer_id or args.model_id,
                checkpoint_ref=args.model_id,
                dataset={"suite": "fewshot", "tasks": _parse_suites(args.fewshot_tasks)},
                command=" ".join(sys.argv),
                metadata={"shots": [int(value) for value in _parse_suites(args.fewshot_shots)]},
            )
            results["completed"]["fewshot"] = str(out_path)

    analytic_builders = {
        "perplexity": (
            compute_perplexity_eval,
            "perplexity_eval.json",
        ),
        "calibration": (
            compute_calibration_eval,
            "calibration_eval.json",
        ),
        "tokenization": (
            compute_tokenization_eval,
            "tokenization_eval.json",
        ),
        "layer_prediction_quality": (
            compute_layer_prediction_quality,
            "layer_prediction_quality.json",
        ),
        "speed": (
            compute_speed_eval,
            "speed_eval.json",
        ),
    }

    for suite_id, (builder, filename) in analytic_builders.items():
        if suite_id not in suites:
            continue
        target = _suite_exists(
            suite_id=suite_id,
            filename=filename,
            model_id=args.model_id,
            run_root=str(resolved_run_root),
        )
        if _maybe_skip(target, force=args.force, skip_existing=args.skip_existing):
            results["skipped"][suite_id] = str(target)
            continue
        if context is None:
            raise RuntimeError(f"{suite_id} requires an evaluation context")
        kwargs = {
            "model": context["model"],
            "sequences": context["sequences"],
            "device": context["runtime_device"],
            "dataset": context["dataset"],
            "max_length": context["max_length"],
        }
        if suite_id == "tokenization":
            kwargs["texts"] = context["texts"]
        payload = builder(**kwargs)
        paths = prepare_eval_paths(
            suite_id=suite_id,
            model_ref=args.model_id,
            run_root=str(resolved_run_root),
        )
        out_path = write_eval_json(
            paths=paths,
            filename=filename,
            payload=payload,
            model_ref=args.model_id,
            tokenizer_ref=args.tokenizer_id or args.model_id,
            checkpoint_ref=args.model_id,
            dataset=context["dataset"],
            command=" ".join(sys.argv),
            metadata={
                "n_sequences": context["n_sequences"],
                "max_length": context["max_length"],
            },
        )
        results["completed"][suite_id] = str(out_path)

    return results


if __name__ == "__main__":
    main()
