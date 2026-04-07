from __future__ import annotations

import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
import json

from mrp.audit import analyze_margins
from mrp.curvature import analyze_curvature
from mrp.eval_harness import run_lm_eval
from mrp.extract import extract_token_stats
from mrp.intrinsic_dimension import analyze_intrinsic_dimension
from mrp.model_inspection import inspect_model
from mrp.tracker import start_run
from mrp.utils import ensure_dir, write_json


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _paths_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def _missing_paths(paths: list[Path]) -> list[str]:
    return [str(path.resolve()) for path in paths if not path.exists()]


def _update_status(status_path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = _now()
    write_json(status_path, payload)


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_extract_outputs(output_root: Path) -> dict[str, Any]:
    token_stats_path = output_root / "token_stats.csv"
    manifest_path = output_root / "manifest.json"
    hidden_state_dir = output_root / "hidden_states"

    missing_paths = _missing_paths([token_stats_path, manifest_path, hidden_state_dir])
    if missing_paths:
        raise FileNotFoundError(
            "extract_token_stats completed without writing required outputs: "
            + ", ".join(missing_paths)
        )

    manifest = _load_manifest(manifest_path)
    hidden_state_files = [Path(raw).expanduser().resolve() for raw in manifest.get("hidden_state_files", [])]
    if not hidden_state_files:
        raise FileNotFoundError(
            "extract_token_stats manifest does not list any hidden_state_files"
        )
    missing_hidden = [str(path) for path in hidden_state_files if not path.exists()]
    if missing_hidden:
        raise FileNotFoundError(
            "extract_token_stats manifest references missing hidden_state_files: "
            + ", ".join(missing_hidden)
        )
    return manifest


def _run_step(
    *,
    status: dict[str, Any],
    status_path: Path,
    name: str,
    outputs: list[Path],
    force: bool,
    runner: Callable[[], Any],
    validator: Callable[[], Any] | None = None,
    tracker=None,
    tracker_step: int | None = None,
) -> Any:
    status["current_step"] = name
    step_status = status["steps"].setdefault(
        name,
        {"outputs": [str(path.resolve()) for path in outputs]},
    )
    step_status["outputs"] = [str(path.resolve()) for path in outputs]

    if not force and _paths_exist(outputs):
        step_status["state"] = "skipped"
        step_status["finished_at"] = _now()
        _update_status(status_path, status)
        if tracker is not None and tracker_step is not None:
            tracker.log_event(
                "phase1_step_skipped",
                step=tracker_step,
                payload={"name": name, "outputs": [str(path.resolve()) for path in outputs]},
            )
        return None

    step_status["state"] = "running"
    step_status["started_at"] = _now()
    _update_status(status_path, status)
    if tracker is not None and tracker_step is not None:
        tracker.log_event(
            "phase1_step_started",
            step=tracker_step,
            payload={"name": name, "outputs": [str(path.resolve()) for path in outputs]},
        )

    try:
        result = runner()
        if validator is not None:
            validated = validator()
            if validated is not None:
                result = validated
        else:
            missing_outputs = _missing_paths(outputs)
            if missing_outputs:
                raise FileNotFoundError(
                    f"{name} completed without writing declared outputs: "
                    + ", ".join(missing_outputs)
                )
    except Exception as exc:
        step_status["state"] = "failed"
        step_status["finished_at"] = _now()
        step_status["error"] = f"{type(exc).__name__}: {exc}"
        step_status["traceback"] = traceback.format_exc()
        status["state"] = "failed"
        _update_status(status_path, status)
        if tracker is not None and tracker_step is not None:
            tracker.log_event(
                "phase1_step_failed",
                step=tracker_step,
                payload={"name": name, "error": f"{type(exc).__name__}: {exc}"},
            )
        raise

    step_status["state"] = "completed"
    step_status["finished_at"] = _now()
    if isinstance(result, dict):
        step_status["summary"] = {
            key: value
            for key, value in result.items()
            if key
            in {
                "load_strategy",
                "processed_sequences",
                "processed_positions",
                "beta_exponent",
                "r_squared",
                "n_layers",
                "manifest_path",
            }
        }
    _update_status(status_path, status)
    if tracker is not None and tracker_step is not None:
        tracker.log_event(
            "phase1_step_completed",
            step=tracker_step,
            payload={"name": name, "summary": step_status.get("summary", {})},
        )
        tracker.log_metrics(
            step=tracker_step,
            values={
                f"phase1/{name}/{key}": float(value)
                for key, value in step_status.get("summary", {}).items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            },
            source="phase1",
        )
        for output_path in outputs:
            if output_path.exists():
                if output_path.suffix.lower() in {".csv"}:
                    tracker.log_table(
                        f"{name}:{output_path.name}",
                        output_path,
                        format=output_path.suffix.lstrip("."),
                    )
                else:
                    tracker.log_artifact(
                        f"{name}:{output_path.name}",
                        output_path,
                        kind="analysis",
                    )
    return result


def run_phase1(
    *,
    model_id: str,
    output_dir: str | Path,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str | None,
    max_sequences: int | None,
    max_length: int,
    reservoir_size: int,
    top_k: int,
    hidden_state_dtype: str,
    device: str,
    trust_remote_code: bool,
    seed: int,
    curvature_neighbor_count: int,
    curvature_variance_threshold: float,
    curvature_plane_neighbors: int,
    curvature_max_points: int | None,
    mle_k1: int,
    mle_k2: int,
    intrinsic_max_points: int | None,
    eval_tasks: str | None,
    eval_limit: float | None,
    eval_batch_size: str,
    eval_device: str,
    force: bool,
) -> dict[str, Any]:
    output_root = ensure_dir(output_dir)
    tracker = start_run(
        output_dir=output_root,
        name=output_root.name,
        run_type="phase1",
        config={
            "model_id": model_id,
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "split": split,
            "max_sequences": max_sequences,
            "max_length": max_length,
            "reservoir_size": reservoir_size,
            "top_k": top_k,
            "hidden_state_dtype": hidden_state_dtype,
            "device": device,
            "curvature_neighbor_count": curvature_neighbor_count,
            "curvature_variance_threshold": curvature_variance_threshold,
            "curvature_plane_neighbors": curvature_plane_neighbors,
            "curvature_max_points": curvature_max_points,
            "mle_k1": mle_k1,
            "mle_k2": mle_k2,
            "intrinsic_max_points": intrinsic_max_points,
            "eval_tasks": eval_tasks,
            "eval_limit": eval_limit,
        },
    )
    status_path = output_root / "phase1_status.json"
    status: dict[str, Any] = {
        "state": "running",
        "started_at": _now(),
        "updated_at": _now(),
        "current_step": None,
        "model_id": model_id,
        "output_dir": str(output_root.resolve()),
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "max_sequences": max_sequences,
        "max_length": max_length,
        "reservoir_size": reservoir_size,
        "steps": {},
    }
    _update_status(status_path, status)

    inspection_path = output_root / "model_inspection.json"
    manifest_path = output_root / "manifest.json"
    token_stats_path = output_root / "token_stats.csv"
    margin_audit_path = output_root / "margin_audit.json"
    margin_curve_path = output_root / "margin_curve.csv"
    curvature_analysis_path = output_root / "curvature_analysis.json"
    curvature_profile_path = output_root / "curvature_profile.csv"
    intrinsic_analysis_path = output_root / "intrinsic_dimension_analysis.json"
    intrinsic_profile_path = output_root / "intrinsic_dimension_profile.csv"

    try:
        _run_step(
            status=status,
            status_path=status_path,
            name="inspect_model",
            outputs=[inspection_path],
            force=force,
            tracker=tracker,
            tracker_step=1,
            runner=lambda: write_json(
                inspection_path,
                inspect_model(
                    model_id,
                    trust_remote_code=trust_remote_code,
                    load_weights=True,
                ),
            ),
        )

        _run_step(
            status=status,
            status_path=status_path,
            name="extract_token_stats",
            outputs=[token_stats_path, manifest_path],
            force=force,
            validator=lambda: _validate_extract_outputs(output_root),
            tracker=tracker,
            tracker_step=2,
            runner=lambda: extract_token_stats(
                model_id=model_id,
                output_dir=output_root,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split=split,
                max_sequences=max_sequences,
                max_length=max_length,
                reservoir_size=reservoir_size,
                top_k=top_k,
                trust_remote_code=trust_remote_code,
                text_column=text_column,
                device=device,
                hidden_state_dtype=hidden_state_dtype,
                seed=seed,
            ),
        )

        _run_step(
            status=status,
            status_path=status_path,
            name="analyze_margins",
            outputs=[margin_audit_path, margin_curve_path],
            force=force,
            tracker=tracker,
            tracker_step=3,
            runner=lambda: analyze_margins(
                token_stats_path,
                output_path=margin_audit_path,
                curve_output_path=margin_curve_path,
            ),
        )

        _run_step(
            status=status,
            status_path=status_path,
            name="analyze_curvature",
            outputs=[curvature_analysis_path, curvature_profile_path],
            force=force,
            tracker=tracker,
            tracker_step=4,
            runner=lambda: analyze_curvature(
                manifest_path,
                output_path=curvature_analysis_path,
                profile_output_path=curvature_profile_path,
                neighbor_count=curvature_neighbor_count,
                variance_threshold=curvature_variance_threshold,
                plane_neighbors=curvature_plane_neighbors,
                max_points=curvature_max_points,
                seed=seed,
            ),
        )

        _run_step(
            status=status,
            status_path=status_path,
            name="analyze_intrinsic_dimension",
            outputs=[intrinsic_analysis_path, intrinsic_profile_path],
            force=force,
            tracker=tracker,
            tracker_step=5,
            runner=lambda: analyze_intrinsic_dimension(
                manifest_path,
                output_path=intrinsic_analysis_path,
                profile_output_path=intrinsic_profile_path,
                max_points=intrinsic_max_points,
                mle_k1=mle_k1,
                mle_k2=mle_k2,
                seed=seed,
            ),
        )

        if eval_tasks:
            eval_slug = eval_tasks.replace(",", "_").replace("/", "_")
            eval_path = output_root / f"lm_eval_{eval_slug}.json"
            _run_step(
                status=status,
                status_path=status_path,
                name="run_lm_eval",
                outputs=[eval_path],
                force=force,
                tracker=tracker,
                tracker_step=6,
                runner=lambda: run_lm_eval(
                    model_id=model_id,
                    tasks=eval_tasks,
                    output_path=eval_path,
                    batch_size=eval_batch_size,
                    device=eval_device,
                    limit=eval_limit,
                    trust_remote_code=trust_remote_code,
                ),
            )
    except Exception as exc:
        if status_path.exists():
            tracker.log_artifact("phase1_status", status_path, kind="status")
        tracker.finish(
            status="failed",
            summary={"error": f"{type(exc).__name__}: {exc}", "status_path": str(status_path.resolve())},
        )
        raise

    status["state"] = "completed"
    status["current_step"] = None
    status["finished_at"] = _now()
    _update_status(status_path, status)
    if status_path.exists():
        tracker.log_artifact("phase1_status", status_path, kind="status")
    tracker.finish(status="completed", summary=status)
    return status
