from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mrp.utils import ensure_dir


SCHEMA_VERSION = 1


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_run_root(path: str | Path | None) -> Path:
    raw = "." if path in (None, "") else path
    resolved = Path(raw).expanduser().resolve()
    if resolved.name == "final_model" or resolved.name.startswith("checkpoint-"):
        return resolved.parent.resolve()
    return resolved


def resolve_model_ref(path: str | Path | None) -> Path | None:
    if path in (None, ""):
        return None
    return Path(path).expanduser().resolve()


def infer_run_root_from_path(path: str | Path | None) -> Path | None:
    if path in (None, ""):
        return None
    resolved = Path(path).expanduser().resolve()
    probe = resolved if resolved.is_dir() else resolved.parent
    for candidate in (probe, *probe.parents):
        if candidate.name == "evals":
            return candidate.parent.resolve()
        if (candidate / ".mrp" / "run.json").exists():
            return candidate.resolve()
        if (
            (candidate / "metrics.jsonl").exists()
            or (candidate / "train_summary.json").exists()
            or (candidate / "phase1_status.json").exists()
        ):
            return candidate.resolve()
    if probe.name == "final_model" or probe.name.startswith("checkpoint-"):
        return probe.parent.resolve()
    return None


def infer_model_label(model_ref: Path | None, run_root: Path) -> str | None:
    if model_ref is None:
        return None
    if model_ref == run_root:
        return None
    try:
        relative = model_ref.relative_to(run_root)
        label = "__".join(relative.parts)
    except ValueError:
        label = model_ref.name or "model"
    return label or None


@dataclass(frozen=True)
class EvalArtifactPaths:
    run_root: Path
    suite_id: str
    suite_parts: tuple[str, ...]
    suite_dir: Path
    index_path: Path
    manifest_path: Path

    def artifact_path(self, filename: str) -> Path:
        return self.suite_dir / filename


def get_eval_paths(
    *,
    run_root: str | Path,
    suite_id: str,
    suite_parts: list[str] | tuple[str, ...] | None = None,
) -> EvalArtifactPaths:
    resolved_run_root = Path(run_root).expanduser().resolve()
    normalized_parts = tuple(suite_parts or (suite_id,))
    suite_dir = resolved_run_root / "evals"
    for part in normalized_parts:
        suite_dir = suite_dir / part
    return EvalArtifactPaths(
        run_root=resolved_run_root,
        suite_id=suite_id,
        suite_parts=normalized_parts,
        suite_dir=suite_dir,
        index_path=resolved_run_root / "evals" / "index.json",
        manifest_path=suite_dir / "eval_manifest.json",
    )


def prepare_eval_paths(
    *,
    suite_id: str,
    model_ref: str | Path | None = None,
    run_root: str | Path | None = None,
    group: str | None = None,
    variant: str | None = None,
) -> EvalArtifactPaths:
    resolved_model_ref = resolve_model_ref(model_ref)
    resolved_run_root = resolve_run_root(run_root or resolved_model_ref or ".")
    suite_parts: list[str] = [suite_id]
    if group:
        suite_parts.append(group)
    if variant:
        suite_parts.append(variant)
    else:
        model_label = infer_model_label(resolved_model_ref, resolved_run_root)
        if model_label:
            suite_parts.append(model_label)
    return get_eval_paths(run_root=resolved_run_root, suite_id=suite_id, suite_parts=suite_parts)


def resolve_suite_output_dir(
    *,
    suite_id: str,
    output_dir: str | Path | None = None,
    model_ref: str | Path | None = None,
    run_root: str | Path | None = None,
    group: str | None = None,
    variant: str | None = None,
) -> tuple[Path, EvalArtifactPaths | None]:
    if output_dir not in (None, ""):
        return Path(output_dir).expanduser().resolve(), None
    paths = prepare_eval_paths(
        suite_id=suite_id,
        model_ref=model_ref,
        run_root=run_root,
        group=group,
        variant=variant,
    )
    return paths.suite_dir, paths


def _load_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _iso_now(),
            "suites": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def update_eval_index(
    *,
    paths: EvalArtifactPaths,
    model_ref: str | Path | None = None,
    tokenizer_ref: str | None = None,
    checkpoint_ref: str | Path | None = None,
    status: str = "completed",
    artifacts: list[Path] | None = None,
) -> None:
    payload = _load_index(paths.index_path)
    suites = payload.setdefault("suites", [])
    artifact_paths = [_relative_to(path, paths.run_root) for path in (artifacts or [])]
    manifest_rel = _relative_to(paths.manifest_path, paths.run_root)
    model_value = str(resolve_model_ref(model_ref)) if model_ref not in (None, "") else None
    checkpoint_value = (
        str(resolve_model_ref(checkpoint_ref))
        if checkpoint_ref not in (None, "")
        else None
    )
    entry = {
        "suite_id": paths.suite_id,
        "suite_parts": list(paths.suite_parts),
        "suite_dir": _relative_to(paths.suite_dir, paths.run_root),
        "manifest_path": manifest_rel,
        "artifacts": artifact_paths,
        "model_ref": model_value,
        "tokenizer_ref": tokenizer_ref,
        "checkpoint_ref": checkpoint_value,
        "status": status,
        "updated_at": _iso_now(),
    }
    suites = [
        item
        for item in suites
        if tuple(item.get("suite_parts") or ()) != paths.suite_parts
    ]
    suites.append(entry)
    suites.sort(key=lambda item: tuple(item.get("suite_parts") or ()))
    payload["schema_version"] = SCHEMA_VERSION
    payload["generated_at"] = _iso_now()
    payload["suites"] = suites
    atomic_write_json(paths.index_path, payload)


def write_eval_manifest(
    *,
    paths: EvalArtifactPaths,
    model_ref: str | Path | None = None,
    tokenizer_ref: str | None = None,
    checkpoint_ref: str | Path | None = None,
    dataset: dict[str, Any] | None = None,
    command: str | None = None,
    status: str = "completed",
    source_artifacts: list[str] | None = None,
    artifacts: list[Path] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": paths.suite_id,
        "suite_parts": list(paths.suite_parts),
        "run_root": str(paths.run_root),
        "suite_dir": str(paths.suite_dir),
        "model_ref": str(resolve_model_ref(model_ref)) if model_ref not in (None, "") else None,
        "tokenizer_ref": tokenizer_ref,
        "checkpoint_ref": (
            str(resolve_model_ref(checkpoint_ref))
            if checkpoint_ref not in (None, "")
            else None
        ),
        "created_at": _iso_now(),
        "dataset": dataset or {},
        "command": command,
        "status": status,
        "source_artifacts": source_artifacts or [],
        "artifacts": [_relative_to(path, paths.run_root) for path in (artifacts or [])],
        "metadata": metadata or {},
    }
    atomic_write_json(paths.manifest_path, payload)
    update_eval_index(
        paths=paths,
        model_ref=model_ref,
        tokenizer_ref=tokenizer_ref,
        checkpoint_ref=checkpoint_ref,
        status=status,
        artifacts=artifacts,
    )
    return payload


def write_eval_json(
    *,
    paths: EvalArtifactPaths,
    filename: str,
    payload: dict[str, Any],
    model_ref: str | Path | None = None,
    tokenizer_ref: str | None = None,
    checkpoint_ref: str | Path | None = None,
    dataset: dict[str, Any] | None = None,
    command: str | None = None,
    status: str = "completed",
    source_artifacts: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    artifact_path = paths.artifact_path(filename)
    atomic_write_json(artifact_path, payload)
    write_eval_manifest(
        paths=paths,
        model_ref=model_ref,
        tokenizer_ref=tokenizer_ref,
        checkpoint_ref=checkpoint_ref,
        dataset=dataset,
        command=command,
        status=status,
        source_artifacts=source_artifacts,
        artifacts=[artifact_path],
        metadata=metadata,
    )
    return artifact_path


def finalize_eval_artifacts(
    *,
    paths: EvalArtifactPaths | None,
    model_ref: str | Path | None = None,
    tokenizer_ref: str | None = None,
    checkpoint_ref: str | Path | None = None,
    dataset: dict[str, Any] | None = None,
    command: str | None = None,
    status: str = "completed",
    source_artifacts: list[str] | None = None,
    artifacts: list[Path] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if paths is None:
        return None
    return write_eval_manifest(
        paths=paths,
        model_ref=model_ref,
        tokenizer_ref=tokenizer_ref,
        checkpoint_ref=checkpoint_ref,
        dataset=dataset,
        command=command,
        status=status,
        source_artifacts=source_artifacts,
        artifacts=artifacts,
        metadata=metadata,
    )
