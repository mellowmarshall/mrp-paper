from __future__ import annotations

import json
import mimetypes
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mrp.utils import ensure_dir, write_json


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _stable_run_id(output_dir: Path) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(output_dir.resolve())))


def _jsonl_append(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    encoded = json.dumps(payload, sort_keys=True) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)


@dataclass
class RunHandle:
    output_dir: Path
    project: str
    name: str
    run_type: str
    run_id: str
    parent_run_id: str | None = None
    fork_step: int | None = None
    config: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    status: str = "running"
    summary: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.meta_dir = ensure_dir(self.output_dir / ".mrp")
        self.artifact_dir = ensure_dir(self.meta_dir / "artifacts")
        self.run_json_path = self.meta_dir / "run.json"
        self.events_path = self.meta_dir / "events.jsonl"
        self._write_run_metadata()

    def _write_run_metadata(self) -> None:
        write_json(
            self.run_json_path,
            {
                "id": self.run_id,
                "name": self.name,
                "project": self.project,
                "run_type": self.run_type,
                "status": self.status,
                "source_path": str(self.output_dir.resolve()),
                "parent_run_id": self.parent_run_id,
                "fork_step": self.fork_step,
                "config": self.config,
                "summary": self.summary,
                "tags": self.tags,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "updated_at": _now(),
            },
        )

    def _append(self, event_type: str, payload: dict[str, Any]) -> None:
        _jsonl_append(
            self.events_path,
            {
                "event_type": event_type,
                "run_id": self.run_id,
                "timestamp": _now(),
                **payload,
            },
            self._lock,
        )

    def log_metric(
        self,
        key: str,
        value: float,
        *,
        step: int,
        timestamp: str | None = None,
        source: str = "sdk",
        checkpoint_id: str | None = None,
    ) -> None:
        self.log_metrics(
            step=step,
            values={key: value},
            timestamp=timestamp,
            source=source,
            checkpoint_id=checkpoint_id,
        )

    def log_metrics(
        self,
        *,
        step: int,
        values: dict[str, float | int],
        timestamp: str | None = None,
        source: str = "sdk",
        checkpoint_id: str | None = None,
    ) -> None:
        numeric_values = {
            key: float(value)
            for key, value in values.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if not numeric_values:
            return
        self._append(
            "metrics",
            {
                "step": int(step),
                "values": numeric_values,
                "source": source,
                "checkpoint_id": checkpoint_id,
                "metric_timestamp": timestamp or _now(),
            },
        )

    def log_event(
        self,
        name: str,
        *,
        payload: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> None:
        self._append(
            "event",
            {
                "name": name,
                "step": step,
                "payload": payload or {},
            },
        )

    def log_checkpoint(
        self,
        path: str | Path,
        *,
        step: int,
        label: str | None = None,
        metrics: dict[str, Any] | None = None,
        parent_checkpoint_id: str | None = None,
        checkpoint_id: str | None = None,
    ) -> str:
        resolved_path = Path(path).resolve()
        actual_id = checkpoint_id or str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{self.run_id}:{resolved_path}:{step}")
        )
        self._append(
            "checkpoint",
            {
                "id": actual_id,
                "step": int(step),
                "label": label or resolved_path.name,
                "path": str(resolved_path),
                "metrics": metrics or {},
                "parent_checkpoint_id": parent_checkpoint_id,
            },
        )
        return actual_id

    def log_artifact(
        self,
        key: str,
        path: str | Path,
        *,
        kind: str | None = None,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        copy: bool = False,
    ) -> str:
        source_path = Path(path).resolve()
        artifact_path = source_path
        if copy and source_path.exists():
            artifact_path = self.artifact_dir / source_path.name
            shutil.copy2(source_path, artifact_path)
        mime, _ = mimetypes.guess_type(str(artifact_path))
        artifact_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{self.run_id}:{key}:{artifact_path}:{checkpoint_id or ''}",
            )
        )
        self._append(
            "artifact",
            {
                "id": artifact_id,
                "artifact_key": key,
                "kind": kind or "file",
                "path": str(artifact_path),
                "checkpoint_id": checkpoint_id,
                "mime": mime or "application/octet-stream",
                "size_bytes": artifact_path.stat().st_size if artifact_path.exists() else 0,
                "metadata": metadata or {},
            },
        )
        return artifact_id

    def log_table(
        self,
        name: str,
        path: str | Path,
        *,
        format: str | None = None,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        resolved_path = Path(path).resolve()
        table_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{self.run_id}:{name}:{resolved_path}:{checkpoint_id or ''}",
            )
        )
        self._append(
            "table",
            {
                "id": table_id,
                "name": name,
                "path": str(resolved_path),
                "format": format or resolved_path.suffix.lstrip(".") or "unknown",
                "checkpoint_id": checkpoint_id,
                "metadata": metadata or {},
            },
        )
        return table_id

    def finish(
        self,
        *,
        status: str = "completed",
        summary: dict[str, Any] | None = None,
    ) -> None:
        self.status = status
        self.summary = summary or self.summary
        self.finished_at = _now()
        self._write_run_metadata()
        self._append(
            "status",
            {
                "status": self.status,
                "summary": self.summary,
                "finished_at": self.finished_at,
            },
        )


def start_run(
    *,
    output_dir: str | Path,
    name: str | None = None,
    project: str = "default",
    run_type: str = "generic",
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    parent_run_id: str | None = None,
    fork_step: int | None = None,
) -> RunHandle:
    resolved_output_dir = ensure_dir(output_dir).resolve()
    return RunHandle(
        output_dir=resolved_output_dir,
        project=project,
        name=name or resolved_output_dir.name,
        run_type=run_type,
        run_id=_stable_run_id(resolved_output_dir),
        parent_run_id=parent_run_id,
        fork_step=fork_step,
        config=config or {},
        tags=tags or [],
    )


def fork_run(
    *,
    output_dir: str | Path,
    parent_run_id: str,
    fork_step: int,
    name: str | None = None,
    project: str = "default",
    run_type: str = "fork",
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> RunHandle:
    return start_run(
        output_dir=output_dir,
        name=name,
        project=project,
        run_type=run_type,
        config=config,
        tags=tags,
        parent_run_id=parent_run_id,
        fork_step=fork_step,
    )
