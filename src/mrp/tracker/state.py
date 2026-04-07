from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ID = "local-mrp"
DEFAULT_EXPERIMENT_ID = "all-runs"
SCRATCH_WORKSPACE_ID = "scratch"
WORKSPACE_PRESETS = (
    "accuracy_loss",
    "margin_distribution",
    "dimensional_analysis",
    "representation_dynamics",
    "prediction_entropy",
    "layerwise_prediction_quality",
    "position_level_comparisons",
    "confused_pairs",
    "mrp_intervention_effectiveness",
    "training_spot_checks",
    "perplexity",
    "calibration",
    "generation_quality",
    "tokenization_efficiency",
    "robustness",
    "speed_efficiency",
    "few_shot_in_context_learning",
)
LEGACY_PRESET_ALIASES = {
    "overview": "accuracy_loss",
    "training_overview": "accuracy_loss",
    "phase1_inspection": "margin_distribution",
    "cyclic_analysis": "dimensional_analysis",
    "intervention_compare": "mrp_intervention_effectiveness",
    "checkpoint_deep_dive": "representation_dynamics",
}
DEFAULT_COLUMNS = (
    "name",
    "run_type",
    "status",
    "started_at",
    "finished_at",
    "parent_run_id",
    "fork_step",
)
DEFAULT_FILTER = {
    "search": "",
    "run_types": [],
    "statuses": [],
    "tags": [],
    "lineage_root_id": None,
    "include_forks": True,
}
DEFAULT_WORKSPACE_STATE = {
    "focus_run_id": None,
    "compare_run_ids": [],
    "reference_run_id": None,
    "selected_metrics": {},
    "visible_columns": list(DEFAULT_COLUMNS),
    "group_by": "none",
    "artifact_tree_mode": "logical",
    "show_only_differences": False,
    "step_stride_n": 0,
}


class TrackerStateError(RuntimeError):
    pass


def _now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    normalized = normalized.strip("-")
    return normalized or "item"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrackerStateError(f"invalid JSON in {path}") from exc
    if not isinstance(payload, dict):
        raise TrackerStateError(f"expected object payload in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _deepcopy_default(payload: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(payload)


def _normalize_workspace_state(payload: dict[str, Any] | None) -> dict[str, Any]:
    state = _deepcopy_default(DEFAULT_WORKSPACE_STATE)
    if payload:
        state.update(payload)
    return state


def default_project_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": PROJECT_ID,
        "name": "Local MRP",
        "default_experiment_id": DEFAULT_EXPERIMENT_ID,
        "default_workspace_id": None,
    }


def default_experiment_payload(*, experiment_id: str, name: str, kind: str = "saved") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": experiment_id,
        "name": name,
        "description": "",
        "kind": kind,
        "filter": _deepcopy_default(DEFAULT_FILTER),
        "sort": "finished_at_desc",
        "group_by": "none",
        "pinned_run_ids": [],
        "updated_at": _now_iso(),
    }


def default_workspace_payload(
    *,
    workspace_id: str,
    name: str,
    experiment_id: str,
    preset_id: str = "accuracy_loss",
) -> dict[str, Any]:
    timestamp = _now_iso()
    return {
        "schema_version": 1,
        "id": workspace_id,
        "name": name,
        "experiment_id": experiment_id,
        "preset_id": preset_id,
        "default_tab": "runs",
        "state": _deepcopy_default(DEFAULT_WORKSPACE_STATE),
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def _project_display_name(project_id: str) -> str:
    return project_id.replace("-", " ").replace("_", " ").title()


def build_builtin_experiments(
    run_type_counts: dict[str, int],
    project_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    experiments = [
        {
            **default_experiment_payload(experiment_id=DEFAULT_EXPERIMENT_ID, name="All Runs", kind="auto"),
            "description": "Every tracked run in the local project.",
            "run_count": sum(run_type_counts.values()),
        }
    ]
    for run_type in sorted(run_type_counts):
        experiments.append(
            {
                **default_experiment_payload(
                    experiment_id=f"run-type-{_slugify(run_type)}",
                    name=f"{run_type.title()} Runs",
                    kind="auto",
                ),
                "description": f"All runs with run_type={run_type}.",
                "filter": {
                    **_deepcopy_default(DEFAULT_FILTER),
                    "run_types": [run_type],
                },
                "run_count": run_type_counts.get(run_type, 0),
            }
        )
    # Group by project (primary organization)
    for project_id in sorted(project_counts or {}):
        experiments.append(
            {
                **default_experiment_payload(
                    experiment_id=f"project-{_slugify(project_id)}",
                    name=_project_display_name(project_id),
                    kind="auto",
                ),
                "description": f"All runs in project {project_id}.",
                "filter": {
                    **_deepcopy_default(DEFAULT_FILTER),
                    "projects": [project_id],
                },
                "run_count": (project_counts or {}).get(project_id, 0),
            }
        )
    return experiments


class TrackerStateStore:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir.resolve()
        self.project_file = self.state_dir / "project.json"
        self.experiments_dir = self.state_dir / "experiments"
        self.workspaces_dir = self.state_dir / "workspaces"

    def _normalize_preset_id(self, preset_id: str | None) -> str:
        resolved = LEGACY_PRESET_ALIASES.get(str(preset_id or ""), str(preset_id or ""))
        return resolved if resolved in WORKSPACE_PRESETS else "accuracy_loss"

    def ensure_layout(self) -> None:
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)
        if not self.project_file.exists():
            _write_json(self.project_file, default_project_payload())

    def load_project(self) -> dict[str, Any]:
        self.ensure_layout()
        payload = default_project_payload()
        payload.update(_read_json(self.project_file))
        return payload

    def save_project(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_layout()
        current = self.load_project()
        current.update(payload)
        current["schema_version"] = 1
        _write_json(self.project_file, current)
        return current

    def list_saved_experiments(self) -> list[dict[str, Any]]:
        self.ensure_layout()
        rows: list[dict[str, Any]] = []
        for path in sorted(self.experiments_dir.glob("*.json")):
            payload = default_experiment_payload(experiment_id=path.stem, name=path.stem)
            payload.update(_read_json(path))
            rows.append(payload)
        return rows

    def get_saved_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        path = self.experiments_dir / f"{experiment_id}.json"
        if not path.exists():
            return None
        payload = default_experiment_payload(experiment_id=experiment_id, name=experiment_id)
        payload.update(_read_json(path))
        return payload

    def _unique_id(self, directory: Path, preferred_id: str) -> str:
        if not (directory / f"{preferred_id}.json").exists():
            return preferred_id
        index = 2
        while (directory / f"{preferred_id}-{index}.json").exists():
            index += 1
        return f"{preferred_id}-{index}"

    def save_experiment(self, payload: dict[str, Any], *, replace: bool = False) -> dict[str, Any]:
        self.ensure_layout()
        experiment_id = _slugify(str(payload.get("id") or payload.get("name") or "experiment"))
        if not replace:
            experiment_id = self._unique_id(self.experiments_dir, experiment_id)
        current = default_experiment_payload(
            experiment_id=experiment_id,
            name=str(payload.get("name") or experiment_id),
        )
        if replace:
            existing = self.get_saved_experiment(experiment_id)
            if existing is not None:
                current.update(existing)
        current.update(payload)
        current["id"] = experiment_id
        current["kind"] = "saved"
        current["schema_version"] = 1
        current["updated_at"] = _now_iso()
        _write_json(self.experiments_dir / f"{experiment_id}.json", current)
        return current

    def delete_experiment(self, experiment_id: str) -> None:
        if not re.fullmatch(r"[a-z0-9-]+", experiment_id):
            raise TrackerStateError(f"invalid experiment id: {experiment_id!r}")
        path = self.experiments_dir / f"{experiment_id}.json"
        if path.exists():
            path.unlink()

    def list_saved_workspaces(self, *, experiment_id: str | None = None) -> list[dict[str, Any]]:
        self.ensure_layout()
        rows: list[dict[str, Any]] = []
        for path in sorted(self.workspaces_dir.glob("*.json")):
            payload = default_workspace_payload(
                workspace_id=path.stem,
                name=path.stem,
                experiment_id=experiment_id or DEFAULT_EXPERIMENT_ID,
            )
            payload.update(_read_json(path))
            payload["preset_id"] = self._normalize_preset_id(payload.get("preset_id"))
            payload["state"] = _normalize_workspace_state(payload.get("state"))
            if experiment_id is None or payload.get("experiment_id") == experiment_id:
                rows.append(payload)
        return rows

    def get_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        path = self.workspaces_dir / f"{workspace_id}.json"
        if not path.exists():
            return None
        payload = default_workspace_payload(
            workspace_id=workspace_id,
            name=workspace_id,
            experiment_id=DEFAULT_EXPERIMENT_ID,
        )
        payload.update(_read_json(path))
        payload["preset_id"] = self._normalize_preset_id(payload.get("preset_id"))
        payload["state"] = _normalize_workspace_state(payload.get("state"))
        return payload

    def save_workspace(self, payload: dict[str, Any], *, replace: bool = False) -> dict[str, Any]:
        self.ensure_layout()
        workspace_id = _slugify(str(payload.get("id") or payload.get("name") or "workspace"))
        if not replace:
            workspace_id = self._unique_id(self.workspaces_dir, workspace_id)
        current = default_workspace_payload(
            workspace_id=workspace_id,
            name=str(payload.get("name") or workspace_id),
            experiment_id=str(payload.get("experiment_id") or DEFAULT_EXPERIMENT_ID),
            preset_id=self._normalize_preset_id(str(payload.get("preset_id") or "accuracy_loss")),
        )
        if replace:
            existing = self.get_workspace(workspace_id)
            if existing is not None:
                current.update(existing)
        current.update(payload)
        current["id"] = workspace_id
        current["preset_id"] = self._normalize_preset_id(current.get("preset_id"))
        current["state"] = _normalize_workspace_state(current.get("state"))
        current["schema_version"] = 1
        current["updated_at"] = _now_iso()
        if "created_at" not in current:
            current["created_at"] = current["updated_at"]
        _write_json(self.workspaces_dir / f"{workspace_id}.json", current)
        return current

    def delete_workspace(self, workspace_id: str) -> None:
        if not re.fullmatch(r"[a-z0-9-]+", workspace_id):
            raise TrackerStateError(f"invalid workspace id: {workspace_id!r}")
        path = self.workspaces_dir / f"{workspace_id}.json"
        if path.exists():
            path.unlink()
