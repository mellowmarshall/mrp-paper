from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from mrp.utils import ensure_dir, write_json


def _resolve_path(base_dir: Path, candidate: str | Path) -> Path:
    path = Path(candidate)
    if path.is_absolute() or path.exists():
        return path
    return (base_dir / path).resolve()


def _load_manifest(manifest_path: str | Path) -> tuple[dict[str, Any], Path]:
    manifest_file = Path(manifest_path).resolve()
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    return payload, manifest_file.parent


def _sample_points(
    hidden_states: np.ndarray,
    *,
    max_points: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_points = hidden_states.shape[0]
    if max_points is None or n_points <= max_points:
        indices = np.arange(n_points, dtype=np.int64)
        return hidden_states, indices

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_points, size=max_points, replace=False))
    return hidden_states[indices], indices


def _neighborhood_basis(
    points: np.ndarray,
    neighborhood_indices: np.ndarray,
    tangent_dim: int,
) -> np.ndarray:
    neighborhood = points[neighborhood_indices]
    centered = neighborhood - neighborhood.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    max_rank = int(np.count_nonzero(singular_values > 1e-12))
    resolved_dim = max(1, min(tangent_dim, max_rank if max_rank else 1, vt.shape[0]))
    return vt[:resolved_dim].T.astype(np.float32, copy=False)


def _summarize(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }

    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _analyze_layer(
    hidden_states: np.ndarray,
    *,
    layer_index: int,
    neighbor_count: int,
    variance_threshold: float,
    plane_neighbors: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_points, hidden_size = hidden_states.shape
    if n_points < 4:
        raise ValueError(f"layer {layer_index} has too few samples: {n_points}")

    resolved_neighbors = min(max(2, neighbor_count), n_points - 1)
    resolved_plane_neighbors = min(max(1, plane_neighbors), resolved_neighbors)

    data = np.asarray(hidden_states, dtype=np.float32)
    tree = cKDTree(data)
    distances, indices = tree.query(data, k=resolved_neighbors + 1)
    distances = np.asarray(distances, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    neighbor_indices = indices[:, 1:]
    neighbor_distances = distances[:, 1:]

    pca_curvature = np.zeros(n_points, dtype=np.float32)
    local_dims = np.ones(n_points, dtype=np.int64)

    for point_index in range(n_points):
        neighborhood_idx = np.concatenate(
            (
                np.asarray([point_index], dtype=np.int64),
                neighbor_indices[point_index],
            )
        )
        neighborhood = data[neighborhood_idx]
        centered = neighborhood - neighborhood.mean(axis=0, keepdims=True)
        _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        variances = np.square(singular_values, dtype=np.float64)
        total_variance = float(variances.sum())
        if total_variance <= 0.0:
            pca_curvature[point_index] = 0.0
            local_dims[point_index] = 1
            continue

        cumulative = np.cumsum(variances) / total_variance
        resolved_dim = int(np.searchsorted(cumulative, variance_threshold, side="left")) + 1
        resolved_dim = max(1, min(resolved_dim, variances.size))
        local_dims[point_index] = resolved_dim
        pca_curvature[point_index] = float(variances[resolved_dim:].sum() / total_variance)

    layer_tangent_dim = int(np.median(local_dims))
    layer_tangent_dim = max(1, min(layer_tangent_dim, resolved_neighbors))

    bases = [
        _neighborhood_basis(
            data,
            np.concatenate(
                (
                    np.asarray([point_index], dtype=np.int64),
                    neighbor_indices[point_index],
                )
            ),
            layer_tangent_dim,
        )
        for point_index in range(n_points)
    ]

    ii_proxy = np.zeros(n_points, dtype=np.float32)
    eps = 1e-6
    for point_index in range(n_points):
        basis_i = bases[point_index]
        q_i = basis_i.shape[1]
        neighbor_scores: list[float] = []

        for neighbor_slot in range(resolved_plane_neighbors):
            neighbor_index = int(neighbor_indices[point_index, neighbor_slot])
            distance_ij = float(neighbor_distances[point_index, neighbor_slot])
            if distance_ij <= eps:
                continue

            basis_j = bases[neighbor_index]
            shared_dim = min(q_i, basis_j.shape[1])
            if shared_dim <= 0:
                continue

            overlap = basis_i[:, :shared_dim].T @ basis_j[:, :shared_dim]
            singular_values = np.linalg.svd(overlap, compute_uv=False)
            singular_values = np.clip(singular_values, -1.0, 1.0)
            principal_angles = np.arccos(singular_values)
            angle_rms = float(np.linalg.norm(principal_angles) / math.sqrt(shared_dim))
            neighbor_scores.append(angle_rms / distance_ij)

        ii_proxy[point_index] = float(np.mean(neighbor_scores)) if neighbor_scores else 0.0

    summary = {
        "layer_index": layer_index,
        "n_points": int(n_points),
        "hidden_size": int(hidden_size),
        "neighbor_count": int(resolved_neighbors),
        "plane_neighbors": int(resolved_plane_neighbors),
        "local_tangent_dim_summary": _summarize(local_dims.astype(np.float64)),
        "layer_tangent_dim_proxy": int(layer_tangent_dim),
        "pca_curvature_summary": _summarize(pca_curvature.astype(np.float64)),
        "second_fundamental_form_proxy_summary": _summarize(ii_proxy.astype(np.float64)),
    }

    rows = [
        {
            "layer_index": layer_index,
            "point_index": point_index,
            "local_tangent_dim": int(local_dims[point_index]),
            "pca_curvature": float(pca_curvature[point_index]),
            "second_fundamental_form_proxy": float(ii_proxy[point_index]),
        }
        for point_index in range(n_points)
    ]
    return summary, rows


def analyze_curvature(
    manifest_path: str | Path,
    *,
    output_path: str | Path,
    profile_output_path: str | Path | None = None,
    neighbor_count: int = 16,
    variance_threshold: float = 0.95,
    plane_neighbors: int = 4,
    max_points: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    if not 0.5 < variance_threshold < 1.0:
        raise ValueError("variance_threshold must lie in (0.5, 1.0)")

    manifest, manifest_dir = _load_manifest(manifest_path)
    hidden_state_files = manifest.get("hidden_state_files", [])
    if not hidden_state_files:
        raise ValueError("manifest does not contain hidden_state_files")

    layer_summaries: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    for layer_path in hidden_state_files:
        resolved_layer_path = _resolve_path(manifest_dir, layer_path)
        with np.load(resolved_layer_path) as payload:
            hidden_states = np.asarray(payload["hidden_states"], dtype=np.float32)

        sampled_hidden_states, sampled_indices = _sample_points(
            hidden_states,
            max_points=max_points,
            seed=seed + len(layer_summaries),
        )
        summary, rows = _analyze_layer(
            sampled_hidden_states,
            layer_index=len(layer_summaries),
            neighbor_count=neighbor_count,
            variance_threshold=variance_threshold,
            plane_neighbors=plane_neighbors,
        )
        for row in rows:
            row["source_point_index"] = int(sampled_indices[row["point_index"]])
        summary["source_path"] = str(resolved_layer_path)
        summary["sampled_points"] = int(sampled_hidden_states.shape[0])
        summary["source_points"] = int(hidden_states.shape[0])
        summary["sampled_index_min"] = int(sampled_indices.min()) if sampled_indices.size else 0
        summary["sampled_index_max"] = int(sampled_indices.max()) if sampled_indices.size else 0
        layer_summaries.append(summary)
        profile_rows.extend(rows)

    pca_layer_means = np.asarray(
        [layer["pca_curvature_summary"]["mean"] for layer in layer_summaries],
        dtype=np.float64,
    )
    ii_layer_means = np.asarray(
        [
            layer["second_fundamental_form_proxy_summary"]["mean"]
            for layer in layer_summaries
        ],
        dtype=np.float64,
    )

    summary = {
        "manifest_path": str(Path(manifest_path).resolve()),
        "neighbor_count": int(neighbor_count),
        "variance_threshold": float(variance_threshold),
        "plane_neighbors": int(plane_neighbors),
        "max_points": None if max_points is None else int(max_points),
        "seed": int(seed),
        "n_layers": int(len(layer_summaries)),
        "layers": layer_summaries,
        "global": {
            "mean_pca_curvature_across_layers": float(pca_layer_means.mean()),
            "median_pca_curvature_across_layers": float(np.median(pca_layer_means)),
            "mean_second_fundamental_form_proxy_across_layers": float(ii_layer_means.mean()),
            "median_second_fundamental_form_proxy_across_layers": float(
                np.median(ii_layer_means)
            ),
        },
        "notes": [
            "PCA curvature is the residual local variance outside the dominant principal subspace.",
            "Second-fundamental-form values here are a tangent-plane-rotation proxy, not an exact differential-geometry estimate.",
            "Hidden states are sampled from the extraction reservoir, so this measures manifold geometry on the sampled token subset rather than the full corpus.",
        ],
    }
    write_json(output_path, summary)

    if profile_output_path is not None:
        profile_target = Path(profile_output_path)
        ensure_dir(profile_target.parent)
        fieldnames = [
            "layer_index",
            "point_index",
            "source_point_index",
            "local_tangent_dim",
            "pca_curvature",
            "second_fundamental_form_proxy",
        ]
        with profile_target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(profile_rows)

    return summary
