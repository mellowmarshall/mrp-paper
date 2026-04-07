from __future__ import annotations

import csv
import json
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


def _summarize(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
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
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p25": float(np.quantile(finite, 0.25)),
        "p75": float(np.quantile(finite, 0.75)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(finite.max()),
    }


def _two_nn(distances: np.ndarray) -> tuple[float, np.ndarray]:
    rho1 = distances[:, 0]
    rho2 = distances[:, 1]
    valid = (rho1 > 0.0) & (rho2 > rho1)
    mu = rho2[valid] / rho1[valid]
    if mu.size == 0:
        raise ValueError("TWO-NN estimator found no valid neighbor ratios")

    logs = np.log(mu)
    denominator = float(logs.sum())
    if denominator <= 0.0:
        raise ValueError("TWO-NN denominator is non-positive")
    estimate = float(mu.size / denominator)
    local_proxy = np.full(distances.shape[0], np.nan, dtype=np.float64)
    local_proxy[valid] = 1.0 / logs
    return estimate, local_proxy


def _mle(distances: np.ndarray, *, k1: int, k2: int) -> tuple[float, np.ndarray, dict[str, float]]:
    # distances are k-nearest-neighbor distances excluding the query point itself
    max_available_k = distances.shape[1]
    resolved_k1 = max(3, k1)
    resolved_k2 = min(max_available_k, k2)
    if resolved_k2 < resolved_k1:
        raise ValueError("not enough neighbors for requested MLE range")

    per_k_estimates: dict[str, float] = {}
    per_k_locals: list[np.ndarray] = []

    for k in range(resolved_k1, resolved_k2 + 1):
        tk = distances[:, k - 1]
        prior = distances[:, : k - 1]
        valid = (tk > 0.0) & np.all(prior > 0.0, axis=1)
        if not np.any(valid):
            continue

        logs = np.log(tk[valid, None] / prior[valid])
        mean_logs = logs.mean(axis=1)
        finite = mean_logs > 0.0
        if not np.any(finite):
            continue

        local_full = np.full(distances.shape[0], np.nan, dtype=np.float64)
        valid_indices = np.flatnonzero(valid)
        finite_mean_logs = mean_logs[finite]
        local_k = 1.0 / finite_mean_logs
        local_full[valid_indices[finite]] = local_k
        per_k_locals.append(local_full)
        # Levina-Bickel's global estimator is the inverse of the mean log-ratio,
        # not the mean of the per-point inverses. The latter is much more
        # sensitive to near-tied neighbor shells and exact duplicates.
        per_k_estimates[f"k_{k}"] = float(1.0 / finite_mean_logs.mean())

    if not per_k_locals:
        raise ValueError("MLE estimator found no valid neighborhoods")

    stacked = np.stack(per_k_locals, axis=0)
    finite_counts = np.isfinite(stacked).sum(axis=0)
    summed = np.nansum(stacked, axis=0)
    local_mean = np.divide(
        summed,
        finite_counts,
        out=np.full(distances.shape[0], np.nan, dtype=np.float64),
        where=finite_counts > 0,
    )
    global_mean = float(np.mean(list(per_k_estimates.values())))
    return global_mean, local_mean.astype(np.float64, copy=False), per_k_estimates


def _analyze_layer(
    hidden_states: np.ndarray,
    *,
    layer_index: int,
    mle_k1: int,
    mle_k2: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_points, hidden_size = hidden_states.shape
    if n_points < max(4, mle_k2 + 1):
        raise ValueError(
            f"layer {layer_index} has too few samples ({n_points}) for mle_k2={mle_k2}"
        )

    data = np.asarray(hidden_states, dtype=np.float32)
    _, inverse, counts = np.unique(data, axis=0, return_inverse=True, return_counts=True)
    duplicate_counts = counts[inverse]
    tree = cKDTree(data)
    distances, _ = tree.query(data, k=mle_k2 + 1)
    distances = np.asarray(distances, dtype=np.float64)[:, 1:]

    two_nn_estimate, two_nn_local_proxy = _two_nn(distances[:, :2])
    mle_estimate, mle_local, mle_by_k = _mle(distances, k1=mle_k1, k2=mle_k2)

    utilization_two_nn = float(two_nn_estimate / hidden_size)
    utilization_mle = float(mle_estimate / hidden_size)

    summary = {
        "layer_index": layer_index,
        "n_points": int(n_points),
        "hidden_size": int(hidden_size),
        "duplicate_point_count": int(np.sum(duplicate_counts > 1)),
        "duplicate_group_count": int(np.sum(counts > 1)),
        "largest_duplicate_group": int(counts.max()),
        "two_nn_dimension": float(two_nn_estimate),
        "two_nn_utilization_ratio": utilization_two_nn,
        "two_nn_local_proxy_summary": _summarize(two_nn_local_proxy),
        "mle_dimension": float(mle_estimate),
        "mle_utilization_ratio": utilization_mle,
        "mle_local_summary": _summarize(mle_local),
        "mle_k_range": {"k1": int(mle_k1), "k2": int(mle_k2)},
        "mle_by_k": mle_by_k,
    }

    row_count = int(hidden_states.shape[0])
    rows = [
        {
            "layer_index": layer_index,
            "point_index": point_index,
            "two_nn_local_proxy": (
                None if not np.isfinite(two_nn_local_proxy[point_index]) else float(two_nn_local_proxy[point_index])
            ),
            "mle_local_dimension": (
                None if not np.isfinite(mle_local[point_index]) else float(mle_local[point_index])
            ),
        }
        for point_index in range(row_count)
    ]
    return summary, rows


def analyze_intrinsic_dimension(
    manifest_path: str | Path,
    *,
    output_path: str | Path,
    profile_output_path: str | Path | None = None,
    max_points: int | None = None,
    mle_k1: int = 5,
    mle_k2: int = 12,
    seed: int = 0,
) -> dict[str, Any]:
    if mle_k1 < 3:
        raise ValueError("mle_k1 must be at least 3")
    if mle_k2 < mle_k1:
        raise ValueError("mle_k2 must be at least mle_k1")

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
            mle_k1=mle_k1,
            mle_k2=mle_k2,
        )
        for row in rows:
            row["source_point_index"] = int(sampled_indices[row["point_index"]])
        summary["source_path"] = str(resolved_layer_path)
        summary["sampled_points"] = int(sampled_hidden_states.shape[0])
        summary["source_points"] = int(hidden_states.shape[0])
        layer_summaries.append(summary)
        profile_rows.extend(rows)

    two_nn_dims = np.asarray(
        [layer["two_nn_dimension"] for layer in layer_summaries],
        dtype=np.float64,
    )
    mle_dims = np.asarray(
        [layer["mle_dimension"] for layer in layer_summaries],
        dtype=np.float64,
    )

    summary = {
        "manifest_path": str(Path(manifest_path).resolve()),
        "max_points": None if max_points is None else int(max_points),
        "mle_k_range": {"k1": int(mle_k1), "k2": int(mle_k2)},
        "seed": int(seed),
        "n_layers": int(len(layer_summaries)),
        "layers": layer_summaries,
        "global": {
            "mean_two_nn_dimension_across_layers": float(two_nn_dims.mean()),
            "median_two_nn_dimension_across_layers": float(np.median(two_nn_dims)),
            "peak_two_nn_dimension": float(two_nn_dims.max()),
            "peak_two_nn_layer": int(np.argmax(two_nn_dims)),
            "mean_mle_dimension_across_layers": float(mle_dims.mean()),
            "median_mle_dimension_across_layers": float(np.median(mle_dims)),
            "peak_mle_dimension": float(mle_dims.max()),
            "peak_mle_layer": int(np.argmax(mle_dims)),
        },
        "notes": [
            "TWO-NN uses the Facco et al. ratio estimator k_hat = n / sum(log(r2 / r1)).",
            "MLE uses the Levina-Bickel global neighborhood-ratio estimator averaged across the requested k range.",
            "Exact duplicate hidden states can create zero-distance neighbors and unstable local inverse-log values, so duplicate counts are reported per layer.",
            "Layer-0 anomalies are common because raw embeddings need not satisfy the manifold assumptions as cleanly as deeper hidden states.",
        ],
    }
    write_json(output_path, summary)

    if profile_output_path is not None:
        target = Path(profile_output_path)
        ensure_dir(target.parent)
        fieldnames = [
            "layer_index",
            "point_index",
            "source_point_index",
            "two_nn_local_proxy",
            "mle_local_dimension",
        ]
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(profile_rows)

    return summary
