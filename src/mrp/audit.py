from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from mrp.utils import ensure_dir, write_json


def _load_margin_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    margins: list[float] = []
    entropies: list[float] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            margins.append(float(row["margin"]))
            entropies.append(float(row["entropy"]))
    return np.asarray(margins, dtype=np.float64), np.asarray(entropies, dtype=np.float64)


def _fit_gap_curve(
    margins: np.ndarray,
    *,
    epsilon_min: float | None,
    epsilon_max: float | None,
    num_points: int,
) -> dict[str, Any]:
    positive_margins = margins[margins > 0]
    if positive_margins.size == 0:
        raise ValueError("all recorded margins are non-positive")

    default_min = max(float(np.quantile(positive_margins, 0.01)), 1e-6)
    default_max = float(np.quantile(margins, 0.25))
    resolved_min = epsilon_min if epsilon_min is not None else default_min
    resolved_max = epsilon_max if epsilon_max is not None else default_max
    if resolved_min <= 0 or resolved_max <= 0:
        raise ValueError("epsilon bounds must be positive")
    if resolved_min >= resolved_max:
        raise ValueError("epsilon_min must be smaller than epsilon_max")

    epsilons = np.logspace(np.log10(resolved_min), np.log10(resolved_max), num_points)
    eta_hat = np.asarray([(margins < epsilon).mean() for epsilon in epsilons])
    fit_mask = (eta_hat > 0.0) & (eta_hat < 1.0)
    if fit_mask.sum() < 2:
        raise ValueError("not enough non-degenerate epsilon values for regression")

    x = np.log10(epsilons[fit_mask])
    y = np.log10(eta_hat[fit_mask])
    design = np.column_stack([x, np.ones_like(x)])
    beta, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    predictions = beta * x + intercept
    residual_sum = float(np.square(y - predictions).sum())
    total_sum = float(np.square(y - y.mean()).sum())
    r_squared = 1.0 - (residual_sum / total_sum if total_sum else 0.0)

    constrained_intercept = float(np.mean(y - x))
    alpha_beta1 = 10.0 ** constrained_intercept

    return {
        "epsilon_min": resolved_min,
        "epsilon_max": resolved_max,
        "num_points": num_points,
        "fit_points": int(fit_mask.sum()),
        "beta_exponent": float(beta),
        "log10_intercept": float(intercept),
        "r_squared": float(r_squared),
        "alpha_assuming_beta_equals_1": float(alpha_beta1),
        "curve": [
            {"epsilon": float(epsilon), "eta_hat": float(eta)}
            for epsilon, eta in zip(epsilons, eta_hat, strict=True)
        ],
    }


def analyze_margins(
    token_stats_path: str | Path,
    *,
    output_path: str | Path,
    curve_output_path: str | Path | None = None,
    epsilon_min: float | None = None,
    epsilon_max: float | None = None,
    num_points: int = 32,
) -> dict[str, Any]:
    margins, entropies = _load_margin_csv(token_stats_path)
    if margins.size == 0:
        raise ValueError("token statistics file is empty")

    spearman = spearmanr(margins, entropies)
    gap_fit = _fit_gap_curve(
        margins,
        epsilon_min=epsilon_min,
        epsilon_max=epsilon_max,
        num_points=num_points,
    )

    summary = {
        "n_positions": int(margins.size),
        "margin_percentiles": {
            "p05": float(np.quantile(margins, 0.05)),
            "p25": float(np.quantile(margins, 0.25)),
            "p50": float(np.quantile(margins, 0.50)),
            "p75": float(np.quantile(margins, 0.75)),
            "p95": float(np.quantile(margins, 0.95)),
        },
        "fraction_margin_lt_0_5": float((margins < 0.5).mean()),
        "entropy_margin_spearman_rho": float(spearman.statistic),
        "entropy_margin_spearman_pvalue": float(spearman.pvalue),
        "gap_fit": gap_fit,
        "notes": [
            "beta_exponent tests whether the small-epsilon scaling is near linear.",
            "alpha_assuming_beta_equals_1 is reported separately so the coefficient is not confused with the free-fit slope.",
        ],
    }
    write_json(output_path, summary)

    if curve_output_path is not None:
        curve_target = Path(curve_output_path)
        ensure_dir(curve_target.parent)
        with curve_target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epsilon", "eta_hat"])
            writer.writeheader()
            for row in gap_fit["curve"]:
                writer.writerow(row)

    return summary

