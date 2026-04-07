"""Unified evaluation with tiered metrics.

Consolidates metrics from:
- scripts/fast_margin_audit.py (fp32 logit recomputation, per-position CSV)
- scripts/standard_eval.py (tier 1-3 metrics, isotropy, velocity, neighborhoods)
- audit.py (analyze_margins for gap curve fitting)
- margin_metrics.py (compute_token_statistics)
- intrinsic_dimension.py (MLE estimator)

Tiers:
  1 (--margins):   accuracy, margin percentiles, entropy, confused pairs, gap curve
  2 (--geometry):  isotropy, intrinsic dimension (MLE), layer-wise prediction accuracy
  3 (--dynamics):  representation velocity, neighborhood stability (needs --prev-checkpoint)
  4 (--benchmarks): lm-harness via run_lm_eval()
  --compare:       flip analysis vs --baseline
"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mrp.audit import analyze_margins
from mrp.model_loading import resolve_output_embeddings
from mrp.shared import (
    load_eval_sequences,
    load_model_flexible,
    resolve_device,
)
from mrp.utils import ensure_dir, write_json


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_BATCH_SIZE: int = 4  # sequences per forward pass to avoid OOM


def _forward_fp32(
    model: torch.nn.Module,
    sequences: list[torch.Tensor],
    device: torch.device,
    *,
    need_hidden: bool = False,
    batch_size: int = _BATCH_SIZE,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]] | None]:
    """Run the model in inference mode and collect fp32 logits via
    ``hidden_states[-1].float() @ lm_head_weight.float().T``.

    Returns ``(all_logits, all_targets, all_hidden_per_seq)`` where each
    element of *all_logits* is ``[seq_len-1, vocab]`` in fp32, *all_targets*
    is ``[seq_len-1]``, and *all_hidden_per_seq* is ``None`` unless
    *need_hidden* is ``True``.
    """
    lm_head_emb = resolve_output_embeddings(model)
    if lm_head_emb is None or not hasattr(lm_head_emb, "weight"):
        raise ValueError("unable to resolve output embeddings for fp32 logit recomputation")
    lm_head_weight = lm_head_emb.weight.detach().float().to(device)

    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_hidden: list[list[torch.Tensor]] | None = [] if need_hidden else None

    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        for seq in sequences:
            input_ids = seq.unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)

            # fp32 logit recomputation from final hidden state
            final_h = outputs.hidden_states[-1][0, :-1, :].float()
            logits_fp32 = final_h @ lm_head_weight.T  # [seq-1, vocab]

            all_logits.append(logits_fp32.cpu())
            all_targets.append(input_ids[0, 1:].cpu())

            if all_hidden is not None:
                # Per-layer mean hidden state for velocity / intrinsic-dim
                layer_means = [
                    hs[0].float().mean(dim=0).cpu()
                    for hs in outputs.hidden_states
                ]
                all_hidden.append(layer_means)

    if model_was_training:
        model.train()

    return all_logits, all_targets, all_hidden


# ---------------------------------------------------------------------------
# Tier 1 -- margins
# ---------------------------------------------------------------------------


def _compute_tier1(
    all_logits: list[torch.Tensor],
    all_targets: list[torch.Tensor],
    tokenizer: Any,
) -> dict[str, Any]:
    """Accuracy, margin percentiles, entropy, accuracy by band, confused pairs,
    correct_rank distribution."""
    margins_list: list[float] = []
    correct_list: list[bool] = []
    top1_ids_list: list[int] = []
    top2_ids_list: list[int] = []
    entropy_list: list[float] = []
    correct_rank_list: list[int] = []

    for logits, targets in zip(all_logits, all_targets):
        top2_vals, top2_idx = torch.topk(logits, 2, dim=-1)
        m = (top2_vals[:, 0] - top2_vals[:, 1]).numpy()
        preds = logits.argmax(dim=-1)
        c = (preds == targets).numpy()
        margins_list.extend(m.tolist())
        correct_list.extend(c.tolist())
        top1_ids_list.extend(top2_idx[:, 0].numpy().tolist())
        top2_ids_list.extend(top2_idx[:, 1].numpy().tolist())

        # Entropy
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        ent = -(probs * log_probs).sum(dim=-1).numpy()
        entropy_list.extend(ent.tolist())

        # Correct rank
        target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        rank = (logits > target_logits.unsqueeze(-1)).sum(dim=-1) + 1
        correct_rank_list.extend(rank.numpy().tolist())

    m_arr = np.array(margins_list)
    c_arr = np.array(correct_list, dtype=bool)
    e_arr = np.array(entropy_list)
    cr_arr = np.array(correct_rank_list)

    # Accuracy by margin band
    bands = [
        (0.0, 0.5, "m_lt_0.5"),
        (0.5, 1.0, "m_0.5_1.0"),
        (1.0, 2.0, "m_1.0_2.0"),
        (2.0, 5.0, "m_2.0_5.0"),
        (5.0, 1e9, "m_5.0_plus"),
    ]
    acc_by_band: dict[str, Any] = {}
    for lo, hi, name in bands:
        mask = (m_arr >= lo) & (m_arr < hi)
        n = int(mask.sum())
        if n > 0:
            acc_by_band[name] = {
                "n": n,
                "correct": int(c_arr[mask].sum()),
                "accuracy": round(float(c_arr[mask].mean()), 6),
            }

    # Confused pairs (margin < 1.0, frequency >= 3)
    pair_data: dict[tuple[int, int], dict[str, list[Any]]] = {}
    for i in range(len(margins_list)):
        if margins_list[i] < 1.0:
            pair = (top1_ids_list[i], top2_ids_list[i])
            if pair not in pair_data:
                pair_data[pair] = {"margins": [], "correct": []}
            pair_data[pair]["margins"].append(margins_list[i])
            pair_data[pair]["correct"].append(correct_list[i])

    confused_pairs: list[dict[str, Any]] = []
    for (t1, t2), data in sorted(pair_data.items(), key=lambda x: -len(x[1]["margins"])):
        if len(data["margins"]) < 3:
            continue
        confused_pairs.append({
            "top1_id": t1,
            "top2_id": t2,
            "top1_token": tokenizer.decode([t1]).strip() if tokenizer else str(t1),
            "top2_token": tokenizer.decode([t2]).strip() if tokenizer else str(t2),
            "frequency": len(data["margins"]),
            "mean_margin": round(float(np.mean(data["margins"])), 6),
            "accuracy": round(float(np.mean(data["correct"])), 6),
        })
        if len(confused_pairs) >= 20:
            break

    # Correct rank distribution
    rank_dist: dict[str, float] = {
        "rank_1_fraction": round(float((cr_arr == 1).mean()), 6),
        "rank_le_5_fraction": round(float((cr_arr <= 5).mean()), 6),
        "rank_le_10_fraction": round(float((cr_arr <= 10).mean()), 6),
        "median_rank": round(float(np.median(cr_arr)), 2),
    }

    return {
        "accuracy": round(float(c_arr.mean()), 6),
        "n_positions": len(m_arr),
        "n_correct": int(c_arr.sum()),
        "margin_percentiles": {
            "p05": round(float(np.quantile(m_arr, 0.05)), 6),
            "p25": round(float(np.quantile(m_arr, 0.25)), 6),
            "p50": round(float(np.median(m_arr)), 6),
            "p75": round(float(np.quantile(m_arr, 0.75)), 6),
            "p95": round(float(np.quantile(m_arr, 0.95)), 6),
        },
        "pr_lt_0.5": round(float((m_arr < 0.5).mean()), 6),
        "accuracy_by_band": acc_by_band,
        "entropy": {
            "mean": round(float(e_arr.mean()), 6),
            "median": round(float(np.median(e_arr)), 6),
            "p05": round(float(np.quantile(e_arr, 0.05)), 6),
            "p95": round(float(np.quantile(e_arr, 0.95)), 6),
        },
        "confused_pairs": confused_pairs,
        "correct_rank": rank_dist,
        # Internal -- used by --compare
        "_correct_list": correct_list,
        "_margins_list": margins_list,
        "_entropy_list": entropy_list,
    }


def _compute_gap_curve(
    margins_list: list[float],
    entropy_list: list[float],
) -> dict[str, Any]:
    """Write a temp CSV, invoke :func:`audit.analyze_margins`, return the
    gap fit results."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="",
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=["margin", "entropy"])
        writer.writeheader()
        for m_val, e_val in zip(margins_list, entropy_list):
            writer.writerow({"margin": "%.6f" % m_val, "entropy": "%.6f" % e_val})
        tmp_path = tmp.name

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_tmp:
        out_path = out_tmp.name

    try:
        result = analyze_margins(
            tmp_path,
            output_path=out_path,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)

    return result.get("gap_fit", {})


# ---------------------------------------------------------------------------
# Tier 2 -- geometry
# ---------------------------------------------------------------------------


def _compute_isotropy(model: torch.nn.Module) -> dict[str, Any]:
    """SVD effective rank, condition number, variance thresholds on the
    embedding matrix."""
    # Try to get the input embedding weight
    embed_weight: torch.Tensor | None = None
    if hasattr(model, "embed_tokens"):
        embed_weight = model.embed_tokens.weight
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_weight = model.model.embed_tokens.weight
    elif hasattr(model, "get_input_embeddings"):
        input_emb = model.get_input_embeddings()
        if input_emb is not None and hasattr(input_emb, "weight"):
            embed_weight = input_emb.weight

    if embed_weight is None:
        return {"error": "could not resolve input embeddings"}

    embed = embed_weight.detach().float().cpu()
    ec = embed - embed.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(ec, full_matrices=False)
    S_np = S.numpy()
    p = S_np / S_np.sum()
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    condition = float(S_np[0] / max(S_np[-1], 1e-12))
    cumvar = np.cumsum(S_np ** 2) / (S_np ** 2).sum()

    return {
        "effective_rank": round(eff_rank, 2),
        "condition_number": round(condition, 2),
        "dims_for_90pct_var": int(np.searchsorted(cumvar, 0.9)) + 1,
        "dims_for_95pct_var": int(np.searchsorted(cumvar, 0.95)) + 1,
        "ambient_dim": int(embed.shape[1]),
    }


def _compute_intrinsic_dimension_mle(
    all_hidden: list[list[torch.Tensor]],
) -> dict[str, Any]:
    """MLE intrinsic dimension estimate from final-layer hidden states."""
    # Gather final-layer mean hidden states
    stacked = torch.stack([layers[-1] for layers in all_hidden])
    points = stacked.numpy()
    if len(points) < 10:
        return {"mle_estimate": None, "note": "too few sequences for MLE"}

    from scipy.spatial.distance import pdist, squareform

    dists = squareform(pdist(points))
    np.fill_diagonal(dists, np.inf)
    estimates: list[float] = []
    for i in range(len(points)):
        sorted_d = np.sort(dists[i])[:10]
        for k in range(3, min(10, len(points) - 1)):
            if sorted_d[k] > 0 and sorted_d[0] > 0:
                lr = np.log(sorted_d[k] / sorted_d[:k])
                lr = lr[np.isfinite(lr)]
                if len(lr) > 0:
                    estimates.append(1.0 / np.mean(lr))
    if not estimates:
        return {"mle_estimate": None, "note": "MLE estimator found no valid neighborhoods"}
    return {"mle_estimate": round(float(np.median(estimates)), 2)}


def _compute_layer_accuracy(
    model: torch.nn.Module,
    sequences: list[torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    """Per-layer prediction accuracy: project each layer's hidden state through
    lm_head and measure top-1 accuracy."""
    lm_head_emb = resolve_output_embeddings(model)
    if lm_head_emb is None or not hasattr(lm_head_emb, "weight"):
        return {"error": "cannot resolve lm_head for layer accuracy"}

    lm_head_weight = lm_head_emb.weight.detach().float().to(device)
    layer_accs: list[list[float]] = []

    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        for seq in sequences:
            ids = seq.unsqueeze(0).to(device)
            out = model(input_ids=ids, output_hidden_states=True, return_dict=True)
            targets = ids[0, 1:].cpu()
            for li, hs in enumerate(out.hidden_states):
                h = hs[0, :-1, :].float()
                preds = (h @ lm_head_weight.T).argmax(dim=-1).cpu()
                acc = float((preds == targets).float().mean())
                if li >= len(layer_accs):
                    layer_accs.append([])
                layer_accs[li].append(acc)

    if model_was_training:
        model.train()

    return {
        "per_layer_accuracy": [
            round(float(np.mean(a)), 6) for a in layer_accs
        ],
    }


# ---------------------------------------------------------------------------
# Tier 3 -- dynamics
# ---------------------------------------------------------------------------


def _compute_velocity(
    all_hidden: list[list[torch.Tensor]],
    all_hidden_prev: list[list[torch.Tensor]],
) -> dict[str, Any]:
    """Per-layer representation velocity (cosine distance between matching
    layers across two checkpoints)."""
    curr = torch.stack([torch.stack(layers) for layers in all_hidden]).mean(dim=0)
    prev = torch.stack([torch.stack(layers) for layers in all_hidden_prev]).mean(dim=0)
    cos_dist = (1.0 - F.cosine_similarity(curr, prev, dim=-1)).numpy()
    return {
        "per_layer_velocity": [round(float(v), 6) for v in cos_dist],
        "mean_velocity": round(float(cos_dist.mean()), 6),
        "max_velocity_layer": int(np.argmax(cos_dist)),
    }


def _compute_neighborhoods(
    model: torch.nn.Module,
    prev_model: torch.nn.Module,
    k: int = 10,
) -> dict[str, Any]:
    """Embedding neighborhood stability between two checkpoints (Jaccard on
    k-nearest neighbors of 500 sampled tokens)."""
    def _get_embed(m: torch.nn.Module) -> torch.Tensor | None:
        if hasattr(m, "embed_tokens"):
            return m.embed_tokens.weight
        if hasattr(m, "model") and hasattr(m.model, "embed_tokens"):
            return m.model.embed_tokens.weight
        if hasattr(m, "get_input_embeddings"):
            inp = m.get_input_embeddings()
            if inp is not None and hasattr(inp, "weight"):
                return inp.weight
        return None

    embed = _get_embed(model)
    prev_embed = _get_embed(prev_model)
    if embed is None or prev_embed is None:
        return {"error": "could not resolve input embeddings for neighborhood stability"}

    embed = embed.detach().float().cpu()
    prev_embed = prev_embed.detach().float().cpu()

    en = embed / embed.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    pn = prev_embed / prev_embed.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    rng = np.random.RandomState(42)
    idx = rng.choice(embed.shape[0], min(500, embed.shape[0]), replace=False)
    stabilities: list[float] = []
    for i in idx:
        curr_nn = set(torch.topk(en[i] @ en.T, k + 1).indices[1:].tolist())
        prev_nn = set(torch.topk(pn[i] @ pn.T, k + 1).indices[1:].tolist())
        stabilities.append(len(curr_nn & prev_nn) / k)

    s = np.array(stabilities)
    return {
        "mean_stability": round(float(s.mean()), 6),
        "median_stability": round(float(np.median(s)), 6),
        "k": k,
    }


# ---------------------------------------------------------------------------
# Flip analysis (--compare)
# ---------------------------------------------------------------------------


def _compute_flip_analysis(
    model_correct: list[bool],
    baseline_correct: list[bool],
) -> dict[str, Any]:
    """W->R, R->W, flip ratio between model and baseline."""
    c = np.array(model_correct, dtype=bool)
    bc = np.array(baseline_correct, dtype=bool)
    w2r = int((~bc & c).sum())
    r2w = int((bc & ~c).sum())
    return {
        "both_right": int((bc & c).sum()),
        "both_wrong": int((~bc & ~c).sum()),
        "model_wins": w2r,
        "baseline_wins": r2w,
        "flip_ratio": round(w2r / max(r2w, 1), 4),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_eval(
    *,
    model_path: str,
    output: str,
    device: str = "cpu",
    config_overrides: dict[str, Any] | None = None,
    tokenizer_id: str | None = None,
    trust_remote_code: bool = False,
    n_sequences: int = 64,
    max_length: int = 512,
    # Tier flags
    margins: bool = True,
    geometry: bool = False,
    dynamics: bool = False,
    benchmarks: bool = False,
    benchmark_tasks: str | None = None,
    benchmark_limit: float | None = None,
    # Compare
    baseline: str | None = None,
    baseline_config: dict[str, Any] | None = None,
    baseline_tokenizer_id: str | None = None,
    # Dynamics
    prev_checkpoint: str | None = None,
    prev_config: dict[str, Any] | None = None,
    prev_tokenizer_id: str | None = None,
) -> dict[str, Any]:
    """Unified model assessment with tiered metrics.

    Parameters
    ----------
    model_path:
        Path to a checkpoint directory, ``model.pt`` file, or HF model ID.
    output:
        Path to write the JSON results file.
    margins:
        Tier 1 -- accuracy, margins, entropy, gap curve, confused pairs.
    geometry:
        Tier 2 -- isotropy, intrinsic dimension, layer-wise accuracy.
    dynamics:
        Tier 3 -- representation velocity, neighborhood stability
        (requires *prev_checkpoint*).
    benchmarks:
        Tier 4 -- lm-harness.
    baseline:
        Path to baseline model for flip analysis (``--compare``).
    prev_checkpoint:
        Previous checkpoint for velocity / neighborhood stability.

    Returns
    -------
    dict[str, Any]
        The complete results dictionary (also written to *output*).
    """
    runtime_device = resolve_device(device)
    need_hidden = geometry or dynamics

    # Load model
    model, tokenizer = load_model_flexible(
        model_path,
        device=device,
        config_overrides=config_overrides,
        tokenizer_id=tokenizer_id,
        trust_remote_code=trust_remote_code,
    )

    # Load sequences for assessment
    sequences = load_eval_sequences(
        tokenizer,
        n_sequences=n_sequences,
        max_length=max_length,
    )

    result: dict[str, Any] = {
        "model_path": str(model_path),
        "n_sequences": len(sequences),
        "n_tokens": sum(s.size(0) for s in sequences),
        "device": str(runtime_device),
    }

    # Forward pass
    all_logits, all_targets, all_hidden = _forward_fp32(
        model, sequences, runtime_device, need_hidden=need_hidden,
    )

    # Tier 1 reference for flip analysis
    tier1: dict[str, Any] | None = None

    # ---- Tier 1: margins ----
    if margins:
        tier1 = _compute_tier1(all_logits, all_targets, tokenizer)
        result["accuracy"] = tier1["accuracy"]
        result["n_positions"] = tier1["n_positions"]
        result["n_correct"] = tier1["n_correct"]
        result["margin_percentiles"] = tier1["margin_percentiles"]
        result["pr_lt_0.5"] = tier1["pr_lt_0.5"]
        result["accuracy_by_band"] = tier1["accuracy_by_band"]
        result["entropy"] = tier1["entropy"]
        result["confused_pairs"] = tier1["confused_pairs"]
        result["correct_rank"] = tier1["correct_rank"]

        # Gap curve via audit.analyze_margins
        try:
            gap_curve = _compute_gap_curve(
                tier1["_margins_list"], tier1["_entropy_list"],
            )
            result["gap_curve"] = gap_curve
        except Exception as exc:
            result["gap_curve_error"] = str(exc)

    # ---- Tier 2: geometry ----
    if geometry:
        result["isotropy"] = _compute_isotropy(model)
        if all_hidden:
            result["intrinsic_dimension"] = _compute_intrinsic_dimension_mle(all_hidden)
        result["layer_accuracy"] = _compute_layer_accuracy(
            model, sequences[:16], runtime_device,
        )

    # ---- Tier 3: dynamics ----
    if dynamics and prev_checkpoint:
        prev_model, _ = load_model_flexible(
            prev_checkpoint,
            device=device,
            config_overrides=prev_config,
            tokenizer_id=prev_tokenizer_id or tokenizer_id,
            trust_remote_code=trust_remote_code,
        )
        _, _, all_hidden_prev = _forward_fp32(
            prev_model, sequences, runtime_device, need_hidden=True,
        )
        if all_hidden and all_hidden_prev:
            result["representation_velocity"] = _compute_velocity(
                all_hidden, all_hidden_prev,
            )
        result["neighborhood_stability"] = _compute_neighborhoods(
            model, prev_model, k=10,
        )
        del prev_model

    # ---- Compare / flip analysis ----
    if baseline and tier1 is not None:
        base_model, _ = load_model_flexible(
            baseline,
            device=device,
            config_overrides=baseline_config,
            tokenizer_id=baseline_tokenizer_id or tokenizer_id,
            trust_remote_code=trust_remote_code,
        )
        base_logits, base_targets, _ = _forward_fp32(
            base_model, sequences, runtime_device, need_hidden=False,
        )
        base_correct: list[bool] = []
        for logits, targets in zip(base_logits, base_targets):
            c = (logits.argmax(dim=-1) == targets).numpy().tolist()
            base_correct.extend(c)
        result["flip_analysis"] = _compute_flip_analysis(
            tier1["_correct_list"], base_correct,
        )
        del base_model

    # ---- Tier 4: benchmarks ----
    if benchmarks:
        from mrp.eval_harness import run_lm_eval

        tasks = benchmark_tasks or "hellaswag,arc_easy"
        bench_output = str(Path(output).with_suffix(".benchmarks.json"))
        bench_result = run_lm_eval(
            model_id=str(model_path),
            tasks=tasks,
            output_path=bench_output,
            device=device,
            trust_remote_code=trust_remote_code,
            limit=benchmark_limit,
        )
        result["benchmarks"] = {
            "tasks": tasks,
            "output_path": bench_output,
            "results": bench_result.get("results", {}),
        }

    # Remove internal fields
    for key in list(result.keys()):
        if key.startswith("_"):
            del result[key]

    # Write output
    ensure_dir(Path(output).parent)
    write_json(output, result)
    return result
