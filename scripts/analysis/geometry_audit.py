"""Geometry audit: margins, entropy, effective dimension.

Runs focused analysis of a trained model's representational geometry
on Wikitext-103 validation. Writes durable JSON outputs for run comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    SRC_ROOT = REPO_ROOT / "src"
    for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

from mrp.factored_transformer import FactoredCausalLM, StandardCausalLM

sys.stdout.reconfigure(line_buffering=True)


def _load_model(model_id, device="cpu"):
    # Select class from config's architecture field
    cfg_path = Path(model_id) / "config.json"
    arch = None
    if cfg_path.exists():
        cfg = json.load(cfg_path.open())
        arch_list = cfg.get("architectures") or []
        arch = arch_list[0] if arch_list else None
    cls = StandardCausalLM if arch == "StandardCausalLM" else FactoredCausalLM
    model = cls.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model


def _collect_stats(model, tokenizer, dataset, n_sequences, max_length,
                   n_hidden_samples_per_seq=32, device="cpu"):
    margins, entropies, top1_correct = [], [], []
    hidden_samples = []

    count = 0
    for example in dataset:
        if count >= n_sequences:
            break
        text = example.get("text", "").strip()
        if len(text) < 100:
            continue
        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_length).input_ids.to(device)
        if tokens.shape[1] < 64:
            continue
        with torch.no_grad():
            out = model(input_ids=tokens, labels=tokens, output_hidden_states=True)
        logits = out.logits[0]
        labels = tokens[0]
        shift_logits = logits[:-1]
        shift_labels = labels[1:]

        top2_vals, _ = shift_logits.topk(2, dim=-1)
        margins.extend((top2_vals[:, 0] - top2_vals[:, 1]).cpu().numpy().tolist())
        log_probs = F.log_softmax(shift_logits, dim=-1)
        entropies.extend((-(log_probs.exp() * log_probs).sum(dim=-1)).cpu().numpy().tolist())
        top1 = shift_logits.argmax(dim=-1)
        top1_correct.extend((top1 == shift_labels).float().cpu().numpy().tolist())

        hidden = out.hidden_states[-1][0].cpu().numpy()
        if hidden.shape[0] > n_hidden_samples_per_seq:
            rng = np.random.default_rng(count)
            idx = rng.choice(hidden.shape[0], n_hidden_samples_per_seq, replace=False)
            hidden_samples.append(hidden[idx])
        else:
            hidden_samples.append(hidden)

        count += 1
        if count % 8 == 0:
            print(f"  {count}/{n_sequences}")

    return {
        "margins": np.array(margins),
        "entropies": np.array(entropies),
        "top1_correct": np.array(top1_correct),
        "hidden": np.concatenate(hidden_samples, axis=0) if hidden_samples else np.zeros((0, 0)),
    }


def _analyze_margins(margins, top1_correct):
    correct_m = margins[top1_correct > 0.5]
    wrong_m = margins[top1_correct < 0.5]
    return {
        "n_positions": int(margins.size),
        "top1_accuracy": float(top1_correct.mean()),
        "percentiles": {f"p{p:02d}": float(np.quantile(margins, p / 100))
                        for p in (5, 25, 50, 75, 95)},
        "mean": float(margins.mean()),
        "std": float(margins.std()),
        "correct": {"n": int(correct_m.size),
                    "mean": float(correct_m.mean()) if correct_m.size > 0 else None,
                    "median": float(np.median(correct_m)) if correct_m.size > 0 else None},
        "wrong": {"n": int(wrong_m.size),
                  "mean": float(wrong_m.mean()) if wrong_m.size > 0 else None,
                  "median": float(np.median(wrong_m)) if wrong_m.size > 0 else None},
        "correct_wrong_gap": (float(correct_m.mean() - wrong_m.mean())
                               if correct_m.size > 0 and wrong_m.size > 0 else None),
    }


def _analyze_entropy(entropies, vocab_size):
    return {
        "n_positions": int(entropies.size),
        "max_possible": float(np.log(vocab_size)),
        "percentiles": {f"p{p:02d}": float(np.quantile(entropies, p / 100))
                        for p in (5, 25, 50, 75, 95)},
        "mean": float(entropies.mean()),
        "std": float(entropies.std()),
        "fraction_of_max": float(entropies.mean() / np.log(vocab_size)),
    }


def _analyze_effective_dim(hidden):
    if hidden.size == 0:
        return {"error": "no hidden states collected"}
    n, d = hidden.shape
    centered = hidden - hidden.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    eigvals = s ** 2
    var_ratio = eigvals / eigvals.sum()
    cum_var = np.cumsum(var_ratio)
    participation_ratio = float(eigvals.sum() ** 2 / (eigvals ** 2).sum())
    thresholds = {}
    for thresh in (0.5, 0.75, 0.9, 0.95, 0.99):
        idx = int(np.argmax(cum_var >= thresh)) + 1
        thresholds[f"dims_for_{int(thresh * 100)}pct_var"] = idx
    return {
        "n_samples": int(n),
        "hidden_dim": int(d),
        "participation_ratio": participation_ratio,
        "participation_ratio_frac": participation_ratio / d,
        "variance_thresholds": thresholds,
        "top10_eigval_ratios": [float(x) for x in var_ratio[:10].tolist()],
    }


def main():
    parser = argparse.ArgumentParser(description="Geometry audit")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tokenizer-id", default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n-sequences", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--n-hidden-samples", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_id}")
    model = _load_model(args.model_id, device=args.device)
    n_params = sum(p.numel() for p in model.parameters())
    vocab_size = model.config.vocab_size
    print(f"  {n_params / 1e6:.1f}M params, vocab={vocab_size}")

    print(f"Loading tokenizer {args.tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

    print(f"Loading {args.dataset_name}/{args.dataset_config} [{args.split}]")
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)

    print(f"Running {args.n_sequences} sequences (max_length={args.max_length})...")
    stats = _collect_stats(model, tokenizer, dataset,
                           n_sequences=args.n_sequences, max_length=args.max_length,
                           n_hidden_samples_per_seq=args.n_hidden_samples,
                           device=args.device)

    margin_result = _analyze_margins(stats["margins"], stats["top1_correct"])
    entropy_result = _analyze_entropy(stats["entropies"], vocab_size)
    eff_dim_result = _analyze_effective_dim(stats["hidden"])

    with (output_dir / "margin_distribution.json").open("w") as f:
        json.dump(margin_result, f, indent=2)
    with (output_dir / "entropy_distribution.json").open("w") as f:
        json.dump(entropy_result, f, indent=2)
    with (output_dir / "effective_dimension.json").open("w") as f:
        json.dump(eff_dim_result, f, indent=2)

    summary = {
        "model_id": str(args.model_id),
        "n_parameters": n_params,
        "vocab_size": vocab_size,
        "dataset": f"{args.dataset_name}/{args.dataset_config}",
        "n_sequences": args.n_sequences,
        "n_positions": int(margin_result["n_positions"]),
        "top1_accuracy": margin_result["top1_accuracy"],
        "margin_median": margin_result["percentiles"]["p50"],
        "margin_correct_wrong_gap": margin_result["correct_wrong_gap"],
        "entropy_mean": entropy_result["mean"],
        "entropy_fraction_of_max": entropy_result["fraction_of_max"],
        "participation_ratio": eff_dim_result.get("participation_ratio"),
        "participation_ratio_frac": eff_dim_result.get("participation_ratio_frac"),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print()
    print(f"Wrote results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
