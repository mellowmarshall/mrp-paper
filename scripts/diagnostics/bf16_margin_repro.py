#!/usr/bin/env python3
"""Minimal reproduction of the bfloat16 margin quantization artifact.

This script demonstrates that:
1. Computing logits via the model's native bf16 lm_head matmul produces
   a severely quantized margin distribution (few hundred unique values).
2. Recomputing the same matmul in float32 from the bf16 hidden states
   recovers a near-continuous margin distribution (one unique value per
   position in most cases).

The root cause is bfloat16's 8-bit mantissa: the lm_head matmul produces
logits whose differences (margins) collapse onto a coarse grid. The step
size is NOT uniform at 0.0625 -- it varies with magnitude because bfloat16
precision is exponent-dependent. At typical logit magnitudes (1-8), the
spacing between adjacent representable values ranges from ~0.008 to 0.0625.

Usage (CPU, ~2GB RAM for a small model):
    python scripts/diagnostics/bf16_margin_repro.py \
        --model-id Qwen/Qwen2.5-0.5B \
        --max-sequences 8

For the paper's subject model (requires ~10GB):
    python scripts/diagnostics/bf16_margin_repro.py \
        --model-id Qwen/Qwen3.5-4B-Base \
        --max-sequences 8
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _count_unique_margins(logits: torch.Tensor) -> tuple[int, int, float, float]:
    """Return (n_positions, n_unique, median_margin, max_margin) from a [S, V] logit tensor."""
    top2 = torch.topk(logits, 2, dim=-1).values  # [S, 2]
    margins = top2[:, 0] - top2[:, 1]
    n = margins.numel()
    n_unique = margins.unique().numel()
    return n, n_unique, float(margins.median()), float(margins.max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-sequences", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    print(f"Loading {args.model_id} in bfloat16 ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.requires_grad_(False)

    # Resolve lm_head weight
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        lm_head_weight = model.lm_head.weight
    elif hasattr(model, "get_output_embeddings"):
        lm_head_weight = model.get_output_embeddings().weight
    else:
        print("ERROR: cannot find lm_head weight", file=sys.stderr)
        sys.exit(1)

    # Simple test corpus
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "It was the best of times, it was the worst of times.",
        "To be, or not to be, that is the question.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
        "Call me Ishmael. Some years ago, never mind how long precisely.",
        "It is a truth universally acknowledged, that a single man in possession of a good fortune.",
        "The sky above the port was the color of television, tuned to a dead channel.",
    ][:args.max_sequences]

    all_bf16_margins = []
    all_fp32_margins = []

    print(f"\nProcessing {len(texts)} sequences (max_length={args.max_length}) ...")
    for i, text in enumerate(texts):
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=args.max_length)
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)

        # Path 1: bf16 logits from model (the artifact)
        bf16_logits = outputs.logits[0, :-1, :]  # [S-1, V], still bf16

        # Path 2: fp32 recomputation from bf16 hidden states
        hidden = outputs.hidden_states[-1][0, :-1, :].float()  # [S-1, d]
        fp32_logits = hidden @ lm_head_weight.detach().float().T  # [S-1, V]

        n_bf16, u_bf16, med_bf16, _ = _count_unique_margins(bf16_logits.float())
        n_fp32, u_fp32, med_fp32, _ = _count_unique_margins(fp32_logits)

        all_bf16_margins.append((n_bf16, u_bf16))
        all_fp32_margins.append((n_fp32, u_fp32))

        print(f"  seq {i}: {n_bf16} positions | "
              f"bf16: {u_bf16} unique ({u_bf16/n_bf16*100:.1f}%) | "
              f"fp32: {u_fp32} unique ({u_fp32/n_fp32*100:.1f}%)")

    total_pos = sum(n for n, _ in all_bf16_margins)
    total_bf16_unique = sum(u for _, u in all_bf16_margins)
    total_fp32_unique = sum(u for _, u in all_fp32_margins)

    print(f"\n{'='*60}")
    print(f"Total positions: {total_pos}")
    print(f"bf16 unique margins (sum across seqs): {total_bf16_unique}")
    print(f"fp32 unique margins (sum across seqs): {total_fp32_unique}")
    print(f"bf16 utilization: {total_bf16_unique/total_pos*100:.1f}%")
    print(f"fp32 utilization: {total_fp32_unique/total_pos*100:.1f}%")
    print(f"Recovery factor: {total_fp32_unique/max(total_bf16_unique, 1):.1f}x")
    print(f"{'='*60}")

    # Show bf16 step sizes at different magnitudes
    print(f"\nbfloat16 precision at different magnitudes:")
    for mag in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        t = torch.tensor(mag, dtype=torch.bfloat16)
        next_val = torch.nextafter(t, torch.tensor(float("inf"), dtype=torch.bfloat16))
        step = float(next_val - t)
        print(f"  magnitude {mag:5.1f}: step = {step:.6f} "
              f"(margin differences near {mag:.0f} quantize to this grid)")

    result = {
        "model_id": args.model_id,
        "total_positions": total_pos,
        "bf16_unique_margins": total_bf16_unique,
        "fp32_unique_margins": total_fp32_unique,
        "bf16_utilization_pct": round(total_bf16_unique / total_pos * 100, 2),
        "fp32_utilization_pct": round(total_fp32_unique / total_pos * 100, 2),
        "recovery_factor": round(total_fp32_unique / max(total_bf16_unique, 1), 1),
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults written to {args.output}")

    # Sanity check
    if total_bf16_unique / total_pos > 0.5:
        print("\nWARNING: bf16 path did not show expected quantization artifact.")
        print("This may happen with very short sequences or models that don't use bf16 matmul.")


if __name__ == "__main__":
    main()
