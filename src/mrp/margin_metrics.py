from __future__ import annotations

from typing import Any

import torch


def compute_token_statistics(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    top_k: int,
) -> dict[str, Any]:
    if logits.ndim != 2:
        raise ValueError(f"expected [seq, vocab] logits, got shape {tuple(logits.shape)}")
    if target_ids.ndim != 1:
        raise ValueError(
            f"expected [seq] target ids, got shape {tuple(target_ids.shape)}"
        )
    if logits.size(0) != target_ids.size(0):
        raise ValueError("logits and target_ids must agree on sequence length")
    if top_k < 2:
        raise ValueError("top_k must be at least 2 to define a margin")

    actual_top_k = min(top_k, logits.size(-1))
    top_k_logits, top_k_indices = torch.topk(logits, k=actual_top_k, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)

    margins = top_k_logits[:, 0] - top_k_logits[:, 1]
    target_logits = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    correct_rank = (logits > target_logits.unsqueeze(-1)).sum(dim=-1) + 1

    return {
        "margins": margins,
        "entropy": entropy,
        "top1_ids": top_k_indices[:, 0],
        "top1_correct": top_k_indices[:, 0].eq(target_ids),
        "correct_rank": correct_rank,
        "top_k_indices": top_k_indices,
        "top_k_logits": top_k_logits,
    }

