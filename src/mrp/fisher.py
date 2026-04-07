"""Shared Fisher information metric distance computation.

This is the canonical implementation. Both training (via compute_fisher_mrp_penalty
in training.py) and analysis scripts should use these functions to ensure numerical
consistency.

Key implementation details that analysis scripts previously missed:
- Embedding normalization before einsum (prevents overflow)
- Logit stabilization (subtract max before softmax)
- Diagonal zeroing before sqrt (prevents NaN gradients)
- eps=1e-8 clamp in sqrt (gradient stability)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def fisher_pairwise_distances(
    logits: torch.Tensor,
    lm_head_weight: torch.Tensor,
    top_k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Fisher-metric pairwise distances for top-k tokens.

    Args:
        logits: [batch, seq, vocab] raw logits (will be cast to fp32)
        lm_head_weight: [vocab, hidden] embedding matrix
        top_k: number of top tokens to consider

    Returns:
        fisher_dist: [batch, seq, k, k] pairwise Fisher distances
        top_k_probs: [batch, seq, k] softmax probabilities
        top_k_indices: [batch, seq, k] token indices
    """
    top_k = min(top_k, logits.size(-1))

    # Stabilize: shift logits so max is 0 (prevents softmax overflow)
    top_k_logits, top_k_indices = torch.topk(logits.float(), top_k, dim=-1)
    top_k_logits = top_k_logits - top_k_logits.max(dim=-1, keepdim=True).values
    top_k_probs = F.softmax(top_k_logits, dim=-1)

    top_k_embeds = lm_head_weight[top_k_indices].float()

    # Normalize embeddings to prevent large intermediate values
    embed_norm = top_k_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    top_k_embeds_normed = top_k_embeds / embed_norm

    # Sigma_k = diag(p) - p p^T  [batch, seq, k, k]
    Sigma_k = torch.diag_embed(top_k_probs) - (
        top_k_probs.unsqueeze(-1) * top_k_probs.unsqueeze(-2)
    )

    # Pairwise embedding diffs (on normalized embeddings)
    delta = top_k_embeds_normed.unsqueeze(-2) - top_k_embeds_normed.unsqueeze(-3)

    # Project through W_k
    proj = torch.einsum("bsijd,bsmd->bsijm", delta, top_k_embeds_normed)

    # Fisher squared distance
    fisher_dist_sq = torch.einsum("bsijm,bsmn,bsijn->bsij", proj, Sigma_k, proj)

    # Zero diagonal BEFORE sqrt, clamp to eps for gradient stability
    eye = torch.eye(top_k, device=logits.device, dtype=fisher_dist_sq.dtype)
    fisher_dist_sq = fisher_dist_sq * (1.0 - eye)
    fisher_dist = torch.sqrt(fisher_dist_sq.clamp(min=1e-8))

    return fisher_dist, top_k_probs, top_k_indices


def fisher_penalty(
    logits: torch.Tensor,
    lm_head_weight: torch.Tensor,
    top_k: int = 5,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the prob-weighted Fisher penalty (scalar).

    Args:
        logits: [batch, seq, vocab]
        lm_head_weight: [vocab, hidden]
        top_k: number of top tokens
        mask: [batch, seq] boolean mask for valid positions

    Returns:
        Scalar penalty value.
    """
    fisher_dist, top_k_probs, _ = fisher_pairwise_distances(logits, lm_head_weight, top_k)

    eye = torch.eye(top_k, device=logits.device, dtype=fisher_dist.dtype)
    prob_outer = top_k_probs.unsqueeze(-1) * top_k_probs.unsqueeze(-2)
    penalty = fisher_dist * prob_outer * (1.0 - eye)
    per_token_penalty = penalty.sum(dim=(-1, -2))

    if mask is None:
        return per_token_penalty.mean()

    if mask.sum() == 0:
        return per_token_penalty.new_zeros(())
    return per_token_penalty.masked_select(mask).mean()
