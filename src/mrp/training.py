from __future__ import annotations

import json as _stdlib_json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import IterableDataset

from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback


def _patch_torch_load_check():
    """Patch transformers 5.3+ torch.load safety check for torch <2.6.
    Only call this when loading our own trusted checkpoints."""
    try:
        replacement = lambda: None

        def _replace(holder: Any) -> None:
            fn = getattr(holder, "check_torch_load_is_safe", None)
            if not callable(fn):
                return
            try:
                fn.__code__ = replacement.__code__
                fn.__defaults__ = replacement.__defaults__
                fn.__kwdefaults__ = replacement.__kwdefaults__
            except Exception:
                holder.check_torch_load_is_safe = replacement

        import transformers.utils.import_utils as _tf_iu
        _replace(_tf_iu)
        # transformers.trainer imports the symbol into module scope, so patch the
        # call site too or resume will still fail on optimizer/scheduler restore.
        import transformers.trainer as _tf_trainer
        _replace(_tf_trainer)

        # Torch 2.5 + transformers 5.3 also trips on resume RNG state because the
        # serialized numpy state is not allowlisted for weights_only=True loads.
        # Exact RNG replay is not required for resumed scratch training here; skip
        # RNG restore rather than failing the entire resume.
        def _skip_rng_state(self, checkpoint: str) -> None:
            return None

        try:
            _tf_trainer.Trainer._load_rng_state = _skip_rng_state
        except Exception:
            pass
        try:
            import transformers as _tf
            _tf.Trainer._load_rng_state = _skip_rng_state
        except Exception:
            pass
        try:
            Trainer._load_rng_state = _skip_rng_state
        except Exception:
            pass
        try:
            _ScratchTrainer._load_rng_state = _skip_rng_state
        except Exception:
            pass
    except Exception:
        pass

from mrp.model_loading import load_text_model, resolve_output_embeddings
from mrp.supervisor_agent import (
    LanguageSupervisorCallback,
    accumulate_metrics,
    average_metrics,
    collect_runtime_metrics,
)
from mrp.tracker import start_run
from mrp.utils import ensure_dir, write_json


DEFAULT_TEXT_COLUMNS = ("text", "content", "document")
DEFAULT_CPU_TRAINABLE_PARAM_LIMIT = 50_000_000
_BLOCK_PREFIX_PATTERNS = (
    re.compile(r"^(?P<prefix>.*?layers)\.(?P<index>\d+)\."),
    re.compile(r"^(?P<prefix>.*?blocks)\.(?P<index>\d+)\."),
    re.compile(r"^(?P<prefix>.*?h)\.(?P<index>\d+)\."),
)


@dataclass(frozen=True)
class TrainableSummary:
    scope: str
    trainable_parameters: int
    total_parameters: int
    trainable_fraction: float
    trainable_parameter_names: list[str]


def _detect_text_column(column_names: list[str], requested: str | None) -> str:
    if requested:
        if requested not in column_names:
            raise ValueError(
                f"text column '{requested}' not found. Available columns: {column_names}"
            )
        return requested

    for candidate in DEFAULT_TEXT_COLUMNS:
        if candidate in column_names:
            return candidate

    raise ValueError(f"unable to infer text column. Available columns: {column_names}")


def _collect_block_prefixes(model: torch.nn.Module) -> dict[str, set[int]]:
    groups: dict[str, set[int]] = {}
    for name, _ in model.named_parameters():
        for pattern in _BLOCK_PREFIX_PATTERNS:
            match = pattern.match(name)
            if match is None:
                continue
            prefix = match.group("prefix") + "."
            groups.setdefault(prefix, set()).add(int(match.group("index")))
            break
    return groups


def _score_block_prefix(prefix: str, layer_ids: set[int]) -> tuple[int, int, int, int]:
    lowered = prefix.lower()
    return (
        int("language_model" in lowered),
        len(layer_ids),
        prefix.count("."),
        len(prefix),
    )


def _select_primary_block_prefix(model: torch.nn.Module) -> tuple[str | None, list[int]]:
    groups = _collect_block_prefixes(model)
    if not groups:
        return None, []
    prefix = max(groups, key=lambda item: _score_block_prefix(item, groups[item]))
    return prefix, sorted(groups[prefix])


def _block_root(prefix: str | None) -> str:
    if not prefix:
        return ""
    container = prefix[:-1]
    if "." not in container:
        return ""
    return container.rsplit(".", 1)[0] + "."


def _find_existing_prefixes(
    names: list[str],
    candidates: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        prefix for prefix in candidates
        if prefix and any(name.startswith(prefix) for name in names)
    )


def _resolve_text_prefixes(model: torch.nn.Module) -> tuple[str, ...]:
    block_prefix, _ = _select_primary_block_prefix(model)
    root = _block_root(block_prefix)
    prefixes: list[str] = []
    if root:
        prefixes.append(root)
    if root and any(name.startswith("lm_head.") for name, _ in model.named_parameters()):
        prefixes.append("lm_head.")
    return tuple(prefixes)


def _resolve_final_norm_prefixes(model: torch.nn.Module) -> tuple[str, ...]:
    block_prefix, _ = _select_primary_block_prefix(model)
    root = _block_root(block_prefix)
    names = [name for name, _ in model.named_parameters()]
    candidates = (
        f"{root}norm.",
        f"{root}final_norm.",
        f"{root}ln_f.",
        f"{root}final_layer_norm.",
        f"{root}decoder.final_layer_norm.",
    )
    return _find_existing_prefixes(names, candidates)


def _resolve_embedding_prefixes(model: torch.nn.Module) -> tuple[str, ...]:
    block_prefix, _ = _select_primary_block_prefix(model)
    root = _block_root(block_prefix)
    names = [name for name, _ in model.named_parameters()]
    candidates = (
        f"{root}embed_tokens.",
        f"{root}wte.",
        f"{root}tok_embeddings.",
        f"{root}word_embeddings.",
    )
    return _find_existing_prefixes(names, candidates)


def _apply_trainable_scope(
    model: torch.nn.Module,
    scope: str,
    *,
    trainable_last_n: int = 1,
) -> TrainableSummary:
    if trainable_last_n <= 0:
        raise ValueError("trainable_last_n must be >= 1")

    block_prefix, layer_ids = _select_primary_block_prefix(model)
    final_norm_prefixes = _resolve_final_norm_prefixes(model)
    embedding_prefixes = _resolve_embedding_prefixes(model)

    allowed_prefixes: tuple[str, ...]
    if scope == "text":
        text_prefixes = _resolve_text_prefixes(model)
        allowed_prefixes = text_prefixes or tuple()
    elif scope == "last_block":
        scope = "last_n_blocks"
        trainable_last_n = 1
    if scope == "last_n_blocks":
        if block_prefix is None or not layer_ids:
            raise ValueError("unable to resolve transformer block prefixes for last_n_blocks")
        selected_layers = layer_ids[-trainable_last_n:]
        allowed_prefixes = tuple(
            f"{block_prefix}{index}." for index in selected_layers
        ) + final_norm_prefixes
    elif scope == "final_norm":
        allowed_prefixes = final_norm_prefixes
    elif scope == "embeddings":
        allowed_prefixes = embedding_prefixes
    elif scope != "text":
        raise ValueError(
            f"unsupported trainable scope '{scope}'. "
            "Expected one of: text, last_block, last_n_blocks, final_norm, embeddings."
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trainable_parameter_names: list[str] = []
    trainable_parameters = 0
    total_parameters = 0

    for name, parameter in model.named_parameters():
        total_parameters += parameter.numel()
        if not allowed_prefixes:
            should_train = (scope == "text")
        else:
            should_train = name.startswith(allowed_prefixes)
        if should_train:
            parameter.requires_grad_(True)
            trainable_parameter_names.append(name)
            trainable_parameters += parameter.numel()

    if trainable_parameters == 0:
        raise ValueError(f"trainable scope '{scope}' selected zero parameters")

    return TrainableSummary(
        scope=scope,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        trainable_fraction=float(trainable_parameters / total_parameters),
        trainable_parameter_names=trainable_parameter_names,
    )


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _normalize_dtype(torch_dtype: str) -> str:
    normalized = torch_dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return "bfloat16"
    if normalized in {"fp16", "float16", "half"}:
        return "float16"
    if normalized in {"fp32", "float32", "float"}:
        return "float32"
    if normalized == "auto":
        return "auto"
    raise ValueError(
        f"unsupported dtype '{torch_dtype}'. Expected auto, bfloat16, float16, or float32."
    )


def _prepare_lm_dataset(
    *,
    tokenizer: Any,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str | None,
    block_size: int,
    max_samples: int | None,
) -> tuple[Dataset, str]:
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    resolved_text_column = _detect_text_column(dataset.column_names, text_column)
    eos_token = tokenizer.eos_token or ""

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        texts: list[str] = []
        for value in batch[resolved_text_column]:
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            if not stripped:
                continue
            texts.append(stripped + eos_token)
        if not texts:
            return {"input_ids": [], "attention_mask": []}
        return tokenizer(texts, add_special_tokens=False)

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"tokenizing {split}",
    )

    def group_texts(examples: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated = {
            key: list(chain.from_iterable(examples[key]))
            for key in examples
        }
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        result = {
            key: [
                values[index : index + block_size]
                for index in range(0, total_length, block_size)
            ]
            for key, values in concatenated.items()
        }
        result["labels"] = [list(chunk) for chunk in result["input_ids"]]
        return result

    packed = tokenized.map(
        group_texts,
        batched=True,
        desc=f"packing {split}",
    )
    if len(packed) == 0:
        raise ValueError(
            f"dataset split '{split}' did not produce any {block_size}-token blocks"
        )
    return packed, resolved_text_column


def _align_for_causal_penalty(
    logits: torch.Tensor,
    labels: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    aligned_logits = logits[..., :-1, :]
    if labels is None:
        return aligned_logits, None
    aligned_labels = labels[..., 1:].contiguous()
    return aligned_logits, aligned_labels


def compute_mrp_penalty(
    logits: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    top_k: int = 5,
) -> torch.Tensor:
    aligned_logits, aligned_labels = _align_for_causal_penalty(logits, labels)
    top_k = min(top_k, aligned_logits.size(-1))
    top_k_logits, top_k_indices = torch.topk(aligned_logits.float(), top_k, dim=-1)
    top_k_probs = F.softmax(top_k_logits, dim=-1)

    top_k_embeds = lm_head_weight[top_k_indices]
    normed = F.normalize(top_k_embeds.float(), dim=-1)
    pairwise_sim = torch.matmul(normed, normed.transpose(-1, -2))
    pairwise_dist = 1.0 - pairwise_sim

    prob_outer = top_k_probs.unsqueeze(-1) * top_k_probs.unsqueeze(-2)
    eye = torch.eye(top_k, device=aligned_logits.device, dtype=prob_outer.dtype)
    penalty = pairwise_dist * prob_outer * (1.0 - eye)
    per_token_penalty = penalty.sum(dim=(-1, -2))

    if aligned_labels is None:
        return per_token_penalty.mean()

    valid_mask = aligned_labels != -100
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return per_token_penalty.new_zeros(())
    return per_token_penalty.masked_select(valid_mask).mean()


def compute_margin_loss(
    logits: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    margin_threshold: float = 1.0,
) -> torch.Tensor:
    """Direct margin maximization loss.

    Maximizes the Voronoi margin (top1 - top2 logit gap) for positions
    where the margin is below threshold. No embedding distance proxy —
    directly optimizes the geometric quantity we measure.

    Returns negative mean margin (so minimizing the loss = maximizing margin).
    """
    aligned_logits, aligned_labels = _align_for_causal_penalty(logits, labels)

    top2_logits = torch.topk(aligned_logits.float(), 2, dim=-1).values
    margins = top2_logits[..., 0] - top2_logits[..., 1]  # [batch, seq]

    # Gate: only penalize positions below threshold
    gate = margins < margin_threshold
    if aligned_labels is not None:
        gate = gate & (aligned_labels != -100)

    if gate.sum() == 0:
        return margins.new_zeros(())

    # Negative margin = minimizing this maximizes the margin
    return -margins.masked_select(gate).mean()


def compute_correct_margin_loss(
    logits: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    margin_threshold: float = 1.0,
) -> torch.Tensor:
    """Correct-only margin maximization.

    Only sharpens boundaries where the model's top-1 prediction is already
    correct. MRP never fights CE by defending wrong answers.

    Equivalent to: -mean(margin[correct & low_margin])
    """
    aligned_logits, aligned_labels = _align_for_causal_penalty(logits, labels)

    top2_logits = torch.topk(aligned_logits.float(), 2, dim=-1).values
    margins = top2_logits[..., 0] - top2_logits[..., 1]

    # Correct mask: top-1 prediction matches the target
    preds = aligned_logits.argmax(dim=-1)
    correct = preds == aligned_labels

    # Gate: correct AND below threshold AND not padding
    gate = correct & (margins < margin_threshold)
    if aligned_labels is not None:
        gate = gate & (aligned_labels != -100)

    if gate.sum() == 0:
        return margins.new_zeros(())

    return -margins.masked_select(gate).mean()


def compute_entropy_penalty(
    logits: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Entropy penalty: sharpen the softmax distribution.

    Minimizes the entropy H(p) = -sum(p log p) of the output distribution,
    pushing probability mass toward the dominant token. This is a geometry-free
    baseline: it widens margins as a side effect of sharpening, without any
    awareness of embedding structure or Fisher metric.

    Returns mean entropy (so minimizing the loss = sharpening the distribution).
    """
    aligned_logits, aligned_labels = _align_for_causal_penalty(logits, labels)

    # Stable log-softmax
    log_probs = F.log_softmax(aligned_logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # [batch, seq]

    # Mask padding
    if aligned_labels is not None:
        valid = aligned_labels != -100
        if valid.sum() == 0:
            return entropy.new_zeros(())
        return entropy.masked_select(valid).mean()

    return entropy.mean()


def compute_fisher_mrp_penalty(
    logits: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    top_k: int = 5,
) -> torch.Tensor:
    """MRP penalty using Fisher information metric distance.

    Instead of cosine distance between embeddings, uses the Fisher-aware
    distance: d_F(i,j)^2 = (w_i - w_j)^T W_k^T Sigma_k W_k (w_i - w_j)
    where Sigma_k = diag(p) - pp^T is the top-k softmax covariance.
    """
    aligned_logits, aligned_labels = _align_for_causal_penalty(logits, labels)
    top_k = min(top_k, aligned_logits.size(-1))

    # Stabilize: shift logits so max is 0 (prevents softmax overflow)
    top_k_logits, top_k_indices = torch.topk(aligned_logits.float(), top_k, dim=-1)
    top_k_logits = top_k_logits - top_k_logits.max(dim=-1, keepdim=True).values
    top_k_probs = F.softmax(top_k_logits, dim=-1)

    top_k_embeds = lm_head_weight[top_k_indices].float()

    # Normalize embeddings to prevent large intermediate values in einsums
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
    eye = torch.eye(top_k, device=aligned_logits.device, dtype=fisher_dist_sq.dtype)
    fisher_dist_sq = fisher_dist_sq * (1.0 - eye)
    fisher_dist = torch.sqrt(fisher_dist_sq.clamp(min=1e-8))

    # Prob-weighted penalty
    prob_outer = top_k_probs.unsqueeze(-1) * top_k_probs.unsqueeze(-2)
    penalty = fisher_dist * prob_outer * (1.0 - eye)
    per_token_penalty = penalty.sum(dim=(-1, -2))

    if aligned_labels is None:
        return per_token_penalty.mean()

    valid_mask = aligned_labels != -100
    if valid_mask.sum() == 0:
        return per_token_penalty.new_zeros(())
    return per_token_penalty.masked_select(valid_mask).mean()


def compute_margin_gated_mrp_penalty(
    logits: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    top_k: int = 5,
    margin_threshold: float = 0.5,
) -> torch.Tensor:
    """MRP penalty applied only to positions with margin below threshold.

    Filters out positions where CE already provides the gradient signal
    (high-uncertainty positions) and focuses on "false confidence" positions
    where the model is confident but the boundary geometry is poor.
    """
    aligned_logits, aligned_labels = _align_for_causal_penalty(logits, labels)
    top_k = min(top_k, aligned_logits.size(-1))
    top_k_logits, top_k_indices = torch.topk(aligned_logits.float(), top_k, dim=-1)
    top_k_probs = F.softmax(top_k_logits, dim=-1)

    # Compute margins to determine the gate
    margins = top_k_logits[..., 0] - top_k_logits[..., 1]  # [batch, seq]

    top_k_embeds = lm_head_weight[top_k_indices]
    normed = F.normalize(top_k_embeds.float(), dim=-1)
    pairwise_sim = torch.matmul(normed, normed.transpose(-1, -2))
    pairwise_dist = 1.0 - pairwise_sim

    prob_outer = top_k_probs.unsqueeze(-1) * top_k_probs.unsqueeze(-2)
    eye = torch.eye(top_k, device=aligned_logits.device, dtype=prob_outer.dtype)
    penalty = pairwise_dist * prob_outer * (1.0 - eye)
    per_token_penalty = penalty.sum(dim=(-1, -2))

    # Gate: only penalize positions with margin below threshold
    margin_mask = margins < margin_threshold
    if aligned_labels is not None:
        margin_mask = margin_mask & (aligned_labels != -100)

    valid_count = int(margin_mask.sum().item())
    if valid_count == 0:
        return per_token_penalty.new_zeros(())
    return per_token_penalty.masked_select(margin_mask).mean()


def compute_depth_mrp_penalty(
    hidden_states: tuple[torch.Tensor, ...],
    lm_head_weight: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    top_k: int = 5,
    target_layers: tuple[int, ...] = (24, 26, 28),
) -> torch.Tensor:
    """MRP penalty computed at intermediate layers, not the final layer.

    Projects intermediate hidden states through lm_head to get "virtual
    logits" at depth, then computes MRP penalty on those. Targets the
    anti-correlated signal where mid-layer ambiguity does not correspond
    to final-layer uncertainty.
    """
    penalties = []
    for layer_idx in target_layers:
        if layer_idx >= len(hidden_states):
            continue

        h = hidden_states[layer_idx]  # [batch, seq, hidden]
        # Project through lm_head to get virtual logits at this depth
        virtual_logits = h.float() @ lm_head_weight.float().T  # [batch, seq, vocab]

        penalty = compute_mrp_penalty(
            virtual_logits,
            lm_head_weight,
            labels=labels,
            top_k=top_k,
        )
        penalties.append(penalty)

    if not penalties:
        return lm_head_weight.new_zeros(())
    return torch.stack(penalties).mean()


# MRP loss mode constants
MRP_MODE_FINAL = "final"
MRP_MODE_MARGIN_GATED = "margin_gated"
MRP_MODE_DEPTH = "depth"
MRP_MODE_COMBINED = "combined"  # depth + margin_gated
MRP_MODE_MARGIN_MAX = "margin_max"  # direct margin maximization
MRP_MODE_CORRECT_MARGIN = "correct_margin"  # only sharpen correct predictions
MRP_MODE_FISHER = "fisher"  # Fisher-metric MRP penalty
MRP_MODE_ENTROPY = "entropy"  # entropy penalty (geometry-free baseline)


class MRPTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        alpha_weight: float,
        mrp_top_k: int,
        mrp_mode: str = MRP_MODE_FINAL,
        mrp_margin_threshold: float = 0.5,
        mrp_target_layers: tuple[int, ...] = (24, 26, 28),
        ce_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alpha_weight = alpha_weight
        self.ce_weight = ce_weight
        self.mrp_top_k = mrp_top_k
        self.mrp_mode = mrp_mode
        self.mrp_margin_threshold = mrp_margin_threshold
        self.mrp_target_layers = mrp_target_layers
        self._latest_loss_components: dict[str, float] = {}

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        need_hidden = self.mrp_mode in (MRP_MODE_DEPTH, MRP_MODE_COMBINED)

        if need_hidden:
            outputs = model(**inputs, output_hidden_states=True)
        else:
            outputs = model(**inputs)

        ce_loss = outputs.loss
        if ce_loss is None:
            raise ValueError("model did not return a cross-entropy loss")

        output_embeddings = resolve_output_embeddings(model)
        if output_embeddings is None or not hasattr(output_embeddings, "weight"):
            raise ValueError("unable to resolve output embeddings for MRP loss")

        lm_head_weight = output_embeddings.weight
        labels = inputs.get("labels")
        components: dict[str, float] = {
            "loss_ce": float(ce_loss.detach().cpu().item()),
        }

        mrp_loss = ce_loss.new_zeros(())

        if self.mrp_mode == MRP_MODE_FINAL:
            mrp_loss = compute_mrp_penalty(
                outputs.logits, lm_head_weight,
                labels=labels, top_k=self.mrp_top_k,
            )

        elif self.mrp_mode == MRP_MODE_MARGIN_GATED:
            mrp_loss = compute_margin_gated_mrp_penalty(
                outputs.logits, lm_head_weight,
                labels=labels, top_k=self.mrp_top_k,
                margin_threshold=self.mrp_margin_threshold,
            )

        elif self.mrp_mode == MRP_MODE_DEPTH:
            mrp_loss = compute_depth_mrp_penalty(
                outputs.hidden_states, lm_head_weight,
                labels=labels, top_k=self.mrp_top_k,
                target_layers=self.mrp_target_layers,
            )

        elif self.mrp_mode == MRP_MODE_COMBINED:
            depth_loss = compute_depth_mrp_penalty(
                outputs.hidden_states, lm_head_weight,
                labels=labels, top_k=self.mrp_top_k,
                target_layers=self.mrp_target_layers,
            )
            gated_loss = compute_margin_gated_mrp_penalty(
                outputs.logits, lm_head_weight,
                labels=labels, top_k=self.mrp_top_k,
                margin_threshold=self.mrp_margin_threshold,
            )
            mrp_loss = depth_loss + gated_loss
            components["loss_mrp_depth"] = float(depth_loss.detach().cpu().item())
            components["loss_mrp_gated"] = float(gated_loss.detach().cpu().item())

        elif self.mrp_mode == MRP_MODE_MARGIN_MAX:
            mrp_loss = compute_margin_loss(
                outputs.logits,
                labels=labels,
                margin_threshold=self.mrp_margin_threshold,
            )

        elif self.mrp_mode == MRP_MODE_CORRECT_MARGIN:
            mrp_loss = compute_correct_margin_loss(
                outputs.logits,
                labels=labels,
                margin_threshold=self.mrp_margin_threshold,
            )

        elif self.mrp_mode == MRP_MODE_FISHER:
            mrp_loss = compute_fisher_mrp_penalty(
                outputs.logits, lm_head_weight,
                labels=labels, top_k=self.mrp_top_k,
            )

        elif self.mrp_mode == MRP_MODE_ENTROPY:
            mrp_loss = compute_entropy_penalty(
                outputs.logits, labels=labels,
            )

        total_loss = (self.ce_weight * ce_loss) + (self.alpha_weight * mrp_loss)
        components["loss_mrp"] = float(mrp_loss.detach().cpu().item())
        components["loss_total"] = float(total_loss.detach().cpu().item())
        components["ce_weight"] = self.ce_weight
        self._latest_loss_components = components

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if self._latest_loss_components:
            merged = dict(logs)
            for key, value in self._latest_loss_components.items():
                merged.setdefault(key, value)
            logs = merged
        super().log(logs, start_time=start_time)


from transformers import TrainerCallback


class _MetricsCallback(TrainerCallback):
    def __init__(self, path: Path, *, run_handle=None, prefix: str = "train") -> None:
        self.path = path
        self.run_handle = run_handle
        self.prefix = prefix
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def on_log(
        self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any
    ) -> None:
        if logs is None:
            return
        import json as _json

        entry = {"step": state.global_step, "epoch": round(state.epoch or 0, 6)}
        entry.update({k: v for k, v in logs.items() if isinstance(v, (int, float))})
        with self.path.open("a") as f:
            f.write(_json.dumps(entry) + "\n")
        if self.run_handle is not None:
            self.run_handle.log_metrics(
                step=int(state.global_step),
                values={
                    f"{self.prefix}/{key}": float(value)
                    for key, value in entry.items()
                    if key != "step" and isinstance(value, (int, float))
                },
                source=self.prefix,
            )


class _GeometricSpotCheckCallback(TrainerCallback):
    """Periodic margin distribution measurement during training.

    Every spot_check_interval steps, runs a forward pass on held-out
    data and computes margin statistics in float32.
    """

    def __init__(
        self,
        path: Path,
        spot_check_data: list[torch.Tensor],
        spot_check_interval: int = 100,
        run_handle=None,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self.spot_check_data = spot_check_data
        self.spot_check_interval = spot_check_interval
        self.run_handle = run_handle

    def on_step_end(
        self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any
    ) -> None:
        if state.global_step % self.spot_check_interval != 0 or model is None:
            return

        import json as _json
        import numpy as _np

        model_was_training = model.training
        model.eval()
        output_emb = resolve_output_embeddings(model)
        if output_emb is None or not hasattr(output_emb, "weight"):
            if model_was_training:
                model.train()
            return
        lm_head_w = output_emb.weight.detach().float()

        all_margins = []
        all_correct = []
        device = next(model.parameters()).device

        with torch.no_grad():
            for seq in self.spot_check_data:
                input_ids = seq.unsqueeze(0).to(device)
                outputs = model(input_ids=input_ids, output_hidden_states=True)
                final_h = outputs.hidden_states[-1][0, :-1, :].float()
                logits = final_h @ lm_head_w.to(device).T
                top2 = torch.topk(logits, 2, dim=-1).values
                margins = (top2[:, 0] - top2[:, 1]).cpu().numpy()
                all_margins.append(margins)
                # Accuracy
                preds = logits.argmax(dim=-1).cpu()
                targets = input_ids[0, 1:].cpu()
                all_correct.append((preds == targets).numpy())

        margins = _np.concatenate(all_margins)
        correct = _np.concatenate(all_correct)
        boundary = margins[margins < 0.25]
        entry = {
            "step": state.global_step,
            "n_positions": len(margins),
            "accuracy": round(float(correct.mean()), 6),
            "median_margin": round(float(_np.median(margins)), 6),
            "pr_lt_0.5": round(float((margins < 0.5).mean()), 6),
            "mean_boundary_margin": round(float(boundary.mean()), 6) if len(boundary) > 0 else None,
            "m_0.05": round(float(_np.quantile(margins, 0.05)), 6),
            "m_0.25": round(float(_np.quantile(margins, 0.25)), 6),
            "low_margin_accuracy": round(float(correct[margins < 0.5].mean()), 6) if (margins < 0.5).sum() > 0 else None,
        }
        with self.path.open("a") as f:
            f.write(_json.dumps(entry) + "\n")
        if self.run_handle is not None:
            self.run_handle.log_metrics(
                step=int(state.global_step),
                values={
                    f"spotcheck/{key}": float(value)
                    for key, value in entry.items()
                    if key != "step" and isinstance(value, (int, float))
                },
                source="spotcheck",
            )

        if model_was_training:
            model.train()


class _TrackerCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir: Path, run_handle) -> None:
        self.output_dir = output_dir
        self.run_handle = run_handle

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.run_handle is None:
            return
        checkpoint_dir = self.output_dir / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            self.run_handle.log_checkpoint(
                checkpoint_dir,
                step=int(state.global_step),
                label=checkpoint_dir.name,
            )


def train_mrp(
    *,
    model_id: str,
    output_dir: str | Path,
    dataset_name: str,
    dataset_config: str | None,
    train_split: str,
    eval_split: str | None,
    text_column: str | None,
    block_size: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    max_steps: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    alpha_weight: float,
    mrp_top_k: int,
    trainable_scope: str,
    torch_dtype: str,
    device: str,
    logging_steps: int,
    save_strategy: str,
    save_steps: int,
    save_model: bool,
    seed: int,
    trust_remote_code: bool,
    gradient_checkpointing: bool = False,
    dataloader_num_workers: int = 0,
    mrp_mode: str = MRP_MODE_FINAL,
    mrp_margin_threshold: float = 0.5,
    mrp_target_layers: str = "24,26,28",
    spot_check_interval: int = 100,
    trainable_last_n: int = 1,
    ce_weight: float = 1.0,
) -> dict[str, Any]:
    try:
        import accelerate  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "training requires `accelerate`. Run `uv sync --extra train` first."
        ) from exc

    output_root = ensure_dir(output_dir)
    runtime_device = _resolve_device(device)
    resolved_dtype = _normalize_dtype(torch_dtype)
    run_handle = start_run(
        output_dir=output_root,
        name=output_root.name,
        run_type="train",
        config={
            "model_id": model_id,
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "train_split": train_split,
            "eval_split": eval_split,
            "block_size": block_size,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "alpha_weight": alpha_weight,
            "mrp_mode": mrp_mode,
            "mrp_margin_threshold": mrp_margin_threshold,
            "mrp_target_layers": mrp_target_layers,
            "mrp_top_k": mrp_top_k,
            "trainable_scope": trainable_scope,
            "trainable_last_n": trainable_last_n,
            "dtype": resolved_dtype,
            "device": str(runtime_device),
        },
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        loaded = load_text_model(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=resolved_dtype,
        )
        model = loaded.model
        model.config.use_cache = False

        if gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        trainable_summary = _apply_trainable_scope(
            model,
            trainable_scope,
            trainable_last_n=trainable_last_n,
        )
        if (
            runtime_device.type == "cpu"
            and trainable_summary.trainable_parameters > DEFAULT_CPU_TRAINABLE_PARAM_LIMIT
        ):
            raise RuntimeError(
                "selected trainable scope is too large for a realistic CPU run. "
                "Use `final_norm` or `last_n_blocks --trainable-last-n 1` for a local smoke test, "
                "or move to GPU/cloud."
            )

        train_dataset, resolved_text_column = _prepare_lm_dataset(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=train_split,
            text_column=text_column,
            block_size=block_size,
            max_samples=max_train_samples,
        )
        eval_dataset = None
        if eval_split:
            eval_dataset, _ = _prepare_lm_dataset(
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split=eval_split,
                text_column=resolved_text_column,
                block_size=block_size,
                max_samples=max_eval_samples,
            )

        warmup_steps = max(0, math.ceil(max_steps * warmup_ratio))

        use_bf16 = resolved_dtype == "bfloat16" and runtime_device.type != "cpu"
        use_fp16 = resolved_dtype == "float16" and runtime_device.type != "cpu"

        training_args = TrainingArguments(
            output_dir=str(output_root),
            do_train=True,
            do_eval=eval_dataset is not None,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            optim="adamw_torch",
            bf16=use_bf16,
            fp16=use_fp16,
            tf32=(runtime_device.type != "cpu"),
            gradient_checkpointing=gradient_checkpointing,
            logging_strategy="steps",
            logging_steps=logging_steps,
            logging_first_step=True,
            save_strategy=save_strategy,
            save_steps=save_steps,
            eval_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            label_names=["labels"],
            seed=seed,
            data_seed=seed,
            use_cpu=(runtime_device.type == "cpu"),
            dataloader_pin_memory=(runtime_device.type != "cpu"),
            dataloader_num_workers=dataloader_num_workers,
            disable_tqdm=False,
        )

        parsed_target_layers = tuple(int(x) for x in mrp_target_layers.split(","))
        metrics_callback = _MetricsCallback(
            output_root / "metrics.jsonl",
            run_handle=run_handle,
            prefix="train",
        )

        # Prepare held-out data for geometric spot checks
        callbacks = [metrics_callback, _TrackerCheckpointCallback(output_root, run_handle)]
        if spot_check_interval > 0:
            spot_check_ds = load_dataset(
                dataset_name, dataset_config, split="validation" if eval_split else "train",
            )
            spot_text_col = _detect_text_column(spot_check_ds.column_names, text_column)
            spot_texts = [r[spot_text_col] for r in spot_check_ds if isinstance(r.get(spot_text_col), str) and r[spot_text_col].strip()][:50]
            # Tokenize individually — no padding, so no pad tokens in margin stats
            spot_sequences = []
            for text in spot_texts[:20]:
                enc = tokenizer(
                    text, return_tensors="pt", max_length=block_size,
                    truncation=True, add_special_tokens=False,
                )
                if enc["input_ids"].size(1) >= 4:
                    spot_sequences.append(enc["input_ids"][0])
            spot_data = spot_sequences  # list of variable-length tensors
            callbacks.append(_GeometricSpotCheckCallback(
                path=output_root / "geometric_spotcheck.jsonl",
                spot_check_data=spot_data,
                spot_check_interval=spot_check_interval,
                run_handle=run_handle,
            ))

        trainer = MRPTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            alpha_weight=alpha_weight,
            ce_weight=ce_weight,
            mrp_top_k=mrp_top_k,
            mrp_mode=mrp_mode,
            mrp_margin_threshold=mrp_margin_threshold,
            mrp_target_layers=parsed_target_layers,
        )

        train_result = trainer.train()
        trainer.save_state()

        eval_metrics: dict[str, Any] | None = None
        if eval_dataset is not None:
            eval_metrics = trainer.evaluate()

        if save_model:
            trainer.save_model(str(output_root / "final_model"))

        summary = {
            "model_id": model_id,
            "load_strategy": loaded.load_strategy,
            "output_dir": str(output_root.resolve()),
            "device": str(runtime_device),
            "dtype": resolved_dtype,
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "train_split": train_split,
            "eval_split": eval_split,
            "text_column": resolved_text_column,
            "block_size": block_size,
            "max_train_samples": max_train_samples,
            "max_eval_samples": max_eval_samples,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "alpha_weight": alpha_weight,
            "mrp_mode": mrp_mode,
            "mrp_margin_threshold": mrp_margin_threshold,
            "mrp_target_layers": list(parsed_target_layers),
            "mrp_top_k": mrp_top_k,
            "bf16": use_bf16,
            "fp16": use_fp16,
            "gradient_checkpointing": gradient_checkpointing,
            "trainable": asdict(trainable_summary),
            "train_dataset_blocks": len(train_dataset),
            "eval_dataset_blocks": None if eval_dataset is None else len(eval_dataset),
            "train_metrics": {
                key: (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value)
                for key, value in train_result.metrics.items()
            },
            "eval_metrics": (
                None
                if eval_metrics is None
                else {
                    key: (
                        float(value)
                        if isinstance(value, (int, float)) and not isinstance(value, bool)
                        else value
                    )
                    for key, value in eval_metrics.items()
                }
            ),
            "log_history": trainer.state.log_history,
            "save_model": save_model,
        }
        write_json(output_root / "train_summary.json", summary)
        run_handle.log_artifact("train_summary", output_root / "train_summary.json", kind="summary")
        run_handle.log_artifact("metrics_jsonl", output_root / "metrics.jsonl", kind="metrics")
        if (output_root / "geometric_spotcheck.jsonl").exists():
            run_handle.log_artifact(
                "geometric_spotcheck_jsonl",
                output_root / "geometric_spotcheck.jsonl",
                kind="metrics",
            )
        if save_model and (output_root / "final_model").exists():
            run_handle.log_artifact("final_model", output_root / "final_model", kind="model")
        run_handle.finish(status="completed", summary=summary)
        return summary
    except Exception as exc:
        if run_handle.status == "running":
            run_handle.log_event(
                "train_failed",
                payload={"error": f"{type(exc).__name__}: {exc}"},
            )
            run_handle.finish(
                status="failed",
                summary={"error": f"{type(exc).__name__}: {exc}"},
            )
        raise


# ── From-scratch training ──


class _StreamingBlockDataset(IterableDataset):
    """Stream from HuggingFace, tokenize and pack into fixed-length blocks."""

    def __init__(self, tokenizer, dataset: str, config: str | None,
                 split: str, block_size: int):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config
        self.split = split
        self.block_size = block_size

    def __iter__(self):
        ds = load_dataset(self.dataset, self.config, split=self.split, streaming=True)
        buffer: list[int] = []
        for example in ds:
            text = example.get("text", "")
            if not text.strip():
                continue
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            buffer.extend(ids)
            while len(buffer) >= self.block_size:
                block = buffer[: self.block_size]
                buffer = buffer[self.block_size :]
                t = torch.tensor(block, dtype=torch.long)
                yield {"input_ids": t, "labels": t}


class _NeuroAnalogCallback(TrainerCallback):
    """Log neuro-analog diagnostic metrics (loops used, active fraction, gate value)."""

    def __init__(self, trainer_ref=None):
        """Optionally receive a ref to the trainer for pre-clip grad logging."""
        super().__init__()
        self._trainer_ref = trainer_ref

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        inner = model.module if hasattr(model, "module") else model
        if hasattr(inner, "set_current_train_step"):
            inner.set_current_train_step(int(state.global_step))

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if model is None or logs is None:
            return
        # The model may be wrapped (e.g., by accelerate)
        inner = model.module if hasattr(model, "module") else model
        # For FactoredCausalLM, loop diagnostics are on inner.model
        backbone = getattr(inner, "model", inner)
        # Pre-clip grad norms (if trainer tracks them)
        if self._trainer_ref is not None:
            pre_shared = getattr(self._trainer_ref, "_last_pre_clip_shared", None)
            pre_other = getattr(self._trainer_ref, "_last_pre_clip_other", None)
            if pre_shared is not None and pre_shared > 0:
                logs["grad_pre_clip_shared"] = pre_shared
            if pre_other is not None and pre_other > 0:
                logs["grad_pre_clip_other"] = pre_other
            # Current max_grad_norm ceiling (supervisor may be tuning it).
            # 0.0 means global clipping disabled (factored+normalize-shared uses
            # per-component clipping instead).
            args_obj = getattr(self._trainer_ref, "args", None)
            if args_obj is not None:
                mgn = float(getattr(args_obj, "max_grad_norm", 0.0))
                if mgn > 0:
                    logs["max_grad_norm"] = mgn
        if hasattr(backbone, "_last_loops_used"):
            logs["loops_used"] = backbone._last_loops_used
            logs["active_frac"] = backbone._last_active_frac
            if backbone._last_gate_val is not None:
                logs["gate_val"] = backbone._last_gate_val
        # Looped reasoning diagnostics (per-loop exit fractions + per-loop CE)
        if hasattr(backbone, "_last_loop0_exit_frac"):
            logs["loop0_exit_frac"] = backbone._last_loop0_exit_frac
            logs["loop1_exit_frac"] = backbone._last_loop1_exit_frac
            logs["loop2_active_frac"] = backbone._last_loop2_active_frac
            # Expose current margin threshold (supervisor may be tuning it)
            if hasattr(backbone, "loop1_margin_threshold"):
                logs["loop1_margin_threshold"] = float(backbone.loop1_margin_threshold.item())
        per_loop_ce = getattr(inner, "_last_per_loop_ce", None)
        if per_loop_ce:
            for i, ce in enumerate(per_loop_ce):
                logs[f"loop{i}_ce"] = ce
            finite_ce = [float(ce) for ce in per_loop_ce if isinstance(ce, (int, float)) and math.isfinite(ce)]
            if len(finite_ce) >= 2:
                # Positive means later loops are outperforming loop 0; negative
                # means refinement is regressing the representation.
                logs["loop_spread"] = finite_ce[0] - finite_ce[-1]
        # Episodic buffer diagnostics
        buffer = getattr(backbone, "episodic_buffer", None)
        if buffer is not None:
            logs["buffer_writes_last_step"] = int(getattr(inner, "_last_buffer_writes", 0))
            logs["buffer_utilization"] = float(buffer.utilization())
            logs["buffer_reject_rate"] = float(buffer.reject_rate())
            logs["buffer_total_written"] = int(buffer.n_written.item())
        # Inter-loop prediction losses (looped model's cosine version)
        il_losses = getattr(inner, "_last_interloop_losses", None)
        if il_losses:
            for i, loss_val in enumerate(il_losses):
                logs[f"interloop_cos_{i}_to_{i + 1}"] = loss_val
        # Homeostatic firing rate diagnostics
        hfr = getattr(backbone, "homeostatic_firing_rate", None)
        if hfr is not None:
            hfr_f = hfr.float()
            if hfr_f.ndim >= 2:
                per_loop_std = hfr_f.std(dim=-1)
                # Use mean per-loop std so the metric stays comparable to the
                # pre-v11 single-vector statistic while still exposing loop
                # spread extremes for debugging.
                logs["homeostatic_firing_rate_std"] = float(per_loop_std.mean().item())
                logs["homeostatic_firing_rate_std_max"] = float(per_loop_std.max().item())
                logs["homeostatic_firing_rate_std_min"] = float(per_loop_std.min().item())
            else:
                logs["homeostatic_firing_rate_std"] = float(hfr_f.std().item())
            logs["homeostatic_firing_rate_min"] = float(hfr_f.min().item())
            logs["homeostatic_firing_rate_max"] = float(hfr_f.max().item())
            cfg = getattr(backbone, "config", None)
            if cfg is not None:
                logs["homeostatic_alpha"] = float(getattr(cfg, "homeostatic_alpha", 1.0))
                logs["homeostatic_tolerance"] = float(getattr(cfg, "homeostatic_tolerance", 1.0))
        # Surprise head diagnostics
        surprise_heads = getattr(backbone, "surprise_heads", None)
        surprise_head = getattr(backbone, "surprise_head", None)
        if surprise_heads is not None or surprise_head is not None:
            aux_loss = getattr(inner, "_last_surprise_aux_loss", None)
            if aux_loss is not None:
                logs["surprise_aux_loss"] = float(aux_loss.detach())
            logs["surprise_p_est_mean"] = float(getattr(inner, "_last_surprise_p_est_mean", 0.0))
            surprise_exit_thresholds = getattr(backbone, "surprise_exit_thresholds", None)
            if surprise_exit_thresholds is not None:
                for i, threshold in enumerate(surprise_exit_thresholds):
                    logs[f"surprise_exit_threshold_{i}"] = float(threshold.item())
            else:
                cfg = getattr(backbone, "config", None)
                if cfg is not None:
                    logs["surprise_exit_threshold"] = float(
                        getattr(cfg, "surprise_exit_threshold", 0.0)
                    )
        # Velocity clip diagnostics
        if self._trainer_ref is not None and getattr(self._trainer_ref, "_velocity_clip", False):
            logs["velocity_clips_applied"] = int(getattr(self._trainer_ref, "_velocity_clips_applied", 0))
            pre = float(getattr(self._trainer_ref, "_last_velocity_clip_pre", 0.0))
            post = float(getattr(self._trainer_ref, "_last_velocity_clip_post", 0.0))
            if pre > 0:
                logs["velocity_clip_last_pre"] = pre
                logs["velocity_clip_last_post"] = post
            prev = getattr(self._trainer_ref, "_prev_grad_norm", None)
            if prev is not None:
                logs["velocity_clip_prev_grad_norm"] = float(prev)
        # MTP per-head losses live on the CausalLM wrapper (inner), not backbone.
        # Each entry is (head_idx, loss_val_or_None). head_idx=0 predicts t+2, etc.
        # None indicates the head was skipped (short sequence or NaN guard).
        mtp_losses = getattr(inner, "_mtp_head_losses", None)
        if mtp_losses:
            for entry in mtp_losses:
                # Backward-compat: older code stored bare floats
                if isinstance(entry, tuple):
                    head_idx, loss_val = entry
                    if loss_val is not None:
                        logs[f"mtp_head{head_idx + 2}_loss"] = loss_val
        mtp_scale = getattr(inner, "_last_mtp_loss_scale", None)
        if mtp_scale is not None:
            logs["mtp_loss_scale"] = float(mtp_scale)
        # Inter-loop predictive coding: per-transition MSE losses.
        # interloop_pred_losses[i] is the MSE between predictor(loop_i) and
        # loop_{i+1}. Higher values = predictor is surprised = loops are doing
        # more distinct things. Diverging over training = loop specialization.
        interloop_losses = getattr(inner, "_interloop_pred_losses", None)
        if interloop_losses:
            for i, loss_val in enumerate(interloop_losses):
                logs[f"interloop_pred_{i}_to_{i + 1}"] = loss_val
        # Sparse coding: report dead-dim count and reset accumulator.
        # _dims_active_mask is a bool tensor [hidden_dim], True for any dim
        # that was active in the window since last log.
        dims_mask = getattr(backbone, "_dims_active_mask", None)
        if dims_mask is not None:
            total_dims = int(dims_mask.numel())
            live_dims = int(dims_mask.sum().item())
            logs["sparsity_dead_dims"] = total_dims - live_dims
            logs["sparsity_live_frac"] = live_dims / max(total_dims, 1)
            # Reset for next window
            backbone._dims_active_mask = None


class _SupervisorCallback(TrainerCallback):
    """Minimal adaptive supervisor. Watches diagnostics, makes narrow adjustments.

    Initial lever set (safe, reversible, well-understood):
      1. Dead-dim monitor → adjust sparsity_k on the model (directly observed metric)
      2. Emergency LR halving on grad spike >3x running mean (v1-style disaster prevention)
      3. Plateau detection (log-only flag, no action taken)
      4. Adaptive grad-norm ceiling (opt-in via tune_grad_clip=True) —
         raises max_grad_norm by 20% when median grad_norm exceeds 1.5x
         the current ceiling (clipping is throttling signal). Raise-only,
         rate-limited (+20%/100 steps), hard-capped at _clip_max. Intended
         for architectures with built-in control mechanisms (per-position
         gating) that can tolerate larger gradient magnitudes.

    Every adjustment is logged to adaptations.jsonl with cooldowns and
    oscillation detection. Bounds: sparsity_k in [0.05, 0.5], LR never
    doubled back up (only halved), max_grad_norm monotonically rising,
    capped at 10.0.
    """

    def __init__(
        self,
        adaptations_path: Path,
        trainer_ref=None,
        tune_grad_clip: bool = False,
        tune_loop_thresholds: bool = False,
        loop_threshold_warmup_steps: int = 200,
        loop_threshold_interval: int = 100,
        loop0_exit_target_low: float = 0.02,
        loop0_exit_target_high: float = 0.15,
        loop1_exit_target_low: float = 0.05,
        loop1_exit_target_high: float = 0.30,
        surprise_exit_threshold_min: float = 0.35,
        surprise_exit_threshold_max: float = 0.80,
        surprise_exit_threshold_step: float = 0.05,
        loop1_margin_threshold_min: float = 0.02,
        loop1_margin_threshold_max: float = 0.30,
        loop1_margin_threshold_step: float = 0.02,
    ):
        super().__init__()
        self.adaptations_path = adaptations_path
        self._trainer_ref = trainer_ref
        self._tune_grad_clip = tune_grad_clip
        self._tune_loop_thresholds = tune_loop_thresholds
        self._loop_threshold_warmup_steps = loop_threshold_warmup_steps
        self._loop_threshold_interval = loop_threshold_interval
        self._loop0_exit_target_low = loop0_exit_target_low
        self._loop0_exit_target_high = loop0_exit_target_high
        self._loop1_exit_target_low = loop1_exit_target_low
        self._loop1_exit_target_high = loop1_exit_target_high
        self._surprise_exit_threshold_min = surprise_exit_threshold_min
        self._surprise_exit_threshold_max = surprise_exit_threshold_max
        self._surprise_exit_threshold_step = surprise_exit_threshold_step
        self._loop1_margin_threshold_min = loop1_margin_threshold_min
        self._loop1_margin_threshold_max = loop1_margin_threshold_max
        self._loop1_margin_threshold_step = loop1_margin_threshold_step
        # Running state
        self._grad_norm_history: list[float] = []
        self._loss_history: list[float] = []
        self._loop2_ce_history: list[float] = []
        self._loop_spread_history: list[float] = []
        self._exit_frac_history: list[float] = []
        self._live_frac_history: list[float] = []
        self._homeo_std_history: list[float] = []
        self._last_sparsity_adjust_step: int = -1000
        self._last_lr_adjust_step: int = -1000
        self._last_clip_adjust_step: int = -1000
        self._last_homeo_adjust_step: int = -1000
        self._last_threshold_adjust_step: int = -1000
        self._sparsity_adjustments: list[tuple[int, float, float]] = []  # (step, old, new)
        self._clip_adjustments: list[tuple[int, float, float]] = []
        self._homeo_adjustments: list[tuple[int, float, float]] = []
        self._threshold_adjustments: list[tuple[int, float, float]] = []
        self._plateau_flag_step: int = -1000
        # Bounds
        self._sparsity_min = 0.05
        self._sparsity_max = 0.5
        self._clip_min = 0.1
        self._clip_max = 50.0
        self._cooldown = 100  # steps between adjustments to same lever
        # Warmup: wait for grad_norm to stabilize before tuning
        self._clip_warmup_steps = 100
        # Minimum history entries before first adjustment (log-entries, not steps)
        self._clip_history_min = 10
        # Write header on first call
        self._header_written = False

    def _log(self, event: dict) -> None:
        """Append one JSON line to adaptations.jsonl."""
        self.adaptations_path.parent.mkdir(parents=True, exist_ok=True)
        with self.adaptations_path.open("a") as f:
            f.write(_stdlib_json.dumps(event) + "\n")

    def _oscillating(self, recent_adjustments: list) -> bool:
        """Detect 3+ reversals in direction among last 5 adjustments."""
        if len(recent_adjustments) < 3:
            return False
        deltas = [new - old for _, old, new in recent_adjustments[-5:]]
        reversals = sum(
            1 for i in range(1, len(deltas)) if deltas[i] * deltas[i - 1] < 0
        )
        return reversals >= 3

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if model is None or logs is None:
            return
        step = int(state.global_step)

        # Track recent grad_norm and loss
        grad_norm = logs.get("grad_norm")
        loss_val = logs.get("loss")
        if isinstance(grad_norm, (int, float)):
            self._grad_norm_history.append(float(grad_norm))
            if len(self._grad_norm_history) > 50:
                self._grad_norm_history.pop(0)
        if isinstance(loss_val, (int, float)):
            self._loss_history.append(float(loss_val))
            if len(self._loss_history) > 200:
                self._loss_history.pop(0)
        loop2_ce = logs.get("loop2_ce")
        if isinstance(loop2_ce, (int, float)):
            self._loop2_ce_history.append(float(loop2_ce))
            if len(self._loop2_ce_history) > 40:
                self._loop2_ce_history.pop(0)
        loop_spread = logs.get("loop_spread")
        if isinstance(loop_spread, (int, float)):
            self._loop_spread_history.append(float(loop_spread))
            if len(self._loop_spread_history) > 40:
                self._loop_spread_history.pop(0)
        exit_frac = 0.0
        loop0_exit = logs.get("loop0_exit_frac")
        loop1_exit = logs.get("loop1_exit_frac")
        if isinstance(loop0_exit, (int, float)):
            exit_frac += float(loop0_exit)
        if isinstance(loop1_exit, (int, float)):
            exit_frac += float(loop1_exit)
        self._exit_frac_history.append(exit_frac)
        if len(self._exit_frac_history) > 40:
            self._exit_frac_history.pop(0)
        live_frac_hist = logs.get("sparsity_live_frac")
        if isinstance(live_frac_hist, (int, float)):
            self._live_frac_history.append(float(live_frac_hist))
            if len(self._live_frac_history) > 40:
                self._live_frac_history.pop(0)
        homeo_std = logs.get("homeostatic_firing_rate_std")
        if isinstance(homeo_std, (int, float)):
            self._homeo_std_history.append(float(homeo_std))
            if len(self._homeo_std_history) > 40:
                self._homeo_std_history.pop(0)

        # --- Lever 1: Dead-dim monitor → adjust sparsity_k ---
        # Use <= 1.0 so the decrease-k branch can fire at live_frac=1.0
        # (the common healthy case). Without this, the decrease-toward-
        # sparsity logic would be dead code.
        live_frac = logs.get("sparsity_live_frac")
        if isinstance(live_frac, (int, float)) and live_frac <= 1.0:
            inner = model.module if hasattr(model, "module") else model
            backbone = getattr(inner, "model", inner)
            cfg = getattr(backbone, "config", None)
            if cfg is not None and getattr(cfg, "sparsity_enabled", False):
                cooldown_ok = step - self._last_sparsity_adjust_step >= self._cooldown
                if cooldown_ok and not self._oscillating(self._sparsity_adjustments):
                    old_k = float(cfg.sparsity_k)
                    new_k = old_k
                    # Too many dead dims → increase k (more active)
                    if live_frac < 0.8 and old_k < self._sparsity_max:
                        new_k = min(old_k + 0.02, self._sparsity_max)
                    # No dead dims + loss plateau → decrease k (more sparse)
                    elif (live_frac >= 0.999 and self._is_plateau() and
                          old_k > self._sparsity_min):
                        new_k = max(old_k - 0.02, self._sparsity_min)
                    if new_k != old_k:
                        cfg.sparsity_k = new_k
                        self._last_sparsity_adjust_step = step
                        self._sparsity_adjustments.append((step, old_k, new_k))
                        self._log({
                            "step": step, "lever": "sparsity_k",
                            "old": old_k, "new": new_k,
                            "reason": f"live_frac={live_frac:.3f}",
                        })

        # --- Lever 2: Emergency LR halving on grad spike ---
        if len(self._grad_norm_history) >= 20 and grad_norm is not None:
            recent_mean = sum(self._grad_norm_history[-20:-1]) / 19
            if recent_mean > 0 and float(grad_norm) > 3.0 * recent_mean:
                cooldown_ok = step - self._last_lr_adjust_step >= self._cooldown
                if cooldown_ok and self._trainer_ref is not None:
                    # Halve LR for next 10 steps by scaling all param groups
                    opt = getattr(self._trainer_ref, "optimizer", None)
                    if opt is not None:
                        for g in opt.param_groups:
                            old_lr = g.get("lr", 0)
                            g["lr"] = old_lr * 0.5
                        self._last_lr_adjust_step = step
                        self._log({
                            "step": step, "lever": "lr_halve",
                            "grad_norm": float(grad_norm),
                            "recent_mean": recent_mean,
                            "reason": "grad_spike_3x",
                        })

        # --- Lever 2.5: Adaptive homeostasis ---
        inner = model.module if hasattr(model, "module") else model
        backbone = getattr(inner, "model", inner)
        cfg = getattr(backbone, "config", None)
        if (cfg is not None
                and getattr(cfg, "adaptive_homeostasis", False)
                and getattr(cfg, "sparsity_homeostatic", False)):
            interval = int(getattr(cfg, "adaptive_homeostasis_interval", self._cooldown))
            warmup = int(getattr(cfg, "adaptive_homeostasis_warmup_steps", 100))
            cooldown_ok = step - self._last_homeo_adjust_step >= interval
            safety_ready = step >= warmup and cooldown_ok
            have_specialization_windows = (
                len(self._loop2_ce_history) >= 20
                and len(self._loop_spread_history) >= 20
            )
            if safety_ready and not self._oscillating(self._homeo_adjustments):
                prev_grad_window = self._grad_norm_history[-20:-10] if len(self._grad_norm_history) >= 20 else []
                current_grad = float(grad_norm) if isinstance(grad_norm, (int, float)) else None
                prev_grad_median = None
                if prev_grad_window:
                    prev_sorted = sorted(prev_grad_window)
                    prev_grad_median = prev_sorted[len(prev_sorted) // 2]
                live_frac_recent = self._live_frac_history[-1] if self._live_frac_history else None
                homeo_std_recent = self._homeo_std_history[-1] if self._homeo_std_history else None
                exit_frac_recent = self._exit_frac_history[-1] if self._exit_frac_history else 0.0

                old_alpha = float(getattr(cfg, "homeostatic_alpha", 1.0))
                old_tol = float(getattr(cfg, "homeostatic_tolerance", 1.0))
                new_alpha = old_alpha
                new_tol = old_tol
                reason = None
                rel_drop = None
                spread_gain = None

                collapse_risk = (
                    (live_frac_recent is not None and live_frac_recent < 0.95)
                    or (homeo_std_recent is not None and homeo_std_recent > 0.030)
                    or (
                        current_grad is not None
                        and prev_grad_median is not None
                        and prev_grad_median > 0
                        and current_grad > 1.8 * prev_grad_median
                    )
                )
                if collapse_risk:
                    new_alpha = min(
                        float(getattr(cfg, "homeostatic_alpha_max", 1.0)),
                        old_alpha + 0.10,
                    )
                    new_tol = max(
                        float(getattr(cfg, "homeostatic_tolerance_min", 1.75)),
                        old_tol - 0.25,
                    )
                    reason = (
                        "safety: tighten homeostasis to resist collapse/instability "
                        f"(live_frac={live_frac_recent}, homeo_std={homeo_std_recent}, "
                        f"grad_norm={current_grad}, prev_grad_median={prev_grad_median})"
                    )
                elif have_specialization_windows:
                    prev_ce = self._loop2_ce_history[-20:-10]
                    curr_ce = self._loop2_ce_history[-10:]
                    prev_spread = self._loop_spread_history[-20:-10]
                    curr_spread = self._loop_spread_history[-10:]
                    prev_ce_mean = sum(prev_ce) / len(prev_ce)
                    curr_ce_mean = sum(curr_ce) / len(curr_ce)
                    rel_drop = (
                        (prev_ce_mean - curr_ce_mean) / max(abs(prev_ce_mean), 1e-8)
                    )
                    spread_gain = (sum(curr_spread) / len(curr_spread)) - (sum(prev_spread) / len(prev_spread))
                    under_specialized = (
                        exit_frac_recent < 0.005
                        and spread_gain < 0.003
                        and (homeo_std_recent is not None and homeo_std_recent < 0.020)
                        and (live_frac_recent is not None and live_frac_recent > 0.99)
                    )
                    if under_specialized:
                        new_alpha = max(
                            float(getattr(cfg, "homeostatic_alpha_min", 0.25)),
                            old_alpha - 0.05,
                        )
                        new_tol = min(
                            float(getattr(cfg, "homeostatic_tolerance_max", 4.0)),
                            old_tol + 0.25,
                        )
                        reason = (
                            "specialization: relax homeostasis because loops remain "
                            f"undifferentiated (rel_drop={rel_drop:.4f}, spread_gain={spread_gain:.4f}, "
                            f"exit_frac={exit_frac_recent:.4f}, homeo_std={homeo_std_recent})"
                        )

                if new_alpha != old_alpha or new_tol != old_tol:
                    if new_alpha != old_alpha:
                        cfg.homeostatic_alpha = new_alpha
                        self._homeo_adjustments.append((step, old_alpha, new_alpha))
                    if new_tol != old_tol:
                        cfg.homeostatic_tolerance = new_tol
                    self._last_homeo_adjust_step = step
                    self._log({
                        "step": step,
                        "lever": "homeostasis",
                        "alpha_old": old_alpha,
                        "alpha_new": new_alpha,
                        "tolerance_old": old_tol,
                        "tolerance_new": new_tol,
                        "rel_drop": rel_drop,
                        "spread_gain": spread_gain,
                        "exit_frac": exit_frac_recent,
                        "live_frac": live_frac_recent,
                        "homeo_std": homeo_std_recent,
                        "grad_norm": current_grad,
                        "reason": reason,
                    })

        # --- Lever 4: Adaptive grad-norm ceiling (opt-in, raise-only) ---
        # The reported grad_norm is pre-clip. If median grad_norm consistently
        # exceeds the current max_grad_norm, clipping is throttling signal
        # and we raise the ceiling.
        #
        # Two modes:
        #   Catch-up: median > 3x clip → jump clip to median/2 in one step.
        #     Handles dramatic magnitude mismatches (e.g., initial clip set
        #     too low vs actual gradient scale). Saves 10+ cooldown cycles.
        #   Fine-tune: median > 1.5x clip → raise by 20%.
        #
        # Raise-only: we never lower the ceiling automatically.
        # Rate-limited: 100-step cooldown between adjustments.
        # Hard cap: self._clip_max (default 50.0).
        if (self._tune_grad_clip and self._trainer_ref is not None
                and len(self._grad_norm_history) >= self._clip_history_min
                and step >= self._clip_warmup_steps):
            cooldown_ok = step - self._last_clip_adjust_step >= self._cooldown
            if cooldown_ok:
                args_obj = getattr(self._trainer_ref, "args", None)
                if args_obj is not None:
                    current_clip = float(getattr(args_obj, "max_grad_norm", 1.0))
                    if current_clip > 0 and current_clip < self._clip_max:
                        window = self._grad_norm_history[-self._clip_history_min:]
                        sorted_window = sorted(window)
                        median_gn = sorted_window[len(sorted_window) // 2]
                        new_clip = current_clip
                        reason = None
                        # Catch-up: dramatic mismatch → jump to median/2
                        if median_gn > 3.0 * current_clip:
                            new_clip = min(median_gn / 2.0, self._clip_max)
                            reason = (
                                f"catch-up: median_gn={median_gn:.2f} > 3x clip={current_clip:.2f}"
                                f" → jump to median/2"
                            )
                        # Fine-tune: modest throttling
                        elif median_gn > 1.5 * current_clip:
                            new_clip = min(current_clip * 1.2, self._clip_max)
                            reason = f"throttling: median_gn={median_gn:.2f} > 1.5x clip={current_clip:.2f}"
                        if new_clip != current_clip and reason is not None:
                            args_obj.max_grad_norm = new_clip
                            self._last_clip_adjust_step = step
                            self._clip_adjustments.append((step, current_clip, new_clip))
                            self._log({
                                "step": step, "lever": "max_grad_norm",
                                "old": current_clip, "new": new_clip,
                                "reason": reason,
                            })

        # --- Lever 5: Adaptive loop-gate thresholds ---
        if self._tune_loop_thresholds:
            inner = model.module if hasattr(model, "module") else model
            backbone = getattr(inner, "model", inner)
            cooldown_ok = (
                step - self._last_threshold_adjust_step
                >= self._loop_threshold_interval
            )
            if (cooldown_ok and step >= self._loop_threshold_warmup_steps
                    and not self._oscillating(self._threshold_adjustments)):
                loop0_exit = logs.get("loop0_exit_frac")
                loop1_exit = logs.get("loop1_exit_frac")
                threshold_changes: list[dict[str, float | int]] = []
                surprise_exit_thresholds = getattr(backbone, "surprise_exit_thresholds", None)
                if surprise_exit_thresholds is not None:
                    gate_specs = [
                        (0, loop0_exit, self._loop0_exit_target_low, self._loop0_exit_target_high),
                        (1, loop1_exit, self._loop1_exit_target_low, self._loop1_exit_target_high),
                    ]
                    for gate_idx, exit_val, target_low, target_high in gate_specs:
                        if gate_idx >= len(surprise_exit_thresholds):
                            continue
                        if not isinstance(exit_val, (int, float)):
                            continue
                        old = float(surprise_exit_thresholds[gate_idx].item())
                        new = old
                        if exit_val < target_low:
                            new = max(
                                self._surprise_exit_threshold_min,
                                old - self._surprise_exit_threshold_step,
                            )
                        elif exit_val > target_high:
                            new = min(
                                self._surprise_exit_threshold_max,
                                old + self._surprise_exit_threshold_step,
                            )
                        if new != old:
                            surprise_exit_thresholds[gate_idx].fill_(new)
                            self._threshold_adjustments.append((step, old, new))
                            threshold_changes.append({
                                "gate": gate_idx,
                                "old": old,
                                "new": new,
                                "exit_frac": float(exit_val),
                                "target_low": float(target_low),
                                "target_high": float(target_high),
                            })
                elif hasattr(backbone, "loop1_margin_threshold"):
                    if isinstance(loop1_exit, (int, float)):
                        old = float(backbone.loop1_margin_threshold.item())
                        new = old
                        if loop1_exit < self._loop1_exit_target_low:
                            new = max(
                                self._loop1_margin_threshold_min,
                                old - self._loop1_margin_threshold_step,
                            )
                        elif loop1_exit > self._loop1_exit_target_high:
                            new = min(
                                self._loop1_margin_threshold_max,
                                old + self._loop1_margin_threshold_step,
                            )
                        if new != old:
                            backbone.loop1_margin_threshold.fill_(new)
                            self._threshold_adjustments.append((step, old, new))
                            threshold_changes.append({
                                "gate": 1,
                                "old": old,
                                "new": new,
                                "exit_frac": float(loop1_exit),
                                "target_low": float(self._loop1_exit_target_low),
                                "target_high": float(self._loop1_exit_target_high),
                            })
                if threshold_changes:
                    self._last_threshold_adjust_step = step
                    self._log({
                        "step": step,
                        "lever": "loop_thresholds",
                        "changes": threshold_changes,
                        "reason": "steer gate exits into target bands",
                    })

        # --- Lever 3: Plateau detection (log only) ---
        if self._is_plateau() and step - self._plateau_flag_step > 500:
            self._plateau_flag_step = step
            self._log({
                "step": step, "lever": "plateau_flag",
                "loss_range_200": (
                    max(self._loss_history) - min(self._loss_history)
                    if self._loss_history else 0
                ),
                "reason": "loss_flat_500steps",
            })

    def _is_plateau(self) -> bool:
        """Loss has moved <1% over last 200 logged steps."""
        if len(self._loss_history) < 200:
            return False
        window = self._loss_history[-200:]
        if not window:
            return False
        lo, hi = min(window), max(window)
        if hi == 0:
            return False
        return (hi - lo) / abs(hi) < 0.01


class _S3CheckpointCallback(TrainerCallback):
    """Upload checkpoints and metrics to S3 in the background after each save."""

    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str,
        metrics_path: Path,
        extra_upload_paths: list[Path] | None = None,
    ):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.metrics_path = metrics_path
        self.extra_upload_paths = extra_upload_paths or []

    def _upload(self, local: str | Path, s3_key: str) -> None:
        local = Path(local)
        uri = f"s3://{self.s3_bucket}/{self.s3_prefix}/{s3_key}"
        if local.is_dir():
            cmd = f"aws s3 sync {local} {uri} --quiet"
        else:
            cmd = f"aws s3 cp {local} {uri} --quiet"
        subprocess.Popen(cmd, shell=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if ckpt_dir.exists():
            self._upload(ckpt_dir, f"checkpoint-{state.global_step}")
        self._upload(self.metrics_path, "metrics.jsonl")
        for extra_path in self.extra_upload_paths:
            if extra_path.exists():
                self._upload(extra_path, extra_path.name)


class _ScratchTrainer(Trainer):
    """Trainer for from-scratch PreTrainedModel models with custom optimizer support.
    PGP loss is handled in the model's forward() via config.pgp_alpha."""

    def __init__(self, *args, optimizer_cls=None, optimizer_kwargs=None,
                 normalize_shared_grad: bool = False,
                 shared_clip: float = 0.1,
                 other_clip: float = 1.0,
                 velocity_clip: bool = False,
                 velocity_clip_floor: float = 15.0,
                 velocity_clip_ratio: float = 2.0,
                 **kwargs):
        self._custom_optimizer_cls = optimizer_cls
        self._custom_optimizer_kwargs = optimizer_kwargs or {}
        self._normalize_shared_grad = normalize_shared_grad
        self._shared_block_params: list | None = None
        self._other_params: list | None = None
        self._shared_grad_scale: float = 1.0
        self._shared_clip = shared_clip
        self._other_clip = other_clip
        # Pre-clip grad norms (set each training_step, read by callback)
        self._last_pre_clip_shared: float = 0.0
        self._last_pre_clip_other: float = 0.0
        # Velocity-based grad clip state
        if normalize_shared_grad and velocity_clip:
            raise ValueError(
                "normalize_shared_grad and velocity_clip cannot both be True: "
                "they conflict on gradient measurement semantics (velocity clip "
                "would measure post-per-component-clipped grads)."
            )
        self._velocity_clip = velocity_clip
        self._velocity_clip_floor = velocity_clip_floor
        self._velocity_clip_ratio = velocity_clip_ratio
        self._prev_grad_norm: float | None = None
        self._velocity_clips_applied: int = 0
        self._last_velocity_clip_pre: float = 0.0
        self._last_velocity_clip_post: float = 0.0
        self._step_metrics_accumulator: dict[str, list[float]] = {}
        self._last_step_metrics: dict[str, float] = {}
        self._last_step_grad_norm: float | None = None
        self._last_raw_training_step_loss: float | None = None
        super().__init__(*args, **kwargs)
        if normalize_shared_grad and hasattr(self.model, 'model') and hasattr(self.model.model, 'shared_block'):
            n = self.model.config.num_shared_instances
            self._shared_grad_scale = 1.0 / n
            shared_ids = set(
                id(p) for p in self.model.model.shared_block.parameters()
            )
            self._shared_block_params = [
                p for p in self.model.parameters() if id(p) in shared_ids
            ]
            self._other_params = [
                p for p in self.model.parameters() if id(p) not in shared_ids
            ]
            print(f"Shared block gradient handling: normalize by 1/{n}, clip to {self._shared_clip}")
            print(f"Other params: clip to {self._other_clip}")
            print(f"  shared params: {sum(p.numel() for p in self._shared_block_params)/1e6:.1f}M")
            print(f"  other params:  {sum(p.numel() for p in self._other_params)/1e6:.1f}M")

    def create_optimizer(self):
        if self._custom_optimizer_cls is not None:
            self.optimizer = self._custom_optimizer_cls(
                self.model.parameters(), **self._custom_optimizer_kwargs
            )
        else:
            super().create_optimizer()
        return self.optimizer

    @staticmethod
    def _compute_total_grad_norm(model) -> float | None:
        grads = [
            p.grad.detach().float().norm(2)
            for p in model.parameters()
            if p.grad is not None
        ]
        if not grads:
            return None
        total = torch.norm(torch.stack(grads), p=2)
        return float(total.item())

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        self._last_raw_training_step_loss = float(loss.detach())
        inner = model.module if hasattr(model, "module") else model
        raw_loss_val = getattr(inner, "_last_total_loss", None)
        accumulate_metrics(
            self._step_metrics_accumulator,
            {"loss": raw_loss_val if raw_loss_val is not None else self._last_raw_training_step_loss},
        )
        accumulate_metrics(
            self._step_metrics_accumulator,
            collect_runtime_metrics(
                model,
                trainer=self,
                prefer_step_aggregates=False,
                include_trainer_state=False,
            ),
        )
        if self._normalize_shared_grad and self._shared_block_params:
            if self.accelerator.sync_gradients:
                # 1. Normalize shared block grads by number of depth calls
                for p in self._shared_block_params:
                    if p.grad is not None:
                        p.grad.mul_(self._shared_grad_scale)
                # 2. Measure pre-clip grad norms per component (diagnostic)
                with torch.no_grad():
                    shared_pre = float(torch.nn.utils.clip_grad_norm_(
                        self._shared_block_params, max_norm=float("inf")
                    ))
                    other_pre = float(torch.nn.utils.clip_grad_norm_(
                        self._other_params, max_norm=float("inf")
                    ))
                self._last_pre_clip_shared = shared_pre
                self._last_pre_clip_other = other_pre
                # 3. Per-component clipping (configurable)
                torch.nn.utils.clip_grad_norm_(self._shared_block_params, max_norm=self._shared_clip)
                torch.nn.utils.clip_grad_norm_(self._other_params, max_norm=self._other_clip)
        # Velocity-based spike damper: clip only when grad_norm spikes abruptly.
        # Condition: grad_norm > floor AND grad_norm > ratio × prev.
        # Action: scale to prev + (current - prev) / 2 (half-step toward spike).
        # Runs AFTER HF Trainer's own clipping (if any); assumes HF clip disabled
        # via max_grad_norm=0.0 when this is enabled.
        if self._velocity_clip and self.accelerator.sync_gradients:
            with torch.no_grad():
                current_gn = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float("inf")
                ))
            if (self._prev_grad_norm is not None
                    and current_gn > self._velocity_clip_floor
                    and current_gn > self._velocity_clip_ratio * self._prev_grad_norm):
                # Spike: dampen to half the distance above prev
                new_norm = self._prev_grad_norm + (current_gn - self._prev_grad_norm) / 2.0
                scale = new_norm / (current_gn + 1e-8)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
                self._velocity_clips_applied += 1
                self._last_velocity_clip_pre = current_gn
                self._last_velocity_clip_post = new_norm
                # Update prev to the POST-clip value so next step's ratio compares
                # against what we actually applied, not the unclipped spike.
                self._prev_grad_norm = new_norm
            else:
                self._prev_grad_norm = current_gn
        if self.accelerator.sync_gradients:
            self._last_step_grad_norm = self._compute_total_grad_norm(model)
            if self._last_step_grad_norm is not None:
                accumulate_metrics(
                    self._step_metrics_accumulator,
                    {"grad_norm": self._last_step_grad_norm},
                )
            optimizer = getattr(self, "optimizer", None)
            if optimizer is not None and getattr(optimizer, "param_groups", None):
                lr = optimizer.param_groups[0].get("lr")
            else:
                lr = None
            if isinstance(lr, (int, float)) and not isinstance(lr, bool):
                accumulate_metrics(
                    self._step_metrics_accumulator,
                    {"learning_rate": float(lr)},
                )
            self._last_step_metrics = average_metrics(self._step_metrics_accumulator)
            self._step_metrics_accumulator = {}
        return loss


def train_from_scratch(
    *,
    model_type: str,
    output_dir: str | Path,
    hidden_size: int,
    num_layers: int,
    tokenizer_id: str,
    dataset_name: str,
    dataset_config: str | None,
    block_size: int,
    max_steps: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    optimizer: str,
    save_steps: int,
    logging_steps: int,
    gradient_checkpointing: bool,
    dataloader_num_workers: int,
    seed: int,
    s3_bucket: str | None,
    s3_prefix: str | None,
    resume_from_checkpoint: str | None = None,
    init_weights_from: str | None = None,
    pgp: bool = False,
    pgp_alpha: float = 0.05,
    pgp_threshold: float = 1.0,
    repulsion: bool = False,
    repulsion_alpha: float = 0.01,
    gravity: bool = False,
    gravity_alpha: float = 0.01,
    covariance: bool = False,
    covariance_alpha: float = 0.01,
    max_loops: int = 1,
    loop_decay: float = 0.5,
    habituation_threshold: float = 0.01,
    normalize_shared_grad: bool = False,
    shared_clip: float = 0.1,
    other_clip: float = 1.0,
    max_grad_norm: float = 1.0,
    supervisor_tune_grad_clip: bool = False,
    velocity_clip: bool = False,
    velocity_clip_floor: float = 15.0,
    velocity_clip_ratio: float = 2.0,
    # Neuro-analog features
    refractory_masking: bool = False,
    refractory_exit_threshold: float = 3.0,
    refractory_use_proxy: bool = True,
    refractory_proxy_threshold: float = 0.05,
    neuromodulatory_gate: bool = False,
    neuromod_gate_hidden: int = 64,
    neuromod_gate_threshold: float = 0.5,
    neuromod_gate_loss_alpha: float = 0.01,
    neuromod_gate_reward_alpha: float = 0.0,
    multi_entry: bool = False,
    multi_entry_points: int = 4,
    cross_loop_cache: bool = False,
    cross_loop_cache_mode: str = "final",
    deep_self_supervision: bool = False,
    deep_supervision_weights: list[float] | None = None,
    surprise_weighted_loss: bool = False,
    surprise_gamma: float = 2.0,
    interloop_pred_enabled: bool = False,
    interloop_pred_weight: float = 0.1,
    interloop_feedback: bool = False,
    sparsity_enabled: bool = False,
    sparsity_k: float = 0.1,
    sparsity_apply_mode: str = "each_loop",
    sparsity_ste: bool = False,
    use_loop_embedding: bool = False,
    sparsity_homeostatic: bool = False,
    homeostatic_tolerance: float = 1.0,
    homeostatic_alpha: float = 1.0,
    adaptive_homeostasis: bool = False,
    adaptive_homeostasis_warmup_steps: int = 100,
    adaptive_homeostasis_interval: int = 100,
    homeostatic_alpha_min: float = 0.25,
    homeostatic_alpha_max: float = 1.0,
    homeostatic_tolerance_min: float = 1.75,
    homeostatic_tolerance_max: float = 4.0,
    mtp_n_future: int = 1,
    mtp_loss_weight: float = 1.0,
    mtp_horizon_decay: float = 1.0,
    mtp_warmup_steps: int = 0,
    mtp_independent_heads: bool = False,
    # Looped reasoning architecture
    num_shared_layers: int = 4,
    num_dedicated_layers: int = 8,
    loop0_confidence_threshold: float = 0.9,
    loop1_margin_threshold_init: float = 0.15,
    episodic_buffer_enabled: bool = False,
    buffer_capacity: int = 4096,
    buffer_top_k: int = 32,
    buffer_novelty_threshold: float = 0.9,
    buffer_improvement_threshold: float = 0.5,
    surprise_head_enabled: bool = False,
    surprise_exit_threshold: float = 0.7,
    surprise_head_loss_weight: float = 0.1,
    surprise_weighted_final_only: bool = False,
    adaptive_loop_thresholds: bool = False,
    adaptive_loop_threshold_warmup_steps: int = 200,
    adaptive_loop_threshold_interval: int = 100,
    loop0_exit_target_low: float = 0.02,
    loop0_exit_target_high: float = 0.15,
    loop1_exit_target_low: float = 0.05,
    loop1_exit_target_high: float = 0.30,
    surprise_exit_threshold_min: float = 0.35,
    surprise_exit_threshold_max: float = 0.80,
    surprise_exit_threshold_step: float = 0.05,
    loop1_margin_threshold_min: float = 0.02,
    loop1_margin_threshold_max: float = 0.30,
    loop1_margin_threshold_step: float = 0.02,
    step_metrics_every_step: bool = False,
    supervisor_mode: str = "off",
    supervisor_model: str = "gpt-5-nano",
    supervisor_api_base: str = "https://api.openai.com/v1",
    supervisor_api_key_env: str = "OPENAI_API_KEY",
    supervisor_decision_interval: int = 1,
    supervisor_decision_warmup_steps: int = 0,
    supervisor_history_steps: int = 64,
    supervisor_transition_horizon: int = 25,
    supervisor_max_output_tokens: int = 1200,
    supervisor_temperature: float = 1.0,
    supervisor_objective_text: str | None = None,
    supervisor_journal_path: str | None = None,
) -> dict[str, Any]:
    """Train a standard, factored, or looped transformer from scratch."""
    from mrp.factored_transformer import (
        FactoredTransformerConfig,
        FactoredCausalLM,
        StandardCausalLM,
    )
    from mrp.looped_reasoning_transformer import (
        LoopedReasoningConfig,
        LoopedReasoningCausalLM,
    )

    output_root = ensure_dir(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory
        print(f"VRAM: {vram / 1e9:.1f} GB")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"Tokenizer: {tokenizer_id}, vocab: {vocab_size}")

    # Model config — derive head counts from hidden size
    num_heads = max(1, hidden_size // 128)
    num_kv_heads = max(1, num_heads // 4)
    config = FactoredTransformerConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=int(hidden_size * 2.6875),
        num_layers=num_layers,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_position_embeddings=block_size,
        num_foundation_layers_start=max(2, num_layers // 6),
        num_foundation_layers_end=max(2, num_layers // 6),
        residual_ratio=0.5,
        pgp_alpha=pgp_alpha if pgp else 0.0,
        pgp_threshold=pgp_threshold,
        repulsion_alpha=repulsion_alpha if repulsion else 0.0,
        gravity_alpha=gravity_alpha if gravity else 0.0,
        covariance_alpha=covariance_alpha if covariance else 0.0,
        max_loops=max_loops,
        loop_decay=loop_decay,
        habituation_threshold=habituation_threshold,
        refractory_masking=refractory_masking,
        refractory_exit_threshold=refractory_exit_threshold,
        refractory_use_proxy=refractory_use_proxy,
        refractory_proxy_threshold=refractory_proxy_threshold,
        neuromodulatory_gate=neuromodulatory_gate,
        neuromod_gate_hidden=neuromod_gate_hidden,
        neuromod_gate_threshold=neuromod_gate_threshold,
        neuromod_gate_loss_alpha=neuromod_gate_loss_alpha,
        neuromod_gate_reward_alpha=neuromod_gate_reward_alpha,
        multi_entry=multi_entry,
        multi_entry_points=multi_entry_points,
        cross_loop_cache=cross_loop_cache,
        cross_loop_cache_mode=cross_loop_cache_mode,
        deep_self_supervision=deep_self_supervision,
        deep_supervision_weights=deep_supervision_weights,
        surprise_weighted_loss=surprise_weighted_loss,
        surprise_gamma=surprise_gamma,
        interloop_pred_enabled=interloop_pred_enabled,
        interloop_pred_weight=interloop_pred_weight,
        interloop_feedback=interloop_feedback,
        sparsity_enabled=sparsity_enabled,
        sparsity_k=sparsity_k,
        sparsity_apply_mode=sparsity_apply_mode,
        sparsity_ste=sparsity_ste,
        mtp_n_future=mtp_n_future,
        mtp_loss_weight=mtp_loss_weight,
        mtp_horizon_decay=mtp_horizon_decay,
        mtp_independent_heads=mtp_independent_heads,
        tokenizer_id=tokenizer_id,
    )

    # Create model
    if model_type == "standard":
        model = StandardCausalLM(config)
    elif model_type == "factored":
        model = FactoredCausalLM(config)
    elif model_type == "looped":
        # Looped reasoning: distinct config (narrower surface area than factored).
        # num_layers arg is reinterpreted as num_shared + num_dedicated.
        looped_config = LoopedReasoningConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=int(hidden_size * 2.6875),
            num_attention_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=block_size,
            num_shared_layers=num_shared_layers,
            num_dedicated_layers=num_dedicated_layers,
            residual_ratio=0.5,
            loop0_confidence_threshold=loop0_confidence_threshold,
            loop1_margin_threshold_init=loop1_margin_threshold_init,
            deep_self_supervision=deep_self_supervision,
            deep_supervision_weights=deep_supervision_weights,
            surprise_weighted_loss=surprise_weighted_loss,
            surprise_gamma=surprise_gamma,
            surprise_weighted_final_only=surprise_weighted_final_only,
            pgp_alpha=pgp_alpha if pgp else 0.0,
            pgp_threshold=pgp_threshold,
            mtp_n_future=mtp_n_future,
            mtp_loss_weight=mtp_loss_weight,
            mtp_horizon_decay=mtp_horizon_decay,
            mtp_warmup_steps=mtp_warmup_steps,
            mtp_independent_heads=mtp_independent_heads,
            sparsity_enabled=sparsity_enabled,
            sparsity_k=sparsity_k,
            sparsity_apply_mode=sparsity_apply_mode,
            sparsity_ste=sparsity_ste,
            interloop_pred_enabled=interloop_pred_enabled,
            interloop_pred_weight=interloop_pred_weight,
            use_loop_embedding=use_loop_embedding,
            sparsity_homeostatic=sparsity_homeostatic,
            homeostatic_tolerance=homeostatic_tolerance,
            homeostatic_alpha=homeostatic_alpha,
            adaptive_homeostasis=adaptive_homeostasis,
            adaptive_homeostasis_warmup_steps=adaptive_homeostasis_warmup_steps,
            adaptive_homeostasis_interval=adaptive_homeostasis_interval,
            homeostatic_alpha_min=homeostatic_alpha_min,
            homeostatic_alpha_max=homeostatic_alpha_max,
            homeostatic_tolerance_min=homeostatic_tolerance_min,
            homeostatic_tolerance_max=homeostatic_tolerance_max,
            episodic_buffer_enabled=episodic_buffer_enabled,
            buffer_capacity=buffer_capacity,
            buffer_top_k=buffer_top_k,
            buffer_novelty_threshold=buffer_novelty_threshold,
            buffer_improvement_threshold=buffer_improvement_threshold,
            surprise_head_enabled=surprise_head_enabled,
            surprise_exit_threshold=surprise_exit_threshold,
            surprise_head_loss_weight=surprise_head_loss_weight,
            tokenizer_id=tokenizer_id,
        )
        model = LoopedReasoningCausalLM(looped_config)
        if init_weights_from:
            # Load weights from a previous checkpoint but keep fresh
            # optimizer/scheduler state. Used to extend training with a
            # new LR schedule.
            import os
            st_path = os.path.join(init_weights_from, "model.safetensors")
            pt_path = os.path.join(init_weights_from, "pytorch_model.bin")
            if os.path.exists(st_path):
                from safetensors.torch import load_file
                state_dict = load_file(st_path)
            elif os.path.exists(pt_path):
                state_dict = torch.load(pt_path, weights_only=True, map_location="cpu")
            else:
                raise FileNotFoundError(
                    f"No weights found at {init_weights_from} "
                    f"(expected model.safetensors or pytorch_model.bin)"
                )
            surprise_heads = getattr(getattr(model, "model", None), "surprise_heads", None)
            if surprise_heads is not None and not any(
                k.startswith("model.surprise_heads.") for k in state_dict
            ):
                legacy_prefix = "model.surprise_head."
                legacy_items = {
                    k[len(legacy_prefix):]: state_dict.pop(k)
                    for k in list(state_dict.keys())
                    if k.startswith(legacy_prefix)
                }
                if legacy_items:
                    for head_idx in range(len(surprise_heads)):
                        for suffix, tensor in legacy_items.items():
                            state_dict[f"model.surprise_heads.{head_idx}.{suffix}"] = tensor.clone()
                    print(
                        "  Replicated legacy surprise_head weights across "
                        f"{len(surprise_heads)} per-gate heads"
                    )
            homeo_fr = state_dict.get("model.homeostatic_firing_rate")
            target_homeo = getattr(getattr(model, "model", None), "homeostatic_firing_rate", None)
            if (homeo_fr is not None and target_homeo is not None
                    and tuple(homeo_fr.shape) != tuple(target_homeo.shape)):
                if homeo_fr.ndim == 1 and target_homeo.ndim == 2 and homeo_fr.shape[-1] == target_homeo.shape[-1]:
                    state_dict["model.homeostatic_firing_rate"] = homeo_fr.unsqueeze(0).repeat(target_homeo.shape[0], 1)
                    print(
                        "  Replicated legacy homeostatic_firing_rate across "
                        f"{target_homeo.shape[0]} loops"
                    )
                else:
                    state_dict.pop("model.homeostatic_firing_rate")
                    print(
                        "  Dropped incompatible legacy homeostatic_firing_rate "
                        f"shape {tuple(homeo_fr.shape)}; reinitializing from config"
                    )
            buffer_runtime_prefixes = (
                "model.episodic_buffer.keys",
                "model.episodic_buffer.values",
                "model.episodic_buffer.write_pos",
                "model.episodic_buffer.n_written",
                "model.episodic_buffer.n_write_attempts",
                "model.episodic_buffer.n_write_rejects",
            )
            dropped_buffer_runtime = [
                key for key in list(state_dict.keys())
                if key in buffer_runtime_prefixes
            ]
            if dropped_buffer_runtime:
                for key in dropped_buffer_runtime:
                    state_dict.pop(key, None)
                print(
                    "  Dropped episodic buffer contents/counters from "
                    "init_weights_from; starting with an empty buffer"
                )
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {init_weights_from}")
            if missing:
                print(f"  Missing keys: {len(missing)} (first 5: {missing[:5]})")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
        config = looped_config  # so summary uses the looped config
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    params = model.param_count()
    print(f"Model: {model_type}, {params['total'] / 1e6:.0f}M params")

    if gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Dataset — streaming from HuggingFace
    train_dataset = _StreamingBlockDataset(
        tokenizer, dataset_name, dataset_config, "train", block_size
    )

    tokens_per_step = per_device_train_batch_size * gradient_accumulation_steps * block_size
    total_tokens = tokens_per_step * max_steps
    print(f"Training: {max_steps} steps, {tokens_per_step:,} tokens/step, "
          f"{total_tokens / 1e9:.1f}B tokens total")
    print(f"Dataset: {dataset_name} ({dataset_config}), streaming")

    # Optimizer
    if optimizer == "mano":
        from mrp.mano import Mano
        opt_cls = Mano
        opt_kwargs = {"lr": learning_rate, "momentum": 0.95, "weight_decay": weight_decay}
        optim_name = "adamw_torch"  # placeholder — overridden by _ScratchTrainer
    else:
        opt_cls = None
        opt_kwargs = {}
        optim_name = "adamw_torch"

    warmup_steps = max(0, math.ceil(max_steps * warmup_ratio))

    training_args = TrainingArguments(
        output_dir=str(output_root),
        do_train=True,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        optim=optim_name,
        bf16=(device.type != "cpu"),
        tf32=(device.type != "cpu"),
        gradient_checkpointing=gradient_checkpointing,
        logging_strategy="steps",
        logging_steps=logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=5,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        seed=seed,
        data_seed=seed,
        use_cpu=(device.type == "cpu"),
        dataloader_pin_memory=(device.type != "cpu"),
        dataloader_num_workers=dataloader_num_workers,
        disable_tqdm=False,
        # Streaming datasets make full dataloader rewind prohibitively slow on resume.
        # We restore model/optimizer/scheduler state from checkpoint and continue
        # without replaying skipped batches.
        ignore_data_skip=bool(resume_from_checkpoint),
        # Disable global clipping when using per-component clipping for factored models
        # factored + normalize_shared_grad uses per-component clipping, so
        # disable HF's global clipping. Otherwise use the configured value.
        # Disable HF global clip when factored uses per-component clipping OR
        # when velocity-based clipping is enabled (they'd fight each other).
        max_grad_norm=(
            0.0 if (normalize_shared_grad and model_type == "factored")
            else 0.0 if velocity_clip
            else max_grad_norm
        ),
    )

    # Callbacks
    metrics_path = output_root / "metrics.jsonl"
    adaptations_path = output_root / "adaptations.jsonl"
    step_metrics_path = output_root / "step_metrics.jsonl"
    supervisor_decisions_path = output_root / "supervisor_decisions.jsonl"
    supervisor_transitions_path = output_root / "supervisor_transitions.jsonl"
    # NOTE: _MetricsCallback must run LAST among callbacks that modify the
    # logs dict, because it serializes logs to disk. NeuroAnalogCallback and
    # SupervisorCallback add per-component metrics to logs during on_log;
    # those are only captured if they run BEFORE _MetricsCallback.
    # HF Trainer calls callbacks in registration order, so we start with
    # an empty list and append _MetricsCallback last below.
    callbacks: list = []
    language_supervisor_callback: LanguageSupervisorCallback | None = None

    extra_upload_paths: list[Path] = []
    if step_metrics_every_step or supervisor_mode != "off":
        extra_upload_paths.append(step_metrics_path)
    if supervisor_mode != "off":
        extra_upload_paths.extend([supervisor_decisions_path, supervisor_transitions_path])

    if s3_bucket and s3_prefix:
        callbacks.append(_S3CheckpointCallback(
            s3_bucket,
            s3_prefix,
            metrics_path,
            extra_upload_paths=extra_upload_paths,
        ))

    trainer = _ScratchTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=callbacks,
        optimizer_cls=opt_cls,
        optimizer_kwargs=opt_kwargs,
        normalize_shared_grad=normalize_shared_grad and model_type == "factored",
        shared_clip=shared_clip,
        other_clip=other_clip,
        velocity_clip=velocity_clip,
        velocity_clip_floor=velocity_clip_floor,
        velocity_clip_ratio=velocity_clip_ratio,
    )

    # Neuro-analog diagnostics (only adds overhead if features are enabled).
    # Added after trainer creation so we can pass trainer_ref for pre-clip grad
    # norm logging.
    # IMPORTANT: _NeuroAnalogCallback MUST be registered before _SupervisorCallback.
    # The supervisor reads sparsity_live_frac from logs dict, which is written by
    # the neuro-analog callback. HF Trainer calls callbacks in registration order
    # during on_log, so neuro-analog must run first to populate the dict.
    if any([refractory_masking, neuromodulatory_gate, multi_entry, cross_loop_cache,
            max_loops > 1, mtp_n_future > 1, sparsity_enabled, interloop_pred_enabled,
            model_type == "looped"]):
        trainer.add_callback(_NeuroAnalogCallback(trainer_ref=trainer))

    # Supervisor: adaptive tuning. Register if either sparsity_enabled (needs
    # sparsity_k adjustment) or supervisor_tune_grad_clip (needs max_grad_norm
    # adjustment) is on. Depends on diagnostics from _NeuroAnalogCallback —
    # see ordering comment above.
    heuristic_supervisor_enabled = (
        supervisor_mode == "off"
        and (sparsity_enabled or supervisor_tune_grad_clip or adaptive_loop_thresholds)
    )
    if heuristic_supervisor_enabled:
        trainer.add_callback(_SupervisorCallback(
            adaptations_path=adaptations_path,
            trainer_ref=trainer,
            tune_grad_clip=supervisor_tune_grad_clip,
            tune_loop_thresholds=adaptive_loop_thresholds,
            loop_threshold_warmup_steps=adaptive_loop_threshold_warmup_steps,
            loop_threshold_interval=adaptive_loop_threshold_interval,
            loop0_exit_target_low=loop0_exit_target_low,
            loop0_exit_target_high=loop0_exit_target_high,
            loop1_exit_target_low=loop1_exit_target_low,
            loop1_exit_target_high=loop1_exit_target_high,
            surprise_exit_threshold_min=surprise_exit_threshold_min,
            surprise_exit_threshold_max=surprise_exit_threshold_max,
            surprise_exit_threshold_step=surprise_exit_threshold_step,
            loop1_margin_threshold_min=loop1_margin_threshold_min,
            loop1_margin_threshold_max=loop1_margin_threshold_max,
            loop1_margin_threshold_step=loop1_margin_threshold_step,
        ))
    elif supervisor_mode != "off" and (
        sparsity_enabled or supervisor_tune_grad_clip or adaptive_loop_thresholds
    ):
        print(
            "Language supervisor enabled; skipping heuristic supervisor callbacks "
            "so the learned controller is the only active optimizer."
        )

    if step_metrics_every_step or supervisor_mode != "off":
        language_supervisor_callback = LanguageSupervisorCallback(
            output_root=output_root,
            trainer_ref=trainer,
            mode=supervisor_mode,
            decision_interval=supervisor_decision_interval,
            decision_warmup_steps=supervisor_decision_warmup_steps,
            history_steps=supervisor_history_steps,
            transition_horizon=supervisor_transition_horizon,
            objective_text=supervisor_objective_text,
            journal_path=supervisor_journal_path,
            model_name=supervisor_model,
            api_base=supervisor_api_base,
            api_key_env=supervisor_api_key_env,
            max_output_tokens=supervisor_max_output_tokens,
            temperature=supervisor_temperature,
        )
        trainer.add_callback(language_supervisor_callback)

    # MetricsCallback registered LAST so it sees all keys added by other
    # callbacks during on_log. Otherwise per-component MTP/interloop/sparsity
    # metrics are written to disk as zeros.
    trainer.add_callback(_MetricsCallback(metrics_path))
    if resume_from_checkpoint:
        _patch_torch_load_check()
    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except Exception as exc:
        if language_supervisor_callback is not None:
            language_supervisor_callback.record_run_failure(exc)
        raise
    trainer.save_state()
    trainer.save_model(str(output_root / "final_model"))

    # Upload final model to S3
    if s3_bucket and s3_prefix:
        cb = _S3CheckpointCallback(
            s3_bucket,
            s3_prefix,
            metrics_path,
            extra_upload_paths=extra_upload_paths,
        )
        cb._upload(output_root / "final_model", "final_model")
        cb._upload(metrics_path, "metrics.jsonl")
        for extra_path in extra_upload_paths:
            if extra_path.exists():
                cb._upload(extra_path, extra_path.name)

    summary = {
        "model_type": model_type,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "params_total": params["total"],
        "tokenizer_id": tokenizer_id,
        "vocab_size": vocab_size,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "block_size": block_size,
        "max_steps": max_steps,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "tokens_per_step": tokens_per_step,
        "total_tokens": total_tokens,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "optimizer": optimizer,
        "gradient_checkpointing": gradient_checkpointing,
        "save_steps": save_steps,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "pgp": pgp,
        "pgp_alpha": pgp_alpha if pgp else 0.0,
        "pgp_threshold": pgp_threshold if pgp else 0.0,
        "seed": seed,
        "adaptive_loop_thresholds": adaptive_loop_thresholds,
        "adaptive_loop_threshold_warmup_steps": adaptive_loop_threshold_warmup_steps,
        "adaptive_loop_threshold_interval": adaptive_loop_threshold_interval,
        "step_metrics_every_step": step_metrics_every_step,
        "supervisor_mode": supervisor_mode,
        "supervisor_model": supervisor_model,
        "supervisor_decision_interval": supervisor_decision_interval,
        "supervisor_decision_warmup_steps": supervisor_decision_warmup_steps,
        "supervisor_history_steps": supervisor_history_steps,
        "supervisor_transition_horizon": supervisor_transition_horizon,
        "supervisor_objective_text": supervisor_objective_text,
        "supervisor_journal_path": supervisor_journal_path,
        "heuristic_supervisor_enabled": heuristic_supervisor_enabled,
        "step_metrics_path": str(step_metrics_path),
        "supervisor_decisions_path": str(supervisor_decisions_path),
        "supervisor_transitions_path": str(supervisor_transitions_path),
        "train_metrics": {
            k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
            for k, v in train_result.metrics.items()
        },
        "log_history": trainer.state.log_history,
    }
    write_json(output_root / "train_summary.json", summary)
    return summary
