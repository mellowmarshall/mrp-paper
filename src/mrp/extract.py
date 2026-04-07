from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from mrp.model_loading import load_text_model, resolve_output_embeddings
from mrp.margin_metrics import compute_token_statistics
from mrp.model_inspection import inspect_model
from mrp.utils import ensure_dir, write_json


DEFAULT_TEXT_COLUMNS = ("text", "content", "document")


@dataclass
class ReservoirSampler:
    capacity: int
    seed: int
    vectors: list[np.ndarray] = field(default_factory=list)
    sequence_ids: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    seen: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def add(
        self,
        vector: np.ndarray,
        *,
        sequence_id: int,
        position: int,
        token_id: int,
    ) -> None:
        self.seen += 1
        if self.capacity <= 0:
            return

        if len(self.vectors) < self.capacity:
            self.vectors.append(vector)
            self.sequence_ids.append(sequence_id)
            self.positions.append(position)
            self.token_ids.append(token_id)
            return

        replacement_index = self._rng.randrange(self.seen)
        if replacement_index >= self.capacity:
            return

        self.vectors[replacement_index] = vector
        self.sequence_ids[replacement_index] = sequence_id
        self.positions[replacement_index] = position
        self.token_ids[replacement_index] = token_id

    def export(self) -> dict[str, np.ndarray]:
        if not self.vectors:
            return {
                "hidden_states": np.empty((0, 0), dtype=np.float32),
                "sequence_ids": np.empty((0,), dtype=np.int64),
                "positions": np.empty((0,), dtype=np.int64),
                "token_ids": np.empty((0,), dtype=np.int64),
            }

        return {
            "hidden_states": np.stack(self.vectors, axis=0),
            "sequence_ids": np.asarray(self.sequence_ids, dtype=np.int64),
            "positions": np.asarray(self.positions, dtype=np.int64),
            "token_ids": np.asarray(self.token_ids, dtype=np.int64),
        }


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _resolve_storage_dtype(dtype_name: str) -> np.dtype:
    mapping = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
    }
    if dtype_name not in mapping:
        raise ValueError(f"unsupported hidden-state dtype: {dtype_name}")
    return mapping[dtype_name]


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

    raise ValueError(
        f"unable to infer text column. Available columns: {column_names}"
    )


def extract_token_stats(
    *,
    model_id: str,
    output_dir: str | Path,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    max_sequences: int | None,
    max_length: int,
    reservoir_size: int,
    top_k: int,
    trust_remote_code: bool,
    text_column: str | None,
    device: str,
    hidden_state_dtype: str,
    seed: int,
) -> dict[str, Any]:
    output_root = ensure_dir(output_dir)
    token_stats_path = output_root / "token_stats.csv"
    hidden_state_dir = ensure_dir(output_root / "hidden_states")
    storage_dtype = _resolve_storage_dtype(hidden_state_dtype)
    runtime_device = _resolve_device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    loaded = load_text_model(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype="auto",
    )
    model = loaded.model
    model.eval()
    model.to(runtime_device)

    # Cache lm_head weight in float32 on CPU for high-precision logit
    # recomputation.  The model forward pass may run in bf16/fp16 but
    # we redo the final projection in fp32 to avoid margin quantization
    # artifacts.  Note: hidden_states[-1] from this model is already
    # post-norm, so only the lm_head projection needs to be redone.
    output_emb = resolve_output_embeddings(model)
    if output_emb is None or not hasattr(output_emb, "weight"):
        raise RuntimeError(
            "unable to resolve output embeddings for fp32 logit recomputation"
        )
    _lm_head_fp32 = output_emb.weight.detach().float().to(runtime_device)

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
    )
    resolved_text_column = _detect_text_column(dataset.column_names, text_column)

    fieldnames = [
        "sequence_id",
        "position",
        "input_token_id",
        "target_token_id",
        "margin",
        "entropy",
        "top1_correct",
        "correct_rank",
        "top_k_token_ids",
        "top_k_logits",
    ]

    reservoirs: dict[int, ReservoirSampler] = {}
    processed_sequences = 0
    processed_positions = 0

    with token_stats_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(dataset, desc="extracting", unit="sequence"):
            if max_sequences is not None and processed_sequences >= max_sequences:
                break

            text = row.get(resolved_text_column)
            if not isinstance(text, str) or not text.strip():
                continue

            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            if encoded["input_ids"].size(1) < 2:
                continue

            encoded = {key: value.to(runtime_device) for key, value in encoded.items()}

            with torch.no_grad():
                outputs = model(
                    **encoded,
                    output_hidden_states=True,
                    return_dict=True,
                )

            hidden_states = outputs.hidden_states
            input_ids = encoded["input_ids"][0].detach().cpu()
            current_ids = input_ids[:-1]
            target_ids = input_ids[1:]

            # Recompute logits in float32 from the last hidden state to avoid
            # bfloat16 quantization artifacts in margin measurements.
            # hidden_states[-1] is already post-norm for this model.
            final_hidden = hidden_states[-1][0, :-1, :].detach().float()
            logits = (final_hidden @ _lm_head_fp32.T).cpu()

            stats = compute_token_statistics(logits, target_ids, top_k=top_k)

            for position in range(logits.size(0)):
                writer.writerow(
                    {
                        "sequence_id": processed_sequences,
                        "position": position,
                        "input_token_id": int(current_ids[position].item()),
                        "target_token_id": int(target_ids[position].item()),
                        "margin": float(stats["margins"][position].item()),
                        "entropy": float(stats["entropy"][position].item()),
                        "top1_correct": int(stats["top1_correct"][position].item()),
                        "correct_rank": int(stats["correct_rank"][position].item()),
                        "top_k_token_ids": " ".join(
                            str(int(token_id))
                            for token_id in stats["top_k_indices"][position].tolist()
                        ),
                        "top_k_logits": " ".join(
                            f"{float(value):.8f}"
                            for value in stats["top_k_logits"][position].tolist()
                        ),
                    }
                )

            for layer_index, layer_hidden_state in enumerate(hidden_states):
                sampler = reservoirs.setdefault(
                    layer_index,
                    ReservoirSampler(capacity=reservoir_size, seed=seed + layer_index),
                )
                layer_vectors = layer_hidden_state[0, :-1, :].detach().cpu()
                for position in range(layer_vectors.size(0)):
                    vector = (
                        layer_vectors[position]
                        .float()
                        .numpy()
                        .astype(storage_dtype, copy=False)
                    )
                    sampler.add(
                        vector,
                        sequence_id=processed_sequences,
                        position=position,
                        token_id=int(current_ids[position].item()),
                    )

            processed_sequences += 1
            processed_positions += int(logits.size(0))

    hidden_state_files: list[str] = []
    for layer_index, sampler in reservoirs.items():
        export = sampler.export()
        destination = hidden_state_dir / f"layer_{layer_index:02d}.npz"
        np.savez_compressed(destination, **export)
        hidden_state_files.append(str(destination))

    inspection = inspect_model(
        model_id,
        trust_remote_code=trust_remote_code,
        load_weights=False,
    )
    manifest = {
        "model": inspection,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "text_column": resolved_text_column,
        "device": str(runtime_device),
        "load_strategy": loaded.load_strategy,
        "max_sequences": max_sequences,
        "max_length": max_length,
        "reservoir_size": reservoir_size,
        "top_k": top_k,
        "hidden_state_dtype": hidden_state_dtype,
        "seed": seed,
        "processed_sequences": processed_sequences,
        "processed_positions": processed_positions,
        "token_stats_path": str(token_stats_path),
        "hidden_state_files": hidden_state_files,
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest
