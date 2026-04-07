from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from mrp.eval_harness import run_lm_eval
from mrp.model_loading import resolve_output_embeddings
from mrp.shared import load_model_flexible, resolve_device


DEFAULT_FEWSHOT_TASKS = [
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "piqa",
    "lambada_openai",
    "truthfulqa_mc1",
    "truthfulqa_mc2",
]


def load_eval_texts_and_sequences(
    *,
    tokenizer: Any,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "validation",
    n_sequences: int = 64,
    max_length: int = 512,
    min_chars: int = 50,
) -> tuple[list[str], list[torch.Tensor]]:
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    texts: list[str] = []
    sequences: list[torch.Tensor] = []
    for example in dataset:
        text = str(example.get("text") or "").strip()
        if len(text) < min_chars:
            continue
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"][0]
        if input_ids.numel() < 32:
            continue
        texts.append(text)
        sequences.append(input_ids)
        if len(sequences) >= n_sequences:
            break
    return texts, sequences


def load_eval_context(
    *,
    model_id: str,
    tokenizer_id: str | None = None,
    device: str = "cpu",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "validation",
    n_sequences: int = 64,
    max_length: int = 512,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    runtime_device = resolve_device(device)
    tokenizer_name = tokenizer_id or model_id
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model, _ = load_model_flexible(
        model_id,
        device=device,
        tokenizer_id=tokenizer_name,
        trust_remote_code=trust_remote_code,
    )
    texts, sequences = load_eval_texts_and_sequences(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        n_sequences=n_sequences,
        max_length=max_length,
    )
    return {
        "model": model,
        "tokenizer": tokenizer,
        "texts": texts,
        "sequences": sequences,
        "runtime_device": runtime_device,
        "dataset": {
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "split": split,
        },
        "max_length": max_length,
        "n_sequences": n_sequences,
        "tokenizer_id": tokenizer_name,
        "model_id": model_id,
    }


def _collect_probability_stats(
    model: torch.nn.Module,
    sequences: list[torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    total_nll = 0.0
    token_count = 0
    confidences: list[float] = []
    corrects: list[int] = []
    sequence_nlls: list[float] = []
    sequence_token_counts: list[int] = []

    model.eval()
    with torch.no_grad():
        for sequence in sequences:
            input_ids = sequence.unsqueeze(0).to(device)
            logits = model(input_ids=input_ids).logits[0, :-1].float()
            labels = input_ids[0, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            probs = log_probs.exp()
            confidence, preds = probs.max(dim=-1)

            total_nll += float(nll.sum().item())
            token_count += int(labels.numel())
            confidences.extend(confidence.detach().cpu().tolist())
            corrects.extend((preds == labels).detach().int().cpu().tolist())
            sequence_nlls.append(float(nll.mean().item()))
            sequence_token_counts.append(int(labels.numel()))

    return {
        "total_nll": total_nll,
        "token_count": token_count,
        "confidences": confidences,
        "corrects": corrects,
        "sequence_nlls": sequence_nlls,
        "sequence_token_counts": sequence_token_counts,
    }


def compute_perplexity_eval(
    *,
    model: torch.nn.Module,
    sequences: list[torch.Tensor],
    device: torch.device,
    dataset: dict[str, Any],
    max_length: int,
) -> dict[str, Any]:
    stats = _collect_probability_stats(model, sequences, device)
    mean_nll = stats["total_nll"] / max(stats["token_count"], 1)
    perplexity = math.exp(mean_nll) if stats["token_count"] else None
    return {
        "schema_version": 1,
        "suite_id": "perplexity_eval",
        "dataset": dataset,
        "max_length": max_length,
        "n_sequences": len(sequences),
        "n_tokens": stats["token_count"],
        "mean_nll": mean_nll if stats["token_count"] else None,
        "overall_perplexity": perplexity,
        "breakdowns": [
            {
                "name": "validation",
                "n_tokens": stats["token_count"],
                "mean_nll": mean_nll if stats["token_count"] else None,
                "perplexity": perplexity,
            }
        ],
    }


def compute_calibration_eval(
    *,
    model: torch.nn.Module,
    sequences: list[torch.Tensor],
    device: torch.device,
    dataset: dict[str, Any],
    max_length: int,
    num_bins: int = 10,
) -> dict[str, Any]:
    stats = _collect_probability_stats(model, sequences, device)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    total = max(len(stats["confidences"]), 1)
    for index in range(num_bins):
        lower = index / num_bins
        upper = (index + 1) / num_bins
        members = [
            (confidence, correct)
            for confidence, correct in zip(stats["confidences"], stats["corrects"], strict=False)
            if (confidence >= lower and confidence < upper)
            or (index == num_bins - 1 and confidence == upper)
        ]
        count = len(members)
        if count:
            mean_confidence = sum(item[0] for item in members) / count
            accuracy = sum(item[1] for item in members) / count
            gap = abs(mean_confidence - accuracy)
            ece += gap * (count / total)
            mce = max(mce, gap)
        else:
            mean_confidence = None
            accuracy = None
        bins.append(
            {
                "lower": round(lower, 4),
                "upper": round(upper, 4),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {
        "schema_version": 1,
        "suite_id": "calibration_eval",
        "dataset": dataset,
        "max_length": max_length,
        "n_tokens": stats["token_count"],
        "ece": ece if stats["token_count"] else None,
        "mce": mce if stats["token_count"] else None,
        "bins": bins,
    }


def compute_tokenization_eval(
    *,
    texts: list[str],
    sequences: list[torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    dataset: dict[str, Any],
    max_length: int,
) -> dict[str, Any]:
    stats = _collect_probability_stats(model, sequences, device)
    char_count = sum(len(text) for text in texts)
    byte_count = sum(len(text.encode("utf-8")) for text in texts)
    token_count = stats["token_count"]
    total_bits = stats["total_nll"] / math.log(2) if stats["token_count"] else 0.0
    return {
        "schema_version": 1,
        "suite_id": "tokenization_eval",
        "dataset": dataset,
        "max_length": max_length,
        "n_sequences": len(sequences),
        "n_tokens": token_count,
        "n_chars": char_count,
        "n_bytes": byte_count,
        "tokens_per_char": (token_count / char_count) if char_count else None,
        "tokens_per_byte": (token_count / byte_count) if byte_count else None,
        "bits_per_char": (total_bits / char_count) if char_count else None,
        "bits_per_byte": (total_bits / byte_count) if byte_count else None,
        "mean_nll": (stats["total_nll"] / token_count) if token_count else None,
    }


def compute_layer_prediction_quality(
    *,
    model: torch.nn.Module,
    sequences: list[torch.Tensor],
    device: torch.device,
    dataset: dict[str, Any],
    max_length: int,
) -> dict[str, Any]:
    lm_head = resolve_output_embeddings(model)
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise ValueError("unable to resolve output embeddings for layer prediction quality")
    # Keep the projection on the model device for large checkpoints so we do
    # not pay repeated CPU matmul costs for every hidden-state layer.
    lm_head_weight = lm_head.weight.detach()
    layer_correct: list[int] = []
    layer_total: list[int] = []
    layer_margins: list[list[float]] = []

    model.eval()
    with torch.no_grad():
        for sequence in sequences:
            input_ids = sequence.unsqueeze(0).to(device)
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            targets = input_ids[0, 1:]
            for layer_index, hidden in enumerate(outputs.hidden_states):
                hidden_states = hidden[0, :-1, :]
                if hidden_states.device != lm_head_weight.device:
                    hidden_states = hidden_states.to(lm_head_weight.device)
                logits = hidden_states @ lm_head_weight.T
                top2_values, top2_indices = logits.topk(2, dim=-1)
                preds = top2_indices[:, 0]
                target_slice = targets.to(preds.device)
                correct = int((preds == target_slice).sum().item())
                margins = (
                    (top2_values[:, 0] - top2_values[:, 1]).detach().float().cpu().tolist()
                )
                while layer_index >= len(layer_correct):
                    layer_correct.append(0)
                    layer_total.append(0)
                    layer_margins.append([])
                layer_correct[layer_index] += correct
                layer_total[layer_index] += int(target_slice.numel())
                layer_margins[layer_index].extend(float(value) for value in margins)
                del hidden_states, logits, top2_values, top2_indices, preds, target_slice
            del outputs, input_ids, targets

    layers: list[dict[str, Any]] = []
    for layer_index, total in enumerate(layer_total):
        margins = layer_margins[layer_index]
        if not total or not margins:
            continue
        sorted_margins = sorted(margins)
        mid = len(sorted_margins) // 2
        if len(sorted_margins) % 2:
            median_margin = sorted_margins[mid]
        else:
            median_margin = (sorted_margins[mid - 1] + sorted_margins[mid]) / 2
        layers.append(
            {
                "layer": layer_index,
                "accuracy": layer_correct[layer_index] / total,
                "median_margin": median_margin,
                "mean_margin": sum(margins) / len(margins),
                "n_positions": total,
            }
        )
    best_layer = max(layers, key=lambda row: float(row["accuracy"]), default=None)
    final_layer = layers[-1] if layers else None
    return {
        "schema_version": 1,
        "suite_id": "layer_prediction_quality",
        "dataset": dataset,
        "max_length": max_length,
        "n_sequences": len(sequences),
        "layers": layers,
        "summary": {
            "best_layer": best_layer["layer"] if best_layer else None,
            "best_layer_accuracy": best_layer["accuracy"] if best_layer else None,
            "final_layer_accuracy": final_layer["accuracy"] if final_layer else None,
            "final_layer_margin": final_layer["median_margin"] if final_layer else None,
            "mean_layer_accuracy": (
                sum(float(row["accuracy"]) for row in layers) / len(layers)
                if layers
                else None
            ),
        },
    }


def compute_speed_eval(
    *,
    model: torch.nn.Module,
    sequences: list[torch.Tensor],
    device: torch.device,
    dataset: dict[str, Any] | None = None,
    max_length: int | None = None,
    batch_size: int = 1,
    decode_tokens: int = 32,
    warmup_iters: int = 1,
    measure_iters: int = 3,
) -> dict[str, Any]:
    if not sequences:
        raise ValueError("at least one evaluation sequence is required for speed benchmarking")
    model.eval()
    runtime_device = device
    sample = sequences[0][: min(128, int(sequences[0].numel()))].unsqueeze(0).to(runtime_device)

    def sync() -> None:
        if runtime_device.type == "cuda":
            torch.cuda.synchronize(runtime_device)

    if runtime_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime_device)

    with torch.no_grad():
        for _ in range(max(warmup_iters, 0)):
            _ = model(input_ids=sample)
        sync()

        start = time.perf_counter()
        for _ in range(max(measure_iters, 1)):
            _ = model(input_ids=sample)
        sync()
        prefill_elapsed = max(time.perf_counter() - start, 1e-9)

        start = time.perf_counter()
        generated = sample
        past_key_values = None
        for _ in range(max(decode_tokens, 1)):
            if past_key_values is None:
                outputs = model(input_ids=generated, use_cache=True)
            else:
                outputs = model(
                    input_ids=generated[:, -1:],
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            past_key_values = getattr(outputs, "past_key_values", None)
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=-1)
        sync()
        decode_elapsed = max(time.perf_counter() - start, 1e-9)

    params = sum(int(parameter.numel()) for parameter in model.parameters())
    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(runtime_device))
        if runtime_device.type == "cuda"
        else None
    )
    return {
        "schema_version": 1,
        "suite_id": "speed_eval",
        "device": str(runtime_device),
        "dtype": str(next(model.parameters()).dtype),
        "batch_size": batch_size,
        "seq_len": int(sample.size(1)),
        "prefill_tokens_per_sec": (sample.size(1) * max(measure_iters, 1)) / prefill_elapsed,
        "decode_tokens_per_sec": max(decode_tokens, 1) / decode_elapsed,
        "peak_memory_bytes": peak_memory_bytes,
        "parameter_count": params,
        "flops_per_token_estimate": float(params * 2),
    }


def compute_fewshot_eval(
    *,
    model_id: str,
    tasks: list[str] | None = None,
    tokenizer_id: str | None = None,
    batch_size: int | str = 1,
    device: str = "cpu",
    limit: int | float | None = None,
    trust_remote_code: bool = False,
    shots: list[int] | None = None,
) -> dict[str, Any]:
    selected_tasks = tasks or list(DEFAULT_FEWSHOT_TASKS)
    shot_values = shots or [0, 5]
    results_by_shot: dict[str, Any] = {}
    mean_score_by_shot: dict[str, float | None] = {}
    for shot in shot_values:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            payload = run_lm_eval(
                model_id=model_id,
                tasks=selected_tasks,
                output_path=temp_path,
                batch_size=batch_size,
                device=device,
                limit=limit,
                num_fewshot=shot,
                trust_remote_code=trust_remote_code,
            )
        finally:
            temp_path.unlink(missing_ok=True)
        task_scores: dict[str, Any] = {}
        numeric_scores: list[float] = []
        for task, task_payload in (payload.get("results") or {}).items():
            if not isinstance(task_payload, dict):
                continue
            metric_value = None
            metric_name = None
            for key in ("acc_norm,none", "acc,none", "mc2,none", "mc1,none"):
                if key in task_payload:
                    metric_name = key
                    metric_value = task_payload[key]
                    break
            task_scores[task] = {
                "metric": metric_name,
                "score": metric_value,
                "raw": task_payload,
            }
            if isinstance(metric_value, (int, float)):
                numeric_scores.append(float(metric_value))
        results_by_shot[str(shot)] = task_scores
        mean_score_by_shot[str(shot)] = (
            sum(numeric_scores) / len(numeric_scores) if numeric_scores else None
        )
    return {
        "schema_version": 1,
        "suite_id": "fewshot_eval",
        "tasks": selected_tasks,
        "shots": shot_values,
        "mean_score_by_shot": mean_score_by_shot,
        "results_by_shot": results_by_shot,
        "model_id": model_id,
        "tokenizer_id": tokenizer_id or model_id,
        "device": device,
        "limit": limit,
    }
