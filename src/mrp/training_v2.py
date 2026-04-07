"""MRP training v2 -- thin wrapper over training.py.

The only new class is ``_MRPTrainerV2``, which replaces the inline
``compute_fisher_mrp_penalty`` with the canonical ``fisher.fisher_penalty``
from fisher.py. Everything else is imported from training.py.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from mrp.fisher import fisher_penalty
from mrp.model_loading import load_text_model, resolve_output_embeddings
from mrp.training import (
    MRP_MODE_COMBINED,
    MRP_MODE_CORRECT_MARGIN,
    MRP_MODE_DEPTH,
    MRP_MODE_ENTROPY,
    MRP_MODE_FINAL,
    MRP_MODE_FISHER,
    MRP_MODE_MARGIN_GATED,
    MRP_MODE_MARGIN_MAX,
    _GeometricSpotCheckCallback,
    _MetricsCallback,
    _S3CheckpointCallback,
    _StreamingBlockDataset,
    _TrackerCheckpointCallback,
    _align_for_causal_penalty,
    _apply_trainable_scope,
    _detect_text_column,
    _normalize_dtype,
    _patch_torch_load_check,
    _prepare_lm_dataset,
    _resolve_device,
    compute_correct_margin_loss,
    compute_depth_mrp_penalty,
    compute_entropy_penalty,
    compute_margin_gated_mrp_penalty,
    compute_margin_loss,
    compute_mrp_penalty,
)
from mrp.tracker import start_run
from mrp.utils import ensure_dir, write_json


class _MRPTrainerV2(Trainer):
    """MRPTrainer variant that uses :func:`fisher.fisher_penalty` (the canonical
    implementation) instead of the inline ``compute_fisher_mrp_penalty`` from
    training.py.

    All other loss modes are identical to ``MRPTrainer``.
    """

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
            # ---- KEY DIFFERENCE from MRPTrainer ----
            # Use canonical fisher.fisher_penalty instead of inline duplicate
            aligned_logits, aligned_labels = _align_for_causal_penalty(
                outputs.logits, labels,
            )
            valid_mask: torch.Tensor | None = None
            if aligned_labels is not None:
                valid_mask = aligned_labels != -100
            mrp_loss = fisher_penalty(
                aligned_logits,
                lm_head_weight,
                top_k=self.mrp_top_k,
                mask=valid_mask,
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


def train_mrp_v2(
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
    # ---- v2 additions ----
    streaming: bool = False,
    lr_schedule: str = "linear",
    s3_bucket: str | None = None,
    s3_prefix: str | None = None,
    save_total_limit: int = 5,
) -> dict[str, Any]:
    """Enhanced MRP continued-pretraining entry point.

    Extends :func:`training.train_mrp` with:

    - ``streaming`` -- use :class:`_StreamingBlockDataset` instead of
      :func:`_prepare_lm_dataset`
    - ``lr_schedule`` -- ``"linear"`` or ``"cosine"``
    - ``s3_bucket`` / ``s3_prefix`` -- adds :class:`_S3CheckpointCallback`
    - ``save_total_limit`` -- maximum number of checkpoints to keep
    - Uses :class:`_MRPTrainerV2` (canonical Fisher penalty)
    """
    _patch_torch_load_check()

    try:
        import accelerate  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "training requires `accelerate`. Run `uv sync --extra train` first."
        ) from exc

    from dataclasses import asdict

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
            "streaming": streaming,
            "lr_schedule": lr_schedule,
            "s3_bucket": s3_bucket,
            "s3_prefix": s3_prefix,
            "save_total_limit": save_total_limit,
        },
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        loaded = load_text_model(
            model_id, trust_remote_code=trust_remote_code, torch_dtype=resolved_dtype,
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

        # ---- Dataset ----
        if streaming:
            train_dataset = _StreamingBlockDataset(
                tokenizer, dataset_name, dataset_config, train_split, block_size,
            )
            resolved_text_column = text_column or "text"
        else:
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
        if eval_split and not streaming:
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
            lr_scheduler_type=lr_schedule,
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
            save_total_limit=save_total_limit,
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
        metrics_path = output_root / "metrics.jsonl"
        metrics_callback = _MetricsCallback(
            metrics_path,
            run_handle=run_handle,
            prefix="train",
        )

        # Prepare held-out data for geometric spot checks
        callbacks = [
            metrics_callback,
            _TrackerCheckpointCallback(output_root, run_handle),
        ]

        if s3_bucket and s3_prefix:
            callbacks.append(
                _S3CheckpointCallback(s3_bucket, s3_prefix, metrics_path)
            )

        if spot_check_interval > 0:
            spot_check_ds = load_dataset(
                dataset_name, dataset_config,
                split="validation" if eval_split else "train",
            )
            spot_text_col = _detect_text_column(
                spot_check_ds.column_names, text_column,
            )
            spot_texts = [
                r[spot_text_col]
                for r in spot_check_ds
                if isinstance(r.get(spot_text_col), str) and r[spot_text_col].strip()
            ][:50]
            spot_sequences: list[torch.Tensor] = []
            for text in spot_texts[:20]:
                enc = tokenizer(
                    text, return_tensors="pt", max_length=block_size,
                    truncation=True, add_special_tokens=False,
                )
                if enc["input_ids"].size(1) >= 4:
                    spot_sequences.append(enc["input_ids"][0])
            callbacks.append(_GeometricSpotCheckCallback(
                path=output_root / "geometric_spotcheck.jsonl",
                spot_check_data=spot_sequences,
                spot_check_interval=spot_check_interval,
                run_handle=run_handle,
            ))

        trainer = _MRPTrainerV2(
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

        summary: dict[str, Any] = {
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
            "streaming": streaming,
            "lr_schedule": lr_schedule,
            "s3_bucket": s3_bucket,
            "s3_prefix": s3_prefix,
            "save_total_limit": save_total_limit,
            "trainable": asdict(trainable_summary),
            "train_dataset_blocks": "streaming" if streaming else len(train_dataset),
            "eval_dataset_blocks": None if eval_dataset is None else len(eval_dataset),
            "train_metrics": {
                key: (
                    float(value)
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    else value
                )
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
        run_handle.log_artifact(
            "train_summary", output_root / "train_summary.json", kind="summary",
        )
        run_handle.log_artifact(
            "metrics_jsonl", output_root / "metrics.jsonl", kind="metrics",
        )
        if (output_root / "geometric_spotcheck.jsonl").exists():
            run_handle.log_artifact(
                "geometric_spotcheck_jsonl",
                output_root / "geometric_spotcheck.jsonl",
                kind="metrics",
            )
        if save_model and (output_root / "final_model").exists():
            run_handle.log_artifact(
                "final_model", output_root / "final_model", kind="model",
            )
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
