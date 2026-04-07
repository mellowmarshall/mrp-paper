from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from mrp.model_loading import load_text_model
from mrp.utils import write_json


def _parse_tasks(tasks: str | list[str]) -> list[str]:
    if isinstance(tasks, str):
        return [task.strip() for task in tasks.split(",") if task.strip()]
    return [task.strip() for task in tasks if task.strip()]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def run_lm_eval(
    *,
    model_id: str,
    tasks: str | list[str],
    output_path: str | Path,
    batch_size: int | str = 1,
    device: str = "cpu",
    torch_dtype: str = "auto",
    limit: int | float | None = None,
    num_fewshot: int | None = None,
    gen_kwargs: str | dict[str, Any] | None = None,
    bootstrap_iters: int = 0,
    log_samples: bool = False,
    trust_remote_code: bool = False,
    verbosity: str = "INFO",
) -> dict[str, Any]:
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise RuntimeError(
            "lm-evaluation-harness is not installed. Run `uv sync --extra eval` first."
        ) from exc

    task_list = _parse_tasks(tasks)
    if not task_list:
        raise ValueError("at least one task is required")

    loaded = load_text_model(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )

    lm = HFLM(
        pretrained=loaded.model,
        tokenizer=tokenizer,
        backend="causal",
        device=device,
        dtype=torch_dtype,
        batch_size=batch_size,
        trust_remote_code=trust_remote_code,
    )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
        gen_kwargs=gen_kwargs,
        bootstrap_iters=bootstrap_iters,
        log_samples=log_samples,
        verbosity=verbosity,
    )
    results["mrp_metadata"] = {
        "model_id": model_id,
        "load_strategy": loaded.load_strategy,
        "tasks": task_list,
        "device": device,
        "dtype": str(torch_dtype),
        "limit": limit,
        "num_fewshot": num_fewshot,
        "gen_kwargs": gen_kwargs,
        "bootstrap_iters": bootstrap_iters,
        "log_samples": log_samples,
    }

    payload = _json_safe(results)
    write_json(output_path, payload)
    return payload
