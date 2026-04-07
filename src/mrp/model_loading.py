from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import transformers
from transformers import AutoConfig, AutoModelForCausalLM

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5ForConditionalGeneration = None


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    config: Any
    load_strategy: str


def _requires_qwen35_wrapper(config: Any) -> bool:
    return (
        type(config).__name__ == "Qwen3_5Config"
        and hasattr(config, "text_config")
        and not hasattr(config, "vocab_size")
    )


def _requires_gemma4_wrapper(config: Any) -> bool:
    return (
        getattr(config, "model_type", None) == "gemma4"
        and hasattr(config, "text_config")
    )


def _resolve_wrapper_loader(config: Any) -> tuple[str, str, str] | None:
    if _requires_qwen35_wrapper(config):
        return (
            "Qwen3_5ForConditionalGeneration",
            "qwen3_5_conditional_wrapper",
            "transformers>=5.3",
        )
    if _requires_gemma4_wrapper(config):
        return (
            "Gemma4ForConditionalGeneration",
            "gemma4_conditional_wrapper",
            "a Gemma 4-capable transformers build",
        )
    return None


def _load_wrapper_model(
    *,
    class_name: str,
    model_id: str,
    trust_remote_code: bool,
    torch_dtype: str,
    low_cpu_mem_usage: bool,
) -> Any:
    model_class = getattr(transformers, class_name, None)
    if model_class is None:
        raise RuntimeError(
            f"{class_name} is not available in this transformers version."
        )
    return model_class.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )


def load_text_model(
    model_id: str,
    *,
    trust_remote_code: bool = False,
    torch_dtype: str = "auto",
    low_cpu_mem_usage: bool = True,
) -> LoadedModel:
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )

    wrapper_spec = _resolve_wrapper_loader(config)
    if wrapper_spec is not None:
        class_name, load_strategy, version_hint = wrapper_spec
        if (
            class_name == "Qwen3_5ForConditionalGeneration"
            and Qwen3_5ForConditionalGeneration is not None
        ):
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
        else:
            try:
                model = _load_wrapper_model(
                    class_name=class_name,
                    model_id=model_id,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{class_name} is required to load {model_id}. "
                    f"Upgrade to {version_hint}."
                ) from exc
        return LoadedModel(
            model=model,
            config=config,
            load_strategy=load_strategy,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    return LoadedModel(
        model=model,
        config=config,
        load_strategy="auto_causal_lm",
    )


def resolve_output_embeddings(model: Any) -> Any:
    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            return output_embeddings

    return getattr(model, "lm_head", None)
