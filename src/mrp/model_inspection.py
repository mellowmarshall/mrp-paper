from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from transformers import AutoConfig

from mrp.model_loading import load_text_model, resolve_output_embeddings


MOE_HINT_FIELDS = (
    "num_experts",
    "num_local_experts",
    "n_experts",
    "moe_num_experts",
    "num_experts_per_tok",
    "num_experts_per_token",
    "expert_interval",
    "decoder_sparse_step",
    "moe_intermediate_size",
    "router_aux_loss_coef",
)


def _nontrivial(value: Any) -> bool:
    return value not in (None, False, 0, 1, "", [], {})


def _config_fields(config: Any, field_names: Iterable[str]) -> dict[str, Any]:
    return {
        name: getattr(config, name)
        for name in field_names
        if hasattr(config, name) and _nontrivial(getattr(config, name))
    }


def _effective_text_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def inspect_model(
    model_id: str,
    *,
    trust_remote_code: bool = False,
    load_weights: bool = True,
) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    config_dict = config.to_dict()
    text_config = _effective_text_config(config)
    text_config_dict = (
        text_config.to_dict() if hasattr(text_config, "to_dict") else config_dict
    )
    moe_hints = _config_fields(text_config, MOE_HINT_FIELDS)
    has_vision_config = hasattr(config, "vision_config")

    summary: dict[str, Any] = {
        "model_id": model_id,
        "architectures": config_dict.get("architectures", []),
        "model_type": config_dict.get("model_type"),
        "is_multimodal_wrapper": has_vision_config,
        "tie_word_embeddings_config": config_dict.get("tie_word_embeddings"),
        "config_moe_hints": moe_hints,
        "ffn_style_guess": "moe" if moe_hints else "dense",
        "text_config": {
            "model_type": text_config_dict.get("model_type"),
            "architectures": text_config_dict.get("architectures"),
            "vocab_size": text_config_dict.get("vocab_size"),
            "hidden_size": text_config_dict.get("hidden_size"),
            "intermediate_size": text_config_dict.get("intermediate_size"),
            "num_hidden_layers": text_config_dict.get("num_hidden_layers"),
            "num_attention_heads": text_config_dict.get("num_attention_heads"),
            "num_key_value_heads": text_config_dict.get("num_key_value_heads"),
            "max_position_embeddings": text_config_dict.get("max_position_embeddings"),
            "tie_word_embeddings": text_config_dict.get("tie_word_embeddings"),
        },
    }

    if not load_weights:
        return summary

    loaded = load_text_model(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype="auto",
    )
    model = loaded.model

    input_embeddings = model.get_input_embeddings()
    output_embeddings = resolve_output_embeddings(model)
    runtime_tied = False
    if (
        input_embeddings is not None
        and output_embeddings is not None
        and hasattr(input_embeddings, "weight")
        and hasattr(output_embeddings, "weight")
    ):
        runtime_tied = (
            input_embeddings.weight.data_ptr()
            == output_embeddings.weight.data_ptr()
        )

    module_types = sorted({type(module).__name__ for module in model.modules()})
    module_type_sample = module_types[:64]
    module_names_lower = {name.lower() for name in module_types}
    runtime_moe = any(
        "expert" in name or "moe" in name or "router" in name
        for name in module_names_lower
    )

    summary["runtime"] = {
        "load_strategy": loaded.load_strategy,
        "model_class": type(model).__name__,
        "parameter_count": int(sum(param.numel() for param in model.parameters())),
        "tied_word_embeddings_runtime": runtime_tied,
        "module_type_sample": module_type_sample,
        "runtime_moe_hints": runtime_moe,
        "ffn_style_runtime_guess": "moe" if runtime_moe else summary["ffn_style_guess"],
    }
    return summary
