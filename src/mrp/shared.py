"""Deduplicated utilities shared across evaluation, training, and CLI.

Consolidates duplicated logic from:
- scripts/standard_eval.py (config construction, load_sequences)
- scripts/comprehensive_checkpoint_analysis.py (config construction)
- training.py (_detect_text_column, _resolve_device, config construction)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_TEXT_CANDIDATES: tuple[str, ...] = ("text", "content", "document")


def load_hf_dataset(
    dataset_name: str,
    dataset_config: str,
    *,
    split: str,
) -> Any:
    """Load a HuggingFace dataset, handling local JSONL files transparently.

    When *dataset_name* is ``"json"``, *dataset_config* is treated as a local
    file path (passed via ``data_files``).  Otherwise the standard
    ``load_dataset(name, config, split=...)`` call is used.
    """
    if dataset_name == "json":
        # For local JSONL files, HF datasets auto-creates only a "train" split.
        # Map data_files explicitly so the caller's split is respected, and
        # default to "train" when the caller doesn't specify otherwise.
        return load_dataset("json", data_files={split: dataset_config}, split=split)
    return load_dataset(dataset_name, dataset_config, split=split)


def _read_checkpoint_config(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalize_torch_dtype(torch_dtype: Any) -> Any:
    if torch_dtype in (None, "auto"):
        return "auto"
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if isinstance(torch_dtype, str):
        normalized = torch_dtype.lower()
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if normalized in mapping:
            return mapping[normalized]
    raise ValueError(f"unsupported torch_dtype: {torch_dtype!r}")


def _apply_model_dtype(model: torch.nn.Module, *, torch_dtype: Any) -> torch.nn.Module:
    resolved_dtype = _normalize_torch_dtype(torch_dtype)
    if resolved_dtype == "auto":
        return model
    return model.to(dtype=resolved_dtype)


def _select_local_mrp_model_class(path: Path) -> type[torch.nn.Module] | None:
    config = _read_checkpoint_config(path)
    architectures = config.get("architectures") or []
    preferred = architectures[0] if architectures else None

    from mrp.factored_transformer import FactoredCausalLM, StandardCausalLM

    if preferred == "FactoredCausalLM":
        return FactoredCausalLM
    if preferred == "StandardCausalLM":
        return StandardCausalLM
    return None


def resolve_device(device_str: str) -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    Handles ``"auto"`` by probing CUDA availability.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def detect_text_column(
    column_names: list[str] | tuple[str, ...],
    requested: str | None = None,
    candidates: tuple[str, ...] = DEFAULT_TEXT_CANDIDATES,
) -> str:
    """Return the text column name, validating *requested* or auto-detecting.

    Raises :class:`ValueError` when the column cannot be resolved.
    """
    if requested:
        if requested not in column_names:
            raise ValueError(
                f"text column '{requested}' not found. "
                f"Available columns: {list(column_names)}"
            )
        return requested

    for candidate in candidates:
        if candidate in column_names:
            return candidate

    raise ValueError(
        f"unable to infer text column from candidates {candidates}. "
        f"Available columns: {list(column_names)}"
    )


def load_eval_sequences(
    tokenizer: Any,
    *,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "validation",
    n_sequences: int = 64,
    max_length: int = 512,
    min_chars: int = 50,
) -> list[torch.Tensor]:
    """Load tokenized sequences from a HuggingFace dataset for evaluation.

    Returns a list of 1-D :class:`torch.Tensor` (``int64``), each with at
    least 32 tokens.
    """
    ds = load_hf_dataset(dataset_name, dataset_config, split=split)
    seqs: list[torch.Tensor] = []
    for example in ds:
        text = example.get("text", "")
        if not text or not text.strip() or len(text) < min_chars:
            continue
        enc = tokenizer(
            text.strip(),
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )
        if enc["input_ids"].size(1) >= 32:
            seqs.append(enc["input_ids"][0])
        if len(seqs) >= n_sequences:
            break
    return seqs


def build_factored_config(
    hidden_size: int,
    num_layers: int,
    vocab_size: int,
    *,
    max_position_embeddings: int = 2048,
    **overrides: Any,
) -> Any:
    """Build a :class:`FactoredTransformerConfig` with derived head counts.

    Derives ``num_heads``, ``num_kv_heads``, ``intermediate_size``, and
    foundation layer counts from *hidden_size* / *num_layers* following the
    project conventions.
    """
    from mrp.factored_transformer import FactoredTransformerConfig

    num_heads = max(1, hidden_size // 128)
    num_kv_heads = max(1, num_heads // 4)
    intermediate_size = int(hidden_size * 2.6875)
    foundation_layers = max(2, num_layers // 6)

    kwargs: dict[str, Any] = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "max_position_embeddings": max_position_embeddings,
        "num_foundation_layers_start": foundation_layers,
        "num_foundation_layers_end": foundation_layers,
        "residual_ratio": 0.5,
    }
    kwargs.update(overrides)
    return FactoredTransformerConfig(**kwargs)


def infer_mrp_architecture_from_state_dict_keys(
    state_dict_keys: list[str] | tuple[str, ...] | set[str],
) -> str:
    """Infer the MRP architecture name from serialized parameter keys."""
    keys = set(state_dict_keys)
    if any(key.startswith("layers.") for key in keys):
        return "StandardCausalLM"
    if any(key.startswith("model.foundation_start.") for key in keys):
        return "FactoredCausalLM"
    if any(key.startswith("model.shared_block.") for key in keys):
        return "FactoredCausalLM"
    if any(key.startswith("model.unique_residuals.") for key in keys):
        return "FactoredCausalLM"
    raise ValueError("unable to infer MRP architecture from state dict keys")


def _numbered_suffixes_for_prefix(
    state_dict_keys: set[str],
    prefix: str,
) -> list[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.")
    matches: set[int] = set()
    for key in state_dict_keys:
        match = pattern.match(key)
        if match:
            matches.add(int(match.group(1)))
    return sorted(matches)


def _infer_head_dim_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    rotary_key: str,
    hidden_size: int,
) -> int:
    inv_freq = state_dict.get(rotary_key)
    if inv_freq is not None and inv_freq.ndim == 1 and inv_freq.numel() > 0:
        return int(inv_freq.numel()) * 2
    if hidden_size % 128 == 0:
        return 128
    return hidden_size


def infer_mrp_config_overrides_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    tokenizer_id: str | None = None,
) -> dict[str, Any]:
    """Infer config overrides for a serialized MRP state dict.

    This is primarily for legacy ``model.pt`` checkpoints that predate
    ``config.json`` export. It aims to reconstruct the structural parameters
    needed to reload and re-export the checkpoint.
    """
    keys = set(state_dict.keys())
    architecture = infer_mrp_architecture_from_state_dict_keys(keys)

    if architecture == "StandardCausalLM":
        embed = state_dict["embed_tokens.weight"]
        hidden_size = int(embed.shape[1])
        vocab_size = int(embed.shape[0])
        layer_ids = _numbered_suffixes_for_prefix(keys, "layers.")
        num_layers = len(layer_ids)
        if num_layers == 0:
            raise ValueError("unable to infer num_layers from standard checkpoint")
        head_dim = _infer_head_dim_from_state_dict(
            state_dict,
            rotary_key="rotary.inv_freq",
            hidden_size=hidden_size,
        )
        q_proj = state_dict["layers.0.attn.q_proj.weight"]
        k_proj = state_dict["layers.0.attn.k_proj.weight"]
        intermediate_size = int(state_dict["layers.0.ffn.gate_proj.weight"].shape[0])
        config_overrides = {
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "num_layers": num_layers,
            "intermediate_size": intermediate_size,
            "num_attention_heads": int(q_proj.shape[0]) // head_dim,
            "num_kv_heads": max(1, int(k_proj.shape[0]) // head_dim),
            "max_position_embeddings": 2048,
            "num_foundation_layers_start": max(2, num_layers // 6),
            "num_foundation_layers_end": max(2, num_layers // 6),
            "residual_ratio": 0.5,
            "architectures": [architecture],
        }
    else:
        embed = state_dict["model.embed_tokens.weight"]
        hidden_size = int(embed.shape[1])
        vocab_size = int(embed.shape[0])
        start_ids = _numbered_suffixes_for_prefix(keys, "model.foundation_start.")
        end_ids = _numbered_suffixes_for_prefix(keys, "model.foundation_end.")
        residual_ids = _numbered_suffixes_for_prefix(keys, "model.unique_residuals.")
        num_layers = len(start_ids) + len(end_ids) + len(residual_ids)
        if num_layers == 0:
            raise ValueError("unable to infer num_layers from factored checkpoint")
        head_dim = _infer_head_dim_from_state_dict(
            state_dict,
            rotary_key="model.rotary.inv_freq",
            hidden_size=hidden_size,
        )
        q_proj = state_dict["model.shared_block.attn.q_proj.weight"]
        k_proj = state_dict["model.shared_block.attn.k_proj.weight"]
        intermediate_size = int(state_dict["model.shared_block.ffn.gate_proj.weight"].shape[0])
        config_overrides = {
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "num_layers": num_layers,
            "intermediate_size": intermediate_size,
            "num_attention_heads": int(q_proj.shape[0]) // head_dim,
            "num_kv_heads": max(1, int(k_proj.shape[0]) // head_dim),
            "max_position_embeddings": 2048,
            "num_foundation_layers_start": len(start_ids),
            "num_foundation_layers_end": len(end_ids),
            "residual_ratio": 0.5,
            "use_depth_embedding": "model.depth_embeddings.weight" in keys,
            "use_depth_conditional_norm": (
                "model.depth_scales" in keys and "model.depth_biases" in keys
            ),
            "architectures": [architecture],
        }

    if tokenizer_id is not None:
        config_overrides["tokenizer_id"] = tokenizer_id
    return config_overrides


def load_model_flexible(
    path: str | Path,
    *,
    device: str = "cpu",
    config_overrides: dict[str, Any] | None = None,
    tokenizer_id: str | None = None,
    trust_remote_code: bool = False,
    torch_dtype: Any = "auto",
) -> tuple[torch.nn.Module, Any]:
    """Load a model from a pretrained directory, a ``model.pt`` state dict, or
    a HuggingFace model ID.

    Returns ``(model, tokenizer)`` with the model on *device* in mode for
    inference.

    Handles three formats:

    1. **HuggingFace pretrained directory** (contains ``config.json``):
       loaded via :func:`~mrp.factored_transformer.StandardCausalLM.from_pretrained`
       or :func:`~mrp.model_loading.load_text_model`.
    2. **State-dict file** (``model.pt`` or ``*.pt``):
       requires *config_overrides* to build a ``FactoredTransformerConfig``.
    3. **HuggingFace Hub model ID** (string that is not an existing path):
       loaded via :func:`~mrp.model_loading.load_text_model`.
    """
    resolved_path = Path(path)
    resolved_device = resolve_device(device)

    # Determine tokenizer
    resolved_config_tokenizer_id = None
    if resolved_path.is_dir() and (resolved_path / "config.json").exists():
        config_payload = _read_checkpoint_config(resolved_path)
        tokenizer_from_config = config_payload.get("tokenizer_id")
        if tokenizer_from_config:
            resolved_config_tokenizer_id = str(tokenizer_from_config)
    tok_id = tokenizer_id or resolved_config_tokenizer_id or str(path)
    tokenizer = AutoTokenizer.from_pretrained(
        tok_id, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Case 1: existing directory with config.json => pretrained
    if resolved_path.is_dir() and (resolved_path / "config.json").exists():
        model_class = _select_local_mrp_model_class(resolved_path)
        if model_class is not None:
            model = model_class.from_pretrained(str(resolved_path))
            model.to(resolved_device)
            model = _apply_model_dtype(model, torch_dtype=torch_dtype)
            model.eval()
            return model, tokenizer

        from mrp.model_loading import load_text_model

        loaded = load_text_model(
            str(resolved_path),
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        model = loaded.model.to(resolved_device)
        model = _apply_model_dtype(model, torch_dtype=torch_dtype)
        model.eval()
        return model, tokenizer

    # Case 2: .pt state dict file
    if resolved_path.is_file() and resolved_path.suffix == ".pt":
        state_dict = torch.load(
            str(resolved_path), weights_only=True, map_location=resolved_device,
        )
        inferred_overrides = infer_mrp_config_overrides_from_state_dict(
            state_dict,
            tokenizer_id=tokenizer_id,
        )
        if config_overrides is not None:
            inferred_overrides.update(config_overrides)
        config = build_factored_config(
            hidden_size=inferred_overrides.get("hidden_size", 2048),
            num_layers=inferred_overrides.get("num_layers", 24),
            vocab_size=inferred_overrides.get("vocab_size", 32000),
            **{
                key: value
                for key, value in inferred_overrides.items()
                if key not in {"hidden_size", "num_layers", "vocab_size"}
            },
        )
        architecture = (
            inferred_overrides.get("architecture")
            or (inferred_overrides.get("architectures") or [None])[0]
            or "StandardCausalLM"
        )
        from mrp.factored_transformer import FactoredCausalLM, StandardCausalLM

        model_class = (
            FactoredCausalLM if architecture == "FactoredCausalLM" else StandardCausalLM
        )
        model = model_class(config)
        model.load_state_dict(state_dict)
        model.to(resolved_device)
        model = _apply_model_dtype(model, torch_dtype=torch_dtype)
        model.eval()
        return model, tokenizer

    # Case 3: HuggingFace Hub model ID (not a local path)
    from mrp.model_loading import load_text_model

    loaded = load_text_model(
        str(path),
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    model = loaded.model.to(resolved_device)
    model = _apply_model_dtype(model, torch_dtype=torch_dtype)
    model.eval()
    return model, tokenizer
