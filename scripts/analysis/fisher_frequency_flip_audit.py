"""Frequency-stratified flip audit for Fisher-polished models.

This compares a source model against a Fisher-polished model on the same held-
out evaluation sequences, then buckets positions by target-token frequency so
aggregate gains can be separated from long-tail regressions.

Usage:
    python scripts/analysis/fisher_frequency_flip_audit.py \
        --source-model results/bl264_ckpt50000 \
        --post-model results/polishing/fisher_precision_fp64/264m_baseline_50k_noprotect/fp32/final_model \
        --run-root results/polishing/fisher_precision_fp64/264m_baseline_50k_noprotect/fp32 \
        --device cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    SRC_ROOT = REPO_ROOT / "src"
    for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

from mrp.eval_artifacts import (
    atomic_write_json,
    finalize_eval_artifacts,
    infer_run_root_from_path,
    resolve_suite_output_dir,
)
from mrp.shared import load_eval_sequences, load_model_flexible


DEFAULT_TOKENIZER_ID = "HuggingFaceTB/SmolLM3-3B"
BUCKET_ORDER = ("freq_1", "freq_2_4", "freq_5_19", "freq_20_99", "freq_100_plus")
BUCKET_LABELS = {
    "freq_1": "1",
    "freq_2_4": "2-4",
    "freq_5_19": "5-19",
    "freq_20_99": "20-99",
    "freq_100_plus": "100+",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frequency-stratified Fisher flip audit.")
    parser.add_argument("--source-model", default=None, help="Original checkpoint or final_model path.")
    parser.add_argument("--post-model", default=None, help="Fisher-polished final_model path.")
    parser.add_argument("--source-token-stats-csv", default=None, help="Optional baseline audit token_stats_fp32.csv.")
    parser.add_argument("--post-token-stats-csv", default=None, help="Optional Fisher audit token_stats_fp32.csv.")
    parser.add_argument("--output-dir", default=None, help="Optional explicit output directory.")
    parser.add_argument("--run-root", default=None, help="Optional explicit run root for canonical eval output.")
    parser.add_argument("--tokenizer-id", default=None, help="Optional tokenizer id or local tokenizer directory.")
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n-sequences", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-chars", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--rare-threshold", type=int, default=4, help="Max corpus frequency treated as rare for token tables.")
    parser.add_argument("--top-k-tokens", type=int, default=20, help="Rows to keep in each token highlight table.")
    return parser


def _resolve_model_path(raw: str | Path) -> str | Path:
    raw_str = str(raw)
    expanded = Path(raw_str).expanduser()
    if not expanded.exists():
        return raw_str
    path = expanded.resolve()
    if path.is_dir():
        if not (path / "config.json").exists() and (path / "model.pt").exists():
            return (path / "model.pt").resolve()
        if any(
            (path / filename).exists()
            for filename in (
                "config.json",
                "model.safetensors",
                "pytorch_model.bin",
                "tokenizer.json",
                "tokenizer_config.json",
            )
        ):
            return path
        return raw_str
    return path


def _resolve_optional_path(raw: str | Path | None) -> Path | None:
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _read_config_tokenizer_id(path: Path) -> str | None:
    config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    tokenizer_id = payload.get("tokenizer_id")
    return str(tokenizer_id) if tokenizer_id else None


def _read_audit_tokenizer_id(path: Path | None) -> str | None:
    if path is None:
        return None
    audit_meta = path.parent / "audit_metadata.json"
    if not audit_meta.exists():
        return None
    try:
        payload = json.loads(audit_meta.read_text(encoding="utf-8"))
    except Exception:
        return None
    tokenizer_id = payload.get("tokenizer_id")
    return str(tokenizer_id) if tokenizer_id else None


def infer_tokenizer_ref(
    *,
    source_model: str | Path | None,
    post_model: str | Path | None,
    source_token_stats_csv: Path | None = None,
    post_token_stats_csv: Path | None = None,
    explicit: str | None = None,
) -> str:
    if explicit:
        return explicit
    if isinstance(post_model, Path) and post_model.is_dir() and ((post_model / "tokenizer.json").exists() or (post_model / "tokenizer_config.json").exists()):
        return str(post_model)
    if isinstance(source_model, Path) and source_model.is_dir() and ((source_model / "tokenizer.json").exists() or (source_model / "tokenizer_config.json").exists()):
        return str(source_model)
    for candidate in (
        _read_config_tokenizer_id(post_model) if isinstance(post_model, Path) else None,
        _read_config_tokenizer_id(source_model) if isinstance(source_model, Path) else None,
        _read_audit_tokenizer_id(post_token_stats_csv),
        _read_audit_tokenizer_id(source_token_stats_csv),
    ):
        if candidate:
            return candidate
    return DEFAULT_TOKENIZER_ID


def _bucket_name(freq: int) -> str:
    if freq <= 1:
        return "freq_1"
    if freq <= 4:
        return "freq_2_4"
    if freq <= 19:
        return "freq_5_19"
    if freq <= 99:
        return "freq_20_99"
    return "freq_100_plus"


def _safe_ratio(num: int, den: int) -> float | None:
    return (num / den) if den else None


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id])
    except Exception:
        return str(token_id)


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def build_reference_token_counts(
    tokenizer: Any,
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
) -> Counter[int]:
    counts: Counter[int] = Counter()
    ds = load_dataset(dataset_name, dataset_config, split=split)
    for example in ds:
        text = (example.get("text") or "").strip()
        if not text:
            continue
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )
        ids = enc["input_ids"][0].tolist()
        counts.update(ids)
    return counts


def collect_predictions(
    model: torch.nn.Module,
    sequences: list[torch.Tensor],
    *,
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    device_obj = torch.device(device)
    with torch.no_grad():
        for seq_idx, seq in enumerate(sequences):
            ids = seq.unsqueeze(0).to(device_obj)
            labels = ids[0, 1:].cpu()
            logits = model(input_ids=ids).logits[0, :-1, :].float().cpu()
            top2_vals, top2_ids = logits.topk(2, dim=-1)
            top1 = top2_ids[:, 0]
            margins = top2_vals[:, 0] - top2_vals[:, 1]
            correct = top1.eq(labels)
            for pos in range(labels.numel()):
                rows.append(
                    {
                        "seq_id": seq_idx,
                        "position": pos,
                        "target_id": int(labels[pos].item()),
                        "pred_id": int(top1[pos].item()),
                        "margin": float(margins[pos].item()),
                        "correct": bool(correct[pos].item()),
                    }
                )
            if (seq_idx + 1) % 10 == 0:
                print(f"  audited {seq_idx + 1}/{len(sequences)} sequences")
    return rows


def load_rows_from_token_stats(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                {
                    "seq_id": int(raw["sequence_id"]),
                    "position": int(raw["position"]),
                    "target_id": int(raw["target_token_id"]),
                    "pred_id": int(raw["top1_token_id"]),
                    "margin": float(raw["margin"]),
                    "correct": raw["top1_is_correct"] == "1",
                }
            )
    return rows


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    pre_correct = sum(1 for row in rows if row["pre_correct"])
    post_correct = sum(1 for row in rows if row["post_correct"])
    w2r = sum(1 for row in rows if (not row["pre_correct"]) and row["post_correct"])
    r2w = sum(1 for row in rows if row["pre_correct"] and (not row["post_correct"]))
    pre_margins = [float(row["pre_margin"]) for row in rows]
    post_margins = [float(row["post_margin"]) for row in rows]
    return {
        "n_positions": count,
        "pre_accuracy": _safe_ratio(pre_correct, count),
        "post_accuracy": _safe_ratio(post_correct, count),
        "delta_accuracy": (_safe_ratio(post_correct, count) - _safe_ratio(pre_correct, count)) if count else None,
        "w2r": w2r,
        "r2w": r2w,
        "net_corrected": w2r - r2w,
        "flip_ratio": _safe_ratio(w2r, r2w) if r2w else (float("inf") if w2r else None),
        "median_pre_margin": _median_or_none(pre_margins),
        "median_post_margin": _median_or_none(post_margins),
        "delta_median_margin": (
            _median_or_none(post_margins) - _median_or_none(pre_margins)
            if pre_margins and post_margins
            else None
        ),
    }


def summarize_tokens(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    rare_threshold: int,
    top_k: int,
) -> dict[str, Any]:
    by_token: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_token[int(row["target_id"])].append(row)

    token_rows: list[dict[str, Any]] = []
    for token_id, token_rows_raw in by_token.items():
        freq = int(token_rows_raw[0]["corpus_frequency"])
        positions = len(token_rows_raw)
        pre_correct = sum(1 for row in token_rows_raw if row["pre_correct"])
        post_correct = sum(1 for row in token_rows_raw if row["post_correct"])
        w2r = sum(1 for row in token_rows_raw if (not row["pre_correct"]) and row["post_correct"])
        r2w = sum(1 for row in token_rows_raw if row["pre_correct"] and (not row["post_correct"]))
        token_rows.append(
            {
                "token_id": token_id,
                "token": _decode_token(tokenizer, token_id),
                "corpus_frequency": freq,
                "positions": positions,
                "pre_accuracy": _safe_ratio(pre_correct, positions),
                "post_accuracy": _safe_ratio(post_correct, positions),
                "delta_accuracy": (_safe_ratio(post_correct, positions) - _safe_ratio(pre_correct, positions)) if positions else None,
                "w2r": w2r,
                "r2w": r2w,
                "net_corrected": w2r - r2w,
                "flip_ratio": _safe_ratio(w2r, r2w) if r2w else (float("inf") if w2r else None),
            }
        )

    rare_rows = [row for row in token_rows if row["corpus_frequency"] <= rare_threshold]
    damaging_rare = sorted(
        [row for row in rare_rows if row["r2w"] > 0],
        key=lambda row: (-row["r2w"], row["w2r"], -row["positions"], row["token"]),
    )[:top_k]
    helpful_rare = sorted(
        [row for row in rare_rows if row["w2r"] > 0],
        key=lambda row: (-row["w2r"], row["r2w"], -row["positions"], row["token"]),
    )[:top_k]
    most_damaging_all = sorted(
        [row for row in token_rows if row["r2w"] > 0],
        key=lambda row: (-row["r2w"], row["w2r"], -row["positions"], row["token"]),
    )[:top_k]
    most_helpful_all = sorted(
        [row for row in token_rows if row["w2r"] > 0],
        key=lambda row: (-row["w2r"], row["r2w"], -row["positions"], row["token"]),
    )[:top_k]

    return {
        "per_token": token_rows,
        "rare_regressions": damaging_rare,
        "rare_improvements": helpful_rare,
        "top_regressions_all": most_damaging_all,
        "top_improvements_all": most_helpful_all,
    }


def build_summary_markdown(
    *,
    source_model: Path,
    post_model: Path,
    tokenizer_ref: str,
    n_positions: int,
    bucket_summaries: dict[str, dict[str, Any]],
    token_summary: dict[str, Any],
    rare_threshold: int,
) -> str:
    lines = [
        "## Frequency-Stratified Fisher Flip Audit",
        "",
        f"- source model: `{source_model}`",
        f"- post model: `{post_model}`",
        f"- tokenizer ref: `{tokenizer_ref}`",
        f"- audited positions: `{n_positions}`",
        "",
        "### Frequency buckets (target token corpus frequency in Wikitext validation)",
        "",
        "```text",
        "Bucket   NPos   PreAcc   PostAcc  dAcc     W->R  R->W  Net   FlipRt   MedPre   MedPost  dMed",
        "------   ----   ------   -------  ------   ----  ----  ---   ------   ------   -------  ------",
    ]
    for bucket in BUCKET_ORDER:
        summary = bucket_summaries[bucket]
        lines.append(
            f"{BUCKET_LABELS[bucket]:<6}   "
            f"{summary['n_positions']:<4d}   "
            f"{_format_ratio(summary['pre_accuracy']):>6}   "
            f"{_format_ratio(summary['post_accuracy']):>7}  "
            f"{_format_ratio(summary['delta_accuracy']):>6}   "
            f"{summary['w2r']:<4d}  "
            f"{summary['r2w']:<4d}  "
            f"{summary['net_corrected']:<4d}  "
            f"{_format_ratio(summary['flip_ratio']):>6}   "
            f"{_format_ratio(summary['median_pre_margin']):>6}   "
            f"{_format_ratio(summary['median_post_margin']):>7}  "
            f"{_format_ratio(summary['delta_median_margin']):>6}"
        )
    lines.extend(["```", ""])

    def _token_block(title: str, rows: list[dict[str, Any]]) -> None:
        lines.extend([f"### {title}", "", "```text"])
        if not rows:
            lines.append("none")
        else:
            lines.append("Token         Freq  Pos  W->R  R->W  Net  PreAcc  PostAcc  FlipRt")
            lines.append("-----------   ----  ---  ----  ----  ---  ------  -------  ------")
            for row in rows:
                token = repr(row["token"])
                if len(token) > 11:
                    token = token[:10] + "…"
                lines.append(
                    f"{token:<11}   "
                    f"{row['corpus_frequency']:<4d}  "
                    f"{row['positions']:<3d}  "
                    f"{row['w2r']:<4d}  "
                    f"{row['r2w']:<4d}  "
                    f"{row['net_corrected']:<3d}  "
                    f"{_format_ratio(row['pre_accuracy']):>6}  "
                    f"{_format_ratio(row['post_accuracy']):>7}  "
                    f"{_format_ratio(row['flip_ratio']):>6}"
                )
        lines.extend(["```", ""])

    _token_block(f"Rare regressions (corpus frequency <= {rare_threshold})", token_summary["rare_regressions"])
    _token_block(f"Rare improvements (corpus frequency <= {rare_threshold})", token_summary["rare_improvements"])
    _token_block("Largest regressions overall", token_summary["top_regressions_all"])
    _token_block("Largest improvements overall", token_summary["top_improvements_all"])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    csv_mode = bool(args.source_token_stats_csv or args.post_token_stats_csv)
    if csv_mode and not (args.source_token_stats_csv and args.post_token_stats_csv):
        parser.error("--source-token-stats-csv and --post-token-stats-csv must be provided together")
    if not csv_mode and not (args.source_model and args.post_model):
        parser.error("either provide --source-model/--post-model or both --source-token-stats-csv/--post-token-stats-csv")

    source_model = _resolve_model_path(args.source_model) if args.source_model else None
    post_model = _resolve_model_path(args.post_model) if args.post_model else None
    source_token_stats_csv = _resolve_optional_path(args.source_token_stats_csv)
    post_token_stats_csv = _resolve_optional_path(args.post_token_stats_csv)
    tokenizer_ref = infer_tokenizer_ref(
        source_model=source_model,
        post_model=post_model,
        source_token_stats_csv=source_token_stats_csv,
        post_token_stats_csv=post_token_stats_csv,
        explicit=args.tokenizer_id,
    )

    model_ref = (
        post_model
        if post_model is not None
        else (post_token_stats_csv.parent.parent if post_token_stats_csv is not None else Path.cwd())
    )
    default_run_root = (
        infer_run_root_from_path(model_ref if isinstance(model_ref, Path) else Path.cwd())
        or (post_token_stats_csv.parent.parent if post_token_stats_csv is not None else None)
        or (source_token_stats_csv.parent.parent if source_token_stats_csv is not None else None)
        or (model_ref.parent if isinstance(model_ref, Path) else Path.cwd())
    )
    output_dir, paths = resolve_suite_output_dir(
        suite_id="frequency_flip_audit",
        output_dir=args.output_dir,
        model_ref=model_ref,
        run_root=args.run_root or default_run_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    if csv_mode:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, trust_remote_code=args.trust_remote_code)
        print("Loading precomputed audit CSVs...")
        pre_rows = load_rows_from_token_stats(source_token_stats_csv)
        post_rows = load_rows_from_token_stats(post_token_stats_csv)
        sequences_count = None
    else:
        print("Loading source model...")
        source_loaded, tokenizer = load_model_flexible(
            source_model,
            tokenizer_id=tokenizer_ref,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
        )

        print("Loading audit sequences...")
        sequences = load_eval_sequences(
            tokenizer,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            n_sequences=args.n_sequences,
            max_length=args.max_length,
            min_chars=args.min_chars,
        )
        sequences_count = len(sequences)
        print(f"Loaded {sequences_count} sequences")

    print("Counting reference token frequencies...")
    reference_counts = build_reference_token_counts(
        tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
    )
    if not csv_mode:
        print("Collecting source predictions...")
        pre_rows = collect_predictions(source_loaded, sequences, device=args.device)
        del source_loaded
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print("Loading post model...")
        post_loaded, _ = load_model_flexible(
            post_model,
            tokenizer_id=tokenizer_ref,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
        )
        print("Collecting post predictions...")
        post_rows = collect_predictions(post_loaded, sequences, device=args.device)
        del post_loaded
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if len(pre_rows) != len(post_rows):
        raise ValueError(f"pre/post position count mismatch: {len(pre_rows)} vs {len(post_rows)}")

    merged_rows: list[dict[str, Any]] = []
    by_bucket: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in BUCKET_ORDER}
    for pre, post in zip(pre_rows, post_rows):
        if pre["seq_id"] != post["seq_id"] or pre["position"] != post["position"] or pre["target_id"] != post["target_id"]:
            raise ValueError("pre/post rows are not aligned")
        freq = int(reference_counts.get(pre["target_id"], 0))
        bucket = _bucket_name(freq)
        row = {
            "seq_id": pre["seq_id"],
            "position": pre["position"],
            "target_id": pre["target_id"],
            "target_token": _decode_token(tokenizer, pre["target_id"]),
            "corpus_frequency": freq,
            "bucket": bucket,
            "pre_pred_id": pre["pred_id"],
            "pre_pred_token": _decode_token(tokenizer, pre["pred_id"]),
            "pre_margin": pre["margin"],
            "pre_correct": pre["correct"],
            "post_pred_id": post["pred_id"],
            "post_pred_token": _decode_token(tokenizer, post["pred_id"]),
            "post_margin": post["margin"],
            "post_correct": post["correct"],
        }
        merged_rows.append(row)
        by_bucket[bucket].append(row)

    bucket_summaries = {bucket: summarize_bucket(by_bucket[bucket]) for bucket in BUCKET_ORDER}
    overall = summarize_bucket(merged_rows)
    token_summary = summarize_tokens(
        merged_rows,
        tokenizer=tokenizer,
        rare_threshold=args.rare_threshold,
        top_k=args.top_k_tokens,
    )

    payload = {
        "source_model": str(source_model) if source_model is not None else None,
        "post_model": str(post_model) if post_model is not None else None,
        "source_token_stats_csv": str(source_token_stats_csv) if source_token_stats_csv is not None else None,
        "post_token_stats_csv": str(post_token_stats_csv) if post_token_stats_csv is not None else None,
        "tokenizer_ref": tokenizer_ref,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "n_sequences": sequences_count,
        "max_length": args.max_length,
        "n_positions": len(merged_rows),
        "rare_threshold": args.rare_threshold,
        "mode": "csv" if csv_mode else "model",
        "bucket_definitions": {bucket: BUCKET_LABELS[bucket] for bucket in BUCKET_ORDER},
        "overall": overall,
        "by_bucket": bucket_summaries,
        "token_summary": token_summary,
        "elapsed_seconds": round(time.time() - start, 3),
    }

    summary_md = build_summary_markdown(
        source_model=source_model or source_token_stats_csv,
        post_model=post_model or post_token_stats_csv,
        tokenizer_ref=tokenizer_ref,
        n_positions=len(merged_rows),
        bucket_summaries=bucket_summaries,
        token_summary=token_summary,
        rare_threshold=args.rare_threshold,
    )

    payload_path = output_dir / "frequency_flip_audit.json"
    token_path = output_dir / "token_frequency_summary.json"
    positions_path = output_dir / "positions.jsonl"
    summary_path = output_dir / "summary.md"
    atomic_write_json(payload_path, payload)
    atomic_write_json(token_path, token_summary)
    with positions_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary_path.write_text(summary_md, encoding="utf-8")

    finalize_eval_artifacts(
        paths=paths,
        model_ref=model_ref,
        tokenizer_ref=tokenizer_ref,
        checkpoint_ref=model_ref,
        dataset={
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "n_sequences": sequences_count,
            "max_length": args.max_length,
            "n_positions": len(merged_rows),
            "frequency_reference_split": args.split,
        },
        command=" ".join(sys.argv),
        status="completed",
        source_artifacts=[
            ref
            for ref in (
                str(source_model) if source_model is not None else None,
                str(source_token_stats_csv) if source_token_stats_csv is not None else None,
            )
            if ref is not None
        ],
        artifacts=[payload_path, token_path, positions_path, summary_path],
        metadata={
            "source_model": str(source_model) if source_model is not None else None,
            "post_model": str(post_model) if post_model is not None else None,
            "source_token_stats_csv": str(source_token_stats_csv) if source_token_stats_csv is not None else None,
            "post_token_stats_csv": str(post_token_stats_csv) if post_token_stats_csv is not None else None,
            "mode": "csv" if csv_mode else "model",
            "rare_threshold": args.rare_threshold,
            "top_k_tokens": args.top_k_tokens,
        },
    )

    print(summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
