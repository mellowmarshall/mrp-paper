"""Heuristic token-class flip audit for polished models.

This compares aligned per-position audit CSVs and groups target tokens into
practical classes such as structural tokens, numbers, function words, content
words, and entity-like words. It is intended as a lightweight token-value
follow-up when full model weights are not needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional in positions-jsonl replay mode
    AutoTokenizer = None

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


CLASS_ORDER = (
    "structural",
    "numeric",
    "function_word",
    "entity_like",
    "content_word",
    "fragment_other",
)

CLASS_LABELS = {
    "structural": "Structural",
    "numeric": "Numeric",
    "function_word": "Function",
    "entity_like": "EntityLike",
    "content_word": "Content",
    "fragment_other": "FragmentOther",
}

FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "to", "of", "in", "on", "at", "by", "for", "from", "with",
    "without", "about", "over", "under", "through", "into", "onto", "off", "up",
    "down", "out", "as", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing", "have", "has", "had", "having", "can", "could",
    "will", "would", "shall", "should", "may", "might", "must", "not", "no", "nor",
    "so", "because", "while", "where", "when", "who", "whom", "whose", "which",
    "what", "why", "how", "it", "its", "it's", "he", "him", "his", "she", "her",
    "hers", "they", "them", "their", "theirs", "we", "us", "our", "ours", "you",
    "your", "yours", "i", "me", "my", "mine", "there", "here", "also", "more",
    "most", "some", "any", "each", "every", "all", "both", "either", "neither",
    "few", "many", "much", "such", "own", "same", "other",
}

NUMERIC_RE = re.compile(r"^[0-9][0-9,./:%+-]*$")
ALPHA_RE = re.compile(r"^[A-Za-z]+$")
ENTITY_RE = re.compile(r"^[A-Z][A-Za-z]+$")
ACRONYM_RE = re.compile(r"^[A-Z]{2,}$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Heuristic token-class flip audit.")
    parser.add_argument("--source-token-stats-csv", default=None, help="Baseline audit token_stats_fp32.csv.")
    parser.add_argument("--post-token-stats-csv", default=None, help="Post-polish audit token_stats_fp32.csv.")
    parser.add_argument(
        "--positions-jsonl",
        default=None,
        help="Optional frequency_flip_audit positions.jsonl to replay without original token_stats CSVs.",
    )
    parser.add_argument("--tokenizer-id", default=None, help="Tokenizer id or local tokenizer directory.")
    parser.add_argument("--output-dir", default=None, help="Optional explicit output directory.")
    parser.add_argument("--run-root", default=None, help="Optional explicit run root for canonical eval output.")
    parser.add_argument("--top-k-tokens", type=int, default=15, help="Rows to keep in token highlight tables.")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def _resolve_path(raw: str | Path) -> Path:
    return Path(raw).expanduser().resolve()


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


def load_rows_from_positions_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            rows.append(
                {
                    "seq_id": int(raw["seq_id"]),
                    "position": int(raw["position"]),
                    "target_id": int(raw["target_id"]),
                    "target_token": str(raw["target_token"]),
                    "pre_pred_id": int(raw["pre_pred_id"]),
                    "pre_pred_token": str(raw["pre_pred_token"]),
                    "pre_margin": float(raw["pre_margin"]),
                    "pre_correct": bool(raw["pre_correct"]),
                    "post_pred_id": int(raw["post_pred_id"]),
                    "post_pred_token": str(raw["post_pred_token"]),
                    "post_margin": float(raw["post_margin"]),
                    "post_correct": bool(raw["post_correct"]),
                }
            )
    return rows


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id])
    except Exception:
        return str(token_id)


def _is_structural(text: str) -> bool:
    stripped = text.strip()
    if stripped == "":
        return True
    for ch in stripped:
        cat = unicodedata.category(ch)
        if not (cat.startswith("P") or cat.startswith("S")):
            return False
    return True


def classify_token(text: str) -> str:
    stripped = text.strip()
    if _is_structural(text):
        return "structural"
    if NUMERIC_RE.match(stripped):
        return "numeric"
    if ALPHA_RE.match(stripped):
        lowered = stripped.lower()
        if lowered in FUNCTION_WORDS:
            return "function_word"
        if ENTITY_RE.match(stripped) or ACRONYM_RE.match(stripped):
            return "entity_like"
        return "content_word"
    if stripped and any(ch.isalpha() for ch in stripped):
        if ENTITY_RE.match(stripped) or ACRONYM_RE.match(stripped):
            return "entity_like"
    return "fragment_other"


def _safe_ratio(num: int, den: int) -> float | None:
    return (num / den) if den else None


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
    top_k: int,
) -> dict[str, Any]:
    by_token: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_token[int(row["target_id"])].append(row)

    token_rows: list[dict[str, Any]] = []
    for token_id, token_rows_raw in by_token.items():
        positions = len(token_rows_raw)
        pre_correct = sum(1 for row in token_rows_raw if row["pre_correct"])
        post_correct = sum(1 for row in token_rows_raw if row["post_correct"])
        w2r = sum(1 for row in token_rows_raw if (not row["pre_correct"]) and row["post_correct"])
        r2w = sum(1 for row in token_rows_raw if row["pre_correct"] and (not row["post_correct"]))
        token_rows.append(
            {
                "token_id": token_id,
                "token": token_rows_raw[0]["target_token"],
                "token_class": token_rows_raw[0]["token_class"],
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

    by_class: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for token_class in CLASS_ORDER:
        class_rows = [row for row in token_rows if row["token_class"] == token_class]
        by_class[token_class] = {
            "top_improvements": sorted(
                [row for row in class_rows if row["w2r"] > 0],
                key=lambda row: (-row["w2r"], row["r2w"], -row["positions"], row["token"]),
            )[:top_k],
            "top_regressions": sorted(
                [row for row in class_rows if row["r2w"] > 0],
                key=lambda row: (-row["r2w"], row["w2r"], -row["positions"], row["token"]),
            )[:top_k],
        }
    return {"per_token": token_rows, "by_class": by_class}


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def build_summary_markdown(
    *,
    source_ref: str,
    post_ref: str,
    tokenizer_ref: str | None,
    n_positions: int,
    class_summaries: dict[str, dict[str, Any]],
    token_summary: dict[str, Any],
) -> str:
    lines = [
        "## Token-Class Flip Audit",
        "",
        f"- source reference: `{source_ref}`",
        f"- post reference: `{post_ref}`",
        f"- tokenizer ref: `{tokenizer_ref}`" if tokenizer_ref else "- tokenizer ref: decoded target tokens from positions payload",
        f"- audited positions: `{n_positions}`",
        "- classes are heuristic and based on decoded target-token strings",
        "",
        "### Token classes",
        "",
        "```text",
        "Class          NPos    PreAcc   PostAcc  dAcc     W->R   R->W  Net    FlipRt   MedPre   MedPost  dMed",
        "-----------    ------  ------   -------  ------   ----   ----  ----   ------   ------   -------  ------",
    ]
    for token_class in CLASS_ORDER:
        summary = class_summaries[token_class]
        lines.append(
            f"{CLASS_LABELS[token_class]:<11}  "
            f"{summary['n_positions']:<6d}  "
            f"{_format_ratio(summary['pre_accuracy']):>6}   "
            f"{_format_ratio(summary['post_accuracy']):>7}  "
            f"{_format_ratio(summary['delta_accuracy']):>6}   "
            f"{summary['w2r']:<4d}   "
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
            lines.append("Token         Pos   W->R  R->W  Net  PreAcc  PostAcc  FlipRt")
            lines.append("-----------   ---   ----  ----  ---  ------  -------  ------")
            for row in rows:
                token = repr(row["token"])
                if len(token) > 11:
                    token = token[:10] + "…"
                lines.append(
                    f"{token:<11}   "
                    f"{row['positions']:<3d}   "
                    f"{row['w2r']:<4d}  "
                    f"{row['r2w']:<4d}  "
                    f"{row['net_corrected']:<3d}  "
                    f"{_format_ratio(row['pre_accuracy']):>6}  "
                    f"{_format_ratio(row['post_accuracy']):>7}  "
                    f"{_format_ratio(row['flip_ratio']):>6}"
                )
        lines.extend(["```", ""])

    for token_class in CLASS_ORDER:
        _token_block(
            f"{CLASS_LABELS[token_class]}: top improvements",
            token_summary["by_class"][token_class]["top_improvements"],
        )
        _token_block(
            f"{CLASS_LABELS[token_class]}: top regressions",
            token_summary["by_class"][token_class]["top_regressions"],
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.positions_jsonl:
        if args.source_token_stats_csv or args.post_token_stats_csv:
            raise ValueError("Use either --positions-jsonl or the source/post token_stats CSV pair, not both.")
        input_positions_path = _resolve_path(args.positions_jsonl)
        tokenizer_ref = args.tokenizer_id
        run_root = args.run_root or infer_run_root_from_path(input_positions_path.parent) or input_positions_path.parent
        candidate_model_ref = Path(run_root).expanduser().resolve() / "final_model"
        model_ref = candidate_model_ref if candidate_model_ref.exists() or args.run_root else input_positions_path.parent
        output_dir, paths = resolve_suite_output_dir(
            suite_id="token_class_flip_audit",
            output_dir=args.output_dir,
            model_ref=model_ref,
            run_root=run_root,
        )
        source_ref = str(input_positions_path)
        post_ref = str(input_positions_path)
    else:
        if not args.source_token_stats_csv or not args.post_token_stats_csv or not args.tokenizer_id:
            raise ValueError(
                "CSV mode requires --source-token-stats-csv, --post-token-stats-csv, and --tokenizer-id."
            )
        if AutoTokenizer is None:
            raise ImportError("transformers is required for CSV mode because target tokens must be decoded.")
        source_csv = _resolve_path(args.source_token_stats_csv)
        post_csv = _resolve_path(args.post_token_stats_csv)
        tokenizer_ref = args.tokenizer_id
        run_root = args.run_root or infer_run_root_from_path(post_csv.parent.parent) or post_csv.parent.parent
        model_ref = post_csv.parent.parent
        output_dir, paths = resolve_suite_output_dir(
            suite_id="token_class_flip_audit",
            output_dir=args.output_dir,
            model_ref=model_ref,
            run_root=run_root,
        )
        source_ref = str(source_csv)
        post_ref = str(post_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    merged_rows: list[dict[str, Any]] = []
    by_class: dict[str, list[dict[str, Any]]] = {token_class: [] for token_class in CLASS_ORDER}
    if args.positions_jsonl:
        position_rows = load_rows_from_positions_jsonl(input_positions_path)
        for raw in position_rows:
            token_text = raw["target_token"]
            token_class = classify_token(token_text)
            row = {
                "seq_id": raw["seq_id"],
                "position": raw["position"],
                "target_id": raw["target_id"],
                "target_token": token_text,
                "token_class": token_class,
                "pre_pred_id": raw["pre_pred_id"],
                "pre_margin": raw["pre_margin"],
                "pre_correct": raw["pre_correct"],
                "post_pred_id": raw["post_pred_id"],
                "post_margin": raw["post_margin"],
                "post_correct": raw["post_correct"],
            }
            merged_rows.append(row)
            by_class[token_class].append(row)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, trust_remote_code=args.trust_remote_code)
        pre_rows = load_rows_from_token_stats(source_csv)
        post_rows = load_rows_from_token_stats(post_csv)
        if len(pre_rows) != len(post_rows):
            raise ValueError(f"pre/post position count mismatch: {len(pre_rows)} vs {len(post_rows)}")
        for pre, post in zip(pre_rows, post_rows):
            if pre["seq_id"] != post["seq_id"] or pre["position"] != post["position"] or pre["target_id"] != post["target_id"]:
                raise ValueError("pre/post rows are not aligned")
            token_text = _decode_token(tokenizer, pre["target_id"])
            token_class = classify_token(token_text)
            row = {
                "seq_id": pre["seq_id"],
                "position": pre["position"],
                "target_id": pre["target_id"],
                "target_token": token_text,
                "token_class": token_class,
                "pre_pred_id": pre["pred_id"],
                "pre_margin": pre["margin"],
                "pre_correct": pre["correct"],
                "post_pred_id": post["pred_id"],
                "post_margin": post["margin"],
                "post_correct": post["correct"],
            }
            merged_rows.append(row)
            by_class[token_class].append(row)

    class_summaries = {token_class: summarize_bucket(by_class[token_class]) for token_class in CLASS_ORDER}
    overall = summarize_bucket(merged_rows)
    token_summary = summarize_tokens(merged_rows, top_k=args.top_k_tokens)

    payload = {
        "source_token_stats_csv": source_ref if not args.positions_jsonl else None,
        "post_token_stats_csv": post_ref if not args.positions_jsonl else None,
        "positions_jsonl": str(input_positions_path) if args.positions_jsonl else None,
        "tokenizer_ref": tokenizer_ref,
        "n_positions": len(merged_rows),
        "class_labels": CLASS_LABELS,
        "overall": overall,
        "by_class": class_summaries,
        "token_summary": token_summary,
        "elapsed_seconds": round(time.time() - start, 3),
    }

    summary_md = build_summary_markdown(
        source_ref=source_ref,
        post_ref=post_ref,
        tokenizer_ref=tokenizer_ref,
        n_positions=len(merged_rows),
        class_summaries=class_summaries,
        token_summary=token_summary,
    )

    payload_path = output_dir / "token_class_flip_audit.json"
    token_path = output_dir / "token_class_summary.json"
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
            "dataset_name": "wikitext",
            "dataset_config": "wikitext-103-raw-v1",
            "split": "validation",
            "n_positions": len(merged_rows),
            "token_classes": list(CLASS_ORDER),
        },
        command=" ".join(sys.argv),
        status="completed",
        source_artifacts=[source_ref],
        artifacts=[payload_path, token_path, positions_path, summary_path],
        metadata={
            "source_token_stats_csv": source_ref if not args.positions_jsonl else None,
            "post_token_stats_csv": post_ref if not args.positions_jsonl else None,
            "positions_jsonl": str(input_positions_path) if args.positions_jsonl else None,
            "top_k_tokens": args.top_k_tokens,
            "classification": "heuristic_decoded_token_classes",
        },
    )

    print(summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
