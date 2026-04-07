"""Cross-tab flip audit by frequency bucket and token class.

Consumes a standardized frequency_flip_audit positions.jsonl artifact and
builds the missing cross-tab:

    frequency bucket x token class

This is mainly useful for answering questions like:

- are the rare tokens that changed actually entity/content tokens?
- are low-frequency entity/content tokens net positive or negative?
- are observed rare-token effects too sparse or generic to interpret?
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

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

BUCKET_ORDER = ("freq_1", "freq_2_4", "freq_5_19", "freq_20_99", "freq_100_plus")
BUCKET_LABELS = {
    "freq_1": "1",
    "freq_2_4": "2-4",
    "freq_5_19": "5-19",
    "freq_20_99": "20-99",
    "freq_100_plus": "100+",
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
    parser = argparse.ArgumentParser(description="Frequency x token-class flip audit.")
    parser.add_argument("--positions-jsonl", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--top-k-tokens", type=int, default=20)
    return parser


def _resolve_path(raw: str | Path) -> Path:
    return Path(raw).expanduser().resolve()


def _safe_ratio(num: int, den: int) -> float | None:
    return (num / den) if den else None


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


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


def load_rows(jsonl_path: Path) -> list[dict[str, Any]]:
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
                    "corpus_frequency": int(raw["corpus_frequency"]),
                    "bucket": str(raw["bucket"]),
                    "token_class": classify_token(str(raw["target_token"])),
                    "pre_margin": float(raw["pre_margin"]),
                    "post_margin": float(raw["post_margin"]),
                    "pre_correct": bool(raw["pre_correct"]),
                    "post_correct": bool(raw["post_correct"]),
                }
            )
    return rows


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
        "unique_tokens": len({int(row["target_id"]) for row in rows}),
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


def summarize_tokens(rows: list[dict[str, Any]], *, top_k: int) -> dict[str, list[dict[str, Any]]]:
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
                "positions": positions,
                "corpus_frequency": token_rows_raw[0]["corpus_frequency"],
                "pre_accuracy": _safe_ratio(pre_correct, positions),
                "post_accuracy": _safe_ratio(post_correct, positions),
                "delta_accuracy": (_safe_ratio(post_correct, positions) - _safe_ratio(pre_correct, positions)) if positions else None,
                "w2r": w2r,
                "r2w": r2w,
                "net_corrected": w2r - r2w,
                "flip_ratio": _safe_ratio(w2r, r2w) if r2w else (float("inf") if w2r else None),
            }
        )
    return {
        "top_improvements": sorted(
            [row for row in token_rows if row["w2r"] > 0],
            key=lambda row: (-row["w2r"], row["r2w"], -row["positions"], row["token"]),
        )[:top_k],
        "top_regressions": sorted(
            [row for row in token_rows if row["r2w"] > 0],
            key=lambda row: (-row["r2w"], row["w2r"], -row["positions"], row["token"]),
        )[:top_k],
    }


def build_summary_markdown(
    *,
    positions_ref: str,
    n_positions: int,
    cell_summaries: dict[str, dict[str, dict[str, Any]]],
    rare_focus: dict[str, Any],
) -> str:
    lines = [
        "## Frequency X Token-Class Flip Audit",
        "",
        f"- positions reference: `{positions_ref}`",
        f"- audited positions: `{n_positions}`",
        "- classes are heuristic and based on decoded target-token strings",
        "",
    ]

    for bucket in BUCKET_ORDER:
        lines.extend(
            [
                f"### Frequency {BUCKET_LABELS[bucket]}",
                "",
                "```text",
                "Class          NPos    UTok   PreAcc   PostAcc  dAcc     W->R   R->W  Net    FlipRt",
                "-----------    ------  -----  ------   -------  ------   ----   ----  ----   ------",
            ]
        )
        for token_class in CLASS_ORDER:
            summary = cell_summaries[bucket][token_class]
            lines.append(
                f"{CLASS_LABELS[token_class]:<11}  "
                f"{summary['n_positions']:<6d}  "
                f"{summary['unique_tokens']:<5d}  "
                f"{_format_ratio(summary['pre_accuracy']):>6}   "
                f"{_format_ratio(summary['post_accuracy']):>7}  "
                f"{_format_ratio(summary['delta_accuracy']):>6}   "
                f"{summary['w2r']:<4d}   "
                f"{summary['r2w']:<4d}  "
                f"{summary['net_corrected']:<4d}  "
                f"{_format_ratio(summary['flip_ratio']):>6}"
            )
        lines.extend(["```", ""])

    def _token_block(title: str, rows: list[dict[str, Any]]) -> None:
        lines.extend([f"### {title}", "", "```text"])
        if not rows:
            lines.append("none")
        else:
            lines.append("Token         Freq  Pos  W->R  R->W  Net  PreAcc  PostAcc")
            lines.append("-----------   ----  ---  ----  ----  ---  ------  -------")
            for row in rows:
                token = repr(row["token"])
                if len(token) > 11:
                    token = token[:10] + "…"
                lines.append(
                    f"{token:<11}   {row['corpus_frequency']:<4d}  {row['positions']:<3d}  "
                    f"{row['w2r']:<4d}  {row['r2w']:<4d}  {row['net_corrected']:<3d}  "
                    f"{_format_ratio(row['pre_accuracy']):>6}  {_format_ratio(row['post_accuracy']):>7}"
                )
        lines.extend(["```", ""])

    for focus_key in ("rare_entity_like", "rare_content_word"):
        focus = rare_focus[focus_key]
        lines.extend(
            [
                f"### {focus['label']}",
                "",
                "```text",
                "BucketUnion     NPos   UTok   dAcc     W->R   R->W  Net    FlipRt",
                "------------    ----   ----   ------   ----   ----  ----   ------",
                f"{focus['label']:<12}  {focus['summary']['n_positions']:<4d}   "
                f"{focus['summary']['unique_tokens']:<4d}   "
                f"{_format_ratio(focus['summary']['delta_accuracy']):>6}   "
                f"{focus['summary']['w2r']:<4d}   "
                f"{focus['summary']['r2w']:<4d}  "
                f"{focus['summary']['net_corrected']:<4d}  "
                f"{_format_ratio(focus['summary']['flip_ratio']):>6}",
                "```",
                "",
            ]
        )
        _token_block(f"{focus['label']}: top improvements", focus["tokens"]["top_improvements"])
        _token_block(f"{focus['label']}: top regressions", focus["tokens"]["top_regressions"])

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    positions_path = _resolve_path(args.positions_jsonl)
    run_root = args.run_root or infer_run_root_from_path(positions_path.parent) or positions_path.parent
    candidate_model_ref = Path(run_root).expanduser().resolve() / "final_model"
    model_ref = candidate_model_ref if candidate_model_ref.exists() or args.run_root else positions_path.parent
    output_dir, paths = resolve_suite_output_dir(
        suite_id="frequency_class_flip_audit",
        output_dir=args.output_dir,
        model_ref=model_ref,
        run_root=run_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    rows = load_rows(positions_path)
    by_bucket_class: dict[str, dict[str, list[dict[str, Any]]]] = {
        bucket: {token_class: [] for token_class in CLASS_ORDER}
        for bucket in BUCKET_ORDER
    }
    for row in rows:
        by_bucket_class[row["bucket"]][row["token_class"]].append(row)

    cell_summaries = {
        bucket: {
            token_class: summarize_bucket(by_bucket_class[bucket][token_class])
            for token_class in CLASS_ORDER
        }
        for bucket in BUCKET_ORDER
    }

    rare_entity_rows = (
        by_bucket_class["freq_1"]["entity_like"]
        + by_bucket_class["freq_2_4"]["entity_like"]
    )
    rare_content_rows = (
        by_bucket_class["freq_1"]["content_word"]
        + by_bucket_class["freq_2_4"]["content_word"]
    )
    rare_focus = {
        "rare_entity_like": {
            "label": "Rare EntityLike",
            "summary": summarize_bucket(rare_entity_rows),
            "tokens": summarize_tokens(rare_entity_rows, top_k=args.top_k_tokens),
        },
        "rare_content_word": {
            "label": "Rare Content",
            "summary": summarize_bucket(rare_content_rows),
            "tokens": summarize_tokens(rare_content_rows, top_k=args.top_k_tokens),
        },
    }

    payload = {
        "positions_jsonl": str(positions_path),
        "n_positions": len(rows),
        "bucket_labels": BUCKET_LABELS,
        "class_labels": CLASS_LABELS,
        "by_bucket_and_class": cell_summaries,
        "rare_focus": rare_focus,
        "elapsed_seconds": round(time.time() - start, 3),
    }

    summary_md = build_summary_markdown(
        positions_ref=str(positions_path),
        n_positions=len(rows),
        cell_summaries=cell_summaries,
        rare_focus=rare_focus,
    )

    payload_path = output_dir / "frequency_class_flip_audit.json"
    rare_path = output_dir / "rare_focus_summary.json"
    summary_path = output_dir / "summary.md"
    atomic_write_json(payload_path, payload)
    atomic_write_json(rare_path, rare_focus)
    summary_path.write_text(summary_md, encoding="utf-8")

    finalize_eval_artifacts(
        paths=paths,
        model_ref=model_ref,
        tokenizer_ref=None,
        checkpoint_ref=model_ref,
        dataset={
            "dataset_name": "wikitext",
            "dataset_config": "wikitext-103-raw-v1",
            "split": "validation",
            "n_positions": len(rows),
            "frequency_buckets": list(BUCKET_ORDER),
            "token_classes": list(CLASS_ORDER),
        },
        command=" ".join(sys.argv),
        status="completed",
        source_artifacts=[str(positions_path)],
        artifacts=[payload_path, rare_path, summary_path],
        metadata={
            "positions_jsonl": str(positions_path),
            "top_k_tokens": args.top_k_tokens,
            "classification": "heuristic_decoded_token_classes",
            "cross_tab": "frequency_bucket_x_token_class",
        },
    )

    print(summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
