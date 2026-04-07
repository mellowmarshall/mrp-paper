"""Analyze whether Fisher polishing widens the same boundary or rotates it.

This script compares a source model against a Fisher-polished model on the same
held-out evaluation sequences and writes a durable per-position artifact plus a
compact summary.

The core buckets are:

- same boundary pair: top1 and top2 unchanged
- runner-up rotation: top1 unchanged, top2 changed
- top1 changed: the decision surface itself changed

Usage:
    python scripts/analysis/fisher_boundary_rotation.py \
        --source-model results/bl264_ckpt50000 \
        --post-model results/polishing/fisher_precision_fp64/264m_baseline_50k_noprotect/fp32/final_model \
        --run-root results/polishing/fisher_precision_fp64/264m_baseline_50k_noprotect/fp32 \
        --device cpu
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Fisher boundary rotation.")
    parser.add_argument("--source-model", required=True, help="Original checkpoint or final_model path.")
    parser.add_argument("--post-model", required=True, help="Fisher-polished final_model path.")
    parser.add_argument("--output-dir", default=None, help="Optional explicit output directory.")
    parser.add_argument("--run-root", default=None, help="Optional explicit run root for canonical eval output.")
    parser.add_argument("--tokenizer-id", default=None, help="Optional tokenizer id or local tokenizer directory.")
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n-sequences", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def _resolve_model_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser().resolve()
    if path.is_dir() and not (path / "config.json").exists() and (path / "model.pt").exists():
        return (path / "model.pt").resolve()
    return path


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


def infer_tokenizer_ref(
    *,
    source_model: Path,
    post_model: Path,
    explicit: str | None = None,
) -> str:
    if explicit:
        return explicit
    if post_model.is_dir() and (
        (post_model / "tokenizer.json").exists()
        or (post_model / "tokenizer_config.json").exists()
    ):
        return str(post_model)
    if source_model.is_dir() and (
        (source_model / "tokenizer.json").exists()
        or (source_model / "tokenizer_config.json").exists()
    ):
        return str(source_model)
    for candidate in (_read_config_tokenizer_id(post_model), _read_config_tokenizer_id(source_model)):
        if candidate:
            return candidate
    return DEFAULT_TOKENIZER_ID


def _forward_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.device.type == "cpu":
        outputs = model(input_ids=input_ids)
    else:
        with torch.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids)
    return outputs.logits[0, :-1, :].float()


def collect_positions(
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
            logits = _forward_logits(model, ids).cpu()
            top2_vals, top2_ids = logits.topk(2, dim=-1)
            top1 = top2_ids[:, 0]
            top2 = top2_ids[:, 1]
            margins = top2_vals[:, 0] - top2_vals[:, 1]
            correct = top1.eq(labels)
            for pos in range(len(labels)):
                rows.append(
                    {
                        "seq_id": seq_idx,
                        "position": pos,
                        "target": int(labels[pos].item()),
                        "top1": int(top1[pos].item()),
                        "top2": int(top2[pos].item()),
                        "margin": float(margins[pos].item()),
                        "correct": bool(correct[pos].item()),
                    }
                )
            if (seq_idx + 1) % 10 == 0:
                print(f"  audited {seq_idx + 1}/{len(sequences)} sequences")
    return rows


def boundary_angle_deg(
    pre_top1: int,
    pre_top2: int,
    post_top1: int,
    post_top2: int,
    *,
    pre_lm_head: torch.Tensor,
    post_lm_head: torch.Tensor,
) -> float:
    pre_normal = (pre_lm_head[pre_top1] - pre_lm_head[pre_top2]).double()
    post_normal = (post_lm_head[post_top1] - post_lm_head[post_top2]).double()
    pre_norm = torch.linalg.vector_norm(pre_normal)
    post_norm = torch.linalg.vector_norm(post_normal)
    if float(pre_norm.item()) == 0.0 or float(post_norm.item()) == 0.0:
        return 0.0
    cosine = torch.clamp(
        torch.dot(pre_normal, post_normal) / (pre_norm * post_norm),
        min=-1.0,
        max=1.0,
    )
    return float(math.degrees(math.acos(float(cosine.item()))))


def classify_boundary_transition(pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    top1_changed = pre["top1"] != post["top1"]
    top2_changed = pre["top2"] != post["top2"]
    same_boundary_pair = (not top1_changed) and (not top2_changed)
    runner_up_rotation = (not top1_changed) and top2_changed
    promotes_pre_top2 = top1_changed and (post["top1"] == pre["top2"])
    swaps_top1_top2 = promotes_pre_top2 and (post["top2"] == pre["top1"])
    if (not pre["correct"]) and post["correct"]:
        correctness_transition = "w2r"
    elif pre["correct"] and (not post["correct"]):
        correctness_transition = "r2w"
    elif pre["correct"] and post["correct"]:
        correctness_transition = "w2w"
    else:
        correctness_transition = "r2r"
    return {
        "top1_changed": top1_changed,
        "top2_changed": top2_changed,
        "same_boundary_pair": same_boundary_pair,
        "runner_up_rotation": runner_up_rotation,
        "boundary_identity_changed": not same_boundary_pair,
        "promotes_pre_top2": promotes_pre_top2,
        "swaps_top1_top2": swaps_top1_top2,
        "correctness_transition": correctness_transition,
    }


def _float_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
        }
    t = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(t.numel()),
        "mean": float(t.mean().item()),
        "median": float(t.median().item()),
        "p25": float(t.quantile(0.25).item()),
        "p75": float(t.quantile(0.75).item()),
    }


def _bucket_summary(rows: list[dict[str, Any]], total: int) -> dict[str, Any]:
    angles = [float(row["boundary_angle_deg"]) for row in rows]
    margin_deltas = [float(row["margin_delta"]) for row in rows]
    widened = [row for row in rows if row["margin_delta"] > 0]
    return {
        "count": len(rows),
        "rate": (len(rows) / total) if total else 0.0,
        "angle_deg": _float_stats(angles),
        "margin_delta": _float_stats(margin_deltas),
        "margin_widened_rate": (len(widened) / len(rows)) if rows else None,
    }


def summarize_rotation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    categories = {
        "same_boundary_pair": [row for row in rows if row["same_boundary_pair"]],
        "runner_up_rotation": [row for row in rows if row["runner_up_rotation"]],
        "top1_changed": [row for row in rows if row["top1_changed"]],
    }
    transitions = {
        label: [row for row in rows if row["correctness_transition"] == label]
        for label in ("w2r", "r2w", "w2w", "r2r")
    }

    summary = {
        "n_positions": total,
        "top1_changed_count": len(categories["top1_changed"]),
        "top1_changed_rate": len(categories["top1_changed"]) / total if total else 0.0,
        "top2_changed_count": sum(1 for row in rows if row["top2_changed"]),
        "top2_changed_rate": sum(1 for row in rows if row["top2_changed"]) / total if total else 0.0,
        "promotes_pre_top2_count": sum(1 for row in rows if row["promotes_pre_top2"]),
        "promotes_pre_top2_rate": sum(1 for row in rows if row["promotes_pre_top2"]) / total if total else 0.0,
        "swaps_top1_top2_count": sum(1 for row in rows if row["swaps_top1_top2"]),
        "swaps_top1_top2_rate": sum(1 for row in rows if row["swaps_top1_top2"]) / total if total else 0.0,
        "same_boundary_pair": _bucket_summary(categories["same_boundary_pair"], total),
        "runner_up_rotation": _bucket_summary(categories["runner_up_rotation"], total),
        "top1_changed": _bucket_summary(categories["top1_changed"], total),
        "boundary_identity_changed_count": sum(1 for row in rows if row["boundary_identity_changed"]),
        "boundary_identity_changed_rate": sum(1 for row in rows if row["boundary_identity_changed"]) / total if total else 0.0,
        "runner_up_rotation_rate_among_boundary_changes": (
            len(categories["runner_up_rotation"]) / max(1, sum(1 for row in rows if row["boundary_identity_changed"]))
        ),
        "overall_angle_deg": _float_stats([float(row["boundary_angle_deg"]) for row in rows]),
        "overall_margin_delta": _float_stats([float(row["margin_delta"]) for row in rows]),
        "correctness_transitions": {},
    }

    for label, bucket_rows in transitions.items():
        summary["correctness_transitions"][label] = {
            "count": len(bucket_rows),
            "rate": len(bucket_rows) / total if total else 0.0,
            "same_boundary_pair": _bucket_summary(
                [row for row in bucket_rows if row["same_boundary_pair"]],
                len(bucket_rows),
            ),
            "runner_up_rotation": _bucket_summary(
                [row for row in bucket_rows if row["runner_up_rotation"]],
                len(bucket_rows),
            ),
            "top1_changed": _bucket_summary(
                [row for row in bucket_rows if row["top1_changed"]],
                len(bucket_rows),
            ),
            "overall_angle_deg": _float_stats([float(row["boundary_angle_deg"]) for row in bucket_rows]),
            "overall_margin_delta": _float_stats([float(row["margin_delta"]) for row in bucket_rows]),
            "promotes_pre_top2_count": sum(1 for row in bucket_rows if row["promotes_pre_top2"]),
            "promotes_pre_top2_rate": (
                sum(1 for row in bucket_rows if row["promotes_pre_top2"]) / len(bucket_rows)
                if bucket_rows
                else 0.0
            ),
            "swaps_top1_top2_count": sum(1 for row in bucket_rows if row["swaps_top1_top2"]),
            "swaps_top1_top2_rate": (
                sum(1 for row in bucket_rows if row["swaps_top1_top2"]) / len(bucket_rows)
                if bucket_rows
                else 0.0
            ),
        }
    return summary


def analyze_boundary_rotation(
    pre_positions: list[dict[str, Any]],
    post_positions: list[dict[str, Any]],
    *,
    pre_lm_head: torch.Tensor,
    post_lm_head: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pre_by_key = {(row["seq_id"], row["position"]): row for row in pre_positions}
    post_by_key = {(row["seq_id"], row["position"]): row for row in post_positions}
    merged_rows: list[dict[str, Any]] = []
    for key, pre in pre_by_key.items():
        post = post_by_key.get(key)
        if post is None:
            continue
        flags = classify_boundary_transition(pre, post)
        angle_deg = boundary_angle_deg(
            pre["top1"],
            pre["top2"],
            post["top1"],
            post["top2"],
            pre_lm_head=pre_lm_head,
            post_lm_head=post_lm_head,
        )
        merged_rows.append(
            {
                "seq_id": key[0],
                "position": key[1],
                "target": pre["target"],
                "pre_top1": pre["top1"],
                "pre_top2": pre["top2"],
                "post_top1": post["top1"],
                "post_top2": post["top2"],
                "pre_correct": pre["correct"],
                "post_correct": post["correct"],
                "pre_margin": pre["margin"],
                "post_margin": post["margin"],
                "margin_delta": post["margin"] - pre["margin"],
                "boundary_angle_deg": angle_deg,
                **flags,
            }
        )
    return merged_rows, summarize_rotation_rows(merged_rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_summary_markdown(
    path: Path,
    *,
    payload: dict[str, Any],
) -> None:
    summary = payload["summary"]
    lines = [
        "# Fisher Boundary Rotation Summary",
        "",
        f"- source model: `{payload['source_model']}`",
        f"- post model: `{payload['post_model']}`",
        f"- tokenizer ref: `{payload['tokenizer_ref']}`",
        f"- dataset: `{payload['dataset_name']}/{payload['dataset_config']}:{payload['split']}`",
        f"- positions: `{summary['n_positions']}`",
        f"- top1 changes that promote the old runner-up: `{summary['promotes_pre_top2_count']}` ({summary['promotes_pre_top2_rate']:.2%})",
        f"- exact top1/top2 swaps: `{summary['swaps_top1_top2_count']}` ({summary['swaps_top1_top2_rate']:.2%})",
        "",
        "```text",
        "Bucket              Count   Rate     MedianAngle  MedianΔMargin  WidenedRate",
        "------------------  ------  -------  -----------  -------------  -----------",
    ]
    for label, key in (
        ("Same boundary", "same_boundary_pair"),
        ("Runner-up rot.", "runner_up_rotation"),
        ("Top1 changed", "top1_changed"),
    ):
        bucket = summary[key]
        angle = bucket["angle_deg"]["median"]
        margin = bucket["margin_delta"]["median"]
        widened = bucket["margin_widened_rate"]
        lines.append(
            f"{label:<18}  {bucket['count']:>6}  {bucket['rate']*100:>6.2f}%  "
            f"{(angle if angle is not None else 0.0):>11.2f}  "
            f"{(margin if margin is not None else 0.0):>13.4f}  "
            f"{((widened or 0.0)*100):>10.2f}%"
        )
    lines.extend(
        [
            "```",
            "",
            "## Correctness Transitions",
            "",
            "```text",
            "Transition  Count   Rate     SamePair  RunnerUpRot  Top1Changed  MedianAngle",
            "----------  ------  -------  --------  -----------  -----------  -----------",
        ]
    )
    for label in ("w2r", "r2w", "w2w", "r2r"):
        bucket = summary["correctness_transitions"][label]
        lines.append(
            f"{label:<10}  {bucket['count']:>6}  {bucket['rate']*100:>6.2f}%  "
            f"{bucket['same_boundary_pair']['count']:>8}  "
            f"{bucket['runner_up_rotation']['count']:>11}  "
            f"{bucket['top1_changed']['count']:>11}  "
            f"{((bucket['overall_angle_deg']['median'] or 0.0)):>11.2f}"
        )
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()

    source_model = _resolve_model_path(args.source_model)
    post_model = _resolve_model_path(args.post_model)
    inferred_run_root = infer_run_root_from_path(post_model) or infer_run_root_from_path(source_model)
    out, canonical_paths = resolve_suite_output_dir(
        suite_id="boundary_rotation",
        output_dir=args.output_dir,
        model_ref=post_model,
        run_root=args.run_root or (str(inferred_run_root) if inferred_run_root else None),
    )
    out.mkdir(parents=True, exist_ok=True)

    tokenizer_ref = infer_tokenizer_ref(
        source_model=source_model,
        post_model=post_model,
        explicit=args.tokenizer_id,
    )

    print(f"Loading post model: {post_model}")
    post_loaded_model, tokenizer = load_model_flexible(
        post_model,
        device=args.device,
        tokenizer_id=tokenizer_ref,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Loading source model: {source_model}")
    source_loaded_model, _ = load_model_flexible(
        source_model,
        device=args.device,
        tokenizer_id=tokenizer_ref,
        trust_remote_code=args.trust_remote_code,
    )

    print("Loading evaluation sequences...")
    sequences = load_eval_sequences(
        tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        n_sequences=args.n_sequences,
        max_length=args.max_length,
    )
    print(f"Loaded {len(sequences)} sequences")

    start = time.time()
    print("Auditing source model...")
    pre_positions = collect_positions(source_loaded_model, sequences, device=args.device)
    print("Auditing post model...")
    post_positions = collect_positions(post_loaded_model, sequences, device=args.device)

    print("Computing boundary-rotation summaries...")
    merged_rows, summary = analyze_boundary_rotation(
        pre_positions,
        post_positions,
        pre_lm_head=source_loaded_model.lm_head.weight.detach().float().cpu(),
        post_lm_head=post_loaded_model.lm_head.weight.detach().float().cpu(),
    )
    elapsed = time.time() - start

    payload = {
        "suite_id": "boundary_rotation",
        "source_model": str(source_model),
        "post_model": str(post_model),
        "tokenizer_ref": tokenizer_ref,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "n_sequences": args.n_sequences,
        "max_length": args.max_length,
        "elapsed_s": round(elapsed, 3),
        "summary": summary,
    }
    atomic_write_json(out / "boundary_rotation.json", payload)
    _write_jsonl(out / "positions.jsonl", merged_rows)
    _write_summary_markdown(out / "summary.md", payload=payload)

    finalize_eval_artifacts(
        paths=canonical_paths,
        model_ref=post_model,
        tokenizer_ref=tokenizer_ref,
        checkpoint_ref=source_model,
        dataset={
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "n_sequences": args.n_sequences,
            "max_length": args.max_length,
        },
        command="python scripts/analysis/fisher_boundary_rotation.py",
        artifacts=[
            out / "boundary_rotation.json",
            out / "positions.jsonl",
            out / "summary.md",
        ],
        metadata={
            "source_model": str(source_model),
            "post_model": str(post_model),
        },
    )

    print("\nFISHER BOUNDARY ROTATION")
    print(f"  positions: {summary['n_positions']}")
    print(
        "  same boundary: {count} ({rate:.2%})".format(
            count=summary["same_boundary_pair"]["count"],
            rate=summary["same_boundary_pair"]["rate"],
        )
    )
    print(
        "  runner-up rotation: {count} ({rate:.2%})".format(
            count=summary["runner_up_rotation"]["count"],
            rate=summary["runner_up_rotation"]["rate"],
        )
    )
    print(
        "  top1 changed: {count} ({rate:.2%})".format(
            count=summary["top1_changed"]["count"],
            rate=summary["top1_changed"]["rate"],
        )
    )
    print(
        "  promote old runner-up: {count} ({rate:.2%})".format(
            count=summary["promotes_pre_top2_count"],
            rate=summary["promotes_pre_top2_rate"],
        )
    )
    print(
        "  w2r runner-up rotation: {count}/{total}".format(
            count=summary["correctness_transitions"]["w2r"]["runner_up_rotation"]["count"],
            total=summary["correctness_transitions"]["w2r"]["count"],
        )
    )
    print(f"  wrote: {out / 'boundary_rotation.json'}")


if __name__ == "__main__":
    main()
