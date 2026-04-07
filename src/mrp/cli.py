from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mrp.audit import analyze_margins
from mrp.curvature import analyze_curvature
from mrp.eval_harness import run_lm_eval
from mrp.extract import extract_token_stats
from mrp.intrinsic_dimension import analyze_intrinsic_dimension
from mrp.model_inspection import inspect_model
from mrp.phase1 import run_phase1
from mrp.training import train_mrp
from mrp.utils import ensure_dir, write_json


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MRP experiment utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-model",
        help="Inspect model architecture traits relevant to the experiment.",
    )
    inspect_parser.add_argument("--model-id", required=True)
    inspect_parser.add_argument("--trust-remote-code", action="store_true")
    inspect_parser.add_argument(
        "--config-only",
        action="store_true",
        help="Skip loading full model weights.",
    )
    inspect_parser.add_argument("--output")

    extract_parser = subparsers.add_parser(
        "extract-token-stats",
        help="Extract token statistics and sampled hidden states.",
    )
    extract_parser.add_argument("--model-id", required=True)
    extract_parser.add_argument("--dataset-name", required=True)
    extract_parser.add_argument("--dataset-config")
    extract_parser.add_argument("--split", default="validation")
    extract_parser.add_argument("--text-column")
    extract_parser.add_argument("--output-dir", required=True)
    extract_parser.add_argument("--max-sequences", type=int, default=128)
    extract_parser.add_argument("--max-length", type=int, default=512)
    extract_parser.add_argument("--reservoir-size", type=int, default=1800)
    extract_parser.add_argument("--top-k", type=int, default=10)
    extract_parser.add_argument("--device", default="auto")
    extract_parser.add_argument(
        "--hidden-state-dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
    )
    extract_parser.add_argument("--seed", type=int, default=0)
    extract_parser.add_argument("--trust-remote-code", action="store_true")

    analyze_parser = subparsers.add_parser(
        "analyze-margins",
        help="Compute expressibility-gap and margin-distribution metrics.",
    )
    analyze_parser.add_argument("--token-stats", required=True)
    analyze_parser.add_argument("--output", required=True)
    analyze_parser.add_argument("--curve-output")
    analyze_parser.add_argument("--epsilon-min", type=float)
    analyze_parser.add_argument("--epsilon-max", type=float)
    analyze_parser.add_argument("--num-points", type=int, default=32)

    curvature_parser = subparsers.add_parser(
        "analyze-curvature",
        help="Compute local-PCA curvature and tangent-plane-rotation proxies.",
    )
    curvature_parser.add_argument("--manifest", required=True)
    curvature_parser.add_argument("--output", required=True)
    curvature_parser.add_argument("--profile-output")
    curvature_parser.add_argument("--neighbor-count", type=int, default=16)
    curvature_parser.add_argument("--variance-threshold", type=float, default=0.95)
    curvature_parser.add_argument("--plane-neighbors", type=int, default=4)
    curvature_parser.add_argument("--max-points", type=int)
    curvature_parser.add_argument("--seed", type=int, default=0)

    dimension_parser = subparsers.add_parser(
        "analyze-intrinsic-dimension",
        help="Compute layerwise intrinsic-dimension estimates with TWO-NN and MLE.",
    )
    dimension_parser.add_argument("--manifest", required=True)
    dimension_parser.add_argument("--output", required=True)
    dimension_parser.add_argument("--profile-output")
    dimension_parser.add_argument("--max-points", type=int)
    dimension_parser.add_argument("--mle-k1", type=int, default=5)
    dimension_parser.add_argument("--mle-k2", type=int, default=12)
    dimension_parser.add_argument("--seed", type=int, default=0)

    eval_parser = subparsers.add_parser(
        "run-lm-eval",
        help="Run lm-evaluation-harness with the wrapper-aware model loader.",
    )
    eval_parser.add_argument("--model-id", required=True)
    eval_parser.add_argument("--tasks", required=True)
    eval_parser.add_argument("--output", required=True)
    eval_parser.add_argument("--batch-size", default="1")
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--dtype", default="auto")
    eval_parser.add_argument("--limit", type=float)
    eval_parser.add_argument("--num-fewshot", type=int)
    eval_parser.add_argument("--gen-kwargs")
    eval_parser.add_argument("--bootstrap-iters", type=int, default=0)
    eval_parser.add_argument("--log-samples", action="store_true")
    eval_parser.add_argument("--verbosity", default="INFO")
    eval_parser.add_argument("--trust-remote-code", action="store_true")

    train_parser = subparsers.add_parser(
        "train-mrp",
        help="Run MRP continued pretraining with a wrapper-aware Trainer path.",
    )
    train_parser.add_argument("--model-id", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--dataset-name", required=True)
    train_parser.add_argument("--dataset-config")
    train_parser.add_argument("--train-split", default="train")
    train_parser.add_argument("--eval-split")
    train_parser.add_argument("--text-column")
    train_parser.add_argument("--block-size", type=int, default=512)
    train_parser.add_argument("--max-train-samples", type=int)
    train_parser.add_argument("--max-eval-samples", type=int)
    train_parser.add_argument("--max-steps", type=int, default=100)
    train_parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    train_parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_parser.add_argument("--learning-rate", type=float, default=1e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.05)
    train_parser.add_argument("--alpha-weight", type=float, default=0.01)
    train_parser.add_argument("--ce-weight", type=float, default=1.0,
                              help="Weight for cross-entropy loss. Set to 0.0 to run pure MRP without CE.")
    train_parser.add_argument("--mrp-top-k", type=int, default=5)
    train_parser.add_argument(
        "--trainable-scope",
        default="text",
        choices=["text", "last_block", "last_n_blocks", "final_norm", "embeddings"],
    )
    train_parser.add_argument(
        "--trainable-last-n",
        type=int,
        default=1,
        help="Number of trailing transformer blocks to unfreeze when using --trainable-scope last_n_blocks.",
    )
    train_parser.add_argument("--dtype", default="auto")
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--logging-steps", type=int, default=1)
    train_parser.add_argument("--save-strategy", default="no")
    train_parser.add_argument("--save-steps", type=int, default=500)
    train_parser.add_argument("--save-model", action="store_true")
    train_parser.add_argument("--gradient-checkpointing", action="store_true")
    train_parser.add_argument("--dataloader-num-workers", type=int, default=0)
    train_parser.add_argument(
        "--mrp-mode",
        default="final",
        choices=["final", "fisher", "margin_gated", "depth", "combined", "margin_max", "correct_margin", "entropy"],
        help="MRP loss variant: final (cosine), fisher (Fisher metric), "
             "margin_gated (threshold-filtered), depth (mid-layer), "
             "combined (depth + margin_gated), margin_max (direct margin maximization), "
             "entropy (entropy penalty, geometry-free baseline).",
    )
    train_parser.add_argument("--mrp-margin-threshold", type=float, default=0.5)
    train_parser.add_argument("--mrp-target-layers", default="24,26,28")
    train_parser.add_argument("--spot-check-interval", type=int, default=100)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--trust-remote-code", action="store_true")
    train_parser.add_argument("--streaming", action="store_true",
                              help="Use streaming dataset (v2 trainer).")
    train_parser.add_argument("--s3-bucket", default=None,
                              help="S3 bucket for checkpoint backup (v2 trainer).")
    train_parser.add_argument("--s3-prefix", default=None,
                              help="S3 prefix/path for this run (v2 trainer).")
    train_parser.add_argument("--lr-schedule", default="linear",
                              choices=["linear", "cosine"],
                              help="Learning rate schedule (v2 trainer).")
    train_parser.add_argument("--save-total-limit", type=int, default=5,
                              help="Max checkpoints to keep (v2 trainer).")

    unified_eval_parser = subparsers.add_parser(
        "eval",
        help="Unified model evaluation with tiered metrics.",
    )
    unified_eval_parser.add_argument("--model-path", required=True,
                                     help="Checkpoint dir, model.pt, or HF model ID.")
    unified_eval_parser.add_argument("--output", required=True,
                                     help="Path to write the JSON results.")
    unified_eval_parser.add_argument("--device", default="cpu")
    unified_eval_parser.add_argument("--tokenizer-id", default=None,
                                     help="Override tokenizer (default: infer from model).")
    unified_eval_parser.add_argument("--trust-remote-code", action="store_true")
    unified_eval_parser.add_argument("--n-sequences", type=int, default=64)
    unified_eval_parser.add_argument("--max-length", type=int, default=512)
    # Tier flags
    unified_eval_parser.add_argument("--margins", action="store_true", default=True,
                                     help="Tier 1: accuracy, margins, entropy (default on).")
    unified_eval_parser.add_argument("--geometry", action="store_true",
                                     help="Tier 2: isotropy, intrinsic dim, layer accuracy.")
    unified_eval_parser.add_argument("--dynamics", action="store_true",
                                     help="Tier 3: velocity, neighborhoods (needs --prev-checkpoint).")
    unified_eval_parser.add_argument("--benchmarks", action="store_true",
                                     help="Tier 4: lm-eval harness.")
    unified_eval_parser.add_argument("--benchmark-tasks", default=None,
                                     help="Comma-separated benchmark tasks for tier 4.")
    unified_eval_parser.add_argument("--benchmark-limit", type=float, default=None,
                                     help="Limit samples for tier 4 benchmarks.")
    # Compare
    unified_eval_parser.add_argument("--baseline", default=None,
                                     help="Baseline model for flip analysis.")
    # Dynamics
    unified_eval_parser.add_argument("--prev-checkpoint", default=None,
                                     help="Previous checkpoint for velocity/neighborhoods.")

    scratch_parser = subparsers.add_parser(
        "train-scratch",
        help="Train a standard or factored transformer from scratch.",
    )
    scratch_parser.add_argument("--model-type", required=True,
                                choices=["standard", "factored", "looped"],
                                help="Architecture: standard (dense), factored (shared middle block), or looped (shared preprocessing + dedicated reasoning with per-position loop gating)")
    scratch_parser.add_argument("--output-dir", required=True)
    scratch_parser.add_argument("--hidden-size", type=int, default=2048,
                                help="Hidden dimension (2048=1.13B, 1024=350M)")
    scratch_parser.add_argument("--num-layers", type=int, default=24,
                                help="Number of layers (24=1.13B, 12=350M)")
    scratch_parser.add_argument("--tokenizer-id", default="HuggingFaceTB/SmolLM3-3B",
                                help="Tokenizer to use (default: SmolLM3 49K vocab)")
    scratch_parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu",
                                help="HuggingFace dataset (default: FineWeb-Edu)")
    scratch_parser.add_argument("--dataset-config", default="sample-10BT",
                                help="Dataset config/subset")
    scratch_parser.add_argument("--block-size", type=int, default=2048)
    scratch_parser.add_argument("--max-steps", type=int, default=50000)
    scratch_parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    scratch_parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    scratch_parser.add_argument("--learning-rate", type=float, default=3e-4)
    scratch_parser.add_argument("--weight-decay", type=float, default=0.1)
    scratch_parser.add_argument("--warmup-ratio", type=float, default=0.01)
    scratch_parser.add_argument("--optimizer", default="mano",
                                choices=["mano", "adamw"],
                                help="Optimizer: mano (manifold) or adamw")
    scratch_parser.add_argument("--save-steps", type=int, default=500,
                                help="Save checkpoint every N steps")
    scratch_parser.add_argument("--logging-steps", type=int, default=10)
    scratch_parser.add_argument("--gradient-checkpointing", action="store_true")
    scratch_parser.add_argument("--dataloader-num-workers", type=int, default=2)
    scratch_parser.add_argument("--seed", type=int, default=42)
    scratch_parser.add_argument("--s3-bucket", default=None,
                                help="S3 bucket for checkpoint backup (optional)")
    scratch_parser.add_argument("--s3-prefix", default=None,
                                help="S3 prefix/path for this run")
    scratch_parser.add_argument("--init-weights-from", default=None,
                                help="Directory to load initial model weights from (fresh optimizer/scheduler, unlike --resume-from-checkpoint). Used to extend a completed run with a new LR schedule.")
    scratch_parser.add_argument("--resume-from-checkpoint", default=None,
                                help="Path to checkpoint dir to resume from")
    scratch_parser.add_argument("--pgp", action="store_true",
                                help="Enable PGP (correct_margin) loss alongside CE")
    scratch_parser.add_argument("--pgp-alpha", type=float, default=0.05,
                                help="PGP loss weight (default: 0.05)")
    scratch_parser.add_argument("--pgp-threshold", type=float, default=1.0,
                                help="Margin threshold for PGP (default: 1.0)")
    scratch_parser.add_argument("--repulsion", action="store_true",
                                help="Enable boundary repulsion loss")
    scratch_parser.add_argument("--repulsion-alpha", type=float, default=0.01,
                                help="Repulsion loss weight (default: 0.01)")
    scratch_parser.add_argument("--gravity", action="store_true",
                                help="Enable semantic gravity loss")
    scratch_parser.add_argument("--gravity-alpha", type=float, default=0.01,
                                help="Gravity loss weight (default: 0.01)")
    scratch_parser.add_argument("--covariance", action="store_true",
                                help="Enable covariance regularization loss")
    scratch_parser.add_argument("--covariance-alpha", type=float, default=0.01,
                                help="Covariance loss weight (default: 0.01)")
    scratch_parser.add_argument("--max-loops", type=int, default=1,
                                help="Number of outer loops for factored model (1=no looping)")
    scratch_parser.add_argument("--loop-decay", type=float, default=0.5,
                                help="Inhibitory decay factor for looping (default: 0.5)")
    scratch_parser.add_argument("--habituation-threshold", type=float, default=0.01,
                                help="Stop looping if delta norm < this (default: 0.01)")
    scratch_parser.add_argument("--normalize-shared-grad", action="store_true",
                                help="Scale shared block gradients by 1/num_shared_instances (factored only)")
    scratch_parser.add_argument("--shared-clip", type=float, default=0.1,
                                help="Per-component grad clip for shared block params (default 0.1)")
    scratch_parser.add_argument("--other-clip", type=float, default=1.0,
                                help="Per-component grad clip for non-shared params (default 1.0, raise to allow more signal)")
    scratch_parser.add_argument("--max-grad-norm", type=float, default=1.0,
                                help="HF Trainer global gradient norm clip (default 1.0). Raise to allow more gradient signal through when using built-in control mechanisms (per-position gating, supervisor). Ignored for factored+normalize-shared-grad (uses per-component clipping).")
    scratch_parser.add_argument("--supervisor-tune-grad-clip", action="store_true",
                                help="Let supervisor dynamically tune max_grad_norm based on observed pre-clip grad_norm. Raises ceiling when clipping throttles signal (median>2x), lowers when clipping rarely fires (median<0.3x). Bounds [0.1, 10.0], ±20%% steps, 100-step cooldown.")
    # Neuro-analog features
    scratch_parser.add_argument("--refractory-masking", action="store_true",
                                help="Enable per-position refractory freezing during loops")
    scratch_parser.add_argument("--refractory-exit-threshold", type=float, default=3.0,
                                help="Margin threshold for freezing (full margin mode)")
    scratch_parser.add_argument("--refractory-proxy", action="store_true", default=True,
                                help="Use delta-norm proxy instead of full margin (default: True)")
    scratch_parser.add_argument("--no-refractory-proxy", dest="refractory_proxy",
                                action="store_false")
    scratch_parser.add_argument("--refractory-proxy-threshold", type=float, default=0.05,
                                help="Per-position delta-norm threshold for proxy exit")
    scratch_parser.add_argument("--neuromod-gate", action="store_true",
                                help="Enable learned neuromodulatory loop gate")
    scratch_parser.add_argument("--neuromod-gate-hidden", type=int, default=64,
                                help="Hidden dim of gate MLP")
    scratch_parser.add_argument("--neuromod-gate-threshold", type=float, default=0.5,
                                help="Gate value below which to stop looping (eval only)")
    scratch_parser.add_argument("--neuromod-gate-reward-alpha", type=float, default=0.0,
                                help="Weight on gate-agreement reward: supervised signal pushing gate toward 1-loop0_accuracy. 0=off, 0.1 recommended")
    scratch_parser.add_argument("--neuromod-gate-loss-alpha", type=float, default=0.01,
                                help="Weight for gate anti-collapse loss")
    scratch_parser.add_argument("--multi-entry", action="store_true",
                                help="Enable multi-depth token embedding injection")
    scratch_parser.add_argument("--multi-entry-points", type=int, default=4,
                                help="Number of entry injection points")
    scratch_parser.add_argument("--cross-loop-cache", action="store_true",
                                help="Enable cross-loop state caching for attention")
    scratch_parser.add_argument("--cross-loop-cache-mode", default="final",
                                choices=["final", "all"],
                                help="Cache mode: final (end-of-loop) or all (every depth)")
    scratch_parser.add_argument("--deep-self-supervision", action="store_true",
                                help="Compute CE loss at each loop iteration (V-JEPA 2.1 inspired)")
    scratch_parser.add_argument("--deep-supervision-weights", type=float, nargs="+", default=None,
                                help="Per-loop weights for deep supervision CE. If unset, defaults vary: uniform [1,1,1] for looped, decay schedule for factored. Example: --deep-supervision-weights 0.25 0.5 1.0")
    scratch_parser.add_argument("--surprise-weighted-loss", action="store_true",
                                help="Weight per-token CE by prediction error (focal loss)")
    scratch_parser.add_argument("--surprise-gamma", type=float, default=2.0,
                                help="Focal loss exponent (higher = more focus on hard tokens)")
    scratch_parser.add_argument("--surprise-weighted-final-only", action="store_true",
                                help="Looped: apply focal weighting only on the final loop loss; earlier loops use plain CE")
    scratch_parser.add_argument("--interloop-pred-enabled", action="store_true",
                                help="Enable inter-loop consistency signal: auxiliary MSE loss training predictors to anticipate each loop's output from the previous loop")
    scratch_parser.add_argument("--interloop-pred-weight", type=float, default=0.1,
                                help="Weight on inter-loop consistency auxiliary loss (default 0.1)")
    scratch_parser.add_argument("--interloop-feedback", action="store_true",
                                help="Let predictor gradients flow back into earlier loop output (closer to true predictive coding, but can cause representation collapse)")
    scratch_parser.add_argument("--sparsity-enabled", action="store_true",
                                help="Enable top-k activation sparsity after each shared block iteration")
    scratch_parser.add_argument("--sparsity-k", type=float, default=0.1,
                                help="Fraction of hidden dims kept active (0.1 = 10%%). Biological range: 0.01-0.1")
    scratch_parser.add_argument("--sparsity-apply-mode", default="each_loop",
                                choices=["each_loop", "each_loop_shared", "final_only", "first_only"],
                                help="When to apply sparsity relative to loops. each_loop=recompute top-k each loop (default), each_loop_shared=compute once and reuse (cross-loop consistency), final_only=after all loops, first_only=after loop 0 only")
    scratch_parser.add_argument("--sparsity-ste", action="store_true",
                                help="Use straight-through estimator so dead dims can recover via gradient. Introduces gradient bias.")
    scratch_parser.add_argument("--use-loop-embedding", action="store_true",
                                help="Looped: add per-loop learned embedding to h at each loop start, giving layers a 'which loop' signal for specialization")
    scratch_parser.add_argument("--sparsity-homeostatic", action="store_true",
                                help="Enable homeostatic firing rate regulation: boost underused dims, dampen overused dims before top-k selection. Produces uniform firing rates → diverse sparse codes → higher effective dimension.")
    scratch_parser.add_argument("--homeostatic-tolerance", type=float, default=1.0,
                                help="Tolerance band for homeostasis. 1.0=regulate all dims toward target (strict). 3.0=only regulate dims firing >3x or <1/3x target (permissive, allows natural concentration).")
    scratch_parser.add_argument("--homeostatic-alpha", type=float, default=1.0,
                                help="Strength of homeostatic scaling. 0.0=off, 1.0=full banded regulation.")
    scratch_parser.add_argument("--adaptive-homeostasis", action="store_true",
                                help="Looped: adapt homeostatic alpha/tolerance during training using collapse-risk and loop-specialization signals")
    scratch_parser.add_argument("--adaptive-homeostasis-warmup-steps", type=int, default=100,
                                help="Looped: steps before adaptive homeostasis can start adjusting (default 100)")
    scratch_parser.add_argument("--adaptive-homeostasis-interval", type=int, default=100,
                                help="Looped: minimum steps between adaptive homeostasis adjustments (default 100)")
    scratch_parser.add_argument("--homeostatic-alpha-min", type=float, default=0.25,
                                help="Looped: lower bound for adaptive homeostasis alpha (default 0.25)")
    scratch_parser.add_argument("--homeostatic-alpha-max", type=float, default=1.0,
                                help="Looped: upper bound for adaptive homeostasis alpha (default 1.0)")
    scratch_parser.add_argument("--homeostatic-tolerance-min", type=float, default=1.75,
                                help="Looped: lower bound for adaptive homeostasis tolerance (default 1.75)")
    scratch_parser.add_argument("--homeostatic-tolerance-max", type=float, default=4.0,
                                help="Looped: upper bound for adaptive homeostasis tolerance (default 4.0)")
    scratch_parser.add_argument("--mtp-n-future", type=int, default=1,
                                help="Total prediction horizon (Meta 2024). 1=standard next-token only. 4=predict t+1..t+4 (main head covers t+1, 3 aux heads cover t+2..t+4)")
    scratch_parser.add_argument("--mtp-loss-weight", type=float, default=1.0,
                                help="Weight on each auxiliary MTP head's loss (default 1.0, equal weighting)")
    scratch_parser.add_argument("--mtp-horizon-decay", type=float, default=1.0,
                                help="Decay per horizon step. 1.0=uniform (Meta), 0.7=near predictions weighted more")
    scratch_parser.add_argument("--mtp-warmup-steps", type=int, default=0,
                                help="Optimizer steps over which to linearly ramp MTP loss from 0 to full weight (default 0 = no warmup)")
    scratch_parser.add_argument("--mtp-independent-heads", action="store_true",
                                help="Use independent Linear(H,V) per head (Meta 2024). Costs 131M params/head but no weight interference. Default: tied projections with transforms (cheaper)")
    # Looped reasoning architecture (model-type=looped)
    scratch_parser.add_argument("--num-shared-layers", type=int, default=4,
                                help="Looped: shared preprocessing layers (single block called N times)")
    scratch_parser.add_argument("--num-dedicated-layers", type=int, default=8,
                                help="Looped: dedicated reasoning layers (unique params each, looped with re-entry)")
    scratch_parser.add_argument("--loop0-confidence-threshold", type=float, default=0.9,
                                help="Looped: exit after loop 0 if max_prob > this (default 0.9)")
    scratch_parser.add_argument("--loop1-margin-threshold-init", type=float, default=0.15,
                                help="Looped: initial exit margin threshold after loop 1 (supervisor-tunable, default 0.15)")
    scratch_parser.add_argument("--episodic-buffer", action="store_true",
                                help="Looped: enable hippocampus-analog episodic buffer (key-value memory queried between loops)")
    scratch_parser.add_argument("--buffer-capacity", type=int, default=4096,
                                help="Looped: episodic buffer capacity (default 4096)")
    scratch_parser.add_argument("--buffer-top-k", type=int, default=32,
                                help="Looped: top-k attention width at buffer retrieval (default 32)")
    scratch_parser.add_argument("--buffer-novelty-threshold", type=float, default=0.9,
                                help="Looped: cosine similarity threshold; writes rejected if max_sim > this (default 0.9)")
    scratch_parser.add_argument("--buffer-improvement-threshold", type=float, default=0.5,
                                help="Looped: CE drop required to write to buffer (loop_0_ce - final_ce > threshold, default 0.5)")
    scratch_parser.add_argument("--surprise-head", action="store_true",
                                help="Looped: enable learned p_correct estimator head for gating (replaces max_prob and margin gates with unified surprise-based gating)")
    scratch_parser.add_argument("--surprise-exit-threshold", type=float, default=0.7,
                                help="Looped: exit loop if the per-gate surprise head predicts p_correct > this (default 0.7)")
    scratch_parser.add_argument("--surprise-head-loss-weight", type=float, default=0.1,
                                help="Looped: weight on surprise head BCE auxiliary loss (default 0.1)")
    scratch_parser.add_argument("--adaptive-loop-thresholds", action="store_true",
                                help="Looped: supervisor-tune active loop gate thresholds to steer exit fractions into target bands")
    scratch_parser.add_argument("--adaptive-loop-threshold-warmup-steps", type=int, default=200,
                                help="Looped: steps before adaptive gate-threshold tuning can start (default 200)")
    scratch_parser.add_argument("--adaptive-loop-threshold-interval", type=int, default=100,
                                help="Looped: minimum steps between gate-threshold adjustments (default 100)")
    scratch_parser.add_argument("--loop0-exit-target-low", type=float, default=0.02,
                                help="Looped: if loop0 exit frac falls below this, make gate 0 easier to exit (default 0.02)")
    scratch_parser.add_argument("--loop0-exit-target-high", type=float, default=0.15,
                                help="Looped: if loop0 exit frac rises above this, make gate 0 harder to exit (default 0.15)")
    scratch_parser.add_argument("--loop1-exit-target-low", type=float, default=0.05,
                                help="Looped: if loop1 exit frac falls below this, make gate 1 easier to exit (default 0.05)")
    scratch_parser.add_argument("--loop1-exit-target-high", type=float, default=0.30,
                                help="Looped: if loop1 exit frac rises above this, make gate 1 harder to exit (default 0.30)")
    scratch_parser.add_argument("--surprise-exit-threshold-min", type=float, default=0.35,
                                help="Looped: lower bound for adaptive surprise gate thresholds (default 0.35)")
    scratch_parser.add_argument("--surprise-exit-threshold-max", type=float, default=0.80,
                                help="Looped: upper bound for adaptive surprise gate thresholds (default 0.80)")
    scratch_parser.add_argument("--surprise-exit-threshold-step", type=float, default=0.05,
                                help="Looped: per-adjustment step for adaptive surprise thresholds (default 0.05)")
    scratch_parser.add_argument("--loop1-margin-threshold-min", type=float, default=0.02,
                                help="Looped: lower bound for adaptive legacy loop1 margin threshold (default 0.02)")
    scratch_parser.add_argument("--loop1-margin-threshold-max", type=float, default=0.30,
                                help="Looped: upper bound for adaptive legacy loop1 margin threshold (default 0.30)")
    scratch_parser.add_argument("--loop1-margin-threshold-step", type=float, default=0.02,
                                help="Looped: per-adjustment step for adaptive legacy loop1 margin threshold (default 0.02)")
    scratch_parser.add_argument("--velocity-clip", action="store_true",
                                help="Use velocity-based grad clip: dampen only sudden spikes (grad_norm > floor AND > ratio * prev), clip to prev+(delta/2). Replaces --max-grad-norm when enabled.")
    scratch_parser.add_argument("--velocity-clip-floor", type=float, default=15.0,
                                help="Velocity clip only fires if grad_norm exceeds this (default 15.0)")
    scratch_parser.add_argument("--velocity-clip-ratio", type=float, default=2.0,
                                help="Velocity clip fires if grad_norm > ratio * prev_grad_norm (default 2.0)")
    scratch_parser.add_argument("--step-metrics-every-step", action="store_true",
                                help="Write `step_metrics.jsonl` every optimizer step using raw per-step metrics instead of `logging_steps` windows.")
    scratch_parser.add_argument("--supervisor-mode", default="off",
                                choices=["off", "shadow", "live"],
                                help="Language-supervisor mode. `off` just disables decisions; `shadow` logs proposals only; `live` applies them.")
    scratch_parser.add_argument("--supervisor-model", default="gpt-5-nano",
                                help="Small language model used as the supervisor brain.")
    scratch_parser.add_argument("--supervisor-api-base", default="https://api.openai.com/v1",
                                help="Chat-completions-compatible API base for the supervisor model.")
    scratch_parser.add_argument("--supervisor-api-key-env", default="OPENAI_API_KEY",
                                help="Environment variable containing the supervisor API key.")
    scratch_parser.add_argument("--supervisor-decision-interval", type=int, default=1,
                                help="Ask the supervisor for a decision every N optimizer steps (default 1).")
    scratch_parser.add_argument("--supervisor-decision-warmup-steps", type=int, default=0,
                                help="Delay supervisor decisions until this many optimizer steps have elapsed.")
    scratch_parser.add_argument("--supervisor-history-steps", type=int, default=64,
                                help="How many recent per-step metric rows to feed into the supervisor prompt.")
    scratch_parser.add_argument("--supervisor-transition-horizon", type=int, default=25,
                                help="How many steps later to score the outcome of a supervisor decision.")
    scratch_parser.add_argument("--supervisor-max-output-tokens", type=int, default=1200,
                                help="Max completion tokens for the supervisor model.")
    scratch_parser.add_argument("--supervisor-temperature", type=float, default=1.0,
                                help="Sampling temperature for the supervisor model.")
    scratch_parser.add_argument("--supervisor-objective-text", default=None,
                                help="Optional freeform objective text for the supervisor.")
    scratch_parser.add_argument("--supervisor-objective-file", default=None,
                                help="Path to a text file containing the supervisor objective.")
    scratch_parser.add_argument("--supervisor-journal-path", default=None,
                                help="Optional experiment journal/path to include in the supervisor context.")

    supervisor_parser = subparsers.add_parser(
        "supervisor",
        help="Build datasets and prompts for the language-model training supervisor.",
    )
    supervisor_subparsers = supervisor_parser.add_subparsers(
        dest="supervisor_command",
        required=True,
    )

    supervisor_sync_parser = supervisor_subparsers.add_parser(
        "sync-s3",
        help="Sync supervisor-relevant artifacts from an S3 prefix into a local cache.",
    )
    supervisor_sync_parser.add_argument("--s3-uri", required=True)
    supervisor_sync_parser.add_argument("--local-dir", required=True)

    supervisor_dataset_parser = supervisor_subparsers.add_parser(
        "build-dataset",
        help="Build a replay dataset from local runs and/or an S3 cache of runs.",
    )
    supervisor_dataset_parser.add_argument("--root", action="append", default=[],
                                           help="Local run root to scan. Can be provided multiple times.")
    supervisor_dataset_parser.add_argument("--s3-uri", default=None,
                                           help="Optional S3 prefix to sync before building the dataset.")
    supervisor_dataset_parser.add_argument("--cache-dir", default=".supervisor-cache",
                                           help="Where S3 artifacts should be synced locally.")
    supervisor_dataset_parser.add_argument("--output", required=True)
    supervisor_dataset_parser.add_argument("--lookback-steps", type=int, default=64)
    supervisor_dataset_parser.add_argument("--outcome-horizon", type=int, default=25)
    supervisor_dataset_parser.add_argument("--objective-text", default=None)
    supervisor_dataset_parser.add_argument("--journal-path", default=None)

    supervisor_prompt_parser = supervisor_subparsers.add_parser(
        "prompt",
        help="Preview the supervisor prompt for a run at a particular step.",
    )
    supervisor_prompt_parser.add_argument("--run-dir", required=True)
    supervisor_prompt_parser.add_argument("--step", type=int, default=None)
    supervisor_prompt_parser.add_argument("--history-steps", type=int, default=64)
    supervisor_prompt_parser.add_argument("--objective-text", default=None)
    supervisor_prompt_parser.add_argument("--journal-path", default=None)

    supervisor_sft_dataset_parser = supervisor_subparsers.add_parser(
        "build-sft-dataset",
        help="Build supervised fine-tuning examples for an open supervisor model.",
    )
    supervisor_sft_dataset_parser.add_argument("--root", action="append", default=[],
                                               help="Local run root to scan. Can be provided multiple times.")
    supervisor_sft_dataset_parser.add_argument("--s3-uri", default=None,
                                               help="Optional S3 prefix to sync before building the dataset.")
    supervisor_sft_dataset_parser.add_argument("--cache-dir", default=".supervisor-cache",
                                               help="Where S3 artifacts should be synced locally.")
    supervisor_sft_dataset_parser.add_argument("--output", required=True)
    supervisor_sft_dataset_parser.add_argument("--lookback-steps", type=int, default=64)
    supervisor_sft_dataset_parser.add_argument("--outcome-horizon", type=int, default=25)
    supervisor_sft_dataset_parser.add_argument("--objective-text", default=None)
    supervisor_sft_dataset_parser.add_argument("--journal-path", default=None)
    supervisor_sft_dataset_parser.add_argument("--include-noop-examples", action="store_true",
                                               help="Also emit hold/no-op supervision rows from windows with no intervention.")
    supervisor_sft_dataset_parser.add_argument("--noop-stride", type=int, default=32,
                                               help="Sample one no-op window every N logged steps (default 32).")
    supervisor_sft_dataset_parser.add_argument("--noop-min-reward", type=float, default=0.0,
                                               help="Minimum future reward for a no-op example to be included (default 0.0).")

    supervisor_train_parser = supervisor_subparsers.add_parser(
        "train-sft",
        help="Fine-tune an open model to act as the language supervisor.",
    )
    supervisor_train_parser.add_argument("--dataset", required=True)
    supervisor_train_parser.add_argument("--output-dir", required=True)
    supervisor_train_parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    supervisor_train_parser.add_argument("--max-length", type=int, default=3072)
    supervisor_train_parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    supervisor_train_parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    supervisor_train_parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    supervisor_train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    supervisor_train_parser.add_argument("--weight-decay", type=float, default=0.0)
    supervisor_train_parser.add_argument("--warmup-ratio", type=float, default=0.03)
    supervisor_train_parser.add_argument("--num-train-epochs", type=float, default=1.0)
    supervisor_train_parser.add_argument("--max-steps", type=int, default=-1)
    supervisor_train_parser.add_argument("--logging-steps", type=int, default=10)
    supervisor_train_parser.add_argument("--save-steps", type=int, default=100)
    supervisor_train_parser.add_argument("--eval-steps", type=int, default=100)
    supervisor_train_parser.add_argument("--eval-fraction", type=float, default=0.05)
    supervisor_train_parser.add_argument("--max-train-examples", type=int, default=None)
    supervisor_train_parser.add_argument("--max-eval-examples", type=int, default=None)
    supervisor_train_parser.add_argument("--seed", type=int, default=0)
    supervisor_train_parser.add_argument("--torch-dtype", default="bfloat16")
    supervisor_train_parser.add_argument("--trust-remote-code", action="store_true")
    supervisor_train_parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    supervisor_train_parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    supervisor_train_parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    supervisor_train_parser.add_argument("--lora-r", type=int, default=16)
    supervisor_train_parser.add_argument("--lora-alpha", type=int, default=32)
    supervisor_train_parser.add_argument("--lora-dropout", type=float, default=0.05)
    supervisor_train_parser.add_argument("--lora-target-modules", nargs="*", default=None)
    supervisor_train_parser.add_argument("--min-reward", type=float, default=None,
                                         help="Optional reward floor for examples included in training.")

    phase1_parser = subparsers.add_parser(
        "run-phase1",
        help="Run the implemented Phase 1 geometry pipeline with step-level resume.",
    )
    phase1_parser.add_argument("--model-id", required=True)
    phase1_parser.add_argument("--output-dir", required=True)
    phase1_parser.add_argument("--dataset-name", default="wikitext")
    phase1_parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    phase1_parser.add_argument("--split", default="validation")
    phase1_parser.add_argument("--text-column")
    phase1_parser.add_argument("--max-sequences", type=int)
    phase1_parser.add_argument("--max-length", type=int, default=512)
    phase1_parser.add_argument("--reservoir-size", type=int, default=1800)
    phase1_parser.add_argument("--top-k", type=int, default=10)
    phase1_parser.add_argument("--hidden-state-dtype", default="float32")
    phase1_parser.add_argument("--device", default="cpu")
    phase1_parser.add_argument("--seed", type=int, default=0)
    phase1_parser.add_argument("--curvature-neighbor-count", type=int, default=16)
    phase1_parser.add_argument("--curvature-variance-threshold", type=float, default=0.95)
    phase1_parser.add_argument("--curvature-plane-neighbors", type=int, default=4)
    phase1_parser.add_argument("--curvature-max-points", type=int)
    phase1_parser.add_argument("--mle-k1", type=int, default=5)
    phase1_parser.add_argument("--mle-k2", type=int, default=12)
    phase1_parser.add_argument("--intrinsic-max-points", type=int)
    phase1_parser.add_argument("--eval-tasks")
    phase1_parser.add_argument("--eval-limit", type=float)
    phase1_parser.add_argument("--eval-batch-size", default="1")
    phase1_parser.add_argument("--eval-device", default="cpu")
    phase1_parser.add_argument("--force", action="store_true")
    phase1_parser.add_argument("--trust-remote-code", action="store_true")

    qeval_parser = subparsers.add_parser(
        "quick-eval",
        help="Quick checkpoint eval: accuracy, margins, SVD, boundary diversity.",
    )
    qeval_parser.add_argument(
        "checkpoint",
        nargs="*",
        default=["."],
        help="Checkpoint dirs to evaluate. Defaults to the current directory.",
    )
    qeval_parser.add_argument("--n-sequences", type=int, default=50)
    qeval_parser.add_argument("--max-length", type=int, default=512)
    qeval_parser.add_argument("--device", default="cpu")
    qeval_parser.add_argument("--tokenizer-id", default="HuggingFaceTB/SmolLM3-3B")
    qeval_parser.add_argument("--json", dest="json_output", action="store_true",
                              help="Also write canonical eval artifacts under <run>/evals/quick_eval/")

    tracker_parser = subparsers.add_parser(
        "tracker",
        help="Run or import the local experiment tracker.",
    )
    tracker_subparsers = tracker_parser.add_subparsers(
        dest="tracker_command",
        required=True,
    )

    tracker_serve_parser = tracker_subparsers.add_parser(
        "serve",
        help="Serve the tracker API and local web app.",
    )
    tracker_serve_parser.add_argument("--host", default="127.0.0.1")
    tracker_serve_parser.add_argument("--port", type=int, default=8876)
    tracker_serve_parser.add_argument(
        "--scan-root",
        dest="scan_roots",
        action="append",
        help="Additional root to scan for runs. Can be passed multiple times.",
    )
    tracker_serve_parser.add_argument("--reload", action="store_true")

    tracker_import_parser = tracker_subparsers.add_parser(
        "import",
        help="Run a one-shot import of existing runs into the tracker.",
    )
    tracker_import_parser.add_argument(
        "--scan-root",
        dest="scan_roots",
        action="append",
        help="Root to scan for runs. Can be passed multiple times.",
    )

    tracker_reindex_parser = tracker_subparsers.add_parser(
        "reindex",
        help="Reset tracker tables and rebuild the local index from disk.",
    )
    tracker_reindex_parser.add_argument(
        "--scan-root",
        dest="scan_roots",
        action="append",
        help="Root to scan for runs. Can be passed multiple times.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "inspect-model":
        payload = inspect_model(
            args.model_id,
            trust_remote_code=args.trust_remote_code,
            load_weights=not args.config_only,
        )
        if args.output:
            target = Path(args.output)
            ensure_dir(target.parent)
            write_json(target, payload)
        _print_json(payload)
        return

    if args.command == "extract-token-stats":
        payload = extract_token_stats(
            model_id=args.model_id,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            max_sequences=args.max_sequences,
            max_length=args.max_length,
            reservoir_size=args.reservoir_size,
            top_k=args.top_k,
            trust_remote_code=args.trust_remote_code,
            text_column=args.text_column,
            device=args.device,
            hidden_state_dtype=args.hidden_state_dtype,
            seed=args.seed,
        )
        _print_json(payload)
        return

    if args.command == "analyze-margins":
        payload = analyze_margins(
            args.token_stats,
            output_path=args.output,
            curve_output_path=args.curve_output,
            epsilon_min=args.epsilon_min,
            epsilon_max=args.epsilon_max,
            num_points=args.num_points,
        )
        _print_json(payload)
        return

    if args.command == "analyze-curvature":
        payload = analyze_curvature(
            args.manifest,
            output_path=args.output,
            profile_output_path=args.profile_output,
            neighbor_count=args.neighbor_count,
            variance_threshold=args.variance_threshold,
            plane_neighbors=args.plane_neighbors,
            max_points=args.max_points,
            seed=args.seed,
        )
        _print_json(payload)
        return

    if args.command == "analyze-intrinsic-dimension":
        payload = analyze_intrinsic_dimension(
            args.manifest,
            output_path=args.output,
            profile_output_path=args.profile_output,
            max_points=args.max_points,
            mle_k1=args.mle_k1,
            mle_k2=args.mle_k2,
            seed=args.seed,
        )
        _print_json(payload)
        return

    if args.command == "run-lm-eval":
        payload = run_lm_eval(
            model_id=args.model_id,
            tasks=args.tasks,
            output_path=args.output,
            batch_size=args.batch_size,
            device=args.device,
            torch_dtype=args.dtype,
            limit=args.limit,
            num_fewshot=args.num_fewshot,
            gen_kwargs=args.gen_kwargs,
            bootstrap_iters=args.bootstrap_iters,
            log_samples=args.log_samples,
            trust_remote_code=args.trust_remote_code,
            verbosity=args.verbosity,
        )
        _print_json(payload)
        return

    if args.command == "train-mrp":
        use_v2 = getattr(args, "streaming", False) or getattr(args, "s3_bucket", None)
        if use_v2:
            from mrp.training_v2 import train_mrp_v2

            payload = train_mrp_v2(
                model_id=args.model_id,
                output_dir=args.output_dir,
                dataset_name=args.dataset_name,
                dataset_config=args.dataset_config,
                train_split=args.train_split,
                eval_split=args.eval_split,
                text_column=args.text_column,
                block_size=args.block_size,
                max_train_samples=args.max_train_samples,
                max_eval_samples=args.max_eval_samples,
                max_steps=args.max_steps,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                alpha_weight=args.alpha_weight,
                ce_weight=args.ce_weight,
                mrp_top_k=args.mrp_top_k,
                trainable_scope=args.trainable_scope,
                trainable_last_n=args.trainable_last_n,
                torch_dtype=args.dtype,
                device=args.device,
                logging_steps=args.logging_steps,
                save_strategy=args.save_strategy,
                save_steps=args.save_steps,
                save_model=args.save_model,
                seed=args.seed,
                trust_remote_code=args.trust_remote_code,
                gradient_checkpointing=args.gradient_checkpointing,
                dataloader_num_workers=args.dataloader_num_workers,
                mrp_mode=args.mrp_mode,
                mrp_margin_threshold=args.mrp_margin_threshold,
                mrp_target_layers=args.mrp_target_layers,
                spot_check_interval=args.spot_check_interval,
                streaming=args.streaming,
                lr_schedule=args.lr_schedule,
                s3_bucket=args.s3_bucket,
                s3_prefix=args.s3_prefix,
                save_total_limit=args.save_total_limit,
            )
        else:
            payload = train_mrp(
                model_id=args.model_id,
                output_dir=args.output_dir,
                dataset_name=args.dataset_name,
                dataset_config=args.dataset_config,
                train_split=args.train_split,
                eval_split=args.eval_split,
                text_column=args.text_column,
                block_size=args.block_size,
                max_train_samples=args.max_train_samples,
                max_eval_samples=args.max_eval_samples,
                max_steps=args.max_steps,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                alpha_weight=args.alpha_weight,
                ce_weight=args.ce_weight,
                mrp_top_k=args.mrp_top_k,
                trainable_scope=args.trainable_scope,
                trainable_last_n=args.trainable_last_n,
                torch_dtype=args.dtype,
                device=args.device,
                logging_steps=args.logging_steps,
                save_strategy=args.save_strategy,
                save_steps=args.save_steps,
                save_model=args.save_model,
                seed=args.seed,
                trust_remote_code=args.trust_remote_code,
                gradient_checkpointing=args.gradient_checkpointing,
                dataloader_num_workers=args.dataloader_num_workers,
                mrp_mode=args.mrp_mode,
                mrp_margin_threshold=args.mrp_margin_threshold,
                mrp_target_layers=args.mrp_target_layers,
                spot_check_interval=args.spot_check_interval,
            )
        _print_json(payload)
        return

    if args.command == "eval":
        from mrp.eval import run_eval as _run_unified_eval

        payload = _run_unified_eval(
            model_path=args.model_path,
            output=args.output,
            device=args.device,
            tokenizer_id=args.tokenizer_id,
            trust_remote_code=args.trust_remote_code,
            n_sequences=args.n_sequences,
            max_length=args.max_length,
            margins=args.margins,
            geometry=args.geometry,
            dynamics=args.dynamics,
            benchmarks=args.benchmarks,
            benchmark_tasks=args.benchmark_tasks,
            benchmark_limit=args.benchmark_limit,
            baseline=args.baseline,
            prev_checkpoint=args.prev_checkpoint,
        )
        _print_json(payload)
        return

    if args.command == "quick-eval":
        from mrp.quick_eval import quick_eval
        quick_eval(
            checkpoints=args.checkpoint,
            n_sequences=args.n_sequences,
            max_length=args.max_length,
            device=args.device,
            tokenizer_id=args.tokenizer_id,
            json_output=args.json_output,
        )
        return

    if args.command == "train-scratch":
        from mrp.training import train_from_scratch
        supervisor_objective_text = args.supervisor_objective_text
        if args.supervisor_objective_file:
            supervisor_objective_text = Path(args.supervisor_objective_file).read_text(
                encoding="utf-8"
            )
        payload = train_from_scratch(
            model_type=args.model_type,
            output_dir=args.output_dir,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            tokenizer_id=args.tokenizer_id,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            block_size=args.block_size,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            optimizer=args.optimizer,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            dataloader_num_workers=args.dataloader_num_workers,
            seed=args.seed,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            resume_from_checkpoint=args.resume_from_checkpoint,
            init_weights_from=args.init_weights_from,
            pgp=args.pgp,
            pgp_alpha=args.pgp_alpha,
            pgp_threshold=args.pgp_threshold,
            repulsion=args.repulsion,
            repulsion_alpha=args.repulsion_alpha,
            gravity=args.gravity,
            gravity_alpha=args.gravity_alpha,
            covariance=args.covariance,
            covariance_alpha=args.covariance_alpha,
            max_loops=args.max_loops,
            loop_decay=args.loop_decay,
            habituation_threshold=args.habituation_threshold,
            normalize_shared_grad=args.normalize_shared_grad,
            shared_clip=args.shared_clip,
            other_clip=args.other_clip,
            max_grad_norm=args.max_grad_norm,
            supervisor_tune_grad_clip=args.supervisor_tune_grad_clip,
            refractory_masking=args.refractory_masking,
            refractory_exit_threshold=args.refractory_exit_threshold,
            refractory_use_proxy=args.refractory_proxy,
            refractory_proxy_threshold=args.refractory_proxy_threshold,
            neuromodulatory_gate=args.neuromod_gate,
            neuromod_gate_hidden=args.neuromod_gate_hidden,
            neuromod_gate_threshold=args.neuromod_gate_threshold,
            neuromod_gate_loss_alpha=args.neuromod_gate_loss_alpha,
            neuromod_gate_reward_alpha=args.neuromod_gate_reward_alpha,
            multi_entry=args.multi_entry,
            multi_entry_points=args.multi_entry_points,
            cross_loop_cache=args.cross_loop_cache,
            cross_loop_cache_mode=args.cross_loop_cache_mode,
            deep_self_supervision=args.deep_self_supervision,
            deep_supervision_weights=args.deep_supervision_weights,
            surprise_weighted_loss=args.surprise_weighted_loss,
            surprise_gamma=args.surprise_gamma,
            surprise_weighted_final_only=args.surprise_weighted_final_only,
            interloop_pred_enabled=args.interloop_pred_enabled,
            interloop_pred_weight=args.interloop_pred_weight,
            interloop_feedback=args.interloop_feedback,
            sparsity_enabled=args.sparsity_enabled,
            sparsity_k=args.sparsity_k,
            sparsity_apply_mode=args.sparsity_apply_mode,
            sparsity_ste=args.sparsity_ste,
            use_loop_embedding=args.use_loop_embedding,
            sparsity_homeostatic=args.sparsity_homeostatic,
            homeostatic_tolerance=args.homeostatic_tolerance,
            homeostatic_alpha=args.homeostatic_alpha,
            adaptive_homeostasis=args.adaptive_homeostasis,
            adaptive_homeostasis_warmup_steps=args.adaptive_homeostasis_warmup_steps,
            adaptive_homeostasis_interval=args.adaptive_homeostasis_interval,
            homeostatic_alpha_min=args.homeostatic_alpha_min,
            homeostatic_alpha_max=args.homeostatic_alpha_max,
            homeostatic_tolerance_min=args.homeostatic_tolerance_min,
            homeostatic_tolerance_max=args.homeostatic_tolerance_max,
            mtp_n_future=args.mtp_n_future,
            mtp_loss_weight=args.mtp_loss_weight,
            mtp_horizon_decay=args.mtp_horizon_decay,
            mtp_warmup_steps=args.mtp_warmup_steps,
            mtp_independent_heads=args.mtp_independent_heads,
            num_shared_layers=args.num_shared_layers,
            num_dedicated_layers=args.num_dedicated_layers,
            loop0_confidence_threshold=args.loop0_confidence_threshold,
            loop1_margin_threshold_init=args.loop1_margin_threshold_init,
            episodic_buffer_enabled=args.episodic_buffer,
            buffer_capacity=args.buffer_capacity,
            buffer_top_k=args.buffer_top_k,
            buffer_novelty_threshold=args.buffer_novelty_threshold,
            buffer_improvement_threshold=args.buffer_improvement_threshold,
            surprise_head_enabled=args.surprise_head,
            surprise_exit_threshold=args.surprise_exit_threshold,
            surprise_head_loss_weight=args.surprise_head_loss_weight,
            adaptive_loop_thresholds=args.adaptive_loop_thresholds,
            adaptive_loop_threshold_warmup_steps=args.adaptive_loop_threshold_warmup_steps,
            adaptive_loop_threshold_interval=args.adaptive_loop_threshold_interval,
            loop0_exit_target_low=args.loop0_exit_target_low,
            loop0_exit_target_high=args.loop0_exit_target_high,
            loop1_exit_target_low=args.loop1_exit_target_low,
            loop1_exit_target_high=args.loop1_exit_target_high,
            surprise_exit_threshold_min=args.surprise_exit_threshold_min,
            surprise_exit_threshold_max=args.surprise_exit_threshold_max,
            surprise_exit_threshold_step=args.surprise_exit_threshold_step,
            loop1_margin_threshold_min=args.loop1_margin_threshold_min,
            loop1_margin_threshold_max=args.loop1_margin_threshold_max,
            loop1_margin_threshold_step=args.loop1_margin_threshold_step,
            velocity_clip=args.velocity_clip,
            velocity_clip_floor=args.velocity_clip_floor,
            velocity_clip_ratio=args.velocity_clip_ratio,
            step_metrics_every_step=args.step_metrics_every_step,
            supervisor_mode=args.supervisor_mode,
            supervisor_model=args.supervisor_model,
            supervisor_api_base=args.supervisor_api_base,
            supervisor_api_key_env=args.supervisor_api_key_env,
            supervisor_decision_interval=args.supervisor_decision_interval,
            supervisor_decision_warmup_steps=args.supervisor_decision_warmup_steps,
            supervisor_history_steps=args.supervisor_history_steps,
            supervisor_transition_horizon=args.supervisor_transition_horizon,
            supervisor_max_output_tokens=args.supervisor_max_output_tokens,
            supervisor_temperature=args.supervisor_temperature,
            supervisor_objective_text=supervisor_objective_text,
            supervisor_journal_path=args.supervisor_journal_path,
        )
        _print_json(payload)
        return

    if args.command == "supervisor":
        from mrp.supervisor_agent import run_supervisor_command

        payload = run_supervisor_command(args)
        _print_json(payload)
        return

    if args.command == "run-phase1":
        payload = run_phase1(
            model_id=args.model_id,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            text_column=args.text_column,
            max_sequences=args.max_sequences,
            max_length=args.max_length,
            reservoir_size=args.reservoir_size,
            top_k=args.top_k,
            hidden_state_dtype=args.hidden_state_dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            seed=args.seed,
            curvature_neighbor_count=args.curvature_neighbor_count,
            curvature_variance_threshold=args.curvature_variance_threshold,
            curvature_plane_neighbors=args.curvature_plane_neighbors,
            curvature_max_points=args.curvature_max_points,
            mle_k1=args.mle_k1,
            mle_k2=args.mle_k2,
            intrinsic_max_points=args.intrinsic_max_points,
            eval_tasks=args.eval_tasks,
            eval_limit=args.eval_limit,
            eval_batch_size=args.eval_batch_size,
            eval_device=args.eval_device,
            force=args.force,
        )
        _print_json(payload)
        return

    if args.command == "tracker":
        from mrp.tracker.cli import run_tracker_command

        run_tracker_command(args)
        return

    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
