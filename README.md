# Geometric Properties of the Voronoi Tessellation in Latent Semantic Manifolds of Large Language Models

**Marshall Brett, MARS Labs**

This repository contains the code, data, and analysis scripts for reproducing the results in the paper.

## Paper

The paper is in `paper/paper.pdf` (source: `paper/paper.md`).

## Setup

```bash
# Requires Python 3.11+ and uv
uv sync
# For training: uv sync --extra train
# For benchmarks: uv sync --extra eval
```

## Reproducing Key Results

### 1. Baseline Geometric Audit (Section 4.1-4.2)

Extract token-level margins from Qwen3.5-4B-Base on WikiText-103 validation:

```bash
uv run mrp extract-token-stats \
  --model-id Qwen/Qwen3.5-4B-Base \
  --dataset-name wikitext --dataset-config wikitext-103-v1 \
  --split validation --max-length 512 --reservoir-size 1800 \
  --top-k 10 --trust-remote-code \
  --output-dir results/baseline
```

Analyze the expressibility gap (log-log scaling, gap coefficient):

```bash
uv run mrp analyze-margins \
  --token-stats results/baseline/token_stats.csv \
  --output results/baseline/margin_audit.json \
  --curve-output results/baseline/gap_curve.csv
```

### 2. MRP Training (Section 3.4)

Fisher information distance MRP at lambda=0.6:

```bash
uv run mrp train-mrp \
  --model-id Qwen/Qwen3.5-4B-Base \
  --output-dir results/fisher_0.6 \
  --dataset-name wikitext --dataset-config wikitext-103-raw-v1 \
  --train-split validation --block-size 512 --max-steps 200 \
  --per-device-train-batch-size 1 --learning-rate 1e-5 \
  --alpha-weight 0.6 --mrp-mode fisher --mrp-top-k 5 \
  --trainable-scope text --trust-remote-code
```

Direct margin maximization at lambda=0.3:

```bash
uv run mrp train-mrp \
  --model-id Qwen/Qwen3.5-4B-Base \
  --output-dir results/marginmax_0.3 \
  --dataset-name wikitext --dataset-config wikitext-103-raw-v1 \
  --train-split validation --block-size 512 --max-steps 200 \
  --per-device-train-batch-size 1 --learning-rate 1e-5 \
  --alpha-weight 0.3 --mrp-mode margin_max \
  --trainable-scope text --trust-remote-code
```

Entropy penalty baseline (geometry-free comparison):

```bash
uv run mrp train-mrp \
  --model-id Qwen/Qwen3.5-4B-Base \
  --output-dir results/entropy_0.6 \
  --dataset-name wikitext --dataset-config wikitext-103-raw-v1 \
  --train-split validation --block-size 512 --max-steps 200 \
  --per-device-train-batch-size 1 --learning-rate 1e-5 \
  --alpha-weight 0.6 --mrp-mode entropy \
  --trainable-scope text --trust-remote-code
```

### 3. Downstream Benchmarks (Section 3.5)

```bash
uv run mrp run-lm-eval \
  --model-id <checkpoint-path> \
  --tasks arc_challenge,hellaswag,winogrande,piqa,lambada_openai,truthfulqa_mc1 \
  --batch-size 4 --trust-remote-code \
  --output results/benchmarks/<run-name>.json
```

### 4. bfloat16 Quantization Artifact (Section 3.3)

Reproduce the bf16 margin quantization artifact on any model:

```bash
python scripts/diagnostics/bf16_margin_repro.py \
  --model-id Qwen/Qwen2.5-0.5B --max-sequences 8
```

### 5. Token-Class Audit (Section 4.11)

```bash
python scripts/analysis/token_class_flip_audit.py \
  --baseline results/baseline_audit/token_stats_fp32.csv \
  --treated results/<run>/audit/token_stats_fp32.csv \
  --output results/<run>/token_class_audit.json
```

## Pre-computed Results

The `results/` directory contains pre-computed audit summaries (JSON) for all 9 dose-response runs reported in the paper:

- `baseline_audit/` — Pre-MRP baseline margin statistics
- `run{N}_{method}_{lambda}/` — Post-MRP margin audits, frequency audits, and token-class audits

## MRP Loss Variants

| Mode | Flag | Description |
|------|------|-------------|
| Fisher | `--mrp-mode fisher` | Fisher information distance maximization (paper's primary method) |
| Margin Max | `--mrp-mode margin_max` | Direct margin maximization (comparison baseline) |
| Entropy | `--mrp-mode entropy` | Entropy penalty, geometry-free sharpening (ablation baseline) |
| CE weight | `--ce-weight 0.0` | Disable cross-entropy to run pure MRP |

## Citation

```bibtex
@article{brett2026geometric,
  title={Geometric Properties of the Voronoi Tessellation in Latent Semantic Manifolds of Large Language Models},
  author={Brett, Marshall},
  year={2026}
}
```

## References

This work builds on the latent semantic manifold framework of:

- Mabrok, M. (2026). Latent Semantic Manifolds in Large Language Models. arXiv:2603.22301.
