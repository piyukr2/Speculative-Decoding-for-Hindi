#!/usr/bin/env bash
# Evaluate EESD: depth analysis + morphological analysis.

set -euo pipefail

MODEL="Qwen/Qwen2-1.5B"
CHECKPOINT="results/checkpoints/exit_heads_final.pt"
EXIT_DEPTHS="2 4 6"
K=5
MAX_EVAL_SAMPLES=500
MAX_NEW_TOKENS=50
OUTPUT_DIR="results"

python3 -m src.evaluate \
  --model_name "$MODEL" \
  --checkpoint "$CHECKPOINT" \
  --exit_depths $EXIT_DEPTHS \
  --K "$K" \
  --max_eval_samples "$MAX_EVAL_SAMPLES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --output_dir "$OUTPUT_DIR"
