#!/usr/bin/env bash
# Run EESD inference demo on a Hindi prompt.

set -euo pipefail

MODEL="Qwen/Qwen2-1.5B"
CHECKPOINT="results/checkpoints/exit_heads_final.pt"
DRAFT_DEPTH=2
K=5
MAX_NEW_TOKENS=100
PROMPT="भारत एक विविध देश है जहाँ कई भाषाएँ बोली जाती हैं।"

python3 -m src.inference \
  --model_name "$MODEL" \
  --checkpoint "$CHECKPOINT" \
  --exit_depths 2 4 6 \
  --draft_depth "$DRAFT_DEPTH" \
  --K "$K" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --prompt "$PROMPT"
