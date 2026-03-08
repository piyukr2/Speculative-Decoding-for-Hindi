"""
EESD Inference Engine.

Implements the speculative decoding loop:
  1. Generate K draft tokens with the early-exit head (cheap).
  2. Verify all K draft tokens with the full model in one forward pass.
  3. Accept tokens up to and including the first mismatch.
  4. Repeat until EOS or max_new_tokens is reached.

Also provides a standard autoregressive baseline for speedup comparison.

Usage:
    python -m src.inference \
        --model_name Qwen/Qwen2-1.5B \
        --checkpoint results/checkpoints/exit_heads_final.pt \
        --draft_depth 2 \
        --K 5 \
        --prompt "भारत एक विविध देश है जहाँ"
"""

from __future__ import annotations

import argparse
import time
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer

from src.model import EarlyExitLM


# ---------------------------------------------------------------------------
# EESD generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eesd_generate(
    model: EarlyExitLM,
    tokenizer,
    prompt: str,
    K: int = 5,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> Tuple[str, int, int, float]:
    """
    Generate text with EESD speculative decoding.

    Returns:
        (generated_text, total_draft_tokens, total_accepted_tokens, wall_time)
    """
    device = next(model.exit_heads.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    total_draft = 0
    total_accepted = 0
    generated_ids = input_ids.clone()

    t0 = time.perf_counter()
    while generated_ids.size(1) - input_ids.size(1) < max_new_tokens:
        # 1. Draft
        draft_ids = model.draft(generated_ids, K=K, temperature=temperature)
        total_draft += draft_ids.size(1)

        # 2. Verify
        accepted_ids = model.verify(generated_ids, draft_ids)
        n_accepted = accepted_ids.size(1)
        total_accepted += n_accepted

        # 3. Append accepted tokens
        generated_ids = torch.cat([generated_ids, accepted_ids], dim=1)

        # 4. Check for EOS
        if tokenizer.eos_token_id in accepted_ids[0].tolist():
            break

    wall_time = time.perf_counter() - t0
    generated_text = tokenizer.decode(
        generated_ids[0, input_ids.size(1):], skip_special_tokens=True
    )
    return generated_text, total_draft, total_accepted, wall_time


# ---------------------------------------------------------------------------
# Autoregressive baseline
# ---------------------------------------------------------------------------

@torch.no_grad()
def autoregressive_generate(
    model: EarlyExitLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
) -> Tuple[str, float]:
    """
    Standard autoregressive generation with the full model.

    Returns:
        (generated_text, wall_time)
    """
    device = next(model.exit_heads.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    t0 = time.perf_counter()
    out = model.base_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    wall_time = time.perf_counter() - t0

    prompt_len = inputs["input_ids"].size(1)
    generated_text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
    return generated_text, wall_time


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EESD inference demo")
    p.add_argument("--model_name", default="Qwen/Qwen2-1.5B")
    p.add_argument("--checkpoint", default="results/checkpoints/exit_heads_final.pt",
                   help="Path to saved exit-head weights")
    p.add_argument("--exit_depths", nargs="+", type=int, default=[8, 16, 22])
    p.add_argument("--draft_depth", type=int, default=8,
                   help="Which exit depth to use for drafting")
    p.add_argument("--K", type=int, default=5,
                   help="Draft length (tokens per speculative step)")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--prompt", type=str,
                   default="भारत एक विविध देश है जहाँ कई भाषाएँ बोली जाती हैं।")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = EarlyExitLM(
        model_name_or_path=args.model_name,
        exit_depths=args.exit_depths,
        draft_depth=args.draft_depth,
    )
    model.load_exit_heads(args.checkpoint)
    model.eval()

    print(f"\nPrompt: {args.prompt}\n")

    # EESD
    text_eesd, n_draft, n_accepted, t_eesd = eesd_generate(
        model, tokenizer, args.prompt,
        K=args.K,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    alpha = n_accepted / n_draft if n_draft > 0 else 0.0
    print(f"[EESD] Generated ({t_eesd:.2f}s):\n  {text_eesd}")
    print(f"  Acceptance rate α = {alpha:.3f}  ({n_accepted}/{n_draft} tokens accepted)")

    # Autoregressive baseline
    text_ar, t_ar = autoregressive_generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    speedup = t_ar / t_eesd if t_eesd > 0 else float("inf")
    print(f"\n[AR]   Generated ({t_ar:.2f}s):\n  {text_ar}")
    print(f"\nSpeedup: {speedup:.2f}x")
