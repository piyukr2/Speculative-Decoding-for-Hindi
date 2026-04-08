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
import sys
import time
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer

from src.model import EarlyExitLM, ThompsonSamplingController


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
    total_matched = 0
    generated_ids = input_ids.clone()

    t0 = time.perf_counter()
    while generated_ids.size(1) - input_ids.size(1) < max_new_tokens:
        # 1. Draft
        draft_ids = model.draft(generated_ids, K=K, temperature=temperature)
        total_draft += draft_ids.size(1)

        # 2. Verify
        accepted_ids = model.verify(generated_ids, draft_ids)
        n_accepted = accepted_ids.size(1)
        # Count only exact matches (not the correction token)
        n_matched = 0
        for i in range(min(n_accepted, draft_ids.size(1))):
            if accepted_ids[0, i].item() == draft_ids[0, i].item():
                n_matched += 1
            else:
                break
        total_matched += n_matched

        # 3. Append accepted tokens
        generated_ids = torch.cat([generated_ids, accepted_ids], dim=1)

        # 4. Check for EOS
        if tokenizer.eos_token_id in accepted_ids[0].tolist():
            break

    wall_time = time.perf_counter() - t0
    generated_text = tokenizer.decode(
        generated_ids[0, input_ids.size(1):], skip_special_tokens=True
    )
    return generated_text, total_draft, total_matched, wall_time


# ---------------------------------------------------------------------------
# EESD with true early exit
# ---------------------------------------------------------------------------

@torch.no_grad()
def eesd_generate_true_exit(
    model: EarlyExitLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    K: int = 3,
    exit_depth: int = 8,
) -> Tuple[str, Dict]:
    """EESD with true early exit — draft step runs only first `exit_depth` layers."""
    device = next(model.exit_heads.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    generated = input_ids.clone()
    matched_total = 0
    accepted_total = 0
    drafted_total = 0
    draft_time_total = 0.0
    verify_time_total = 0.0

    start_time = time.time()

    while generated.size(1) - input_ids.size(1) < max_new_tokens:
        # --- DRAFT phase: run only first `exit_depth` layers ---
        t0 = time.time()
        draft_tokens = []
        cur_ids = generated.clone()
        for _ in range(K):
            logits, _ = model.partial_forward(cur_ids, exit_depth)
            draft_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
            draft_tokens.append(draft_token)
            cur_ids = torch.cat([cur_ids, draft_token], dim=1)
        draft_time_total += time.time() - t0

        draft_ids = torch.cat(draft_tokens, dim=1)  # [1, K]
        drafted_total += K

        # --- VERIFY phase: full model forward on generated + draft_tokens ---
        t0 = time.time()
        full_input = torch.cat([generated, draft_ids], dim=1)  # [1, T+K]
        full_out = model.base_model(input_ids=full_input)
        full_logits = full_out.logits  # [1, T+K, V]

        T = generated.size(1)
        accepted = []
        n_matched = 0
        for i in range(K):
            full_token = full_logits[:, T + i - 1, :].argmax(dim=-1)  # [1]
            draft_token = draft_ids[:, i]  # [1]
            if full_token.item() == draft_token.item():
                accepted.append(draft_token.unsqueeze(1))
                n_matched += 1
            else:
                accepted.append(full_token.unsqueeze(1))  # take full model's token
                break
        verify_time_total += time.time() - t0

        matched_total += n_matched
        accepted_total += len(accepted)
        accepted_ids = torch.cat(accepted, dim=1)  # [1, len(accepted)]
        generated = torch.cat([generated, accepted_ids], dim=1)

        # Check for EOS
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in accepted_ids[0].tolist():
            break

    elapsed = time.time() - start_time
    overhead_time = elapsed - draft_time_total - verify_time_total
    new_tokens = generated.size(1) - input_ids.size(1)

    generated_text = tokenizer.decode(
        generated[0, input_ids.size(1):], skip_special_tokens=True
    )

    stats = {
        "alpha": matched_total / drafted_total if drafted_total > 0 else 0.0,
        "time": elapsed,
        "tokens_per_sec": new_tokens / elapsed if elapsed > 0 else 0.0,
        "draft_time": draft_time_total,
        "verify_time": verify_time_total,
        "overhead_time": overhead_time,
        "total_drafted": drafted_total,
        "total_matched": matched_total,
        "total_accepted": accepted_total,
        "new_tokens": new_tokens,
    }

    return generated_text, stats


# ---------------------------------------------------------------------------
# EESD with Thompson Sampling depth selection
# ---------------------------------------------------------------------------

@torch.no_grad()
def eesd_generate_thompson(
    model: EarlyExitLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    K: int = 3,
) -> Tuple[str, Dict]:
    """EESD with true early exit + Thompson Sampling depth selection."""
    device = next(model.exit_heads.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    controller = ThompsonSamplingController(model.exit_depths)

    generated = input_ids.clone()
    matched_total = 0
    accepted_total = 0
    drafted_total = 0
    draft_time_total = 0.0
    verify_time_total = 0.0
    depth_usage = {d: 0 for d in model.exit_depths}
    per_depth_accepted = {d: 0 for d in model.exit_depths}
    per_depth_drafted = {d: 0 for d in model.exit_depths}

    start_time = time.time()

    while generated.size(1) - input_ids.size(1) < max_new_tokens:
        # Select depth via Thompson Sampling
        exit_depth = controller.select_depth()
        depth_usage[exit_depth] += 1

        # --- DRAFT phase: run only first `exit_depth` layers ---
        t0 = time.time()
        draft_tokens = []
        cur_ids = generated.clone()
        for _ in range(K):
            logits, _ = model.partial_forward(cur_ids, exit_depth)
            draft_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            draft_tokens.append(draft_token)
            cur_ids = torch.cat([cur_ids, draft_token], dim=1)
        draft_time_total += time.time() - t0

        draft_ids = torch.cat(draft_tokens, dim=1)
        drafted_total += K
        per_depth_drafted[exit_depth] += K

        # --- VERIFY phase: full model forward ---
        t0 = time.time()
        full_input = torch.cat([generated, draft_ids], dim=1)
        full_out = model.base_model(input_ids=full_input)
        full_logits = full_out.logits

        T = generated.size(1)
        accepted = []
        n_matched = 0
        for i in range(K):
            full_token = full_logits[:, T + i - 1, :].argmax(dim=-1)
            draft_token = draft_ids[:, i]
            if full_token.item() == draft_token.item():
                accepted.append(draft_token.unsqueeze(1))
                n_matched += 1
            else:
                accepted.append(full_token.unsqueeze(1))
                break
        verify_time_total += time.time() - t0

        matched_total += n_matched
        accepted_total += len(accepted)
        per_depth_accepted[exit_depth] += n_matched

        # Update Thompson Sampling controller
        controller.update(exit_depth, n_matched, K)

        accepted_ids = torch.cat(accepted, dim=1)
        generated = torch.cat([generated, accepted_ids], dim=1)

        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in accepted_ids[0].tolist():
            break

    elapsed = time.time() - start_time
    overhead_time = elapsed - draft_time_total - verify_time_total
    new_tokens = generated.size(1) - input_ids.size(1)

    generated_text = tokenizer.decode(
        generated[0, input_ids.size(1):], skip_special_tokens=True
    )

    per_depth_alpha = {
        d: round(per_depth_accepted[d] / per_depth_drafted[d], 4) if per_depth_drafted[d] > 0 else 0.0
        for d in model.exit_depths
    }

    stats = {
        "alpha": matched_total / drafted_total if drafted_total > 0 else 0.0,
        "time": elapsed,
        "tokens_per_sec": new_tokens / elapsed if elapsed > 0 else 0.0,
        "draft_time": draft_time_total,
        "verify_time": verify_time_total,
        "overhead_time": overhead_time,
        "total_drafted": drafted_total,
        "total_matched": matched_total,
        "total_accepted": accepted_total,
        "new_tokens": new_tokens,
        "depth_usage": depth_usage,
        "per_depth_accepted": per_depth_accepted,
        "per_depth_drafted": per_depth_drafted,
        "per_depth_alpha": per_depth_alpha,
    }

    return generated_text, stats


# ---------------------------------------------------------------------------
# Entropy-based confidence exit
# ---------------------------------------------------------------------------

@torch.no_grad()
def eesd_generate_entropy_exit(
    model: EarlyExitLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    K: int = 3,
    entropy_threshold: float = 1.5,
) -> Tuple[str, Dict]:
    """EESD with entropy-based adaptive depth selection.

    At each draft token, tries shallowest exit head first.
    If entropy of output distribution > threshold, goes deeper.
    This avoids Thompson Sampling's cold-start problem.
    """
    device = next(model.exit_heads.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    generated = input_ids.clone()
    matched_total = 0
    drafted_total = 0
    draft_time_total = 0.0
    verify_time_total = 0.0
    depth_usage = {d: 0 for d in model.exit_depths}
    per_depth_accepted = {d: 0 for d in model.exit_depths}
    per_depth_drafted = {d: 0 for d in model.exit_depths}

    start_time = time.time()

    while generated.size(1) - input_ids.size(1) < max_new_tokens:
        # --- DRAFT phase: entropy-based depth selection ---
        t0 = time.time()
        draft_tokens = []
        draft_chosen_depths = []
        cur_ids = generated.clone()
        for _ in range(K):
            chosen_depth = None
            draft_token = None
            for depth in sorted(model.exit_depths):
                logits, _ = model.partial_forward(cur_ids, depth)
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).item()
                if entropy < entropy_threshold or depth == max(model.exit_depths):
                    chosen_depth = depth
                    draft_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    break
            depth_usage[chosen_depth] += 1
            per_depth_drafted[chosen_depth] += 1
            draft_tokens.append(draft_token)
            draft_chosen_depths.append(chosen_depth)
            cur_ids = torch.cat([cur_ids, draft_token], dim=1)
        draft_time_total += time.time() - t0

        draft_ids = torch.cat(draft_tokens, dim=1)
        drafted_total += K

        # --- VERIFY phase: full model forward ---
        t0 = time.time()
        full_input = torch.cat([generated, draft_ids], dim=1)
        full_out = model.base_model(input_ids=full_input)
        full_logits = full_out.logits

        T = generated.size(1)
        accepted = []
        n_matched = 0
        for i in range(K):
            full_token = full_logits[:, T + i - 1, :].argmax(dim=-1)
            draft_token = draft_ids[:, i]
            if full_token.item() == draft_token.item():
                accepted.append(draft_token.unsqueeze(1))
                n_matched += 1
                per_depth_accepted[draft_chosen_depths[i]] += 1
            else:
                accepted.append(full_token.unsqueeze(1))
                break
        verify_time_total += time.time() - t0

        matched_total += n_matched
        accepted_ids = torch.cat(accepted, dim=1)
        generated = torch.cat([generated, accepted_ids], dim=1)

        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in accepted_ids[0].tolist():
            break

    elapsed = time.time() - start_time
    overhead_time = elapsed - draft_time_total - verify_time_total
    new_tokens = generated.size(1) - input_ids.size(1)

    generated_text = tokenizer.decode(
        generated[0, input_ids.size(1):], skip_special_tokens=True
    )

    per_depth_alpha = {
        d: round(per_depth_accepted[d] / per_depth_drafted[d], 4) if per_depth_drafted[d] > 0 else 0.0
        for d in model.exit_depths
    }

    stats = {
        "alpha": matched_total / drafted_total if drafted_total > 0 else 0.0,
        "time": elapsed,
        "tokens_per_sec": new_tokens / elapsed if elapsed > 0 else 0.0,
        "draft_time": draft_time_total,
        "verify_time": verify_time_total,
        "overhead_time": overhead_time,
        "total_drafted": drafted_total,
        "total_matched": matched_total,
        "total_accepted": matched_total,
        "new_tokens": new_tokens,
        "depth_usage": depth_usage,
        "per_depth_accepted": per_depth_accepted,
        "per_depth_drafted": per_depth_drafted,
        "per_depth_alpha": per_depth_alpha,
    }

    return generated_text, stats


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

    # Quick test for true early exit
    if "--test_true_exit" in sys.argv:
        print("\n--- True Early Exit Test ---")
        text_te, stats_te = eesd_generate_true_exit(
            model, tokenizer, "भारत एक",
            max_new_tokens=20, K=3, exit_depth=8,
        )
        print(f"Text: {text_te}")
        print(f"Stats: {stats_te}")
