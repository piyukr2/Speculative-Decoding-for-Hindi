"""
Draft-Model Speculative Decoding Baseline.

Uses Qwen2-0.5B as draft model and Qwen2-1.5B as verifier.
Standard speculative decoding per Leviathan et al. 2023.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_models():
    """Load draft (0.5B), verifier (1.5B), and shared tokenizer."""
    draft_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    draft_model.eval()

    verifier_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    verifier_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return draft_model, verifier_model, tokenizer


@torch.no_grad()
def speculative_decode_step(draft_model, verifier_model, input_ids, K):
    """One speculative decoding step: draft K tokens, verify, accept greedily.

    Returns:
        new_tokens: [1, n_accepted] accepted token ids.
        num_accepted: number of accepted tokens.
        num_drafted: K (always).
    """
    device = input_ids.device
    cur_ids = input_ids.clone()

    # --- Draft K tokens autoregressively ---
    draft_tokens = []
    for _ in range(K):
        out = draft_model(input_ids=cur_ids)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
        draft_tokens.append(next_token)
        cur_ids = torch.cat([cur_ids, next_token], dim=1)

    draft_ids = torch.cat(draft_tokens, dim=1)  # [1, K]

    # --- Verify with full model in one forward pass ---
    full_input = torch.cat([input_ids, draft_ids], dim=1)  # [1, T+K]
    full_out = verifier_model(input_ids=full_input)
    full_logits = full_out.logits  # [1, T+K, V]

    T = input_ids.size(1)
    accepted = []
    n_matched = 0
    for i in range(K):
        verifier_token = full_logits[:, T + i - 1, :].argmax(dim=-1)  # [1]
        draft_token = draft_ids[:, i]  # [1]
        if verifier_token.item() == draft_token.item():
            accepted.append(draft_token.unsqueeze(1))
            n_matched += 1
        else:
            accepted.append(verifier_token.unsqueeze(1))  # take verifier's token
            break

    new_tokens = torch.cat(accepted, dim=1)  # [1, len(accepted)]
    return new_tokens, n_matched, K


def run_baseline(args):
    print("Loading models...")
    draft_model, verifier_model, tokenizer = load_models()
    device = next(verifier_model.parameters()).device

    print("Loading XL-Sum Hindi test set...")
    dataset = load_dataset("csebuetnlp/xlsum", "hindi", split="test")

    results_per_prompt = []
    total_accepted = 0
    total_drafted = 0
    total_time = 0.0

    for idx in range(min(args.num_samples, len(dataset))):
        text = dataset[idx]["text"]
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        prompt_ids = tokens["input_ids"][:, :50].to(device)  # first 50 tokens as prompt

        generated = prompt_ids.clone()
        prompt_accepted = 0
        prompt_drafted = 0

        t0 = time.time()
        while generated.size(1) - prompt_ids.size(1) < args.max_new_tokens:
            new_tokens, n_accepted, n_drafted = speculative_decode_step(
                draft_model, verifier_model, generated, args.K
            )
            prompt_accepted += n_accepted
            prompt_drafted += n_drafted
            generated = torch.cat([generated, new_tokens], dim=1)

            # Check EOS
            if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in new_tokens[0].tolist():
                break
        elapsed = time.time() - t0

        new_token_count = generated.size(1) - prompt_ids.size(1)
        alpha = prompt_accepted / prompt_drafted if prompt_drafted > 0 else 0.0

        results_per_prompt.append({
            "idx": idx,
            "alpha": alpha,
            "time": elapsed,
            "new_tokens": new_token_count,
            "tokens_per_sec": new_token_count / elapsed if elapsed > 0 else 0.0,
            "accepted": prompt_accepted,
            "drafted": prompt_drafted,
        })

        total_accepted += prompt_accepted
        total_drafted += prompt_drafted
        total_time += elapsed

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{args.num_samples}] α={alpha:.3f} | {new_token_count} tokens in {elapsed:.2f}s")

    # Peak VRAM
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

    avg_alpha = total_accepted / total_drafted if total_drafted > 0 else 0.0
    avg_tokens_per_sec = sum(r["tokens_per_sec"] for r in results_per_prompt) / len(results_per_prompt)

    summary = {
        "method": "draft_model_baseline",
        "draft_model": "Qwen/Qwen2-0.5B",
        "verifier_model": "Qwen/Qwen2-1.5B",
        "K": args.K,
        "max_new_tokens": args.max_new_tokens,
        "num_samples": len(results_per_prompt),
        "avg_alpha": avg_alpha,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "total_time": total_time,
        "peak_vram_gb": peak_vram_gb,
        "per_prompt": results_per_prompt,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Draft-Model Baseline Summary ===")
    print(f"Samples: {len(results_per_prompt)}")
    print(f"Avg α: {avg_alpha:.3f}")
    print(f"Avg tokens/sec: {avg_tokens_per_sec:.2f}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draft-model speculative decoding baseline")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--output_path", type=str, default="outputs/draft_model_results.json")
    args = parser.parse_args()
    run_baseline(args)
