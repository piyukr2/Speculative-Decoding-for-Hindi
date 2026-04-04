"""
Unified evaluation for all EESD methods.

7 methods: autoregressive, draft-model, EESD-heavy-hook, EESD-heavy-true-exit,
           EESD-bottleneck-true-exit, EESD-thompson, EESD-entropy-exit
Plus: K-ablation, morphological analysis, losslessness verification,
      latency breakdown, prompt-length ablation, cross-lingual comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from transformers import AutoModelForCausalLM

from src.model import EarlyExitLM, BottleneckExitHead, ThompsonSamplingController
from src.inference import eesd_generate, eesd_generate_true_exit, eesd_generate_thompson, eesd_generate_entropy_exit
from src.draft_model_baseline import speculative_decode_step


def load_eval_prompts(num_samples, tokenizer):
    """Load XL-Sum Hindi test split, return list of (prompt_text, input_ids) tuples."""
    import json as _json
    from urllib.request import urlopen
    from datasets import Dataset as HFDataset

    _api = "https://datasets-server.huggingface.co/parquet?dataset=csebuetnlp/xlsum&config=hindi&split=test"
    with urlopen(_api, timeout=60) as _resp:
        _parquet_urls = [f["url"] for f in _json.loads(_resp.read())["parquet_files"]]
    dfs = [pd.read_parquet(url) for url in _parquet_urls]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    dataset = HFDataset.from_pandas(df, preserve_index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompts = []
    for idx in range(min(num_samples, len(dataset))):
        text = dataset[idx]["text"]
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = tokens["input_ids"][:, :50].to(device)  # first 50 tokens as prompt
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        prompts.append((prompt_text, input_ids))

    return prompts


@torch.no_grad()
def run_autoregressive(prompts, tokenizer, model, args):
    """Generate token-by-token with full model. Returns summary dict."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_tokens = 0
    t0 = time.time()

    for prompt_text, input_ids in prompts:
        out = model.base_model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        total_tokens += out.size(1) - input_ids.size(1)

    elapsed = time.time() - t0
    vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    return {
        "method": "autoregressive",
        "time_sec": round(elapsed, 2),
        "tokens_per_sec": round(total_tokens / elapsed, 2) if elapsed > 0 else 0.0,
        "total_tokens": total_tokens,
        "vram_mb": round(vram_mb, 1),
    }


@torch.no_grad()
def run_draft_model(prompts, tokenizer, args):
    """Speculative decoding with Qwen2-0.5B draft + Qwen2-1.5B verifier."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    draft_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B", torch_dtype=torch.float16, device_map="auto",
    )
    draft_model.eval()

    verifier_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B", torch_dtype=torch.float16, device_map="auto",
    )
    verifier_model.eval()

    total_tokens = 0
    total_accepted = 0
    total_drafted = 0
    t0 = time.time()

    for prompt_text, input_ids in prompts:
        generated = input_ids.clone()
        while generated.size(1) - input_ids.size(1) < args.max_new_tokens:
            new_tokens, n_accepted, n_drafted = speculative_decode_step(
                draft_model, verifier_model, generated, args.K
            )
            total_accepted += n_accepted
            total_drafted += n_drafted
            generated = torch.cat([generated, new_tokens], dim=1)
            if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in new_tokens[0].tolist():
                break
        total_tokens += generated.size(1) - input_ids.size(1)

    elapsed = time.time() - t0
    vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    del draft_model, verifier_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "method": "draft_model",
        "alpha": round(total_accepted / total_drafted, 4) if total_drafted > 0 else 0.0,
        "time_sec": round(elapsed, 2),
        "tokens_per_sec": round(total_tokens / elapsed, 2) if elapsed > 0 else 0.0,
        "total_tokens": total_tokens,
        "vram_mb": round(vram_mb, 1),
    }


@torch.no_grad()
def run_eesd_heavy_hook(prompts, model, tokenizer, args):
    """EESD with heavy exit heads, hook-based drafting (runs all layers)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.load_exit_heads("EESD/exit_heads_final.pt")
    model.eval()

    total_tokens = 0
    total_accepted = 0
    total_drafted = 0
    t0 = time.time()

    for prompt_text, input_ids in prompts:
        text, n_draft, n_accepted, wall = eesd_generate(
            model, tokenizer, prompt_text,
            K=args.K, max_new_tokens=args.max_new_tokens, temperature=0.0,
        )
        total_drafted += n_draft
        total_accepted += n_accepted
        total_tokens += len(tokenizer.encode(text))

    elapsed = time.time() - t0
    vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    return {
        "method": "eesd_heavy_hook",
        "alpha": round(total_accepted / total_drafted, 4) if total_drafted > 0 else 0.0,
        "time_sec": round(elapsed, 2),
        "tokens_per_sec": round(total_tokens / elapsed, 2) if elapsed > 0 else 0.0,
        "total_tokens": total_tokens,
        "vram_mb": round(vram_mb, 1),
    }


@torch.no_grad()
def run_eesd_heavy_true_exit(prompts, model, tokenizer, args):
    """EESD with heavy exit heads, true early exit (partial_forward, runs only d layers)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.load_exit_heads("EESD/exit_heads_final.pt")
    model.eval()

    total_tokens = 0
    total_accepted = 0
    total_drafted = 0
    draft_time = 0.0
    verify_time = 0.0
    overhead_time = 0.0
    t0 = time.time()

    for prompt_text, input_ids in prompts:
        text, stats = eesd_generate_true_exit(
            model, tokenizer, prompt_text,
            max_new_tokens=args.max_new_tokens, K=args.K, exit_depth=22,
        )
        total_drafted += stats["total_drafted"]
        total_accepted += stats["total_accepted"]
        total_tokens += stats["new_tokens"]
        draft_time += stats["draft_time"]
        verify_time += stats["verify_time"]
        overhead_time += stats["overhead_time"]

    elapsed = time.time() - t0
    vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    return {
        "method": "eesd_heavy_true_exit",
        "alpha": round(total_accepted / total_drafted, 4) if total_drafted > 0 else 0.0,
        "time_sec": round(elapsed, 2),
        "tokens_per_sec": round(total_tokens / elapsed, 2) if elapsed > 0 else 0.0,
        "total_tokens": total_tokens,
        "vram_mb": round(vram_mb, 1),
        "draft_time": round(draft_time, 2),
        "verify_time": round(verify_time, 2),
        "overhead_time": round(overhead_time, 2),
    }


@torch.no_grad()
def run_eesd_bottleneck_true_exit(prompts, model, tokenizer, args):
    """EESD with bottleneck exit heads, true early exit (partial_forward)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Replace exit heads with bottleneck variants and load weights
    hidden_size = model.base_model.config.hidden_size
    vocab_size = model.base_model.config.vocab_size
    original_exit_heads = model.exit_heads  # save to restore later
    model.exit_heads = torch.nn.ModuleDict(
        {str(d): BottleneckExitHead(hidden_size, vocab_size) for d in model.exit_depths}
    )
    model.load_exit_heads("EESD/bottleneck_exit_heads_final.pt")
    model.eval()

    total_tokens = 0
    total_accepted = 0
    total_drafted = 0
    draft_time = 0.0
    verify_time = 0.0
    overhead_time = 0.0
    t0 = time.time()

    for prompt_text, input_ids in prompts:
        text, stats = eesd_generate_true_exit(
            model, tokenizer, prompt_text,
            max_new_tokens=args.max_new_tokens, K=args.K, exit_depth=22,
        )
        total_drafted += stats["total_drafted"]
        total_accepted += stats["total_accepted"]
        total_tokens += stats["new_tokens"]
        draft_time += stats["draft_time"]
        verify_time += stats["verify_time"]
        overhead_time += stats["overhead_time"]

    elapsed = time.time() - t0
    vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    # Restore heavy exit heads so subsequent methods work correctly
    model.exit_heads = original_exit_heads

    return {
        "method": "eesd_bottleneck_true_exit",
        "alpha": round(total_accepted / total_drafted, 4) if total_drafted > 0 else 0.0,
        "time_sec": round(elapsed, 2),
        "tokens_per_sec": round(total_tokens / elapsed, 2) if elapsed > 0 else 0.0,
        "total_tokens": total_tokens,
        "vram_mb": round(vram_mb, 1),
        "draft_time": round(draft_time, 2),
        "verify_time": round(verify_time, 2),
        "overhead_time": round(overhead_time, 2),
    }


@torch.no_grad()
def run_eesd_thompson(prompts, model, tokenizer, args):
    """EESD with Thompson Sampling depth selection + true early exit."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Load heavy exit heads (swap to bottleneck if it has better α)
    model.load_exit_heads("EESD/exit_heads_final.pt")
    model.eval()

    total_tokens = 0
    total_accepted = 0
    total_drafted = 0
    draft_time = 0.0
    verify_time = 0.0
    overhead_time = 0.0
    depth_usage = {d: 0 for d in model.exit_depths}
    t0 = time.time()

    for prompt_text, input_ids in prompts:
        text, stats = eesd_generate_thompson(
            model, tokenizer, prompt_text,
            max_new_tokens=args.max_new_tokens, K=args.K,
        )
        total_drafted += stats["total_drafted"]
        total_accepted += stats["total_accepted"]
        total_tokens += stats["new_tokens"]
        draft_time += stats["draft_time"]
        verify_time += stats["verify_time"]
        overhead_time += stats["overhead_time"]
        for d in model.exit_depths:
            depth_usage[d] += stats["depth_usage"].get(d, 0)

    elapsed = time.time() - t0
    vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    return {
        "method": "eesd_thompson",
        "alpha": round(total_accepted / total_drafted, 4) if total_drafted > 0 else 0.0,
        "time_sec": round(elapsed, 2),
        "tokens_per_sec": round(total_tokens / elapsed, 2) if elapsed > 0 else 0.0,
        "total_tokens": total_tokens,
        "vram_mb": round(vram_mb, 1),
        "draft_time": round(draft_time, 2),
        "verify_time": round(verify_time, 2),
        "overhead_time": round(overhead_time, 2),
        "depth_usage": depth_usage,
    }


@torch.no_grad()
def run_eesd_entropy_exit(prompts, model, tokenizer, args):
    """EESD with entropy-based adaptive depth selection."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.load_exit_heads("EESD/exit_heads_final.pt")
    model.eval()
    total_tokens = 0
    total_accepted = 0
    total_drafted = 0
    draft_time = 0.0
    verify_time = 0.0
    overhead_time = 0.0
    depth_usage = {d: 0 for d in model.exit_depths}
    t0 = time.time()
    for prompt_text, input_ids in prompts:
        text, stats = eesd_generate_entropy_exit(
            model, tokenizer, prompt_text,
            max_new_tokens=args.max_new_tokens, K=args.K,
        )
        total_drafted += stats["total_drafted"]
        total_accepted += stats["total_accepted"]
        total_tokens += stats["new_tokens"]
        draft_time += stats["draft_time"]
        verify_time += stats["verify_time"]
        overhead_time += stats["overhead_time"]
        for d in model.exit_depths:
            depth_usage[d] += stats["depth_usage"].get(d, 0)
    elapsed = time.time() - t0
    vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    return {
        "method": "eesd_entropy_exit",
        "alpha": round(total_accepted / total_drafted, 4) if total_drafted > 0 else 0.0,
        "time_sec": round(elapsed, 2),
        "tokens_per_sec": round(total_tokens / elapsed, 2) if elapsed > 0 else 0.0,
        "total_tokens": total_tokens,
        "vram_mb": round(vram_mb, 1),
        "draft_time": round(draft_time, 2),
        "verify_time": round(verify_time, 2),
        "overhead_time": round(overhead_time, 2),
        "depth_usage": depth_usage,
    }


@torch.no_grad()
def cross_lingual_comparison(model, tokenizer, args):
    """Compare EESD performance on Hindi vs English using same model and exit heads."""
    import json as _json
    from urllib.request import urlopen

    model.load_exit_heads("EESD/exit_heads_final.pt")
    model.eval()
    device = next(model.exit_heads.parameters()).device

    _api = "https://datasets-server.huggingface.co/parquet?dataset=csebuetnlp/xlsum&config=english&split=test"
    with urlopen(_api, timeout=60) as _resp:
        _parquet_urls = [f["url"] for f in _json.loads(_resp.read())["parquet_files"]]
    dfs = [pd.read_parquet(url) for url in _parquet_urls]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    en_prompts = []
    for idx in range(min(50, len(df))):
        text = df.iloc[idx]["text"]
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = tokens["input_ids"][:, :50].to(device)
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        en_prompts.append((prompt_text, input_ids))

    en_accepted = 0
    en_drafted = 0
    en_tokens = 0
    t0_en = time.time()
    for prompt_text, input_ids in en_prompts:
        _, stats = eesd_generate_true_exit(
            model, tokenizer, prompt_text,
            max_new_tokens=args.max_new_tokens, K=3, exit_depth=22,
        )
        en_accepted += stats["total_accepted"]
        en_drafted += stats["total_drafted"]
        en_tokens += stats["new_tokens"]
    en_time = time.time() - t0_en
    en_alpha = en_accepted / en_drafted if en_drafted > 0 else 0.0

    hi_prompts = load_eval_prompts(50, tokenizer)
    hi_accepted = 0
    hi_drafted = 0
    hi_tokens = 0
    t0_hi = time.time()
    for prompt_text, input_ids in hi_prompts:
        _, stats = eesd_generate_true_exit(
            model, tokenizer, prompt_text,
            max_new_tokens=args.max_new_tokens, K=3, exit_depth=22,
        )
        hi_accepted += stats["total_accepted"]
        hi_drafted += stats["total_drafted"]
        hi_tokens += stats["new_tokens"]
    hi_time = time.time() - t0_hi
    hi_alpha = hi_accepted / hi_drafted if hi_drafted > 0 else 0.0

    return {
        "english_alpha": round(en_alpha, 4),
        "hindi_alpha": round(hi_alpha, 4),
        "english_samples": len(en_prompts),
        "hindi_samples": len(hi_prompts),
        "english_tokens_per_sec": round(en_tokens / en_time, 2) if en_time > 0 else 0.0,
        "hindi_tokens_per_sec": round(hi_tokens / hi_time, 2) if hi_time > 0 else 0.0,
    }


@torch.no_grad()
def k_ablation(prompts, model, tokenizer, args):
    """Sweep K=1..5, measure α and speedup relative to autoregressive baseline."""
    subset = prompts[:50]

    # Autoregressive baseline time on same subset
    ar_result = run_autoregressive(subset, tokenizer, model, args)
    ar_time = ar_result["time_sec"]

    results = {}
    for K in [1, 2, 3, 4, 5]:
        total_accepted = 0
        total_drafted = 0
        total_tokens_k = 0
        t0 = time.time()
        for prompt_text, input_ids in subset:
            _, stats = eesd_generate_true_exit(
                model, tokenizer, prompt_text,
                max_new_tokens=args.max_new_tokens, K=K, exit_depth=22,
            )
            total_accepted += stats["total_accepted"]
            total_drafted += stats["total_drafted"]
            total_tokens_k += stats["new_tokens"]
        elapsed = time.time() - t0
        alpha = total_accepted / total_drafted if total_drafted > 0 else 0.0
        speedup = ar_time / elapsed if elapsed > 0 else 0.0
        results[K] = {
            "alpha": round(alpha, 4),
            "speedup": round(speedup, 3),
            "time_sec": round(elapsed, 2),
            "tokens_per_sec": round(total_tokens_k / elapsed, 2) if elapsed > 0 else 0.0,
        }

    return {"k_ablation": results}


def analyze_morphology(generated_tokens_list, acceptance_mask_list, tokenizer):
    """Classify draft tokens into Hindi morphological categories and compute per-category α."""
    POSTPOSITIONS = {"ने", "को", "से", "में", "पर", "का", "की", "के", "लिए", "तक", "द्वारा"}
    VERB_SUFFIXES = ("ता", "ती", "ते", "ना", "कर", "या", "ये", "ई")

    categories = {
        "postpositions": {"accepted": 0, "total": 0},
        "verb_forms": {"accepted": 0, "total": 0},
        "compounds": {"accepted": 0, "total": 0},
        "other": {"accepted": 0, "total": 0},
    }

    for tokens, mask in zip(generated_tokens_list, acceptance_mask_list):
        for tok_id, accepted in zip(tokens, mask):
            text = tokenizer.decode([tok_id]).strip()
            if not text:
                continue

            if text in POSTPOSITIONS:
                cat = "postpositions"
            elif text.endswith(VERB_SUFFIXES):
                cat = "verb_forms"
            elif len(text) >= 5 and any("\u0900" <= ch <= "\u097F" for ch in text):
                cat = "compounds"
            else:
                cat = "other"

            categories[cat]["total"] += 1
            if accepted:
                categories[cat]["accepted"] += 1

    result = {}
    for cat, counts in categories.items():
        alpha = counts["accepted"] / counts["total"] if counts["total"] > 0 else 0.0
        result[cat] = {"alpha": round(alpha, 4), "count": counts["total"]}

    return result


@torch.no_grad()
def verify_losslessness(prompts, model, tokenizer, args):
    """Verify that greedy EESD produces identical tokens to greedy autoregressive."""
    subset = prompts[:50]
    exact_match = 0
    mismatch = 0
    mismatched_positions = []

    for idx, (prompt_text, input_ids) in enumerate(subset):
        # (a) Autoregressive greedy
        ar_out = model.base_model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        ar_tokens = ar_out[0, input_ids.size(1):].tolist()

        # (b) EESD true-exit greedy — capture raw token IDs
        eesd_generated = input_ids.clone()
        while eesd_generated.size(1) - input_ids.size(1) < args.max_new_tokens:
            draft_tokens = []
            cur_ids = eesd_generated.clone()
            for _ in range(args.K):
                logits, _ = model.partial_forward(cur_ids, 22)
                dt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                draft_tokens.append(dt)
                cur_ids = torch.cat([cur_ids, dt], dim=1)
            draft_ids = torch.cat(draft_tokens, dim=1)

            full_input = torch.cat([eesd_generated, draft_ids], dim=1)
            full_out = model.base_model(input_ids=full_input)
            full_logits = full_out.logits
            T_pos = eesd_generated.size(1)
            accepted = []
            for i in range(args.K):
                full_token = full_logits[:, T_pos + i - 1, :].argmax(dim=-1)
                draft_token = draft_ids[:, i]
                if full_token.item() == draft_token.item():
                    accepted.append(draft_token.unsqueeze(1))
                else:
                    accepted.append(full_token.unsqueeze(1))
                    break
            accepted_ids = torch.cat(accepted, dim=1)
            eesd_generated = torch.cat([eesd_generated, accepted_ids], dim=1)
            if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in accepted_ids[0].tolist():
                break

        eesd_tokens = eesd_generated[0, input_ids.size(1):].tolist()

        # Compare token-by-token up to min length
        min_len = min(len(ar_tokens), len(eesd_tokens))
        is_match = (len(ar_tokens) == len(eesd_tokens))
        first_diff = None
        for j in range(min_len):
            if ar_tokens[j] != eesd_tokens[j]:
                is_match = False
                first_diff = j
                break

        if is_match and first_diff is None:
            exact_match += 1
        else:
            mismatch += 1
            if len(mismatched_positions) < 10:
                mismatched_positions.append({
                    "prompt_idx": idx,
                    "first_diff_pos": first_diff if first_diff is not None else min_len,
                    "ar_len": len(ar_tokens),
                    "eesd_len": len(eesd_tokens),
                })

            if mismatch <= 3:
                print(f"\n[MISMATCH] Prompt {idx}:")
                print(f"  AR   ({len(ar_tokens)} tokens): {tokenizer.decode(ar_tokens[:20])}...")
                print(f"  EESD ({len(eesd_tokens)} tokens): {tokenizer.decode(eesd_tokens[:20])}...")
                if first_diff is not None:
                    print(f"  First diff at position {first_diff}: "
                          f"AR={tokenizer.decode([ar_tokens[first_diff]])} vs "
                          f"EESD={tokenizer.decode([eesd_tokens[first_diff]])}")

    total = len(subset)
    return {
        "total_prompts": total,
        "exact_match": exact_match,
        "mismatch": mismatch,
        "match_rate": round(exact_match / total, 4) if total > 0 else 0.0,
        "mismatched_positions": mismatched_positions,
    }


@torch.no_grad()
def prompt_length_ablation(prompts, model, tokenizer, args):
    """Bin prompts by input length, report α and speedup per bin."""
    bins = {"short": [], "medium": [], "long": []}

    for prompt_text, input_ids in prompts:
        n_tokens = input_ids.size(1)
        if n_tokens <= 30:
            bins["short"].append((prompt_text, input_ids))
        elif n_tokens <= 50:
            bins["medium"].append((prompt_text, input_ids))
        else:
            bins["long"].append((prompt_text, input_ids))

    result = {}
    for bin_name, bin_prompts in bins.items():
        if len(bin_prompts) < 5:
            result[bin_name] = {"alpha": None, "speedup": None, "count": len(bin_prompts),
                                "note": "insufficient samples"}
            continue

        # Autoregressive baseline for this bin
        ar_result = run_autoregressive(bin_prompts, tokenizer, model, args)
        ar_time = ar_result["time_sec"]

        # EESD true-exit at depth 22
        total_accepted = 0
        total_drafted = 0
        t0 = time.time()
        for prompt_text, input_ids in bin_prompts:
            _, stats = eesd_generate_true_exit(
                model, tokenizer, prompt_text,
                max_new_tokens=args.max_new_tokens, K=3, exit_depth=22,
            )
            total_accepted += stats["total_accepted"]
            total_drafted += stats["total_drafted"]
        elapsed = time.time() - t0

        alpha = total_accepted / total_drafted if total_drafted > 0 else 0.0
        speedup = ar_time / elapsed if elapsed > 0 else 0.0
        result[bin_name] = {
            "alpha": round(alpha, 4),
            "speedup": round(speedup, 3),
            "count": len(bin_prompts),
        }

    return result


class _Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self._stdout = sys.stdout
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        self._file = open(log_path, "w", buffering=1)  # line-buffered
    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        self._file.close()


def _checkpoint(all_results, output_path):
    """Save current results to a .partial file after each step."""
    partial_path = output_path.replace(".json", "_partial.json")
    os.makedirs(os.path.dirname(partial_path) if os.path.dirname(partial_path) else ".", exist_ok=True)
    with open(partial_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


def _sync_to_drive(paths, drive_dir="/content/drive/MyDrive/EESD_project/outputs"):
    """Copy result files to Google Drive if available."""
    try:
        from google.colab import drive as _drive  # noqa: F401 — import triggers mount check
        if not os.path.exists("/content/drive/MyDrive"):
            print("  Google Drive not mounted — skipping Drive sync.")
            return
        os.makedirs(drive_dir, exist_ok=True)
        import shutil
        for p in paths:
            if os.path.exists(p):
                dest = os.path.join(drive_dir, os.path.basename(p))
                shutil.copy2(p, dest)
                print(f"  Saved to Drive: {dest}")
    except Exception as e:
        print(f"  Drive sync skipped: {e}")


def main(args):
    # --- Logging setup: tee stdout to a log file ---
    log_path = args.output_path.replace(".json", ".log")
    tee = _Tee(log_path)
    sys.stdout = tee
    print(f"Logging to {log_path}")
    print(f"Args: {vars(args)}")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = EarlyExitLM(
        model_name_or_path="Qwen/Qwen2-1.5B",
        exit_depths=[8, 16, 22],
        draft_depth=8,
    )

    print(f"Loading {args.eval_samples} eval prompts...")
    prompts = load_eval_prompts(args.eval_samples, tokenizer)

    all_results = {}

    # --- 1. Autoregressive baseline ---
    print("\n[1/7] Autoregressive baseline...")
    ar = run_autoregressive(prompts, tokenizer, model, args)
    ar_time = ar["time_sec"]
    ar["speedup"] = 1.0
    all_results["autoregressive"] = ar
    print(f"  {ar['tokens_per_sec']} tok/s, {ar_time}s total")
    _checkpoint(all_results, args.output_path)

    # --- 2. Draft model ---
    print("\n[2/7] Draft model (Qwen2-0.5B + Qwen2-1.5B)...")
    dm = run_draft_model(prompts, tokenizer, args)
    dm["speedup"] = round(ar_time / dm["time_sec"], 3) if dm["time_sec"] > 0 else 0.0
    all_results["draft_model"] = dm
    print(f"  α={dm['alpha']}, speedup={dm['speedup']}x")
    _checkpoint(all_results, args.output_path)

    # --- 3. EESD heavy hook ---
    print("\n[3/7] EESD heavy hook...")
    eh = run_eesd_heavy_hook(prompts, model, tokenizer, args)
    eh["speedup"] = round(ar_time / eh["time_sec"], 3) if eh["time_sec"] > 0 else 0.0
    all_results["eesd_heavy_hook"] = eh
    print(f"  α={eh['alpha']}, speedup={eh['speedup']}x")
    _checkpoint(all_results, args.output_path)

    # --- 4. EESD heavy true exit ---
    print("\n[4/7] EESD heavy true exit...")
    model.load_exit_heads("EESD/exit_heads_final.pt")
    ht = run_eesd_heavy_true_exit(prompts, model, tokenizer, args)
    ht["speedup"] = round(ar_time / ht["time_sec"], 3) if ht["time_sec"] > 0 else 0.0
    all_results["eesd_heavy_true_exit"] = ht
    print(f"  α={ht['alpha']}, speedup={ht['speedup']}x")
    _checkpoint(all_results, args.output_path)

    # --- 5. EESD bottleneck true exit ---
    print("\n[5/7] EESD bottleneck true exit...")
    try:
        bt = run_eesd_bottleneck_true_exit(prompts, model, tokenizer, args)
        bt["speedup"] = round(ar_time / bt["time_sec"], 3) if bt["time_sec"] > 0 else 0.0
        all_results["eesd_bottleneck_true_exit"] = bt
        print(f"  α={bt['alpha']}, speedup={bt['speedup']}x")
    except FileNotFoundError:
        print("  SKIPPED — bottleneck checkpoint not found (EESD/bottleneck_exit_heads_final.pt)")
        all_results["eesd_bottleneck_true_exit"] = {"method": "eesd_bottleneck_true_exit", "skipped": True}
    _checkpoint(all_results, args.output_path)

    # --- 6. EESD Thompson ---
    print("\n[6/7] EESD Thompson...")
    # Reload heavy heads for Thompson
    model.load_exit_heads("EESD/exit_heads_final.pt")
    th = run_eesd_thompson(prompts, model, tokenizer, args)
    th["speedup"] = round(ar_time / th["time_sec"], 3) if th["time_sec"] > 0 else 0.0
    all_results["eesd_thompson"] = th
    print(f"  α={th['alpha']}, speedup={th['speedup']}x, depth_usage={th['depth_usage']}")
    _checkpoint(all_results, args.output_path)

    # --- 7. EESD entropy exit ---
    print("\n[7/7] EESD entropy exit...")
    model.load_exit_heads("EESD/exit_heads_final.pt")
    ee = run_eesd_entropy_exit(prompts, model, tokenizer, args)
    ee["speedup"] = round(ar_time / ee["time_sec"], 3) if ee["time_sec"] > 0 else 0.0
    all_results["eesd_entropy_exit"] = ee
    print(f"  α={ee['alpha']}, speedup={ee['speedup']}x, depth_usage={ee['depth_usage']}")
    _checkpoint(all_results, args.output_path)

    # --- Cross-lingual comparison ---
    print("\n--- Cross-lingual comparison ---")
    cross_ling = cross_lingual_comparison(model, tokenizer, args)
    all_results["cross_lingual"] = cross_ling
    print(f"  English α={cross_ling['english_alpha']}, Hindi α={cross_ling['hindi_alpha']}")

    # --- Morphological analysis ---
    print("\n--- Morphological analysis ---")
    model.load_exit_heads("EESD/exit_heads_final.pt")
    model.eval()
    morph_tokens_list = []
    morph_mask_list = []
    for prompt_text, input_ids in prompts[:50]:
        cur_ids = input_ids.clone()
        draft_tokens = []
        for _ in range(args.K):
            logits, _ = model.partial_forward(cur_ids, 22)
            dt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            draft_tokens.append(dt)
            cur_ids = torch.cat([cur_ids, dt], dim=1)
        draft_ids = torch.cat(draft_tokens, dim=1)

        full_input = torch.cat([input_ids, draft_ids], dim=1)
        full_out = model.base_model(input_ids=full_input)
        full_logits = full_out.logits
        T = input_ids.size(1)
        acceptance_mask = []
        for i in range(args.K):
            full_token = full_logits[:, T + i - 1, :].argmax(dim=-1)
            accepted = (full_token.item() == draft_ids[:, i].item())
            acceptance_mask.append(accepted)
            if not accepted:
                break
        while len(acceptance_mask) < args.K:
            acceptance_mask.append(False)

        morph_tokens_list.append(draft_ids[0].tolist())
        morph_mask_list.append(acceptance_mask)

    morph_results = analyze_morphology(morph_tokens_list, morph_mask_list, tokenizer)
    all_results["morphological_analysis"] = morph_results
    for cat, v in morph_results.items():
        print(f"  {cat}: α={v['alpha']}, count={v['count']}")

    # --- Losslessness verification ---
    print("\n--- Losslessness verification ---")
    model.load_exit_heads("EESD/exit_heads_final.pt")
    lossless = verify_losslessness(prompts, model, tokenizer, args)
    all_results["losslessness"] = lossless
    print(f"  Match rate: {lossless['match_rate']} ({lossless['exact_match']}/{lossless['total_prompts']})")

    # --- K-ablation ---
    print("\n--- K-ablation ---")
    model.draft_depth = 22
    k_abl = k_ablation(prompts, model, tokenizer, args)
    all_results["k_ablation"] = k_abl["k_ablation"]
    for k, v in k_abl["k_ablation"].items():
        print(f"  K={k}: α={v['alpha']}, speedup={v['speedup']}x")

    # --- Prompt-length ablation ---
    print("\n--- Prompt-length ablation ---")
    pl_abl = prompt_length_ablation(prompts, model, tokenizer, args)
    all_results["prompt_length_ablation"] = pl_abl
    for bin_name, v in pl_abl.items():
        print(f"  {bin_name}: α={v.get('alpha')}, speedup={v.get('speedup')}, n={v['count']}")

    # --- Latency breakdown ---
    latency = {}
    for method_name in ["eesd_heavy_true_exit", "eesd_bottleneck_true_exit", "eesd_thompson", "eesd_entropy_exit"]:
        r = all_results[method_name]
        if "draft_time" in r:
            latency[method_name] = {
                "draft_time": r["draft_time"],
                "verify_time": r["verify_time"],
                "overhead_time": r["overhead_time"],
            }
    all_results["latency_breakdown"] = latency

    # --- Save ---
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {args.output_path}")

    # --- Summary table ---
    print(f"\n{'='*70}")
    print(f"{'Method':<30} {'α':>8} {'Speedup':>10} {'tok/s':>10} {'VRAM MB':>10}")
    print(f"{'-'*70}")
    for name in ["autoregressive", "draft_model", "eesd_heavy_hook",
                  "eesd_heavy_true_exit", "eesd_bottleneck_true_exit", "eesd_thompson",
                  "eesd_entropy_exit"]:
        r = all_results.get(name, {})
        if r.get("skipped"):
            print(f"{name:<30} {'SKIPPED':>8}")
            continue
        alpha = r.get("alpha", "—")
        speedup = r.get("speedup", "—")
        tps = r.get("tokens_per_sec", "—")
        vram = r.get("vram_mb", "—")
        print(f"{name:<30} {alpha:>8} {speedup:>10} {tps:>10} {vram:>10}")
    print(f"{'='*70}")

    # --- Sync to Google Drive ---
    print("\n--- Syncing to Google Drive ---")
    _sync_to_drive([
        args.output_path,
        args.output_path.replace(".json", "_partial.json"),
        args.output_path.replace(".json", ".log"),
    ])

    # Restore stdout and close log file
    sys.stdout = tee._stdout
    tee.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified EESD evaluation")
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--output_path", type=str, default="outputs/ablation_results.json")
    args = parser.parse_args()
    main(args)
