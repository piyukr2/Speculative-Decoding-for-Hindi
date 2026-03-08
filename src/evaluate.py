"""
Evaluation script: acceptance rate, speedup, and morphological analysis.

Experiments (from the paper):
  1. Depth Analysis – evaluate α and speedup at exit depths {2, 4, 6}
  2. Morphological Analysis – stratify α by token type
     (inflected verbs, compound words, postpositions)

Usage:
    python -m src.evaluate \
        --model_name Qwen/Qwen2-1.5B \
        --checkpoint results/checkpoints/exit_heads_final.pt \
        --exit_depths 2 4 6 \
        --K 5 \
        --max_eval_samples 500 \
        --output_dir results
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from src.model import EarlyExitLM
from src.data import get_eval_dataloader
from src.inference import eesd_generate, autoregressive_generate


# ---------------------------------------------------------------------------
# Hindi morphological token categorisation (heuristic, Unicode-based)
# ---------------------------------------------------------------------------

# Devanagari Unicode range: U+0900–U+097F
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")

# Common Hindi postpositions (written as separate tokens in most tokenisers)
POSTPOSITIONS = {
    "में", "को", "से", "का", "की", "के", "पर", "ने", "तक",
    "के लिए", "के बारे में", "के साथ", "के बाद", "के पहले",
}

# Common verb endings in Devanagari (infinitive / conjugated suffixes)
VERB_SUFFIX_RE = re.compile(r"(ना|ता|ती|ते|या|यी|येंगे|एगा|एगी|कर|ने)$")

# Compound / sandhi marker – two Devanagari words joined (simple heuristic)
COMPOUND_RE = re.compile(r"[\u0900-\u097F]{5,}")  # long words often compounds


def classify_token(token: str) -> str:
    """Return one of: 'postposition', 'verb', 'compound', 'other'."""
    text = token.strip()
    if text in POSTPOSITIONS:
        return "postposition"
    if DEVANAGARI_RE.fullmatch(text):
        if VERB_SUFFIX_RE.search(text):
            return "verb"
        if COMPOUND_RE.fullmatch(text):
            return "compound"
    return "other"


# ---------------------------------------------------------------------------
# Depth Analysis
# ---------------------------------------------------------------------------

def depth_analysis(
    model: EarlyExitLM,
    tokenizer,
    eval_texts: List[str],
    exit_depths: List[int],
    K: int,
    max_new_tokens: int,
    device: torch.device,
) -> Dict[int, Dict]:
    """
    For each exit depth, compute mean acceptance rate and speedup.

    Returns dict: {depth: {"alpha": float, "speedup": float, "eesd_time": float, "ar_time": float}}
    """
    results = {}

    # Compute autoregressive baseline once
    print("\n[Depth Analysis] Computing autoregressive baseline …")
    ar_times = []
    for text in eval_texts:
        _, t = autoregressive_generate(model, tokenizer, text, max_new_tokens=max_new_tokens)
        ar_times.append(t)
    mean_ar_time = sum(ar_times) / len(ar_times)

    for depth in exit_depths:
        print(f"\n  Testing exit depth {depth} …")
        original_depth = model.draft_depth
        model.draft_depth = depth

        alphas = []
        eesd_times = []
        for text in eval_texts:
            _, n_draft, n_accepted, t = eesd_generate(
                model, tokenizer, text, K=K, max_new_tokens=max_new_tokens
            )
            if n_draft > 0:
                alphas.append(n_accepted / n_draft)
            eesd_times.append(t)

        mean_alpha = sum(alphas) / len(alphas) if alphas else 0.0
        mean_eesd_time = sum(eesd_times) / len(eesd_times)
        speedup = mean_ar_time / mean_eesd_time if mean_eesd_time > 0 else 0.0

        results[depth] = {
            "alpha": mean_alpha,
            "speedup": speedup,
            "eesd_time": mean_eesd_time,
            "ar_time": mean_ar_time,
        }
        print(
            f"    Depth {depth}: α={mean_alpha:.3f}, speedup={speedup:.2f}x, "
            f"EESD={mean_eesd_time:.2f}s, AR={mean_ar_time:.2f}s"
        )
        model.draft_depth = original_depth

    return results


# ---------------------------------------------------------------------------
# Morphological Analysis
# ---------------------------------------------------------------------------

def morphological_analysis(
    model: EarlyExitLM,
    tokenizer,
    eval_texts: List[str],
    K: int,
    max_new_tokens: int,
    device: torch.device,
) -> Dict[str, Dict]:
    """
    Stratify acceptance rates by Hindi token category.

    Returns dict: {category: {"accepted": int, "total": int, "alpha": float}}
    """
    category_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"accepted": 0, "total": 0})

    print("\n[Morphological Analysis] Running …")
    for text in eval_texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        draft_ids = model.draft(input_ids, K=K)
        accepted_ids = model.verify(input_ids, draft_ids)

        draft_tokens = tokenizer.convert_ids_to_tokens(draft_ids[0].tolist())
        n_accepted = accepted_ids.size(1)

        for i, token in enumerate(draft_tokens):
            category = classify_token(token)
            category_stats[category]["total"] += 1
            if i < n_accepted:
                category_stats[category]["accepted"] += 1

    results = {}
    for cat, stats in category_stats.items():
        total = stats["total"]
        accepted = stats["accepted"]
        results[cat] = {
            "accepted": accepted,
            "total": total,
            "alpha": accepted / total if total > 0 else 0.0,
        }
        print(f"  {cat:15s}: α={results[cat]['alpha']:.3f}  ({accepted}/{total})")

    return results


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace):
    print(f"\n{'='*60}")
    print("EESD Evaluation")
    print(f"  Model      : {args.model_name}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Exit depths: {args.exit_depths}")
    print(f"  K          : {args.K}")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = EarlyExitLM(
        model_name_or_path=args.model_name,
        exit_depths=args.exit_depths,
        draft_depth=args.exit_depths[0],
    )
    model.load_exit_heads(args.checkpoint)
    model.eval()

    # Load eval texts
    from src.data import XLSumHindiDataset
    ds = XLSumHindiDataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_eval_samples,
        cache_dir=args.cache_dir,
    )
    eval_texts = [ds.texts[i] for i in range(len(ds))]
    print(f"Loaded {len(eval_texts)} evaluation samples from XL-Sum Hindi.\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Experiment 1 – Depth Analysis
    depth_results = depth_analysis(
        model, tokenizer, eval_texts,
        exit_depths=args.exit_depths,
        K=args.K,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    all_results["depth_analysis"] = depth_results

    # Experiment 2 – Morphological Analysis (use best depth)
    best_depth = max(depth_results, key=lambda d: depth_results[d]["alpha"])
    model.draft_depth = best_depth
    print(f"\nUsing depth {best_depth} (best α) for morphological analysis.")
    morph_results = morphological_analysis(
        model, tokenizer, eval_texts,
        K=args.K,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    all_results["morphological_analysis"] = morph_results

    # Save results
    results_path = out_dir / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {results_path}")

    # Pretty-print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nDepth Analysis:")
    for d, r in depth_results.items():
        print(f"  Depth {d:2d}: α={r['alpha']:.3f}, speedup={r['speedup']:.2f}x")
    print("\nMorphological Analysis:")
    for cat, r in morph_results.items():
        print(f"  {cat:15s}: α={r['alpha']:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate EESD for Hindi")
    p.add_argument("--model_name", default="Qwen/Qwen2-1.5B")
    p.add_argument("--checkpoint", default="results/checkpoints/exit_heads_final.pt")
    p.add_argument("--exit_depths", nargs="+", type=int, default=[8, 16, 22])
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_eval_samples", type=int, default=1000)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--cache_dir", default=None)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
