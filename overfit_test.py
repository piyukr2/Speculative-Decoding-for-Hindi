"""
Overfit Test — Pipeline Sanity Check

Trains the EESD exit heads on exactly 100 samples for many epochs using soft
distillation (KL divergence), then evaluates on the same 100 samples.
If the pipeline is correct, the model should memorise the tiny dataset and show:
  • Training loss ≈ 0 at all exit depths
  • Validation loss ≈ 0  (same data)
  • Acceptance rate ≈ 1.0 at all exit depths

Per-depth loss and acceptance rate are printed and saved after every epoch.

Usage:
    python overfit_test.py                              # defaults
    python overfit_test.py --epochs 100 --num_samples 50
    python overfit_test.py --model_name Qwen/Qwen2-1.5B --batch_size 4 --temperature 2.0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
try:
    get_ipython()  # exists in Jupyter/Kaggle notebooks
except NameError:
    matplotlib.use("Agg")  # headless terminal — no display
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.model import EarlyExitLM
from src.data import AI4BharatHindiDataset
from src.train import distillation_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_overfit_loader(
    tokenizer,
    num_samples: int,
    max_length: int,
    batch_size: int,
    cache_dir: str | None,
) -> DataLoader:
    """Load the full dataset, then take the first `num_samples` rows."""
    full_ds = AI4BharatHindiDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=num_samples,      # stream only what we need
        cache_dir=cache_dir,
    )
    print(f"Dataset loaded: {len(full_ds)} samples (requested {num_samples})")
    return DataLoader(
        full_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,       # small dataset, no need for workers
        pin_memory=True,
        drop_last=False,
    )


@torch.no_grad()
def compute_val_loss(
    model: EarlyExitLM,
    loader: DataLoader,
    exit_depths: list[int],
    device: torch.device,
    temperature: float = 2.0,
) -> tuple:
    """Mean distillation loss (avg and per-depth) over the entire loader."""
    model.exit_heads.eval()
    total_loss = 0.0
    depth_totals = {str(d): 0.0 for d in exit_depths}
    n_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        loss, depth_losses = distillation_loss(out["exit_logits"], out["full_logits"], exit_depths, temperature)
        total_loss += loss.item()
        for d, dl in depth_losses.items():
            depth_totals[d] += dl
        n_batches += 1
    model.exit_heads.train()
    n = max(n_batches, 1)
    return total_loss / n, {d: depth_totals[d] / n for d in depth_totals}


@torch.no_grad()
def compute_acceptance_rate(
    model: EarlyExitLM,
    loader: DataLoader,
    device: torch.device,
    exit_depths: list[int],
    K: int = 5,
) -> dict:
    """
    Compute token acceptance rate at each exit depth (greedy draft).

    Returns:
        dict mapping str(depth) -> {"alpha": float, "accepted": int, "total": int}
    """
    model.exit_heads.eval()
    original_depth = model.draft_depth
    results = {}

    for depth in exit_depths:
        model.draft_depth = depth
        total_drafted = 0
        total_accepted = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            for i in range(input_ids.size(0)):
                ids = input_ids[i : i + 1]          # [1, T]
                draft_ids    = model.draft(ids, K=K, temperature=0.0)
                accepted_ids = model.verify(ids, draft_ids)
                total_drafted  += draft_ids.size(1)
                total_accepted += accepted_ids.size(1)
        results[str(depth)] = {
            "alpha":    total_accepted / max(total_drafted, 1),
            "accepted": total_accepted,
            "total":    total_drafted,
        }

    model.draft_depth = original_depth
    model.exit_heads.train()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    print(f"\n{'=' * 60}")
    print("EESD Overfit Test — Pipeline Sanity Check")
    print(f"  Model       : {args.model_name}")
    print(f"  Exit depths : {args.exit_depths}")
    print(f"  Num samples : {args.num_samples}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Draft K     : {args.K}")
    print(f"  Draft depth : {args.draft_depth}")
    print(f"  Temperature : {args.temperature}")
    print(f"{'=' * 60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Tokenizer ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ─────────────────────────────────────────────────────────
    model = EarlyExitLM(
        model_name_or_path=args.model_name,
        exit_depths=args.exit_depths,
        draft_depth=args.draft_depth,
    )
    model.exit_heads = model.exit_heads.to(device)

    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Loading exit heads from {args.resume_from} …")
        model.load_exit_heads(args.resume_from)

    # ── Data (same loader for train / val / test) ─────────────────────
    loader = build_overfit_loader(
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )

    # ── Optimizer (8-bit AdamW to fit in T4 VRAM) ─────────────────────
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.trainable_parameters(), lr=args.lr, weight_decay=0.01
        )
        print("Using 8-bit AdamW (saves ~4 GB optimizer memory)")
    except ImportError:
        from torch.optim import AdamW
        print("WARNING: bitsandbytes not found, using standard AdamW (may OOM)")
        optimizer = AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=0.01)

    # ── Output dir ────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint_100_overfit.pt"
    loss_hist_path = out_dir / "overfit_loss_history.json"

    # Load existing loss history (for resume)
    if loss_hist_path.exists():
        with open(loss_hist_path) as f:
            loss_history = json.load(f)
    else:
        loss_history = []

    # ─────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────
    print("\n── Training ──\n")
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.exit_heads.train()
        epoch_loss = 0.0
        epoch_depth_losses = {str(d): 0.0 for d in args.exit_depths}

        pbar = tqdm(
            enumerate(loader, 1),
            total=len(loader),
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="step",
            dynamic_ncols=True,
            leave=False,
        )
        for step, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss, depth_losses = distillation_loss(
                exit_logits=out["exit_logits"],
                full_logits=out["full_logits"],
                exit_depths=args.exit_depths,
                temperature=args.temperature,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            step_loss = loss.item()
            del out, loss  # free activation memory before optimizer step
            torch.cuda.empty_cache()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += step_loss
            for d, dl in depth_losses.items():
                epoch_depth_losses[d] += dl
            postfix = {f"d{d}": f"{epoch_depth_losses[d]/step:.3f}"
                       for d in sorted(epoch_depth_losses, key=int)}
            pbar.set_postfix(postfix)

        avg_train_loss = epoch_loss / len(loader)
        avg_depth_losses = {d: epoch_depth_losses[d] / len(loader) for d in epoch_depth_losses}

        # Validation loss (same data, no grad)
        val_loss, val_depth_losses = compute_val_loss(model, loader, args.exit_depths, device, args.temperature)

        # Acceptance rate per depth (greedy, same 100 samples)
        ar_results = compute_acceptance_rate(model, loader, device, exit_depths=args.exit_depths, K=args.K)

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss

        # Save loss history and plot
        loss_history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_depth_losses": {d: round(avg_depth_losses[d], 6) for d in avg_depth_losses},
            "val_depth_losses":   {d: round(val_depth_losses[d], 6) for d in val_depth_losses},
            "acceptance_rates":   {str(d): round(ar_results[str(d)]["alpha"], 4) for d in args.exit_depths},
        })
        with open(loss_hist_path, "w") as f:
            json.dump(loss_history, f, indent=2)

        # Plot per-depth loss curves
        ep_list    = [h["epoch"] for h in loss_history]
        depth_keys = sorted(loss_history[0].get("train_depth_losses", {}).keys(), key=int)
        colors     = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, dk in enumerate(depth_keys):
            c = colors[i % len(colors)]
            ax.plot(ep_list, [h["train_depth_losses"][dk] for h in loss_history],
                    "o-",  label=f"Train d{dk}", color=c, linewidth=2)
            ax.plot(ep_list, [h["val_depth_losses"][dk] for h in loss_history],
                    "s--", label=f"Val d{dk}",   color=c, linewidth=1.5, alpha=0.75)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("KL Loss")
        ax.set_title("Overfit Test — KL Loss per Exit Depth")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = fig_dir / "overfit_loss_curve.png"
        fig.savefig(plot_path, dpi=120)
        plt.show()
        plt.close(fig)

        depth_str = "  ".join(f"d{d}={avg_depth_losses[d]:.6f}" for d in sorted(avg_depth_losses, key=int))
        val_str   = "  ".join(f"d{d}={val_depth_losses[d]:.6f}" for d in sorted(val_depth_losses, key=int))
        alpha_str = "  ".join(f"d{d}={ar_results[str(d)]['alpha']:.4f}" for d in sorted(args.exit_depths))
        print(
            f"Epoch {epoch:>4d}/{args.epochs} │ "
            f"Train: {avg_train_loss:.6f}  {depth_str}\n"
            f"             │ Val:   {val_loss:.6f}  {val_str}\n"
            f"             │ α:     {alpha_str}"
        )

        # Save checkpoint after every epoch (overwrites previous)
        model.save_exit_heads(str(ckpt_path))

    print(f"\nFinal checkpoint saved → {ckpt_path}")

    # ─────────────────────────────────────────────────────────────────
    # Evaluation — acceptance rate on the same 100 samples
    # ─────────────────────────────────────────────────────────────────
    print("\n── Evaluation ──\n")

    # Re-compute final train & val loss (should match)
    final_train_loss, final_depth_losses = compute_val_loss(model, loader, args.exit_depths, device, args.temperature)
    final_val_loss = final_train_loss  # same data

    # Acceptance rate per depth
    ar_results = compute_acceptance_rate(model, loader, device, exit_depths=args.exit_depths, K=args.K)

    # ─────────────────────────────────────────────────────────────────
    # Report
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("OVERFIT TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  Samples          : {args.num_samples}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  Final train loss : {final_train_loss:.6f}")
    for d in sorted(final_depth_losses, key=int):
        print(f"    Depth {d:>2s}      : {final_depth_losses[d]:.6f}")
    print(f"  Final val loss   : {final_val_loss:.6f}")
    print(f"  Acceptance rates :")
    for d in sorted(ar_results, key=int):
        r = ar_results[d]
        print(f"    Depth {d:>2s}      : α={r['alpha']:.4f}  ({r['accepted']}/{r['total']} tokens)")
    print(f"{'=' * 60}")

    # Verdict
    mean_alpha = sum(r["alpha"] for r in ar_results.values()) / len(ar_results)
    loss_ok  = final_train_loss < 0.1
    alpha_ok = mean_alpha >= 0.5

    if loss_ok and alpha_ok:
        print("\n✅ OVERFIT TEST PASSED")
        print("   The training pipeline can memorise a small dataset and")
        print("   the exit heads produce high-acceptance drafts.")
    elif loss_ok and not alpha_ok:
        print("\n⚠️  OVERFIT TEST PARTIAL PASS")
        print(f"   Loss is near zero ({final_train_loss:.6f}) — training works.")
        print(f"   But mean acceptance rate is low ({mean_alpha:.4f}) — draft/verify may have a bug.")
    else:
        print("\n❌ OVERFIT TEST FAILED")
        print(f"   Train loss did not converge ({final_train_loss:.6f}).")
        print("   The training pipeline likely has a bug.")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overfit test: verify EESD pipeline by memorising 100 samples"
    )
    p.add_argument("--model_name", default="Qwen/Qwen2-1.5B")
    p.add_argument("--exit_depths", nargs="+", type=int, default=[8, 16, 22])
    p.add_argument("--draft_depth", type=int, default=8)
    p.add_argument("--num_samples", type=int, default=100,
                   help="Number of samples to overfit on (default: 100)")
    p.add_argument("--epochs", type=int, default=50,
                   help="Number of training epochs (default: 50)")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=2.0,
                   help="Distillation temperature for soft KL targets (default: 2.0)")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--K", type=int, default=5,
                   help="Draft length for acceptance rate evaluation (default: 5)")
    p.add_argument("--output_dir", default="results/checkpoints")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--resume_from", default=None,
                   help="Path to checkpoint to resume training from")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
