"""
Training script for bottleneck exit heads.

Identical to src/train.py except:
  - Uses BottleneckExitHead (LayerNorm → Linear(1536,256) → GELU → Linear(256,V))
    instead of ExitHead (LayerNorm → Linear(1536,V))
  - 5 epochs (instead of 3)
  - Checkpoints: bottleneck_exit_heads_epoch{N}.pt

Everything else is the same: KL distillation, 8-bit AdamW, T=2.0.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.model import EarlyExitLM, BottleneckExitHead
from src.data import get_train_dataloader
from src.train import distillation_loss, _probe_acceptance_rate, _save_loss_history, _plot_loss_curve


def _replace_exit_heads_with_bottleneck(model: EarlyExitLM) -> None:
    """Replace the default ExitHead modules with BottleneckExitHead."""
    hidden_size = model.base_model.config.hidden_size
    vocab_size = model.base_model.config.vocab_size
    model.exit_heads = nn.ModuleDict(
        {str(d): BottleneckExitHead(hidden_size, vocab_size) for d in model.exit_depths}
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    print(f"\n{'='*60}")
    print("Bottleneck Exit Heads – Training")
    print(f"  Model      : {args.model_name}")
    print(f"  Exit depths: {args.exit_depths}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Grad accum : {args.gradient_accumulation_steps}")
    print(f"  Effective  : {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  LR         : {args.lr}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Platform   : {args.platform}")
    if args.resume_from:
        print(f"  Resume from: {args.resume_from} (epoch {args.start_epoch})")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    use_bf16 = False
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {gpu_mem:.1f} GB")
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            use_bf16 = True
            print("Enabled TF32 matmul + cuDNN and bf16 autocast (Ampere+ GPU)")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model — create with default ExitHead, then swap in BottleneckExitHead
    model = EarlyExitLM(
        model_name_or_path=args.model_name,
        exit_depths=args.exit_depths,
        load_in_8bit=getattr(args, 'load_in_8bit', False),
    )
    _replace_exit_heads_with_bottleneck(model)
    model.exit_heads = model.exit_heads.to(device)

    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Loading bottleneck exit heads from {args.resume_from} …")
        model.load_exit_heads(args.resume_from)

    # Dataloader
    train_loader = get_train_dataloader(
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        pin_memory=(device.type == "cuda"),
    )

    # Optimizer (only exit heads) — use 8-bit AdamW to save ~4 GB VRAM
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.trainable_parameters(), lr=args.lr, weight_decay=0.01
        )
        print("Using 8-bit AdamW (saves ~4 GB optimizer memory)")
    except ImportError:
        print("WARNING: bitsandbytes not found, using standard AdamW (may OOM)")
        optimizer = AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=0.01)

    remaining_epochs = args.epochs - args.start_epoch + 1
    total_steps = len(train_loader) * remaining_epochs
    warmup_steps = max(1, total_steps // 20)

    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    def _warmup_lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_lr_lambda)

    scaler = torch.amp.GradScaler(enabled=use_bf16)

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load existing loss history (for resume)
    loss_hist_path = out_dir / "bottleneck_loss_history.json"
    if loss_hist_path.exists():
        with open(loss_hist_path) as f:
            loss_history = json.load(f)
        loss_history = [h for h in loss_history if h["epoch"] < args.start_epoch]
    else:
        loss_history = []

    grad_accum = args.gradient_accumulation_steps

    # ---- Training loop ----
    global_step = 0
    for epoch in range(args.start_epoch, args.epochs + 1):
        model.exit_heads.train()
        epoch_loss = 0.0
        epoch_depth_losses = {str(d): 0.0 for d in args.exit_depths}
        t0 = time.time()
        optimizer.zero_grad()
        print(f"Starting epoch {epoch} training loop ({len(train_loader)} steps) …")

        pbar = tqdm(
            enumerate(train_loader, 1),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="step",
            dynamic_ncols=True,
        )
        for step, batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            if step == 1:
                print("First batch loaded, running forward pass …")

            amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_bf16):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                loss, depth_losses = distillation_loss(
                    exit_logits=out["exit_logits"],
                    full_logits=out["full_logits"],
                    exit_depths=args.exit_depths,
                    temperature=args.temperature,
                )
            scaled_loss = loss / grad_accum
            scaler.scale(scaled_loss).backward()
            step_loss = loss.item()

            if step % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                del out, loss, scaled_loss
                scaler.step(optimizer)
                scaler.update()
                if global_step < warmup_steps:
                    warmup_scheduler.step()
                else:
                    scheduler.step()
                optimizer.zero_grad()

            epoch_loss += step_loss
            for d, dl in depth_losses.items():
                epoch_depth_losses[d] += dl
            global_step += 1

            current_lr = scheduler.get_last_lr()[0]
            avg_loss_so_far = epoch_loss / step
            postfix = {"loss": f"{avg_loss_so_far:.4f}"}
            postfix.update({f"d{d}": f"{epoch_depth_losses[d]/step:.3f}"
                       for d in sorted(epoch_depth_losses, key=int)})
            postfix["lr"] = f"{current_lr:.2e}"
            pbar.set_postfix(postfix)

        # Compute validation loss
        model.exit_heads.eval()
        val_loss_total = 0.0
        val_depth_totals = {str(d): 0.0 for d in args.exit_depths}
        val_batches = 0
        with torch.no_grad():
            for vb in train_loader:
                vids = vb["input_ids"].to(device)
                vmask = vb["attention_mask"].to(device)
                vout = model(input_ids=vids, attention_mask=vmask)
                vl, vdepth = distillation_loss(vout["exit_logits"], vout["full_logits"], args.exit_depths, args.temperature)
                val_loss_total += vl.item()
                for d, dl in vdepth.items():
                    val_depth_totals[d] += dl
                val_batches += 1
                if val_batches >= 50:
                    break
        val_loss = val_loss_total / max(val_batches, 1)
        avg_val_depth_losses = {d: val_depth_totals[d] / max(val_batches, 1) for d in val_depth_totals}
        model.exit_heads.train()

        # Probe acceptance rate
        model.exit_heads.eval()
        alpha_results = _probe_acceptance_rate(model, tokenizer, device, n_samples=5, K=5)
        model.exit_heads.train()

        # Save checkpoint
        ckpt_path = out_dir / f"bottleneck_exit_heads_epoch{epoch}.pt"
        model.save_exit_heads(str(ckpt_path))
        avg_loss = epoch_loss / len(train_loader)
        avg_depth_losses = {d: epoch_depth_losses[d] / len(train_loader) for d in epoch_depth_losses}
        depth_str = "  ".join(f"d{d}={avg_depth_losses[d]:.4f}" for d in sorted(avg_depth_losses, key=int))
        val_str   = "  ".join(f"d{d}={avg_val_depth_losses[d]:.4f}" for d in sorted(avg_val_depth_losses, key=int))
        alpha_str = "  ".join(f"d{d}=α{alpha_results.get(d, 0.0):.3f}" for d in sorted(alpha_results, key=int))
        print(f"\n[Epoch {epoch}] Train: {avg_loss:.4f}  {depth_str}")
        print(f"           Val:   {val_loss:.4f}  {val_str}")
        print(f"           Alpha: {alpha_str}")
        print(f"Checkpoint saved → {ckpt_path}")

        # Save loss history and plot
        loss_history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_depth_losses": {d: round(avg_depth_losses[d], 6) for d in avg_depth_losses},
            "val_depth_losses":   {d: round(avg_val_depth_losses[d], 6) for d in avg_val_depth_losses},
            "acceptance_rate":    alpha_results,
        })
        _save_loss_history(loss_history, loss_hist_path)
        plot_path = fig_dir / "bottleneck_loss_curve.png"
        _plot_loss_curve(loss_history, plot_path)
        print(f"Loss history → {loss_hist_path}")
        print(f"Loss curve   → {plot_path}")

        # Save training state for resumption
        state_path = out_dir / "bottleneck_training_state.json"
        with open(state_path, "w") as f:
            json.dump({
                "completed_epoch": epoch,
                "train_loss": avg_loss,
                "train_depth_losses": avg_depth_losses,
                "val_loss": val_loss,
                "val_depth_losses": avg_val_depth_losses,
                "global_step": global_step,
                "checkpoint": str(ckpt_path),
            }, f, indent=2)
        print(f"Training state saved → {state_path}\n")

    # Save final
    final_path = out_dir / "bottleneck_exit_heads_final.pt"
    model.save_exit_heads(str(final_path))
    print(f"Training complete. Final checkpoint → {final_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train bottleneck exit heads for Hindi")
    p.add_argument("--model_name", default="Qwen/Qwen2-1.5B")
    p.add_argument("--exit_depths", nargs="+", type=int, default=[8, 16, 22])
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_samples", type=int, default=167_000)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--output_dir", default="results/checkpoints")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--input_dir", default=None)
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--resume_from", default=None)
    p.add_argument("--start_epoch", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--platform", choices=["local", "kaggle", "colab"], default="local")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
