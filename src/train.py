"""
Training script: distillation-based supervision for early-exit heads.

Objective — soft distillation via KL divergence (Hinton et al., 2015):
    L = (1/K) Σ_{k ∈ depths}  T² · KL( softmax(f_full/T) ‖ softmax(f_k/T) )

where f_full are the frozen full-model logits, f_k are the exit-head logits,
and T is the distillation temperature (default 2.0).  Soft targets transfer
the full probability distribution — not just the argmax — from teacher to
student, giving the exit heads a much richer training signal.

Only the exit-head parameters are updated; the base model stays frozen.

Supports:
  - Checkpoint resumption (--resume_from) for Kaggle 9-hour session limits
  - Gradient accumulation (--gradient_accumulation_steps) for larger effective batch sizes
  - Platform-aware defaults (--platform kaggle|colab|local)

Usage:
    python -m src.train \
        --model_name Qwen/Qwen2-1.5B \
        --exit_depths 2 4 6 \
        --batch_size 4 \
        --epochs 3 \
        --lr 1e-4 \
        --output_dir results/checkpoints

    # Resume from checkpoint (after Kaggle session restart):
    python -m src.train \
        --resume_from results/checkpoints/exit_heads_epoch1.pt \
        --start_epoch 2
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.model import EarlyExitLM
from src.data import get_train_dataloader


# ---------------------------------------------------------------------------
# Quick acceptance-rate probe (runs a few samples through draft → verify)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _probe_acceptance_rate(
    model: EarlyExitLM,
    tokenizer,
    device: torch.device,
    n_samples: int = 5,
    K: int = 5,
) -> dict:
    """Run draft→verify on a few fixed prompts, return per-depth α."""
    prompts = [
        "भारत एक विविध देश है जहाँ",
        "आज के समय में प्रौद्योगिकी",
        "हिंदी भाषा का महत्व",
        "शिक्षा हर व्यक्ति का",
        "भारतीय संस्कृति में",
    ][:n_samples]
    results = {}
    for depth in model.exit_depths:
        old_depth = model.draft_depth
        model.draft_depth = depth
        total_draft = 0
        total_accepted = 0
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            draft_ids = model.draft(ids, K=K)
            accepted_ids = model.verify(ids, draft_ids)
            total_draft += draft_ids.size(1)
            total_accepted += accepted_ids.size(1)
        alpha = total_accepted / max(total_draft, 1)
        results[str(depth)] = round(alpha, 3)
        model.draft_depth = old_depth
    return results


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def distillation_loss(
    exit_logits: dict,
    full_logits: torch.Tensor,
    exit_depths: list,
    temperature: float = 2.0,
) -> tuple:
    """
    Soft distillation loss: KL divergence between each exit head's output
    distribution and the frozen full model's output distribution.

    Using soft targets instead of hard argmax labels transfers richer
    information — the full probability distribution encodes which alternative
    tokens are plausible, not just the single most likely one.

        L = T² · KL( softmax(full_logits/T) ‖ softmax(exit_logits/T) )

    The T² factor (Hinton et al., 2015) compensates for the reduced gradient
    magnitude at higher temperatures so that the effective learning rate stays
    consistent across temperature choices.

    Args:
        exit_logits : dict mapping str(depth) -> [B, T, V]
        full_logits : [B, T, V] full-model logits (no-grad)
        exit_depths : list of int depths
        temperature : softmax temperature; higher = softer targets (default 2.0)

    Returns:
        (avg_loss, depth_losses)
        avg_loss    : scalar tensor averaged over all depths — use for backprop
        depth_losses: dict mapping str(depth) -> float — use for logging
    """
    with torch.no_grad():
        # Soft teacher distribution — cast to float32 for numerical stability
        teacher_probs = F.softmax(full_logits.float() / temperature, dim=-1)  # [B, T, V]

    depth_losses: dict = {}
    total_loss = torch.tensor(0.0, device=full_logits.device)
    for depth in exit_depths:
        logits = exit_logits[str(depth)]  # [B, T, V]
        student_log_probs = F.log_softmax(logits.float() / temperature, dim=-1)  # [B, T, V]
        # KL(teacher ‖ student), averaged over tokens; multiply by T² to restore gradient scale
        loss = F.kl_div(
            student_log_probs.view(-1, student_log_probs.size(-1)),
            teacher_probs.view(-1, teacher_probs.size(-1)),
            reduction="batchmean",
        ) * (temperature ** 2)
        depth_losses[str(depth)] = loss.item()
        total_loss = total_loss + loss

    return total_loss / len(exit_depths), depth_losses


# ---------------------------------------------------------------------------
# Loss tracking helpers
# ---------------------------------------------------------------------------

def _save_loss_history(history: list[dict], path: Path) -> None:
    """Append-safe save of loss history to JSON."""
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def _plot_loss_curve(history: list[dict], path: Path) -> None:
    """Overwrite a train/val loss curve PNG with one line per exit depth."""
    epochs     = [h["epoch"] for h in history]
    depth_keys = sorted(
        history[0].get("train_depth_losses", {}).keys(), key=int
    ) if history and history[0].get("train_depth_losses") else []
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    fig, ax = plt.subplots(figsize=(9, 5))
    if depth_keys:
        for i, dk in enumerate(depth_keys):
            c = colors[i % len(colors)]
            ax.plot(epochs,
                    [h["train_depth_losses"][dk] for h in history],
                    "o-",  label=f"Train d{dk}", color=c, linewidth=2)
            if "val_depth_losses" in history[0]:
                ax.plot(epochs,
                        [h["val_depth_losses"][dk] for h in history],
                        "s--", label=f"Val d{dk}",   color=c, linewidth=1.5, alpha=0.75)
    else:
        # Fallback for history entries without per-depth data
        ax.plot(epochs, [h["train_loss"] for h in history], "o-",  label="Train loss", linewidth=2)
        ax.plot(epochs, [h.get("val_loss", h["train_loss"]) for h in history],
                "s--", label="Val loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss per Exit Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    print(f"\n{'='*60}")
    print("Early-Exit Speculative Decoding – Training")
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
        # Enable TF32 for Ampere+ GPUs (H100, A100, RTX 30xx/40xx)
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 8:  # Ampere or newer
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            use_bf16 = True
            print("Enabled TF32 matmul + cuDNN and bf16 autocast (Ampere+ GPU)")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = EarlyExitLM(
        model_name_or_path=args.model_name,
        exit_depths=args.exit_depths,
        load_in_8bit=getattr(args, 'load_in_8bit', False),
    )
    # Move exit heads to device (base model already placed via device_map="auto")
    model.exit_heads = model.exit_heads.to(device)

    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Loading exit heads from {args.resume_from} …")
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
    # Remaining steps when resuming
    remaining_epochs = args.epochs - args.start_epoch + 1
    total_steps = len(train_loader) * remaining_epochs
    warmup_steps = max(1, total_steps // 20)  # 5% warmup

    # Cosine annealing for better convergence on longer runs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    # Warmup wrapper: linearly ramp LR for the first warmup_steps
    _base_lr = args.lr
    def _warmup_lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_lr_lambda)

    # GradScaler for mixed precision
    scaler = torch.amp.GradScaler(enabled=use_bf16)

    # NOTE: torch.compile removed — the 1-2 min JIT warmup per epoch isn't
    # worth it for only 7 epochs on A100.  TF32 + bf16 already give ~90% of
    # the possible speedup.

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load existing loss history (for resume)
    loss_hist_path = out_dir / "loss_history.json"
    if loss_hist_path.exists():
        with open(loss_hist_path) as f:
            loss_history = json.load(f)
        # Remove entries for epochs we're about to retrain
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
            # Scale loss for gradient accumulation
            scaled_loss = loss / grad_accum
            scaler.scale(scaled_loss).backward()
            step_loss = loss.item()  # save before potential del

            if step % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                del out, loss, scaled_loss  # free activation memory before optimizer step
                scaler.step(optimizer)
                scaler.update()
                # Step both schedulers
                if global_step < warmup_steps:
                    warmup_scheduler.step()
                else:
                    scheduler.step()
                optimizer.zero_grad()

            epoch_loss += step_loss  # log unscaled loss
            for d, dl in depth_losses.items():
                epoch_depth_losses[d] += dl
            global_step += 1

            # Update progress bar postfix (per-depth running averages + loss)
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
                if val_batches >= 50:  # cap at 50 batches to save time
                    break
        val_loss = val_loss_total / max(val_batches, 1)
        avg_val_depth_losses = {d: val_depth_totals[d] / max(val_batches, 1) for d in val_depth_totals}
        model.exit_heads.train()

        # Probe acceptance rate (quick draft→verify on a few prompts)
        model.exit_heads.eval()
        alpha_results = _probe_acceptance_rate(model, tokenizer, device, n_samples=5, K=5)
        model.exit_heads.train()

        # Save checkpoint after each epoch
        ckpt_path = out_dir / f"exit_heads_epoch{epoch}.pt"
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

        # Save loss history (incremental) and plot
        loss_history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_depth_losses": {d: round(avg_depth_losses[d], 6) for d in avg_depth_losses},
            "val_depth_losses":   {d: round(avg_val_depth_losses[d], 6) for d in avg_val_depth_losses},
            "acceptance_rate":    alpha_results,
        })
        _save_loss_history(loss_history, loss_hist_path)
        plot_path = fig_dir / "loss_curve.png"
        _plot_loss_curve(loss_history, plot_path)
        print(f"Loss history → {loss_hist_path}")
        print(f"Loss curve   → {plot_path}")

        # Save training state for resumption
        state_path = out_dir / "training_state.json"
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
    final_path = out_dir / "exit_heads_final.pt"
    model.save_exit_heads(str(final_path))
    print(f"Training complete. Final checkpoint → {final_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EESD exit heads for Hindi")
    p.add_argument("--model_name", default="Qwen/Qwen2-1.5B",
                   help="HuggingFace model name or local path (default: Qwen/Qwen2-1.5B)")
    p.add_argument("--exit_depths", nargs="+", type=int, default=[8, 16, 22],
                   help="Layer depths for exit heads (default: 2 4 6)")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=2.0,
                   help="Distillation temperature for soft KL targets (default: 2.0)")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_samples", type=int, default=167_000,
                   help="Max training samples (~5M tokens at ~30 tok/sent)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--output_dir", default="results/checkpoints")
    p.add_argument("--cache_dir", default=None,
                   help="HuggingFace datasets cache directory")
    p.add_argument("--input_dir", default=None,
                   help="Directory containing checkpoint to resume from")
    p.add_argument("--load_in_8bit", action="store_true",
                   help="Load model in 8-bit mode to reduce memory usage")
    # Checkpoint resumption (for Kaggle / Colab session limits)
    p.add_argument("--resume_from", default=None,
                   help="Path to exit-head checkpoint to resume training from")
    p.add_argument("--start_epoch", type=int, default=1,
                   help="Epoch to start from when resuming (default: 1)")
    # Gradient accumulation
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="Number of gradient accumulation steps (default: 1)")
    # Platform hint
    p.add_argument("--platform", choices=["local", "kaggle", "colab"], default="local",
                   help="Target platform for auto-tuned defaults")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
