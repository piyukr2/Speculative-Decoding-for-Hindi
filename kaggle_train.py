"""
Training Runner for EESD Hindi.

Supports Kaggle, Google Colab, and local environments.
Designed to be run as:
    !python kaggle_train.py

Or imported and called from a notebook cell. It handles:
  - Platform-specific paths and settings
  - Automatic checkpoint resumption across sessions
  - GPU detection and VRAM reporting
  - H100/A100 optimizations (bf16, tf32, larger batches, cosine schedule)

Place this file in the root of the EESD_v3 project directory.
"""

import os
import sys
import json
from pathlib import Path

# ── Ensure CWD is the project root (needed for `python -m src.train`) ─
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# ── Kaggle / Colab environment detection ──────────────────────────────

def detect_platform() -> str:
    """Detect whether we are running on Kaggle, Colab, or local."""
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass
    return "local"


PLATFORM = detect_platform()
print(f"Detected platform: {PLATFORM}")

# ── Platform-specific defaults ────────────────────────────────────────

DEFAULTS = {
    "kaggle": {
        "batch_size": 2,
        "num_workers": 2,
        "max_samples": 167_000,
        "gradient_accumulation_steps": 4,
        "lr": "1e-4",
        "epochs": 3,
        "input_dir": "/kaggle/input/datasets/kspsvln/epoch-2",
        "output_dir": "results/checkpoints",
        "cache_dir": "/kaggle/working/cache",
    },
    "colab": {
        "batch_size": 8,
        "num_workers": 4,
        "max_samples": 167_000,
        "gradient_accumulation_steps": 2,
        "lr": "2e-4",
        "epochs": 10,
        "input_dir": "/content/drive/MyDrive/INLP26/",
        "output_dir": "/content/drive/MyDrive/INLP26/results/checkpoints",
        "cache_dir": "/content/drive/MyDrive/INLP26/cache",
    },
    "local": {
        "batch_size": 1,
        "num_workers": 2,
        "max_samples": 167_000,
        "gradient_accumulation_steps": 1,
        "lr": "1e-4",
        "epochs": 3,
        "input_dir": "results/checkpoints",
        "output_dir": "results/checkpoints",
        "cache_dir": None,
    },
}

cfg = DEFAULTS[PLATFORM]

# ── Check for existing checkpoint to resume from ─────────────────────

def find_resume_checkpoint(input_dir: str, output_dir: str):
    """Look for the latest completed epoch checkpoint and training state."""
    # Check output_dir first (in-progress training), then input_dir (uploaded checkpoint)
    for search_dir in [output_dir, input_dir]:
        state_path = Path(search_dir) / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            ckpt = state.get("checkpoint")
            epoch = state.get("completed_epoch", 0)
            if ckpt and Path(ckpt).exists():
                print(f"Found checkpoint: {ckpt} (completed epoch {epoch})")
                return ckpt, epoch + 1  # resume from next epoch
    # Also look for exit_heads_final.pt or exit_heads_epoch*.pt directly
    for search_dir in [output_dir, input_dir]:
        d = Path(search_dir)
        if (d / "exit_heads_final.pt").exists():
            print(f"Found final checkpoint in {search_dir}")
            return str(d / "exit_heads_final.pt"), 4  # assume epoch3 done
        epoch_ckpts = sorted(d.glob("exit_heads_epoch*.pt"))
        if epoch_ckpts:
            last = epoch_ckpts[-1]
            # Extract epoch number from filename
            try:
                ep = int(last.stem.replace("exit_heads_epoch", ""))
            except ValueError:
                ep = 3
            print(f"Found checkpoint: {last} (epoch {ep})")
            return str(last), ep + 1
    return None, 1


resume_from, start_epoch = find_resume_checkpoint(cfg["input_dir"], cfg["output_dir"])

if resume_from:
    print(f"Will resume from epoch {start_epoch}")
else:
    print("Starting fresh training (no checkpoint found)")

# ── Build the training command ────────────────────────────────────────

cmd_parts = [
    sys.executable, "-m", "src.train",
    "--model_name", "Qwen/Qwen2-1.5B",
    "--exit_depths", "8", "16", "22",
    "--batch_size", str(cfg["batch_size"]),
    "--epochs", str(cfg["epochs"]),
    "--lr", cfg["lr"],
    "--max_length", "256",
    "--max_samples", str(cfg["max_samples"]),
    "--num_workers", str(cfg["num_workers"]),
    "--output_dir", cfg["output_dir"],
    "--log_every", "100",
    "--gradient_accumulation_steps", str(cfg["gradient_accumulation_steps"]),
    "--platform", PLATFORM,
]

if cfg["cache_dir"]:
    cmd_parts += ["--cache_dir", cfg["cache_dir"]]

if resume_from:
    cmd_parts += ["--resume_from", resume_from, "--start_epoch", str(start_epoch)]

# ── Run training ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    cmd = " ".join(cmd_parts)
    print(f"\nRunning: {cmd}\n")
    # PYTHONUNBUFFERED=1 ensures print/tqdm output appears in real time
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd_parts, check=True, env=env)
