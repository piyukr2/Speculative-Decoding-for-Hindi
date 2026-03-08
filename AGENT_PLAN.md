# EESD Hindi — Project Plan & Progress Tracker

## Agent Instructions for Any IDE or Environment

This file is the single source of truth for project progress, status, and next steps. To ensure seamless collaboration and agent handoff across VS Code, PyCharm, Jupyter, Kaggle, or any other IDE:

- **Always update this file** after completing any step, regardless of your editor or platform.
- **Mark progress** using `[ ]` (not started), `[/]` (in progress), and `[x]` (done) for each step in the Execution Plan below.
- **Add notes** under any step if you encounter issues, deviations, or special instructions.
- **Paths are relative to the project root**; do not use IDE-specific features or absolute paths.
- **All code, config, and documentation changes must be reflected here**. If you change code, update the plan; if you change the plan, update the code/docs as needed.
- **No IDE-specific actions are required**—all agents and collaborators should follow the same workflow.

This ensures that anyone can pick up the project in any environment and continue seamlessly.

> **Purpose**: This file is the single source of truth for the project's goal, plan, execution steps, and current progress. Any agent or collaborator working on this codebase should read this file first and update it after completing each step.

> [!IMPORTANT]
> **Consistency Rules** — follow these for every code change:
> 1. If a code change affects Kaggle notebook cells, update `KAGGLE_CELLS.md`.
> 2. If a code change affects this plan or its steps, update `AGENT_PLAN.md`.
> 3. If a code change affects user-facing docs, update `README.md`.
> 4. All files must stay consistent with each other after every change.
> 5. If a change does **not** require an update to a file, do **not** touch that file.

---

## Project Goal

Implement and evaluate **Early-Exit Speculative Decoding (EESD)** for **Hindi** text generation. The project adds lightweight exit heads at transformer layers {8, 16, 22} of a frozen **Qwen/Qwen2-1.5B** model, trains them via cross-entropy distillation, and measures:
1. Token acceptance rate (α) at each exit depth
2. Inference speedup over standard autoregressive decoding
3. Acceptance rate stratified by Hindi morphological category (postpositions, verbs, compound words)

**Base model**: `Qwen/Qwen2-1.5B` (~1.5B parameters, frozen)  
**Training objective**: Soft distillation — `T² · KL( softmax(full_logits/T) ‖ softmax(exit_logits/T) )`, T=2.0
**Training data**: AI4Bharat IndicCorpv2 Hindi (~167K sentences, ~5M tokens)
**Evaluation data**: XL-Sum Hindi (500 test samples)  
**Target platform**: Kaggle (free tier, T4/P100 16 GB VRAM)

---

## Codebase Structure

```
EESD_v3/
├── src/
│   ├── model.py        # EarlyExitLM: base model + exit heads
│   ├── data.py         # AI4Bharat (train) & XL-Sum Hindi (eval) loaders
│   ├── train.py        # Distillation training (with checkpoint resumption)
│   ├── inference.py    # EESD + autoregressive generation
│   └── evaluate.py     # Depth analysis + morphological analysis
├── configs/default.yaml
├── scripts/            # Shell launchers (inference, evaluate)
├── kaggle_train.py     # Kaggle-aware training runner with auto-resume
├── overfit_test.py     # Pipeline sanity check (100-sample overfit test)
├── notebooks/
│   └── analysis.ipynb  # Results visualisation
├── AGENT_PLAN.md       # ← THIS FILE
└── results/            # Checkpoints, logs, figures
```

---

## Execution Plan

### Phase 1: Environment Setup
- [x] Create project structure and all source files
- [x] Create `requirements.txt` with all dependencies (includes `bitsandbytes` for 8-bit AdamW)
- [x] Adapt codebase for Kaggle (batch_size=2, grad_accum=4, num_workers=2, checkpoint resumption)
- [x] Upload codebase to Kaggle as dataset `eesd-v3`
- [x] Install dependencies on Kaggle (`pip install -q pymorphy2 conllu indicnlp bitsandbytes`)
- [x] Verify GPU availability — 2× Tesla T4 (15 GB each), model uses GPU 0 only

### Phase 2: Data Preparation
- [ ] Load AI4Bharat IndicCorpv2 Hindi training data (167K samples, streaming)
- [ ] Load XL-Sum Hindi evaluation data (500 test samples)
- [ ] Verify tokenizer works correctly with Devanagari text

### Phase 3: Training (IN PROGRESS — running on Kaggle)
- [ ] **Sanity check**: Run `overfit_test.py` (100 samples, 50 epochs) to verify pipeline
- [/] **Epoch 1**: Train exit heads (distillation loss, ~41K steps at batch_size=2)
  - Checkpoint: `results/checkpoints/exit_heads_epoch1.pt`
- [ ] **Epoch 2**: Continue training (resume if session expired)
  - Checkpoint: `results/checkpoints/exit_heads_epoch2.pt`
- [ ] **Epoch 3**: Final epoch
  - Checkpoint: `results/checkpoints/exit_heads_epoch3.pt`
- [ ] Save final weights: `results/checkpoints/exit_heads_final.pt`

### Phase 4: Inference
- [ ] Run EESD inference demo with Hindi prompt
- [ ] Run autoregressive baseline for comparison
- [ ] Verify acceptance rate and speedup metrics are computed

### Phase 5: Evaluation
- [ ] **Experiment 1 — Depth Analysis**: Sweep exit depths {2, 4, 6}, measure α and speedup
- [ ] **Experiment 2 — Morphological Analysis**: Stratify α by token category (postpositions, verbs, compounds, other)
- [ ] Save results to `results/evaluation_results.json`

### Phase 6: Visualisation & Reporting
- [ ] Generate `results/figures/depth_analysis.png` (bar charts of α and speedup)
- [ ] Generate `results/figures/morphological_analysis.png` (horizontal bar chart by category)
- [ ] Fill in results tables in `README.md`

---

## Key Configuration

| Parameter | Value | Notes |
|---|---|---|
| `--model_name` | `Qwen/Qwen2-1.5B` | ~1.5B params, frozen |
| `--exit_depths` | `2 4 6` | Layers for exit heads |
| `--batch_size` | `2` | For Kaggle T4 (15 GB VRAM) |
| `--gradient_accumulation_steps` | `4` | Effective batch = 2 × 4 = 8 |
| `--epochs` | `3` | Total training epochs |
| `--lr` | `1e-4` | 8-bit AdamW with linear warmup |
| `--temperature` | `2.0` | KL distillation temperature (Hinton et al., 2015) |
| `--max_samples` | `167000` | ~5M tokens |
| `--max_length` | `256` | Max sequence length |

---

## Key Decisions & Learnings

1. **Kaggle input path**: `/kaggle/input/datasets/piyukr/eesd-v3` (not `/kaggle/input/eesd-v3`)
2. **Do NOT run `pip install -r requirements.txt`** on Kaggle — it overwrites GPU PyTorch with CPU version. Install only `pymorphy2 conllu indicnlp bitsandbytes`.
3. **8-bit AdamW required**: 3 exit heads × 233M params = 700M trainable params. Standard AdamW needs 5.6 GB for FP32 momentum buffers → OOM. 8-bit AdamW (bitsandbytes) cuts this to ~1.4 GB.
4. **`total_memory` not `total_mem`**: PyTorch 2.9+ renamed the attribute.
5. **Deleted `scripts/train_exit_head.sh`** in favor of `kaggle_train.py` (platform-aware, auto-resume).
7. **Soft distillation (KL) replaces hard CE**: Hard CE uses `argmax(full_logits)` as the target — throws away all probability mass except the top-1 token. KL divergence uses the full softened distribution as target, giving exit heads a much richer signal. The T² factor (Hinton et al., 2015) restores gradient magnitudes that are otherwise reduced by the 1/T scaling inside `log_softmax`. `overfit_test.py` also prints and saves per-depth acceptance rate after **every** epoch.
6. **`device_map="auto"` splits across both T4s**: Despite Qwen2-1.5B (~3 GB fp16) fitting on one T4, `device_map="auto"` distributes layers across both GPUs on Kaggle. Later layers (and thus their captured hidden states) land on `cuda:1` while exit heads default to `cuda:0` → `RuntimeError: Expected all tensors to be on the same device`. Fixed by using `.to(self.norm.weight)` in `ExitHead.forward()`, which moves the captured hidden state to match both the **device and dtype** of the exit head in one call.

---

## How to Resume Training

If a Kaggle session expires mid-training, the system saves `training_state.json` after each epoch. To resume:

```bash
python3 -m src.train \
    --resume_from results/checkpoints/exit_heads_epoch{N}.pt \
    --start_epoch {N+1}
```

Or simply re-run `kaggle_train.py` which auto-detects the latest checkpoint.

---

## Agent Instructions

1. **Read this file first** before starting any work
2. **Check the execution plan** to find the next uncompleted step (`[ ]`)
3. **Mark steps in-progress** with `[/]` when starting them
4. **Mark steps complete** with `[x]` when done
5. **Add notes** under any step if there were issues or deviations
6. **Update the Results section** below when experiments complete

---

## Results

*(To be filled after experiments)*

### Depth Analysis

| Exit Depth | Acceptance Rate (α) | Speedup |
|:---:|:---:|:---:|
| 2 | — | — |
| 4 | — | — |
| 6 | — | — |

### Morphological Analysis

| Token Category | Acceptance Rate (α) | Count |
|---|:---:|:---:|
| Postpositions | — | — |
| Inflected verbs | — | — |
| Compound words | — | — |
| Other | — | — |
