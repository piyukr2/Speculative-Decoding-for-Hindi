# Early-Exit Speculative Decoding for Hindi (EESD)

An empirical study of Early-Exit Speculative Decoding (EESD) on morphologically rich Hindi text generation. This project implements the framework of [Liu et al. (2024)](https://aclanthology.org/2024.findings-acl.195/) adapted for Hindi using the AI4Bharat IndicCorpv2 training corpus and XL-Sum Hindi evaluation set.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Datasets](#datasets)
5. [Training](#training)
6. [Inference](#inference)
7. [Evaluation](#evaluation)
8. [Results & Analysis](#results--analysis)
9. [References](#references)

---

## Overview

Autoregressive LLMs generate tokens one-at-a-time, causing high latency. **Speculative decoding** accelerates this by using a cheap *draft model* to propose candidate tokens, then verifying them in one parallel forward pass of the full model.

**EESD** eliminates the separate draft model by using the LLM's own intermediate representations — adding lightweight *exit heads* (LayerNorm + Linear) at depths {8, 16, 22} of the 28-layer Qwen2-1.5B architecture.

This project investigates how well EESD works for **Hindi**, which poses unique challenges:
- Complex inflectional morphology and Devanagari script
- Subject-Object-Verb (SOV) word order (differs from English SVO)
- Large, dense vocabulary with long compound words

**Research Questions:**
1. What token acceptance rate (α) does EESD achieve for Hindi at each exit depth?
2. Does EESD yield measurable inference speedups over autoregressive decoding?
3. Which Hindi morphological categories (verbs, postpositions, compounds) have the lowest acceptance rates?

---

## Project Structure

```
EESD_v3/
├── EESD.pdf                     # Original research proposal
├── AGENT_PLAN.md                # Project plan & progress tracker
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── default.yaml             # All hyperparameters in one place
│
├── src/
│   ├── __init__.py
│   ├── model.py                 # EarlyExitLM: base model + exit heads
│   ├── data.py                  # AI4Bharat (train) & XL-Sum Hindi (eval)
│   ├── train.py                 # KL distillation training loop
│   ├── inference.py             # EESD + autoregressive generation
│   └── evaluate.py              # Depth analysis + morphological analysis
│
├── kaggle_train.py              # Platform-aware training runner with auto-resume
├── overfit_test.py              # Sanity check: 100 samples, 50 epochs
│
├── notebooks/
│   └── analysis.ipynb           # Plots: depth analysis & morphological analysis
│
├── report.tex                   # ACL-style project report
├── report.pdf                   # Compiled report
│
└── results_final/               # Final evaluation outputs
    ├── evaluation_results.json
    ├── inference_results_{8,16,22}.csv
    └── evaluation.txt
```

---

## Setup

### Local Setup

```bash
cd EESD_v3
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Kaggle Setup

> **Important:** Do NOT run `pip install -r requirements.txt` on Kaggle — it overwrites GPU PyTorch with a CPU version.

```bash
pip install -q pymorphy2 conllu indicnlp bitsandbytes
```

### Verify Installation

```bash
python -c "import torch, transformers; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

---

## Datasets

| Dataset | Purpose | Size | HuggingFace ID |
|---|---|---|---|
| AI4Bharat IndicCorpv2 (Hindi) | Exit-head training | 167K sentences (~5M tokens) | `ai4bharat/IndicCorpv2` / config `indiccorp_v2` / split `hin_Deva` |
| XL-Sum Hindi | Formal evaluation | 500 test samples | `csebuetnlp/xlsum` / config `hindi` |

Both datasets are downloaded automatically on first use.

---

## Training

Exit heads are trained to mimic the full model's output using **soft KL-divergence distillation** ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)):

```
L = (1/|D|) * Σ_{k ∈ {8,16,22}} T² · KL( softmax(f_full/T) ‖ softmax(f_k/T) )
```

where T=2.0 is the distillation temperature. The base LLM is **fully frozen**; only the three exit heads are updated.

### Run Training

```bash
# On Kaggle (recommended):
python kaggle_train.py

# Locally with custom args:
python -m src.train \
    --model_name Qwen/Qwen2-1.5B \
    --exit_depths 8 16 22 \
    --batch_size 2 \
    --epochs 10 \
    --lr 1e-4 \
    --max_samples 167000 \
    --gradient_accumulation_steps 4 \
    --output_dir results/checkpoints
```

**Key hyperparameters:**

| Parameter | Value | Description |
|---|---|---|
| `--model_name` | `Qwen/Qwen2-1.5B` | Base LLM (~1.5B params, frozen) |
| `--exit_depths` | `8 16 22` | Layer indices for exit heads |
| `--batch_size` | `2` | Per-GPU batch size (Kaggle T4) |
| `--gradient_accumulation_steps` | `4` | Effective batch size = 8 |
| `--epochs` | `10` | Training epochs |
| `--lr` | `1e-4` | 8-bit AdamW learning rate |
| `--temperature` | `2.0` | KL distillation temperature |
| `--max_samples` | `167000` | ~5M tokens at 30 tok/sent |

Checkpoints are saved to `results/checkpoints/exit_heads_epoch{N}.pt` after each epoch and `exit_heads_final.pt` at the end. Training auto-resumes from the latest checkpoint across Kaggle sessions.

---

## Inference

### Run EESD Inference

```bash
python -m src.inference \
    --model_name Qwen/Qwen2-1.5B \
    --checkpoint results/checkpoints/exit_heads_final.pt \
    --exit_depths 8 16 22 \
    --draft_depth 8 \
    --K 5 \
    --max_new_tokens 100 \
    --prompt "भारत एक विविध देश है जहाँ कई भाषाएँ बोली जाती हैं।"
```

**EESD inference loop:**

```
for each speculative step:
  1. Run forward pass, hook hidden state at layer d → exit head → K draft tokens
  2. Run full model on [prompt + K draft tokens] in one forward pass
  3. Accept tokens sequentially until first mismatch
  4. Append accepted tokens; repeat until EOS or max_new_tokens
```

| Flag | Default | Description |
|---|---|---|
| `--draft_depth` | `8` | Exit layer to use for drafting |
| `--K` | `5` | Draft tokens per speculative step |
| `--temperature` | `1.0` | Sampling temperature (0 = greedy) |

---

## Evaluation

### Run Full Evaluation

```bash
python -m src.evaluate \
    --model_name Qwen/Qwen2-1.5B \
    --checkpoint results/checkpoints/exit_heads_final.pt \
    --exit_depths 8 16 22 \
    --K 5 \
    --max_eval_samples 500 \
    --output_dir results
```

This runs two experiments:

#### Experiment 1 — Depth Analysis

Sweeps exit depths `{8, 16, 22}` and reports:
- **α** (acceptance rate): fraction of draft tokens accepted by the full model
- **Speedup**: ratio of autoregressive wall-clock time to EESD time

#### Experiment 2 — Morphological Analysis

Stratifies acceptance rates by Hindi token category:
- **Postpositions** (में, को, से, …)
- **Inflected verbs** (tokens ending in -ना, -ता, -ते, -या, …)
- **Compound words** (long Devanagari tokens, ≥5 characters)
- **Other** — numbers, punctuation, rare tokens

Results are saved to `results/evaluation_results.json`.

---

## Results & Analysis

### Training Convergence (10 epochs)

| Metric | Overall | Depth 8 | Depth 16 | Depth 22 |
|---|:---:|:---:|:---:|:---:|
| Train Loss | 0.664 | 0.758 | 0.760 | 0.474 |
| Val Loss   | 0.656 | 0.750 | 0.751 | 0.467 |

Loss decreased from 2.71 (epoch 1) to 0.66 (epoch 10). Depth 22 converges fastest (~0.47), while depths 8 and 16 plateau at ~0.76.

### Large-Scale Inference (101 Hindi Prompts, K=3)

| Exit Depth | Mean α | Mean Speedup | EESD Time (s) | AR Time (s) |
|:---:|:---:|:---:|:---:|:---:|
| 8  | 0.565 ± 0.042 | 0.41× | 8.15  | 3.33 |
| 16 | 0.589 ± 0.043 | 0.41× | 11.53 | 4.65 |
| **22** | **0.646 ± 0.049** | **0.47×** | **7.18** | 3.36 |

### Formal Evaluation (XL-Sum Hindi)

| Exit Depth | α | Speedup | EESD Time (s) | AR Time (s) |
|:---:|:---:|:---:|:---:|:---:|
| 8  | 0.531 | 0.26× | 10.38 | 2.69 |
| 16 | 0.548 | 0.26× | 10.17 | 2.69 |
| **22** | **0.610** | **0.29×** | **9.25** | 2.69 |

> **Note on speedup:** Speedup is <1× because the current implementation runs all 28 layers during drafting (with a hook at the exit layer). True early-exit (running only layers 1–d) would require custom CUDA kernels. The acceptance rate α is the meaningful metric — it measures how well exit heads approximate the full model.

### Key Findings

1. **Depth 22 is the best draft depth**: highest α (0.646), fewest draft tokens needed, and lowest EESD wall time.
2. **Acceptance rate scales with depth**: α increases monotonically from depth 8 → 16 → 22.
3. **Consistent across prompts**: standard deviation of α is small (0.04–0.05).
4. **No overfitting**: validation loss closely tracks training loss throughout all 10 epochs.

---

## References

| Paper | Description |
|---|---|
| Liu et al. (2024a) — [EESD](https://aclanthology.org/2024.findings-acl.195/) | Core EESD method with Thompson Sampling |
| Leviathan et al. (2023) — [Speculative Decoding](https://arxiv.org/abs/2211.17192) | Original speculative decoding |
| Chen et al. (2023) — [Speculative Sampling](https://arxiv.org/abs/2302.01318) | Speculative sampling variant |
| Liu et al. (2024b) — [Kangaroo](https://arxiv.org/abs/2404.02901) | Double early-exiting (NeurIPS 2024) |
| Elhoushi et al. (2024) — [LayerSkip](https://arxiv.org/abs/2404.16710) | Self-speculative decoding via layer skipping |
| Cai et al. (2024) — [Medusa](https://arxiv.org/abs/2401.10774) | Multiple decoding heads |
| Hinton et al. (2015) — [Distillation](https://arxiv.org/abs/1503.02531) | Knowledge distillation |
| Doddapaneni et al. (2023) — [AI4Bharat](https://aclanthology.org/2023.acl-long.693/) | IndicCorp monolingual corpora |
