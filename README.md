# Early-Exit Speculative Decoding for Hindi (EESD)

Comprehensive evaluation of Early-Exit Speculative Decoding for Hindi using Qwen2-1.5B.
This repository includes training, inference, unified multi-method benchmarking, visualization notebooks, and ACL-style report artifacts.

- GitHub: https://github.com/piyukr2/Speculative-Decoding-for-Hindi
- Exit-head checkpoints: https://huggingface.co/piyukr/EESD
- Colab notebook (train/evaluate/inference): https://colab.research.google.com/drive/1lWGMBimmVMaQ-RQI9R8hEys3JtVxEMgG?usp=drive_link
- Google Drive (source files, models): https://drive.google.com/drive/folders/1NF3m1lP8Xn9u3rXBdm-T0FYSSVC_S6Wd?usp=drive_link

## Highlights

- 12 decoding methods evaluated under one pipeline (AR, draft-model baseline, hook/true-exit, Thompson, UCB, weighted Thompson, bottleneck variants).
- Best acceptance (101 prompts): BN-TrueExit at 70.9%.
- Best speedup and throughput (101 prompts): Thomp-BN at 1.130x and 30.40 tok/s.
- Robustness on 500 prompts: Thomp-BN remains above unity speedup at 1.041x.
- Final report artifacts are exported in results_final/plots and results_final/tables.

## Repository Layout

```text
EESD_v3/
├── src/
│   ├── model.py                  # EarlyExitLM, bottleneck heads, controllers
│   ├── train.py                  # Exit-head distillation training
│   ├── inference.py              # EESD decoding variants
│   ├── evaluate.py               # Legacy depth + morphology evaluation
│   └── evaluate_all.py           # Unified 12-method evaluation
├── visual_analysis.ipynb         # Final analysis and plot/table export notebook
├── report.tex                    # Main report source
├── Execution_notebook.ipynb      # notebook which calls the functions to perform training, evaluation, inference
├── results_final/
│   ├── plots/                    # Final figures for report
│   └── tables/                   # Final CSV tables
└── requirements.txt
```

## Setup

### Google Colab (recommended)

All training, evaluation, and inference were performed on Google Colab with GPU runtime. The trained models are stored on Google Drive.

1. Open the [Colab notebook](https://colab.research.google.com/drive/1lWGMBimmVMaQ-RQI9R8hEys3JtVxEMgG?usp=drive_link).
2. Mount your Google Drive (the notebook handles this).
3. Source files and trained model checkpoints are available in the [Google Drive folder](https://drive.google.com/drive/folders/1NF3m1lP8Xn9u3rXBdm-T0FYSSVC_S6Wd?usp=drive_link).
4. Run the notebook cells to train, evaluate, or run inference on the models.

### Local

1. Download the project directory from the [Google Drive folder](https://drive.google.com/drive/folders/1NF3m1lP8Xn9u3rXBdm-T0FYSSVC_S6Wd?usp=drive_link) (includes source files, trained model checkpoints, and outputs).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run `Execution_notebook.ipynb` to perform training, evaluation, or inference.

## Data and Base Model

- Training corpus: AI4Bharat IndicCorpv2 Hindi split (hin_Deva)
- Evaluation set: XL-Sum Hindi
- Base LLM: Qwen/Qwen2-1.5B
- Draft baseline model (for 2-model speculative decoding): Qwen/Qwen2-0.5B

## Training, Evaluation, and Inference

All training, evaluation, and inference are performed via `Execution_notebook.ipynb`. The notebook calls the underlying functions in `src/` and handles setup, model loading, and output saving. Open it on Colab or locally and follow the cells for:

- **Training** exit heads via KL-divergence distillation
- **Evaluation** of all 12 decoding methods (101-prompt and 500-prompt benchmarks)
- **Inference** on custom Hindi prompts

# Models

- Hugging Face model: [piyukr26/INLP](https://huggingface.co/piyukr26/INLP/tree/main)

## Final Report Artifacts

Primary final assets used in the report:

- Figures: results_final/plots
- Tables: results_final/tables
- Summary table CSV: results_final/tables/summary_table.csv
- Morphological table CSV: results_final/tables/morphological_upos_table.csv

To regenerate figures/tables from existing outputs, run the export cells in visual_analysis.ipynb.

## Main Results Snapshot

### 101-prompt benchmark

| Method | Acceptance | Speedup | Throughput (tok/s) |
|---|---:|---:|---:|
| BN-TrueExit | 70.9% | 0.796x | 21.50 |
| Draft Model | 65.2% | 0.863x | 23.18 |
| WThomp-BN | 62.9% | 0.876x | 24.13 |
| UCB-BN | 62.3% | 0.919x | 24.81 |
| Thomp-BN | 54.8% | 1.130x | 30.40 |

### 500-prompt robustness

| Method | Acceptance | Speedup |
|---|---:|---:|
| BN-TrueExit | 68.8% | 0.751x |
| Thomp-BN | 54.5% | 1.041x |
| Thompson | 48.4% | 0.985x |

Interpretation:

- BN-TrueExit is best when acceptance/fidelity is prioritized.
- Thomp-BN is best when wall-clock latency and throughput are prioritized.
- Bottleneck variants consistently outperform hook counterparts across adaptive policies.

## Reproducibility Notes

- Exit depths used in final experiments: 8, 16, 22
- Bottleneck hidden dimension: 256
- UCB exploration constant: 1.41
- Weighted Thompson score weights: correctness 0.92, time 0.08
- Generation draft length for main comparisons: K = 3

## References

- Liu et al. (2024), EESD: https://aclanthology.org/2024.findings-acl.195/
- Leviathan et al. (2023), Speculative Decoding: https://arxiv.org/abs/2211.17192
- Hinton et al. (2015), Distillation: https://arxiv.org/abs/1503.02531
