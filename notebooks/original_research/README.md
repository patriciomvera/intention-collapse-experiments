# Original Research: Intention Collapse Framework

This directory contains the notebooks and experiments for the **Intention Collapse** paper, which introduces a unified framework for understanding reasoning in Large Language Models.

## Paper

> **Intention Collapse: Intention-Level Metrics for Reasoning in Language Models**
>
> P. M. Vera
>
> [Read on arXiv](https://arxiv.org/abs/2601.01011)

## Framework Overview

The **Intention Collapse** framework analyzes the many-to-one mapping from internal intention states *I* to external language *L*. We introduce three model-agnostic metrics:

1. **H_int(I)**: Intention entropy - Shannon entropy of next-token distribution
2. **dim_eff(I)**: Effective dimensionality - PCA-based participation ratio
3. **Recov(I;Z)**: Recoverability - Linear probe AUROC for predicting success

## Experimental Structure

### Pilot Study (`pilot/`)

Initial validation on a small scale:
- **Model**: Mistral-7B-Instruct
- **Benchmark**: GSM8K (200 problems)
- **Conditions**: Baseline, CoT, Babble control

**Purpose**: Validate that intention metrics are implementable and show expected patterns.

### Scaled Experiments (`scaled/`)

Full 3×3 experimental design:
- **Models**: Mistral-7B, LLaMA-3.1-8B, Qwen-2.5-7B
- **Benchmarks**: GSM8K, MATH, ARC-Challenge
- **Conditions**: Baseline, CoT, Babble control
- **Total**: 9 model×benchmark combinations

**Purpose**: Validate framework across models and task types for publication.

## Key Findings

From the scaled experiments:

1. **CoT is not universally beneficial**: Large gains on GSM8K but consistent degradations on ARC-Challenge
2. **Distinct entropy regimes**: Mistral shows lower-entropy CoT while LLaMA shows higher-entropy CoT
3. **Recoverability dissociation**: High probe AUROC can co-occur with degraded accuracy

## Directory Contents

```
original_research/
├── README.md                   # This file
├── pilot/                      # Initial pilot study
│   ├── README.md
│   └── 01_pilot_gsm8k.ipynb   # Pilot experiment
└── scaled/                     # Full 3×3 experiments
    ├── README.md
    ├── METHODOLOGICAL_CLARIFICATIONS.md
    ├── 01_run_experiments.ipynb              # Run single experiment
    ├── 02_consolidate_results.ipynb          # Consolidate all results
    └── reviewer_response_recalculations.ipynb # Post-review analysis
```

## Running the Experiments

### Pilot Study (Quick Start)

```python
# In Colab or local Jupyter
jupyter notebook pilot/01_pilot_gsm8k.ipynb

# Follow notebook instructions
# Expected time: 30-45 minutes on GPU
```

### Scaled Experiments (Full Replication)

**Step 1**: Run individual experiments (9 times)

```python
# Open scaled/01_run_experiments.ipynb
# Set MODEL_FAMILY and BENCHMARK (only 2 variables!)
MODEL_FAMILY = 'mistral'  # or 'llama', 'qwen'
BENCHMARK = 'gsm8k'       # or 'math', 'arc'

# Run all cells
# Repeat for all 9 combinations
```

**Step 2**: Consolidate results

```python
# After all 9 experiments complete
# Open scaled/02_consolidate_results.ipynb
# Run all cells to generate paper figures and tables
```

**Total compute**: ~6-7 hours on Colab H100

## Experiment Checklist

Run these 9 combinations:

- [ ] mistral + gsm8k
- [ ] mistral + math
- [ ] mistral + arc
- [ ] llama + gsm8k
- [ ] llama + math
- [ ] llama + arc
- [ ] qwen + gsm8k
- [ ] qwen + math
- [ ] qwen + arc

## Metrics Computed

All experiments compute:

| Metric | Description | Expected Pattern |
|--------|-------------|------------------|
| **Accuracy** | % correct answers | CoT > Baseline |
| **H_int(I)** | Intention entropy (bits) | CoT < Baseline (more decided) |
| **dim_eff(I)** | Effective dimensionality | CoT ≥ Baseline (richer state) |
| **Recov(I;Z)** | Probe AUROC | CoT > Baseline (latent signal) |

## Hardware Requirements

| Experiment | GPU Memory | Recommended Hardware |
|------------|------------|---------------------|
| Pilot | 12-15 GB | Colab Free (T4/L4) |
| Scaled (per run) | 12-15 GB | Colab Free (T4/L4) or H100 |
| Consolidation | CPU only | Any |

## Outputs

Results are saved to Google Drive (or local `results/` directory):

```
results/original_research/
├── pilot/
│   ├── mistral_gsm8k_results.json
│   ├── mistral_gsm8k_activations.npz
│   └── mistral_gsm8k_summary.png
└── scaled/
    ├── splits/                    # Dataset indices (reproducibility)
    ├── mistral_gsm8k_results.json # Detailed results
    ├── [... 8 more combinations ...]
    └── paper_outputs/             # Publication figures
        ├── accuracy_heatmap.pdf
        ├── entropy_distributions.pdf
        ├── probe_auroc.pdf
        └── results_table.tex
```

## Methodological Notes

The scaled experiments address issues from the pilot:

- ✅ **No data leakage**: Pipeline with per-fold normalization
- ✅ **Symbolic evaluation**: Sympy for MATH benchmark
- ✅ **Robust extraction**: Multiple regex patterns, last match
- ✅ **Reproducibility**: Fixed seeds, saved indices, version tracking

See [`scaled/METHODOLOGICAL_CLARIFICATIONS.md`](scaled/METHODOLOGICAL_CLARIFICATIONS.md) for details.

## Citation

If you use these experiments:

```bibtex
@article{vera2025intention,
  title={Intention Collapse: Intention-Level Metrics for Reasoning in Language Models},
  author={Vera, Patricio M.},
  journal={arXiv preprint arXiv:2601.01011},
  year={2025}
}
```

## Next Steps

After completing these experiments:

1. **Adaptive Router**: See [`../adaptive_router/`](../adaptive_router/) for applying intention entropy to routing
2. **Extended Analysis**: Cross-model comparisons, failure mode analysis
3. **Future Work**: State-dependent temperature, latent recovery with quirky models

## Support

For questions about the original research experiments:
1. Review the paper: https://arxiv.org/abs/2601.01011
2. Check methodological clarifications: [`scaled/METHODOLOGICAL_CLARIFICATIONS.md`](scaled/METHODOLOGICAL_CLARIFICATIONS.md)
3. Open an issue: https://github.com/patriciomvera/intention-collapse-experiments/issues
