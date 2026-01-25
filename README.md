# Intention Collapse Experiments

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patriciomvera/intention-collapse-experiments/blob/main/notebooks/scaled/01_run_experiments.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.01011-b31b1b.svg)](https://arxiv.org/abs/2601.01011)

Empirical validation of the **Intention Collapse** framework for understanding reasoning in Large Language Models.

## Paper

> **Intention Collapse: Intention-Level Metrics for Reasoning in Language Models**
>
> P. M. Vera
>
> [Read on arXiv](https://arxiv.org/abs/2601.01011) | [PDF](docs/paper/Intention_Collapse.pdf)

## Abstract

Language generation maps a rich, high-dimensional internal state to a single token sequence. We study this many-to-one mapping through the lens of **intention collapse**: the projection from an internal intention space *I* to an external language space *L*. We introduce three cheap, model-agnostic metrics computed on a pre-collapse state *I*:

1. **Intention entropy** H_int(I) - Shannon entropy of the next-token distribution at the pre-collapse moment
2. **Effective dimensionality** dim_eff(I) - PCA-based participation ratio capturing geometric richness
3. **Recoverability** Recov(I;Z) - Linear probe AUROC for predicting eventual success

We evaluate these metrics in a 3x3 study across models (Mistral-7B, LLaMA-3.1-8B, Qwen-2.5-7B) and benchmarks (GSM8K, ARC-Challenge, AQUA-RAT), comparing baseline, chain-of-thought (CoT), and a babble control (n=200 items per cell).

### Key Findings

- **CoT is not universally beneficial**: Large gains on GSM8K but consistent degradations on ARC-Challenge
- **Distinct entropy regimes across models**: Mistral shows lower-entropy CoT while LLaMA shows higher-entropy CoT
- **Recoverability can dissociate from accuracy**: High probe AUROC can co-occur with degraded CoT accuracy, suggesting internal signal exists but isn't reliably converted to final decisions

## Repository Structure

```
intention-collapse-experiments/
├── README.md                    # This file
├── CLAUDE.md                    # Project context for AI assistants
├── CONTRIBUTING.md              # Contribution guidelines
├── requirements.txt             # Python dependencies
├── .gitignore
│
├── configs/
│   └── experiment_config.yaml   # Experiment hyperparameters
│
├── docs/
│   ├── README.md
│   └── paper/
│       └── Intention_Collapse.pdf   # Current manuscript
│
├── notebooks/
│   ├── colab_quick_test.ipynb       # Setup validation
│   ├── pilot/                       # Initial validation (200 GSM8K, 1 model)
│   │   ├── 01_pilot_gsm8k.ipynb
│   │   └── README.md
│   └── scaled/                      # Full 3x3 experiments
│       ├── 01_run_experiments.ipynb
│       ├── 02_consolidate_results.ipynb
│       ├── reviewer_response_recalculations.ipynb
│       ├── METHODOLOGICAL_CLARIFICATIONS.md
│       └── README.md
│
├── results/
│   ├── data/                        # JSON results & activations
│   └── figures/                     # Publication-quality plots
│
└── src/
    ├── __init__.py
    ├── activation_hooks.py          # Activation extraction utilities
    ├── checkpoint_utils.py          # Checkpoint management
    ├── data_utils.py                # Dataset loading and processing
    ├── metrics.py                   # Intention metrics computation
    ├── probing.py                   # Linear probe training/evaluation
    ├── shared_utils.py              # Core experiment utilities
    └── visualization.py             # Plotting functions
```

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Local Setup

```bash
# Clone repository
git clone https://github.com/patriciomvera/intention-collapse-experiments.git
cd intention-collapse-experiments

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token (required for gated models)
export HF_TOKEN="your_token_here"
```

### Dependencies

Core ML frameworks:
- `torch>=2.0.0`
- `transformers>=4.36.0`
- `accelerate>=0.25.0`
- `bitsandbytes>=0.41.0` (for 4-bit quantization)

Scientific computing:
- `numpy>=1.24.0`
- `scipy>=1.11.0`
- `scikit-learn>=1.3.0`
- `pandas>=2.0.0`

Visualization:
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`

## Running Experiments

### Option 1: Google Colab (Recommended)

1. Click the Colab badge above or open [`01_run_experiments.ipynb`](notebooks/scaled/01_run_experiments.ipynb)
2. Set `MODEL_FAMILY` and `BENCHMARK` in Section 2
3. Runtime > Run all
4. Repeat for all 9 model-benchmark combinations
5. Consolidate results with [`02_consolidate_results.ipynb`](notebooks/scaled/02_consolidate_results.ipynb)

### Option 2: Local Execution

```bash
# Run experiments via Jupyter
jupyter notebook notebooks/scaled/01_run_experiments.ipynb

# Or run pilot study first
jupyter notebook notebooks/pilot/01_pilot_gsm8k.ipynb
```

### Experimental Design

| Models | Benchmarks | Conditions |
|--------|------------|------------|
| Mistral-7B-Instruct | GSM8K (free-response math) | Baseline |
| LLaMA-3.1-8B-Instruct | ARC-Challenge (multiple-choice) | Chain-of-Thought |
| Qwen-2.5-7B-Instruct | AQUA-RAT (multiple-choice math) | Babble (control) |

### Hardware Requirements

| Environment | GPU Memory | Configuration |
|-------------|------------|---------------|
| Colab Free | 12-15 GB | 4-bit, batch_size=1 |
| Colab Pro | 24-40 GB | 8-bit or fp16, batch_size=4 |
| Local A100 | 40-80 GB | Full precision, batch_size=8+ |

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{vera2025intention,
  title={Intention Collapse: Intention-Level Metrics for Reasoning in Language Models},
  author={Vera, Patricio M.},
  journal={arXiv preprint arXiv:2601.01011},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contact

**Patricio M. Vera**
- GitHub: [@patriciomvera](https://github.com/patriciomvera)
- Email: patricio.vera@gwu.edu
- Institution: George Washington University

For questions, open an issue or start a discussion.

---

**Status**: Active Development | Preparing for EMNLP/ACL 2026
