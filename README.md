# Intention Collapse Experiments

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patriciomvera/intention-collapse-experiments/blob/main/notebooks/adaptive_router/colab_demo.ipynb)
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
├── COLAB_INSTALL.md            # Google Colab installation guide
├── CONTRIBUTING.md              # Contribution guidelines
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Package configuration
├── setup.py                    # Package setup
│
├── configs/
│   └── experiment_config.yaml   # Experiment hyperparameters
│
├── docs/                       # Documentation
│   ├── README.md
│   ├── constrained_decoding.md
│   ├── option_normalized_entropy.md
│   ├── self_consistency.md
│   └── paper/
│       └── Intention_Collapse.pdf
│
├── examples/                   # Standalone demo scripts
│   ├── README.md
│   ├── router_demo.py
│   ├── option_normalized_demo.py
│   ├── self_consistency_demo.py
│   └── constrained_decoding_demo.py
│
├── notebooks/                  # Jupyter notebooks
│   ├── README.md
│   ├── original_research/      # Intention Collapse framework (Paper)
│   │   ├── README.md
│   │   ├── pilot/             # Initial validation (1 model × 1 benchmark)
│   │   │   ├── README.md
│   │   │   └── 01_pilot_gsm8k.ipynb
│   │   └── scaled/            # Full experiments (3 models × 3 benchmarks)
│   │       ├── README.md
│   │       ├── METHODOLOGICAL_CLARIFICATIONS.md
│   │       ├── 01_run_experiments.ipynb
│   │       ├── 02_consolidate_results.ipynb
│   │       └── reviewer_response_recalculations.ipynb
│   └── adaptive_router/        # Adaptive routing experiments
│       ├── README.md
│       ├── colab_demo.ipynb   # Quick Colab demo
│       ├── quick_test.ipynb   # Setup validation
│       └── full_experiment.ipynb  # Complete router evaluation
│
├── scripts/                    # Standalone Python scripts
│   ├── README.md
│   ├── colab_setup.py         # Automated Colab installation
│   ├── verify_colab_setup.py  # Installation verification
│   └── run_router_experiment.py  # Non-interactive experiment runner
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── activation_hooks.py    # Activation extraction
│   ├── checkpoint_utils.py    # Checkpoint management
│   ├── data_utils.py          # Dataset loading
│   ├── metrics.py             # Intention metrics
│   ├── probing.py             # Linear probes
│   ├── shared_utils.py        # Core utilities
│   ├── visualization.py       # Plotting functions
│   ├── controls/              # Self-consistency baseline
│   │   ├── __init__.py
│   │   └── self_consistency.py
│   ├── decoding/              # Constrained decoding
│   │   ├── __init__.py
│   │   └── constrained.py
│   └── router/                # Adaptive router
│       ├── __init__.py
│       ├── README.md
│       └── adaptive_router.py
│
├── tests/                      # Test suite
│   ├── README.md
│   ├── test_imports.py
│   ├── test_option_normalized_entropy.py
│   └── test_package_structure.py
│
└── results/                    # Experiment results (gitignored)
    ├── original_research/      # Paper experiment outputs
    └── adaptive_router/        # Router experiment outputs
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

### Google Colab Setup

For quick experimentation in Google Colab:

**Automated Setup (Recommended):**
```python
!wget -q https://raw.githubusercontent.com/patriciomvera/intention-collapse-experiments/main/scripts/colab_setup.py
!python colab_setup.py
```

**Manual Setup:**
```python
!git clone -b main https://github.com/patriciomvera/intention-collapse-experiments.git
!pip install -e /content/intention-collapse-experiments/

# Verify installation
from src.router import AdaptiveInferenceRouter, RouteDecision
from src.metrics import compute_intention_entropy
from src.controls import self_consistency_baseline
from src.decoding import constrained_mc_generation
print("[OK] Ready to run experiments!")
```

**Quick Start Notebooks:**
- **Adaptive Router Demo:** [`notebooks/adaptive_router/colab_demo.ipynb`](notebooks/adaptive_router/colab_demo.ipynb) - 5 minute demo
- **Original Research:** [`notebooks/original_research/pilot/01_pilot_gsm8k.ipynb`](notebooks/original_research/pilot/01_pilot_gsm8k.ipynb) - Paper experiments

See [`COLAB_INSTALL.md`](COLAB_INSTALL.md) for detailed installation instructions.

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

### Quick Start: Adaptive Router

Try the router in ~5 minutes:
```bash
# In Google Colab or local Jupyter
jupyter notebook notebooks/adaptive_router/colab_demo.ipynb
```

See [`notebooks/adaptive_router/README.md`](notebooks/adaptive_router/README.md) for details.

### Original Research: Intention Collapse Framework

**Pilot Study (Quick Start):**
```bash
# Initial validation experiment
jupyter notebook notebooks/original_research/pilot/01_pilot_gsm8k.ipynb
```

**Scaled Experiments (Full Replication):**
```bash
# Run individual experiments (repeat 9 times)
jupyter notebook notebooks/original_research/scaled/01_run_experiments.ipynb
# Configure MODEL_FAMILY and BENCHMARK in the notebook

# Consolidate all 9 experiments
jupyter notebook notebooks/original_research/scaled/02_consolidate_results.ipynb
```

See [`notebooks/original_research/README.md`](notebooks/original_research/README.md) for complete details.

### Command-Line Experiments

For non-interactive execution:
```bash
# Run router experiment from command line
python scripts/run_router_experiment.py --n_problems 200 --model qwen

# See all options
python scripts/run_router_experiment.py --help
```

See [`scripts/README.md`](scripts/README.md) for more options.

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
