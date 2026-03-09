# Notebooks

This directory contains Jupyter notebooks for running experiments and demonstrations of the Intention Collapse framework and Adaptive Inference Router.

## Directory Structure

```
notebooks/
├── original_research/          # Intention Collapse framework validation (Paper)
│   ├── pilot/                 # Initial pilot study (1 model × 1 benchmark)
│   └── scaled/                # Full experiments (3 models × 3 benchmarks)
└── adaptive_router/           # Adaptive Router experiments (New)
    ├── colab_demo.ipynb       # Quick Colab demo
    └── full_experiment.ipynb  # Complete router evaluation
```

## Quick Start

### For Adaptive Router (New Users)

If you want to see the Adaptive Inference Router in action:

1. **Google Colab Demo**: Open [`adaptive_router/colab_demo.ipynb`](adaptive_router/colab_demo.ipynb)
   - Fully self-contained
   - Runs in 3-5 minutes
   - No setup required

2. **Full Experiment**: Open [`adaptive_router/full_experiment.ipynb`](adaptive_router/full_experiment.ipynb)
   - Complete evaluation on GSM8K
   - Compares 4 strategies
   - Requires GPU (recommended: H100)

### For Original Research (Paper Replication)

If you want to replicate the Intention Collapse paper experiments:

1. **Pilot Study**: [`original_research/pilot/`](original_research/pilot/)
   - Initial validation on 200 GSM8K problems
   - 1 model (Mistral-7B)
   - Good starting point to understand the framework

2. **Scaled Experiments**: [`original_research/scaled/`](original_research/scaled/)
   - Full 3×3 design (3 models × 3 benchmarks)
   - Publication-ready figures and tables
   - Requires multiple GPU hours

## Notebook Descriptions

### Adaptive Router

| Notebook | Purpose | Time | Hardware |
|----------|---------|------|----------|
| `adaptive_router/colab_demo.ipynb` | Quick demo of entropy-based routing | 3-5 min | CPU/GPU |
| `adaptive_router/full_experiment.ipynb` | Complete router evaluation (200 problems) | 1-2 hours | GPU recommended |

**Key Features:**
- Measures intention entropy H_int(I) before generation
- Routes to direct answer (low entropy) or CoT (high entropy)
- Tracks accuracy, token usage, and efficiency

### Original Research

| Notebook | Purpose | Time | Hardware |
|----------|---------|------|----------|
| `original_research/pilot/01_pilot_gsm8k.ipynb` | Initial validation study | 30-45 min | GPU required |
| `original_research/scaled/01_run_experiments.ipynb` | Run one model×benchmark combo | 30-45 min | GPU required |
| `original_research/scaled/02_consolidate_results.ipynb` | Consolidate all 9 experiments | 5-10 min | CPU |

**Key Metrics:**
- **H_int(I)**: Intention entropy
- **dim_eff(I)**: Effective dimensionality
- **Recov(I;Z)**: Recoverability (probe AUROC)

## Installation

### Google Colab

```python
# Clone repository
!git clone -b main https://github.com/patriciomvera/intention-collapse-experiments.git

# Install package
!pip install -e /content/intention-collapse-experiments/

# Navigate to notebooks
%cd intention-collapse-experiments/notebooks
```

### Local

```bash
# Clone repository
git clone https://github.com/patriciomvera/intention-collapse-experiments.git
cd intention-collapse-experiments

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/
```

## Choosing the Right Notebook

**I want to...**

- **Try the router quickly** → `adaptive_router/colab_demo.ipynb`
- **Evaluate router performance** → `adaptive_router/full_experiment.ipynb`
- **Understand intention collapse** → `original_research/pilot/01_pilot_gsm8k.ipynb`
- **Replicate paper results** → `original_research/scaled/01_run_experiments.ipynb`

## Resources

- **Installation Guide**: [`../COLAB_INSTALL.md`](../COLAB_INSTALL.md)
- **Main README**: [`../README.md`](../README.md)
- **Examples**: [`../examples/`](../examples/)
- **Documentation**: [`../docs/`](../docs/)

## Support

For issues or questions:
1. Check the notebook's own README or comments
2. Review the main project documentation
3. Open an issue: https://github.com/patriciomvera/intention-collapse-experiments/issues
