# Intention Collapse Experiments

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patriciomvera/intention-collapse-experiments/blob/main/notebooks/scaled/01_run_experiments.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2501.xxxxx)

Empirical validation of the **Intention Collapse** framework for understanding reasoning in Large Language Models.

## 📄 Paper

> **Intention Collapse: Intention-Level Metrics for Reasoning in Language Models**
>
> P. M. Vera • 2025
>
> [Read on arXiv](https://arxiv.org/abs/2601.01011) | [PDF](docs/paper/Intention_Collapse_v2.pdf)

**Abstract**: We propose *intention collapse* as a unifying framework for analyzing language model reasoning: a two-stage process where a high-dimensional internal state I (intention) is irreversibly projected into a concrete linguistic output through a collapse operator κ. We introduce three model-agnostic intention metrics—H_int(I), dim_eff(I), and Recov(I;Z)—and validate them across multiple models and benchmarks.

## 🎯 Overview

The framework distinguishes:
- **Intention Formation**: Building a rich internal state I from prompt, context, and parameters
- **Intention Collapse**: Irreversible projection κ : I → y (linguistic output)

This perspective unifies contemporary reasoning techniques (CoT, STaR, Quiet-STaR, process supervision, test-time training) as interventions on I *before* collapse.

## 📊 Key Findings

**Pilot Study** (200 GSM8K problems, Mistral-7B):
- CoT improves accuracy: 5.5% → 53.0%
- CoT reduces intention entropy: 1.42 → 0.37 bits
- CoT increases global dimensionality: 2.43 → 2.85
- Linear probe recovers latent correctness: AUROC = 0.65 [0.57-0.72]

**Scaled Experiments** (3 models × 3 benchmarks, in progress):
- Cross-model validation of intention collapse signatures
- Benchmark generalization (GSM8K, MATH, ARC-Challenge)
- Methodological improvements addressing reviewer feedback

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

**Run Scaled Experiments:**

1. Click the badge above or open in Colab: [`01_run_experiments.ipynb`](notebooks/scaled/01_run_experiments.ipynb)
2. Set `MODEL_FAMILY` and `BENCHMARK` in Section 2
3. Runtime > Run all (~40 min on H100)
4. Repeat for all 9 combinations
5. Consolidate with [`02_consolidate_results.ipynb`](notebooks/scaled/02_consolidate_results.ipynb)

**Explore Pilot Study:**

Open [`notebooks/pilot/01_pilot_gsm8k.ipynb`](notebooks/pilot/01_pilot_gsm8k.ipynb) for the initial validation experiment.

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/patriciomvera/intention-collapse-experiments.git
cd intention-collapse-experiments

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token
export HF_TOKEN="your_token_here"

# Run experiments
jupyter notebook notebooks/scaled/01_run_experiments.ipynb
```

## 📁 Repository Structure

```
intention-collapse-experiments/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
│
├── src/
│   ├── shared_utils.py                # Single source of truth (v3.0, 1347 lines)
│   └── [other utilities]              # Visualization, data loaders
│
├── notebooks/
│   ├── README.md                      # Notebooks overview
│   ├── pilot/                         # Initial validation (200 GSM8K, 1 model)
│   │   ├── README.md
│   │   └── 01_pilot_gsm8k.ipynb
│   ├── scaled/                        # Full agenda (3 models × 3 benchmarks)
│   │   ├── README.md
│   │   ├── 01_run_experiments.ipynb   # Execution notebook
│   │   └── 02_consolidate_results.ipynb  # Analysis & figures
│   └── colab_quick_test.ipynb         # Setup validation
│
├── results/
│   ├── pilot/                         # Pilot study outputs
│   └── scaled/                        # Scaled experiment outputs
│       ├── figures/                   # Publication-quality plots
│       ├── raw_data/                  # JSON results & activations
│       └── checkpoints/               # Incremental saves
│
├── docs/
│   ├── README.md
│   ├── paper/
│   │   └── Intention_Collapse_v2.pdf  # Current manuscript
│   └── EXPERIMENT_GUIDE.md            # Detailed reproduction guide
│
└── configs/
    └── experiment_config.yaml         # Hyperparameters
```

## 📈 Intention Metrics

Three model-agnostic metrics for quantifying pre-collapse states:

### 1. Intention Entropy H_int(I)
Shannon entropy of the next-token distribution immediately before first emission:
```python
H_int(I) = -Σ p(y|I) log₂ p(y|I)
```
*Lower entropy → more decided intention*

### 2. Effective Dimensionality dim_eff(I)
PCA-based dimensionality of hidden activations:
```python
dim_eff(I) = smallest k where Σᵏλᵢ / Σλᵢ ≥ 0.9
```
*Higher dimensionality → richer internal representation*

### 3. Latent Recoverability Recov(I;Z)
Linear probe accuracy (AUROC) for predicting task outcomes:
```python
Recov(I;Z) = AUROC(probe(I), Z)
```
*Higher AUROC → more latent information preserved pre-collapse*

## 🔬 Experimental Design

### Pilot Study
- **Model**: Mistral-7B-Instruct-v0.3 (4-bit)
- **Benchmark**: GSM8K (200 problems)
- **Conditions**: Baseline, CoT, Babble
- **Status**: ✅ Complete

### Scaled Experiments (Current)
- **Models**: Mistral-7B, Llama-3.1-8B, Qwen-2.5-7B
- **Benchmarks**: GSM8K, MATH, ARC-Challenge
- **Design**: 3×3 matrix = 9 runs
- **Conditions**: Baseline, CoT, Babble (per run)
- **Status**: 🔄 In Progress

### Future Work
- Experiment 4.2: State-dependent temperature policies
- Experiment 4.3: Latent recovery with quirky models
- Multi-modal extension (vision + language)

## 🧪 Requirements

### Software
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.40+
- CUDA 11.8+ (for GPU acceleration)

### Hardware

| Environment | GPU Memory | Config |
|-------------|------------|--------|
| Colab Free | 12-15 GB | 4-bit, batch_size=1 |
| Colab Pro | 24-40 GB | 8-bit or fp16, batch_size=4 |
| Local A100 | 40-80 GB | Full precision, batch_size=8+ |

**Recommended**: Google Colab with H100 GPU (~$10/month Pro subscription)

## 🤝 Contributing

Contributions welcome! Areas of interest:

- [ ] Additional models (Claude, GPT-4, Gemini via API)
- [ ] Additional benchmarks (MMLU, HumanEval)
- [ ] Experiment 4.2 implementation
- [ ] Experiment 4.3 implementation
- [ ] Statistical significance tests
- [ ] Cross-lingual validation

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📚 Citation

If you use this code or findings:

```bibtex
@article{vera2025intention,
  title={Intention Collapse: Intention-Level Metrics for Reasoning in Language Models},
  author={Vera, Patricio M.},
  journal={arXiv preprint arXiv:2601.01011},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Hugging Face for model hosting and `transformers` library
- Google Colab for accessible GPU compute
- Anthropic's Claude for research assistance and code review
- OpenAI (GSM8K), Hendrycks et al. (MATH), AI2 (ARC) for benchmark datasets

## 📧 Contact

**Patricio M. Vera**
- GitHub: [@patriciomvera](https://github.com/patriciomvera)
- Email: patricio.vera@gwu.edu
- Institution: George Washington University

For questions, open an issue or start a discussion.

---

**Status**: 🔄 Active Development | Last updated: January 2025
