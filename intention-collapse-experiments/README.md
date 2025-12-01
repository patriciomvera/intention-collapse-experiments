# Intention Collapse Experiments

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/intention-collapse-experiments/blob/main/notebooks/01_intention_metrics.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Empirical validation of the **Intention Collapse** framework for understanding reasoning in Large Language Models (LLMs).

## 📄 Paper

This repository accompanies the paper:

> **Intention Collapse: A Unified Framework for Understanding Reasoning in Language Models**
> 
> *Abstract*: We propose intention collapse as the process by which a cognitive system—biological or artificial—compresses a vast, implicit configuration of meaning into a single concrete linguistic message. This framework provides a unified lens for understanding contemporary reasoning techniques in LLMs.

## 🎯 Overview

The Intention Collapse framework distinguishes between:
- **Intention State (I)**: A high-dimensional internal representation aggregating context, memory, and intermediate computation
- **Collapse Operator (κ)**: The irreversible projection from I into a concrete linguistic output

This repository implements experiments to validate key predictions of the framework:

| Experiment | Description | Status |
|------------|-------------|--------|
| 4.1 | Correlating intention metrics with reasoning accuracy | ✅ Implemented |
| 4.2 | State-dependent collapse variability | 🔄 Planned |
| 4.3 | Latent knowledge recovery pre/post collapse | 🔄 Planned |

## 📊 Metrics Implemented

Three model-agnostic metrics for quantifying intention states:

1. **Intention Entropy** $H_{int}(I)$: Shannon entropy of the predicted next-token distribution
2. **Effective Dimensionality** $dim_{eff}(I)$: PCA-based dimensionality of hidden activations
3. **Latent Recoverability** $Recov(I; Z)$: Linear probe accuracy for predicting task outcomes

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badge above, or:

1. Open [Google Colab](https://colab.research.google.com/)
2. File → Open notebook → GitHub tab
3. Enter: `YOUR_USERNAME/intention-collapse-experiments`
4. Select `notebooks/01_intention_metrics.ipynb`

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/intention-collapse-experiments.git
cd intention-collapse-experiments

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set Hugging Face token
export HF_TOKEN="your_token_here"

# Run experiments
python src/run_experiment.py --config configs/experiment_config.yaml
```

## 📁 Repository Structure

```
intention-collapse-experiments/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
├── docs/
│   ├── paper/                          # Place the original paper here
│   └── README.md                       # Documentation guide
├── notebooks/
│   ├── 01_intention_metrics.ipynb      # Experiment 4.1: Metric correlation
│   ├── 02_collapse_variability.ipynb   # Experiment 4.2 (planned)
│   └── 03_latent_recovery.ipynb        # Experiment 4.3 (planned)
├── src/
│   ├── __init__.py
│   ├── metrics.py                      # Intention metric implementations
│   ├── activation_hooks.py             # Hidden state extraction utilities
│   ├── probing.py                      # Linear probe training
│   ├── data_utils.py                   # Dataset loading and preprocessing
│   └── visualization.py                # Publication-quality figures
├── configs/
│   └── experiment_config.yaml          # Experiment hyperparameters
└── results/
    ├── figures/                        # Generated plots
    └── data/                           # Experiment outputs
```

## 🔧 Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU with ≥12GB VRAM (or Google Colab T4)
- Hugging Face account (free)

### Hardware Notes

| Environment | GPU Memory | Recommended Config |
|-------------|------------|-------------------|
| Colab Free | 12-15 GB | 4-bit quantization, batch_size=1 |
| Colab Pro | 24-40 GB | 8-bit or fp16, batch_size=4 |
| Local A100 | 40-80 GB | Full precision, batch_size=8+ |

## 📈 Expected Results

Based on the theoretical framework, we hypothesize:

1. **dimeff(I)** and **Recov(I;Z)** increase under enhanced reasoning methods (CoT, STaR)
2. **Hint(I)** shows U-shaped pattern: rising during exploration, falling as intention crystallizes
3. Intention metrics correlate more strongly with accuracy than token count alone

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- [ ] Implement Experiment 4.2 (collapse variability)
- [ ] Implement Experiment 4.3 (latent recovery)
- [ ] Add support for additional models (Llama-3, Qwen-2)
- [ ] Add MATH benchmark support
- [ ] Improve visualization options
- [ ] Add statistical significance tests

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{intention_collapse_2025,
  title={Intention Collapse: A Unified Framework for Understanding Reasoning in Language Models},
  author={[Author Names]},
  journal={arXiv preprint},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for model hosting and the `transformers` library
- Google Colab for accessible GPU compute
- The open-source ML community for foundational tools

## 📧 Contact

For questions or collaboration inquiries, please open an issue or start a discussion.

---

**Note**: Replace `YOUR_USERNAME` with your actual GitHub username after creating the repository.
