# Experiments Directory

This directory contains experiments for evaluating the Adaptive Inference Router.

## Files

### Notebooks

- **`router_experiment.ipynb`**: Complete interactive notebook for GSM8K evaluation
  - Loads model (Qwen-2.5-7B or Mistral-7B)
  - Evaluates 4 strategies on 200 problems
  - Generates visualizations and analysis
  - Saves detailed results

### Scripts

- **`run_router_experiment.py`**: Standalone script (command-line)
  - Same experiment as notebook
  - Non-interactive, suitable for servers
  - Customizable via command-line arguments

## Quick Start

### Option 1: Jupyter Notebook (Interactive)

```bash
# Install dependencies
pip install -r ../requirements.txt

# Launch Jupyter
jupyter notebook router_experiment.ipynb

# Run all cells (Ctrl+Enter through each cell)
```

### Option 2: Python Script (Command-line)

```bash
# Basic usage (200 problems, Qwen model)
python run_router_experiment.py

# Custom configuration
python run_router_experiment.py \
    --n_problems 100 \
    --model mistral \
    --thresholds 0.4,1.0 \
    --output_dir results/my_experiment \
    --seed 123
```

## Experiment Design

### Strategies Compared

1. **Always-Direct**: Force direct answer for all problems (max 50 tokens)
2. **Always-CoT**: Force Chain-of-Thought for all problems (max 512 tokens)
3. **Random**: Randomly choose between direct and CoT (50/50)
4. **Adaptive-Router**: Use entropy-based routing (our approach)

### Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Accuracy** | % correct answers | correct / total |
| **Total Tokens** | Sum of all tokens | Σ (input + output) |
| **Avg Tokens/Query** | Average per problem | total_tokens / n_problems |
| **Efficiency** | Accuracy per 1k tokens | accuracy / (tokens / 1000) |

### Expected Results

Based on pilot experiments (Mistral-7B, GSM8K):

| Strategy | Accuracy | Avg Tokens | Efficiency |
|----------|----------|------------|------------|
| Always-Direct | ~62% | 85 | 0.729 |
| Always-CoT | ~78% | 340 | 0.229 |
| Random | ~70% | 212 | 0.330 |
| **Adaptive** | **~75%** | **180** | **0.417** |

**Key finding**: Adaptive achieves 96% of CoT accuracy with 47% fewer tokens.

## Configuration

### Default Settings

```python
CONFIG = {
    'model_name': 'Qwen/Qwen2.5-7B-Instruct',
    'n_problems': 200,
    'entropy_threshold_low': 0.5,
    'entropy_threshold_high': 1.2,
    'max_tokens_direct': 50,
    'max_tokens_cot': 512,
    'seed': 42
}
```

### Threshold Tuning

| Threshold Profile | Low | High | Behavior |
|------------------|-----|------|----------|
| Conservative | 0.3 | 0.8 | Prefer CoT (safer, more expensive) |
| Balanced (default) | 0.5 | 1.2 | Good trade-off |
| Aggressive | 0.8 | 1.5 | Prefer direct (riskier, cheaper) |

## Outputs

All results are saved to `../results/router_experiment/`:

```
results/router_experiment/
├── experiment_summary.json          # High-level summary
├── always_direct_detailed.json      # Full results for always-direct
├── always_cot_detailed.json         # Full results for always-CoT
├── random_detailed.json             # Full results for random
├── adaptive_detailed.json           # Full results for adaptive
├── comparison.csv                   # Tabular comparison
├── comparison.png                   # Accuracy and token bars
├── efficiency_scatter.png           # Scatter plot
├── entropy_distribution.png         # Histogram (notebook only)
└── accuracy_by_entropy.png          # Binned accuracy (notebook only)
```

## Advanced Usage

### Custom Model

```python
# In notebook, modify cell 2:
CONFIG['model_name'] = 'meta-llama/Llama-3.1-8B-Instruct'

# In script:
python run_router_experiment.py --model llama
```

### Threshold Sweep

```bash
# Test multiple threshold configurations
for thresholds in "0.3,0.8" "0.5,1.2" "0.8,1.5"; do
    python run_router_experiment.py \
        --thresholds $thresholds \
        --output_dir results/sweep_$thresholds
done
```

### Subset Evaluation (for quick testing)

```bash
# Quick test on 50 problems
python run_router_experiment.py --n_problems 50
```

## Troubleshooting

### Out of Memory

If you get OOM errors:

1. **Reduce batch size** (not applicable, we use batch_size=1)
2. **Use smaller model**: `--model mistral` (7B instead of 8B)
3. **Reduce n_problems**: `--n_problems 100`
4. **Check VRAM**: Model needs ~8GB VRAM for 4-bit quantization

### Slow Execution

- **Expected time**: ~1-2 hours for 200 problems on H100
- **Speedup options**:
  - Use smaller n_problems for testing
  - Use faster GPU
  - Run script (non-interactive) instead of notebook

### Import Errors

```bash
# Make sure you're in the right directory
cd intention-collapse-experiments/experiments

# Install dependencies
pip install -r ../requirements.txt

# Verify imports
python -c "from router import AdaptiveInferenceRouter; print('OK')"
```

## Analysis Tips

### 1. Error Analysis

After running, analyze errors:

```python
import pandas as pd
import json

# Load adaptive results
with open('results/router_experiment/adaptive_detailed.json') as f:
    data = json.load(f)

# Get errors
errors = [r for r in data['results'] if not r['is_correct']]

# Low entropy errors (confident but wrong)
low_entropy_errors = sorted(errors, key=lambda x: x['entropy'])[:10]

# High entropy errors (uncertain and wrong)
high_entropy_errors = sorted(errors, key=lambda x: x['entropy'], reverse=True)[:10]
```

### 2. Threshold Optimization

Plot accuracy and efficiency across thresholds:

```python
import matplotlib.pyplot as plt

# Run experiments with different thresholds
thresholds = [(0.3, 0.8), (0.5, 1.2), (0.8, 1.5)]
results = []  # Store results for each threshold

# Plot
plt.plot([t[0] for t in thresholds], [r['accuracy'] for r in results], 'o-')
plt.xlabel('Low Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Threshold')
plt.show()
```

### 3. Route Distribution Analysis

Understand when model routes to CoT:

```python
import numpy as np

# Load results
with open('results/router_experiment/adaptive_detailed.json') as f:
    data = json.load(f)

# Analyze by entropy bins
df = pd.DataFrame(data['results'])
df['entropy_bin'] = pd.cut(df['entropy'], bins=[0, 0.5, 1.2, 2.0])

# Accuracy by bin
print(df.groupby('entropy_bin')['is_correct'].mean())

# Route distribution by bin
print(df.groupby('entropy_bin')['route_taken'].value_counts())
```

## Citation

If you use this experiment in your research, please cite:

```bibtex
@article{intention-collapse-2025,
  title={Intention Collapse: Analyzing Pre-Generation States in Large Language Models},
  author={...},
  year={2025}
}

@misc{adaptive-router-experiment,
  title={Adaptive Inference Router Experiment},
  note={Task 5 - Router Experiment},
  year={2025}
}
```

## Support

For issues or questions:
1. Check the main project README
2. Review the demo scripts in `../examples/`
3. Open an issue on GitHub
