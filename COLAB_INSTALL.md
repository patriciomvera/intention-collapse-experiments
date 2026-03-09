# Google Colab Installation Guide

This document explains how to install and use the `intention-collapse` package in Google Colab.

## Quick Installation (1 cell)

### Method 1: Using automatic script (Recommended)

```python
# Download and install automatically
!wget -q https://raw.githubusercontent.com/patriciomvera/intention-collapse-experiments/main/scripts/colab_setup.py
!python colab_setup.py
```

### Method 2: Manual installation

```python
import sys

# Clone and install
!rm -rf intention-collapse-experiments
!git clone -q -b main https://github.com/patriciomvera/intention-collapse-experiments.git
!pip install -q -e /content/intention-collapse-experiments/

# Configure path (necessary in Colab)
sys.path.insert(0, '/content/intention-collapse-experiments')

# Verify imports
from src.router import AdaptiveInferenceRouter, RouteDecision
from src.metrics import compute_intention_entropy
from src.controls import self_consistency_baseline
from src.decoding import constrained_mc_generation

print("✅ Installation successful! Ready to run experiments.")
```

## Example Notebooks

### Option 1: Router Experiment (Recommended)
Complete notebook with functional example of the Adaptive Inference Router:

```python
# In Colab, run:
!git clone -b main https://github.com/patriciomvera/intention-collapse-experiments.git
%cd intention-collapse-experiments/notebooks/adaptive_router
# Open: colab_demo.ipynb
```

### Option 2: Complete Experiments
To replicate the paper experiments:

```python
# In Colab, run:
!git clone -b main https://github.com/patriciomvera/intention-collapse-experiments.git
%cd intention-collapse-experiments/notebooks/original_research/scaled
# Open: 01_run_experiments.ipynb
```

## Package Structure

```python
# Import main modules
from src.router import AdaptiveInferenceRouter, RouteDecision, RouterResult
from src.metrics import compute_intention_entropy, IntentionMetrics
from src.controls import self_consistency_baseline, SelfConsistencyResult
from src.decoding import constrained_mc_generation
```

## Minimal Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.router import AdaptiveInferenceRouter

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create router
router = AdaptiveInferenceRouter(
    model=model,
    tokenizer=tokenizer,
    entropy_threshold_low=0.5,
    entropy_threshold_high=1.2
)

# Run adaptive inference
result = router.generate(
    question="What is 2 + 2?",
    max_tokens_cot=50
)

print(f"Route: {result.route_taken}")
print(f"Entropy: {result.intention_entropy:.3f}")
print(f"Answer: {result.extracted_answer}")
```

## Dependencies

The `pip install -e .` command automatically installs:

- `torch>=2.0.0`
- `transformers>=4.36.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0`
- `datasets>=2.14.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`

## Troubleshooting

### Error: "No module named 'src'"

**Cause:** The package is not installed correctly.

**Solution:**
```python
!pip install -e /content/intention-collapse-experiments/
```

### Error: "No module named 'torch'"

**Cause:** Dependencies were not installed.

**Solution:**
```python
# Install with dependencies explicitly
!pip install -e /content/intention-collapse-experiments/
```

### Error: "attempted relative import with no known parent package"

**Cause:** You are trying to run src/ files directly.

**Solution:** Import from the installed package:
```python
# ✅ Correct
from src.router import AdaptiveInferenceRouter

# ❌ Incorrect
%run src/router/adaptive_router.py
```

### Verify that everything works

Run this test script:

```python
import sys
import subprocess

# Test 1: Verify package installation
try:
    import src
    print(f"✅ Package 'src' installed (version {src.__version__})")
except ImportError as e:
    print(f"❌ Cannot import 'src': {e}")
    sys.exit(1)

# Test 2: Verify critical imports
try:
    from src.router import AdaptiveInferenceRouter, RouteDecision
    print("✅ src.router imports OK")
except ImportError as e:
    print(f"❌ src.router import failed: {e}")
    sys.exit(1)

try:
    from src.metrics import compute_intention_entropy
    print("✅ src.metrics imports OK")
except ImportError as e:
    print(f"❌ src.metrics import failed: {e}")
    sys.exit(1)

try:
    from src.controls import self_consistency_baseline
    print("✅ src.controls imports OK")
except ImportError as e:
    print(f"❌ src.controls import failed: {e}")
    sys.exit(1)

try:
    from src.decoding import constrained_mc_generation
    print("✅ src.decoding imports OK")
except ImportError as e:
    print(f"❌ src.decoding import failed: {e}")
    sys.exit(1)

print("\n✅ ALL TESTS PASSED - Ready to run experiments!")
```

## Resources

- **Quick Start Notebook:** `notebooks/adaptive_router/colab_demo.ipynb`
- **Full Experiments:** `notebooks/original_research/scaled/01_run_experiments.ipynb`
- **Documentation:** [README.md](README.md)
- **Paper:** [arXiv:2601.01011](https://arxiv.org/abs/2601.01011)

## Support

If you encounter problems:

1. Verify you are using Python 3.10+
2. Verify the branch is `main`
3. Run the verification script above
4. Report issues at: https://github.com/patriciomvera/intention-collapse-experiments/issues
