# Scripts

Standalone Python scripts for running experiments and setup tasks.

## Files

### Setup Scripts

#### `colab_setup.py`
Automated installation script for Google Colab.

**Usage**:
```python
# In Colab
!wget -q https://raw.githubusercontent.com/patriciomvera/intention-collapse-experiments/main/scripts/colab_setup.py
!python colab_setup.py
```

**What it does**:
1. Clones the repository
2. Installs the package with dependencies
3. Configures Python path
4. Verifies all imports work
5. Reports success/failure

**Output**:
```
=======================================================================
AUTOMATIC INSTALLATION - Adaptive Inference Router
=======================================================================

[1/4] Cleaning previous installation...
[2/4] Cloning repository from GitHub...
   ✅ Repository cloned
[3/4] Installing intention-collapse package...
   ✅ Package installed
[4/4] Configuring Python path...
   ✅ Path configured

=======================================================================
INSTALLATION VERIFICATION
=======================================================================
✅ All imports successful!
✅ RouteDecision available: ['direct', 'cot', 'unknown']

🎉 INSTALLATION COMPLETE - Ready to run!
```

#### `verify_colab_setup.py`
Verification script to test installation completeness.

**Usage**:
```python
# After installation
!python scripts/verify_colab_setup.py
```

**Checks**:
- Package importability
- All submodules accessible
- Dependencies installed
- Version compatibility

### Experiment Scripts

#### `run_router_experiment.py`
Standalone script for running router experiments (non-interactive).

**Usage**:
```bash
# Basic usage (200 problems, Qwen model)
python scripts/run_router_experiment.py

# Custom configuration
python scripts/run_router_experiment.py \
    --n_problems 100 \
    --model mistral \
    --thresholds 0.4,1.0 \
    --output_dir results/my_experiment \
    --seed 123

# Help
python scripts/run_router_experiment.py --help
```

**Arguments**:
```
--n_problems     Number of GSM8K problems to evaluate (default: 200)
--model          Model to use: qwen, mistral, llama, gpt2 (default: qwen)
--thresholds     Low,High entropy thresholds (default: 0.5,1.2)
--output_dir     Where to save results (default: results/router_experiment)
--seed           Random seed (default: 42)
--max_tokens_direct   Max tokens for direct answers (default: 50)
--max_tokens_cot      Max tokens for CoT (default: 512)
```

**Example**:
```bash
# Quick test with GPT-2 on 50 problems
python scripts/run_router_experiment.py \
    --model gpt2 \
    --n_problems 50 \
    --output_dir results/quick_test

# Full evaluation with conservative thresholds
python scripts/run_router_experiment.py \
    --model mistral \
    --n_problems 200 \
    --thresholds 0.3,0.8 \
    --output_dir results/conservative
```

**Output**:
Creates a complete experiment directory with:
- `experiment_summary.json` - Overall metrics
- `*_detailed.json` - Per-problem results for each strategy
- `comparison.csv` - Tabular comparison
- `*.png` - Visualization plots

**When to use**:
- Running on servers (non-interactive)
- Batch processing multiple configurations
- Automated experiments in pipelines
- Command-line preference over notebooks

## Directory Structure After Setup

```
intention-collapse-experiments/
├── scripts/
│   ├── README.md                  # This file
│   ├── colab_setup.py            # Automated Colab installation
│   ├── verify_colab_setup.py     # Installation verification
│   └── run_router_experiment.py  # Standalone experiment runner
├── notebooks/                     # Jupyter notebooks (interactive)
├── examples/                      # Quick demos
└── src/                          # Source code
```

## When to Use Scripts vs Notebooks vs Examples

### Use Scripts (`scripts/`) when:
- Running non-interactively (servers, clusters)
- Automating experiments
- Need command-line arguments
- CI/CD integration

### Use Notebooks (`notebooks/`) when:
- Interactive exploration
- Step-by-step analysis
- Visualization during execution
- Teaching/learning

### Use Examples (`examples/`) when:
- Quick standalone demos
- Understanding API usage
- Copy-paste starting points
- Minimal dependencies

## Installation

Scripts are included when you install the package:

```bash
git clone https://github.com/patriciomvera/intention-collapse-experiments.git
cd intention-collapse-experiments
pip install -e .

# Scripts are now available in scripts/
python scripts/run_router_experiment.py --help
```

## Development

### Adding New Scripts

1. Create script in `scripts/` directory
2. Add shebang: `#!/usr/bin/env python3`
3. Make executable: `chmod +x scripts/your_script.py`
4. Document in this README
5. Add example usage

### Script Template

```python
#!/usr/bin/env python3
"""
Brief description of what this script does.

Usage:
    python scripts/your_script.py [options]
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--option", type=str, help="Option description")
    args = parser.parse_args()

    # Your code here
    print(f"Running with option: {args.option}")

if __name__ == "__main__":
    main()
```

## Batch Experiment Examples

### Threshold Sweep

```bash
# Test multiple threshold configurations
for thresholds in "0.3,0.8" "0.5,1.2" "0.8,1.5"; do
    python scripts/run_router_experiment.py \
        --thresholds $thresholds \
        --output_dir results/sweep_$thresholds \
        --n_problems 200
done
```

### Multi-Model Comparison

```bash
# Compare all models
for model in qwen mistral llama; do
    python scripts/run_router_experiment.py \
        --model $model \
        --output_dir results/model_$model \
        --n_problems 200
done
```

### Quick Tests

```bash
# Fast iteration for development
python scripts/run_router_experiment.py \
    --model gpt2 \
    --n_problems 10 \
    --output_dir results/dev_test
```

## Troubleshooting

**Script not found**
```bash
# Verify you're in the repository root
pwd  # Should show .../intention-collapse-experiments

# List scripts
ls scripts/

# Run with explicit path
python scripts/run_router_experiment.py
```

**Import errors**
```bash
# Install package first
pip install -e .

# Verify installation
python -c "from src.router import AdaptiveInferenceRouter; print('OK')"
```

**Permission denied**
```bash
# Make script executable
chmod +x scripts/your_script.py

# Or run with python
python scripts/your_script.py
```

## Resources

- **Notebooks**: For interactive versions, see [`../notebooks/`](../notebooks/)
- **Examples**: For quick demos, see [`../examples/`](../examples/)
- **Documentation**: [`../docs/`](../docs/)
- **Source Code**: [`../src/`](../src/)

## Support

For issues with scripts:
1. Check the script's `--help` output
2. Review this README
3. Try the equivalent notebook version
4. Open an issue: https://github.com/patriciomvera/intention-collapse-experiments/issues
