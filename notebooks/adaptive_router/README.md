# Adaptive Inference Router

This directory contains experiments and demonstrations of the **Adaptive Inference Router**, which uses intention entropy to automatically route between inference strategies.

## Concept

The Adaptive Router measures **intention entropy H_int(I)** before generating a response:

- **Low entropy (< 0.5 bits)** → Model is confident → Use direct answer (cheap, fast)
- **High entropy (> 1.2 bits)** → Model is uncertain → Use Chain-of-Thought (more tokens, better reasoning)

This approach aims to achieve the accuracy of CoT while using fewer tokens than always-CoT.

## Notebooks

### 1. `colab_demo.ipynb` - Quick Demo

**Purpose**: Interactive demonstration of the router in ~5 minutes

**Features**:
- Fully self-contained (automatic installation)
- Runs on CPU or GPU
- 4 test questions with different complexity levels
- Visualizes routing decisions

**Usage**:
```python
# Open in Google Colab
# Click: Runtime > Run all
# Wait 3-5 minutes
```

**Perfect for**:
- First-time users
- Understanding how the router works
- Quick experimentation

### 2. `full_experiment.ipynb` - Complete Evaluation

**Purpose**: Comprehensive evaluation on GSM8K dataset

**Features**:
- 200 GSM8K problems
- 4 strategies compared:
  1. Always-Direct (baseline)
  2. Always-CoT (expensive baseline)
  3. Random routing (control)
  4. Adaptive routing (our approach)
- Detailed metrics and visualizations
- Statistical analysis

**Usage**:
```python
# Requires GPU (recommended: H100)
# Expected time: 1-2 hours

# Configure in notebook:
CONFIG = {
    'model_name': 'Qwen/Qwen2.5-7B-Instruct',
    'n_problems': 200,
    'entropy_threshold_low': 0.5,
    'entropy_threshold_high': 1.2,
}

# Run all cells
```

**Perfect for**:
- Evaluating router performance
- Tuning entropy thresholds
- Comparing strategies

## Quick Start

### Google Colab (Recommended)

**Option 1: Run demo directly**
```python
!git clone -b main https://github.com/patriciomvera/intention-collapse-experiments.git
%cd intention-collapse-experiments/notebooks/adaptive_router
# Open colab_demo.ipynb and run all cells
```

**Option 2: Full experiment**
```python
!git clone -b main https://github.com/patriciomvera/intention-collapse-experiments.git
%cd intention-collapse-experiments/notebooks/adaptive_router
# Open full_experiment.ipynb
# Configure model and thresholds
# Run all cells
```

### Local Jupyter

```bash
# Clone repository
git clone https://github.com/patriciomvera/intention-collapse-experiments.git
cd intention-collapse-experiments

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/adaptive_router/colab_demo.ipynb
```

## Expected Results

Based on pilot experiments (Mistral-7B, GSM8K):

| Strategy | Accuracy | Avg Tokens/Query | Efficiency Score |
|----------|----------|------------------|------------------|
| Always-Direct | ~62% | 85 | 0.729 |
| Always-CoT | ~78% | 340 | 0.229 |
| Random | ~70% | 212 | 0.330 |
| **Adaptive Router** | **~75%** | **180** | **0.417** |

**Key Insight**: Adaptive routing achieves 96% of CoT accuracy with 47% fewer tokens.

## How It Works

```python
from src.router import AdaptiveInferenceRouter

# 1. Initialize router with thresholds
router = AdaptiveInferenceRouter(
    model=model,
    tokenizer=tokenizer,
    entropy_threshold_low=0.5,   # Below: use direct
    entropy_threshold_high=1.2   # Above: use CoT
)

# 2. Router measures entropy and decides
result = router.generate(
    question="If John has 12 apples and gives away 1/3, how many remain?",
    max_tokens_cot=512
)

# 3. Check decision
print(f"Entropy: {result.intention_entropy:.3f} bits")
print(f"Route: {result.route_taken}")  # 'direct' or 'cot'
print(f"Answer: {result.extracted_answer}")
```

## Configuration

### Threshold Profiles

| Profile | Low | High | Behavior | Use When |
|---------|-----|------|----------|----------|
| Conservative | 0.3 | 0.8 | Prefer CoT (safer) | Accuracy critical |
| **Balanced** | **0.5** | **1.2** | **Good trade-off** | **Default** |
| Aggressive | 0.8 | 1.5 | Prefer direct (cheaper) | Cost sensitive |

### Supported Models

Any HuggingFace causal LM, tested with:
- Mistral-7B-Instruct
- Qwen-2.5-7B-Instruct
- LLaMA-3.1-8B-Instruct
- GPT-2 (for quick testing)

## Outputs

Experiments save results to `../../results/adaptive_router/`:

```
results/adaptive_router/
├── experiment_summary.json          # High-level metrics
├── always_direct_detailed.json      # Detailed results per strategy
├── always_cot_detailed.json
├── random_detailed.json
├── adaptive_detailed.json
├── comparison.csv                   # Tabular comparison
└── visualizations/                  # Plots
    ├── comparison.png
    ├── efficiency_scatter.png
    └── entropy_distribution.png
```

## Customization

### Custom Dataset

```python
# Replace GSM8K with your own questions
test_questions = [
    {
        "question": "Your custom question here",
        "expected_answer": "Expected answer (optional)",
        "expected_route": "DIRECT" or "COT"
    },
    # ... more questions
]
```

### Custom Thresholds

```python
# Experiment with different thresholds
router = AdaptiveInferenceRouter(
    model=model,
    tokenizer=tokenizer,
    entropy_threshold_low=0.3,   # More conservative
    entropy_threshold_high=0.9   # More conservative
)
```

### Custom Prompts

```python
# Modify prompts in router.generate()
result = router.generate(
    question="Your question",
    direct_prompt="Answer briefly: {question}",
    cot_prompt="Think step by step:\n{question}"
)
```

## Performance Tips

### For Quick Testing
- Use GPT-2 or small models
- Reduce `n_problems` to 50
- Run on Colab Free tier

### For Serious Evaluation
- Use 7B+ instruction-tuned models
- Full 200 problems
- Colab Pro with H100 GPU
- Monitor entropy distributions

## Troubleshooting

**Router always chooses same route**
- Check entropy distribution with `router.stats['entropy_history']`
- Adjust thresholds based on observed entropy range

**Low accuracy across all strategies**
- Model may be too small for the task
- Try larger model (7B+)
- Check answer extraction regex

**Out of memory**
- Use 4-bit quantization: `model = AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True)`
- Reduce max_tokens_cot
- Use smaller model

## Advanced Usage

### Threshold Sweep

Test multiple threshold configurations:

```python
thresholds = [(0.3, 0.8), (0.5, 1.2), (0.8, 1.5)]

for low, high in thresholds:
    router = AdaptiveInferenceRouter(
        model, tokenizer,
        entropy_threshold_low=low,
        entropy_threshold_high=high
    )
    # Run experiments...
```

### Error Analysis

Analyze when router makes wrong decisions:

```python
import json

with open('results/adaptive_router/adaptive_detailed.json') as f:
    data = json.load(f)

# Find low-entropy errors (confident but wrong)
low_entropy_errors = [
    r for r in data['results']
    if not r['is_correct'] and r['entropy'] < 0.5
]

# Find high-entropy errors (uncertain and wrong)
high_entropy_errors = [
    r for r in data['results']
    if not r['is_correct'] and r['entropy'] > 1.2
]
```

## Related

- **Paper**: The router is based on the Intention Collapse framework - see [`../original_research/`](../original_research/)
- **Examples**: Standalone scripts in [`../../examples/router_demo.py`](../../examples/router_demo.py)
- **Source Code**: [`../../src/router/`](../../src/router/)
- **Documentation**: [`../../docs/`](../../docs/)

## Citation

```bibtex
@misc{adaptive-router-2025,
  title={Adaptive Inference Router Using Intention Entropy},
  author={Intention Collapse Research Team},
  year={2025},
  note={Part of Intention Collapse Experiments}
}
```

## Support

For issues with the router:
1. Check the demo notebook comments
2. Review [`../../src/router/README.md`](../../src/router/README.md)
3. Try the standalone example: [`../../examples/router_demo.py`](../../examples/router_demo.py)
4. Open an issue: https://github.com/patriciomvera/intention-collapse-experiments/issues
