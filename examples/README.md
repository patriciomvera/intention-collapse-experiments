# Examples

Standalone demo scripts showing how to use the Intention Collapse framework and Adaptive Inference Router.

## Files

### Router Examples

#### `router_demo.py`
Quick demonstration of the Adaptive Inference Router.

**Usage**:
```bash
python examples/router_demo.py
```

**What it demonstrates**:
- Loading a model (GPT-2 for speed)
- Creating an AdaptiveInferenceRouter
- Running inference with entropy-based routing
- Interpreting router decisions

**Output**:
```
Loading model...
Model loaded: gpt2

Creating router with thresholds:
  Low:  0.5 (below → direct)
  High: 1.2 (above → CoT)

Testing router with 4 questions...

[Question 1] What is 2 + 2?
  Entropy: 0.32 bits
  Route: DIRECT (low entropy → confident)
  Answer: 4
  Tokens used: 15

[Question 2] Complex math problem...
  Entropy: 1.45 bits
  Route: COT (high entropy → uncertain, use reasoning)
  Answer: 42
  Tokens used: 127
```

### Controls Examples

#### `self_consistency_demo.py`
Demonstrates self-consistency baseline for improved accuracy.

**Usage**:
```bash
python examples/self_consistency_demo.py
```

**What it demonstrates**:
- Multiple sampling from the same question
- Majority voting for final answer
- Confidence scores from agreement
- When self-consistency helps

**Concepts**:
- Samples K different responses
- Extracts answers from each
- Returns most common answer
- Provides agreement statistics

#### `option_normalized_demo.py`
Shows option-normalized entropy for multiple-choice questions.

**Usage**:
```bash
python examples/option_normalized_demo.py
```

**What it demonstrates**:
- Standard entropy vs option-normalized entropy
- Why option-normalized is better for MC questions
- Separating competence from compliance

**Key insight**: Standard entropy confuses two problems:
1. Does the model know the answer? (competence)
2. Will the model format correctly? (compliance)

Option-normalized entropy measures only competence.

### Decoding Examples

#### `constrained_decoding_demo.py`
Demonstrates constrained decoding for multiple-choice answers.

**Usage**:
```bash
python examples/constrained_decoding_demo.py
```

**What it demonstrates**:
- Forcing outputs to be valid choices [A, B, C, D]
- LogitsProcessor for constrained generation
- Perfect formatting compliance

**When to use**:
- Multiple-choice benchmarks (ARC, AQUA)
- When you need guaranteed valid outputs
- Eliminates formatting errors

## Quick Start

### Run All Examples

```bash
# Install dependencies first
pip install -e .

# Run each demo
python examples/router_demo.py
python examples/self_consistency_demo.py
python examples/option_normalized_demo.py
python examples/constrained_decoding_demo.py
```

### Modify for Your Use Case

Each example is a standalone script you can copy and adapt:

```bash
# Copy example as starting point
cp examples/router_demo.py my_custom_router.py

# Modify for your needs
# - Change model
# - Change questions
# - Adjust thresholds
# - Add custom logic

python my_custom_router.py
```

## Example Comparison

| Example | Time | Requires GPU | Teaches |
|---------|------|--------------|---------|
| `router_demo.py` | ~30 sec | No | Entropy-based routing |
| `self_consistency_demo.py` | ~1 min | No | Majority voting |
| `option_normalized_demo.py` | ~30 sec | No | MC entropy calculation |
| `constrained_decoding_demo.py` | ~30 sec | No | Forced valid outputs |

## When to Use Examples vs Notebooks vs Scripts

### Use Examples (`examples/`) when:
- Learning the API
- Quick standalone demonstrations
- Copy-paste starting points
- Understanding a specific component

### Use Notebooks (`notebooks/`) when:
- Interactive exploration
- Step-by-step analysis
- Complete experiments
- Visualization

### Use Scripts (`scripts/`) when:
- Non-interactive batch processing
- Server/cluster execution
- Command-line arguments
- Automation

## Code Structure

All examples follow this pattern:

```python
# 1. Imports
from src.router import AdaptiveInferenceRouter
from transformers import AutoModel, AutoTokenizer

# 2. Setup
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 3. Main demonstration
router = AdaptiveInferenceRouter(model, tokenizer)
result = router.generate(question="...")

# 4. Show results
print(f"Route: {result.route_taken}")
print(f"Answer: {result.extracted_answer}")
```

## Customization

### Change Model

```python
# In any example, replace model name
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
```

### Add Your Questions

```python
# Modify test_questions list
test_questions = [
    "Your custom question 1",
    "Your custom question 2",
    # ... more questions
]
```

### Adjust Parameters

```python
# Router thresholds
router = AdaptiveInferenceRouter(
    model, tokenizer,
    entropy_threshold_low=0.3,   # More conservative
    entropy_threshold_high=0.9
)

# Self-consistency samples
result = self_consistency_baseline(
    model, tokenizer,
    question="...",
    n_samples=10  # More samples = higher confidence
)
```

## Dependencies

All examples require:
- `torch>=2.0.0`
- `transformers>=4.36.0`
- The `intention-collapse` package installed

Install with:
```bash
pip install -e .
```

## Output

Examples print to console (no file output). For saving results, see:
- **Notebooks**: `notebooks/` - Save figures and data
- **Scripts**: `scripts/` - Command-line output options

## Advanced Usage

### Combine Multiple Techniques

```python
# Example: Router + Self-Consistency + Constrained Decoding
from src.router import AdaptiveInferenceRouter
from src.controls import self_consistency_baseline
from src.decoding import constrained_mc_generation

# 1. Measure entropy
router = AdaptiveInferenceRouter(model, tokenizer)
entropy = router.compute_intention_entropy(question)

# 2. If uncertain, use self-consistency
if entropy > 1.2:
    result = self_consistency_baseline(
        model, tokenizer, question, n_samples=5
    )
else:
    result = router.generate(question)

# 3. For MC questions, use constrained decoding
if is_multiple_choice(question):
    answer = constrained_mc_generation(
        model, tokenizer, question, valid_options=['A', 'B', 'C', 'D']
    )
```

### Batch Processing

```python
# Process multiple questions
questions = ["Question 1", "Question 2", "Question 3"]

results = []
for question in questions:
    result = router.generate(question)
    results.append({
        'question': question,
        'answer': result.extracted_answer,
        'entropy': result.intention_entropy,
        'route': result.route_taken
    })

# Analyze results
import pandas as pd
df = pd.DataFrame(results)
print(df.groupby('route')['entropy'].mean())
```

## Troubleshooting

**Import errors**
```bash
# Make sure package is installed
pip install -e .

# Verify
python -c "from src.router import AdaptiveInferenceRouter; print('OK')"
```

**Model download fails**
```bash
# Set HuggingFace token for gated models
export HF_TOKEN="your_token_here"

# Or use public models
# Change model name to "gpt2" in examples
```

**Out of memory**
```python
# Use smaller model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Or 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    load_in_4bit=True
)
```

## Next Steps

After trying examples:

1. **Dive deeper**: Try the [notebooks](../notebooks/) for complete experiments
2. **Run at scale**: Use [scripts](../scripts/) for batch processing
3. **Read the paper**: [arXiv:2601.01011](https://arxiv.org/abs/2601.01011)
4. **Explore source**: See [`../src/`](../src/) for implementation details

## Contributing

To add a new example:

1. Create `examples/your_example.py`
2. Follow the existing pattern
3. Keep it under 200 lines
4. Add docstring explaining purpose
5. Update this README
6. Test with `python examples/your_example.py`

## Resources

- **Source Code**: [`../src/`](../src/)
- **Documentation**: [`../docs/`](../docs/)
- **Notebooks**: [`../notebooks/`](../notebooks/)
- **Scripts**: [`../scripts/`](../scripts/)

## Support

For issues with examples:
1. Check the example's docstring
2. Review this README
3. Try simpler model (gpt2) first
4. Open an issue: https://github.com/patriciomvera/intention-collapse-experiments/issues
