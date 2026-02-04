# Option-Normalized Entropy for Multiple-Choice Questions

## Motivation

Standard intention entropy H_int(I) measures uncertainty over the **entire vocabulary** (50,000+ tokens). For multiple-choice questions, this conflates two distinct issues:

1. **COMPETENCE**: Does the model know which option is correct?
2. **COMPLIANCE**: Will the model format its answer correctly?

**Option-normalized entropy** H_opt(I) isolates competence by measuring uncertainty **only over valid options** [A, B, C, D, E].

## Mathematical Definition

### Standard Entropy

```
H_int(I) = -Σ_{y ∈ V} p(y|I,x) log₂ p(y|I,x)
```

where V is the entire vocabulary (~50k tokens)

### Option-Normalized Entropy

```
H_opt(I) = -Σ_{o ∈ O} p(o|I,x) log₂ p(o|I,x)

where:
  O = {token_id(A), token_id(B), token_id(C), token_id(D)}
  p(o|I,x) = softmax(logits[o]) / Σ_{o' ∈ O} softmax(logits[o'])
```

### Theoretical Bounds

| Number of Options | Maximum Entropy | Uniform Distribution |
|-------------------|-----------------|----------------------|
| 2 (binary)        | 1.00 bits       | p(A) = p(B) = 0.5    |
| 3                 | 1.58 bits       | p(each) = 0.33       |
| 4 (ARC)           | 2.00 bits       | p(each) = 0.25       |
| 5 (AQUA)          | 2.32 bits       | p(each) = 0.20       |

**Key insight**: H_opt ≤ 2.32 bits (for 5 options), while H_int can be 5-8 bits!

## Entropy Decomposition

To understand the relationship between standard and option-normalized entropy:

```python
decomposition = compute_entropy_decomposition(logits, tokenizer, options=['A','B','C','D'])

# Returns:
{
    'standard': 3.45,                    # H_int over top-100 tokens
    'option_normalized': 1.82,           # H_opt over [A,B,C,D]
    'ratio': 0.53,                       # H_opt / H_int
    'option_probability_mass': 0.68,     # Σ p(o) for o in options
    'most_likely_option': 'C',
    'confidence': 0.42,                  # p(C)
    'option_probs': {
        'A': 0.15,
        'B': 0.21,
        'C': 0.42,  # ← Most likely
        'D': 0.22
    }
}
```

### Interpretation of Ratio

| Ratio Value | Interpretation | Implication |
|-------------|----------------|-------------|
| ratio > 0.8 | High ratio | Uncertainty is mainly about **which option** (competence issue) |
| ratio < 0.3 | Low ratio | Uncertainty is mainly **outside options** (compliance issue) |
| 0.3 ≤ ratio ≤ 0.8 | Medium ratio | Mixed uncertainty (both issues present) |

### Interpretation of Probability Mass

| Probability Mass | Interpretation |
|-----------------|----------------|
| > 0.7 | Model focused on valid options (good formatting) |
| 0.5 - 0.7 | Model somewhat focused on options |
| < 0.5 | Model may generate invalid format |

## Usage

### Basic Usage

```python
from metrics import compute_option_normalized_entropy

# Get model logits
inputs = tokenizer(prompt, return_tensors="pt")
logits = model(**inputs).logits[0, -1, :]  # Last token

# Compute option-normalized entropy
entropy = compute_option_normalized_entropy(
    logits,
    tokenizer,
    options=['A', 'B', 'C', 'D']
)

print(f"Option-normalized entropy: {entropy:.3f} bits")
# Output: "Option-normalized entropy: 1.234 bits"
```

### With Probabilities

```python
entropy, probs = compute_option_normalized_entropy(
    logits,
    tokenizer,
    options=['A', 'B', 'C', 'D'],
    return_probs=True
)

print(f"Entropy: {entropy:.3f} bits")
for option, prob in probs.items():
    print(f"  {option}: {prob:.1%}")

# Output:
# Entropy: 1.234 bits
#   A: 15.0%
#   B: 25.0%
#   C: 45.0%  ← Most likely
#   D: 15.0%
```

### Full Decomposition

```python
from metrics import compute_entropy_decomposition

decomp = compute_entropy_decomposition(
    logits,
    tokenizer,
    options=['A', 'B', 'C', 'D']
)

print(f"Standard entropy:    {decomp['standard']:.3f} bits")
print(f"Option entropy:      {decomp['option_normalized']:.3f} bits")
print(f"Ratio:               {decomp['ratio']:.3f}")
print(f"Prob mass on opts:   {decomp['option_probability_mass']:.1%}")
print(f"Most likely:         {decomp['most_likely_option']}")
print(f"Confidence:          {decomp['confidence']:.1%}")
```

### Integration with Router

```python
from router import AdaptiveInferenceRouter

# Router with option-normalized entropy (recommended for MC)
router = AdaptiveInferenceRouter(
    model=model,
    tokenizer=tokenizer,
    benchmark='arc',  # or 'aqua'
    use_option_normalized_entropy=True,  # ← Enable option-normalized
    entropy_threshold_low=0.8,           # Adjusted for MC (was 0.5)
    entropy_threshold_high=1.5,          # Adjusted for MC (was 1.2)
    verbose=True
)

result = router.generate(
    question="Which property can be measured with a ruler?",
    choices="A. mass\nB. temperature\nC. length\nD. volume",
    ground_truth="C"
)

print(f"Route: {result.route_taken}")  # Uses option-normalized entropy
```

## Threshold Adjustment for MC Questions

Since option-normalized entropy has different bounds than standard entropy, **you must adjust thresholds**:

| Benchmark Type | Standard Thresholds | Option-Normalized Thresholds | Rationale |
|---------------|---------------------|------------------------------|-----------|
| Open-ended (GSM8K) | low=0.5, high=1.2 | N/A (use standard) | Full vocabulary needed |
| 4-option MC (ARC) | low=0.5, high=1.2 | low=0.8, high=1.5 | Max entropy = 2.0 bits |
| 5-option MC (AQUA) | low=0.5, high=1.2 | low=0.9, high=1.6 | Max entropy = 2.32 bits |

**Rule of thumb**:
- For n options: max_entropy = log₂(n)
- Set high threshold to ~0.65 × max_entropy
- Set low threshold to ~0.35 × max_entropy

## Example: Competence vs Compliance

### Scenario 1: Competence Issue (High H_opt)

```
Question: "What is the capital of Bhutan?"
Options: A. Thimphu  B. Kathmandu  C. Dhaka  D. Colombo

Standard entropy:    3.2 bits
Option entropy:      1.9 bits
Ratio:               0.59
Prob mass on opts:   0.85
Most likely:         A (35%)

Interpretation: Model is uncertain about the answer (competence issue).
                But it's focused on valid options (good compliance).
Action:          Route to CoT to help model reason through geography.
```

### Scenario 2: Compliance Issue (Low H_opt)

```
Question: "What is 2 + 2?"
Options: A. 3  B. 4  C. 5  D. 6

Standard entropy:    4.1 bits  (high!)
Option entropy:      0.3 bits  (low!)
Ratio:               0.07
Prob mass on opts:   0.45
Most likely:         B (90% of option mass)

Interpretation: Model knows the answer (B=4) but lots of probability
                is on non-option tokens (formatting uncertainty).
Action:          Route to direct (model knows answer) but consider
                constrained decoding to force valid token.
```

## Implementation Details

### Token ID Resolution

Different tokenizers encode options differently:
- Some: `'A'` → token 32
- Others: `' A'` (with space) → token 32
- Others: `'A'` → token 45, `' A'` → token 67

The `get_option_token_ids()` function handles this:

```python
option_ids = get_option_token_ids(tokenizer, options=['A','B','C','D'])
# Returns: {'A': 32, 'B': 33, 'C': 34, 'D': 35}

# Tries multiple candidates and picks the most common encoding
```

### Numerical Stability

Like standard entropy, we use:
1. Softmax only over option logits (not full vocabulary)
2. Epsilon for log stability (1e-10)
3. Log base 2 for bits

```python
option_logits = torch.stack([logits[option_ids[o]] for o in options])
probs = F.softmax(option_logits, dim=-1)
entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
```

## Experimental Validation

### Hypothesis

> "For multiple-choice questions where the model knows the answer,
>  option-normalized entropy should be LOW even if standard entropy is HIGH."

### Experiment Design

1. Take 200 ARC-Challenge questions
2. For each question:
   - Compute H_int (standard entropy)
   - Compute H_opt (option-normalized entropy)
   - Record if model got answer correct
3. Analyze correlation:
   - H_int vs correctness
   - H_opt vs correctness

### Expected Results

| Metric | Correlation with Correctness | Routing Quality |
|--------|----------------------------|----------------|
| H_int  | Moderate (r ≈ -0.3 to -0.5) | Noisy signal |
| H_opt  | Strong (r ≈ -0.6 to -0.8)   | Clean signal |

**Key finding**: H_opt should be a **better predictor** of correctness than H_int for MC questions.

## Related Work

### Constrained Decoding (Task 4)

Option-normalized entropy tells us about **intention**, but the model might still generate invalid format. Combine with:

```python
# Measure intention with option-normalized entropy
entropy = compute_option_normalized_entropy(logits, tokenizer, ['A','B','C','D'])

# Force valid output with constrained decoding
output = constrained_mc_generation(model, tokenizer, prompt, valid_options=['A','B','C','D'])
```

### Self-Consistency (Task 6)

For high H_opt (uncertain about answer):

```python
if entropy > 1.5:  # High uncertainty
    # Use self-consistency: generate multiple answers, take majority vote
    answer = self_consistency_baseline(model, prompt, n_samples=5)
else:  # Low uncertainty
    # Use greedy decoding
    answer = model.generate(prompt, do_sample=False)
```

## API Reference

### `compute_option_normalized_entropy(logits, tokenizer, options, temperature=1.0, return_probs=False)`

Compute entropy over valid multiple-choice options only.

**Parameters:**
- `logits` (torch.Tensor): Model logits, shape (vocab_size,)
- `tokenizer`: HuggingFace tokenizer
- `options` (List[str]): Valid option letters (e.g., ['A','B','C','D'])
- `temperature` (float): Softmax temperature (default 1.0)
- `return_probs` (bool): Whether to return probability dict

**Returns:**
- If `return_probs=False`: float (entropy in bits)
- If `return_probs=True`: Tuple[float, Dict[str, float]]

### `compute_entropy_decomposition(logits, tokenizer, options, top_k=100)`

Decompose entropy into standard vs option-normalized components.

**Returns:** Dictionary with:
- `'standard'`: Standard H_int(I) over top-k tokens
- `'option_normalized'`: H_opt over valid options
- `'ratio'`: H_opt / H_int
- `'option_probability_mass'`: Total probability on valid options
- `'option_probs'`: Probability distribution over options
- `'most_likely_option'`: Argmax of option probabilities
- `'confidence'`: Max probability among options

### `get_option_token_ids(tokenizer, options=['A','B','C','D','E'])`

Get token IDs for multiple-choice options, handling different tokenization schemes.

**Returns:** Dict[str, int] mapping option letter to token ID

## Files

- `src/metrics.py`: Implementation of option-normalized entropy functions
- `examples/option_normalized_demo.py`: Demonstration script with 4 demos
- `docs/option_normalized_entropy.md`: This documentation

## References

- Standard intention entropy: See paper Section 2.2
- Multiple-choice evaluation: See paper Section 3.2 (ARC-Challenge, AQUA)
- Entropy decomposition: Novel contribution (Task 3)

## Citation

If you use option-normalized entropy, please cite both the Intention Collapse paper and mention this extension:

```bibtex
@article{intention-collapse-2025,
  title={Intention Collapse: Analyzing Pre-Generation States in Large Language Models},
  author={...},
  year={2025}
}

@misc{option-normalized-entropy,
  title={Option-Normalized Entropy for Multiple-Choice Questions},
  note={Extension to Intention Collapse framework, Task 3},
  year={2025}
}
```
