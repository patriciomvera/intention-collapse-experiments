# Self-Consistency Control Baseline

## Motivation

Traditional "babble" baseline in intention collapse experiments generates meaningless output:

```
Prompt: "Solve: 2 + 2 = ?"
Babble: "Numbers are interesting. Two plus two... math concepts... thinking about addition..."
```

This is a **weak control** because:
- It's meaningless (not a genuine attempt to solve)
- Uses same tokens but provides no useful comparison
- Doesn't test if model CAN solve without explicit reasoning

**Self-consistency** provides a **rigorous control**:
- Generate N diverse answers with sampling
- Take majority vote
- Same token budget, but genuine attempts to solve

## Method

### Algorithm

```
Self-Consistency(prompt, N=5, temperature=0.7):
    answers = []
    for i in 1 to N:
        generation = model.generate(prompt, temperature=T)
        answer = extract_answer(generation)
        answers.append(answer)

    final_answer = majority_vote(answers)
    confidence = max_count / N

    return final_answer, confidence
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_samples` | 5 | Number of diverse samples |
| `temperature` | 0.7 | Sampling temperature (0 = greedy, 1 = random) |
| `max_new_tokens` | 256 | Tokens per sample |

**Temperature selection**:
- Too low (< 0.3): Samples too similar, no diversity
- Too high (> 1.0): Samples too random, poor quality
- Optimal: 0.5-0.8 for good diversity + quality

### Majority Vote

```python
answers = ["42", "42", "43", "42", "40"]

# Normalize
normalized = ["42", "42", "43", "42", "40"]

# Count
counts = {"42": 3, "43": 1, "40": 1}

# Winner
final_answer = "42"  # Most common
confidence = 3/5 = 0.6  # 60% agreement
```

## Mathematical Formulation

### Answer Distribution Entropy

Measures diversity of answers:

```
H(A) = -Σ p(a) log₂ p(a)

where p(a) = count(a) / N
```

**Interpretation**:
- H = 0: All samples agree (perfect consensus)
- H = log₂(N): Uniform distribution (maximum uncertainty)
- H ≈ 1-2: Moderate diversity (typical)

**Example**:
```python
# Perfect consensus
answers = ["42", "42", "42", "42", "42"]
H = 0 bits  # No uncertainty

# Uniform distribution
answers = ["42", "43", "44", "45", "46"]
H = log₂(5) = 2.32 bits  # Maximum uncertainty

# Typical case
answers = ["42", "42", "42", "43", "43"]
H ≈ 0.97 bits  # Some uncertainty
```

### Confidence Score

```
confidence = max(count(a)) / N

Range: [1/N, 1.0]
- 1.0 = unanimous
- 0.6 = strong majority (3/5)
- 0.4 = weak majority (2/5)
- 0.2 = no clear winner (tie with 5 options)
```

## Usage

### Basic Usage

```python
from controls import self_consistency_baseline

result = self_consistency_baseline(
    model,
    tokenizer,
    prompt,
    n_samples=5,
    temperature=0.7,
    ground_truth="42"
)

print(f"Answer: {result.final_answer}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Vote distribution: {result.answer_counts}")
```

**Output**:
```
Answer: 42
Confidence: 60.0%
Vote distribution: {'42': 3, '43': 1, '44': 1}
```

### With Diagnostics

```python
result = self_consistency_baseline(...)

print(f"All answers: {result.all_answers}")
print(f"Answer entropy: {result.answer_entropy:.3f} bits")
print(f"Total tokens: {result.total_tokens}")
print(f"Avg tokens/sample: {result.avg_tokens_per_sample:.1f}")

if result.confidence < 0.5:
    print("⚠️  Low confidence - no clear majority")

if result.answer_entropy > 1.5:
    print("⚠️  High diversity - model very uncertain")
```

### Compare with CoT

```python
from controls import compare_self_consistency_vs_cot

comparison = compare_self_consistency_vs_cot(
    model,
    tokenizer,
    prompt,
    n_samples=5,
    ground_truth="42"
)

# Outputs:
# Self-Consistency: answer=42, confidence=60%, tokens=1250
# CoT: answer=42, tokens=450
# Analysis: SC uses 2.8x tokens but has higher confidence
```

## Integration with Router

### Strategy: Three-Tier Routing

```python
# Low entropy → Direct
if H_int < 0.5:
    answer = router.generate(question, force_route='direct')

# Medium entropy → CoT
elif H_int < 1.2:
    answer = router.generate(question, force_route='cot')

# High entropy → Self-Consistency
else:
    result = self_consistency_baseline(
        model, tokenizer, prompt,
        n_samples=5,
        temperature=0.7
    )
    answer = result.final_answer
```

**Rationale**:
- **Direct** (H < 0.5): Model confident, no reasoning needed
- **CoT** (0.5 ≤ H < 1.2): Model uncertain, reasoning helps
- **Self-Consistency** (H ≥ 1.2): Model very uncertain, need robustness

### Token Budget

| Strategy | Tokens/Query | When to Use |
|----------|-------------|-------------|
| Direct | ~50 | H < 0.5 |
| CoT | ~340 | 0.5 ≤ H < 1.2 |
| Self-Consistency (N=5) | ~1250 | H ≥ 1.2 |

**Trade-off**:
- Self-consistency is expensive (3.7x CoT)
- But more robust for very uncertain problems
- Use sparingly (only top ~10% hardest problems)

## Experimental Results

### Hypothesis

> "Self-consistency provides a stronger control than babble by generating
>  N genuine attempts to solve, making it a fair comparison to CoT."

### Expected Results (GSM8K, 200 problems)

| Method | Accuracy | Avg Tokens | Efficiency |
|--------|----------|------------|------------|
| Babble | 0% | 200 | 0.000 |
| Self-Consistency (N=3) | 68% | 750 | 0.091 |
| Self-Consistency (N=5) | 72% | 1250 | 0.058 |
| CoT | 78% | 340 | 0.229 |

**Key findings**:
- Self-consistency >> babble (meaningful baseline)
- Self-consistency approaches CoT accuracy
- But less efficient (3.7x tokens for +4 pp accuracy)

### N Samples Trade-off

| N | Accuracy | Total Tokens | Efficiency | Notes |
|---|----------|--------------|------------|-------|
| 1 | 62% | 250 | 0.248 | No benefit |
| 3 | 68% | 750 | 0.091 | Good trade-off |
| 5 | 72% | 1250 | 0.058 | Diminishing returns |
| 10 | 74% | 2500 | 0.030 | Not worth it |

**Optimal**: N=3-5 samples provides good balance.

## Comparison: Babble vs Self-Consistency vs CoT

### Babble (Original Baseline)

```
Prompt: "Solve: What is 25% of 80?"
Babble output: "Numbers and percentages... thinking about math...
                 twenty-five and eighty... calculations..."
Answer extracted: [None]
Correct: ❌ (meaningless)
```

**Issues**:
- Not a genuine attempt to solve
- Wastes tokens without providing insight
- Can't distinguish competence from compliance

### Self-Consistency (Our Control)

```
Prompt: "Solve: What is 25% of 80?"

Sample 1: "25% of 80 is 20"         → Answer: 20 ✓
Sample 2: "0.25 * 80 = 20"          → Answer: 20 ✓
Sample 3: "80/4 = 20"               → Answer: 20 ✓
Sample 4: "25/100 * 80 = 20"        → Answer: 20 ✓
Sample 5: "Quarter of 80 is 20"     → Answer: 20 ✓

Majority vote: 20 (confidence: 100%)
Correct: ✓
```

**Advantages**:
- Genuine attempts to solve
- Tests model's raw capability
- Confidence score provides additional signal
- Can identify when model is genuinely uncertain

### Chain-of-Thought (Target)

```
Prompt: "Solve: What is 25% of 80? Show your work."
CoT output: "To find 25% of 80:
             Step 1: Convert 25% to decimal: 25/100 = 0.25
             Step 2: Multiply: 0.25 × 80 = 20

             The answer is 20."
Answer: 20 ✓
Correct: ✓
```

**Advantages**:
- Explicit reasoning shown
- More interpretable
- Single generation (more efficient)

## Best Practices

### 1. Use for High-Uncertainty Problems Only

```python
# Adaptive strategy
if entropy > 1.5:
    # Very uncertain → need robustness
    result = self_consistency_baseline(model, tokenizer, prompt, n_samples=5)
elif entropy > 0.8:
    # Moderately uncertain → reasoning helps
    result = router.generate(question, force_route='cot')
else:
    # Confident → direct answer
    result = router.generate(question, force_route='direct')
```

### 2. Monitor Confidence and Diversity

```python
result = self_consistency_baseline(...)

if result.confidence < 0.4:
    print("⚠️  Warning: No clear consensus")
    # Consider:
    # - Increasing N
    # - Using different temperature
    # - Problem may be genuinely ambiguous

if result.answer_entropy > 2.0:
    print("⚠️  Warning: Very high diversity")
    # Model is very uncertain, answers all over the place
```

### 3. Tune N Based on Budget

```python
# Budget-constrained
n_samples = 3  # Minimum for majority vote

# Standard
n_samples = 5  # Good balance

# High-stakes
n_samples = 7  # More robust, diminishing returns
```

### 4. Temperature Selection

```python
# Conservative (less diversity, more similar)
temperature = 0.5

# Balanced (good diversity + quality)
temperature = 0.7  # Recommended

# Aggressive (high diversity, lower quality)
temperature = 1.0
```

## Limitations

### 1. Expensive

Self-consistency with N=5 uses **3.7x more tokens** than single CoT:
- CoT: 1 generation × 340 tokens = 340 tokens
- Self-Consistency: 5 generations × 250 tokens = 1250 tokens

**Mitigation**: Use only for hardest problems (top 10-20%)

### 2. No Reasoning Trace

Unlike CoT, self-consistency doesn't show reasoning:
- Can't debug why model got answer
- Less interpretable
- Can't learn from reasoning

**Mitigation**: Optionally save all generations for post-hoc analysis

### 3. May Not Help for All Problems

Some problems need reasoning, not diversity:
- Complex multi-step math
- Problems requiring specific domain knowledge
- Questions with multiple valid approaches

**Mitigation**: Combine with CoT (generate reasoning paths, take majority vote)

## Advanced Usage

### Self-Consistency with CoT

Generate N reasoning paths, extract answer from each, take majority vote:

```python
# Modify prompt to request reasoning
cot_prompt = prompt.replace("Answer:", "Let's solve step by step.\n\nAnswer:")

result = self_consistency_baseline(
    model,
    tokenizer,
    cot_prompt,  # Request reasoning in each sample
    n_samples=5,
    temperature=0.7,
    max_new_tokens=400  # Allow longer for reasoning
)

# Now we have:
# - Diverse reasoning paths in result.all_generations
# - Majority vote answer in result.final_answer
# - Confidence in result.confidence
```

### Weighted Voting

Weight answers by confidence or quality:

```python
# Standard majority vote
counts = Counter(answers)  # Equal weight

# Weighted by length (longer = more confident)
weights = [len(gen) for gen in generations]
weighted_counts = defaultdict(float)
for answer, weight in zip(answers, weights):
    weighted_counts[answer] += weight

winner = max(weighted_counts.items(), key=lambda x: x[1])[0]
```

### Iterative Refinement

Use self-consistency iteratively:

```python
# Round 1: Generate N answers
result1 = self_consistency_baseline(..., n_samples=5)

if result1.confidence < 0.6:
    # Round 2: Generate N more answers
    result2 = self_consistency_baseline(..., n_samples=5)

    # Combine votes
    all_answers = result1.all_answers + result2.all_answers
    final_answer, confidence, counts = majority_vote(all_answers)
```

## API Reference

### `self_consistency_baseline(model, tokenizer, prompt, ...)`

Main function for self-consistency generation.

**Parameters**:
- `model`: HuggingFace model
- `tokenizer`: HuggingFace tokenizer
- `prompt` (str): Input prompt
- `n_samples` (int, default=5): Number of samples
- `temperature` (float, default=0.7): Sampling temperature
- `max_new_tokens` (int, default=256): Max tokens per sample
- `benchmark` (str, default='gsm8k'): Benchmark type
- `ground_truth` (str, optional): Ground truth for evaluation
- `verbose` (bool, default=False): Print progress

**Returns**: `SelfConsistencyResult` with:
- `final_answer`: Majority vote answer
- `confidence`: Vote share of winner
- `all_generations`: All generated texts
- `all_answers`: All extracted answers
- `answer_counts`: Vote distribution
- `answer_entropy`: Diversity metric
- `total_tokens`: Total token usage
- `is_correct`: Correctness (if ground_truth provided)

### `majority_vote(answers, benchmark)`

Perform majority vote on answers.

**Returns**: Tuple of (winner, confidence, counts)

### `aggregate_answers(all_answers, benchmark)`

Aggregate answers with entropy computation.

**Returns**: Tuple of (final_answer, confidence, counts, entropy)

### `compare_self_consistency_vs_cot(model, tokenizer, prompt, ...)`

Compare self-consistency vs single CoT generation.

**Returns**: Dictionary with comparison results

## Files

- `src/controls/self_consistency.py`: Implementation
- `src/controls/__init__.py`: Module exports
- `examples/self_consistency_demo.py`: Demos (4 interactive demos)
- `docs/self_consistency.md`: This documentation

## References

- Wang et al. (2022): "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
- Intention Collapse paper: Section on control baselines

## Citation

```bibtex
@article{wang2022self,
  title={Self-Consistency Improves Chain of Thought Reasoning in Language Models},
  author={Wang, Xuezhi and Wei, Jason and Schuurmans, Dale and Le, Quoc and Chi, Ed and Narang, Sharan and Chowdhery, Aakanksha and Zhou, Denny},
  journal={arXiv preprint arXiv:2203.11171},
  year={2022}
}

@misc{self-consistency-control,
  title={Self-Consistency Control Baseline for Intention Collapse},
  note={Task 6 - Rigorous control to replace babble baseline},
  year={2025}
}
```
