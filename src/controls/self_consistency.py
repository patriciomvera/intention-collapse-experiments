"""
Self-Consistency Control Baseline

Implements self-consistency (Wang et al., 2022) as a rigorous control
baseline for intention collapse experiments.

Key Insight:
    Instead of "babble" (meaningless output), use self-consistency:
    - Generate N diverse answers with sampling (temperature > 0)
    - Extract answer from each generation
    - Take majority vote

    This provides a stronger baseline that uses the same number of tokens
    as enhanced (CoT) but without explicit reasoning.

Reference:
    Wang et al. (2022): "Self-Consistency Improves Chain of Thought
    Reasoning in Language Models"

Comparison:
    - Babble: "Stream of consciousness about numbers..." (meaningless)
    - Self-Consistency: N independent answers + majority vote (rigorous)
    - CoT: Explicit reasoning then answer (our comparison target)
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
from collections import Counter
import re


@dataclass
class SelfConsistencyResult:
    """Result from self-consistency generation."""

    # Aggregated result
    final_answer: str
    confidence: float  # Vote share of winning answer (e.g., 0.6 = 3/5)

    # Individual generations
    all_generations: List[str]
    all_answers: List[str]
    answer_counts: Dict[str, int]

    # Metadata
    n_samples: int
    temperature: float
    total_tokens: int
    avg_tokens_per_sample: float

    # Entropy of answer distribution (higher = more diverse)
    answer_entropy: float

    # For evaluation
    is_correct: Optional[bool] = None
    ground_truth: Optional[str] = None


def extract_answer_from_generation(
    generated_text: str,
    benchmark: str = "gsm8k"
) -> str:
    """
    Extract answer from a single generation.

    Args:
        generated_text: Generated text
        benchmark: Benchmark type ('gsm8k', 'arc', 'aqua')

    Returns:
        Extracted answer string
    """
    # Pattern 1: "#### X" format (GSM8K style)
    pattern_hash = r'####\s*([^\n]+)'
    match = re.search(pattern_hash, generated_text)
    if match:
        return match.group(1).strip()

    # Pattern 2: "\boxed{X}" format (MATH style)
    pattern_box = r'\\boxed\{([^}]+)\}'
    match = re.search(pattern_box, generated_text)
    if match:
        return match.group(1).strip()

    # Pattern 3: "The answer is X" or "Final answer: X"
    pattern_answer = r'(?:answer is|final answer:|answer:)\s*([A-E\d\.,\-]+)'
    match = re.search(pattern_answer, generated_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 4: For MC questions, look for standalone letter
    if benchmark in ['arc', 'aqua']:
        pattern_letter = r'\b([A-E])\b'
        matches = re.findall(pattern_letter, generated_text)
        if matches:
            return matches[-1]  # Take last occurrence

    # Pattern 5: Last number (for math problems)
    if benchmark in ['gsm8k', 'math']:
        pattern_number = r'-?\d+\.?\d*'
        numbers = re.findall(pattern_number, generated_text)
        if numbers:
            return numbers[-1]

    # Fallback: empty string
    return ""


def normalize_answer(answer: str, benchmark: str = "gsm8k") -> str:
    """
    Normalize answer for comparison.

    Args:
        answer: Answer string
        benchmark: Benchmark type

    Returns:
        Normalized answer
    """
    answer = answer.strip().lower()

    # Remove common punctuation
    answer = answer.replace(',', '').replace('.', '').replace(' ', '')

    # For numbers, convert to float then back to string to handle "42.0" vs "42"
    if benchmark in ['gsm8k', 'math']:
        try:
            num = float(answer)
            # Remove trailing zeros
            if num == int(num):
                answer = str(int(num))
            else:
                answer = str(num)
        except (ValueError, OverflowError):
            pass

    # For MC, just take first letter
    if benchmark in ['arc', 'aqua']:
        if answer and answer[0] in 'abcde':
            answer = answer[0]

    return answer


def majority_vote(
    answers: List[str],
    benchmark: str = "gsm8k"
) -> Tuple[str, float, Dict[str, int]]:
    """
    Perform majority vote on answers.

    Args:
        answers: List of answer strings
        benchmark: Benchmark type

    Returns:
        Tuple of (winning_answer, confidence, answer_counts)

    Confidence:
        - 1.0 = unanimous
        - 0.5 = tie (2/4)
        - 0.6 = 3/5
    """
    # Normalize all answers
    normalized = [normalize_answer(a, benchmark) for a in answers]

    # Filter out empty answers
    normalized = [a for a in normalized if a]

    if not normalized:
        return "", 0.0, {}

    # Count occurrences
    counts = Counter(normalized)

    # Get winner
    most_common = counts.most_common(1)
    if most_common:
        winner, count = most_common[0]
        confidence = count / len(normalized)
    else:
        winner = ""
        confidence = 0.0

    return winner, confidence, dict(counts)


def compute_answer_entropy(answer_counts: Dict[str, int]) -> float:
    """
    Compute entropy of answer distribution.

    High entropy = diverse answers (model uncertain)
    Low entropy = consensus (model confident)

    Args:
        answer_counts: Dictionary of answer -> count

    Returns:
        Entropy in bits
    """
    if not answer_counts:
        return 0.0

    total = sum(answer_counts.values())
    probs = [count / total for count in answer_counts.values()]

    # Shannon entropy
    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)

    return entropy


def aggregate_answers(
    all_answers: List[str],
    benchmark: str = "gsm8k"
) -> Tuple[str, float, Dict[str, int], float]:
    """
    Aggregate multiple answers via majority vote.

    Args:
        all_answers: List of extracted answers
        benchmark: Benchmark type

    Returns:
        Tuple of (final_answer, confidence, counts, entropy)
    """
    final_answer, confidence, counts = majority_vote(all_answers, benchmark)
    entropy = compute_answer_entropy(counts)

    return final_answer, confidence, counts, entropy


def self_consistency_baseline(
    model,
    tokenizer,
    prompt: str,
    n_samples: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    benchmark: str = "gsm8k",
    ground_truth: Optional[str] = None,
    verbose: bool = False,
    **generation_kwargs
) -> SelfConsistencyResult:
    """
    Generate answer using self-consistency (majority vote over multiple samples).

    This is a rigorous control baseline that uses the same total number of
    tokens as enhanced/CoT conditions but generates N independent answers
    instead of one reasoned answer.

    Process:
    1. Generate N answers with sampling (temperature > 0)
    2. Extract answer from each generation
    3. Take majority vote
    4. Return aggregated result

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        n_samples: Number of samples to generate (default: 5)
        temperature: Sampling temperature (default: 0.7)
        max_new_tokens: Max tokens per sample
        benchmark: Benchmark type ('gsm8k', 'arc', 'aqua')
        ground_truth: Ground truth answer (for evaluation)
        verbose: Print progress
        **generation_kwargs: Additional args for model.generate()

    Returns:
        SelfConsistencyResult with aggregated answer and metadata

    Example:
        >>> result = self_consistency_baseline(
        ...     model, tokenizer, prompt,
        ...     n_samples=5,
        ...     temperature=0.7
        ... )
        >>> print(f"Final answer: {result.final_answer}")
        >>> print(f"Confidence: {result.confidence:.1%}")
        >>> print(f"Vote distribution: {result.answer_counts}")

    Reference:
        Wang et al. (2022): "Self-Consistency Improves Chain of Thought
        Reasoning in Language Models"

        Key insight: Diverse reasoning paths can converge to same answer.
        Majority vote over multiple samples improves robustness.
    """
    if verbose:
        print(f"\nSelf-Consistency: generating {n_samples} samples...")

    all_generations = []
    all_answers = []
    total_tokens = 0

    # Tokenize prompt once
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    # Generate N samples
    for i in range(n_samples):
        if verbose:
            print(f"  Sample {i+1}/{n_samples}...", end='')

        # Generation config
        gen_kwargs = {
            'max_new_tokens': max_new_tokens,
            'do_sample': True,  # Enable sampling
            'temperature': temperature,
            'top_p': 0.95,  # Nucleus sampling
            'pad_token_id': tokenizer.eos_token_id,
            **generation_kwargs
        }

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # Decode
        generated_ids = outputs[0, input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract answer
        answer = extract_answer_from_generation(generated_text, benchmark)

        all_generations.append(generated_text)
        all_answers.append(answer)
        total_tokens += input_length + len(generated_ids)

        if verbose:
            print(f" answer: {answer}")

    # Aggregate via majority vote
    final_answer, confidence, counts, entropy = aggregate_answers(
        all_answers, benchmark
    )

    if verbose:
        print(f"\n  Majority vote: {final_answer} (confidence: {confidence:.1%})")
        print(f"  Vote distribution: {counts}")
        print(f"  Answer entropy: {entropy:.3f} bits")

    # Evaluate if ground truth provided
    is_correct = None
    if ground_truth is not None:
        is_correct = normalize_answer(final_answer, benchmark) == normalize_answer(ground_truth, benchmark)
        if verbose:
            print(f"  Correct: {is_correct}")

    # Compute average tokens per sample
    avg_tokens_per_sample = total_tokens / n_samples

    result = SelfConsistencyResult(
        final_answer=final_answer,
        confidence=confidence,
        all_generations=all_generations,
        all_answers=all_answers,
        answer_counts=counts,
        n_samples=n_samples,
        temperature=temperature,
        total_tokens=total_tokens,
        avg_tokens_per_sample=avg_tokens_per_sample,
        answer_entropy=entropy,
        is_correct=is_correct,
        ground_truth=ground_truth
    )

    return result


def compare_self_consistency_vs_cot(
    model,
    tokenizer,
    prompt: str,
    n_samples: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    benchmark: str = "gsm8k",
    ground_truth: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compare self-consistency vs single CoT generation.

    This is useful for understanding the trade-off:
    - Self-consistency: N diverse samples, majority vote
    - CoT: 1 reasoned sample, longer generation

    Args:
        model: Model
        tokenizer: Tokenizer
        prompt: Prompt (should request reasoning for CoT)
        n_samples: Samples for self-consistency
        temperature: Temperature for self-consistency
        max_new_tokens: Max tokens per sample
        benchmark: Benchmark type
        ground_truth: Ground truth
        verbose: Print details

    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*80)
    print("COMPARISON: Self-Consistency vs CoT")
    print("="*80)

    # 1. Self-Consistency
    print("\n1. Self-Consistency (N independent samples + majority vote):")
    sc_result = self_consistency_baseline(
        model, tokenizer, prompt,
        n_samples=n_samples,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        benchmark=benchmark,
        ground_truth=ground_truth,
        verbose=verbose
    )

    # 2. CoT (single sample, greedy, longer generation)
    print("\n2. Chain-of-Thought (1 reasoned sample, greedy):")

    # Adjust prompt for CoT (ensure it asks for reasoning)
    cot_prompt = prompt
    if "step by step" not in prompt.lower():
        # Add reasoning instruction if not present
        cot_prompt = prompt.replace("Answer:", "Let's solve this step by step.\n\nAnswer:")

    inputs = tokenizer(cot_prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    # Generate with greedy decoding (temperature=0)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens * 2,  # Allow longer for reasoning
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0, input_length:]
    cot_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    cot_answer = extract_answer_from_generation(cot_text, benchmark)
    cot_tokens = input_length + len(generated_ids)

    cot_correct = None
    if ground_truth:
        cot_correct = normalize_answer(cot_answer, benchmark) == normalize_answer(ground_truth, benchmark)

    if verbose:
        print(f"  Answer: {cot_answer}")
        print(f"  Tokens: {cot_tokens}")
        if cot_correct is not None:
            print(f"  Correct: {cot_correct}")

    # Comparison
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)

    comparison = {
        'self_consistency': {
            'answer': sc_result.final_answer,
            'confidence': sc_result.confidence,
            'total_tokens': sc_result.total_tokens,
            'avg_tokens_per_sample': sc_result.avg_tokens_per_sample,
            'answer_entropy': sc_result.answer_entropy,
            'correct': sc_result.is_correct
        },
        'cot': {
            'answer': cot_answer,
            'total_tokens': cot_tokens,
            'correct': cot_correct
        }
    }

    print(f"\nSelf-Consistency:")
    print(f"  Answer: {sc_result.final_answer}")
    print(f"  Confidence: {sc_result.confidence:.1%}")
    print(f"  Total tokens: {sc_result.total_tokens}")
    print(f"  Correct: {sc_result.is_correct}")

    print(f"\nChain-of-Thought:")
    print(f"  Answer: {cot_answer}")
    print(f"  Total tokens: {cot_tokens}")
    print(f"  Correct: {cot_correct}")

    # Analysis
    print(f"\nANALYSIS:")

    if sc_result.total_tokens and cot_tokens:
        token_ratio = sc_result.total_tokens / cot_tokens
        print(f"  Token ratio (SC/CoT): {token_ratio:.2f}x")

    if sc_result.is_correct is not None and cot_correct is not None:
        if sc_result.is_correct and cot_correct:
            print(f"  ✅ Both correct!")
        elif sc_result.is_correct and not cot_correct:
            print(f"  ✅ Self-Consistency correct, CoT wrong!")
        elif not sc_result.is_correct and cot_correct:
            print(f"  ⚠️  CoT correct, Self-Consistency wrong")
        else:
            print(f"  ❌ Both wrong")

    if sc_result.confidence < 0.5:
        print(f"  ⚠️  Low confidence ({sc_result.confidence:.1%}) - no clear majority")
    elif sc_result.confidence == 1.0:
        print(f"  ✅ Perfect consensus (all samples agree)")

    if sc_result.answer_entropy > 1.5:
        print(f"  ⚠️  High answer diversity (entropy={sc_result.answer_entropy:.2f})")

    print("="*80)

    return comparison
