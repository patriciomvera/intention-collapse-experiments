"""
Unit tests for option-normalized entropy functions

Run with:
    pytest tests/test_option_normalized_entropy.py
    or
    python tests/test_option_normalized_entropy.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from unittest.mock import Mock


def test_option_normalized_entropy_uniform():
    """Test that uniform distribution over 4 options gives log2(4) = 2.0 bits."""
    from metrics import compute_option_normalized_entropy

    # Create uniform logits for 4 options
    vocab_size = 50000
    logits = torch.zeros(vocab_size)

    # Mock tokenizer that returns simple token IDs
    tokenizer = Mock()
    tokenizer.encode = Mock(side_effect=lambda x, **kwargs: [ord(x.strip())])

    # Set option logits to be equal (uniform distribution)
    option_ids = {
        'A': ord('A'),
        'B': ord('B'),
        'C': ord('C'),
        'D': ord('D')
    }

    for token_id in option_ids.values():
        logits[token_id] = 0.0  # All equal logits

    # Compute entropy
    entropy = compute_option_normalized_entropy(
        logits, tokenizer, options=['A', 'B', 'C', 'D']
    )

    # Should be log2(4) = 2.0 bits
    expected = np.log2(4)
    assert abs(entropy - expected) < 0.01, f"Expected {expected}, got {entropy}"
    print(f"✓ Uniform distribution test passed: {entropy:.3f} bits ≈ {expected:.3f} bits")


def test_option_normalized_entropy_certain():
    """Test that certain distribution (p=1 for one option) gives 0 bits."""
    from metrics import compute_option_normalized_entropy

    vocab_size = 50000
    logits = torch.full((vocab_size,), -100.0)  # Very low logits

    tokenizer = Mock()
    tokenizer.encode = Mock(side_effect=lambda x, **kwargs: [ord(x.strip())])

    option_ids = {
        'A': ord('A'),
        'B': ord('B'),
        'C': ord('C'),
        'D': ord('D')
    }

    # Make option A have very high logit (certain)
    logits[option_ids['A']] = 100.0
    for opt in ['B', 'C', 'D']:
        logits[option_ids[opt]] = -100.0

    entropy = compute_option_normalized_entropy(
        logits, tokenizer, options=['A', 'B', 'C', 'D']
    )

    # Should be close to 0 bits
    assert entropy < 0.1, f"Expected ~0, got {entropy}"
    print(f"✓ Certain distribution test passed: {entropy:.3f} bits ≈ 0.0 bits")


def test_option_normalized_entropy_with_probs():
    """Test that return_probs=True returns valid probability distribution."""
    from metrics import compute_option_normalized_entropy

    vocab_size = 50000
    logits = torch.zeros(vocab_size)

    tokenizer = Mock()
    tokenizer.encode = Mock(side_effect=lambda x, **kwargs: [ord(x.strip())])

    option_ids = {
        'A': ord('A'),
        'B': ord('B'),
        'C': ord('C'),
        'D': ord('D')
    }

    # Set different logits for each option
    logits[option_ids['A']] = 2.0
    logits[option_ids['B']] = 1.0
    logits[option_ids['C']] = 0.5
    logits[option_ids['D']] = 0.0

    entropy, probs = compute_option_normalized_entropy(
        logits, tokenizer, options=['A', 'B', 'C', 'D'], return_probs=True
    )

    # Check that probabilities sum to 1
    prob_sum = sum(probs.values())
    assert abs(prob_sum - 1.0) < 0.001, f"Probabilities should sum to 1, got {prob_sum}"

    # Check that all probabilities are in [0, 1]
    for opt, prob in probs.items():
        assert 0 <= prob <= 1, f"Probability for {opt} out of bounds: {prob}"

    # Check that A has highest probability (highest logit)
    assert probs['A'] == max(probs.values()), "Option A should have highest probability"

    print(f"✓ Probability distribution test passed:")
    for opt, prob in sorted(probs.items()):
        print(f"    {opt}: {prob:.3f}")


def test_entropy_decomposition():
    """Test entropy decomposition function."""
    from metrics import compute_entropy_decomposition

    vocab_size = 50000
    logits = torch.randn(vocab_size)  # Random logits

    tokenizer = Mock()
    tokenizer.encode = Mock(side_effect=lambda x, **kwargs: [ord(x.strip())])

    option_ids = {
        'A': ord('A'),
        'B': ord('B'),
        'C': ord('C'),
        'D': ord('D')
    }

    # Give options high logits (so they have significant probability mass)
    for token_id in option_ids.values():
        logits[token_id] = 5.0

    decomp = compute_entropy_decomposition(
        logits, tokenizer, options=['A', 'B', 'C', 'D'], top_k=100
    )

    # Check that all required keys are present
    required_keys = [
        'standard', 'option_normalized', 'ratio',
        'option_probability_mass', 'option_probs',
        'most_likely_option', 'confidence'
    ]
    for key in required_keys:
        assert key in decomp, f"Missing key: {key}"

    # Check that ratio is in reasonable range
    assert 0 <= decomp['ratio'] <= 1, f"Ratio should be in [0,1], got {decomp['ratio']}"

    # Check that probability mass is positive
    assert 0 < decomp['option_probability_mass'] <= 1, \
        f"Probability mass should be in (0,1], got {decomp['option_probability_mass']}"

    # Check that most_likely_option is one of the options
    assert decomp['most_likely_option'] in ['A', 'B', 'C', 'D'], \
        f"Invalid most likely option: {decomp['most_likely_option']}"

    print(f"✓ Entropy decomposition test passed:")
    print(f"    Standard entropy:    {decomp['standard']:.3f} bits")
    print(f"    Option entropy:      {decomp['option_normalized']:.3f} bits")
    print(f"    Ratio:               {decomp['ratio']:.3f}")
    print(f"    Prob mass:           {decomp['option_probability_mass']:.3f}")


def test_get_option_token_ids():
    """Test token ID extraction for options."""
    from metrics import get_option_token_ids

    tokenizer = Mock()

    # Mock encode function that handles different formats
    def mock_encode(text, add_special_tokens=False):
        # Simple mock: return ord of first letter
        if text:
            return [ord(text.strip()[0])]
        return []

    tokenizer.encode = mock_encode

    option_ids = get_option_token_ids(tokenizer, options=['A', 'B', 'C', 'D'])

    # Check that we got IDs for all options
    assert len(option_ids) == 4, f"Expected 4 option IDs, got {len(option_ids)}"

    # Check that all options are present
    for opt in ['A', 'B', 'C', 'D']:
        assert opt in option_ids, f"Missing option: {opt}"

    # Check that IDs are integers
    for opt, token_id in option_ids.items():
        assert isinstance(token_id, int), f"Token ID for {opt} is not an integer: {token_id}"

    print(f"✓ Token ID extraction test passed:")
    for opt, token_id in sorted(option_ids.items()):
        print(f"    {opt} → {token_id}")


def test_five_options():
    """Test with 5 options (AQUA-style)."""
    from metrics import compute_option_normalized_entropy

    vocab_size = 50000
    logits = torch.zeros(vocab_size)

    tokenizer = Mock()
    tokenizer.encode = Mock(side_effect=lambda x, **kwargs: [ord(x.strip())])

    # Uniform distribution over 5 options
    for opt in ['A', 'B', 'C', 'D', 'E']:
        logits[ord(opt)] = 0.0

    entropy = compute_option_normalized_entropy(
        logits, tokenizer, options=['A', 'B', 'C', 'D', 'E']
    )

    # Should be log2(5) ≈ 2.32 bits
    expected = np.log2(5)
    assert abs(entropy - expected) < 0.01, f"Expected {expected}, got {entropy}"
    print(f"✓ Five-option test passed: {entropy:.3f} bits ≈ {expected:.3f} bits")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING OPTION-NORMALIZED ENTROPY TESTS")
    print("="*80 + "\n")

    tests = [
        ("Uniform distribution (4 options)", test_option_normalized_entropy_uniform),
        ("Certain distribution", test_option_normalized_entropy_certain),
        ("Probability distribution", test_option_normalized_entropy_with_probs),
        ("Entropy decomposition", test_entropy_decomposition),
        ("Token ID extraction", test_get_option_token_ids),
        ("Five options (AQUA)", test_five_options),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\nTest: {test_name}")
        print("-" * 80)
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
