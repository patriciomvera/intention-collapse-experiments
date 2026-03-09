"""
Control Baselines for Intention Collapse Experiments

This module provides rigorous control baselines to compare against
enhanced (CoT) conditions.

Traditional "babble" baseline (stream of consciousness without solving)
is replaced with self-consistency: generating multiple answers and
taking majority vote.
"""

from .self_consistency import (
    self_consistency_baseline,
    SelfConsistencyResult,
    aggregate_answers,
    majority_vote,
    extract_answer_from_generation
)

__all__ = [
    'self_consistency_baseline',
    'SelfConsistencyResult',
    'aggregate_answers',
    'majority_vote',
    'extract_answer_from_generation'
]
