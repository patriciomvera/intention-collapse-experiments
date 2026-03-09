"""
Adaptive Inference Routing

This module implements an adaptive routing system that uses intention entropy
H_int(I) as a "traffic cop" to decide between inference strategies:
- Direct answer (low entropy → model is confident)
- Chain-of-Thought (high entropy → model needs reasoning)

This transforms the observational Intention Collapse framework into a
practical engineering tool for cost-effective inference.
"""

from .adaptive_router import (
    AdaptiveInferenceRouter,
    RouteDecision,
    RouterResult
)

__all__ = [
    'AdaptiveInferenceRouter',
    'RouteDecision',
    'RouterResult'
]
