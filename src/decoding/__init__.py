"""
Constrained Decoding for Multiple-Choice Questions

This module implements constrained decoding to force the model to generate
only valid multiple-choice option tokens (A, B, C, D, E).

This solves "compliance" issues where the model knows the answer but
generates invalid format.
"""

from .constrained import (
    constrained_mc_generation,
    MultipleChoiceLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    extract_first_option_token
)

__all__ = [
    'constrained_mc_generation',
    'MultipleChoiceLogitsProcessor',
    'PrefixConstrainedLogitsProcessor',
    'extract_first_option_token'
]
