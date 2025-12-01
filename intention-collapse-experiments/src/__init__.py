"""
Intention Collapse Experiments

A framework for measuring and analyzing intention states in Large Language Models.
"""

__version__ = "0.1.0"
__author__ = "Intention Collapse Research Team"

from .metrics import (
    compute_intention_entropy,
    compute_effective_dimensionality,
    IntentionMetrics
)
from .activation_hooks import ActivationExtractor
from .probing import LinearProbe, train_recoverability_probe
from .data_utils import load_gsm8k, extract_answer, evaluate_answer
from .visualization import (
    plot_metrics_comparison,
    plot_correlation_matrix,
    plot_entropy_trajectory,
    create_publication_figure
)

__all__ = [
    "compute_intention_entropy",
    "compute_effective_dimensionality", 
    "IntentionMetrics",
    "ActivationExtractor",
    "LinearProbe",
    "train_recoverability_probe",
    "load_gsm8k",
    "extract_answer",
    "evaluate_answer",
    "plot_metrics_comparison",
    "plot_correlation_matrix",
    "plot_entropy_trajectory",
    "create_publication_figure"
]
