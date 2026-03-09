"""
Intention Collapse Experiments

A framework for measuring and analyzing intention states in Large Language Models.

Usage:
    from src.router import AdaptiveInferenceRouter
    from src.metrics import compute_intention_entropy
    from src.controls import self_consistency_baseline
    from src.decoding import constrained_mc_generation
"""

__version__ = "2.0.0"
__author__ = "Intention Collapse Research Team"

# Lazy imports - only load when explicitly accessed
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

def __getattr__(name):
    """Lazy import pattern - only load modules when accessed."""
    if name in ["compute_intention_entropy", "compute_effective_dimensionality", "IntentionMetrics"]:
        from .metrics import compute_intention_entropy, compute_effective_dimensionality, IntentionMetrics
        return locals()[name]
    elif name == "ActivationExtractor":
        from .activation_hooks import ActivationExtractor
        return ActivationExtractor
    elif name in ["LinearProbe", "train_recoverability_probe"]:
        from .probing import LinearProbe, train_recoverability_probe
        return locals()[name]
    elif name in ["load_gsm8k", "extract_answer", "evaluate_answer"]:
        from .data_utils import load_gsm8k, extract_answer, evaluate_answer
        return locals()[name]
    elif name in ["plot_metrics_comparison", "plot_correlation_matrix", "plot_entropy_trajectory", "create_publication_figure"]:
        from .visualization import plot_metrics_comparison, plot_correlation_matrix, plot_entropy_trajectory, create_publication_figure
        return locals()[name]
    raise AttributeError(f"module 'src' has no attribute '{name}'")
