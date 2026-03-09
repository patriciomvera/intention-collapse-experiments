"""
Intention Metrics Implementation

This module implements the three core metrics from the Intention Collapse framework:
1. Intention Entropy H_int(I): Shannon entropy of next-token distribution
2. Effective Dimensionality dim_eff(I): PCA-based dimensionality measure  
3. Latent Recoverability Recov(I; Z): Linear probe accuracy

Reference: Section 2.2 "Three Families of Intention Metrics"
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass


@dataclass
class IntentionMetrics:
    """Container for all intention metrics for a single example."""
    entropy: float
    dim_eff: float
    recoverability: Optional[float] = None

    # Multiple-choice specific metrics
    option_normalized_entropy: Optional[float] = None
    option_probs: Optional[Dict[str, float]] = None
    option_probability_mass: Optional[float] = None

    # Additional diagnostic information
    entropy_trajectory: Optional[List[float]] = None
    top_token_probs: Optional[Dict[str, float]] = None
    pca_explained_variance: Optional[np.ndarray] = None


def compute_intention_entropy(
    logits: torch.Tensor,
    top_k: int = 100,
    temperature: float = 1.0
) -> float:
    """
    Compute intention entropy H_int(I) from logits.
    
    The intention entropy measures the "decidedness" of the model's intention
    state. Lower entropy indicates a more decided intention.
    
    Args:
        logits: Model output logits, shape (vocab_size,) or (seq_len, vocab_size)
        top_k: Number of top tokens to consider (for numerical stability)
        temperature: Temperature for softmax (default 1.0)
        
    Returns:
        Shannon entropy in bits
        
    Reference:
        H_int(I) ≜ H[p_θ(y_1 | I, x)]
        "Lower entropy indicates a more decided intention."
    """
    # Handle sequence dimension - take last position
    if logits.dim() == 2:
        logits = logits[-1]
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Get top-k logits for numerical stability
    if top_k > 0 and top_k < logits.size(-1):
        top_logits, _ = torch.topk(scaled_logits, top_k)
        # Apply softmax only to top-k
        probs = F.softmax(top_logits, dim=-1)
    else:
        probs = F.softmax(scaled_logits, dim=-1)
    
    # Compute Shannon entropy: H = -Σ p(x) log₂ p(x)
    # Add small epsilon for numerical stability
    eps = 1e-10
    log_probs = torch.log2(probs + eps)
    entropy = -torch.sum(probs * log_probs).item()
    
    return entropy


def compute_entropy_trajectory(
    all_logits: List[torch.Tensor],
    top_k: int = 100
) -> List[float]:
    """
    Compute entropy at each step of generation.

    This allows us to observe the U-shaped curve predicted by the framework:
    entropy rises during exploration, then falls as intention crystallizes.

    Args:
        all_logits: List of logits at each generation step
        top_k: Number of top tokens to consider

    Returns:
        List of entropy values for each step
    """
    return [compute_intention_entropy(logits, top_k=top_k) for logits in all_logits]


def get_option_token_ids(
    tokenizer,
    options: List[str] = ['A', 'B', 'C', 'D', 'E']
) -> Dict[str, int]:
    """
    Get token IDs for multiple choice options.

    Args:
        tokenizer: HuggingFace tokenizer
        options: List of option letters (default: ['A', 'B', 'C', 'D', 'E'])

    Returns:
        Dictionary mapping option letter to token ID

    Note:
        Some tokenizers may encode 'A' differently than ' A' (with space).
        This function tries both and returns the most likely token ID.
    """
    option_ids = {}

    for option in options:
        # Try different encodings
        candidates = [
            option,           # "A"
            f" {option}",     # " A" (with leading space)
            f"{option}.",     # "A." (with period)
            f" {option}.",    # " A." (with space and period)
        ]

        # Tokenize each candidate and take the first token
        token_ids = []
        for candidate in candidates:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if ids:
                token_ids.append(ids[0])

        # Use the most common token ID (mode)
        if token_ids:
            from collections import Counter
            most_common = Counter(token_ids).most_common(1)[0][0]
            option_ids[option] = most_common

    return option_ids


def compute_option_normalized_entropy(
    logits: torch.Tensor,
    tokenizer,
    options: List[str] = ['A', 'B', 'C', 'D'],
    temperature: float = 1.0,
    return_probs: bool = False
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    Compute intention entropy normalized over valid multiple-choice options only.

    This is the key metric for multiple-choice questions. Instead of computing
    entropy over the entire vocabulary (50k+ tokens), we only consider the
    valid option tokens [A, B, C, D] or [A, B, C, D, E].

    Key Insight:
        - Regular H_int(I) may be high because of vocabulary uncertainty
        - Option-normalized entropy focuses on uncertainty among valid answers
        - This separates "compliance" (format issues) from "competence" (knowledge)

    Args:
        logits: Model output logits, shape (vocab_size,) or (seq_len, vocab_size)
        tokenizer: HuggingFace tokenizer for getting option token IDs
        options: List of valid option letters (default: ['A', 'B', 'C', 'D'])
        temperature: Temperature for softmax (default 1.0)
        return_probs: If True, also return probability distribution over options

    Returns:
        If return_probs=False: Shannon entropy in bits over valid options
        If return_probs=True: Tuple of (entropy, {option: probability})

    Example:
        >>> logits = model(**inputs).logits[:, -1, :]
        >>> entropy = compute_option_normalized_entropy(logits, tokenizer, ['A','B','C','D'])
        >>> # entropy is now only considering uncertainty between A, B, C, D
        >>> # Low entropy → model is confident in one option
        >>> # High entropy (~2 bits for 4 options) → model is guessing uniformly

    Reference:
        "For multiple-choice questions, standard entropy conflates two issues:
         (1) whether the model knows which option is correct (competence)
         (2) whether the model will format its answer correctly (compliance)

         Option-normalized entropy isolates competence by measuring uncertainty
         only over the valid answer space."
    """
    # Handle sequence dimension - take last position
    if logits.dim() == 2:
        logits = logits[-1]

    # Get token IDs for each option
    option_ids = get_option_token_ids(tokenizer, options)

    if len(option_ids) == 0:
        raise ValueError(f"Could not find token IDs for any options: {options}")

    # Extract logits for option tokens only
    option_logits = []
    valid_options = []

    for option in options:
        if option in option_ids:
            token_id = option_ids[option]
            option_logits.append(logits[token_id])
            valid_options.append(option)

    if len(option_logits) == 0:
        raise ValueError(f"No valid option tokens found in vocabulary")

    # Stack into tensor
    option_logits = torch.stack(option_logits)

    # Apply temperature scaling
    scaled_logits = option_logits / temperature

    # Softmax over options only
    probs = F.softmax(scaled_logits, dim=-1)

    # Compute Shannon entropy: H = -Σ p(x) log₂ p(x)
    eps = 1e-10
    log_probs = torch.log2(probs + eps)
    entropy = -torch.sum(probs * log_probs).item()

    if return_probs:
        # Convert to dictionary
        prob_dict = {opt: prob.item() for opt, prob in zip(valid_options, probs)}
        return entropy, prob_dict

    return entropy


def compute_entropy_decomposition(
    logits: torch.Tensor,
    tokenizer,
    options: List[str] = ['A', 'B', 'C', 'D'],
    top_k: int = 100
) -> Dict[str, float]:
    """
    Decompose entropy into option-normalized vs residual components.

    This diagnostic helps understand whether high entropy is due to:
    - Uncertainty about the answer (high option-normalized entropy)
    - Formatting/compliance issues (high residual entropy)

    Args:
        logits: Model output logits
        tokenizer: HuggingFace tokenizer
        options: Valid option letters
        top_k: Top-k for standard entropy

    Returns:
        Dictionary with:
        - 'standard': Standard H_int(I) over top-k tokens
        - 'option_normalized': Entropy over valid options only
        - 'option_probability_mass': Total probability on valid options
        - 'ratio': option_normalized / standard

    Interpretation:
        - ratio ≈ 1.0 → Most uncertainty is about which option (competence issue)
        - ratio << 1.0 → Much uncertainty is outside options (compliance issue)
        - option_probability_mass < 0.5 → Model not focused on valid options
    """
    # Compute standard entropy
    standard_entropy = compute_intention_entropy(logits, top_k=top_k)

    # Compute option-normalized entropy and probabilities
    option_entropy, option_probs = compute_option_normalized_entropy(
        logits, tokenizer, options, return_probs=True
    )

    # Total probability mass on valid options
    option_prob_mass = sum(option_probs.values())

    # Compute ratio
    ratio = option_entropy / standard_entropy if standard_entropy > 0 else 0.0

    return {
        'standard': standard_entropy,
        'option_normalized': option_entropy,
        'option_probability_mass': option_prob_mass,
        'ratio': ratio,
        'option_probs': option_probs,
        'most_likely_option': max(option_probs.items(), key=lambda x: x[1])[0],
        'confidence': max(option_probs.values())  # Probability of most likely option
    }


def compute_effective_dimensionality(
    activations: np.ndarray,
    variance_threshold: float = 0.90
) -> Tuple[int, np.ndarray]:
    """
    Compute effective intention dimensionality dim_eff(I) using PCA.
    
    Higher dim_eff correlates with multi-faceted reasoning and predicts
    correct final answers on math and code benchmarks.
    
    Args:
        activations: Hidden state activations, shape (n_samples, hidden_dim)
                    or (n_layers, n_samples, hidden_dim)
        variance_threshold: Fraction of variance to capture (default 0.90)
        
    Returns:
        Tuple of (effective_dim, explained_variance_ratio)
        
    Reference:
        "The effective dimensionality is the smallest k such that
         Σᵢ₌₁ᵏ λᵢ / Σⱼ λⱼ ≥ 0.9"
    """
    # Flatten if multi-layer
    if activations.ndim == 3:
        n_layers, n_samples, hidden_dim = activations.shape
        activations = activations.reshape(n_layers * n_samples, hidden_dim)
    
    # Ensure we have enough samples
    n_samples, n_features = activations.shape
    n_components = min(n_samples, n_features)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(activations)
    
    # Compute cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find smallest k such that cumulative variance >= threshold
    dim_eff = np.searchsorted(cumulative_variance, variance_threshold) + 1
    
    # Ensure we don't exceed the number of components
    dim_eff = min(dim_eff, n_components)
    
    return dim_eff, pca.explained_variance_ratio_


def compute_participation_ratio(activations: np.ndarray) -> float:
    """
    Alternative dimensionality measure: Participation Ratio.
    
    PR = (Σ λᵢ)² / Σ λᵢ²
    
    This gives a continuous measure of effective dimensionality,
    avoiding the need for an arbitrary variance threshold.
    
    Args:
        activations: Hidden state activations, shape (n_samples, hidden_dim)
        
    Returns:
        Participation ratio (effective number of dimensions)
    """
    if activations.ndim == 3:
        n_layers, n_samples, hidden_dim = activations.shape
        activations = activations.reshape(n_layers * n_samples, hidden_dim)
    
    # Compute covariance and eigenvalues
    centered = activations - activations.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    
    # Filter out near-zero eigenvalues
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # Participation ratio
    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    
    return pr


def get_top_token_probabilities(
    logits: torch.Tensor,
    tokenizer,
    k: int = 10
) -> Dict[str, float]:
    """
    Get the top-k tokens and their probabilities.
    
    Useful for understanding what alternatives the model considered
    before collapse.
    
    Args:
        logits: Model output logits for a single position
        tokenizer: Tokenizer for decoding token IDs
        k: Number of top tokens to return
        
    Returns:
        Dictionary mapping token strings to probabilities
    """
    if logits.dim() == 2:
        logits = logits[-1]
    
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k)
    
    result = {}
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        token = tokenizer.decode([idx])
        result[token] = prob
    
    return result


class IntentionMetricsComputer:
    """
    Unified class for computing all intention metrics.
    
    This class handles the extraction and computation of metrics
    for both baseline and enhanced (CoT) conditions.
    """
    
    def __init__(
        self,
        variance_threshold: float = 0.90,
        entropy_top_k: int = 100,
        tokenizer = None
    ):
        """
        Initialize the metrics computer.
        
        Args:
            variance_threshold: Threshold for dim_eff computation
            entropy_top_k: Number of top tokens for entropy
            tokenizer: Optional tokenizer for decoding (for diagnostics)
        """
        self.variance_threshold = variance_threshold
        self.entropy_top_k = entropy_top_k
        self.tokenizer = tokenizer
        
        # Storage for batch computation
        self._activations_buffer = []
        self._logits_buffer = []
    
    def compute_single(
        self,
        logits: torch.Tensor,
        activations: np.ndarray,
        include_diagnostics: bool = False
    ) -> IntentionMetrics:
        """
        Compute all metrics for a single example.
        
        Args:
            logits: Model output logits
            activations: Hidden state activations
            include_diagnostics: Whether to include additional info
            
        Returns:
            IntentionMetrics dataclass with all metrics
        """
        # Compute entropy
        entropy = compute_intention_entropy(
            logits, 
            top_k=self.entropy_top_k
        )
        
        # Compute effective dimensionality
        # For single example, use activation sequence as samples
        if activations.ndim == 1:
            activations = activations.reshape(1, -1)
        
        dim_eff, explained_var = compute_effective_dimensionality(
            activations,
            variance_threshold=self.variance_threshold
        )
        
        metrics = IntentionMetrics(
            entropy=entropy,
            dim_eff=dim_eff
        )
        
        if include_diagnostics:
            metrics.pca_explained_variance = explained_var
            if self.tokenizer is not None:
                metrics.top_token_probs = get_top_token_probabilities(
                    logits, self.tokenizer
                )
        
        return metrics
    
    def compute_batch(
        self,
        all_logits: List[torch.Tensor],
        all_activations: List[np.ndarray]
    ) -> Tuple[List[IntentionMetrics], int, np.ndarray]:
        """
        Compute metrics for a batch and also compute global dim_eff.
        
        The global dim_eff is computed across all examples to capture
        the overall richness of the intention space.
        
        Args:
            all_logits: List of logits for each example
            all_activations: List of activations for each example
            
        Returns:
            Tuple of (individual_metrics, global_dim_eff, global_explained_var)
        """
        individual_metrics = []
        
        for logits, acts in zip(all_logits, all_activations):
            metrics = self.compute_single(logits, acts)
            individual_metrics.append(metrics)
        
        # Compute global dimensionality across all examples
        stacked_activations = np.vstack(all_activations)
        global_dim_eff, global_explained_var = compute_effective_dimensionality(
            stacked_activations,
            variance_threshold=self.variance_threshold
        )
        
        return individual_metrics, global_dim_eff, global_explained_var


def compare_conditions(
    baseline_metrics: List[IntentionMetrics],
    enhanced_metrics: List[IntentionMetrics],
    baseline_correct: List[bool],
    enhanced_correct: List[bool]
) -> Dict[str, Dict[str, float]]:
    """
    Compare metrics between baseline and enhanced conditions.
    
    Args:
        baseline_metrics: Metrics from baseline condition
        enhanced_metrics: Metrics from enhanced (CoT) condition
        baseline_correct: Correctness labels for baseline
        enhanced_correct: Correctness labels for enhanced
        
    Returns:
        Dictionary with statistical comparisons
    """
    results = {}
    
    # Extract metric values
    baseline_entropy = [m.entropy for m in baseline_metrics]
    enhanced_entropy = [m.entropy for m in enhanced_metrics]
    baseline_dim = [m.dim_eff for m in baseline_metrics]
    enhanced_dim = [m.dim_eff for m in enhanced_metrics]
    
    # Basic statistics
    results['entropy'] = {
        'baseline_mean': np.mean(baseline_entropy),
        'baseline_std': np.std(baseline_entropy),
        'enhanced_mean': np.mean(enhanced_entropy),
        'enhanced_std': np.std(enhanced_entropy),
        'diff': np.mean(enhanced_entropy) - np.mean(baseline_entropy)
    }
    
    results['dim_eff'] = {
        'baseline_mean': np.mean(baseline_dim),
        'baseline_std': np.std(baseline_dim),
        'enhanced_mean': np.mean(enhanced_dim),
        'enhanced_std': np.std(enhanced_dim),
        'diff': np.mean(enhanced_dim) - np.mean(baseline_dim)
    }
    
    # Accuracy
    results['accuracy'] = {
        'baseline': np.mean(baseline_correct),
        'enhanced': np.mean(enhanced_correct),
        'improvement': np.mean(enhanced_correct) - np.mean(baseline_correct)
    }
    
    # Correlation with correctness
    from scipy import stats
    
    # Point-biserial correlation for entropy vs correctness
    r_entropy_base, p_entropy_base = stats.pointbiserialr(
        baseline_correct, baseline_entropy
    )
    r_entropy_enh, p_entropy_enh = stats.pointbiserialr(
        enhanced_correct, enhanced_entropy
    )
    
    results['correlations'] = {
        'entropy_correctness_baseline': {'r': r_entropy_base, 'p': p_entropy_base},
        'entropy_correctness_enhanced': {'r': r_entropy_enh, 'p': p_entropy_enh}
    }
    
    return results
