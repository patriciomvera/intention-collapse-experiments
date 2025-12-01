"""
Linear Probing for Latent Knowledge Recoverability

This module implements linear probes to measure how much task-relevant
information is present in the intention state I but not necessarily
verbalized in the final output.

Reference: Section 2.2 "Latent Knowledge Recoverability Recov(I; Z)"
    "Train linear or shallow probes on frozen I to predict downstream 
     variables Z (final answer correctness, required lemma, hallucination 
     flag, etc.) that are known to the model but not necessarily verbalized."
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class ProbeResults:
    """Results from training a linear probe."""
    accuracy: float
    cv_scores: np.ndarray
    cv_mean: float
    cv_std: float
    auc_roc: Optional[float] = None
    feature_importance: Optional[np.ndarray] = None
    
    def __str__(self):
        return (
            f"ProbeResults(accuracy={self.accuracy:.3f}, "
            f"cv_mean={self.cv_mean:.3f}±{self.cv_std:.3f})"
        )


class LinearProbe:
    """
    Linear probe for binary classification tasks.
    
    Used to measure latent recoverability by training a simple linear
    classifier on hidden states to predict task-relevant variables.
    """
    
    def __init__(
        self,
        regularization: float = 1.0,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize the linear probe.
        
        Args:
            regularization: L2 regularization strength (C = 1/regularization)
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.regularization = regularization
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Create pipeline with standardization
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=1.0/regularization,
                max_iter=1000,
                random_state=random_state,
                solver='lbfgs'
            ))
        ])
        
        self._is_fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> 'LinearProbe':
        """
        Fit the probe on training data.
        
        Args:
            X: Activation features, shape (n_samples, n_features)
            y: Binary labels, shape (n_samples,)
            
        Returns:
            self
        """
        self.pipeline.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new data."""
        if not self._is_fitted:
            raise ValueError("Probe must be fitted before prediction")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for new data."""
        if not self._is_fitted:
            raise ValueError("Probe must be fitted before prediction")
        return self.pipeline.predict_proba(X)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        compute_auc: bool = True
    ) -> ProbeResults:
        """
        Evaluate probe performance with cross-validation.
        
        Args:
            X: Activation features
            y: Binary labels
            compute_auc: Whether to compute AUC-ROC
            
        Returns:
            ProbeResults with accuracy and CV scores
        """
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.pipeline, X, y,
            cv=self.cv_folds,
            scoring='accuracy'
        )
        
        # Fit on all data for final accuracy
        self.fit(X, y)
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        
        # AUC-ROC if requested
        auc = None
        if compute_auc and len(np.unique(y)) == 2:
            try:
                proba = self.predict_proba(X)[:, 1]
                auc = roc_auc_score(y, proba)
            except Exception:
                pass
        
        # Feature importance (coefficient magnitudes)
        classifier = self.pipeline.named_steps['classifier']
        feature_importance = np.abs(classifier.coef_).flatten()
        
        return ProbeResults(
            accuracy=accuracy,
            cv_scores=cv_scores,
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            auc_roc=auc,
            feature_importance=feature_importance
        )


def train_recoverability_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    regularization: float = 1.0,
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[ProbeResults, LinearProbe]:
    """
    Train a linear probe to measure latent recoverability.
    
    This function trains a probe to predict task-relevant labels from
    hidden activations, measuring Recov(I; Z).
    
    Args:
        activations: Hidden state activations, shape (n_samples, n_features)
                    or (n_layers, n_samples, n_features)
        labels: Target labels (e.g., correctness), shape (n_samples,)
        test_size: Fraction of data for testing
        regularization: L2 regularization strength
        cv_folds: Number of CV folds
        random_state: Random seed
        
    Returns:
        Tuple of (ProbeResults, fitted LinearProbe)
    """
    # Flatten multi-layer activations
    if activations.ndim == 3:
        n_layers, n_samples, n_features = activations.shape
        activations = activations.transpose(1, 0, 2).reshape(n_samples, -1)
    
    # Ensure labels are numpy array
    labels = np.asarray(labels).astype(int)
    
    # Create and evaluate probe
    probe = LinearProbe(
        regularization=regularization,
        cv_folds=cv_folds,
        random_state=random_state
    )
    
    results = probe.evaluate(activations, labels)
    
    return results, probe


def compare_recoverability(
    pre_collapse_activations: np.ndarray,
    post_collapse_predictions: np.ndarray,
    true_labels: np.ndarray,
    regularization: float = 1.0
) -> Dict[str, ProbeResults]:
    """
    Compare recoverability before and after collapse.
    
    This implements the key test from Section 4.3: whether pre-collapse
    activations contain more task-relevant information than post-collapse
    outputs.
    
    Args:
        pre_collapse_activations: Hidden states before collapse
        post_collapse_predictions: Model's verbalized predictions
        true_labels: Ground truth labels
        regularization: L2 regularization for probes
        
    Returns:
        Dictionary with probe results for pre and post collapse
    """
    results = {}
    
    # Pre-collapse probe: predict true labels from activations
    pre_results, _ = train_recoverability_probe(
        pre_collapse_activations,
        true_labels,
        regularization=regularization
    )
    results['pre_collapse'] = pre_results
    
    # Post-collapse: just compute accuracy of verbalized predictions
    post_accuracy = accuracy_score(true_labels, post_collapse_predictions)
    results['post_collapse'] = ProbeResults(
        accuracy=post_accuracy,
        cv_scores=np.array([post_accuracy]),
        cv_mean=post_accuracy,
        cv_std=0.0
    )
    
    # Compute recoverability gap
    results['recoverability_gap'] = pre_results.cv_mean - post_accuracy
    
    return results


class MultiTaskProbe:
    """
    Multi-task probe for predicting multiple downstream variables.
    
    Useful for understanding what different types of information
    are recoverable from the intention state.
    """
    
    def __init__(
        self,
        task_names: List[str],
        regularization: float = 1.0
    ):
        """
        Initialize multi-task probe.
        
        Args:
            task_names: Names of tasks to probe for
            regularization: L2 regularization strength
        """
        self.task_names = task_names
        self.probes = {
            name: LinearProbe(regularization=regularization)
            for name in task_names
        }
        self.results: Dict[str, ProbeResults] = {}
    
    def fit_all(
        self,
        X: np.ndarray,
        labels_dict: Dict[str, np.ndarray]
    ) -> Dict[str, ProbeResults]:
        """
        Fit probes for all tasks.
        
        Args:
            X: Activation features
            labels_dict: Dictionary mapping task names to labels
            
        Returns:
            Dictionary of results for each task
        """
        for name in self.task_names:
            if name not in labels_dict:
                raise ValueError(f"Missing labels for task: {name}")
            
            results = self.probes[name].evaluate(X, labels_dict[name])
            self.results[name] = results
        
        return self.results
    
    def summary(self) -> str:
        """Generate summary of all probe results."""
        lines = ["Multi-Task Probe Results:", "-" * 40]
        for name, results in self.results.items():
            lines.append(f"{name}: {results.cv_mean:.3f} ± {results.cv_std:.3f}")
        return "\n".join(lines)


def layer_wise_probing(
    activations: np.ndarray,
    labels: np.ndarray,
    layer_names: Optional[List[str]] = None,
    regularization: float = 1.0
) -> Dict[str, ProbeResults]:
    """
    Probe each layer separately to find where information is encoded.
    
    Args:
        activations: Shape (n_layers, n_samples, n_features)
        labels: Shape (n_samples,)
        layer_names: Optional names for layers
        regularization: L2 regularization
        
    Returns:
        Dictionary mapping layer names to probe results
    """
    if activations.ndim != 3:
        raise ValueError("Expected 3D activations (n_layers, n_samples, n_features)")
    
    n_layers = activations.shape[0]
    if layer_names is None:
        layer_names = [f"layer_{i}" for i in range(n_layers)]
    
    results = {}
    for i, name in enumerate(layer_names):
        layer_acts = activations[i]
        probe_results, _ = train_recoverability_probe(
            layer_acts, labels, regularization=regularization
        )
        results[name] = probe_results
    
    return results
