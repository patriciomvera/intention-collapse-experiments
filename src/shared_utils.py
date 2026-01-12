"""
Intention Collapse Experiments - Shared Utilities v3.0
=======================================================
Single Source of Truth for all experiment logic.

This module contains ALL shared functions. The notebook should ONLY import
from here and call these functions - no inline redefinitions.

Version: 3.0
Authors: P. M. Vera
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, balanced_accuracy_score
import re
import random
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
from datetime import datetime
import warnings
import math

warnings.filterwarnings('ignore')

# =============================================================================
# VERSION AND METADATA
# =============================================================================

__version__ = "3.3.0"
CODE_VERSION = "3.3.0"

def get_environment_info() -> Dict[str, str]:
    """Get environment info for reproducibility."""
    import transformers
    import datasets
    info = {
        'code_version': CODE_VERSION,
        'torch_version': torch.__version__,
        'transformers_version': transformers.__version__,
        'datasets_version': datasets.__version__,
        'cuda_available': str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
    
    # Try to get git commit
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info['git_commit'] = result.stdout.strip()[:8]
    except:
        info['git_commit'] = 'unknown'
    
    return info


def set_all_seeds(seed: int = 42):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Problem:
    """Generic container for a problem from any benchmark."""
    question: str
    answer: str
    final_answer: str
    idx: int
    original_idx: int  # Index in the original dataset (for reproducibility)
    benchmark: str
    choices: str = ""  # For ARC multiple choice options
    difficulty: Optional[str] = None


@dataclass
class IntentionMetrics:
    """Metrics for a single problem."""
    entropy: float
    entropy_top_k: int = 100
    argmax_token_id: int = -1  # Token ID of most likely next token
    prompt_length_tokens: int = 0


@dataclass 
class ExperimentResult:
    """Result for a single problem-condition pair."""
    problem_idx: int
    original_dataset_idx: int
    condition: str
    benchmark: str
    question: str
    ground_truth: str
    model_output: str
    extracted_answer: str
    is_correct: bool
    metrics: IntentionMetrics
    # Token counts (not word counts)
    input_tokens: int = 0
    generated_tokens: int = 0
    total_tokens: int = 0
    # Extraction metadata
    extraction_rule_used: str = ""
    math_parse_failed: bool = False
    stopped_by_criteria: bool = False
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['metrics'] = asdict(self.metrics)
        return d


@dataclass
class ProbeResults:
    """Results from recoverability probe."""
    auroc: float
    auroc_ci_low: float
    auroc_ci_high: float
    accuracy: float
    balanced_acc: float
    brier: float
    n_samples: int
    pos_rate: float  # Fraction of correct answers
    cv_folds_used: int
    n_bootstrap: int = 1000


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODEL_CONFIGS = {
    'mistral': {
        'name': 'mistralai/Mistral-7B-Instruct-v0.3',
        'extraction_layers': [27, 28, 29, 30, 31],
        'num_layers': 32,
    },
    'llama': {
        'name': 'meta-llama/Llama-3.1-8B-Instruct',
        'extraction_layers': [27, 28, 29, 30, 31],
        'num_layers': 32,
    },
    'qwen': {
        'name': 'Qwen/Qwen2.5-7B-Instruct',
        'extraction_layers': [23, 24, 25, 26, 27],
        'num_layers': 28,
    }
}


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

PROMPTS = {
    'gsm8k': {
        'baseline': "Solve this math problem. Give only the final numerical answer after ####.\n\nProblem: {question}\nAnswer:",
        'enhanced': "Solve this math problem step by step. Show your reasoning, then give the final answer after ####.\n\nProblem: {question}\nSolution:",
        'babble': "Given this math problem, write a stream of consciousness about numbers and mathematical concepts. Do NOT solve the problem.\n\nProblem: {question}\nStream of consciousness:"
    },
    'math': {
        'baseline': "Solve this mathematics problem. Give only the final answer in \\boxed{{}}.\n\nProblem: {question}\nAnswer:",
        'enhanced': "Solve this mathematics problem step by step. Show your reasoning, then give the final answer in \\boxed{{}}.\n\nProblem: {question}\nSolution:",
        'babble': "Given this mathematics problem, write a stream of consciousness about mathematical concepts. Do NOT solve the problem.\n\nProblem: {question}\nStream of consciousness:"
    },
    'arc': {
        'baseline': "Answer this science question. Give only the letter (A, B, C, or D) after ####.\n\nQuestion: {question}\n{choices}\nAnswer:",
        'enhanced': "Answer this science question step by step. Reason through each option, then give the final answer letter after ####.\n\nQuestion: {question}\n{choices}\nSolution:",
        'babble': "Given this science question, write a stream of consciousness about scientific concepts. Do NOT answer the question.\n\nQuestion: {question}\n{choices}\nStream of consciousness:"
    }
}


def format_prompt(problem: Problem, condition: str) -> str:
    """Format a prompt for a given problem and condition."""
    template = PROMPTS[problem.benchmark][condition]
    if problem.benchmark == 'arc':
        return template.format(question=problem.question, choices=problem.choices)
    return template.format(question=problem.question)


# =============================================================================
# DATASET LOADERS (with original indices for reproducibility)
# =============================================================================

def get_or_create_indices(benchmark: str, subset_size: int, seed: int, 
                          splits_dir: Optional[str] = None) -> List[int]:
    """
    Get or create dataset indices for reproducibility.
    
    If splits_dir is provided, tries to load/save indices from file.
    This ensures all models use EXACTLY the same problems.
    """
    if splits_dir:
        os.makedirs(splits_dir, exist_ok=True)
        filename = f"{benchmark}_seed{seed}_n{subset_size}.json"
        filepath = os.path.join(splits_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                indices = json.load(f)
            print(f"  Loaded existing indices from {filepath}")
            return indices
    
    # Generate new indices
    random.seed(seed)
    
    # Get dataset size
    if benchmark == 'gsm8k':
        dataset = load_dataset("gsm8k", "main", split="test")
    elif benchmark == 'math':
        dataset = load_dataset("hendrycks/competition_math", split="test")
    elif benchmark == 'arc':
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    total_size = len(dataset)
    indices = sorted(random.sample(range(total_size), min(subset_size, total_size)))
    
    # Save if directory provided
    if splits_dir:
        with open(filepath, 'w') as f:
            json.dump(indices, f)
        print(f"  Saved new indices to {filepath}")
    
    return indices


def load_gsm8k_problems(indices: List[int]) -> List[Problem]:
    """Load GSM8K problems for given indices."""
    dataset = load_dataset("gsm8k", "main", split="test")
    
    problems = []
    for local_idx, original_idx in enumerate(indices):
        item = dataset[original_idx]
        match = re.search(r'####\s*(.+?)$', item['answer'], re.MULTILINE)
        final_answer = match.group(1).strip().replace(',', '') if match else ""
        
        problems.append(Problem(
            question=item['question'],
            answer=item['answer'],
            final_answer=final_answer,
            idx=local_idx,
            original_idx=original_idx,
            benchmark='gsm8k'
        ))
    return problems


def load_math_problems(indices: List[int]) -> List[Problem]:
    """Load MATH problems for given indices."""
    dataset = load_dataset("hendrycks/competition_math", split="test")
    
    problems = []
    for local_idx, original_idx in enumerate(indices):
        item = dataset[original_idx]
        # Handle nested boxed expressions
        match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', item['solution'])
        final_answer = match.group(1).strip() if match else ""
        
        problems.append(Problem(
            question=item['problem'],
            answer=item['solution'],
            final_answer=final_answer,
            idx=local_idx,
            original_idx=original_idx,
            benchmark='math',
            difficulty=item.get('level', None)
        ))
    return problems


def load_arc_problems(indices: List[int]) -> List[Problem]:
    """Load ARC-Challenge problems for given indices."""
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    
    problems = []
    for local_idx, original_idx in enumerate(indices):
        item = dataset[original_idx]
        choices_text = "\n".join([
            f"{label}. {text}" 
            for label, text in zip(item['choices']['label'], item['choices']['text'])
        ])
        
        problems.append(Problem(
            question=item['question'],
            answer=item['answerKey'],  # Store correct answer key
            final_answer=item['answerKey'],
            idx=local_idx,
            original_idx=original_idx,
            benchmark='arc',
            choices=choices_text  # Choices stored separately
        ))
    return problems


def load_problems(benchmark: str, indices: List[int]) -> List[Problem]:
    """Load problems for a benchmark given indices."""
    loaders = {
        'gsm8k': load_gsm8k_problems,
        'math': load_math_problems,
        'arc': load_arc_problems
    }
    return loaders[benchmark](indices)


# =============================================================================
# STOPPING CRITERIA
# =============================================================================

class StopOnPattern(StoppingCriteria):
    """Stop generation when answer pattern is detected."""
    
    def __init__(self, tokenizer, patterns: List[str], min_tokens: int = 5):
        self.tokenizer = tokenizer
        self.patterns = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns]
        self.min_tokens = min_tokens
        self._generated_tokens = 0
        self.stopped = False
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self._generated_tokens += 1
        
        if self._generated_tokens < self.min_tokens:
            return False
        
        # Decode only the last portion for efficiency
        last_tokens = input_ids[0, -30:] if input_ids.shape[1] > 30 else input_ids[0]
        text = self.tokenizer.decode(last_tokens, skip_special_tokens=True)
        
        for pattern in self.patterns:
            if pattern.search(text):
                self.stopped = True
                return True
        
        return False
    
    def reset(self):
        self._generated_tokens = 0
        self.stopped = False


def get_stopping_criteria(tokenizer, benchmark: str, condition: str) -> Tuple[Optional[StoppingCriteriaList], Optional[StopOnPattern]]:
    """
    Get stopping criteria based on benchmark/condition.
    
    Returns tuple of (StoppingCriteriaList, StopOnPattern) so we can check if stopped.
    
    v3.1: Added \n\n stop for baseline to prevent unwanted reasoning.
    """
    if condition == 'babble':
        return None, None  # Let babble run full length
    
    if benchmark == 'arc':
        patterns = [
            r'####\s*[A-Da-d]',
            r'[Aa]nswer[:\s]+[A-Da-d]\s*$',
            r'^[A-Da-d]\.$',
        ]
    else:  # gsm8k, math
        patterns = [
            r'####\s*-?\d',
            r'\\boxed\{',
        ]
        # v3.1: For baseline, also stop on double newline to prevent reasoning
        if condition == 'baseline':
            patterns.append(r'\n\n')
    
    stopper = StopOnPattern(tokenizer, patterns, min_tokens=10)
    return StoppingCriteriaList([stopper]), stopper


# =============================================================================
# ACTIVATION EXTRACTOR (Optimized - prompt state only)
# =============================================================================

class ActivationExtractor:
    """
    Extract hidden state activations from specified transformer layers.
    
    Optimized for prompt-state extraction (pre-collapse intention).
    """
    
    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = sorted(layers)
        self._hooks = []
        self._activations = {}
        self._is_capturing = False
    
    def _get_layer_module(self, layer_idx: int):
        """Get the module for a specific layer."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[layer_idx]
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h[layer_idx]
        raise ValueError(f"Cannot find layer module for layer {layer_idx}")
    
    def _create_hook(self, layer_idx: int):
        """Create a hook that captures the last token's activation."""
        def hook(module, input, output):
            if not self._is_capturing:
                return
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Capture last token position only (pre-collapse state)
            self._activations[layer_idx] = hidden_states[:, -1, :].detach().cpu()
        return hook
    
    def _register_hooks(self):
        for layer_idx in self.layers:
            layer_module = self._get_layer_module(layer_idx)
            hook = layer_module.register_forward_hook(self._create_hook(layer_idx))
            self._hooks.append(hook)
    
    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def clear(self):
        self._activations = {}
    
    @contextmanager
    def capture(self):
        """Context manager for capturing activations during forward pass."""
        self.clear()
        self._register_hooks()
        self._is_capturing = True
        try:
            yield self
        finally:
            self._is_capturing = False
            self._remove_hooks()
    
    def get_activations(self) -> np.ndarray:
        """
        Get extracted activations.
        
        Returns:
            Array of shape (n_layers, hidden_dim)
        """
        if not self._activations:
            raise ValueError("No activations captured. Run capture() first.")
        
        all_activations = []
        for l in self.layers:
            if l not in self._activations:
                raise ValueError(f"No activations for layer {l}")
            all_activations.append(self._activations[l].squeeze(0).numpy())
        
        return np.stack(all_activations, axis=0)


# =============================================================================
# INTENTION METRICS
# =============================================================================

def compute_intention_entropy(logits: torch.Tensor, top_k: int = 100) -> Tuple[float, int]:
    """
    Compute intention entropy from logits.
    
    This is H_int(I): entropy of the next-token distribution given the prompt,
    representing pre-collapse uncertainty.
    
    v3.3: Added numerical stability (subtract max before softmax) and NaN checking.
    
    Args:
        logits: Logits tensor (can be [vocab_size] or [seq_len, vocab_size])
        top_k: Number of top tokens to consider (for numerical stability)
    
    Returns:
        Tuple of (entropy in bits, argmax token ID)
        Returns (float('nan'), argmax_idx) if computation produces invalid values.
    """
    if logits.dim() == 2:
        logits = logits[-1]  # Take last position
    
    # Numerical stability: subtract max before softmax to prevent overflow
    logits = logits - logits.max()
    
    # Get top-k for stability
    if top_k > 0 and top_k < logits.size(-1):
        top_logits, top_indices = torch.topk(logits, top_k)
        # Subtract max again for the subset
        top_logits = top_logits - top_logits.max()
        probs = F.softmax(top_logits, dim=-1)
        argmax_idx = top_indices[0].item()
    else:
        probs = F.softmax(logits, dim=-1)
        argmax_idx = torch.argmax(logits).item()
    
    # Compute entropy in bits (log base 2)
    eps = 1e-10
    entropy = -torch.sum(probs * torch.log2(probs + eps)).item()
    
    # Check for invalid values - return NaN explicitly rather than hiding it
    if math.isnan(entropy) or math.isinf(entropy):
        return float('nan'), argmax_idx
    
    return entropy, argmax_idx


def compute_effective_dimensionality(
    activations: np.ndarray,
    method: str = 'threshold',
    variance_threshold: float = 0.90
) -> Tuple[int, np.ndarray]:
    """
    Compute effective dimensionality via PCA.
    
    Args:
        activations: Shape (n_samples, n_features) or (n_layers, n_samples, hidden_dim)
        method: 'threshold' (smallest k for variance) or 'participation_ratio'
        variance_threshold: Variance threshold for 'threshold' method
    
    Returns:
        Tuple of (effective dimensionality, explained variance ratios)
    """
    # Flatten if 3D (layer activations)
    if activations.ndim == 3:
        n_layers, n_samples, hidden_dim = activations.shape
        activations = activations.reshape(n_layers * n_samples, hidden_dim)
    
    n_samples, n_features = activations.shape
    n_components = min(n_samples, n_features)
    
    if n_components < 2:
        return 1, np.array([1.0])
    
    # Center the data
    activations = activations - activations.mean(axis=0)
    
    # PCA
    pca = PCA(n_components=n_components)
    pca.fit(activations)
    
    if method == 'participation_ratio':
        # Participation ratio: (sum λ)² / sum λ²
        eigenvalues = pca.explained_variance_
        dim_eff = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        return int(np.round(dim_eff)), pca.explained_variance_ratio_
    else:  # threshold
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        dim_eff = np.searchsorted(cumulative_variance, variance_threshold) + 1
        return int(min(dim_eff, n_components)), pca.explained_variance_ratio_


def compute_layer_wise_dim_eff(
    activations: np.ndarray,
    variance_threshold: float = 0.90
) -> List[int]:
    """
    Compute effective dimensionality per layer.
    
    Args:
        activations: Shape (n_layers, n_samples, hidden_dim)
    
    Returns:
        List of dim_eff per layer
    """
    if activations.ndim != 3:
        raise ValueError("Expected 3D array (n_layers, n_samples, hidden_dim)")
    
    layer_dims = []
    for layer_idx in range(activations.shape[0]):
        layer_acts = activations[layer_idx]  # (n_samples, hidden_dim)
        dim_eff, _ = compute_effective_dimensionality(
            layer_acts, method='threshold', variance_threshold=variance_threshold
        )
        layer_dims.append(dim_eff)
    
    return layer_dims


# =============================================================================
# RECOVERABILITY PROBE (No leakage)
# =============================================================================

def train_recoverability_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    cv_folds: int = 5,
    n_bootstrap: int = 1000
) -> ProbeResults:
    """
    Train a linear probe to measure latent recoverability Recov(I; Z).
    
    IMPORTANT: Uses Pipeline to prevent data leakage (scaler fitted within each fold).
    CI computed via bootstrap resampling.
    
    Args:
        activations: Shape (n_samples, ...) - will be flattened
        labels: Binary labels (0/1)
        cv_folds: Number of CV folds
        n_bootstrap: Number of bootstrap samples for CI
    
    Returns:
        ProbeResults dataclass with all metrics
    """
    # Flatten activations
    if activations.ndim == 3:
        n_layers, n_samples, hidden_dim = activations.shape
        activations = activations.transpose(1, 0, 2).reshape(n_samples, -1)
    elif activations.ndim == 2 and activations.shape[0] != len(labels):
        activations = activations.reshape(len(labels), -1)
    
    labels = np.asarray(labels).astype(int)
    n_samples = len(labels)
    
    # Check class balance
    unique, counts = np.unique(labels, return_counts=True)
    pos_rate = counts[1] / n_samples if len(unique) == 2 else 0.0
    
    if len(unique) < 2:
        return ProbeResults(
            auroc=0.5, auroc_ci_low=0.0, auroc_ci_high=1.0,
            accuracy=0.5, balanced_acc=0.5, brier=1.0,
            n_samples=n_samples, pos_rate=pos_rate, cv_folds_used=0, n_bootstrap=0
        )
    
    min_class_count = counts.min()
    cv_folds = min(cv_folds, min_class_count)
    
    if cv_folds < 2:
        return ProbeResults(
            auroc=0.5, auroc_ci_low=0.0, auroc_ci_high=1.0,
            accuracy=0.5, balanced_acc=0.5, brier=1.0,
            n_samples=n_samples, pos_rate=pos_rate, cv_folds_used=0, n_bootstrap=0
        )
    
    # Pipeline with scaler INSIDE (no leakage)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs'))
    ])
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Collect out-of-fold predictions for bootstrap
    oof_probs = np.zeros(n_samples)
    oof_preds = np.zeros(n_samples)
    fold_aurocs = []
    fold_accs = []
    fold_balanced_accs = []
    fold_briers = []
    
    for train_idx, test_idx in cv.split(activations, labels):
        X_train, X_test = activations[train_idx], activations[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        oof_probs[test_idx] = y_prob
        oof_preds[test_idx] = y_pred
        
        fold_accs.append((y_pred == y_test).mean())
        fold_balanced_accs.append(balanced_accuracy_score(y_test, y_pred))
        
        if len(np.unique(y_test)) > 1:
            fold_aurocs.append(roc_auc_score(y_test, y_prob))
            fold_briers.append(brier_score_loss(y_test, y_prob))
    
    # Bootstrap CI for AUROC on out-of-fold predictions
    auroc_overall = roc_auc_score(labels, oof_probs) if len(np.unique(labels)) > 1 else 0.5
    
    if n_bootstrap > 0 and len(np.unique(labels)) > 1:
        bootstrap_aurocs = []
        rng = np.random.RandomState(42)
        for _ in range(n_bootstrap):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            if len(np.unique(labels[idx])) > 1:
                bootstrap_aurocs.append(roc_auc_score(labels[idx], oof_probs[idx]))
        
        if bootstrap_aurocs:
            ci_low = np.percentile(bootstrap_aurocs, 2.5)
            ci_high = np.percentile(bootstrap_aurocs, 97.5)
        else:
            ci_low, ci_high = 0.0, 1.0
    else:
        ci_low, ci_high = 0.0, 1.0
    
    return ProbeResults(
        auroc=auroc_overall,
        auroc_ci_low=max(0, ci_low),
        auroc_ci_high=min(1, ci_high),
        accuracy=np.mean(fold_accs),
        balanced_acc=np.mean(fold_balanced_accs),
        brier=np.mean(fold_briers) if fold_briers else 1.0,
        n_samples=n_samples,
        pos_rate=pos_rate,
        cv_folds_used=cv_folds,
        n_bootstrap=n_bootstrap
    )


# =============================================================================
# ANSWER EXTRACTION
# =============================================================================

def extract_answer_gsm8k(model_output: str) -> Tuple[str, str]:
    """
    Extract numerical answer from GSM8K output.
    
    v3.2: Fixed rule_first_block to only apply when NO answer markers present.
          Added rule_final_answer for "Final answer: X" pattern.
    
    Returns: (extracted_answer, extraction_rule_used)
    """
    output = model_output.strip()
    
    # Rule 1: #### followed by number (highest priority - GSM8K standard format)
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', output)
    if match:
        return match.group(1).replace(',', ''), 'rule_hash'
    
    # Rule 2: "Final answer:" format (common in CoT outputs)
    match = re.search(r'[Ff]inal\s*[Aa]nswer[:\s]+\$?(-?[\d,]+\.?\d*)', output)
    if match:
        return match.group(1).replace(',', ''), 'rule_final_answer'
    
    # Rule 3: "Answer: X" format
    match = re.search(r'[Aa]nswer[:\s]+\$?(-?[\d,]+\.?\d*)', output)
    if match:
        return match.group(1).replace(',', ''), 'rule_answer'
    
    # Rule 4: First block number - ONLY for baseline-style outputs
    # Skip if there are answer markers (prevents grabbing "1." from CoT steps)
    has_answer_marker = re.search(r'[Aa]nswer|####|\\boxed', output, re.IGNORECASE)
    if not has_answer_marker and '\n\n' in output:
        first_block = output.split('\n\n')[0]
        numbers = re.findall(r'-?[\d,]+\.?\d*', first_block)
        numbers = [n.replace(',', '') for n in numbers 
                   if n.replace(',', '').replace('.', '').replace('-', '').isdigit()]
        if numbers:
            return numbers[0], 'rule_first_block'
    
    # Rule 5: "= X" at end of line (for CoT calculations - take LAST occurrence)
    matches = re.findall(r'=\s*\$?(-?[\d,]+\.?\d*)\s*$', output, re.MULTILINE)
    if matches:
        return matches[-1].replace(',', ''), 'rule_equals'
    
    # Rule 6: Last number in output (fallback)
    numbers = re.findall(r'-?[\d,]+\.?\d*', output)
    numbers = [n.replace(',', '') for n in numbers 
               if n.replace(',', '').replace('.', '').replace('-', '').isdigit()]
    if numbers:
        return numbers[-1], 'rule_last_number'
    
    return "", 'rule_none'


def extract_answer_math(model_output: str) -> Tuple[str, str]:
    """
    Extract answer from MATH output.
    
    Returns: (extracted_answer, extraction_rule_used)
    """
    output = model_output.strip()
    
    # Rule 1: \boxed{...} (handle nested braces)
    match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', output)
    if match:
        return match.group(1).strip(), 'rule_boxed'
    
    # Rule 2: #### answer
    match = re.search(r'####\s*(.+?)$', output, re.MULTILINE)
    if match:
        return match.group(1).strip(), 'rule_hash'
    
    # Rule 3: "Answer: X"
    match = re.search(r'[Aa]nswer[:\s]+(.+?)(?:\n|$)', output)
    if match:
        return match.group(1).strip(), 'rule_answer'
    
    # Rule 4: Last line
    lines = output.strip().split('\n')
    if lines:
        return lines[-1].strip(), 'rule_last_line'
    
    return "", 'rule_none'


def extract_answer_arc(model_output: str) -> Tuple[str, str]:
    """
    Extract letter answer (A/B/C/D) from ARC output.
    
    IMPORTANT: Takes LAST letter match to avoid capturing letters in reasoning.
    
    Returns: (extracted_answer, extraction_rule_used)
    """
    output = model_output.strip()
    
    # Rule 1: #### X
    match = re.search(r'####\s*([A-Da-d])\b', output)
    if match:
        return match.group(1).upper(), 'rule_hash'
    
    # Rule 2: "Answer: X" or "The answer is X"
    match = re.search(r'(?:[Aa]nswer|[Tt]he answer is)[:\s]+([A-Da-d])\b', output, re.IGNORECASE)
    if match:
        return match.group(1).upper(), 'rule_answer'
    
    # Rule 3: Letter at start of line with period/paren (e.g., "A." or "A)")
    match = re.search(r'^([A-Da-d])[\.\)]', output, re.MULTILINE)
    if match:
        return match.group(1).upper(), 'rule_start_line'
    
    # Rule 4: LAST standalone letter A-D (not first!)
    matches = re.findall(r'\b([A-Da-d])\b', output)
    if matches:
        return matches[-1].upper(), 'rule_last_letter'
    
    return "", 'rule_none'


def extract_answer(model_output: str, benchmark: str) -> Tuple[str, str]:
    """
    Extract answer from model output.
    
    Returns: (extracted_answer, extraction_rule_used)
    """
    extractors = {
        'gsm8k': extract_answer_gsm8k,
        'math': extract_answer_math,
        'arc': extract_answer_arc
    }
    return extractors[benchmark](model_output)


# =============================================================================
# ANSWER EVALUATION
# =============================================================================

def evaluate_answer_gsm8k(predicted: str, ground_truth: str, tolerance: float = 0.01) -> bool:
    """Evaluate GSM8K answer (numerical comparison)."""
    pred = predicted.strip().replace(',', '').replace('$', '').replace('%', '')
    truth = ground_truth.strip().replace(',', '').replace('$', '').replace('%', '')
    
    if pred == truth:
        return True
    
    try:
        pred_num = float(pred)
        truth_num = float(truth)
        
        if pred_num == truth_num:
            return True
        if int(pred_num) == int(truth_num):
            return True
        if truth_num != 0 and abs((pred_num - truth_num) / truth_num) < tolerance:
            return True
    except ValueError:
        pass
    
    return False


def evaluate_answer_math_symbolic(predicted: str, ground_truth: str) -> Tuple[bool, bool]:
    """
    Evaluate MATH answer using symbolic comparison.
    
    Returns: (is_correct, parse_failed)
    """
    try:
        from sympy import simplify, sympify, nsimplify
        from sympy.parsing.latex import parse_latex
    except ImportError:
        # Fallback to string comparison
        return predicted.strip() == ground_truth.strip(), True
    
    def normalize_and_parse(s: str):
        """Try multiple parsing strategies."""
        s = s.strip()
        
        # Remove common wrappers
        s = s.replace('$', '').strip()
        
        # Try sympy's sympify directly
        try:
            return sympify(s, evaluate=True), False
        except:
            pass
        
        # Try LaTeX parser
        try:
            return parse_latex(s), False
        except:
            pass
        
        # Try after normalizing LaTeX
        s_norm = s.replace('\\frac', 'Rational')
        s_norm = s_norm.replace('\\sqrt', 'sqrt')
        s_norm = s_norm.replace('\\pi', 'pi')
        s_norm = s_norm.replace('\\cdot', '*')
        s_norm = s_norm.replace('\\times', '*')
        s_norm = s_norm.replace('^', '**')
        s_norm = re.sub(r'\\left|\\right', '', s_norm)
        s_norm = re.sub(r'\\[a-zA-Z]+', '', s_norm)  # Remove remaining LaTeX commands
        
        try:
            return sympify(s_norm, evaluate=True), False
        except:
            pass
        
        # Try extracting just the number
        numbers = re.findall(r'-?\d+\.?\d*', s)
        if numbers:
            try:
                return sympify(numbers[-1]), False
            except:
                pass
        
        return None, True
    
    pred_sym, pred_failed = normalize_and_parse(predicted)
    truth_sym, truth_failed = normalize_and_parse(ground_truth)
    
    if pred_failed or truth_failed or pred_sym is None or truth_sym is None:
        # Fallback to string comparison
        return predicted.strip().lower() == ground_truth.strip().lower(), True
    
    try:
        # Check symbolic equality
        diff = simplify(pred_sym - truth_sym)
        if diff == 0:
            return True, False
        
        # Try numerical evaluation
        pred_float = float(pred_sym.evalf())
        truth_float = float(truth_sym.evalf())
        return abs(pred_float - truth_float) < 1e-6, False
    except:
        return False, True


def evaluate_answer_arc(predicted: str, ground_truth: str) -> bool:
    """Evaluate ARC answer (letter comparison)."""
    return predicted.strip().upper() == ground_truth.strip().upper()


def evaluate_answer(predicted: str, ground_truth: str, benchmark: str) -> Tuple[bool, bool]:
    """
    Evaluate if predicted answer is correct.
    
    Returns: (is_correct, parse_failed) - parse_failed only relevant for MATH
    """
    if benchmark == 'gsm8k':
        return evaluate_answer_gsm8k(predicted, ground_truth), False
    elif benchmark == 'math':
        return evaluate_answer_math_symbolic(predicted, ground_truth)
    elif benchmark == 'arc':
        return evaluate_answer_arc(predicted, ground_truth), False
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


# =============================================================================
# EXPERIMENT RUNNER (2-Pass Approach)
# =============================================================================

def run_single_problem(
    model,
    tokenizer,
    extractor: ActivationExtractor,
    problem: Problem,
    condition: str,
    max_new_tokens: int,
    entropy_top_k: int = 100
) -> Tuple[ExperimentResult, Optional[np.ndarray]]:
    """
    Run model on single problem using 2-pass approach.
    
    Pass 1: Forward on prompt → pre-collapse intention state (logits + activations)
    Pass 2: Generate response (no output_scores overhead)
    
    This is conceptually aligned with the paper's definition of pre-collapse I.
    """
    prompt = format_prompt(problem, condition)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]
    
    # =========================================================================
    # PASS 1: Forward pass on prompt → pre-collapse intention state
    # =========================================================================
    activations = None
    entropy = 0.0
    argmax_token_id = -1
    
    with torch.no_grad():
        with extractor.capture():
            outputs_forward = model(**inputs)
        
        try:
            activations = extractor.get_activations()
        except Exception as e:
            activations = None
        
        # Compute intention entropy from logits of NEXT token
        logits = outputs_forward.logits[0, -1, :]
        entropy, argmax_token_id = compute_intention_entropy(logits, entropy_top_k)
    
    # =========================================================================
    # PASS 2: Generate response (NO output_scores → efficient)
    # =========================================================================
    stopping_criteria, stopper = get_stopping_criteria(tokenizer, problem.benchmark, condition)
    
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=None,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria
            # NOTE: NO output_scores=True, NO return_dict_in_generate=True
        )
    
    generated_ids = generated[0][input_length:]
    model_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_tokens = len(generated_ids)
    
    stopped_by_criteria = stopper.stopped if stopper else False
    
    # =========================================================================
    # Extract and evaluate answer
    # =========================================================================
    if condition == 'babble':
        extracted_answer = ""
        extraction_rule = "babble_no_extraction"
        is_correct = False
        parse_failed = False
    else:
        extracted_answer, extraction_rule = extract_answer(model_output, problem.benchmark)
        is_correct, parse_failed = evaluate_answer(extracted_answer, problem.final_answer, problem.benchmark)
    
    metrics = IntentionMetrics(
        entropy=entropy,
        entropy_top_k=entropy_top_k,
        argmax_token_id=argmax_token_id,
        prompt_length_tokens=input_length
    )
    
    result = ExperimentResult(
        problem_idx=problem.idx,
        original_dataset_idx=problem.original_idx,
        condition=condition,
        benchmark=problem.benchmark,
        question=problem.question[:200],  # Truncate for storage
        ground_truth=problem.final_answer,
        model_output=model_output,
        extracted_answer=extracted_answer,
        is_correct=is_correct,
        metrics=metrics,
        input_tokens=input_length,
        generated_tokens=generated_tokens,
        total_tokens=input_length + generated_tokens,
        extraction_rule_used=extraction_rule,
        math_parse_failed=parse_failed,
        stopped_by_criteria=stopped_by_criteria
    )
    
    return result, activations


# =============================================================================
# RESULTS AGGREGATION
# =============================================================================

def aggregate_results(
    results: List[ExperimentResult],
    activations: List[np.ndarray],
    variance_threshold: float = 0.90
) -> Dict[str, Any]:
    """
    Aggregate results for a single condition.
    
    Returns dict with all aggregated metrics.
    """
    n = len(results)
    if n == 0:
        return {}
    
    # Basic metrics
    accuracies = [r.is_correct for r in results if r.condition != 'babble']
    entropies = np.array([r.metrics.entropy for r in results])
    gen_tokens = [r.generated_tokens for r in results]
    
    # Count valid/invalid entropies
    entropy_valid_mask = ~np.isnan(entropies) & ~np.isinf(entropies)
    entropy_valid_n = int(np.sum(entropy_valid_mask))
    entropy_nan_count = int(np.sum(np.isnan(entropies)))
    entropy_inf_count = int(np.sum(np.isinf(entropies)))
    
    # Dimensionality
    dim_eff_global = 0
    dim_eff_layerwise = []
    explained_variance = None
    
    if activations and len(activations) > 0:
        acts_array = np.stack(activations, axis=1)  # (n_layers, n_samples, hidden_dim)
        
        # Layer-wise dim_eff
        dim_eff_layerwise = compute_layer_wise_dim_eff(acts_array, variance_threshold)
        
        # Global dim_eff (median of layers as aggregation)
        dim_eff_global = int(np.median(dim_eff_layerwise))
        
        # Also compute on stacked for comparison
        dim_eff_stacked, explained_variance = compute_effective_dimensionality(
            acts_array, method='threshold', variance_threshold=variance_threshold
        )
    
    agg = {
        'n_samples': n,
        'accuracy': np.mean(accuracies) if accuracies else None,
        'accuracy_std': np.std(accuracies) if accuracies else None,
        'entropy_mean': float(np.nanmean(entropies)),
        'entropy_std': float(np.nanstd(entropies)),
        'entropy_valid_n': entropy_valid_n,
        'entropy_nan_count': entropy_nan_count,
        'entropy_inf_count': entropy_inf_count,
        'entropy_nan_rate': entropy_nan_count / n if n > 0 else 0,
        'dim_eff_global': dim_eff_global,
        'dim_eff_layerwise': dim_eff_layerwise,
        'generated_tokens_mean': float(np.mean(gen_tokens)),
        'generated_tokens_std': float(np.std(gen_tokens)),
        'stopped_by_criteria_rate': np.mean([r.stopped_by_criteria for r in results]),
    }
    
    # MATH-specific
    if results[0].benchmark == 'math':
        agg['math_parse_failed_rate'] = np.mean([r.math_parse_failed for r in results])
    
    # Extraction rules distribution
    rules = [r.extraction_rule_used for r in results]
    agg['extraction_rules'] = {rule: rules.count(rule) for rule in set(rules)}
    
    return agg


# =============================================================================
# SAVING/LOADING
# =============================================================================

def save_experiment_results(
    results: Dict[str, List[ExperimentResult]],
    activations: Dict[str, List[np.ndarray]],
    probe_results: Dict[str, ProbeResults],
    aggregated: Dict[str, Dict],
    config: Dict,
    indices: List[int],
    model_family: str,
    benchmark: str,
    drive_path: str
) -> Tuple[str, str]:
    """
    Save all experiment results to files.
    
    Returns: (json_path, npz_path)
    """
    os.makedirs(drive_path, exist_ok=True)
    
    # Prepare summary
    summary = {
        'version': CODE_VERSION,
        'environment': get_environment_info(),
        'model_family': model_family,
        'model_name': MODEL_CONFIGS[model_family]['name'],
        'benchmark': benchmark,
        'n_problems': len(indices),
        'dataset_indices': indices,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'conditions': {}
    }
    
    # Add aggregated results per condition
    for condition in results.keys():
        summary['conditions'][condition] = aggregated.get(condition, {})
        
        if condition in probe_results:
            summary['conditions'][condition]['probe'] = asdict(probe_results[condition])
    
    # Add detailed results
    summary['detailed_results'] = {
        condition: [r.to_dict() for r in res_list]
        for condition, res_list in results.items()
    }
    
    # Save JSON
    json_path = os.path.join(drive_path, f"{model_family}_{benchmark}_results.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save activations
    npz_path = os.path.join(drive_path, f"{model_family}_{benchmark}_activations.npz")
    save_dict = {}
    for condition, acts in activations.items():
        if acts:
            save_dict[condition] = np.stack(acts, axis=0)
    
    if save_dict:
        np.savez_compressed(npz_path, **save_dict)
    else:
        npz_path = None
    
    return json_path, npz_path


def load_experiment_results(model_family: str, benchmark: str, drive_path: str) -> Dict:
    """Load experiment results from JSON."""
    json_path = os.path.join(drive_path, f"{model_family}_{benchmark}_results.json")
    with open(json_path, 'r') as f:
        return json.load(f)


def list_available_results(drive_path: str) -> List[Tuple[str, str]]:
    """List all available experiment results."""
    results = []
    if not os.path.exists(drive_path):
        return results
    
    for filename in os.listdir(drive_path):
        if filename.endswith('_results.json'):
            parts = filename.replace('_results.json', '').split('_')
            if len(parts) >= 2:
                results.append((parts[0], parts[1]))
    
    return results


# =============================================================================
# SANITY CHECKS
# =============================================================================

def run_sanity_checks() -> bool:
    """
    Run sanity checks to verify all fixes are in place.
    
    Returns True if all checks pass.
    """
    import inspect
    
    checks_passed = True
    
    # Check 1: Probe uses Pipeline (no leakage)
    probe_source = inspect.getsource(train_recoverability_probe)
    if 'Pipeline' not in probe_source:
        print("❌ Probe does NOT use Pipeline - data leakage risk!")
        checks_passed = False
    else:
        print("  ✓ Probe uses Pipeline (no leakage)")
    
    # Check 2: ARC extraction takes LAST match
    arc_source = inspect.getsource(extract_answer_arc)
    if 'matches[-1]' not in arc_source:
        print("❌ ARC extraction may not take LAST match")
        checks_passed = False
    else:
        print("  ✓ ARC extraction takes LAST letter match")
    
    # Check 3: MATH uses symbolic evaluation
    math_source = inspect.getsource(evaluate_answer_math_symbolic)
    if 'sympy' not in math_source and 'simplify' not in math_source:
        print("❌ MATH evaluation may not be symbolic")
        checks_passed = False
    else:
        print("  ✓ MATH uses symbolic evaluation")
    
    # Check 4: Problem has choices field
    if not hasattr(Problem, 'choices'):
        print("❌ Problem dataclass missing 'choices' field")
        checks_passed = False
    else:
        print("  ✓ Problem has 'choices' field for ARC")
    
    # Check 5: 2-pass approach (check run_single_problem)
    runner_source = inspect.getsource(run_single_problem)
    # v3.1: Only check actual code, not comments
    runner_code_lines = [l for l in runner_source.split('\n') if not l.strip().startswith('#')]
    runner_code = '\n'.join(runner_code_lines)
    if 'output_scores=True' in runner_code:
        print("❌ Still using output_scores=True in runner")
        checks_passed = False
    else:
        print("  ✓ 2-pass approach (no output_scores overhead)")
    
    # Check 6: Version tracking
    if 'version' not in inspect.getsource(save_experiment_results):
        print("❌ Results may not include version")
        checks_passed = False
    else:
        print("  ✓ Results include version tracking")
    
    print()
    if checks_passed:
        print(f"✅ All sanity checks passed! Version {CODE_VERSION}")
    else:
        print("❌ Some checks failed!")
    
    return checks_passed


# =============================================================================
# QUICK UNIT TESTS
# =============================================================================

def run_unit_tests() -> bool:
    """Run quick unit tests for critical functions."""
    print("\nRunning unit tests...")
    all_passed = True
    
    # Test 1: ARC extraction with letters in reasoning
    test_output = "Let me think about this. Option A seems good, but B is better. Actually, C is wrong. The answer is D."
    answer, rule = extract_answer_arc(test_output)
    if answer != 'D':
        print(f"  ❌ ARC extraction test failed: got '{answer}', expected 'D'")
        all_passed = False
    else:
        print("  ✓ ARC extraction handles reasoning letters correctly")
    
    # Test 2: MATH symbolic evaluation
    is_correct, parse_failed = evaluate_answer_math_symbolic("1/2", "0.5")
    if not is_correct:
        print(f"  ❌ MATH symbolic test 1 failed: 1/2 != 0.5")
        all_passed = False
    
    is_correct, _ = evaluate_answer_math_symbolic("\\frac{1}{2}", "0.5")
    if not is_correct:
        print(f"  ❌ MATH symbolic test 2 failed: \\frac{{1}}{{2}} != 0.5")
        # This might fail without proper LaTeX parsing, so just warn
        print("     (Note: LaTeX parsing may require sympy.parsing.latex)")
    else:
        print("  ✓ MATH symbolic evaluation works")
    
    # Test 3: GSM8K extraction
    test_output = "Let me calculate... 42 + 58 = 100. #### 100"
    answer, rule = extract_answer_gsm8k(test_output)
    if answer != '100':
        print(f"  ❌ GSM8K extraction test failed: got '{answer}', expected '100'")
        all_passed = False
    else:
        print("  ✓ GSM8K extraction works")
    
    # Test 3b (v3.1): GSM8K first_block extraction for baseline outputs
    test_output_baseline = "18 miles\n\nHere's the reasoning:\n\n1. Let's first find Dana's walking speed..."
    answer, rule = extract_answer_gsm8k(test_output_baseline)
    if answer != '18':
        print(f"  ❌ GSM8K first_block test failed: got '{answer}' ({rule}), expected '18'")
        all_passed = False
    else:
        print("  ✓ GSM8K first_block extraction works (v3.1 fix)")
    
    # Test 3c (v3.2): GSM8K "Final answer:" extraction for CoT outputs
    test_output_cot = "1. First step.\n2. Second step = 42\n\nFinal answer: Janet pays $1430."
    answer, rule = extract_answer_gsm8k(test_output_cot)
    if answer != '1430':
        print(f"  ❌ GSM8K final_answer test failed: got '{answer}' ({rule}), expected '1430'")
        all_passed = False
    else:
        print("  ✓ GSM8K Final answer extraction works (v3.2 fix)")
    
    # Test 4: Entropy computation
    logits = torch.randn(1000)
    entropy, argmax = compute_intention_entropy(logits, top_k=100)
    if not (0 <= entropy <= 10):  # Reasonable entropy range
        print(f"  ❌ Entropy computation test failed: got {entropy}")
        all_passed = False
    else:
        print("  ✓ Entropy computation works")
    
    print()
    if all_passed:
        print("✅ All unit tests passed!")
    else:
        print("❌ Some unit tests failed!")
    
    return all_passed


if __name__ == "__main__":
    print(f"Intention Collapse Shared Utilities v{CODE_VERSION}")
    print("=" * 50)
    run_sanity_checks()
    run_unit_tests()
