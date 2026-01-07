"""
Intention Collapse Experiments - Shared Utilities v3.0.1
Single source of truth for all experimental functions.
"""

import torch
import numpy as np
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.stats import entropy as scipy_entropy
from datasets import load_dataset
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATASET UTILITIES
# ============================================================================

def load_gsm8k_split(split: str = "test", sample_size: Optional[int] = None, seed: int = 42):
    """
    Load GSM8K dataset with optional sampling.
    
    Args:
        split: Dataset split ('test', 'train')
        sample_size: Number of samples to select (None = all)
        seed: Random seed for sampling
        
    Returns:
        dataset: HuggingFace dataset object
    """
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    
    if sample_size is not None and sample_size < len(dataset):
        # Ensure reproducible sampling
        indices = list(range(len(dataset)))
        np.random.seed(seed)
        selected_indices = sorted(np.random.choice(indices, sample_size, replace=False))
        dataset = dataset.select(selected_indices)
    
    return dataset


def extract_ground_truth(answer_string: str) -> str:
    """
    Extract numerical answer from GSM8K answer format.
    
    Args:
        answer_string: Full answer string (e.g., "explanation #### 42")
        
    Returns:
        Numerical answer as string
    """
    parts = answer_string.split('####')
    if len(parts) > 1:
        return parts[-1].strip()
    return answer_string.strip()


def extract_predicted_answer(generated_text: str, regime: str = "cot") -> Optional[str]:
    """
    Extract numerical answer from generated text.
    
    Args:
        generated_text: Model-generated text
        regime: Generation regime ('baseline', 'cot', 'babble')
        
    Returns:
        Extracted answer or None
    """
    # For CoT, look for #### pattern
    if "####" in generated_text:
        parts = generated_text.split("####")
        answer_part = parts[-1]
    else:
        answer_part = generated_text
    
    # Extract last number
    numbers = re.findall(r'-?\d+\.?\d*', answer_part)
    if numbers:
        return numbers[-1].strip()
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Remove whitespace, commas, and trailing zeros
    answer = answer.replace(",", "").replace(" ", "").strip()
    try:
        # Try to convert to float to handle 42.0 vs 42
        return str(float(answer))
    except:
        return answer


def check_correctness(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    return pred_norm == gt_norm


# ============================================================================
# ACTIVATION EXTRACTION
# ============================================================================

def extract_activations_with_hooks(
    model,
    tokenizer,
    prompt: str,
    layer_indices: List[int],
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    return_first_step_only: bool = True
) -> Tuple[Dict[int, torch.Tensor], str, torch.Tensor]:
    """
    Extract hidden activations from specified layers during generation.
    
    Args:
        model: Loaded transformer model
        tokenizer: Corresponding tokenizer
        prompt: Input prompt
        layer_indices: List of layer indices to extract from (e.g., [27, 28, 29, 30, 31])
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        return_first_step_only: If True, only return activations from first decoding step
        
    Returns:
        activations: Dict mapping layer_idx -> tensor of activations
        generated_text: Full generated output
        logits: Logits from first token (for entropy calculation)
    """
    activations = {layer_idx: [] for layer_idx in layer_indices}
    first_logits = None
    
    def create_hook(layer_idx):
        def hook_fn(module, input, output):
            # output[0] is the hidden states tensor (batch, seq_len, hidden_dim)
            hidden_states = output[0].detach().cpu()
            activations[layer_idx].append(hidden_states)
        return hook_fn
    
    # Register hooks
    hooks = []
    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]
        hook = layer.register_forward_hook(create_hook(layer_idx))
        hooks.append(hook)
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        # Get first token logits for entropy calculation
        outputs = model(**inputs)
        first_logits = outputs.logits[0, -1, :].detach().cpu()  # Last position of prompt
        
        # Full generation
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=False  # We get them via hooks
        )
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Decode output
    generated_text = tokenizer.decode(generated.sequences[0], skip_special_tokens=True)
    
    # Process activations: take only first decoding step if requested
    if return_first_step_only:
        processed_activations = {}
        for layer_idx in layer_indices:
            if len(activations[layer_idx]) > 0:
                # First element, last token position
                first_step = activations[layer_idx][0][:, -1, :]  # (batch=1, hidden_dim)
                processed_activations[layer_idx] = first_step.squeeze(0)  # (hidden_dim,)
        activations = processed_activations
    
    return activations, generated_text, first_logits


# ============================================================================
# INTENTION METRICS
# ============================================================================

def compute_intention_entropy(logits: torch.Tensor) -> float:
    """
    Compute entropy H_int(I) from next-token logits.
    
    Args:
        logits: Logit tensor for next token (vocab_size,)
        
    Returns:
        Entropy in bits
    """
    probs = torch.softmax(logits, dim=-1).numpy()
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    H = scipy_entropy(probs, base=2)  # bits
    return float(H)


def compute_effective_dimensionality(
    activations: Dict[int, torch.Tensor],
    variance_threshold: float = 0.9,
    per_layer: bool = False
) -> Dict[str, float]:
    """
    Compute effective dimensionality dim_eff(I) using PCA.
    
    Args:
        activations: Dict of layer_idx -> activation tensor
        variance_threshold: Variance threshold for dimensionality (default 0.9)
        per_layer: If True, compute per-layer; if False, compute global
        
    Returns:
        Dict with 'global' and/or per-layer dimensionalities
    """
    results = {}
    
    if per_layer:
        for layer_idx, act in activations.items():
            if act.ndim == 1:
                # Single sample: reshape to (1, features)
                act = act.reshape(1, -1)
            elif act.ndim == 2:
                # Multiple samples: (n_samples, features)
                pass
            else:
                raise ValueError(f"Unexpected activation shape: {act.shape}")
            
            if act.shape[0] < 2:
                # Cannot compute PCA with single sample
                results[f'layer_{layer_idx}'] = float('nan')
                continue
            
            act_np = act.numpy()
            pca = PCA()
            pca.fit(act_np)
            
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            dim_eff = np.searchsorted(cumvar, variance_threshold) + 1
            results[f'layer_{layer_idx}'] = float(dim_eff)
    
    else:
        # Global: concatenate all layers
        all_acts = []
        for layer_idx in sorted(activations.keys()):
            act = activations[layer_idx]
            if act.ndim == 1:
                all_acts.append(act.numpy())
            elif act.ndim == 2:
                # If multiple samples, take mean
                all_acts.append(act.mean(dim=0).numpy())
        
        if len(all_acts) == 0:
            results['global'] = float('nan')
        else:
            # Stack into (n_layers, hidden_dim)
            act_matrix = np.stack(all_acts, axis=0)
            
            if act_matrix.shape[0] < 2:
                results['global'] = float('nan')
            else:
                pca = PCA()
                pca.fit(act_matrix)
                
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                dim_eff = np.searchsorted(cumvar, variance_threshold) + 1
                results['global'] = float(dim_eff)
    
    return results


# ============================================================================
# PROBING FOR LATENT RECOVERABILITY
# ============================================================================

def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    regularization: float = 1.0,
    max_iter: int = 1000
) -> Dict[str, float]:
    """
    Train linear probe to predict correctness from pre-collapse state.
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,)
        X_test: Test features
        y_test: Test labels
        regularization: L2 regularization (C parameter)
        max_iter: Maximum iterations
        
    Returns:
        Dict with accuracy, AUROC, balanced_accuracy
    """
    # Normalize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    # Train probe
    probe = LogisticRegression(
        C=regularization,
        max_iter=max_iter,
        random_state=42,
        class_weight='balanced'  # Handle imbalance
    )
    probe.fit(X_train_norm, y_train)
    
    # Evaluate
    y_pred = probe.predict(X_test_norm)
    y_prob = probe.predict_proba(X_test_norm)[:, 1]
    
    accuracy = (y_pred == y_test).mean()
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    # AUROC (handle case where all labels are same class)
    try:
        auroc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auroc = float('nan')
    
    return {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'auroc': float(auroc)
    }


# ============================================================================
# EXPERIMENT ORCHESTRATION
# ============================================================================

def run_single_problem(
    model,
    tokenizer,
    question: str,
    ground_truth: str,
    regimes: Dict[str, Dict[str, Any]],
    layer_indices: List[int]
) -> Dict[str, Any]:
    """
    Run inference on a single problem across multiple regimes.
    
    Args:
        model: Loaded model
        tokenizer: Tokenizer
        question: Problem question
        ground_truth: Ground truth answer
        regimes: Dict of regime_name -> {prompt_template, max_tokens}
        layer_indices: Layers to extract from
        
    Returns:
        Dict with results for each regime
    """
    results = {
        'question': question,
        'ground_truth': ground_truth,
        'regimes': {}
    }
    
    for regime_name, config in regimes.items():
        prompt_template = config['prompt_template']
        max_tokens = config['max_tokens']
        
        prompt = prompt_template.format(question=question)
        
        try:
            activations, generated_text, first_logits = extract_activations_with_hooks(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                layer_indices=layer_indices,
                max_new_tokens=max_tokens,
                temperature=0.0,  # Greedy
                return_first_step_only=True
            )
            
            # Compute metrics
            H_int = compute_intention_entropy(first_logits)
            dim_eff_global = compute_effective_dimensionality(activations, per_layer=False)
            
            # Extract answer
            predicted_answer = extract_predicted_answer(generated_text, regime=regime_name)
            correct = check_correctness(predicted_answer, ground_truth)
            
            results['regimes'][regime_name] = {
                'generated_text': generated_text,
                'predicted_answer': predicted_answer,
                'correct': correct,
                'intention_entropy': H_int,
                'dim_eff_global': dim_eff_global.get('global', float('nan')),
                'output_length': len(tokenizer.encode(generated_text)),
                'activations_available': True
            }
            
            # Store activations separately (not in JSON)
            results['regimes'][regime_name]['_activations'] = activations
            
        except Exception as e:
            results['regimes'][regime_name] = {
                'error': str(e),
                'activations_available': False
            }
    
    return results


def save_results(results: List[Dict], output_path: Path):
    """Save results to JSON (excluding activation tensors)."""
    # Deep copy and remove activation tensors
    results_json = []
    for item in results:
        item_copy = item.copy()
        item_copy['regimes'] = {}
        for regime_name, regime_data in item['regimes'].items():
            regime_copy = regime_data.copy()
            regime_copy.pop('_activations', None)
            item_copy['regimes'][regime_name] = regime_copy
        results_json.append(item_copy)
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)


def aggregate_results(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across all problems.
    
    Returns:
        Dict of regime_name -> aggregated metrics
    """
    aggregated = {}
    
    # Get regime names
    regime_names = list(results[0]['regimes'].keys()) if len(results) > 0 else []
    
    for regime_name in regime_names:
        correct_count = 0
        total_count = 0
        entropies = []
        dim_effs = []
        output_lengths = []
        
        for item in results:
            regime_data = item['regimes'].get(regime_name, {})
            
            if regime_data.get('activations_available', False):
                total_count += 1
                if regime_data.get('correct', False):
                    correct_count += 1
                
                entropies.append(regime_data.get('intention_entropy', float('nan')))
                dim_effs.append(regime_data.get('dim_eff_global', float('nan')))
                output_lengths.append(regime_data.get('output_length', 0))
        
        aggregated[regime_name] = {
            'accuracy': correct_count / total_count if total_count > 0 else 0.0,
            'mean_intention_entropy': float(np.nanmean(entropies)) if entropies else float('nan'),
            'mean_dim_eff_global': float(np.nanmean(dim_effs)) if dim_effs else float('nan'),
            'mean_output_length': float(np.mean(output_lengths)) if output_lengths else 0.0,
            'n_problems': total_count
        }
    
    return aggregated


# ============================================================================
# PERSISTENCE: SPLITS AND CHECKPOINTS
# ============================================================================

def get_or_create_splits(
    dataset,
    splits_dir: Path,
    experiment_name: str = "default",
    seed: int = 42
) -> Dict[str, List[int]]:
    """
    Get or create reproducible train/val/test splits.
    
    Args:
        dataset: Dataset to split
        splits_dir: Directory to store split indices
        experiment_name: Name for this experiment's splits
        seed: Random seed
        
    Returns:
        Dict with 'train', 'val', 'test' index lists
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits_file = splits_dir / f"{experiment_name}_splits.json"
    
    if splits_file.exists():
        # Load existing
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        print(f"✓ Loaded existing splits from {splits_file}")
    else:
        # Create new
        n = len(dataset)
        indices = list(range(n))
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        # 70/15/15 split
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        
        splits = {
            'train': indices[:n_train],
            'val': indices[n_train:n_train + n_val],
            'test': indices[n_train + n_val:],
            'seed': seed
        }
        
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"✓ Created and saved new splits to {splits_file}")
    
    return splits


# ============================================================================
# VERSION INFO
# ============================================================================

__version__ = "3.0.1"

def get_version_info():
    return {
        'version': __version__,
        'description': 'Intention Collapse experiments - single source of truth utilities'
    }
