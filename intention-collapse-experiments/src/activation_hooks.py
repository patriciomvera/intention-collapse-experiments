"""
Activation Extraction Utilities

This module provides tools for extracting hidden state activations from
transformer models during inference. These activations constitute the
"intention state" I in the Intention Collapse framework.

Reference: Section 2.1 "Intention State in Practice"
    I_t = (h_t, KV_t, R_t, c_t) ∈ I
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from contextlib import contextmanager


class ActivationExtractor:
    """
    Extract hidden state activations from specified transformer layers.
    
    This class uses PyTorch hooks to capture intermediate activations
    during the forward pass without modifying the model.
    
    Usage:
        extractor = ActivationExtractor(model, layers=[27, 28, 29, 30, 31])
        with extractor.capture():
            outputs = model.generate(...)
        activations = extractor.get_activations()
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        layers: List[int],
        extraction_point: str = "post_layernorm"
    ):
        """
        Initialize the activation extractor.
        
        Args:
            model: The transformer model (HuggingFace format)
            layers: List of layer indices to extract from (0-indexed)
            extraction_point: Where to extract activations:
                - "post_layernorm": After layer normalization (default)
                - "residual": Residual stream output
                - "attention": Attention output only
                - "mlp": MLP output only
        """
        self.model = model
        self.layers = sorted(layers)
        self.extraction_point = extraction_point
        
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._activations: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
        self._is_capturing = False
        
        # Detect model architecture
        self._detect_architecture()
    
    def _detect_architecture(self):
        """Detect the model architecture and set appropriate layer paths."""
        model_type = self.model.config.model_type.lower()
        
        if model_type in ["mistral", "llama", "qwen2"]:
            # These models have similar architectures
            self._layer_module_template = "model.layers.{}"
            self._hidden_states_attr = None  # We'll use hooks
        elif model_type == "gpt2":
            self._layer_module_template = "transformer.h.{}"
        else:
            # Default to common pattern
            self._layer_module_template = "model.layers.{}"
        
        # Verify layers exist
        max_layers = self.model.config.num_hidden_layers
        for layer in self.layers:
            if layer >= max_layers:
                raise ValueError(
                    f"Layer {layer} does not exist. Model has {max_layers} layers."
                )
    
    def _get_layer_module(self, layer_idx: int) -> torch.nn.Module:
        """Get the module for a specific layer."""
        parts = self._layer_module_template.format(layer_idx).split(".")
        module = self.model
        for part in parts:
            module = getattr(module, part)
        return module
    
    def _create_hook(self, layer_idx: int) -> Callable:
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            if not self._is_capturing:
                return
            
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Store the hidden states (detached, on CPU for memory)
            self._activations[layer_idx].append(
                hidden_states.detach().cpu()
            )
        
        return hook
    
    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        for layer_idx in self.layers:
            layer_module = self._get_layer_module(layer_idx)
            hook = layer_module.register_forward_hook(self._create_hook(layer_idx))
            self._hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def clear(self):
        """Clear stored activations."""
        self._activations = {l: [] for l in self.layers}
    
    @contextmanager
    def capture(self):
        """
        Context manager for capturing activations during forward pass.
        
        Usage:
            with extractor.capture():
                outputs = model.generate(...)
        """
        self.clear()
        self._register_hooks()
        self._is_capturing = True
        
        try:
            yield self
        finally:
            self._is_capturing = False
            self._remove_hooks()
    
    def get_activations(
        self,
        layer: Optional[int] = None,
        aggregate: str = "last"
    ) -> np.ndarray:
        """
        Get extracted activations.
        
        Args:
            layer: Specific layer to get (None for all layers)
            aggregate: How to aggregate across sequence positions:
                - "last": Last token position (default, for causal LMs)
                - "mean": Mean across positions
                - "all": Return all positions
                
        Returns:
            Numpy array of activations
        """
        if layer is not None:
            layers_to_get = [layer]
        else:
            layers_to_get = self.layers
        
        all_activations = []
        
        for l in layers_to_get:
            if not self._activations[l]:
                raise ValueError(f"No activations captured for layer {l}")
            
            # Concatenate all captured tensors for this layer
            layer_acts = torch.cat(self._activations[l], dim=0)
            
            # Aggregate across sequence dimension
            if aggregate == "last":
                layer_acts = layer_acts[:, -1, :]  # Last position
            elif aggregate == "mean":
                layer_acts = layer_acts.mean(dim=1)
            elif aggregate == "all":
                pass  # Keep all positions
            else:
                raise ValueError(f"Unknown aggregation: {aggregate}")
            
            all_activations.append(layer_acts.numpy())
        
        # Stack layers
        if len(all_activations) == 1:
            return all_activations[0]
        else:
            return np.stack(all_activations, axis=0)
    
    def get_activation_trajectory(self, layer: int) -> List[np.ndarray]:
        """
        Get activations as a trajectory across generation steps.
        
        Useful for observing how the intention state evolves during
        the thinking phase.
        
        Args:
            layer: Layer index to get trajectory for
            
        Returns:
            List of activation arrays, one per generation step
        """
        if not self._activations[layer]:
            raise ValueError(f"No activations captured for layer {layer}")
        
        return [t.numpy() for t in self._activations[layer]]


class LogitsExtractor:
    """
    Extract logits at each generation step.
    
    This is useful for computing intention entropy trajectory
    and observing the U-shaped entropy curve.
    """
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize logits extractor.
        
        Args:
            model: The transformer model
        """
        self.model = model
        self._logits_history: List[torch.Tensor] = []
        self._hook: Optional[torch.utils.hooks.RemovableHandle] = None
        self._is_capturing = False
    
    def _get_lm_head(self) -> torch.nn.Module:
        """Get the language model head module."""
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'lm_head'):
            return self.model.model.lm_head
        else:
            raise AttributeError("Could not find lm_head in model")
    
    def _create_hook(self) -> Callable:
        """Create hook for capturing logits."""
        def hook(module, input, output):
            if self._is_capturing:
                # Output shape: (batch, seq, vocab)
                # We want the last position
                last_logits = output[:, -1, :].detach().cpu()
                self._logits_history.append(last_logits)
        return hook
    
    def clear(self):
        """Clear stored logits."""
        self._logits_history = []
    
    @contextmanager
    def capture(self):
        """Context manager for capturing logits during generation."""
        self.clear()
        lm_head = self._get_lm_head()
        self._hook = lm_head.register_forward_hook(self._create_hook())
        self._is_capturing = True
        
        try:
            yield self
        finally:
            self._is_capturing = False
            if self._hook is not None:
                self._hook.remove()
                self._hook = None
    
    def get_logits_trajectory(self) -> List[torch.Tensor]:
        """Get the sequence of logits from generation."""
        return self._logits_history
    
    def get_final_logits(self) -> torch.Tensor:
        """Get logits from the last generation step."""
        if not self._logits_history:
            raise ValueError("No logits captured")
        return self._logits_history[-1]


class CombinedExtractor:
    """
    Combined extractor for both activations and logits.
    
    Provides a unified interface for capturing all information needed
    to compute intention metrics.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        layers: List[int]
    ):
        """
        Initialize combined extractor.
        
        Args:
            model: The transformer model
            layers: Layer indices for activation extraction
        """
        self.activation_extractor = ActivationExtractor(model, layers)
        self.logits_extractor = LogitsExtractor(model)
        self.model = model
    
    @contextmanager
    def capture(self):
        """Capture both activations and logits."""
        with self.activation_extractor.capture():
            with self.logits_extractor.capture():
                yield self
    
    def get_activations(self, **kwargs) -> np.ndarray:
        """Get extracted activations."""
        return self.activation_extractor.get_activations(**kwargs)
    
    def get_logits_trajectory(self) -> List[torch.Tensor]:
        """Get logits trajectory."""
        return self.logits_extractor.get_logits_trajectory()
    
    def get_final_logits(self) -> torch.Tensor:
        """Get final logits."""
        return self.logits_extractor.get_final_logits()
    
    def clear(self):
        """Clear all stored data."""
        self.activation_extractor.clear()
        self.logits_extractor.clear()
