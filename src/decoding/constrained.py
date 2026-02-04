"""
Constrained Decoding Implementation

Forces the model to generate only valid multiple-choice option tokens.

Key Insight:
    Even when the model has low option-normalized entropy (knows the answer),
    it may generate invalid format like "The correct answer is C" instead of "C".

    Constrained decoding FORCES the model to emit only valid option tokens,
    completely eliminating compliance issues.

Reference:
    - Standard generation: p(y|x) over full vocabulary
    - Constrained generation: p(y|x, y ∈ O) where O = {A, B, C, D, E}
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple, Union, Any
from dataclasses import dataclass
import re

try:
    from transformers import LogitsProcessor, LogitsProcessorList
except ImportError:
    # Fallback for older transformers
    class LogitsProcessor:
        def __call__(self, input_ids, scores):
            return scores

    class LogitsProcessorList(list):
        pass


@dataclass
class ConstrainedGenerationResult:
    """Result from constrained generation."""
    generated_text: str
    option_selected: str
    token_id_used: int
    logits_before_constraint: torch.Tensor
    logits_after_constraint: torch.Tensor
    probability_before: float  # p(option) before constraint
    probability_after: float   # p(option) after constraint (should be ~1.0)
    was_most_likely: bool      # Was this option the model's top choice?


class MultipleChoiceLogitsProcessor(LogitsProcessor):
    """
    Logits processor that restricts generation to valid MC option tokens.

    This processor sets logits of all non-option tokens to -inf,
    forcing the model to only sample from valid options.

    Usage:
        processor = MultipleChoiceLogitsProcessor(tokenizer, valid_options=['A','B','C','D'])
        outputs = model.generate(
            inputs,
            logits_processor=LogitsProcessorList([processor]),
            max_new_tokens=1
        )
    """

    def __init__(
        self,
        tokenizer,
        valid_options: List[str] = ['A', 'B', 'C', 'D'],
        allow_eos: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the processor.

        Args:
            tokenizer: HuggingFace tokenizer
            valid_options: List of valid option letters
            allow_eos: Whether to allow EOS token (for early stopping)
            verbose: Print debugging information
        """
        self.tokenizer = tokenizer
        self.valid_options = valid_options
        self.allow_eos = allow_eos
        self.verbose = verbose

        # Get token IDs for options
        self.option_token_ids = self._get_option_token_ids()

        if self.verbose:
            print(f"MultipleChoiceLogitsProcessor initialized:")
            print(f"  Valid options: {valid_options}")
            print(f"  Token IDs: {self.option_token_ids}")

    def _get_option_token_ids(self) -> Dict[str, int]:
        """Get token IDs for each option."""
        option_ids = {}

        for option in self.valid_options:
            # Try different encodings
            candidates = [
                option,           # "A"
                f" {option}",     # " A"
                f"{option}.",     # "A."
                f" {option}.",    # " A."
            ]

            token_ids = []
            for candidate in candidates:
                ids = self.tokenizer.encode(candidate, add_special_tokens=False)
                if ids:
                    token_ids.append(ids[0])

            if token_ids:
                # Use most common
                from collections import Counter
                most_common = Counter(token_ids).most_common(1)[0][0]
                option_ids[option] = most_common

        return option_ids

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Process logits to constrain to valid options.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            scores: Logits for next token (batch_size, vocab_size)

        Returns:
            Modified logits with only valid options allowed
        """
        # Create mask for valid tokens
        vocab_size = scores.shape[-1]
        batch_size = scores.shape[0]

        # Initialize mask to -inf (disallow all tokens)
        mask = torch.full_like(scores, float('-inf'))

        # Allow valid option tokens
        for option, token_id in self.option_token_ids.items():
            if token_id < vocab_size:
                mask[:, token_id] = 0.0  # 0 = no penalty

        # Optionally allow EOS token
        if self.allow_eos:
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is not None and eos_token_id < vocab_size:
                mask[:, eos_token_id] = 0.0

        # Apply mask
        constrained_scores = scores + mask

        if self.verbose and batch_size == 1:
            # Show top tokens before and after
            probs_before = F.softmax(scores[0], dim=-1)
            probs_after = F.softmax(constrained_scores[0], dim=-1)

            print(f"\n  Logits processing:")
            print(f"    Valid token IDs: {list(self.option_token_ids.values())}")

            # Show option probabilities
            print(f"    Option probabilities:")
            for opt, tid in self.option_token_ids.items():
                prob_before = probs_before[tid].item()
                prob_after = probs_after[tid].item()
                print(f"      {opt}: {prob_before:.1%} → {prob_after:.1%}")

        return constrained_scores


class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    """
    More flexible processor that allows a prefix before the option token.

    For example, allows "The answer is C" by constraining only after
    detecting a trigger phrase.

    This is useful if you want the model to show some reasoning but
    still guarantee a valid final answer.
    """

    def __init__(
        self,
        tokenizer,
        valid_options: List[str] = ['A', 'B', 'C', 'D'],
        trigger_phrases: List[str] = ["answer is", "Answer:", "####"],
        max_prefix_length: int = 20,
        verbose: bool = False
    ):
        """
        Initialize prefix-constrained processor.

        Args:
            tokenizer: HuggingFace tokenizer
            valid_options: Valid option letters
            trigger_phrases: Phrases that trigger constraint
            max_prefix_length: Maximum tokens before constraint kicks in
            verbose: Print debugging info
        """
        self.tokenizer = tokenizer
        self.valid_options = valid_options
        self.trigger_phrases = trigger_phrases
        self.max_prefix_length = max_prefix_length
        self.verbose = verbose

        # Get option token IDs
        self.mc_processor = MultipleChoiceLogitsProcessor(
            tokenizer, valid_options, allow_eos=False, verbose=False
        )

        self.triggered = False

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Process logits with prefix awareness."""

        # Decode current sequence
        current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        # Check if we should trigger constraint
        should_constrain = False

        # Trigger 1: Detected trigger phrase
        for phrase in self.trigger_phrases:
            if phrase.lower() in current_text.lower():
                should_constrain = True
                break

        # Trigger 2: Reached max prefix length
        generated_length = input_ids.shape[1]
        if generated_length >= self.max_prefix_length:
            should_constrain = True

        if should_constrain:
            if self.verbose and not self.triggered:
                print(f"  Constraint triggered! Generated so far: '{current_text}'")
            self.triggered = True
            return self.mc_processor(input_ids, scores)
        else:
            # No constraint yet, allow free generation
            return scores


def extract_first_option_token(
    generated_ids: torch.Tensor,
    tokenizer,
    valid_options: List[str] = ['A', 'B', 'C', 'D']
) -> Optional[str]:
    """
    Extract the first valid option token from generated IDs.

    Args:
        generated_ids: Generated token IDs
        tokenizer: Tokenizer
        valid_options: Valid option letters

    Returns:
        First valid option found, or None
    """
    # Get option token IDs
    from ..metrics import get_option_token_ids
    option_ids = get_option_token_ids(tokenizer, valid_options)

    # Reverse lookup: token_id -> option
    id_to_option = {v: k for k, v in option_ids.items()}

    # Find first valid option in generated sequence
    for token_id in generated_ids.squeeze().tolist():
        if token_id in id_to_option:
            return id_to_option[token_id]

    return None


def constrained_mc_generation(
    model,
    tokenizer,
    prompt: str,
    valid_options: List[str] = ['A', 'B', 'C', 'D'],
    strategy: str = "force_first_token",
    max_new_tokens: int = 1,
    return_diagnostics: bool = False,
    verbose: bool = False,
    **generation_kwargs
) -> Union[str, ConstrainedGenerationResult]:
    """
    Generate multiple-choice answer with constrained decoding.

    This is the main function for constrained MC generation. It forces
    the model to only generate valid option tokens, eliminating compliance issues.

    Strategies:
        - "force_first_token": Force ONLY first token to be valid option (strictest)
        - "prefix_allowed": Allow reasoning prefix, then constrain (flexible)
        - "anywhere": Allow generation, extract first valid option

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        valid_options: Valid option letters (e.g., ['A','B','C','D'])
        strategy: Constraint strategy (see above)
        max_new_tokens: Maximum tokens to generate
        return_diagnostics: Return full ConstrainedGenerationResult
        verbose: Print debugging information
        **generation_kwargs: Additional args for model.generate()

    Returns:
        If return_diagnostics=False: Selected option letter (str)
        If return_diagnostics=True: ConstrainedGenerationResult with full info

    Example:
        >>> answer = constrained_mc_generation(
        ...     model, tokenizer, prompt,
        ...     valid_options=['A','B','C','D'],
        ...     strategy='force_first_token'
        ... )
        >>> print(answer)  # 'C'

        >>> result = constrained_mc_generation(
        ...     model, tokenizer, prompt,
        ...     valid_options=['A','B','C','D'],
        ...     return_diagnostics=True
        ... )
        >>> print(f"Selected: {result.option_selected}")
        >>> print(f"Probability before constraint: {result.probability_before:.1%}")
        >>> print(f"Was most likely: {result.was_most_likely}")
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    # Get logits BEFORE constraint (for diagnostics)
    with torch.no_grad():
        outputs_unconstrained = model(**inputs)
        logits_before = outputs_unconstrained.logits[0, -1, :].cpu()

    # Create appropriate processor based on strategy
    if strategy == "force_first_token":
        # Strictest: force first token to be option
        processor = MultipleChoiceLogitsProcessor(
            tokenizer,
            valid_options=valid_options,
            allow_eos=False,
            verbose=verbose
        )
        logits_processor = LogitsProcessorList([processor])
        max_tokens = 1  # Only generate one token

    elif strategy == "prefix_allowed":
        # Allow reasoning, then constrain
        processor = PrefixConstrainedLogitsProcessor(
            tokenizer,
            valid_options=valid_options,
            trigger_phrases=["answer is", "Answer:", "####", "letter"],
            max_prefix_length=max_new_tokens - 1,
            verbose=verbose
        )
        logits_processor = LogitsProcessorList([processor])
        max_tokens = max_new_tokens

    elif strategy == "anywhere":
        # Free generation, just extract first valid option
        logits_processor = None
        max_tokens = max_new_tokens

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Generate with constraint
    gen_kwargs = {
        'max_new_tokens': max_tokens,
        'do_sample': False,  # Greedy for reproducibility
        'pad_token_id': tokenizer.eos_token_id,
        **generation_kwargs
    }

    if logits_processor is not None:
        gen_kwargs['logits_processor'] = logits_processor

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Extract generated tokens (skip input)
    generated_ids = outputs[0, input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract option
    option_selected = extract_first_option_token(
        generated_ids, tokenizer, valid_options
    )

    if option_selected is None:
        # Fallback: try parsing from text
        for opt in valid_options:
            if opt in generated_text:
                option_selected = opt
                break

    if verbose:
        print(f"\n  Generated: '{generated_text}'")
        print(f"  Selected option: {option_selected}")

    # Simple return
    if not return_diagnostics:
        return option_selected if option_selected else ""

    # Full diagnostics
    from ..metrics import get_option_token_ids, compute_option_normalized_entropy

    option_ids = get_option_token_ids(tokenizer, valid_options)

    if option_selected and option_selected in option_ids:
        token_id_used = option_ids[option_selected]

        # Get probabilities before constraint
        probs_before = F.softmax(logits_before, dim=-1)
        prob_before = probs_before[token_id_used].item()

        # Was it most likely?
        option_probs_before = {
            opt: probs_before[tid].item()
            for opt, tid in option_ids.items()
        }
        most_likely_before = max(option_probs_before.items(), key=lambda x: x[1])[0]
        was_most_likely = (option_selected == most_likely_before)

        # After constraint (should be ~1.0 for selected option)
        processor_temp = MultipleChoiceLogitsProcessor(
            tokenizer, valid_options, verbose=False
        )
        logits_after = processor_temp(
            inputs.input_ids,
            logits_before.unsqueeze(0)
        ).squeeze(0)
        probs_after = F.softmax(logits_after, dim=-1)
        prob_after = probs_after[token_id_used].item()

    else:
        # Fallback values
        token_id_used = -1
        prob_before = 0.0
        prob_after = 0.0
        was_most_likely = False
        logits_after = logits_before

    result = ConstrainedGenerationResult(
        generated_text=generated_text,
        option_selected=option_selected if option_selected else "",
        token_id_used=token_id_used,
        logits_before_constraint=logits_before,
        logits_after_constraint=logits_after,
        probability_before=prob_before,
        probability_after=prob_after,
        was_most_likely=was_most_likely
    )

    return result


def compare_free_vs_constrained(
    model,
    tokenizer,
    prompt: str,
    valid_options: List[str] = ['A', 'B', 'C', 'D'],
    ground_truth: Optional[str] = None,
    max_new_tokens: int = 50
) -> Dict[str, Any]:
    """
    Compare free generation vs constrained generation.

    Useful for diagnosing compliance issues.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        prompt: Input prompt
        valid_options: Valid options
        ground_truth: Correct answer (for evaluation)
        max_new_tokens: Max tokens for free generation

    Returns:
        Dictionary with comparison results
    """
    # Free generation
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs_free = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    free_text = tokenizer.decode(
        outputs_free[0, input_length:],
        skip_special_tokens=True
    )

    # Extract option from free text
    free_option = None
    for opt in valid_options:
        if opt in free_text:
            free_option = opt
            break

    # Constrained generation
    constrained_result = constrained_mc_generation(
        model, tokenizer, prompt,
        valid_options=valid_options,
        strategy='force_first_token',
        return_diagnostics=True,
        verbose=False
    )

    # Compare
    comparison = {
        'free_generation': {
            'text': free_text,
            'extracted_option': free_option,
            'valid_format': free_option in valid_options if free_option else False,
            'correct': free_option == ground_truth if ground_truth and free_option else None
        },
        'constrained_generation': {
            'text': constrained_result.generated_text,
            'extracted_option': constrained_result.option_selected,
            'valid_format': True,  # Always valid by construction
            'correct': constrained_result.option_selected == ground_truth if ground_truth else None,
            'probability': constrained_result.probability_before,
            'was_most_likely': constrained_result.was_most_likely
        },
        'compliance_issue': free_option != constrained_result.option_selected,
        'ground_truth': ground_truth
    }

    return comparison
