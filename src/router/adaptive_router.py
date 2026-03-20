"""
Adaptive Inference Router

Routes queries to direct answer or Chain-of-Thought based on intention entropy H_int(I).
This is the core practical application of the Intention Collapse framework.

Key Insight:
    - Low H_int(I) → Model has clear intention → Use direct answer (cheap)
    - High H_int(I) → Model is uncertain → Use CoT reasoning (expensive but necessary)

Reference:
    "Intention collapse occurs when the model's internal representation
     crystallizes into a specific plan before generating the first token."
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from enum import Enum

# Import existing utilities from the repo
from ..metrics import (
    compute_intention_entropy,
    compute_option_normalized_entropy,
    compute_entropy_decomposition
)


class RouteDecision(str, Enum):
    """Routing decision based on intention entropy."""
    DIRECT = "direct"      # Low entropy → use baseline/direct answer
    COT = "cot"           # High entropy → use chain-of-thought
    UNKNOWN = "unknown"    # Could not compute entropy


@dataclass
class RouterResult:
    """Result from adaptive router including generation and metadata."""

    # Generation output
    generated_text: str
    extracted_answer: str

    # Routing decision
    route_taken: RouteDecision
    intention_entropy: float

    # Cost metrics
    input_tokens: int
    output_tokens: int
    total_tokens: int

    # Additional metadata
    confidence_score: Optional[float] = None  # Distance from thresholds
    generation_time: Optional[float] = None
    top_tokens: Optional[Dict[str, float]] = None

    # For evaluation
    is_correct: Optional[bool] = None
    ground_truth: Optional[str] = None


class AdaptiveInferenceRouter:
    """
    Adaptive router that uses intention entropy as a traffic cop.

    This router performs a lightweight forward pass on the prompt to compute
    H_int(I), then routes to either:
    - Direct answer (for confident, low-entropy states)
    - Chain-of-Thought (for uncertain, high-entropy states)

    Usage:
        router = AdaptiveInferenceRouter(model, tokenizer)
        result = router.generate("What is 2 + 2?")
        print(f"Route: {result.route_taken}, Answer: {result.extracted_answer}")
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        entropy_threshold_low: float = 0.5,
        entropy_threshold_high: float = 1.2,
        device: Optional[str] = None,
        benchmark: str = "gsm8k",
        use_option_normalized_entropy: Optional[bool] = None,
        use_constrained_decoding: Optional[bool] = None,
        verbose: bool = False
    ):
        """
        Initialize the adaptive router.

        Args:
            model: HuggingFace transformer model (AutoModelForCausalLM)
            tokenizer: HuggingFace tokenizer
            entropy_threshold_low: Entropy below this → use direct answer
            entropy_threshold_high: Entropy above this → use CoT
            device: Device to run on (None = auto-detect)
            benchmark: Benchmark type for prompt formatting ('gsm8k', 'arc', 'aqua')
            use_option_normalized_entropy: Use option-normalized entropy for MC questions
                                           (None = auto-detect based on benchmark)
            use_constrained_decoding: Force valid MC options for MC benchmarks
                                     (None = auto-detect based on benchmark)
            verbose: Print routing decisions

        Threshold Design:
            - H_int < low: Clear intention, use direct (e.g., 0.5 bits)
            - low ≤ H_int < high: Uncertain, prefer CoT (e.g., 0.5-1.2 bits)
            - H_int ≥ high: Very uncertain, definitely use CoT (e.g., > 1.2 bits)

            Typical entropy ranges (from paper):
            - GSM8K baseline: 0.2-0.8 bits (mean ~0.4)
            - GSM8K enhanced: 0.4-1.5 bits (mean ~0.8)
            - ARC baseline: 0.3-1.0 bits

            For option-normalized entropy (MC questions):
            - Max entropy for 4 options: log2(4) = 2.0 bits
            - Max entropy for 5 options: log2(5) = 2.32 bits
            - Adjust thresholds accordingly (e.g., low=0.8, high=1.5)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.entropy_threshold_low = entropy_threshold_low
        self.entropy_threshold_high = entropy_threshold_high
        self.benchmark = benchmark
        self.verbose = verbose

        # Auto-detect if we should use option-normalized entropy
        if use_option_normalized_entropy is None:
            # Use for multiple-choice benchmarks
            self.use_option_normalized_entropy = benchmark in ['arc', 'aqua']
        else:
            self.use_option_normalized_entropy = use_option_normalized_entropy

        # Auto-detect if we should use constrained decoding
        if use_constrained_decoding is None:
            # Use for multiple-choice benchmarks
            self.use_constrained_decoding = benchmark in ['arc', 'aqua']
        else:
            self.use_constrained_decoding = use_constrained_decoding

        # Define valid options per benchmark
        self.mc_options = {
            'arc': ['A', 'B', 'C', 'D'],
            'aqua': ['A', 'B', 'C', 'D', 'E']
        }

        # Auto-detect device
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

        # Set model to eval mode
        self.model.eval()

        # Statistics tracking
        self.stats = {
            'total_queries': 0,
            'direct_routes': 0,
            'cot_routes': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'entropy_history': []
        }

    def reset_statistics(self):
        """Reset statistics tracking for new experiment runs."""
        self.stats = {
            'total_queries': 0,
            'direct_routes': 0,
            'cot_routes': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'entropy_history': []
        }

    def compute_intention_entropy(
        self,
        prompt: str,
        top_k: int = 100,
        temperature: float = 1.0,
        return_decomposition: bool = False
    ) -> tuple[float, torch.Tensor, Optional[Dict]]:
        """
        Compute intention entropy H_int(I) from prompt.

        This performs a single forward pass on the prompt and computes
        the entropy of the next-token distribution.

        For multiple-choice benchmarks (ARC, AQUA), automatically uses
        option-normalized entropy if enabled.

        Args:
            prompt: Input prompt text
            top_k: Number of top tokens to consider for standard entropy
            temperature: Temperature for softmax
            return_decomposition: Return full entropy decomposition (for MC)

        Returns:
            Tuple of (entropy in bits, logits tensor, optional decomposition dict)
        """
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Forward pass (no gradient needed)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Get logits for the last token (pre-collapse state)
            last_logits = logits[0, -1, :]

        # Compute entropy based on benchmark type
        decomposition = None

        if self.use_option_normalized_entropy and self.benchmark in self.mc_options:
            # Use option-normalized entropy for MC questions
            options = self.mc_options[self.benchmark]

            if return_decomposition:
                # Get full decomposition
                decomposition = compute_entropy_decomposition(
                    last_logits,
                    self.tokenizer,
                    options=options,
                    top_k=top_k
                )
                entropy = decomposition['option_normalized']
            else:
                # Just get option-normalized entropy
                entropy = compute_option_normalized_entropy(
                    last_logits,
                    self.tokenizer,
                    options=options,
                    temperature=temperature
                )
        else:
            # Use standard entropy
            entropy = compute_intention_entropy(
                last_logits,
                top_k=top_k,
                temperature=temperature
            )

        return entropy, last_logits, decomposition

    def route(
        self,
        prompt: str,
        force_route: Optional[RouteDecision] = None
    ) -> tuple[RouteDecision, float, Optional[float]]:
        """
        Decide routing strategy based on intention entropy.

        Args:
            prompt: Input prompt text
            force_route: Force a specific route (for ablation studies)

        Returns:
            Tuple of (route_decision, entropy, confidence_score)

        Confidence Score:
            - Measures how far entropy is from decision boundaries
            - Positive = confident in decision, negative = near boundary
        """
        # If forced route, skip computation
        if force_route is not None:
            return force_route, float('nan'), None

        # Compute intention entropy
        try:
            entropy, _, _ = self.compute_intention_entropy(prompt)
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error computing entropy: {e}")
            return RouteDecision.UNKNOWN, float('nan'), None

        # Store entropy for statistics
        self.stats['entropy_history'].append(entropy)

        # Routing decision logic
        if entropy < self.entropy_threshold_low:
            decision = RouteDecision.DIRECT
            # Confidence: how far below threshold
            confidence = (self.entropy_threshold_low - entropy) / self.entropy_threshold_low
        elif entropy < self.entropy_threshold_high:
            # In uncertain zone, prefer CoT
            decision = RouteDecision.COT
            # Confidence: how far above low threshold
            confidence = (entropy - self.entropy_threshold_low) / \
                        (self.entropy_threshold_high - self.entropy_threshold_low)
        else:
            # Very high entropy, definitely CoT
            decision = RouteDecision.COT
            # Confidence: how far above high threshold
            confidence = min((entropy - self.entropy_threshold_high) / self.entropy_threshold_high, 1.0)

        if self.verbose:
            print(f"📊 Entropy: {entropy:.3f} bits → Route: {decision.value} (confidence: {confidence:.2f})")

        return decision, entropy, confidence

    def _format_prompt(self, question: str, condition: str, choices: str = "") -> str:
        """
        Format prompt based on benchmark and condition.

        Args:
            question: Question text
            condition: 'baseline' or 'enhanced'
            choices: Multiple choice options (for ARC/AQUA)

        Returns:
            Formatted prompt string
        """
        # Prompt templates (simplified from shared_utils.py)
        templates = {
            'gsm8k': {
                'baseline': "Solve this math problem. Give only the final numerical answer after ####.\n\nProblem: {question}\nAnswer:",
                'enhanced': "Solve this math problem step by step. Show your reasoning, then give the final answer after ####.\n\nProblem: {question}\nSolution:"
            },
            'arc': {
                'baseline': "Answer this science question. Give only the letter (A, B, C, or D) after ####.\n\nQuestion: {question}\n{choices}\nAnswer:",
                'enhanced': "Answer this science question step by step. Reason through each option, then give the final answer letter after ####.\n\nQuestion: {question}\n{choices}\nSolution:"
            },
            'aqua': {
                'baseline': "Solve this algebra problem. Give only the letter (A, B, C, D, or E) after ####.\n\nProblem: {question}\n{choices}\nAnswer:",
                'enhanced': "Solve this algebra problem step by step. Show your work, then give the final answer letter after ####.\n\nProblem: {question}\n{choices}\nSolution:"
            }
        }

        template = templates.get(self.benchmark, templates['gsm8k'])[condition]

        if choices:
            return template.format(question=question, choices=choices)
        return template.format(question=question)

    def generate(
        self,
        question: str,
        choices: str = "",
        force_route: Optional[RouteDecision] = None,
        ground_truth: Optional[str] = None,
        max_tokens_direct: int = 50,
        max_tokens_cot: int = 512,
        **generation_kwargs
    ) -> RouterResult:
        """
        Generate answer using adaptive routing.

        This is the main method that orchestrates the entire pipeline:
        1. Compute intention entropy from prompt
        2. Route to direct or CoT
        3. Generate response
        4. Extract answer
        5. Return result with metadata

        Args:
            question: Question text
            choices: Multiple choice options (for ARC/AQUA)
            force_route: Force specific route (for baselines)
            ground_truth: Ground truth answer (for evaluation)
            max_tokens_direct: Max tokens for direct answer
            max_tokens_cot: Max tokens for CoT
            **generation_kwargs: Additional args for model.generate()

        Returns:
            RouterResult with generation output and metadata
        """
        import time
        start_time = time.time()

        # Update statistics
        self.stats['total_queries'] += 1

        # Step 1: Route decision (using baseline prompt for entropy computation)
        baseline_prompt = self._format_prompt(question, 'baseline', choices)
        decision, entropy, confidence = self.route(baseline_prompt, force_route)

        # Update routing statistics
        if decision == RouteDecision.DIRECT:
            self.stats['direct_routes'] += 1
        elif decision == RouteDecision.COT:
            self.stats['cot_routes'] += 1

        # Step 2: Format final prompt based on decision
        if decision == RouteDecision.DIRECT:
            prompt = baseline_prompt
            max_new_tokens = max_tokens_direct
            condition = 'baseline'
        elif decision == RouteDecision.COT:
            prompt = self._format_prompt(question, 'enhanced', choices)
            max_new_tokens = max_tokens_cot
            condition = 'enhanced'
        else:
            # Fallback to direct if unknown
            prompt = baseline_prompt
            max_new_tokens = max_tokens_direct
            condition = 'baseline'

        # Step 3: Generate response
        # Use constrained decoding for MC questions if enabled
        if self.use_constrained_decoding and self.benchmark in self.mc_options:
            # Import constrained decoding
            try:
                from ..decoding import constrained_mc_generation

                # Get valid options
                valid_options = self.mc_options[self.benchmark]

                # Choose strategy based on route
                if decision == RouteDecision.DIRECT:
                    strategy = 'force_first_token'  # Strictest, 1 token
                    max_new_tokens = 1
                else:
                    strategy = 'prefix_allowed'  # Allow reasoning, then constrain

                # Generate with constraint
                constrained_result = constrained_mc_generation(
                    self.model,
                    self.tokenizer,
                    prompt,
                    valid_options=valid_options,
                    strategy=strategy,
                    max_new_tokens=max_new_tokens,
                    return_diagnostics=True,
                    verbose=self.verbose
                )

                generated_text = constrained_result.generated_text
                input_length = len(self.tokenizer.encode(prompt))
                output_length = len(self.tokenizer.encode(generated_text, add_special_tokens=False))

            except ImportError:
                # Fallback to standard generation
                if self.verbose:
                    print("⚠️  Constrained decoding not available, using standard generation")
                self.use_constrained_decoding = False
                # Fall through to standard generation below

        # Standard generation (for non-MC or if constrained decoding disabled)
        if not (self.use_constrained_decoding and self.benchmark in self.mc_options):
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            input_length = inputs.input_ids.shape[1]

            # Default generation parameters
            gen_kwargs = {
                'max_new_tokens': max_new_tokens,
                'do_sample': False,  # Greedy decoding for reproducibility
                'pad_token_id': self.tokenizer.eos_token_id,
                'temperature': None,  # Greedy
                'top_p': None,
                **generation_kwargs
            }

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            # Decode output
            generated_ids = outputs[0, input_length:]  # Skip input tokens
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            output_length = len(generated_ids)

        total_tokens = input_length + output_length

        # Update token statistics
        self.stats['total_input_tokens'] += input_length
        self.stats['total_output_tokens'] += output_length

        generation_time = time.time() - start_time

        # Step 4: Extract answer (simplified version)
        extracted_answer = self._extract_answer(generated_text, prompt + generated_text)

        # Step 5: Evaluate if ground truth provided
        is_correct = None
        if ground_truth is not None:
            is_correct = self._evaluate_answer(extracted_answer, ground_truth)

        # Construct result
        result = RouterResult(
            generated_text=generated_text,
            extracted_answer=extracted_answer,
            route_taken=decision,
            intention_entropy=entropy,
            input_tokens=input_length,
            output_tokens=output_length,
            total_tokens=total_tokens,
            confidence_score=confidence,
            generation_time=generation_time,
            is_correct=is_correct,
            ground_truth=ground_truth
        )

        if self.verbose:
            print(f"✅ Generated in {generation_time:.2f}s | "
                  f"Tokens: {input_length} + {output_length} = {total_tokens} | "
                  f"Answer: {extracted_answer}")

        return result

    def _extract_answer(self, generated_text: str, full_text: str) -> str:
        """
        Extract the final answer from generated text.

        Delegates to the production-grade extractors in shared_utils.py,
        which handle all edge cases (commas, $, "Final answer:", "= X", etc.)

        Args:
            generated_text: Generated text only (not including the prompt)
            full_text: Full text including prompt (unused, kept for API compat)

        Returns:
            Extracted answer string (cleaned, no commas or currency symbols)
        """
        try:
            from ..shared_utils import extract_answer
            answer, _ = extract_answer(generated_text, self.benchmark)
            return answer
        except Exception:
            pass

        # Fallback if shared_utils not available
        import re
        numbers = re.findall(r'-?[\d,]+\.?\d*', generated_text)
        numbers = [n.replace(',', '') for n in numbers
                   if n.replace(',', '').replace('.', '').replace('-', '').isdigit()]
        return numbers[-1] if numbers else ""

    def _evaluate_answer(self, predicted: str, ground_truth: str) -> bool:
        """
        Evaluate if predicted answer matches ground truth.

        Delegates to production-grade evaluators in shared_utils.py.

        Args:
            predicted: Predicted answer (already cleaned by _extract_answer)
            ground_truth: Ground truth answer

        Returns:
            True if correct, False otherwise
        """
        try:
            from ..shared_utils import evaluate_answer
            is_correct, _ = evaluate_answer(predicted, ground_truth, self.benchmark)
            return is_correct
        except Exception:
            pass

        # Fallback: numerical comparison with comma/$ stripping
        pred = predicted.strip().replace(',', '').replace('$', '').replace('%', '')
        truth = ground_truth.strip().replace(',', '').replace('$', '').replace('%', '')
        if pred == truth:
            return True
        try:
            return abs(float(pred) - float(truth)) < 1e-6
        except ValueError:
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get routing statistics.

        Returns:
            Dictionary with statistics:
            - Total queries
            - Direct vs CoT routing breakdown
            - Token usage
            - Entropy statistics
        """
        total = self.stats['total_queries']
        if total == 0:
            return self.stats

        stats = self.stats.copy()
        stats['routing_distribution'] = {
            'direct_pct': 100 * self.stats['direct_routes'] / total,
            'cot_pct': 100 * self.stats['cot_routes'] / total
        }

        if self.stats['entropy_history']:
            stats['entropy_stats'] = {
                'mean': np.mean(self.stats['entropy_history']),
                'std': np.std(self.stats['entropy_history']),
                'min': np.min(self.stats['entropy_history']),
                'max': np.max(self.stats['entropy_history']),
                'median': np.median(self.stats['entropy_history'])
            }

        stats['token_efficiency'] = {
            'avg_input_tokens': self.stats['total_input_tokens'] / total,
            'avg_output_tokens': self.stats['total_output_tokens'] / total,
            'total_tokens': self.stats['total_input_tokens'] + self.stats['total_output_tokens']
        }

        return stats

    def reset_statistics(self):
        """Reset all statistics counters."""
        self.stats = {
            'total_queries': 0,
            'direct_routes': 0,
            'cot_routes': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'entropy_history': []
        }

    def set_thresholds(self, low: float, high: float):
        """
        Update entropy thresholds.

        Useful for tuning after initial experiments.

        Args:
            low: New low threshold
            high: New high threshold
        """
        if low >= high:
            raise ValueError(f"Low threshold ({low}) must be < high threshold ({high})")

        self.entropy_threshold_low = low
        self.entropy_threshold_high = high

        if self.verbose:
            print(f"🔧 Updated thresholds: low={low:.2f}, high={high:.2f}")
