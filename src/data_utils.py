"""
Data Utilities for Intention Collapse Experiments

This module handles loading datasets (GSM8K, MATH, ARC) and
evaluating model outputs against ground truth answers.
"""

import re
import random
from typing import List, Dict, Tuple, Optional, Any
from datasets import load_dataset
from dataclasses import dataclass


@dataclass
class MathProblem:
    """Container for a math problem from GSM8K or MATH."""
    question: str
    answer: str  # Full solution with reasoning
    final_answer: str  # Just the numerical answer
    idx: int  # Index in dataset


def load_gsm8k(
    split: str = "test",
    subset_size: Optional[int] = None,
    seed: int = 42
) -> List[MathProblem]:
    """
    Load GSM8K dataset.
    
    GSM8K contains grade school math word problems with step-by-step solutions.
    
    Args:
        split: Dataset split ("train" or "test")
        subset_size: Number of examples to load (None for all)
        seed: Random seed for subsetting
        
    Returns:
        List of MathProblem objects
    """
    # Load dataset
    dataset = load_dataset("gsm8k", "main", split=split)
    
    # Subset if requested
    if subset_size is not None and subset_size < len(dataset):
        random.seed(seed)
        indices = random.sample(range(len(dataset)), subset_size)
        dataset = dataset.select(indices)
    
    problems = []
    for idx, item in enumerate(dataset):
        # Extract final answer from solution
        # GSM8K format: solution ends with "#### <answer>"
        solution = item['answer']
        final_answer = extract_gsm8k_answer(solution)
        
        problems.append(MathProblem(
            question=item['question'],
            answer=solution,
            final_answer=final_answer,
            idx=idx
        ))
    
    return problems


def extract_gsm8k_answer(solution: str) -> str:
    """
    Extract the final numerical answer from a GSM8K solution.
    
    GSM8K solutions end with "#### <answer>".
    
    Args:
        solution: Full solution text
        
    Returns:
        Extracted numerical answer as string
    """
    # Look for #### pattern
    match = re.search(r'####\s*(.+?)$', solution, re.MULTILINE)
    if match:
        answer = match.group(1).strip()
        # Clean up commas in numbers (e.g., "1,000" -> "1000")
        answer = answer.replace(',', '')
        return answer
    
    # Fallback: try to find last number in solution
    numbers = re.findall(r'-?\d+\.?\d*', solution)
    if numbers:
        return numbers[-1]
    
    return ""


def extract_answer(
    model_output: str,
    method: str = "auto"
) -> str:
    """
    Extract numerical answer from model output.
    
    Args:
        model_output: Raw model output text
        method: Extraction method:
            - "auto": Try multiple methods
            - "last_number": Take last number in output
            - "boxed": Look for \\boxed{} format
            - "answer_tag": Look for "Answer:" or "answer:" prefix
            
    Returns:
        Extracted answer as string
    """
    # Clean output
    output = model_output.strip()
    
    if method == "auto":
        # Try each method in order
        for m in ["answer_tag", "boxed", "last_number"]:
            result = extract_answer(output, method=m)
            if result:
                return result
        return ""
    
    elif method == "last_number":
        # Find all numbers and take the last one
        numbers = re.findall(r'-?\d+\.?\d*', output)
        if numbers:
            return numbers[-1].replace(',', '')
        return ""
    
    elif method == "boxed":
        # LaTeX \boxed{} format (common in MATH dataset)
        match = re.search(r'\\boxed\{([^}]+)\}', output)
        if match:
            return match.group(1).strip().replace(',', '')
        return ""
    
    elif method == "answer_tag":
        # Look for "Answer: X" or "answer: X" or "The answer is X"
        patterns = [
            r'[Aa]nswer[:\s]+(-?\d+\.?\d*)',
            r'[Tt]he answer is[:\s]+(-?\d+\.?\d*)',
            r'=\s*(-?\d+\.?\d*)\s*$',
            r'####\s*(-?\d+\.?\d*)',
        ]
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1).replace(',', '')
        return ""
    
    else:
        raise ValueError(f"Unknown extraction method: {method}")


def evaluate_answer(
    predicted: str,
    ground_truth: str,
    tolerance: float = 1e-6
) -> bool:
    """
    Check if predicted answer matches ground truth.
    
    Args:
        predicted: Model's predicted answer
        ground_truth: Correct answer
        tolerance: Tolerance for floating point comparison
        
    Returns:
        True if answers match, False otherwise
    """
    # Clean both answers
    pred_clean = predicted.strip().replace(',', '').replace('$', '')
    truth_clean = ground_truth.strip().replace(',', '').replace('$', '')
    
    # Try exact string match first
    if pred_clean == truth_clean:
        return True
    
    # Try numerical comparison
    try:
        pred_num = float(pred_clean)
        truth_num = float(truth_clean)
        
        # Exact match for integers
        if pred_num == truth_num:
            return True
        
        # Tolerance-based match for floats
        if abs(pred_num - truth_num) < tolerance:
            return True
        
        # Relative tolerance for large numbers
        if truth_num != 0 and abs((pred_num - truth_num) / truth_num) < tolerance:
            return True
            
    except ValueError:
        pass
    
    return False


def format_prompt(
    question: str,
    template: str,
    few_shot_examples: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Format a question into a prompt using the specified template.
    
    Args:
        question: The math problem
        template: Prompt template with {question} placeholder
        few_shot_examples: Optional list of example Q&A pairs
        
    Returns:
        Formatted prompt string
    """
    # Add few-shot examples if provided
    if few_shot_examples:
        examples_text = ""
        for ex in few_shot_examples:
            examples_text += f"Problem: {ex['question']}\n"
            examples_text += f"Solution: {ex['answer']}\n\n"
        template = examples_text + template
    
    return template.format(question=question)


def get_few_shot_examples(n: int = 3, seed: int = 42) -> List[Dict[str, str]]:
    """
    Get few-shot examples from GSM8K train set.
    
    Args:
        n: Number of examples
        seed: Random seed
        
    Returns:
        List of example dictionaries
    """
    train_data = load_dataset("gsm8k", "main", split="train")
    random.seed(seed)
    indices = random.sample(range(len(train_data)), n)
    
    examples = []
    for idx in indices:
        item = train_data[idx]
        examples.append({
            'question': item['question'],
            'answer': item['answer']
        })
    
    return examples


@dataclass
class EvaluationResult:
    """Results from evaluating model on a problem."""
    problem_idx: int
    question: str
    ground_truth: str
    model_output: str
    extracted_answer: str
    is_correct: bool
    
    # Optional metrics (filled in during experiment)
    entropy: Optional[float] = None
    dim_eff: Optional[int] = None


def evaluate_batch(
    problems: List[MathProblem],
    model_outputs: List[str]
) -> Tuple[List[EvaluationResult], float]:
    """
    Evaluate a batch of model outputs.
    
    Args:
        problems: List of MathProblem objects
        model_outputs: List of model output strings
        
    Returns:
        Tuple of (list of EvaluationResults, overall accuracy)
    """
    if len(problems) != len(model_outputs):
        raise ValueError("Number of problems and outputs must match")
    
    results = []
    correct_count = 0
    
    for problem, output in zip(problems, model_outputs):
        extracted = extract_answer(output)
        is_correct = evaluate_answer(extracted, problem.final_answer)
        
        if is_correct:
            correct_count += 1
        
        results.append(EvaluationResult(
            problem_idx=problem.idx,
            question=problem.question,
            ground_truth=problem.final_answer,
            model_output=output,
            extracted_answer=extracted,
            is_correct=is_correct
        ))
    
    accuracy = correct_count / len(problems) if problems else 0.0
    
    return results, accuracy


class DataCollector:
    """
    Collect and organize data from experiments for later analysis.
    """
    
    def __init__(self):
        self.results: Dict[str, List[EvaluationResult]] = {
            'baseline': [],
            'enhanced': []
        }
        self.activations: Dict[str, List] = {
            'baseline': [],
            'enhanced': []
        }
        self.logits: Dict[str, List] = {
            'baseline': [],
            'enhanced': []
        }
    
    def add_result(
        self,
        condition: str,
        result: EvaluationResult,
        activations: Optional[Any] = None,
        logits: Optional[Any] = None
    ):
        """Add a single result."""
        self.results[condition].append(result)
        if activations is not None:
            self.activations[condition].append(activations)
        if logits is not None:
            self.logits[condition].append(logits)
    
    def get_accuracy(self, condition: str) -> float:
        """Get accuracy for a condition."""
        results = self.results[condition]
        if not results:
            return 0.0
        return sum(r.is_correct for r in results) / len(results)
    
    def get_correctness_labels(self, condition: str) -> List[bool]:
        """Get list of correctness labels."""
        return [r.is_correct for r in self.results[condition]]
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            'baseline_accuracy': self.get_accuracy('baseline'),
            'enhanced_accuracy': self.get_accuracy('enhanced'),
            'baseline_n': len(self.results['baseline']),
            'enhanced_n': len(self.results['enhanced']),
            'improvement': self.get_accuracy('enhanced') - self.get_accuracy('baseline')
        }
