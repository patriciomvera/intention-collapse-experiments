"""
Adaptive Router Experiment - Standalone Script

Evaluates the Adaptive Inference Router on GSM8K problems.

Usage:
    python experiments/run_router_experiment.py [--n_problems 200] [--model qwen]

Options:
    --n_problems: Number of problems to evaluate (default: 200)
    --model: Model to use: qwen, mistral, llama (default: qwen)
    --thresholds: Comma-separated low,high thresholds (default: 0.5,1.2)
    --output_dir: Output directory (default: ../results/router_experiment)
    --seed: Random seed (default: 42)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import random
from collections import defaultdict
from datetime import datetime

from router import AdaptiveInferenceRouter, RouteDecision


# Model configurations
MODELS = {
    'qwen': 'Qwen/Qwen2.5-7B-Instruct',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
    'llama': 'meta-llama/Llama-3.1-8B-Instruct'
}


def load_model_and_tokenizer(model_name, quantization='4bit'):
    """Load model and tokenizer."""
    print(f"\nLoading model: {model_name}")

    if quantization == '4bit':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif quantization == '8bit':
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    print(f"✓ Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


def load_gsm8k_problems(n_problems, seed=42):
    """Load GSM8K problems."""
    print(f"\nLoading GSM8K dataset ({n_problems} problems)...")

    dataset = load_dataset("gsm8k", "main", split="test")

    random.seed(seed)
    indices = sorted(random.sample(range(len(dataset)), n_problems))

    problems = []
    for idx in indices:
        item = dataset[idx]
        answer_parts = item['answer'].split('####')
        if len(answer_parts) == 2:
            ground_truth = answer_parts[1].strip()
        else:
            import re
            numbers = re.findall(r'-?\d+\.?\d*', item['answer'])
            ground_truth = numbers[-1] if numbers else ""

        problems.append({
            'idx': idx,
            'question': item['question'],
            'full_answer': item['answer'],
            'ground_truth': ground_truth
        })

    print(f"✓ Loaded {len(problems)} problems")
    return problems


def evaluate_strategy(strategy_name, problems, router, force_route=None):
    """Evaluate a routing strategy."""
    results = []
    total_tokens = 0
    correct_count = 0
    route_counts = defaultdict(int)
    entropy_history = []

    print(f"\n{'='*80}")
    print(f"Evaluating: {strategy_name}")
    print(f"{'='*80}")

    for i, problem in enumerate(tqdm(problems, desc=strategy_name)):
        try:
            result = router.generate(
                question=problem['question'],
                ground_truth=problem['ground_truth'],
                force_route=force_route
            )

            total_tokens += result.total_tokens
            if result.is_correct:
                correct_count += 1

            route_counts[result.route_taken.value] += 1
            entropy_history.append(result.intention_entropy)

            results.append({
                'problem_idx': i,
                'question': problem['question'][:100] + '...',
                'ground_truth': problem['ground_truth'],
                'extracted_answer': result.extracted_answer,
                'is_correct': result.is_correct,
                'route_taken': result.route_taken.value,
                'entropy': result.intention_entropy,
                'total_tokens': result.total_tokens,
                'input_tokens': result.input_tokens,
                'output_tokens': result.output_tokens
            })

        except Exception as e:
            print(f"\nError on problem {i}: {e}")
            results.append({
                'problem_idx': i,
                'is_correct': False,
                'total_tokens': 0,
                'entropy': float('nan')
            })

    accuracy = correct_count / len(problems)
    avg_tokens = total_tokens / len(problems)
    efficiency = accuracy / (total_tokens / 1000)

    summary = {
        'strategy': strategy_name,
        'n_problems': len(problems),
        'correct': correct_count,
        'accuracy': accuracy,
        'total_tokens': total_tokens,
        'avg_tokens': avg_tokens,
        'efficiency': efficiency,
        'route_distribution': dict(route_counts),
        'entropy_mean': np.nanmean(entropy_history),
        'entropy_std': np.nanstd(entropy_history)
    }

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.1%} ({correct_count}/{len(problems)})")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg tokens/query: {avg_tokens:.1f}")
    print(f"  Efficiency: {efficiency:.4f}")
    print(f"  Route distribution: {dict(route_counts)}")

    return {'summary': summary, 'results': results}


def evaluate_random_strategy(problems, router, seed=42):
    """Evaluate random routing."""
    random.seed(seed)

    results = []
    total_tokens = 0
    correct_count = 0
    route_counts = defaultdict(int)

    print(f"\n{'='*80}")
    print(f"Evaluating: Random Routing")
    print(f"{'='*80}")

    for i, problem in enumerate(tqdm(problems, desc="Random")):
        try:
            force_route = random.choice([RouteDecision.DIRECT, RouteDecision.COT])

            result = router.generate(
                question=problem['question'],
                ground_truth=problem['ground_truth'],
                force_route=force_route
            )

            total_tokens += result.total_tokens
            if result.is_correct:
                correct_count += 1

            route_counts[result.route_taken.value] += 1

            results.append({
                'problem_idx': i,
                'is_correct': result.is_correct,
                'total_tokens': result.total_tokens,
                'route_taken': result.route_taken.value
            })
        except Exception as e:
            print(f"\nError on problem {i}: {e}")
            results.append({'problem_idx': i, 'is_correct': False, 'total_tokens': 0})

    accuracy = correct_count / len(problems)
    avg_tokens = total_tokens / len(problems)
    efficiency = accuracy / (total_tokens / 1000)

    summary = {
        'strategy': 'Random',
        'accuracy': accuracy,
        'total_tokens': total_tokens,
        'avg_tokens': avg_tokens,
        'efficiency': efficiency,
        'route_distribution': dict(route_counts)
    }

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg tokens/query: {avg_tokens:.1f}")
    print(f"  Efficiency: {efficiency:.4f}")

    return {'summary': summary, 'results': results}


def create_visualizations(all_results, output_dir):
    """Create visualization plots."""
    print("\nCreating visualizations...")

    strategies = [r['strategy'] for r in all_results]
    accuracies = [r['accuracy'] * 100 for r in all_results]
    avg_tokens = [r['avg_tokens'] for r in all_results]
    colors = ['#3498db', '#e74c3c', '#95a5a6', '#2ecc71']

    # Plot 1: Accuracy and Token comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = ax1.bar(strategies, accuracies, color=colors, alpha=0.8)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    bars2 = ax2.bar(strategies, avg_tokens, color=colors, alpha=0.8)
    ax2.set_ylabel('Avg Tokens per Query', fontsize=12)
    ax2.set_title('Token Usage Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.0f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=300, bbox_inches='tight')
    print("  ✓ Saved: comparison.png")

    # Plot 2: Efficiency scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, result in enumerate(all_results):
        ax.scatter(result['avg_tokens'], result['accuracy'] * 100,
                   s=300, c=colors[i], alpha=0.7, edgecolors='black', linewidth=2)
        ax.annotate(result['strategy'], (result['avg_tokens'], result['accuracy'] * 100),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold')

    ax.set_xlabel('Average Tokens per Query', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Accuracy vs Token Efficiency', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_scatter.png'), dpi=300, bbox_inches='tight')
    print("  ✓ Saved: efficiency_scatter.png")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Router Experiment")
    parser.add_argument('--n_problems', type=int, default=200, help="Number of problems")
    parser.add_argument('--model', type=str, default='qwen', choices=['qwen', 'mistral', 'llama'])
    parser.add_argument('--thresholds', type=str, default='0.5,1.2', help="low,high thresholds")
    parser.add_argument('--output_dir', type=str, default='../results/router_experiment')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Parse thresholds
    threshold_low, threshold_high = map(float, args.thresholds.split(','))

    print("\n" + "="*80)
    print("ADAPTIVE ROUTER EXPERIMENT")
    print("="*80)
    print(f"Model: {MODELS[args.model]}")
    print(f"Problems: {args.n_problems}")
    print(f"Thresholds: low={threshold_low}, high={threshold_high}")
    print(f"Output: {args.output_dir}")
    print(f"Seed: {args.seed}")

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_model_and_tokenizer(MODELS[args.model])

    # Load problems
    problems = load_gsm8k_problems(args.n_problems, args.seed)

    # Initialize router
    print("\nInitializing router...")
    router = AdaptiveInferenceRouter(
        model=model,
        tokenizer=tokenizer,
        entropy_threshold_low=threshold_low,
        entropy_threshold_high=threshold_high,
        benchmark='gsm8k',
        verbose=False
    )
    print("✓ Router initialized")

    # Run experiments
    all_results = []

    # 1. Always Direct
    router.reset_statistics()
    results_direct = evaluate_strategy("Always-Direct", problems, router, RouteDecision.DIRECT)
    all_results.append(results_direct['summary'])

    # 2. Always CoT
    router.reset_statistics()
    results_cot = evaluate_strategy("Always-CoT", problems, router, RouteDecision.COT)
    all_results.append(results_cot['summary'])

    # 3. Random
    router.reset_statistics()
    results_random = evaluate_random_strategy(problems, router, args.seed)
    all_results.append(results_random['summary'])

    # 4. Adaptive
    router.reset_statistics()
    results_adaptive = evaluate_strategy("Adaptive-Router", problems, router, None)
    all_results.append(results_adaptive['summary'])

    # Print comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    df = pd.DataFrame(all_results)
    df = df[['strategy', 'accuracy', 'total_tokens', 'avg_tokens', 'efficiency']]
    print(df.to_string(index=False))

    # Print key findings
    adaptive = results_adaptive['summary']
    cot = results_cot['summary']

    accuracy_diff = (adaptive['accuracy'] - cot['accuracy']) * 100
    token_savings = (1 - adaptive['total_tokens'] / cot['total_tokens']) * 100
    efficiency_improvement = (adaptive['efficiency'] / cot['efficiency'] - 1) * 100

    print(f"\n📊 ADAPTIVE vs ALWAYS-COT:")
    print(f"  Accuracy: {accuracy_diff:+.1f} percentage points")
    print(f"  Token savings: {token_savings:.1f}%")
    print(f"  Efficiency improvement: {efficiency_improvement:+.1f}%")

    # Save results
    print(f"\nSaving results to {args.output_dir}...")

    # Summary
    with open(os.path.join(args.output_dir, 'experiment_summary.json'), 'w') as f:
        json.dump({
            'config': vars(args),
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }, f, indent=2)
    print("  ✓ Saved: experiment_summary.json")

    # Detailed results
    for name, results in [
        ('always_direct', results_direct),
        ('always_cot', results_cot),
        ('random', results_random),
        ('adaptive', results_adaptive)
    ]:
        with open(os.path.join(args.output_dir, f'{name}_detailed.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  ✓ Saved: {name}_detailed.json")

    # CSV
    df.to_csv(os.path.join(args.output_dir, 'comparison.csv'), index=False)
    print("  ✓ Saved: comparison.csv")

    # Visualizations
    create_visualizations(all_results, args.output_dir)

    print("\n" + "="*80)
    print("✅ EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
