"""
Simple demo of the Adaptive Inference Router

This script demonstrates how to use the router for cost-effective inference.

Usage:
    python examples/router_demo.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from router import AdaptiveInferenceRouter, RouteDecision


def load_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Load model and tokenizer with 4-bit quantization."""
    print(f"Loading model: {model_name}...")

    from transformers import BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

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


def demo_basic_usage():
    """Demo 1: Basic usage with simple math problems."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Usage")
    print("="*70)

    # Load model
    model, tokenizer = load_model()

    # Create router
    router = AdaptiveInferenceRouter(
        model=model,
        tokenizer=tokenizer,
        entropy_threshold_low=0.5,
        entropy_threshold_high=1.2,
        benchmark='gsm8k',
        verbose=True
    )

    # Test problems (easy → hard)
    problems = [
        ("What is 2 + 2?", "4"),  # Easy, should route to DIRECT
        ("What is 15 * 23?", "345"),  # Medium, might go either way
        ("A train travels 60 miles in 45 minutes. What is its speed in miles per hour?", "80"),  # Harder, should route to CoT
    ]

    results = []
    for question, ground_truth in problems:
        print(f"\n📝 Question: {question}")
        print(f"   Ground truth: {ground_truth}")

        result = router.generate(
            question=question,
            ground_truth=ground_truth
        )

        results.append(result)

        print(f"   ├─ Route: {result.route_taken.value}")
        print(f"   ├─ Entropy: {result.intention_entropy:.3f} bits")
        print(f"   ├─ Answer: {result.extracted_answer}")
        print(f"   ├─ Correct: {'✓' if result.is_correct else '✗'}")
        print(f"   └─ Tokens: {result.input_tokens} + {result.output_tokens} = {result.total_tokens}")

    # Print summary statistics
    print("\n" + "-"*70)
    stats = router.get_statistics()
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   Direct routes: {stats['direct_routes']} ({stats['routing_distribution']['direct_pct']:.1f}%)")
    print(f"   CoT routes: {stats['cot_routes']} ({stats['routing_distribution']['cot_pct']:.1f}%)")
    print(f"   Total tokens: {stats['token_efficiency']['total_tokens']}")
    print(f"   Accuracy: {sum(r.is_correct for r in results)}/{len(results)} = {100*sum(r.is_correct for r in results)/len(results):.1f}%")

    return router, results


def demo_comparison_with_baselines():
    """Demo 2: Compare router with always-direct and always-CoT baselines."""
    print("\n" + "="*70)
    print("DEMO 2: Comparison with Baselines")
    print("="*70)

    # Load model
    model, tokenizer = load_model()

    # Create router
    router = AdaptiveInferenceRouter(
        model=model,
        tokenizer=tokenizer,
        entropy_threshold_low=0.5,
        entropy_threshold_high=1.2,
        benchmark='gsm8k',
        verbose=False  # Disable verbose for cleaner output
    )

    # Test on a small dataset
    test_problems = [
        ("What is 5 + 7?", "12"),
        ("What is 12 * 12?", "144"),
        ("If John has 3 apples and Mary gives him 5 more, how many apples does John have?", "8"),
        ("A rectangle has width 4 and length 9. What is its area?", "36"),
        ("Calculate 15% of 200.", "30"),
    ]

    strategies = {
        'Adaptive Router': None,  # Let router decide
        'Always Direct': RouteDecision.DIRECT,
        'Always CoT': RouteDecision.COT,
    }

    results_by_strategy = {}

    for strategy_name, force_route in strategies.items():
        print(f"\n{strategy_name}:")
        router.reset_statistics()

        results = []
        for question, ground_truth in test_problems:
            result = router.generate(
                question=question,
                ground_truth=ground_truth,
                force_route=force_route
            )
            results.append(result)

        results_by_strategy[strategy_name] = results

        # Calculate metrics
        accuracy = sum(r.is_correct for r in results) / len(results)
        total_tokens = sum(r.total_tokens for r in results)
        avg_tokens = total_tokens / len(results)
        efficiency = accuracy / (total_tokens / 1000)  # Accuracy per 1k tokens

        print(f"  Accuracy: {accuracy:.1%} ({sum(r.is_correct for r in results)}/{len(results)})")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Avg tokens/query: {avg_tokens:.1f}")
        print(f"  Efficiency: {efficiency:.4f} (accuracy per 1k tokens)")

    # Print comparison
    print("\n" + "-"*70)
    print("COMPARISON:")
    print(f"{'Strategy':<20} {'Accuracy':>10} {'Total Tokens':>15} {'Efficiency':>12}")
    print("-"*70)

    for strategy_name, results in results_by_strategy.items():
        accuracy = sum(r.is_correct for r in results) / len(results)
        total_tokens = sum(r.total_tokens for r in results)
        efficiency = accuracy / (total_tokens / 1000)

        print(f"{strategy_name:<20} {accuracy:>9.1%} {total_tokens:>15} {efficiency:>12.4f}")

    return results_by_strategy


def demo_threshold_tuning():
    """Demo 3: Show how different thresholds affect routing."""
    print("\n" + "="*70)
    print("DEMO 3: Threshold Tuning")
    print("="*70)

    # Load model
    model, tokenizer = load_model()

    # Test problem
    question = "If a car travels 150 miles in 2.5 hours, what is its average speed?"
    ground_truth = "60"

    # Try different threshold configurations
    threshold_configs = [
        (0.3, 0.8, "Conservative (prefer CoT)"),
        (0.5, 1.2, "Balanced (default)"),
        (0.8, 1.5, "Aggressive (prefer Direct)"),
    ]

    print(f"\n📝 Test question: {question}")
    print(f"   Ground truth: {ground_truth}")

    for low, high, description in threshold_configs:
        print(f"\n{description} [low={low}, high={high}]:")

        router = AdaptiveInferenceRouter(
            model=model,
            tokenizer=tokenizer,
            entropy_threshold_low=low,
            entropy_threshold_high=high,
            benchmark='gsm8k',
            verbose=False
        )

        result = router.generate(question=question, ground_truth=ground_truth)

        print(f"  Entropy: {result.intention_entropy:.3f} bits")
        print(f"  Route: {result.route_taken.value}")
        print(f"  Tokens: {result.total_tokens}")
        print(f"  Correct: {'✓' if result.is_correct else '✗'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive Router Demo")
    parser.add_argument(
        '--demo',
        type=int,
        choices=[1, 2, 3],
        help="Which demo to run (1=basic, 2=comparison, 3=threshold tuning)"
    )
    args = parser.parse_args()

    print("\n🚀 ADAPTIVE INFERENCE ROUTER DEMO")
    print("="*70)
    print("This demo shows how to use intention entropy H_int(I) as a")
    print("'traffic cop' to route between direct answer and Chain-of-Thought.")
    print("="*70)

    try:
        if args.demo == 1 or args.demo is None:
            demo_basic_usage()

        if args.demo == 2 or args.demo is None:
            demo_comparison_with_baselines()

        if args.demo == 3 or args.demo is None:
            demo_threshold_tuning()

        print("\n" + "="*70)
        print("✅ Demo completed!")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
