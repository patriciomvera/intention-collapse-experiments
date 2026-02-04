"""
Self-Consistency Control Baseline Demonstration

This script demonstrates self-consistency as a rigorous control baseline
for intention collapse experiments.

Key Idea:
    Instead of "babble" (meaningless output), use self-consistency:
    - Generate N diverse answers with sampling
    - Take majority vote
    - Compare against single CoT generation

Usage:
    python examples/self_consistency_demo.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from controls import self_consistency_baseline, compare_self_consistency_vs_cot
from router import AdaptiveInferenceRouter


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

    print(f"✓ Model loaded on {next(model.parameters()).device}\n")
    return model, tokenizer


def demo_basic_self_consistency():
    """Demo 1: Basic self-consistency usage."""
    print("="*80)
    print("DEMO 1: Basic Self-Consistency")
    print("="*80)

    model, tokenizer = load_model()

    # Test problem
    prompt = """Solve this math problem. Give the final numerical answer after ####.

Problem: A baker makes 12 loaves of bread in the morning and 8 loaves in the afternoon. If each loaf uses 3 cups of flour, how many cups of flour did the baker use in total?
Answer:"""

    print(f"Question: A baker makes 12 loaves of bread in the morning and 8 loaves in the afternoon...")
    print(f"Ground truth: 60")
    print("\n" + "-"*80)

    # Generate with self-consistency
    result = self_consistency_baseline(
        model,
        tokenizer,
        prompt,
        n_samples=5,
        temperature=0.7,
        max_new_tokens=100,
        benchmark='gsm8k',
        ground_truth='60',
        verbose=True
    )

    print("\n" + "-"*80)
    print("FINAL RESULT:")
    print(f"  Answer: {result.final_answer}")
    print(f"  Confidence: {result.confidence:.1%} ({result.answer_counts})")
    print(f"  Correct: {'✓' if result.is_correct else '✗'}")
    print(f"  Total tokens: {result.total_tokens}")
    print(f"  Avg tokens/sample: {result.avg_tokens_per_sample:.1f}")
    print(f"  Answer entropy: {result.answer_entropy:.3f} bits")


def demo_self_consistency_vs_cot():
    """Demo 2: Compare self-consistency vs CoT."""
    print("\n" + "="*80)
    print("DEMO 2: Self-Consistency vs Chain-of-Thought")
    print("="*80)

    model, tokenizer = load_model()

    # Test problems
    problems = [
        {
            'prompt': """Solve this math problem step by step. Give the final answer after ####.

Problem: Sarah has 24 candies. She gives 1/3 of them to her friend and eats 4 herself. How many candies does she have left?
Answer:""",
            'ground_truth': '12'
        },
        {
            'prompt': """Solve this math problem step by step. Give the final answer after ####.

Problem: A store sells notebooks for $3 each. If you buy 5 notebooks, you get a 20% discount on the total price. How much do you pay for 5 notebooks?
Answer:""",
            'ground_truth': '12'
        }
    ]

    for i, problem in enumerate(problems, 1):
        print(f"\n{'='*80}")
        print(f"Problem {i}")
        print(f"{'='*80}")

        comparison = compare_self_consistency_vs_cot(
            model,
            tokenizer,
            problem['prompt'],
            n_samples=5,
            temperature=0.7,
            max_new_tokens=150,
            benchmark='gsm8k',
            ground_truth=problem['ground_truth'],
            verbose=False
        )


def demo_confidence_analysis():
    """Demo 3: Analyze confidence and diversity."""
    print("\n" + "="*80)
    print("DEMO 3: Confidence and Diversity Analysis")
    print("="*80)

    model, tokenizer = load_model()

    # Easy problem (should have high confidence)
    easy_prompt = """Solve this math problem. Give the final answer after ####.

Problem: What is 10 + 5?
Answer:"""

    # Hard problem (should have lower confidence)
    hard_prompt = """Solve this math problem. Give the final answer after ####.

Problem: A rectangular garden is 15 meters long and 8 meters wide. A path 1 meter wide is built around the outside of the garden. What is the area of the path in square meters?
Answer:"""

    problems = [
        ("Easy", easy_prompt, "15"),
        ("Hard", hard_prompt, "62")
    ]

    for difficulty, prompt, ground_truth in problems:
        print(f"\n{'-'*80}")
        print(f"{difficulty} Problem")
        print(f"{'-'*80}")

        result = self_consistency_baseline(
            model,
            tokenizer,
            prompt,
            n_samples=5,
            temperature=0.7,
            max_new_tokens=150,
            benchmark='gsm8k',
            ground_truth=ground_truth,
            verbose=True
        )

        print(f"\n  Confidence: {result.confidence:.1%}")
        print(f"  Answer entropy: {result.answer_entropy:.3f} bits")

        if result.confidence == 1.0:
            print(f"  → Perfect consensus (all samples agree)")
        elif result.confidence >= 0.6:
            print(f"  → Strong majority")
        elif result.confidence >= 0.4:
            print(f"  → Weak majority")
        else:
            print(f"  → No clear majority (tie)")

        if result.answer_entropy > 1.5:
            print(f"  → High diversity (model very uncertain)")
        elif result.answer_entropy > 0.8:
            print(f"  → Moderate diversity")
        else:
            print(f"  → Low diversity (model confident)")


def demo_integration_with_router():
    """Demo 4: Integrate self-consistency with adaptive router."""
    print("\n" + "="*80)
    print("DEMO 4: Integration with Adaptive Router")
    print("="*80)
    print("For high entropy problems, use self-consistency instead of single CoT\n")

    model, tokenizer = load_model()

    # Test problems with different entropy levels
    problems = [
        {
            'question': "What is 25 * 4?",
            'ground_truth': "100",
            'expected_entropy': 'low'
        },
        {
            'question': "A train travels 180 miles in 3 hours. Another train travels 240 miles in 4 hours. Which train is faster and by how much (in mph)?",
            'ground_truth': "60",  # First train is faster by 0 mph (they're equal)
            'expected_entropy': 'high'
        }
    ]

    for i, problem in enumerate(problems, 1):
        print(f"\n{'-'*80}")
        print(f"Problem {i}: {problem['question'][:60]}...")
        print(f"Expected entropy: {problem['expected_entropy']}")
        print(f"{'-'*80}")

        # Create router
        router = AdaptiveInferenceRouter(
            model=model,
            tokenizer=tokenizer,
            entropy_threshold_low=0.5,
            entropy_threshold_high=1.2,
            benchmark='gsm8k',
            verbose=False
        )

        # Get routing decision
        prompt = f"Solve this math problem step by step. Give the final answer after ####.\\n\\nProblem: {problem['question']}\\nAnswer:"

        # Compute entropy
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]

        from metrics import compute_intention_entropy
        entropy = compute_intention_entropy(logits)

        print(f"\n  Entropy: {entropy:.3f} bits")

        # Routing decision
        if entropy < 0.5:
            strategy = "Direct"
            print(f"  → Route: DIRECT (entropy < 0.5)")

            result = router.generate(
                question=problem['question'],
                ground_truth=problem['ground_truth']
            )

            print(f"  → Answer: {result.extracted_answer}")
            print(f"  → Tokens: {result.total_tokens}")
            print(f"  → Correct: {'✓' if result.is_correct else '✗'}")

        elif entropy < 1.2:
            strategy = "CoT"
            print(f"  → Route: COT (0.5 ≤ entropy < 1.2)")

            result = router.generate(
                question=problem['question'],
                ground_truth=problem['ground_truth']
            )

            print(f"  → Answer: {result.extracted_answer}")
            print(f"  → Tokens: {result.total_tokens}")
            print(f"  → Correct: {'✓' if result.is_correct else '✗'}")

        else:
            strategy = "Self-Consistency"
            print(f"  → Route: SELF-CONSISTENCY (entropy ≥ 1.2)")
            print(f"  → Using majority vote over 5 samples...")

            sc_result = self_consistency_baseline(
                model,
                tokenizer,
                prompt,
                n_samples=5,
                temperature=0.7,
                max_new_tokens=200,
                benchmark='gsm8k',
                ground_truth=problem['ground_truth'],
                verbose=False
            )

            print(f"  → Answer: {sc_result.final_answer}")
            print(f"  → Confidence: {sc_result.confidence:.1%}")
            print(f"  → Tokens: {sc_result.total_tokens}")
            print(f"  → Correct: {'✓' if sc_result.is_correct else '✗'}")

    print("\n" + "="*80)
    print("ROUTING STRATEGY:")
    print("="*80)
    print("  H_int < 0.5:         Direct answer (fast, cheap)")
    print("  0.5 ≤ H_int < 1.2:   Chain-of-Thought (reasoned, moderate cost)")
    print("  H_int ≥ 1.2:         Self-Consistency (robust, expensive)")
    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Self-Consistency Demo")
    parser.add_argument(
        '--demo',
        type=int,
        choices=[1, 2, 3, 4],
        help="Which demo to run"
    )
    args = parser.parse_args()

    print("\n🎯 SELF-CONSISTENCY CONTROL BASELINE")
    print("="*80)
    print("This demo shows self-consistency as a rigorous control baseline")
    print("for intention collapse experiments.")
    print("="*80)

    try:
        if args.demo == 1 or args.demo is None:
            demo_basic_self_consistency()

        if args.demo == 2 or args.demo is None:
            demo_self_consistency_vs_cot()

        if args.demo == 3 or args.demo is None:
            demo_confidence_analysis()

        if args.demo == 4 or args.demo is None:
            demo_integration_with_router()

        print("\n" + "="*80)
        print("✅ Demo completed!")
        print("="*80)
        print("\n📚 Key Takeaways:")
        print("   1. Self-consistency generates N diverse samples + majority vote")
        print("   2. More robust than single CoT for uncertain problems")
        print("   3. Confidence score indicates agreement level")
        print("   4. Answer entropy measures diversity")
        print("   5. Can be integrated with router for high-entropy problems")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
