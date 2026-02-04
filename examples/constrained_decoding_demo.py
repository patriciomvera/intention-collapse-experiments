"""
Constrained Decoding Demonstration

This script demonstrates how constrained decoding eliminates compliance issues
in multiple-choice questions by forcing the model to generate only valid option tokens.

Usage:
    python examples/constrained_decoding_demo.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from decoding import constrained_mc_generation, compare_free_vs_constrained
from metrics import compute_option_normalized_entropy, compute_entropy_decomposition


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


def demo_basic_constrained_generation():
    """Demo 1: Basic constrained generation."""
    print("="*80)
    print("DEMO 1: Basic Constrained Generation")
    print("="*80)

    model, tokenizer = load_model()

    prompt = """Answer this science question. Give only the letter (A, B, C, or D).

Question: Which property of an object can be measured using a ruler?
A. mass
B. temperature
C. length
D. volume
Answer:"""

    print(f"Question prompt:\n{prompt}\n")
    print("Correct answer: C\n")
    print("-"*80)

    # Strategy 1: Force first token
    print("\n1️⃣  FORCE FIRST TOKEN (strictest)")
    print("   Only allows ['A','B','C','D'] as first generated token\n")

    result = constrained_mc_generation(
        model, tokenizer, prompt,
        valid_options=['A', 'B', 'C', 'D'],
        strategy='force_first_token',
        return_diagnostics=True,
        verbose=True
    )

    print(f"\n   Generated text: '{result.generated_text}'")
    print(f"   Selected option: {result.option_selected}")
    print(f"   Probability (before constraint): {result.probability_before:.1%}")
    print(f"   Probability (after constraint):  {result.probability_after:.1%}")
    print(f"   Was model's top choice: {result.was_most_likely}")

    # Strategy 2: Prefix allowed
    print("\n" + "-"*80)
    print("\n2️⃣  PREFIX ALLOWED (flexible)")
    print("   Allows reasoning, then constrains after trigger phrase\n")

    result2 = constrained_mc_generation(
        model, tokenizer, prompt,
        valid_options=['A', 'B', 'C', 'D'],
        strategy='prefix_allowed',
        max_new_tokens=20,
        return_diagnostics=True,
        verbose=True
    )

    print(f"\n   Generated text: '{result2.generated_text}'")
    print(f"   Selected option: {result2.option_selected}")


def demo_free_vs_constrained():
    """Demo 2: Compare free vs constrained generation."""
    print("\n" + "="*80)
    print("DEMO 2: Free vs Constrained Generation")
    print("="*80)
    print("This demo shows how constrained decoding fixes compliance issues.\n")

    model, tokenizer = load_model()

    # Test questions (some may have compliance issues)
    questions = [
        {
            'prompt': """Answer this science question. Give only the letter (A, B, C, or D).

Question: What gas do plants absorb from the air?
A. oxygen
B. nitrogen
C. carbon dioxide
D. helium
Answer:""",
            'options': ['A', 'B', 'C', 'D'],
            'correct': 'C'
        },
        {
            'prompt': """Answer this science question. Give only the letter (A, B, C, or D).

Question: Which of these is a renewable energy source?
A. coal
B. oil
C. solar
D. natural gas
Answer:""",
            'options': ['A', 'B', 'C', 'D'],
            'correct': 'C'
        }
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n{'-'*80}")
        print(f"Question {i}")
        print(f"Correct answer: {q['correct']}")
        print(f"{'-'*80}")

        comparison = compare_free_vs_constrained(
            model, tokenizer,
            q['prompt'],
            valid_options=q['options'],
            ground_truth=q['correct'],
            max_new_tokens=50
        )

        # Display free generation
        print(f"\n🆓 FREE GENERATION:")
        print(f"   Generated: '{comparison['free_generation']['text'][:100]}...'")
        print(f"   Extracted option: {comparison['free_generation']['extracted_option']}")
        print(f"   Valid format: {comparison['free_generation']['valid_format']}")
        if comparison['free_generation']['correct'] is not None:
            correct_symbol = "✓" if comparison['free_generation']['correct'] else "✗"
            print(f"   Correct: {correct_symbol}")

        # Display constrained generation
        print(f"\n🔒 CONSTRAINED GENERATION:")
        print(f"   Generated: '{comparison['constrained_generation']['text']}'")
        print(f"   Extracted option: {comparison['constrained_generation']['extracted_option']}")
        print(f"   Valid format: {comparison['constrained_generation']['valid_format']} (always true)")
        if comparison['constrained_generation']['correct'] is not None:
            correct_symbol = "✓" if comparison['constrained_generation']['correct'] else "✗"
            print(f"   Correct: {correct_symbol}")
        print(f"   Probability: {comparison['constrained_generation']['probability']:.1%}")
        print(f"   Was top choice: {comparison['constrained_generation']['was_most_likely']}")

        # Diagnosis
        if comparison['compliance_issue']:
            print(f"\n⚠️  COMPLIANCE ISSUE DETECTED!")
            print(f"   Free generation produced different format")
            print(f"   Constrained decoding fixed this ✓")
        else:
            print(f"\n✅ No compliance issue (both matched)")


def demo_entropy_plus_constrained():
    """Demo 3: Combine option-normalized entropy with constrained decoding."""
    print("\n" + "="*80)
    print("DEMO 3: Option-Normalized Entropy + Constrained Decoding")
    print("="*80)
    print("Combining both techniques for optimal MC question handling.\n")

    model, tokenizer = load_model()

    prompt = """Answer this science question. Give only the letter (A, B, C, or D).

Question: Which planet is closest to the Sun?
A. Venus
B. Mercury
C. Earth
D. Mars
Answer:"""

    print(f"Question: Which planet is closest to the Sun?")
    print(f"Correct answer: B (Mercury)\n")
    print("-"*80)

    # Step 1: Analyze intention with option-normalized entropy
    print("\n📊 STEP 1: Analyze Intention State")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]

    decomp = compute_entropy_decomposition(
        logits, tokenizer, options=['A', 'B', 'C', 'D']
    )

    print(f"\n   Standard entropy:        {decomp['standard']:.3f} bits")
    print(f"   Option-normalized:       {decomp['option_normalized']:.3f} bits")
    print(f"   Ratio:                   {decomp['ratio']:.3f}")
    print(f"   Probability mass:        {decomp['option_probability_mass']:.1%}")
    print(f"   Most likely (unconstrained): {decomp['most_likely_option']} ({decomp['confidence']:.1%})")

    print(f"\n   Option probabilities (before constraint):")
    for opt in ['A', 'B', 'C', 'D']:
        prob = decomp['option_probs'][opt]
        bar = "█" * int(prob * 40)
        marker = " ← Correct" if opt == 'B' else ""
        print(f"     {opt}: {prob:5.1%} {bar}{marker}")

    # Diagnosis
    print(f"\n   💡 DIAGNOSIS:")
    if decomp['option_normalized'] < 0.8:
        print(f"      Low entropy ({decomp['option_normalized']:.2f} < 0.8)")
        print(f"      → Model is CONFIDENT about answer")
        diagnosis = "confident"
    elif decomp['option_normalized'] > 1.5:
        print(f"      High entropy ({decomp['option_normalized']:.2f} > 1.5)")
        print(f"      → Model is UNCERTAIN about answer")
        diagnosis = "uncertain"
    else:
        print(f"      Medium entropy (0.8 ≤ {decomp['option_normalized']:.2f} ≤ 1.5)")
        print(f"      → Model has some uncertainty")
        diagnosis = "medium"

    if decomp['ratio'] > 0.7:
        print(f"      High ratio ({decomp['ratio']:.2f}) → Competence issue")
    elif decomp['ratio'] < 0.3:
        print(f"      Low ratio ({decomp['ratio']:.2f}) → Compliance issue likely")
    else:
        print(f"      Medium ratio ({decomp['ratio']:.2f}) → Mixed uncertainty")

    # Step 2: Generate with constraint
    print(f"\n🔒 STEP 2: Generate with Constrained Decoding")

    result = constrained_mc_generation(
        model, tokenizer, prompt,
        valid_options=['A', 'B', 'C', 'D'],
        strategy='force_first_token',
        return_diagnostics=True,
        verbose=False
    )

    print(f"\n   Selected option: {result.option_selected}")
    print(f"   Was top choice: {result.was_most_likely}")
    print(f"   Correct: {'✓' if result.option_selected == 'B' else '✗'}")

    # Step 3: Recommendation
    print(f"\n💡 RECOMMENDATION:")
    if diagnosis == "confident" and decomp['ratio'] < 0.5:
        print(f"   ✅ Model knows answer but might have format issues")
        print(f"   ✅ Use: Direct answer + Constrained decoding")
        print(f"   ✅ This saves tokens and guarantees valid format")
    elif diagnosis == "uncertain":
        print(f"   ⚠️  Model is uncertain about the answer")
        print(f"   ✅ Use: CoT reasoning first, then constrained decoding")
        print(f"   ✅ This helps model reason through options")
    else:
        print(f"   ➡️  Medium uncertainty")
        print(f"   ✅ Use: Constrained decoding to ensure valid output")


def demo_integration_with_router():
    """Demo 4: Integration with adaptive router."""
    print("\n" + "="*80)
    print("DEMO 4: Integration with Adaptive Router")
    print("="*80)
    print("Using constrained decoding with the adaptive inference router.\n")

    model, tokenizer = load_model()

    # Import router
    from router import AdaptiveInferenceRouter

    print("Creating router with constrained decoding support...\n")

    # Note: We'll need to add constrained decoding support to the router
    # For now, show manual integration

    question = "Which of these is a mammal?"
    choices = "A. snake\nB. dolphin\nC. frog\nD. eagle"
    correct = "B"

    prompt_baseline = f"""Answer this science question. Give only the letter (A, B, C, or D).

Question: {question}
{choices}
Answer:"""

    print(f"Question: {question}")
    print(f"Correct: {correct}\n")
    print("-"*80)

    # Option 1: Router decides, then apply constrained decoding
    print("\n📍 APPROACH 1: Route decision → Constrained generation")

    # Compute entropy for routing
    inputs = tokenizer(prompt_baseline, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]

    option_entropy = compute_option_normalized_entropy(
        logits, tokenizer, options=['A', 'B', 'C', 'D']
    )

    print(f"\n   Option entropy: {option_entropy:.3f} bits")

    # Routing decision
    if option_entropy < 0.8:
        route = "direct"
        max_tokens = 1
        print(f"   Route: DIRECT (entropy < 0.8)")
    else:
        route = "cot"
        max_tokens = 50
        print(f"   Route: COT (entropy ≥ 0.8)")

    # Generate with constraint
    if route == "direct":
        result = constrained_mc_generation(
            model, tokenizer, prompt_baseline,
            valid_options=['A', 'B', 'C', 'D'],
            strategy='force_first_token',
            return_diagnostics=True
        )
        print(f"   Generated: '{result.generated_text}'")
        print(f"   Answer: {result.option_selected}")
        print(f"   Correct: {'✓' if result.option_selected == correct else '✗'}")
        print(f"   Tokens used: {max_tokens}")
    else:
        # For CoT, allow reasoning then constrain
        prompt_cot = prompt_baseline.replace(
            "Give only the letter",
            "Think step by step, then give the letter"
        )
        result = constrained_mc_generation(
            model, tokenizer, prompt_cot,
            valid_options=['A', 'B', 'C', 'D'],
            strategy='prefix_allowed',
            max_new_tokens=max_tokens,
            return_diagnostics=True
        )
        print(f"   Generated: '{result.generated_text[:100]}...'")
        print(f"   Answer: {result.option_selected}")
        print(f"   Correct: {'✓' if result.option_selected == correct else '✗'}")

    print(f"\n💡 BENEFITS:")
    print(f"   ✅ Adaptive routing saves tokens when possible")
    print(f"   ✅ Constrained decoding guarantees valid format")
    print(f"   ✅ Combined: Optimal efficiency + reliability")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Constrained Decoding Demo")
    parser.add_argument(
        '--demo',
        type=int,
        choices=[1, 2, 3, 4],
        help="Which demo to run"
    )
    args = parser.parse_args()

    print("\n🔒 CONSTRAINED DECODING DEMONSTRATION")
    print("="*80)
    print("This demo shows how to force models to generate only valid MC options,")
    print("completely eliminating compliance (formatting) issues.")
    print("="*80)

    try:
        if args.demo == 1 or args.demo is None:
            demo_basic_constrained_generation()

        if args.demo == 2 or args.demo is None:
            demo_free_vs_constrained()

        if args.demo == 3 or args.demo is None:
            demo_entropy_plus_constrained()

        if args.demo == 4 or args.demo is None:
            demo_integration_with_router()

        print("\n" + "="*80)
        print("✅ Demo completed!")
        print("="*80)
        print("\n📚 Key Takeaways:")
        print("   1. Constrained decoding eliminates compliance issues")
        print("   2. Force first token: strictest, guarantees valid output")
        print("   3. Prefix allowed: flexible, allows reasoning first")
        print("   4. Combine with option-normalized entropy for best results")
        print("   5. Integration with router: optimal efficiency + reliability")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
