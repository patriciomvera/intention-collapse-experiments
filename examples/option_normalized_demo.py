"""
Demonstration of Option-Normalized Entropy for Multiple-Choice Questions

This script demonstrates the difference between standard entropy H_int(I)
and option-normalized entropy H_opt(I) for multiple-choice questions.

Key Insight:
    Standard entropy conflates two issues:
    1. COMPETENCE: Does the model know the correct answer?
    2. COMPLIANCE: Will the model format the answer correctly?

    Option-normalized entropy isolates competence by measuring uncertainty
    only over the valid answer space [A, B, C, D].

Usage:
    python examples/option_normalized_demo.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from metrics import (
    compute_intention_entropy,
    compute_option_normalized_entropy,
    compute_entropy_decomposition,
    get_option_token_ids
)


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


def demo_entropy_comparison():
    """Demo 1: Compare standard vs option-normalized entropy."""
    print("="*80)
    print("DEMO 1: Standard vs Option-Normalized Entropy")
    print("="*80)

    model, tokenizer = load_model()

    # Multiple choice question
    question = """Answer this science question. Give only the letter (A, B, C, or D) after ####.

Question: Which property of an object can be measured using a ruler?
A. mass
B. temperature
C. length
D. volume
Answer:"""

    print(f"Question:\n{question}\n")
    print("Correct answer: C (length)\n")
    print("-"*80)

    # Tokenize and get logits
    inputs = tokenizer(question, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token logits

    # Compute standard entropy
    standard_entropy = compute_intention_entropy(logits, top_k=100)

    # Compute option-normalized entropy with probabilities
    option_entropy, option_probs = compute_option_normalized_entropy(
        logits, tokenizer, options=['A', 'B', 'C', 'D'], return_probs=True
    )

    # Get full decomposition
    decomposition = compute_entropy_decomposition(
        logits, tokenizer, options=['A', 'B', 'C', 'D']
    )

    # Display results
    print(f"📊 ENTROPY METRICS:")
    print(f"   Standard H_int(I):          {standard_entropy:.3f} bits")
    print(f"   Option-normalized H_opt(I): {option_entropy:.3f} bits")
    print(f"   Ratio (H_opt / H_std):      {decomposition['ratio']:.3f}")
    print(f"\n   Option probability mass:    {decomposition['option_probability_mass']:.1%}")
    print(f"   Most likely option:         {decomposition['most_likely_option']}")
    print(f"   Confidence:                 {decomposition['confidence']:.1%}")

    print(f"\n📈 OPTION PROBABILITIES:")
    for option, prob in sorted(option_probs.items()):
        marker = " ← Correct answer!" if option == 'C' else ""
        bar = "█" * int(prob * 50)
        print(f"   {option}: {prob:6.1%} {bar}{marker}")

    print(f"\n💡 INTERPRETATION:")
    if decomposition['ratio'] > 0.8:
        print("   High ratio → Uncertainty is mainly about which option to choose")
        print("   This indicates a COMPETENCE issue (model unsure of answer)")
    elif decomposition['ratio'] < 0.3:
        print("   Low ratio → Uncertainty is mainly outside valid options")
        print("   This indicates a COMPLIANCE issue (formatting uncertainty)")
    else:
        print("   Medium ratio → Mixed uncertainty (both competence and compliance)")

    if decomposition['option_probability_mass'] > 0.7:
        print(f"\n   High probability mass on options ({decomposition['option_probability_mass']:.1%})")
        print("   → Model is focused on valid answer choices ✓")
    else:
        print(f"\n   Low probability mass on options ({decomposition['option_probability_mass']:.1%})")
        print("   → Model may generate invalid format ⚠️")

    # Show theoretical maximum
    max_entropy_4_options = 2.0  # log2(4)
    print(f"\n   Theoretical max entropy (4 options): {max_entropy_4_options:.2f} bits")
    print(f"   Current option entropy:              {option_entropy:.2f} bits")
    print(f"   Uncertainty level:                   {option_entropy/max_entropy_4_options:.1%}")


def demo_easy_vs_hard_questions():
    """Demo 2: Compare entropy for easy vs hard questions."""
    print("\n" + "="*80)
    print("DEMO 2: Easy vs Hard Multiple-Choice Questions")
    print("="*80)

    model, tokenizer = load_model()

    questions = [
        {
            'prompt': """Answer this science question. Give only the letter (A, B, C, or D) after ####.

Question: What is the primary gas in Earth's atmosphere?
A. nitrogen
B. oxygen
C. carbon dioxide
D. helium
Answer:""",
            'correct': 'A',
            'difficulty': 'Easy (elementary knowledge)'
        },
        {
            'prompt': """Answer this science question. Give only the letter (A, B, C, or D) after ####.

Question: Which process is primarily responsible for the formation of fossil fuels?
A. photosynthesis
B. combustion
C. decomposition
D. evaporation
Answer:""",
            'correct': 'C',
            'difficulty': 'Hard (requires reasoning)'
        }
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n{'-'*80}")
        print(f"Question {i}: {q['difficulty']}")
        print(f"Correct answer: {q['correct']}")
        print(f"{'-'*80}")

        # Get logits
        inputs = tokenizer(q['prompt'], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]

        # Compute decomposition
        decomp = compute_entropy_decomposition(
            logits, tokenizer, options=['A', 'B', 'C', 'D']
        )

        print(f"\n  Option-normalized entropy: {decomp['option_normalized']:.3f} bits")
        print(f"  Most likely option:        {decomp['most_likely_option']} ({decomp['confidence']:.1%})")
        print(f"  Correct:                   {'✓' if decomp['most_likely_option'] == q['correct'] else '✗'}")

        print(f"\n  Option probabilities:")
        for option in ['A', 'B', 'C', 'D']:
            prob = decomp['option_probs'][option]
            marker = " ← Model's choice" if option == decomp['most_likely_option'] else ""
            marker += " ← Correct" if option == q['correct'] else ""
            bar = "█" * int(prob * 40)
            print(f"    {option}: {prob:5.1%} {bar}{marker}")


def demo_routing_decision():
    """Demo 3: How option-normalized entropy affects routing."""
    print("\n" + "="*80)
    print("DEMO 3: Option-Normalized Entropy for Routing Decisions")
    print("="*80)

    model, tokenizer = load_model()

    # Import router
    from router import AdaptiveInferenceRouter

    # Create two routers: one with standard entropy, one with option-normalized
    print("\nCreating two routers:")
    print("  1. Router with STANDARD entropy (baseline)")
    print("  2. Router with OPTION-NORMALIZED entropy (recommended for MC)")

    router_standard = AdaptiveInferenceRouter(
        model=model,
        tokenizer=tokenizer,
        benchmark='arc',
        use_option_normalized_entropy=False,  # Use standard entropy
        entropy_threshold_low=0.5,
        entropy_threshold_high=1.2,
        verbose=False
    )

    router_option_norm = AdaptiveInferenceRouter(
        model=model,
        tokenizer=tokenizer,
        benchmark='arc',
        use_option_normalized_entropy=True,   # Use option-normalized
        entropy_threshold_low=0.8,            # Adjusted for MC
        entropy_threshold_high=1.5,           # Max is 2.0 for 4 options
        verbose=False
    )

    # Test question
    question = """Which property of an object can be measured using a ruler?"""
    choices = """A. mass
B. temperature
C. length
D. volume"""

    print(f"\n{'-'*80}")
    print(f"Test Question: {question}")
    print(f"Choices:\n{choices}")
    print(f"Correct: C")
    print(f"{'-'*80}")

    # Test with standard entropy router
    print("\n1️⃣  STANDARD ENTROPY ROUTER:")
    result_std = router_standard.generate(
        question=question,
        choices=choices,
        ground_truth="C"
    )
    print(f"   Entropy: {result_std.intention_entropy:.3f} bits")
    print(f"   Route:   {result_std.route_taken.value}")
    print(f"   Tokens:  {result_std.total_tokens}")

    # Test with option-normalized entropy router
    print("\n2️⃣  OPTION-NORMALIZED ENTROPY ROUTER:")
    result_opt = router_option_norm.generate(
        question=question,
        choices=choices,
        ground_truth="C"
    )
    print(f"   Entropy: {result_opt.intention_entropy:.3f} bits")
    print(f"   Route:   {result_opt.route_taken.value}")
    print(f"   Tokens:  {result_opt.total_tokens}")

    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON:")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'Standard':<20} {'Option-Normalized':<20}")
    print(f"{'-'*80}")
    print(f"{'Entropy':<30} {result_std.intention_entropy:<20.3f} {result_opt.intention_entropy:<20.3f}")
    print(f"{'Route Decision':<30} {result_std.route_taken.value:<20} {result_opt.route_taken.value:<20}")
    print(f"{'Total Tokens':<30} {result_std.total_tokens:<20} {result_opt.total_tokens:<20}")

    savings = result_std.total_tokens - result_opt.total_tokens
    if savings > 0:
        print(f"\n💰 Token savings with option-normalized: {savings} tokens ({100*savings/result_std.total_tokens:.1f}%)")
    elif savings < 0:
        print(f"\n⚠️  Option-normalized used {-savings} more tokens")
    else:
        print(f"\n➡️  Both routers made the same decision")


def demo_token_ids():
    """Demo 4: Show how option tokens are encoded."""
    print("\n" + "="*80)
    print("DEMO 4: Option Token IDs in Different Tokenizers")
    print("="*80)

    model, tokenizer = load_model()

    print(f"\nTokenizer: {tokenizer.__class__.__name__}")
    print(f"Vocabulary size: {len(tokenizer)}\n")

    # Get option token IDs
    option_ids = get_option_token_ids(tokenizer, options=['A', 'B', 'C', 'D', 'E'])

    print("Option token IDs:")
    print(f"{'Option':<10} {'Token ID':<15} {'Decoded':<20} {'Alternatives':<30}")
    print("-"*80)

    for option in ['A', 'B', 'C', 'D', 'E']:
        token_id = option_ids[option]
        decoded = tokenizer.decode([token_id])

        # Show alternative encodings
        alternatives = []
        for candidate in [option, f" {option}", f"{option}.", f" {option}."]:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if ids and ids[0] == token_id:
                alternatives.append(f'"{candidate}"')

        print(f"{option:<10} {token_id:<15} {repr(decoded):<20} {', '.join(alternatives):<30}")

    print("\n💡 Note: The function automatically handles different tokenization schemes")
    print("   and selects the most appropriate token ID for each option.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Option-Normalized Entropy Demo")
    parser.add_argument(
        '--demo',
        type=int,
        choices=[1, 2, 3, 4],
        help="Which demo to run (1=comparison, 2=easy vs hard, 3=routing, 4=token IDs)"
    )
    args = parser.parse_args()

    print("\n🎯 OPTION-NORMALIZED ENTROPY DEMONSTRATION")
    print("="*80)
    print("This demo shows how option-normalized entropy improves routing decisions")
    print("for multiple-choice questions by separating competence from compliance.")
    print("="*80)

    try:
        if args.demo == 1 or args.demo is None:
            demo_entropy_comparison()

        if args.demo == 2 or args.demo is None:
            demo_easy_vs_hard_questions()

        if args.demo == 3 or args.demo is None:
            demo_routing_decision()

        if args.demo == 4 or args.demo is None:
            demo_token_ids()

        print("\n" + "="*80)
        print("✅ Demo completed!")
        print("="*80)
        print("\n📚 Key Takeaways:")
        print("   1. Standard entropy conflates competence and compliance issues")
        print("   2. Option-normalized entropy isolates uncertainty about the answer")
        print("   3. For MC questions, use option-normalized with adjusted thresholds")
        print("   4. Typical threshold adjustment: low=0.8, high=1.5 (vs 0.5, 1.2)")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
