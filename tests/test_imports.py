"""
Test that all critical imports work in Colab.

This test validates that after `pip install -e .`, all main modules
can be imported without errors.
"""

import sys
from pathlib import Path

# Add root directory to path (for local testing without installation)
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))


def test_core_imports():
    """Test main package imports."""
    print("Testing core package imports...")

    # Test 1: Router module
    try:
        from src.router import AdaptiveInferenceRouter, RouteDecision, RouterResult
        print("[OK] src.router imports successful")
    except ImportError as e:
        print(f"[FAIL] src.router import failed: {e}")
        raise

    # Test 2: Metrics module
    try:
        from src.metrics import compute_intention_entropy, IntentionMetrics
        print("[OK] src.metrics imports successful")
    except ImportError as e:
        print(f"[FAIL] src.metrics import failed: {e}")
        raise

    # Test 3: Controls module
    try:
        from src.controls import self_consistency_baseline, SelfConsistencyResult
        print("[OK] src.controls imports successful")
    except ImportError as e:
        print(f"[FAIL] src.controls import failed: {e}")
        raise

    # Test 4: Decoding module
    try:
        from src.decoding import constrained_mc_generation, MultipleChoiceLogitsProcessor
        print("[OK] src.decoding imports successful")
    except ImportError as e:
        print(f"[FAIL] src.decoding import failed: {e}")
        raise

    print("\n" + "="*50)
    print("[SUCCESS] ALL IMPORTS SUCCESSFUL!")
    print("="*50)
    print("\nReady to run experiments in Google Colab.")


def test_basic_functionality():
    """Test that basic classes can be instantiated."""
    print("\nTesting basic functionality...")

    from src.router import RouteDecision
    assert RouteDecision.DIRECT == "direct"
    assert RouteDecision.COT == "cot"
    print("[OK] RouteDecision enum works")

    print("[OK] Basic functionality checks passed")


if __name__ == "__main__":
    try:
        test_core_imports()
        test_basic_functionality()
    except Exception as e:
        print(f"\n[FAIL] Tests failed: {e}")
        sys.exit(1)
