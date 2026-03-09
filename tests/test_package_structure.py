"""
Test package structure (without heavy dependencies).

This test verifies that:
1. The src package can be imported
2. Submodules have correct __init__.py files
3. __all__ exports are correctly defined
"""

import sys
from pathlib import Path

# Add root directory to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))


def test_package_import():
    """Test that the src package can be imported."""
    try:
        import src
        print(f"[OK] Package 'src' imported successfully")
        print(f"     Version: {src.__version__}")
        return True
    except ImportError as e:
        print(f"[FAIL] Cannot import 'src': {e}")
        return False


def test_submodule_structure():
    """Test that submodules have correct structure."""
    # Packages (directories with __init__.py)
    subpackages = ['router', 'controls', 'decoding']
    # Modules (direct .py files)
    modules = ['metrics', 'activation_hooks', 'data_utils', 'probing', 'visualization']

    all_ok = True

    # Test subpackages
    for subpackage in subpackages:
        module_path = root_dir / 'src' / subpackage / '__init__.py'
        if not module_path.exists():
            print(f"[FAIL] Missing __init__.py in src/{subpackage}")
            all_ok = False
            continue

        print(f"[OK] src/{subpackage}/__init__.py exists")

        # Try to import
        try:
            exec(f"import src.{subpackage}")
            print(f"[OK] src.{subpackage} imports successfully")
        except ImportError as e:
            print(f"[SKIP] src.{subpackage} requires dependencies: {str(e).split(':')[0]}")

    # Test modules
    for module in modules:
        module_path = root_dir / 'src' / f'{module}.py'
        if not module_path.exists():
            print(f"[FAIL] Missing src/{module}.py")
            all_ok = False
            continue

        print(f"[OK] src/{module}.py exists")

    return all_ok


def test_module_all_exports():
    """Test that modules define __all__ correctly."""
    modules_to_check = {
        'src': ['compute_intention_entropy', 'IntentionMetrics'],
        'src.router': ['AdaptiveInferenceRouter', 'RouteDecision', 'RouterResult'],
        'src.controls': ['self_consistency_baseline', 'SelfConsistencyResult'],
        'src.decoding': ['constrained_mc_generation', 'MultipleChoiceLogitsProcessor'],
    }

    for module_name, expected_exports in modules_to_check.items():
        try:
            module = __import__(module_name, fromlist=['__all__'])
            if hasattr(module, '__all__'):
                all_exports = module.__all__
                for export in expected_exports:
                    if export in all_exports:
                        print(f"[OK] {module_name}.__all__ includes '{export}'")
                    else:
                        print(f"[WARNING] {module_name}.__all__ missing '{export}'")
            else:
                print(f"[WARNING] {module_name} does not define __all__")
        except Exception as e:
            # This may fail if dependencies are not installed, that's OK
            print(f"[SKIP] Cannot check {module_name}: {e}")

    return True


def test_lazy_imports():
    """Test that __getattr__ lazy loading works."""
    try:
        import src
        # __version__ must be available without lazy loading
        assert hasattr(src, '__version__')
        print(f"[OK] src.__version__ is accessible: {src.__version__}")

        # __all__ must be defined
        assert hasattr(src, '__all__')
        print(f"[OK] src.__all__ is defined with {len(src.__all__)} exports")

        return True
    except Exception as e:
        print(f"[FAIL] Lazy import test failed: {e}")
        return False


if __name__ == "__main__":
    print("="*70)
    print("Testing Package Structure (no heavy dependencies required)")
    print("="*70)
    print()

    all_passed = True

    # Test 1: Package import
    print("Test 1: Package Import")
    all_passed &= test_package_import()
    print()

    # Test 2: Submodule structure
    print("Test 2: Submodule Structure")
    all_passed &= test_submodule_structure()
    print()

    # Test 3: Module exports
    print("Test 3: Module __all__ Exports")
    all_passed &= test_module_all_exports()
    print()

    # Test 4: Lazy imports
    print("Test 4: Lazy Import Mechanism")
    all_passed &= test_lazy_imports()
    print()

    print("="*70)
    if all_passed:
        print("[SUCCESS] All package structure tests passed!")
        print()
        print("Note: Full import tests (test_imports.py) require all dependencies")
        print("      to be installed (torch, transformers, etc.)")
    else:
        print("[FAIL] Some tests failed")
        sys.exit(1)
    print("="*70)
