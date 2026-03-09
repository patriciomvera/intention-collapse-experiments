#!/usr/bin/env python
"""
Script de verificación para Google Colab.

Este script verifica que la instalación del paquete intention-collapse
funcione correctamente en Google Colab.

Uso en Colab:
    !python /content/intention-collapse-experiments/verify_colab_setup.py
"""

import sys


def check_python_version():
    """Verificar versión de Python."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        print(f"[FAIL] Python {major}.{minor} detected. Requires Python 3.10+")
        return False
    print(f"[OK] Python {major}.{minor} (requirement: 3.10+)")
    return True


def check_package_import():
    """Verificar que el paquete src se puede importar."""
    try:
        import src
        print(f"[OK] Package 'src' v{src.__version__} imported")
        return True
    except ImportError as e:
        print(f"[FAIL] Cannot import 'src': {e}")
        print("       Run: pip install -e /content/intention-collapse-experiments/")
        return False


def check_dependencies():
    """Verificar dependencias principales."""
    dependencies = {
        'torch': 'torch',
        'transformers': 'transformers',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
    }

    all_ok = True
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"[OK] {package} installed")
        except ImportError:
            print(f"[FAIL] {package} not installed")
            all_ok = False

    return all_ok


def check_critical_imports():
    """Verificar imports críticos del paquete."""
    imports_to_test = [
        ("src.router", ["AdaptiveInferenceRouter", "RouteDecision", "RouterResult"]),
        ("src.metrics", ["compute_intention_entropy", "IntentionMetrics"]),
        ("src.controls", ["self_consistency_baseline", "SelfConsistencyResult"]),
        ("src.decoding", ["constrained_mc_generation", "MultipleChoiceLogitsProcessor"]),
    ]

    all_ok = True
    for module_name, items in imports_to_test:
        try:
            module = __import__(module_name, fromlist=items)
            for item in items:
                if not hasattr(module, item):
                    print(f"[FAIL] {module_name}.{item} not found")
                    all_ok = False
                    continue
            print(f"[OK] {module_name} imports successful")
        except Exception as e:
            print(f"[FAIL] {module_name} import failed: {e}")
            all_ok = False

    return all_ok


def check_basic_functionality():
    """Test funcionalidad básica."""
    try:
        from src.router import RouteDecision

        # Test enum values
        assert RouteDecision.DIRECT == "direct"
        assert RouteDecision.COT == "cot"

        print("[OK] Basic functionality works (RouteDecision enum)")
        return True
    except Exception as e:
        print(f"[FAIL] Basic functionality test failed: {e}")
        return False


def main():
    """Ejecutar todos los tests."""
    print("=" * 70)
    print("Google Colab Setup Verification")
    print("=" * 70)
    print()

    all_passed = True

    # Test 1: Python version
    print("[Test 1] Python Version")
    all_passed &= check_python_version()
    print()

    # Test 2: Package import
    print("[Test 2] Package Import")
    package_ok = check_package_import()
    all_passed &= package_ok
    print()

    if not package_ok:
        print("=" * 70)
        print("[CRITICAL] Package not installed. Cannot proceed with other tests.")
        print()
        print("Run this command first:")
        print("  !pip install -e /content/intention-collapse-experiments/")
        print("=" * 70)
        sys.exit(1)

    # Test 3: Dependencies
    print("[Test 3] Dependencies")
    all_passed &= check_dependencies()
    print()

    # Test 4: Critical imports
    print("[Test 4] Critical Imports")
    all_passed &= check_critical_imports()
    print()

    # Test 5: Basic functionality
    print("[Test 5] Basic Functionality")
    all_passed &= check_basic_functionality()
    print()

    # Summary
    print("=" * 70)
    if all_passed:
        print("[SUCCESS] All tests passed!")
        print()
        print("You are ready to run experiments. Try:")
        print("  - notebooks/colab_router_experiment.ipynb")
        print("  - notebooks/scaled/01_run_experiments.ipynb")
    else:
        print("[FAIL] Some tests failed. See above for details.")
        print()
        print("Common fixes:")
        print("  1. Install package: pip install -e /content/intention-collapse-experiments/")
        print("  2. Restart runtime: Runtime > Restart runtime")
        print("  3. Check Python version: python --version")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
