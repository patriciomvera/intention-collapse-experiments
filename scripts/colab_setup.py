#!/usr/bin/env python3
"""
Google Colab Setup Script
Handles installation and import configuration for Colab environments.
"""

import sys
import os
import subprocess


def install_package():
    """Install the intention-collapse package."""
    print("🔧 Installing intention-collapse package...")

    # Remove old installation
    subprocess.run(
        ["rm", "-rf", "/content/intention-collapse-experiments"],
        capture_output=True
    )

    # Clone repository
    print("📥 Cloning repository...")
    result = subprocess.run(
        [
            "git", "clone", "-q", "-b", "v2-router-experiments",
            "https://github.com/patriciomvera/intention-collapse-experiments.git",
            "/content/intention-collapse-experiments"
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Git clone failed: {result.stderr}")
        return False

    # Install package
    print("📦 Installing package...")
    result = subprocess.run(
        ["pip", "install", "-e", "/content/intention-collapse-experiments/"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Installation failed: {result.stderr}")
        return False

    print("✅ Package installed successfully")
    return True


def configure_imports():
    """Configure Python path for imports."""
    repo_path = "/content/intention-collapse-experiments"

    # Ensure the package directory is in sys.path
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
        print(f"✅ Added {repo_path} to sys.path")

    return True


def verify_installation():
    """Verify that all imports work correctly."""
    print("\n🔍 Verifying installation...")

    try:
        from src.router import AdaptiveInferenceRouter, RouteDecision
        from src.metrics import compute_intention_entropy
        from src.controls import self_consistency_baseline
        from src.decoding import constrained_mc_generation

        print("✅ All critical imports successful!")
        print(f"✅ RouteDecision options: {list(RouteDecision)}")
        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def setup_colab():
    """Complete setup process for Google Colab."""
    print("=" * 70)
    print("Google Colab Setup - Intention Collapse")
    print("=" * 70)
    print()

    # Step 1: Install
    if not install_package():
        print("\n❌ Setup failed at installation step")
        return False

    # Step 2: Configure paths
    if not configure_imports():
        print("\n❌ Setup failed at path configuration step")
        return False

    # Step 3: Verify
    if not verify_installation():
        print("\n❌ Setup failed at verification step")
        return False

    print("\n" + "=" * 70)
    print("✅ SETUP COMPLETE - Ready to run experiments!")
    print("=" * 70)
    print("\nYou can now use:")
    print("  - notebooks/colab_router_experiment.ipynb")
    print("  - notebooks/scaled/01_run_experiments.ipynb")
    print()

    return True


if __name__ == "__main__":
    success = setup_colab()
    sys.exit(0 if success else 1)
