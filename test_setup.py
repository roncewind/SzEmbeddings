#!/usr/bin/env python3
"""
Quick test script to verify the environment is set up correctly.

Before running, ensure Senzing environment is loaded:
    source ~/senzingv4/setupEnv

Run with: python test_setup.py
"""
from __future__ import annotations

import sys


def test_imports() -> bool:
    """Test that all required packages can be imported."""
    print("Testing imports...")
    errors = []

    try:
        import numpy as np
        print(f"  ✓ numpy {np.__version__}")
    except ImportError as e:
        errors.append(f"  ✗ numpy: {e}")

    try:
        import orjson
        print(f"  ✓ orjson")
    except ImportError as e:
        errors.append(f"  ✗ orjson: {e}")

    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError as e:
        errors.append(f"  ✗ torch: {e}")

    try:
        import sentence_transformers
        print(f"  ✓ sentence-transformers {sentence_transformers.__version__}")
    except ImportError as e:
        errors.append(f"  ✗ sentence-transformers: {e}")

    try:
        import senzing
        print(f"  ✓ senzing")
    except ImportError as e:
        errors.append(f"  ✗ senzing: {e}")
        errors.append("    → Did you run: source ~/senzingv4/setupEnv ?")

    try:
        import senzing_core
        print(f"  ✓ senzing-core")
    except ImportError as e:
        errors.append(f"  ✗ senzing-core: {e}")
        errors.append("    → Did you run: source ~/senzingv4/setupEnv ?")

    if errors:
        print("\nImport errors:")
        for err in errors:
            print(err)
        return False

    print("  All imports successful!\n")
    return True


def test_cuda() -> None:
    """Test CUDA availability."""
    print("Testing CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
        else:
            print("  ℹ CUDA not available (will use CPU)")
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
    print()


def test_local_imports() -> bool:
    """Test that local modules can be imported."""
    print("Testing local imports...")
    errors = []

    try:
        from sz_utils import get_embedding, get_embeddings, format_seconds_to_hhmmss, get_senzing_config
        print("  ✓ sz_utils")
    except ImportError as e:
        errors.append(f"  ✗ sz_utils: {e}")

    if errors:
        print("\nLocal import errors:")
        for err in errors:
            print(err)
        return False

    print("  All local imports successful!\n")
    return True


def test_senzing_config() -> bool:
    """Test that Senzing config environment variable is set."""
    print("Testing Senzing configuration...")
    import os
    config = os.getenv("SENZING_ENGINE_CONFIGURATION_JSON")
    if config:
        print("  ✓ SENZING_ENGINE_CONFIGURATION_JSON is set")
        # Don't print the actual config as it may contain sensitive info
        print(f"    Config length: {len(config)} characters")
        return True
    else:
        print("  ✗ SENZING_ENGINE_CONFIGURATION_JSON is NOT set")
        print("    Set this environment variable before running the scripts")
        return False


def test_embedding_function() -> bool:
    """Test the embedding function with a minimal model (if available)."""
    print("\nTesting embedding function...")
    try:
        from sentence_transformers import SentenceTransformer
        from sz_utils import get_embedding

        # Use a tiny model for testing (downloads ~90MB on first run)
        print("  Loading test model (all-MiniLM-L6-v2)...")
        print("  (This may take a moment on first run)")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        test_name = "John Doe"
        embedding = get_embedding(test_name, model)

        print(f"  ✓ Embedding generated for '{test_name}'")
        print(f"    Shape: {embedding.shape}")
        print(f"    Dtype: {embedding.dtype}")
        print(f"    First 5 values: {embedding[:5]}")
        return True
    except Exception as e:
        print(f"  ✗ Error testing embedding: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("SzEmbeddings Environment Test")
    print("=" * 60 + "\n")

    all_passed = True

    all_passed &= test_imports()
    test_cuda()
    all_passed &= test_local_imports()
    config_ok = test_senzing_config()

    # Ask user if they want to test embedding (downloads model)
    print("\n" + "-" * 60)
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        all_passed &= test_embedding_function()
    else:
        print("Skipping embedding test (run with --full to include)")
        print("Note: --full will download a ~90MB test model on first run")

    print("\n" + "=" * 60)
    if all_passed and config_ok:
        print("✓ All tests passed! Ready to run.")
        sys.exit(0)
    elif all_passed and not config_ok:
        print("⚠ Imports OK but SENZING_ENGINE_CONFIGURATION_JSON not set")
        sys.exit(1)
    else:
        print("✗ Some tests failed. Check errors above.")
        sys.exit(1)
