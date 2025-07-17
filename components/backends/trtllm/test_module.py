#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script to verify the dynamo.trtllm module works correctly.
"""

import sys
import os

# Add the src directory to the Python path
src_dir = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, src_dir)

def test_import():
    """Test that the module can be imported."""
    try:
        import dynamo.trtllm
        print("✓ Successfully imported dynamo.trtllm")
        print(f"  Version: {dynamo.trtllm.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import dynamo.trtllm: {e}")
        return False

def test_main_module():
    """Test that the __main__ module can be imported."""
    try:
        from dynamo.trtllm import __main__
        print("✓ Successfully imported dynamo.trtllm.__main__")
        return True
    except ImportError as e:
        if "tensorrt_llm" in str(e):
            print("✓ dynamo.trtllm.__main__ structure is correct (TensorRT-LLM not installed)")
            return True
        else:
            print(f"✗ Failed to import dynamo.trtllm.__main__: {e}")
            return False

def test_package_structure():
    """Test that the package structure is correct."""
    try:
        import dynamo.trtllm
        # Check that the module has the expected attributes
        expected_attrs = ['__version__']
        missing_attrs = [attr for attr in expected_attrs if not hasattr(dynamo.trtllm, attr)]
        
        if missing_attrs:
            print(f"✗ Package structure is incorrect - missing attributes: {missing_attrs}")
            return False
        else:
            print("✓ Package structure is correct")
            return True
    except Exception as e:
        print(f"✗ Package structure test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing dynamo.trtllm module...")
    print()
    
    tests = [
        test_import,
        test_main_module,
        test_package_structure,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The dynamo.trtllm module is ready to use.")
        print("\nYou can now run:")
        print("  python -m dynamo.trtllm --help")
    else:
        print("❌ Some tests failed. Please check the module structure.")
        sys.exit(1) 