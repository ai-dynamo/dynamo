#!/usr/bin/env python3
"""
Simple test script to validate Claude API connection.
Tests with a hardcoded fake error to verify the classification system works.
"""

import os
import sys
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from opensearch_upload.error_classification import Config
from opensearch_upload.error_classification.claude_client import ClaudeClient


# Fake error logs to test with
# NOTE: Testing with just 1 error to avoid NVIDIA API rate limits
FAKE_ERRORS = {
    "infrastructure_error": """
ERROR collecting tests/test_server.py
ImportError while importing test module 'tests/test_server.py'.
Hint: make sure your test suite is properly configured and can be imported.
Traceback (most recent call last):
  File "/usr/lib/python3.11/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1206, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1178, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1149, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
  File "/home/runner/.local/lib/python3.11/site-packages/_pytest/assertion/rewrite.py", line 178, in exec_module
    exec(co, module.__dict__)
  File "tests/test_server.py", line 5, in <module>
    from vllm.engine import LLMEngine
ModuleNotFoundError: No module named 'vllm'
""",
    "timeout": """
=========================== FAILURES ===========================
________________________ test_batch_inference ________________________
test_batch_inference timed out after 300.00 seconds

The test exceeded the maximum time limit. This could indicate:
- A deadlock in the code
- An infinite loop
- A very slow operation
- Network timeout waiting for response

Consider increasing the timeout or investigating the root cause.
""",
    "assertion_failure": """
=========================== FAILURES ===========================
________________________ test_accuracy ________________________

tests/test_accuracy.py:45: AssertionError
    def test_accuracy():
        result = model.predict(test_data)
        expected_accuracy = 0.95
        actual_accuracy = 0.89
>       assert actual_accuracy >= expected_accuracy, f"Accuracy {actual_accuracy} below threshold {expected_accuracy}"
E       AssertionError: Accuracy 0.89 below threshold 0.95

tests/test_accuracy.py:45: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_accuracy.py::test_accuracy - AssertionError: Accuracy 0.89 below threshold 0.95
"""
}


def test_claude_connection():
    """Test Claude API connection with a fake error."""

    print("=" * 70)
    print("TESTING CLAUDE API CONNECTION")
    print("=" * 70)
    print()

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ERROR: ANTHROPIC_API_KEY not set")
        print("   Please set the secret in GitHub repository settings")
        return False

    print("✅ ANTHROPIC_API_KEY found")
    print(f"   Key length: {len(api_key)} chars")
    print(f"   Key prefix: {api_key[:15]}...")
    print()

    # Initialize configuration
    print("📝 Initializing configuration...")
    try:
        config = Config.from_env()
        print(f"   Model: {config.anthropic_model}")
        print(f"   Max RPM: {config.max_rpm}")
        print()
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False

    # Initialize Claude client
    print("🔌 Connecting to Claude API...")
    try:
        client = ClaudeClient(config)
        print("   ✅ ClaudeClient initialized")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize Claude client: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test classification with each fake error
    # NOTE: Only test first error to avoid rate limits
    results = []
    test_count = 0
    max_tests = 1  # Limit to 1 test to avoid NVIDIA API rate limits

    for error_type, error_text in FAKE_ERRORS.items():
        if test_count >= max_tests:
            print(f"⏭️  Skipping remaining tests to avoid rate limits")
            break
        test_count += 1
        print("-" * 70)
        print(f"🧪 Testing with fake {error_type}...")
        print("-" * 70)
        print()

        # Show error snippet
        print("Error text (first 200 chars):")
        print(error_text[:200] + "...")
        print()

        # Classify the error
        print("🤖 Calling Claude API...")
        try:
            error_context = {
                "source_type": "test",
                "job_name": "test-job",
                "step_name": "test-step",
                "framework": "pytest"
            }

            result = client.classify_error(
                error_text=error_text,
                error_context=error_context,
                use_cache=True
            )

            print("✅ Classification successful!")
            print()
            print(f"   📊 Results:")
            print(f"      Category: {result.primary_category}")
            print(f"      Confidence: {result.confidence_score:.1%}")
            print(f"      Summary: {result.root_cause_summary}")
            print()
            print(f"   💰 Token Usage:")
            print(f"      Prompt tokens: {result.prompt_tokens:,}")
            print(f"      Completion tokens: {result.completion_tokens:,}")
            print(f"      Cached tokens: {result.cached_tokens:,}")
            print()

            # Check if classification matches expected
            if result.primary_category == error_type:
                print(f"   ✅ Classification CORRECT (expected {error_type})")
            else:
                print(f"   ⚠️  Classification mismatch: got {result.primary_category}, expected {error_type}")

            results.append({
                "expected": error_type,
                "actual": result.primary_category,
                "confidence": result.confidence_score,
                "match": result.primary_category == error_type
            })

            # Add delay to avoid rate limiting (NVIDIA API has strict limits)
            time.sleep(2)

        except Exception as e:
            print(f"❌ Classification failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "expected": error_type,
                "actual": "ERROR",
                "confidence": 0.0,
                "match": False
            })

        print()

    # Summary
    print("=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print()

    total = len(results)
    correct = sum(1 for r in results if r["match"])
    avg_confidence = sum(r["confidence"] for r in results) / total if total > 0 else 0

    print(f"Total tests: {total}")
    print(f"Correct classifications: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"Average confidence: {avg_confidence:.1%}")
    print()

    print("Details:")
    for r in results:
        status = "✅" if r["match"] else "❌"
        print(f"  {status} Expected: {r['expected']}, Got: {r['actual']}, Confidence: {r['confidence']:.0%}")
    print()

    # Final result
    print("=" * 70)
    if correct == total:
        print("✅ ALL TESTS PASSED - Claude API connection working perfectly!")
    elif correct > 0:
        print(f"⚠️  PARTIAL SUCCESS - {correct}/{total} tests passed")
    else:
        print("❌ ALL TESTS FAILED - Check API key and configuration")
    print("=" * 70)

    return correct == total


if __name__ == "__main__":
    success = test_claude_connection()
    sys.exit(0 if success else 1)
