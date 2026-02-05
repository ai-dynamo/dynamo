#!/usr/bin/env python3
"""
Local testing script for error classification system.

This script tests the full classification pipeline with Claude API locally
before deploying to GitHub workflows.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python3 opensearch_upload/test_classification_locally.py
"""

import os
import sys
from typing import List, Dict, Any

# Add to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensearch_upload.error_classification import (
    Config,
    ErrorClassifier,
    ErrorContext,
    ErrorDeduplicator,
    GitHubAnnotator,
    AnnotationConfig,
    ERROR_CATEGORIES,
)


# Sample errors covering all 10 categories
SAMPLE_ERRORS = [
    {
        "name": "CUDA Out of Memory",
        "text": """RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 15.77 GiB of which 1.23 GiB is free. Including non-PyTorch memory, this process has 13.54 GiB memory in use. Of the allocated memory 12.45 GiB is allocated by PyTorch, and 234.56 MiB is reserved by PyTorch but unallocated.""",
        "expected_category": "resource_exhaustion",
        "source_type": "pytest",
        "framework": "vllm",
        "test_name": "test_large_batch_inference",
    },
    {
        "name": "Package Version Conflict",
        "text": """ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
torchvision 0.14.0 requires torch==1.13.0, but you have torch 2.0.0 which is incompatible.
ERROR: Could not find a version that satisfies the requirement torch>=2.0.0,<2.1.0
ERROR: No matching distribution found for torch>=2.0.0,<2.1.0""",
        "expected_category": "dependency_error",
        "source_type": "buildkit",
        "framework": "vllm",
    },
    {
        "name": "Test Timeout",
        "text": """FAILED tests/integration/test_model_serving.py::test_large_batch_processing - Timeout: Test exceeded 300 seconds
E   pytest_timeout.TimeoutError: test_large_batch_processing did not complete within 300 seconds
E   Last log entry: "Processing batch 45/100..."
E   The test appears to be stuck in an infinite loop or deadlock.""",
        "expected_category": "timeout",
        "source_type": "pytest",
        "framework": "sglang",
        "test_name": "test_large_batch_processing",
    },
    {
        "name": "Connection Refused",
        "text": """requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=8000): Max retries exceeded with url: /v1/completions (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f8a0b2c3d60>: Failed to establish a new connection: [Errno 111] Connection refused'))
During handling of the above exception, another exception occurred:
Failed to connect to model server after 10 retries""",
        "expected_category": "network_error",
        "source_type": "pytest",
        "framework": "trtllm",
        "test_name": "test_api_endpoint",
    },
    {
        "name": "Assertion Failure",
        "text": """tests/unit/test_tokenizer.py:45: AssertionError: Token count mismatch
E   assert 42 == 43
E   +42
E   -43
E   Expected 43 tokens but got 42 tokens
E   Input text: "Hello, world! This is a test."
E   Tokens: ['Hello', ',', ' world', '!', ...]""",
        "expected_category": "assertion_failure",
        "source_type": "pytest",
        "framework": "vllm",
        "test_name": "test_tokenizer_count",
    },
    {
        "name": "Rust Compilation Error",
        "text": """error[E0425]: cannot find value `tokenizer` in this scope
  --> src/processors/tokenization.rs:123:9
   |
123 |         tokenizer.encode(text)
   |         ^^^^^^^^^ not found in this scope
   |
help: consider importing this struct
   |
1   | use crate::Tokenizer;
   |

error: could not compile `dynamo-tokenizer` due to previous error""",
        "expected_category": "compilation_error",
        "source_type": "rust_test",
        "framework": "rust",
    },
    {
        "name": "Segmentation Fault",
        "text": """Fatal Python error: Segmentation fault

Current thread 0x00007f8b9c0a1234 (most recent call first):
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/functional.py", line 2345 in linear
  File "/workspace/dynamo/vllm/model.py", line 89 in forward
  File "/workspace/dynamo/vllm/engine.py", line 234 in generate
  File "tests/test_inference.py", line 56 in test_model_forward

Segmentation fault (core dumped)
/bin/bash: line 1: 12345 Segmentation fault""",
        "expected_category": "runtime_error",
        "source_type": "pytest",
        "framework": "vllm",
        "test_name": "test_model_forward",
    },
    {
        "name": "Docker Build Failure",
        "text": """ERROR [build 8/12] RUN apt-get update && apt-get install -y cuda-toolkit-11-8
#12 0.856 Err:1 http://archive.ubuntu.com/ubuntu jammy InRelease
#12 0.856   Temporary failure resolving 'archive.ubuntu.com'
#12 1.234 Reading package lists...
#12 1.456 E: Failed to fetch http://archive.ubuntu.com/ubuntu/dists/jammy/InRelease
failed to solve with frontend dockerfile.v0: failed to build LLB: executor failed running [/bin/sh -c apt-get update && apt-get install -y cuda-toolkit-11-8]: exit code: 100""",
        "expected_category": "infrastructure_error",
        "source_type": "buildkit",
        "framework": "vllm",
    },
    {
        "name": "Config File Not Found",
        "text": """FileNotFoundError: [Errno 2] No such file or directory: '/workspace/config/model_config.yaml'
  File "/workspace/dynamo/config_loader.py", line 12, in load_config
    with open('/workspace/config/model_config.yaml', 'r') as f:
  File "/workspace/dynamo/main.py", line 45, in main
    config = load_config()

The configuration file is required but was not found at the expected path.
Please ensure MODEL_CONFIG_PATH environment variable is set correctly.""",
        "expected_category": "configuration_error",
        "source_type": "pytest",
        "framework": "sglang",
    },
    {
        "name": "Flaky Test",
        "text": """tests/parallel/test_concurrent_requests.py::test_race_condition FAILED [100%]

=================================== FAILURES ===================================
__________________________ test_race_condition _______________________________

RuntimeError: Race condition detected: attempted to read from shared tensor while write operation was in progress from another thread
Thread 1: Writing to tensor at index 42
Thread 2: Reading from same tensor

Note: This test passes ~75% of the time when run individually but fails more frequently when run in parallel with other tests.
Flaky test detected by pytest-flakefinder after 10 runs (7 passed, 3 failed)""",
        "expected_category": "flaky_test",
        "source_type": "pytest",
        "framework": "trtllm",
        "test_name": "test_race_condition",
    },
]


def create_error_contexts() -> List[ErrorContext]:
    """Create ErrorContext objects from sample errors."""
    contexts = []

    for sample in SAMPLE_ERRORS:
        context = ErrorContext(
            error_text=sample["text"],
            source_type=sample["source_type"],
            framework=sample.get("framework"),
            test_name=sample.get("test_name"),
            workflow_id="test-workflow-123",
            job_id="test-job-456",
            workflow_name="Local Testing",
            job_name="test-classification",
            repo="ai-dynamo/dynamo",
            branch="test-branch",
            commit_sha="abc123def456",
            user_alias="test-user",
            metadata={"expected_category": sample["expected_category"]},
        )
        contexts.append(context)

    return contexts


def test_deduplication():
    """Test error deduplication with similar errors."""
    print("\n" + "=" * 80)
    print("TEST 1: ERROR DEDUPLICATION")
    print("=" * 80)

    deduplicator = ErrorDeduplicator()

    # Create two similar errors with different timestamps and paths
    error1 = "RuntimeError at 2025-01-15 10:30:45: CUDA out of memory (GPU 0) in /workspace/file.py:123"
    error2 = "RuntimeError at 2026-02-03 14:20:10: CUDA out of memory (GPU 0) in /home/user/file.py:456"

    # Normalize and hash
    normalized1 = deduplicator.normalize_error_text(error1)
    normalized2 = deduplicator.normalize_error_text(error2)
    hash1 = deduplicator.compute_error_hash(error1)
    hash2 = deduplicator.compute_error_hash(error2)

    print(f"\nOriginal Error 1:\n  {error1}")
    print(f"\nNormalized Error 1:\n  {normalized1}")
    print(f"\nHash 1: {hash1}")

    print(f"\nOriginal Error 2:\n  {error2}")
    print(f"\nNormalized Error 2:\n  {normalized2}")
    print(f"\nHash 2: {hash2}")

    if hash1 == hash2:
        print("\n✅ PASS: Same hash despite different timestamps/paths (deduplication works!)")
    else:
        print("\n❌ FAIL: Different hashes for similar errors")

    return hash1 == hash2


def test_classification(config: Config) -> bool:
    """Test error classification with Claude API."""
    print("\n" + "=" * 80)
    print("TEST 2: ERROR CLASSIFICATION WITH CLAUDE API")
    print("=" * 80)

    # Create classifier (without OpenSearch)
    classifier = ErrorClassifier(config, opensearch_client=None)

    # Create error contexts
    error_contexts = create_error_contexts()

    print(f"\nTesting classification of {len(error_contexts)} sample errors...")
    print(f"Categories: {', '.join(ERROR_CATEGORIES)}")

    results = []
    all_passed = True

    for i, error_context in enumerate(error_contexts, 1):
        sample = SAMPLE_ERRORS[i - 1]
        print(f"\n{'-' * 80}")
        print(f"[{i}/{len(error_contexts)}] {sample['name']}")
        print(f"Expected: {sample['expected_category']}")

        try:
            # Classify the error
            classification = classifier.classify_error(
                error_context,
                use_cache=True,
                classification_method="test"
            )

            # Check result
            correct = classification.primary_category == sample['expected_category']
            status = "✅ PASS" if correct else "❌ FAIL"

            print(f"\nResult:")
            print(f"  Category: {classification.primary_category}")
            print(f"  Confidence: {classification.confidence_score:.2%}")
            print(f"  Root Cause: {classification.root_cause_summary[:100]}...")
            print(f"  {status}")

            if classification.prompt_tokens:
                print(f"\nAPI Usage:")
                print(f"  Prompt tokens: {classification.prompt_tokens}")
                print(f"  Cached tokens: {classification.cached_tokens}")
                cache_rate = (classification.cached_tokens / classification.prompt_tokens * 100) if classification.prompt_tokens > 0 else 0
                print(f"  Cache hit rate: {cache_rate:.1f}%")

            results.append({
                "name": sample["name"],
                "expected": sample["expected_category"],
                "actual": classification.primary_category,
                "confidence": classification.confidence_score,
                "correct": correct,
            })

            if not correct:
                all_passed = False

        except Exception as e:
            print(f"\n❌ ERROR: Failed to classify: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    # Summary
    print(f"\n{'=' * 80}")
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)

    correct_count = sum(1 for r in results if r["correct"])
    total_count = len(results)
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0

    print(f"\nAccuracy: {correct_count}/{total_count} ({accuracy:.1f}%)")

    print("\nResults:")
    for r in results:
        status = "✅" if r["correct"] else "❌"
        print(f"  {status} {r['name']}")
        print(f"     Expected: {r['expected']}, Got: {r['actual']} ({r['confidence']:.1%} confidence)")

    if accuracy >= 80:
        print(f"\n✅ PASS: Classification accuracy is acceptable (≥80%)")
    else:
        print(f"\n❌ FAIL: Classification accuracy is too low (<80%)")

    return all_passed


def test_github_annotator():
    """Test GitHub annotator (without actually creating annotations)."""
    print("\n" + "=" * 80)
    print("TEST 3: GITHUB ANNOTATIONS (DRY RUN)")
    print("=" * 80)

    annotator = GitHubAnnotator(AnnotationConfig(enabled=False))

    print(f"\nGitHub Context:")
    print(f"  Available: {annotator.is_available()}")
    print(f"  Token: {'set' if annotator.github_token else 'not set'}")
    print(f"  Repo: {annotator.repo or 'not set'}")
    print(f"  SHA: {annotator.sha or 'not set'}")

    # Create a sample classification to test annotation formatting
    from opensearch_upload.error_classification.classifier import ErrorClassification
    sample_classification = ErrorClassification(
        error_id="test-123",
        error_hash="abc123",
        primary_category="dependency_error",
        confidence_score=0.87,
        root_cause_summary="Package version conflict between torch 2.0.0 and torchvision 0.14.0",
        error_source="pytest",
        framework="vllm",
        test_name="test_model_inference",
    )

    # Test annotation formatting
    message = annotator.format_annotation_message(sample_classification)
    print(f"\nSample Annotation Message:")
    print("-" * 80)
    print(message)
    print("-" * 80)

    # Test annotation creation (won't actually create)
    annotation = annotator.create_annotation(sample_classification)
    if annotation:
        print(f"\nAnnotation Structure:")
        print(f"  Path: {annotation['path']}")
        print(f"  Level: {annotation['annotation_level']}")
        print(f"  Title: {annotation['title']}")
        print(f"\n✅ PASS: Annotation formatting works correctly")
    else:
        print(f"\n⚠️  SKIP: Annotation creation skipped (not in GitHub Actions)")

    return True


def main():
    """Run all tests."""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test error classification system locally")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt for API calls")
    args = parser.parse_args()

    print("=" * 80)
    print("ERROR CLASSIFICATION SYSTEM - LOCAL TESTING")
    print("=" * 80)

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n❌ ERROR: ANTHROPIC_API_KEY environment variable is required")
        print("\nPlease set your Claude API key:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        print("\nGet your API key from: https://console.anthropic.com/")
        sys.exit(1)

    print(f"\n✅ ANTHROPIC_API_KEY is set")
    print(f"   Model: claude-sonnet-4-5-20250929")

    # Load config
    try:
        config = Config(
            anthropic_api_key=api_key,
            anthropic_model="claude-sonnet-4-5-20250929",
            max_error_length=10000,
        )
        print(f"✅ Configuration loaded successfully")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load configuration: {e}")
        sys.exit(1)

    # Run tests
    all_passed = True

    try:
        # Test 1: Deduplication
        if not test_deduplication():
            all_passed = False

        # Test 2: Classification (calls Claude API)
        print(f"\n⚠️  WARNING: The next test will make API calls to Claude")
        print(f"   Estimated cost: ~$0.03 for 10 classifications")
        print(f"   (With caching: ~$0.003 after first request)")
        if not args.yes:
            input("\nPress Enter to continue or Ctrl+C to cancel...")
        else:
            print("   (Auto-continuing with --yes flag)")

        if not test_classification(config):
            all_passed = False

        # Test 3: GitHub Annotations
        if not test_github_annotator():
            all_passed = False

    except KeyboardInterrupt:
        print("\n\n⚠️  Testing cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
        print("\nThe error classification system is working correctly.")
        print("You can now deploy it to GitHub workflows.")
        print("\nNext steps:")
        print("1. Add ANTHROPIC_API_KEY to GitHub Secrets")
        print("2. Add error classification step to workflow YAML")
        print("3. Test with a real workflow failure")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nPlease fix the issues before deploying to GitHub workflows.")
        print("Check the error messages above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
