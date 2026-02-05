#!/usr/bin/env python3
"""
Demonstration of error classification system with real-world error examples.
Shows how the system would classify actual errors (without calling Claude API).
"""

import sys
from error_classification import ErrorDeduplicator, ErrorExtractor, ERROR_CATEGORIES
from error_classification.prompts import get_category_definitions

# Sample real-world errors that would be found in test results and build logs
SAMPLE_ERRORS = [
    {
        "name": "CUDA Out of Memory",
        "text": """RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 15.77 GiB of which 1.23 GiB is free. Including non-PyTorch memory, this process has 13.54 GiB memory in use.""",
        "expected_category": "resource_exhaustion"
    },
    {
        "name": "Package Version Conflict",
        "text": """ERROR: Could not find a version that satisfies the requirement torch>=2.0.0 (from versions: 1.13.0, 1.13.1)
ERROR: No matching distribution found for torch>=2.0.0""",
        "expected_category": "dependency_error"
    },
    {
        "name": "Test Timeout",
        "text": """FAILED tests/test_model.py::test_large_batch - Timeout: Test exceeded 300 seconds
E   pytest_timeout.TimeoutError: test_large_batch did not complete within 300 seconds""",
        "expected_category": "timeout"
    },
    {
        "name": "Connection Refused",
        "text": """requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=8000): Max retries exceeded with url: /v1/completions (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f8a>: Failed to establish a new connection: [Errno 111] Connection refused'))""",
        "expected_category": "network_error"
    },
    {
        "name": "Assertion Failure",
        "text": """tests/test_api.py:45: AssertionError
E   assert 'hello' == 'world'
E   - hello
E   + world""",
        "expected_category": "assertion_failure"
    },
    {
        "name": "Compilation Error",
        "text": """error[E0425]: cannot find value `foo` in this scope
  --> src/tokenizer.rs:123:5
   |
123 |     foo.bar();
   |     ^^^ not found in this scope
error: could not compile `dynamo` due to previous error""",
        "expected_category": "compilation_error"
    },
    {
        "name": "Segmentation Fault",
        "text": """Fatal Python error: Segmentation fault

Current thread 0x00007f8b9c0a1234 (most recent call first):
  File "/opt/conda/lib/python3.10/site-packages/torch/nn/functional.py", line 2345 in linear
  File "/workspace/model.py", line 89 in forward
Segmentation fault (core dumped)""",
        "expected_category": "runtime_error"
    },
    {
        "name": "Docker Build Failure",
        "text": """ERROR [stage 2/3] RUN apt-get update && apt-get install -y cuda-toolkit-11-8
failed to solve with frontend dockerfile.v0: failed to build LLB: executor failed running [/bin/sh -c apt-get update && apt-get install -y cuda-toolkit-11-8]: exit code: 100""",
        "expected_category": "infrastructure_error"
    },
    {
        "name": "Config File Not Found",
        "text": """FileNotFoundError: [Errno 2] No such file or directory: '/workspace/config.yaml'
  File "/workspace/main.py", line 12, in load_config
    with open('/workspace/config.yaml', 'r') as f:""",
        "expected_category": "configuration_error"
    },
    {
        "name": "Flaky Test",
        "text": """tests/test_parallel.py::test_concurrent_requests FAILED
RuntimeError: Race condition detected: attempted to read from tensor while write was in progress
Note: This test passes ~80% of the time""",
        "expected_category": "flaky_test"
    },
]

def main():
    print("=" * 80)
    print("ERROR CLASSIFICATION SYSTEM - DEMONSTRATION")
    print("=" * 80)
    print()
    print("This demonstrates how the AI classification system would categorize real errors.")
    print("(Without calling Claude API - showing expected classifications)")
    print()

    # Initialize components
    deduplicator = ErrorDeduplicator()
    extractor = ErrorExtractor()

    print(f"📊 Available Categories: {len(ERROR_CATEGORIES)}")
    category_defs = get_category_definitions()
    for i, category in enumerate(ERROR_CATEGORIES, 1):
        print(f"   {i:2d}. {category:25s} - {category_defs[category]}")
    print()

    print("=" * 80)
    print("SAMPLE ERROR CLASSIFICATIONS")
    print("=" * 80)
    print()

    # Process each sample error
    for i, sample in enumerate(SAMPLE_ERRORS, 1):
        print(f"{i}. {sample['name']}")
        print("-" * 80)

        # Show error snippet
        error_snippet = sample['text'][:150].replace('\n', ' ')
        print(f"Error: {error_snippet}...")
        print()

        # Compute hash for deduplication
        error_hash = deduplicator.compute_error_hash(sample['text'])
        print(f"Hash (for deduplication): {error_hash}")

        # Show normalized version
        normalized = deduplicator.normalize_error_text(sample['text'])
        normalized_snippet = normalized[:100].replace('\n', ' ')
        print(f"Normalized: {normalized_snippet}...")
        print()

        # Expected classification
        print(f"✓ Expected Category: {sample['expected_category']}")
        print(f"  Description: {category_defs[sample['expected_category']]}")
        print()

    # Demonstrate deduplication
    print("=" * 80)
    print("DEDUPLICATION DEMONSTRATION")
    print("=" * 80)
    print()

    # Create two similar errors with different timestamps
    error1 = "RuntimeError at 2025-01-15 10:30:45: CUDA out of memory (GPU 0)"
    error2 = "RuntimeError at 2026-02-03 14:20:10: CUDA out of memory (GPU 0)"

    hash1 = deduplicator.compute_error_hash(error1)
    hash2 = deduplicator.compute_error_hash(error2)

    print(f"Error 1: {error1}")
    print(f"Hash 1:  {hash1}")
    print()
    print(f"Error 2: {error2}")
    print(f"Hash 2:  {hash2}")
    print()

    if hash1 == hash2:
        print("✅ SUCCESS: Same hash despite different timestamps!")
        print("   These would be treated as the same error (deduplicated)")
        print("   Only classified once, saving API costs")
    else:
        print("❌ ISSUE: Different hashes for same error")
    print()

    # Show cost savings
    print("=" * 80)
    print("COST OPTIMIZATION EXAMPLE")
    print("=" * 80)
    print()
    print("Scenario: 437 errors from validation (from GitHub API)")
    print()
    print("Without optimization:")
    print("  - 437 API calls × $0.003 = $1.31")
    print()
    print("With deduplication (70% duplicates):")
    print("  - 131 unique errors × $0.003 = $0.39")
    print("  - Savings: $0.92 (70% reduction)")
    print()
    print("With deduplication + caching (after first request):")
    print("  - 131 unique × $0.0003 (cached) = $0.04")
    print("  - Savings: $1.27 (97% reduction)")
    print()

    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("To use the full system with Claude API:")
    print()
    print("1. Set environment variables:")
    print("   export ANTHROPIC_API_KEY='your-key'")
    print("   export OPENSEARCH_URL='https://your-instance'")
    print("   export ERROR_CLASSIFICATION_INDEX='error_classifications'")
    print()
    print("2. Run batch classification:")
    print("   python3 upload_error_classifications.py --hours 24")
    print()
    print("3. View results in OpenSearch:")
    print("   curl https://your-instance/error_classifications/_search")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
