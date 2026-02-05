#!/usr/bin/env python3
"""
Test error classification with NVIDIA API.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensearch_upload.error_classification import (
    Config,
    ErrorClassifier,
    ErrorContext,
)

# Read NVIDIA API key
with open(os.path.expanduser("~/.claude2"), "r") as f:
    api_key = f.read().strip()

print("=" * 80)
print("NVIDIA API INTEGRATION TEST")
print("=" * 80)
print(f"\n✅ API Key loaded: {api_key[:10]}...\n")

# Create config for NVIDIA API (OpenAI-compatible format)
config = Config(
    anthropic_api_key=api_key,
    anthropic_model="aws/anthropic/claude-opus-4-5",
    api_format="openai",  # Use OpenAI-compatible format
    api_base_url="https://inference-api.nvidia.com/v1",
    max_error_length=10000,
)

print(f"API Format: {config.api_format}")
print(f"Base URL: {config.api_base_url}")
print(f"Model: {config.anthropic_model}\n")

# Create classifier (without OpenSearch)
classifier = ErrorClassifier(config, opensearch_client=None)

# Test with a simple error
test_error = ErrorContext(
    error_text="""RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has a total capacity of 15.77 GiB of which 1.23 GiB is free.""",
    source_type="pytest",
    framework="vllm",
    test_name="test_inference",
    workflow_id="test-123",
    job_id="test-job-456",
    workflow_name="Test Run",
    job_name="test-nvidia-api",
)

print("Testing error classification...")
print(f"Error: {test_error.error_text[:100]}...\n")

try:
    # Classify the error
    classification = classifier.classify_error(
        test_error,
        use_cache=False,
        classification_method="test"
    )

    print("=" * 80)
    print("✅ SUCCESS! NVIDIA API Classification Works!")
    print("=" * 80)
    print(f"\nCategory: {classification.primary_category}")
    print(f"Confidence: {classification.confidence_score:.2%}")
    print(f"Root Cause: {classification.root_cause_summary}")
    print(f"\nAPI Usage:")
    print(f"  Prompt tokens: {classification.prompt_tokens}")
    print(f"  Completion tokens: {classification.completion_tokens}")
    print(f"  Total tokens: {classification.prompt_tokens + classification.completion_tokens}")
    print(f"  Model: {classification.model_version}")
    print(f"\n✅ The error classification system now works with NVIDIA API!")

except Exception as e:
    print("=" * 80)
    print("❌ ERROR")
    print("=" * 80)
    print(f"\n{e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)
