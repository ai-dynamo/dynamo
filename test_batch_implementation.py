#!/usr/bin/env python3
"""
Test script for batch error classification implementation.
Validates the new full log analysis functionality.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opensearch_upload.error_classification import Config
from opensearch_upload.error_classification.claude_client import ClaudeClient
from opensearch_upload.error_classification.classifier import ErrorClassifier
from opensearch_upload.error_classification.github_annotator import GitHubAnnotator


def test_prompt_import():
    """Test that the new prompt is available."""
    from opensearch_upload.error_classification.prompts import SYSTEM_PROMPT_FULL_LOG_ANALYSIS

    print("✅ Test 1: SYSTEM_PROMPT_FULL_LOG_ANALYSIS imported successfully")
    print(f"   Prompt length: {len(SYSTEM_PROMPT_FULL_LOG_ANALYSIS)} chars")

    # Verify it contains expected content
    assert "analyze COMPLETE GitHub Actions job logs" in SYSTEM_PROMPT_FULL_LOG_ANALYSIS
    assert "errors_found" in SYSTEM_PROMPT_FULL_LOG_ANALYSIS
    assert "total_errors" in SYSTEM_PROMPT_FULL_LOG_ANALYSIS
    print("   ✅ Prompt contains expected structure")


def test_claude_client_method():
    """Test that ClaudeClient has the new method."""
    config = Config(
        anthropic_api_key="test-key",
        anthropic_model="claude-sonnet-4-5-20250929"
    )

    client = ClaudeClient(config)

    # Check method exists
    assert hasattr(client, 'analyze_full_job_log')
    assert hasattr(client, '_build_full_log_prompt')
    assert hasattr(client, '_parse_full_log_response')

    print("✅ Test 2: ClaudeClient has analyze_full_job_log method")


def test_classifier_method():
    """Test that ErrorClassifier has the new method."""
    config = Config(
        anthropic_api_key="test-key",
        anthropic_model="claude-sonnet-4-5-20250929"
    )

    classifier = ErrorClassifier(config, opensearch_client=None)

    # Check method exists
    assert hasattr(classifier, 'classify_job_from_full_log')

    print("✅ Test 3: ErrorClassifier has classify_job_from_full_log method")


def test_annotator_pr_methods():
    """Test that GitHubAnnotator has PR comment methods."""
    annotator = GitHubAnnotator()

    # Check methods exist
    assert hasattr(annotator, 'create_pr_comment')
    assert hasattr(annotator, '_get_pr_number')
    assert hasattr(annotator, '_build_summary_markdown')
    assert hasattr(annotator, '_group_by_severity')
    assert hasattr(annotator, '_truncate')

    print("✅ Test 4: GitHubAnnotator has PR comment methods")


def test_pr_markdown_generation():
    """Test markdown generation logic."""
    from dataclasses import dataclass

    @dataclass
    class MockClassification:
        error_id: str
        job_name: str
        step_name: str
        primary_category: str
        confidence_score: float
        root_cause_summary: str
        error_hash: str = "test_hash"

    # Create mock classifications
    classifications = [
        MockClassification(
            error_id="1",
            job_name="test-job-1",
            step_name="Run tests",
            primary_category="infrastructure_error",
            confidence_score=0.85,
            root_cause_summary="Test infrastructure failed"
        ),
        MockClassification(
            error_id="2",
            job_name="test-job-2",
            step_name="Build",
            primary_category="timeout",
            confidence_score=0.92,
            root_cause_summary="Build exceeded time limit"
        ),
        MockClassification(
            error_id="3",
            job_name="test-job-3",
            step_name="Unit tests",
            primary_category="assertion_failure",
            confidence_score=0.78,
            root_cause_summary="Test assertion failed"
        ),
    ]

    annotator = GitHubAnnotator()
    markdown = annotator._build_summary_markdown(classifications, {})

    # Verify markdown structure
    assert "## 🤖 AI Error Classification Summary" in markdown
    assert "Found **3 unique error(s)**" in markdown
    assert "### 🔴 Critical" in markdown
    assert "### 🟠 Important" in markdown
    assert "### 🔵 Informational" in markdown
    assert "📊 Classification Statistics" in markdown

    print("✅ Test 5: PR markdown generation works correctly")
    print(f"   Generated markdown: {len(markdown)} chars")


def test_severity_grouping():
    """Test severity grouping logic."""
    from dataclasses import dataclass

    @dataclass
    class MockClassification:
        primary_category: str
        error_hash: str = "test"

    classifications = [
        MockClassification(primary_category="infrastructure_error"),
        MockClassification(primary_category="compilation_error"),
        MockClassification(primary_category="timeout"),
        MockClassification(primary_category="configuration_error"),
        MockClassification(primary_category="assertion_failure"),
        MockClassification(primary_category="flaky_test"),
    ]

    annotator = GitHubAnnotator()
    critical, important, informational = annotator._group_by_severity(classifications)

    assert len(critical) == 2  # infrastructure, compilation
    assert len(important) == 2  # timeout, configuration
    assert len(informational) == 2  # assertion, flaky_test

    print("✅ Test 6: Severity grouping works correctly")
    print(f"   Critical: {len(critical)}, Important: {len(important)}, Informational: {len(informational)}")


def test_json_parsing():
    """Test JSON response parsing."""
    config = Config(
        anthropic_api_key="test-key",
        anthropic_model="claude-sonnet-4-5-20250929"
    )

    client = ClaudeClient(config)

    # Test valid JSON
    valid_response = """
    {
        "errors_found": [
            {
                "step": "Run tests",
                "primary_category": "infrastructure_error",
                "confidence_score": 0.85,
                "root_cause_summary": "Test failed",
                "log_excerpt": "ERROR: Test collection failed"
            }
        ],
        "total_errors": 1
    }
    """

    result = client._parse_full_log_response(valid_response)

    assert result["total_errors"] == 1
    assert len(result["errors_found"]) == 1
    assert result["errors_found"][0]["primary_category"] == "infrastructure_error"

    print("✅ Test 7: JSON parsing works correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("TESTING BATCH ERROR CLASSIFICATION IMPLEMENTATION")
    print("=" * 70)
    print()

    try:
        test_prompt_import()
        print()

        test_claude_client_method()
        print()

        test_classifier_method()
        print()

        test_annotator_pr_methods()
        print()

        test_pr_markdown_generation()
        print()

        test_severity_grouping()
        print()

        test_json_parsing()
        print()

        print("=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        return 0

    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
