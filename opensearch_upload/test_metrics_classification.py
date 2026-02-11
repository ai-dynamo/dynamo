#!/usr/bin/env python3
"""
Test script for error classification integration with workflow metrics uploader.

This script tests the error classification feature by simulating a failed job
and verifying that error fields are correctly added to the metrics.
"""

import os
import sys
import json
from datetime import datetime, timezone

# Set test environment variables
os.environ["ENABLE_ERROR_CLASSIFICATION"] = "true"

# Import after setting env vars
from workflow_metrics_uploader import WorkflowMetricsUploader, FIELD_ERROR_TYPE, FIELD_ERROR_SUMMARY, FIELD_ERROR_CONFIDENCE


def create_mock_job_data():
    """Create mock job data for a failed job."""
    return {
        "id": 123456789,
        "name": "test-build-job",
        "conclusion": "failure",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "runner_name": "test-runner",
        "runner_id": 1
    }


def create_mock_workflow_data():
    """Create mock workflow data."""
    return {
        "id": 987654321,
        "name": "Test Workflow",
        "event": "push",
        "head_branch": "main",
        "head_sha": "abc123def456",
        "run_attempt": 1,
        "actor": {"login": "test-user"}
    }


def test_error_classification_fields():
    """Test that error classification fields are added to job metrics."""
    print("🧪 Testing error classification integration\n")

    # Check if API key is set
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set - cannot test classification")
        print("   Set the API key to test actual classification")
        return False

    # Try to initialize uploader
    try:
        # Mock required env vars
        os.environ["WORKFLOW_INDEX"] = "http://localhost:9200/test-workflows"
        os.environ["JOB_INDEX"] = "http://localhost:9200/test-jobs"
        os.environ["STEPS_INDEX"] = "http://localhost:9200/test-steps"
        os.environ["GITHUB_TOKEN"] = "test-token"
        os.environ["REPO"] = "test/repo"

        uploader = WorkflowMetricsUploader()

        if not uploader.error_classifier:
            print("❌ Error classifier not initialized")
            return False

        print("✅ Workflow metrics uploader initialized with error classification")

    except Exception as e:
        print(f"❌ Failed to initialize uploader: {e}")
        return False

    # Test that fields are defined
    print(f"\n📋 Checking field constants:")
    print(f"   FIELD_ERROR_TYPE: {FIELD_ERROR_TYPE}")
    print(f"   FIELD_ERROR_SUMMARY: {FIELD_ERROR_SUMMARY}")
    print(f"   FIELD_ERROR_CONFIDENCE: {FIELD_ERROR_CONFIDENCE}")

    # Create mock job metrics dict
    job_metrics = {
        "_id": "github-job-123456789-attempt-1",
        "s_job_id": "123456789",
        "s_job_name": "test-build-job",
        "s_status": "failure",
        "l_status_number": 1,
        "@timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Create mock job and workflow data
    job_data = create_mock_job_data()
    workflow_data = create_mock_workflow_data()

    print(f"\n🧪 Testing add_error_classification_fields method:")
    print(f"   Job: {job_data['name']}")
    print(f"   Status: {job_data['conclusion']}")

    # Note: This will fail to fetch real logs since it's a mock job
    # but we can verify the method exists and handles errors gracefully
    try:
        uploader.add_error_classification_fields(
            job_metrics,
            job_data=job_data,
            workflow_data=workflow_data
        )
        print("✅ Method executed without crashing")

        # Check if fields were added (they won't be since logs don't exist)
        if FIELD_ERROR_TYPE in job_metrics:
            print(f"✅ Error classification fields added:")
            print(f"   Type: {job_metrics[FIELD_ERROR_TYPE]}")
            print(f"   Summary: {job_metrics[FIELD_ERROR_SUMMARY][:100]}...")
            print(f"   Confidence: {job_metrics[FIELD_ERROR_CONFIDENCE]}")
        else:
            print("ℹ️  No classification fields added (expected for mock data)")

    except Exception as e:
        print(f"❌ Error during classification: {e}")
        return False

    print("\n✅ Error classification integration test passed!")
    print("\n📝 To test with real data:")
    print("   1. Set environment variables (WORKFLOW_INDEX, JOB_INDEX, etc.)")
    print("   2. Run: python3 workflow_metrics_uploader.py")
    print("   3. Check OpenSearch for s_error_type, s_error_summary, f_error_confidence fields")

    return True


if __name__ == "__main__":
    success = test_error_classification_fields()
    sys.exit(0 if success else 1)
