#!/usr/bin/env python3
"""
Quick error classification during CI runs.

Called when a job fails to classify critical errors immediately.
Only classifies infrastructure and build errors - defers test failures to batch.

Usage:
    export ANTHROPIC_API_KEY=<key>
    export ERROR_CLASSIFICATION_INDEX=<index>
    export ENABLE_ERROR_CLASSIFICATION=true
    python3 .github/workflows/classify_errors_on_failure.py
"""

import glob
import os
import sys
from typing import List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from opensearchpy import OpenSearch
except ImportError:
    print("⚠️  opensearch-py not installed, skipping classification")
    sys.exit(0)

from opensearch_upload.error_classification import (
    Config,
    ErrorClassifier,
    ErrorExtractor,
    ErrorContext,
    create_index_if_not_exists,
    GitHubAnnotator,
    AnnotationConfig,
)


def create_opensearch_client(config: Config) -> OpenSearch:
    """Create OpenSearch client."""
    if not config.opensearch_url:
        print("⚠️  OPENSEARCH_URL not set, skipping upload")
        return None

    auth = None
    if config.opensearch_username and config.opensearch_password:
        auth = (config.opensearch_username, config.opensearch_password)

    return OpenSearch(
        hosts=[config.opensearch_url],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        ssl_show_warn=False,
    )


def extract_ci_errors() -> List[ErrorContext]:
    """
    Extract errors from current CI run context.

    Looks for:
    - test-results/*.xml (pytest JUnit)
    - build-logs/*.log (BuildKit logs)
    """
    extractor = ErrorExtractor()
    errors = []

    # Get context from environment
    context = {
        "workflow_id": os.getenv("GITHUB_RUN_ID"),
        "job_id": os.getenv("GITHUB_JOB"),
        "workflow_name": os.getenv("GITHUB_WORKFLOW"),
        "job_name": os.getenv("GITHUB_JOB"),
        "repo": os.getenv("GITHUB_REPOSITORY"),
        "branch": os.getenv("GITHUB_REF_NAME"),
        "commit_sha": os.getenv("GITHUB_SHA"),
        "user_alias": os.getenv("GITHUB_ACTOR"),
    }

    # Extract from test results
    test_results_dir = "test-results"
    if os.path.exists(test_results_dir):
        for xml_file in glob.glob(f"{test_results_dir}/*.xml"):
            print(f"📋 Extracting errors from {xml_file}")
            test_errors = extractor.extract_from_junit_xml(xml_file, context)
            errors.extend(test_errors)
            print(f"  Found {len(test_errors)} errors")

    # Extract from build logs
    build_logs_dir = "build-logs"
    if os.path.exists(build_logs_dir):
        for log_file in glob.glob(f"{build_logs_dir}/*.log"):
            print(f"📋 Extracting errors from {log_file}")
            with open(log_file, 'r') as f:
                log_content = f.read()
            build_errors = extractor.extract_from_buildkit_log(log_content, context)
            errors.extend(build_errors)
            print(f"  Found {len(build_errors)} errors")

    return errors


def classify_and_upload_critical_errors():
    """Main function for CI quick classification with GitHub annotations."""
    print("=" * 60)
    print("CI ERROR CLASSIFICATION")
    print("=" * 60)

    # Check if enabled
    if not os.getenv("ENABLE_ERROR_CLASSIFICATION", "").lower() == "true":
        print("⚠️  Error classification not enabled (ENABLE_ERROR_CLASSIFICATION != true)")
        return

    try:
        # Load config
        config = Config.from_env()

        # Create OpenSearch client
        opensearch_client = create_opensearch_client(config)

        if opensearch_client and config.error_classification_index:
            create_index_if_not_exists(
                opensearch_client,
                config.error_classification_index
            )

        # Initialize classifier and annotator
        classifier = ErrorClassifier(config, opensearch_client)
        annotator = GitHubAnnotator(AnnotationConfig.from_env())

        # Extract errors from CI artifacts
        errors = extract_ci_errors()

        if not errors:
            print("✅ No errors found in CI artifacts")
            return

        print(f"📊 Found {len(errors)} total errors")

        # Filter for critical errors that should be classified in real-time
        critical_errors = [
            error for error in errors
            if classifier.should_classify_realtime(error)
        ]

        if not critical_errors:
            print("✅ No critical errors to classify in real-time")
            print("   (Test failures will be classified in batch)")
            return

        print(f"🚨 {len(critical_errors)} critical errors to classify")

        # Classify all critical errors
        classifications = []
        error_contexts = {}

        for i, error in enumerate(critical_errors, 1):
            print(f"  [{i}/{len(critical_errors)}] Classifying {error.source_type} error...")

            try:
                # Classify
                classification = classifier.classify_error(
                    error,
                    use_cache=True,
                    classification_method="realtime"
                )

                classifications.append(classification)
                error_contexts[classification.error_id] = error

                # Upload to OpenSearch
                if opensearch_client and config.error_classification_index:
                    doc = classification.to_opensearch_doc()
                    opensearch_client.index(
                        index=config.error_classification_index,
                        id=doc.get("_id"),
                        body=doc,
                    )

                    print(f"  ✅ Classified as: {classification.primary_category} "
                          f"(confidence: {classification.confidence_score:.2f})")
                else:
                    print(f"  ✅ Classified as: {classification.primary_category} "
                          f"(confidence: {classification.confidence_score:.2f})")
                    print(f"  ⚠️  Not uploaded (OpenSearch not configured)")

            except Exception as e:
                print(f"  ✗ Failed to classify error: {e}")
                continue

        # Create GitHub annotations for all classifications
        if classifications:
            print("\n📝 Creating GitHub annotations...")
            try:
                success = annotator.create_check_run_with_annotations(
                    classifications,
                    error_contexts
                )
                if success:
                    print("✅ GitHub annotations created successfully")
                else:
                    print("⚠️  GitHub annotations not created (may be disabled or unavailable)")
            except Exception as e:
                print(f"⚠️  Failed to create GitHub annotations: {e}")
                # Don't fail the workflow

        print("=" * 60)
        print(f"✅ Classified {len(classifications)} critical errors")
        if annotator.is_available():
            print(f"📝 GitHub annotations: {'enabled' if annotator.config.enabled else 'disabled'}")
        print("=" * 60)

    except Exception as e:
        print(f"✗ Error during classification: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail the workflow on classification errors
        sys.exit(0)


if __name__ == "__main__":
    classify_and_upload_critical_errors()
