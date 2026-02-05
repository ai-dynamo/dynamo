#!/usr/bin/env python3
"""
Batch Error Classification Cronjob

Runs periodically to classify unprocessed errors from OpenSearch.
Fetches failed tests, jobs, and steps, deduplicates, classifies with Claude,
and uploads results to OpenSearch.

Usage:
    export ANTHROPIC_API_KEY=<your-key>
    export OPENSEARCH_URL=<url>
    export ERROR_CLASSIFICATION_INDEX=<index>
    python3 upload_error_classifications.py --hours 24
"""

import argparse
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from opensearchpy import OpenSearch
except ImportError:
    print("✗ Error: opensearch-py not installed")
    print("  Install with: pip install opensearch-py")
    sys.exit(1)

from error_classification import (
    Config,
    ErrorClassifier,
    ErrorExtractor,
    ErrorContext,
    create_index_if_not_exists,
)


class BatchErrorClassifier:
    """Batch error classifier for cronjob processing."""

    def __init__(self, config: Config):
        """
        Initialize batch classifier.

        Args:
            config: Configuration object
        """
        self.config = config

        # Initialize OpenSearch client
        self.opensearch = self._create_opensearch_client()

        # Initialize classifier
        self.classifier = ErrorClassifier(config, self.opensearch)

        # Initialize extractor
        self.extractor = ErrorExtractor()

        # Create index if needed
        if config.error_classification_index:
            create_index_if_not_exists(
                self.opensearch,
                config.error_classification_index
            )

    def _create_opensearch_client(self) -> OpenSearch:
        """Create OpenSearch client from config."""
        if not self.config.opensearch_url:
            raise ValueError("OPENSEARCH_URL is required")

        auth = None
        if self.config.opensearch_username and self.config.opensearch_password:
            auth = (self.config.opensearch_username, self.config.opensearch_password)

        return OpenSearch(
            hosts=[self.config.opensearch_url],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            ssl_show_warn=False,
        )

    def find_unclassified_errors(self, hours_back: int = 24) -> List[ErrorContext]:
        """
        Query OpenSearch for errors without classifications.

        Searches:
        1. Tests index: failed tests with error messages
        2. Jobs index: failed jobs with annotations
        3. Steps index: failed steps
        4. Layers index: error status layers

        Args:
            hours_back: Hours to look back

        Returns:
            List of ErrorContext objects
        """
        print(f"🔍 Searching for unclassified errors (past {hours_back}h)...")

        errors = []
        min_timestamp = datetime.now(timezone.utc) - timedelta(hours=hours_back)

        # 1. Search tests index for failed tests
        errors.extend(self._find_failed_tests(min_timestamp))

        # 2. Search jobs index for failed jobs with annotations
        errors.extend(self._find_failed_jobs(min_timestamp))

        # 3. Search steps index for failed steps
        errors.extend(self._find_failed_steps(min_timestamp))

        print(f"✓ Found {len(errors)} unclassified errors")
        return errors

    def _find_failed_tests(self, min_timestamp: datetime) -> List[ErrorContext]:
        """Find failed tests from tests index."""
        errors = []

        try:
            # Note: Assuming tests index name from environment
            tests_index = os.getenv("TESTS_INDEX")
            if not tests_index:
                print("  ⚠️  TESTS_INDEX not set, skipping test errors")
                return errors

            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"s_test_status": "failed"}},
                            {"exists": {"field": "s_error_message"}},
                            {"range": {"@timestamp": {"gte": min_timestamp.isoformat()}}}
                        ]
                    }
                },
                "size": 1000,
                "_source": [
                    "s_test_name", "s_error_message", "s_workflow_id",
                    "s_job_id", "s_step_id", "s_workflow_name", "s_job_name",
                    "s_repo", "s_branch", "s_pr_id", "s_commit_sha",
                    "s_user_alias", "@timestamp"
                ]
            }

            response = self.opensearch.search(index=tests_index, body=query)
            hits = response.get("hits", {}).get("hits", [])

            print(f"  Found {len(hits)} failed tests")

            for hit in hits:
                source = hit["_source"]

                # Detect framework from test name
                test_name = source.get("s_test_name", "")
                framework = self._detect_framework(test_name)

                error_context = ErrorContext(
                    error_text=source.get("s_error_message", ""),
                    source_type="pytest",
                    test_name=test_name,
                    framework=framework,
                    workflow_id=source.get("s_workflow_id"),
                    job_id=source.get("s_job_id"),
                    step_id=source.get("s_step_id"),
                    job_name=source.get("s_job_name"),
                    workflow_name=source.get("s_workflow_name"),
                    repo=source.get("s_repo"),
                    branch=source.get("s_branch"),
                    pr_id=source.get("s_pr_id"),
                    commit_sha=source.get("s_commit_sha"),
                    user_alias=source.get("s_user_alias"),
                    timestamp=source.get("@timestamp"),
                )

                errors.append(error_context)

        except Exception as e:
            print(f"  ⚠️  Error querying tests index: {e}")

        return errors

    def _find_failed_jobs(self, min_timestamp: datetime) -> List[ErrorContext]:
        """Find failed jobs from jobs index."""
        errors = []

        try:
            jobs_index = os.getenv("JOB_INDEX")
            if not jobs_index:
                print("  ⚠️  JOB_INDEX not set, skipping job errors")
                return errors

            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"s_status": "failure"}},
                            {"exists": {"field": "s_annotation_messages"}},
                            {"range": {"@timestamp": {"gte": min_timestamp.isoformat()}}}
                        ]
                    }
                },
                "size": 1000,
                "_source": [
                    "s_job_name", "s_annotation_messages", "s_workflow_id",
                    "s_job_id", "s_workflow_name", "s_repo", "s_branch",
                    "s_pr_id", "s_commit_sha", "s_user_alias", "@timestamp"
                ]
            }

            response = self.opensearch.search(index=jobs_index, body=query)
            hits = response.get("hits", {}).get("hits", [])

            print(f"  Found {len(hits)} failed jobs with annotations")

            for hit in hits:
                source = hit["_source"]

                # Extract errors from annotation messages
                annotations_text = source.get("s_annotation_messages", "")
                if not annotations_text:
                    continue

                # Parse pipe-separated annotations
                annotation_errors = self.extractor.extract_from_annotation_messages(
                    annotations_text,
                    context={
                        "workflow_id": source.get("s_workflow_id"),
                        "job_id": source.get("s_job_id"),
                        "job_name": source.get("s_job_name"),
                        "workflow_name": source.get("s_workflow_name"),
                        "repo": source.get("s_repo"),
                        "branch": source.get("s_branch"),
                        "pr_id": source.get("s_pr_id"),
                        "commit_sha": source.get("s_commit_sha"),
                        "user_alias": source.get("s_user_alias"),
                    }
                )

                errors.extend(annotation_errors)

        except Exception as e:
            print(f"  ⚠️  Error querying jobs index: {e}")

        return errors

    def _find_failed_steps(self, min_timestamp: datetime) -> List[ErrorContext]:
        """Find failed steps from steps index."""
        errors = []

        try:
            steps_index = os.getenv("STEPS_INDEX")
            if not steps_index:
                print("  ⚠️  STEPS_INDEX not set, skipping step errors")
                return errors

            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"s_status": "failure"}},
                            {"range": {"@timestamp": {"gte": min_timestamp.isoformat()}}}
                        ]
                    }
                },
                "size": 500,  # Fewer steps typically
                "_source": [
                    "s_step_name", "s_workflow_id", "s_job_id", "s_step_id",
                    "s_workflow_name", "s_job_name", "s_repo", "s_branch",
                    "s_pr_id", "s_commit_sha", "s_user_alias", "@timestamp"
                ]
            }

            response = self.opensearch.search(index=steps_index, body=query)
            hits = response.get("hits", {}).get("hits", [])

            print(f"  Found {len(hits)} failed steps")

            # Note: Steps typically don't have error text directly
            # We rely on test/job level errors for actual classification

        except Exception as e:
            print(f"  ⚠️  Error querying steps index: {e}")

        return errors

    def deduplicate_and_classify(
        self,
        errors: List[ErrorContext]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate errors and classify unique ones.

        Args:
            errors: List of error contexts

        Returns:
            List of classification documents for OpenSearch
        """
        if not errors:
            return []

        print(f"🤖 Classifying errors...")

        # Group by error hash for deduplication
        error_groups = defaultdict(list)

        for error in errors:
            error_hash = self.classifier.deduplicator.compute_error_hash(
                error.error_text
            )
            error_groups[error_hash].append(error)

        print(f"📊 {len(errors)} total errors → {len(error_groups)} unique")

        # Classify unique errors
        classifications = []

        for i, (error_hash, group) in enumerate(error_groups.items(), 1):
            print(f"  [{i}/{len(error_groups)}] Hash {error_hash[:8]}... ({len(group)} occurrences)")

            # Use first error as representative
            representative = group[0]

            try:
                # Classify
                classification = self.classifier.classify_error(
                    representative,
                    use_cache=True,
                    classification_method="batch"
                )

                # Update occurrence count
                classification.occurrence_count = len(group)

                # Convert to OpenSearch document
                doc = classification.to_opensearch_doc()
                classifications.append(doc)

            except Exception as e:
                print(f"  ✗ Failed to classify: {e}")
                continue

        return classifications

    def upload_classifications(
        self,
        classifications: List[Dict[str, Any]]
    ) -> int:
        """
        Upload classifications to OpenSearch.

        Args:
            classifications: List of classification documents

        Returns:
            Number of successfully uploaded documents
        """
        if not classifications:
            return 0

        if not self.config.error_classification_index:
            print("  ⚠️  ERROR_CLASSIFICATION_INDEX not set, skipping upload")
            return 0

        print(f"📤 Uploading {len(classifications)} classifications...")

        success_count = 0

        for doc in classifications:
            try:
                self.opensearch.index(
                    index=self.config.error_classification_index,
                    id=doc.get("_id"),
                    body=doc,
                    refresh=False,  # Don't refresh immediately
                )
                success_count += 1

            except Exception as e:
                print(f"  ✗ Failed to upload document: {e}")
                continue

        # Refresh index once at the end
        try:
            self.opensearch.indices.refresh(index=self.config.error_classification_index)
        except Exception as e:
            print(f"  ⚠️  Failed to refresh index: {e}")

        print(f"✅ Uploaded {success_count}/{len(classifications)} classifications")
        return success_count

    def process_batch(self, hours_back: int = 24):
        """
        Main batch processing function.

        Args:
            hours_back: Hours to look back for errors
        """
        print("=" * 60)
        print("BATCH ERROR CLASSIFICATION")
        print("=" * 60)
        print(f"⏰ {datetime.now(timezone.utc).isoformat()}")
        print(f"🔍 Looking back {hours_back} hours")
        print("")

        # Find unclassified errors
        errors = self.find_unclassified_errors(hours_back)

        if not errors:
            print("✅ No errors to classify")
            return

        # Deduplicate and classify
        classifications = self.deduplicate_and_classify(errors)

        if not classifications:
            print("✗ No classifications produced")
            return

        # Upload to OpenSearch
        uploaded = self.upload_classifications(classifications)

        # Summary
        print("")
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total errors found: {len(errors)}")
        print(f"Unique errors: {len(classifications)}")
        print(f"Successfully uploaded: {uploaded}")
        print("=" * 60)

    def _detect_framework(self, test_name: str) -> str:
        """Detect framework from test name."""
        test_lower = test_name.lower()

        if "vllm" in test_lower:
            return "vllm"
        elif "sglang" in test_lower or "sgl" in test_lower:
            return "sglang"
        elif "trtllm" in test_lower or "tensorrt" in test_lower:
            return "trtllm"
        elif "rust" in test_lower:
            return "rust"

        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch classify errors from OpenSearch"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours to look back (default: 24)"
    )

    args = parser.parse_args()

    try:
        # Load config from environment
        config = Config.from_env()

        # Validate
        errors = config.validate()
        if errors:
            print(f"✗ Configuration errors: {', '.join(errors)}")
            sys.exit(1)

        # Run batch processing
        classifier = BatchErrorClassifier(config)
        classifier.process_batch(hours_back=args.hours)

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
