#!/usr/bin/env python3
"""
Workflow-level error classification.

Runs at the end of a workflow to classify all failures from completed jobs.
Creates comprehensive GitHub annotations for all errors in the workflow.

Usage:
    export ANTHROPIC_API_KEY=<key>
    export GITHUB_TOKEN=<token>
    export ENABLE_ERROR_CLASSIFICATION=true
    python3 .github/workflows/classify_workflow_errors.py
"""

import os
import sys
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from opensearchpy import OpenSearch
except ImportError:
    print("⚠️  opensearch-py not installed, skipping OpenSearch upload")
    OpenSearch = None

from opensearch_upload.error_classification import (
    Config,
    ErrorClassifier,
    ErrorExtractor,
    ErrorContext,
    create_index_if_not_exists,
    GitHubAnnotator,
    AnnotationConfig,
)


def create_opensearch_client(config: Config) -> Optional[OpenSearch]:
    """Create OpenSearch client."""
    if not OpenSearch or not config.opensearch_url:
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


def fetch_workflow_jobs() -> List[Dict[str, Any]]:
    """
    Fetch all jobs in the current workflow run.

    Returns:
        List of job objects from GitHub API
    """
    github_token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    run_id = os.getenv("GITHUB_RUN_ID")

    if not all([github_token, repo, run_id]):
        print("❌ Missing required GitHub environment variables")
        print(f"   GITHUB_TOKEN: {'set' if github_token else 'missing'}")
        print(f"   GITHUB_REPOSITORY: {repo or 'missing'}")
        print(f"   GITHUB_RUN_ID: {run_id or 'missing'}")
        return []

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    jobs_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs"

    try:
        print(f"📡 Fetching workflow jobs from GitHub API...")
        response = requests.get(jobs_url, headers=headers, timeout=10)

        if response.status_code != 200:
            print(f"❌ Failed to fetch jobs: HTTP {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return []

        jobs_data = response.json()
        jobs = jobs_data.get("jobs", [])

        print(f"✅ Found {len(jobs)} jobs in workflow")
        return jobs

    except Exception as e:
        print(f"❌ Error fetching jobs: {e}")
        return []


def fetch_job_logs(job: Dict[str, Any], github_token: str, repo: str) -> Optional[str]:
    """
    Fetch logs for a specific job.

    Args:
        job: Job object from GitHub API
        github_token: GitHub token for authentication
        repo: Repository name (owner/repo)

    Returns:
        Log content as string, or None if failed
    """
    job_id = job.get("id")
    job_name = job.get("name")

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    logs_url = f"https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs"

    try:
        print(f"  📥 Fetching logs for: {job_name}")
        response = requests.get(logs_url, headers=headers, timeout=30)

        if response.status_code != 200:
            print(f"  ⚠️  Failed to fetch logs: HTTP {response.status_code}")
            return None

        log_content = response.text
        print(f"  ✅ Fetched {len(log_content)} bytes of logs")
        return log_content

    except Exception as e:
        print(f"  ⚠️  Error fetching logs: {e}")
        return None


def extract_errors_from_workflow() -> List[ErrorContext]:
    """
    Extract errors from all failed jobs in the workflow.

    Returns:
        List of ErrorContext objects
    """
    extractor = ErrorExtractor()
    errors = []

    # Get GitHub context
    github_token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")

    context = {
        "workflow_id": os.getenv("GITHUB_RUN_ID"),
        "workflow_name": os.getenv("GITHUB_WORKFLOW"),
        "repo": repo,
        "branch": os.getenv("GITHUB_REF_NAME"),
        "commit_sha": os.getenv("GITHUB_SHA"),
        "user_alias": os.getenv("GITHUB_ACTOR"),
    }

    # Fetch all jobs
    jobs = fetch_workflow_jobs()

    if not jobs:
        print("⚠️  No jobs found")
        return errors

    # Filter for failed jobs
    failed_jobs = [
        job for job in jobs
        if job.get("conclusion") == "failure"
    ]

    if not failed_jobs:
        print("✅ No failed jobs in workflow")
        return errors

    print(f"\n🔍 Found {len(failed_jobs)} failed jobs:")
    for job in failed_jobs:
        print(f"   - {job.get('name')}")

    print(f"\n📋 Extracting errors from failed jobs...")

    # Extract errors from each failed job
    for job in failed_jobs:
        job_name = job.get("name")
        job_id = job.get("id")

        # Update context with job info
        job_context = {
            **context,
            "job_id": str(job_id),
            "job_name": job_name,
        }

        # Fetch job logs
        log_content = fetch_job_logs(job, github_token, repo)

        if not log_content:
            continue

        # Extract errors from logs
        job_errors = extractor.extract_from_github_job_logs(log_content, job_context)

        if job_errors:
            errors.extend(job_errors)
            print(f"  ✅ Extracted {len(job_errors)} errors from {job_name}")
        else:
            # If no specific errors extracted, create a generic error for the failed job
            generic_error = ErrorContext(
                error_text=f"Job '{job_name}' failed.\n\nSee job logs for details.",
                source_type="github_job_log",
                workflow_id=job_context.get("workflow_id"),
                job_id=job_context.get("job_id"),
                job_name=job_name,
                repo=job_context.get("repo"),
                workflow_name=job_context.get("workflow_name"),
                branch=job_context.get("branch"),
                commit_sha=job_context.get("commit_sha"),
                user_alias=job_context.get("user_alias"),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            errors.append(generic_error)
            print(f"  ℹ️  Created generic error for {job_name}")

    return errors


def classify_and_annotate_workflow_errors():
    """Main function for workflow-level error classification."""
    print("=" * 70)
    print("WORKFLOW ERROR CLASSIFICATION")
    print("=" * 70)
    print(f"Workflow: {os.getenv('GITHUB_WORKFLOW')}")
    print(f"Run ID: {os.getenv('GITHUB_RUN_ID')}")
    print(f"Repository: {os.getenv('GITHUB_REPOSITORY')}")
    print("=" * 70)

    # Check if enabled
    if not os.getenv("ENABLE_ERROR_CLASSIFICATION", "").lower() == "true":
        print("⚠️  Error classification not enabled")
        print("   Set ENABLE_ERROR_CLASSIFICATION=true to enable")
        return

    try:
        # Load config
        config = Config.from_env()

        # Create OpenSearch client
        opensearch_client = create_opensearch_client(config)

        if opensearch_client and config.error_classification_index:
            print(f"✅ OpenSearch client configured")
            create_index_if_not_exists(
                opensearch_client,
                config.error_classification_index
            )
        else:
            print("ℹ️  OpenSearch not configured, skipping upload")

        # Initialize classifier and annotator
        classifier = ErrorClassifier(config, opensearch_client)
        annotator = GitHubAnnotator(AnnotationConfig.from_env())

        # Extract errors from all failed jobs
        print("\n" + "=" * 70)
        errors = extract_errors_from_workflow()

        if not errors:
            print("\n✅ No errors found in workflow")
            print("=" * 70)
            return

        print(f"\n📊 Found {len(errors)} total errors across all jobs")
        print("=" * 70)

        # Classify all errors
        print(f"\n🤖 Classifying errors with Claude...")
        classifications = []
        error_contexts = {}

        for i, error in enumerate(errors, 1):
            job_name = error.job_name or "Unknown"
            print(f"\n  [{i}/{len(errors)}] Classifying error from: {job_name}")
            print(f"  Source: {error.source_type}")

            try:
                # Classify
                classification = classifier.classify_error(
                    error,
                    use_cache=True,
                    classification_method="workflow_summary"
                )

                classifications.append(classification)
                error_contexts[classification.error_id] = error

                print(f"  ✅ Category: {classification.primary_category}")
                print(f"     Confidence: {classification.confidence_score:.2%}")
                print(f"     Summary: {classification.root_cause_summary[:100]}...")

                # Upload to OpenSearch
                if opensearch_client and config.error_classification_index:
                    doc = classification.to_opensearch_doc()
                    opensearch_client.index(
                        index=config.error_classification_index,
                        id=doc.get("_id"),
                        body=doc,
                    )

            except Exception as e:
                print(f"  ❌ Failed to classify: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Create GitHub annotations for all classifications
        if classifications:
            print("\n" + "=" * 70)
            print("📝 Creating GitHub annotations...")
            print("=" * 70)

            try:
                success = annotator.create_check_run_with_annotations(
                    classifications,
                    error_contexts
                )

                if success:
                    print("✅ GitHub annotations created successfully")
                    print(f"   {len(classifications)} errors annotated")
                else:
                    print("⚠️  GitHub annotations not created")
                    print("   (May be disabled or unavailable)")

            except Exception as e:
                print(f"⚠️  Failed to create GitHub annotations: {e}")
                import traceback
                traceback.print_exc()

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"✅ Classified {len(classifications)}/{len(errors)} errors")

        if classifications:
            # Count by category
            categories = {}
            for c in classifications:
                cat = c.primary_category
                categories[cat] = categories.get(cat, 0) + 1

            print(f"\n📊 Errors by category:")
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                print(f"   - {cat}: {count}")

        if opensearch_client:
            print(f"\n💾 Results uploaded to OpenSearch")

        if annotator.is_available():
            print(f"📝 GitHub annotations: {'enabled' if annotator.config.enabled else 'disabled'}")

        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during classification: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail the workflow on classification errors
        sys.exit(0)


if __name__ == "__main__":
    classify_and_annotate_workflow_errors()
