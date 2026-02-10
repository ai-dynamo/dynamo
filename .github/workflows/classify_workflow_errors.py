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
import concurrent.futures
from functools import partial
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


def fetch_job_steps(job: Dict[str, Any], github_token: str, repo: str) -> List[Dict[str, Any]]:
    """
    Fetch step details for a specific job.

    Args:
        job: Job object from GitHub API
        github_token: GitHub token for authentication
        repo: Repository name (owner/repo)

    Returns:
        List of step objects with name, conclusion, number
    """
    job_id = job.get("id")

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # The job object already contains steps
    steps = job.get("steps", [])

    if not steps:
        print(f"  ⚠️  No steps found in job data")
        return []

    # Filter for failed steps, excluding classification steps
    failed_steps = [
        step for step in steps
        if step.get("conclusion") == "failure"
        and "Classify" not in step.get("name", "")  # Skip classification steps
        and "Error Classification" not in step.get("name", "")
    ]

    return failed_steps


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


def extract_failed_step_logs(full_log: str, step: Dict[str, Any]) -> str:
    """
    Extract log section for a specific failed step from full job log.

    GitHub Actions logs use ##[group] markers to separate steps.

    Args:
        full_log: Full job log content
        step: Step object with name and number

    Returns:
        Log content for that step only
    """
    step_name = step.get("name", "")
    step_number = step.get("number", 0)

    # GitHub logs format: ##[group]Step name
    # Try to find the step section
    import re

    # Look for the step by name
    step_pattern = rf"##\[group\]{re.escape(step_name)}.*?##\[endgroup\]"
    match = re.search(step_pattern, full_log, re.DOTALL | re.MULTILINE)

    if match:
        return match.group(0)

    # Fallback: if we can't parse by markers, look for step name in log
    # and extract surrounding context (next 5000 chars)
    name_pos = full_log.find(step_name)
    if name_pos != -1:
        # Get context around the step name
        start = max(0, name_pos - 500)
        end = min(len(full_log), name_pos + 5000)
        return full_log[start:end]

    # Last resort: return full log (filtered later)
    return full_log


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

        # Get failed steps for this job
        failed_steps = fetch_job_steps(job, github_token, repo)

        if not failed_steps:
            print(f"  ℹ️  No failed steps found in {job_name} (or all failures from classification steps)")
            # Create a generic error for the job since we can't identify specific failed steps
            generic_error = ErrorContext(
                error_text=f"Job '{job_name}' failed but no specific failed steps identified.\n\nSee job logs for details.",
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
            continue

        print(f"  🔍 Found {len(failed_steps)} failed step(s) in {job_name}:")
        for step in failed_steps:
            print(f"     - {step.get('name')}")

        # Fetch full job logs
        log_content = fetch_job_logs(job, github_token, repo)

        if not log_content:
            continue

        # Extract errors from each failed step
        step_error_count = 0
        for step in failed_steps:
            step_name = step.get("name")
            step_number = step.get("number")

            # Extract logs for this specific step
            step_log = extract_failed_step_logs(log_content, step)

            # Update context with step info
            step_context = {
                **job_context,
                "step_id": str(step_number),
                "step_name": step_name,
                "full_step_log": step_log,  # Store full log for potential second pass
            }

            # Extract errors from this step's logs
            step_errors = extractor.extract_from_github_job_logs(step_log, step_context)

            # Only take the FIRST error from each failed step to avoid over-classification
            # Each failed step should produce at most 1 error for classification
            if step_errors:
                # Store full log in metadata for potential re-analysis
                step_errors[0].metadata = step_errors[0].metadata or {}
                step_errors[0].metadata['full_step_log'] = step_log

                errors.append(step_errors[0])  # Only take first error
                step_error_count += 1
                if len(step_errors) > 1:
                    print(f"     ℹ️  Step '{step_name}' had {len(step_errors)} error patterns, using first one only")

        if step_error_count > 0:
            print(f"  ✅ Extracted {step_error_count} error(s) from {job_name}")
        else:
            # No errors extracted from failed steps - create generic error
            step_names = ", ".join([s.get("name", "unknown") for s in failed_steps])
            generic_error = ErrorContext(
                error_text=f"Job '{job_name}' failed in step(s): {step_names}\n\nNo specific error pattern identified. See job logs for details.",
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


def validate_classifications(classifications: List[Any], errors: List[ErrorContext]) -> List[str]:
    """
    Validate classifications for consistency and quality.

    Returns list of validation issues found.
    """
    issues = []

    # Build error hash to classification mapping
    from opensearch_upload.error_classification.deduplicator import ErrorDeduplicator
    dedup = ErrorDeduplicator(None, None)

    hash_to_classifications = {}
    for i, (classification, error) in enumerate(zip(classifications, errors)):
        error_hash = dedup.compute_error_hash(error.error_text)

        if error_hash not in hash_to_classifications:
            hash_to_classifications[error_hash] = []

        hash_to_classifications[error_hash].append({
            'index': i + 1,
            'job': error.job_name,
            'category': classification.primary_category,
            'confidence': classification.confidence_score,
            'error_text': error.error_text[:100]
        })

    # Check for duplicate errors with different classifications
    for error_hash, group in hash_to_classifications.items():
        if len(group) > 1:
            # Multiple errors with same hash
            categories = set(item['category'] for item in group)

            if len(categories) > 1:
                # Same error, different classifications - VALIDATION FAILURE
                jobs = ", ".join([item['job'] for item in group])
                cats = ", ".join([f"{item['category']}({item['confidence']:.0%})" for item in group])

                issues.append(
                    f"Duplicate errors classified differently:\n"
                    f"      Jobs: {jobs}\n"
                    f"      Classifications: {cats}\n"
                    f"      Error: {group[0]['error_text']}..."
                )
            else:
                # Same error, same classification - GOOD
                print(f"  ✅ Deduplication validated: {len(group)} jobs with identical error -> {group[0]['category']}")

    # Enhanced Confidence Validation
    confidences = [c.confidence_score for c in classifications]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    print(f"  📊 Confidence distribution:")
    print(f"     Average: {avg_confidence:.1%}")
    print(f"     Min: {min(confidences):.1%}, Max: {max(confidences):.1%}")

    # Check 1: Very low confidence classifications
    low_confidence = [
        (i + 1, c.primary_category, c.confidence_score, errors[i].job_name)
        for i, c in enumerate(classifications)
        if c.confidence_score < 0.6
    ]

    if low_confidence:
        for idx, cat, conf, job in low_confidence:
            issues.append(
                f"⚠️  Low confidence (#{idx}): {job} -> {cat} ({conf:.0%})"
            )

    # Check 2: Average confidence too low (suggests systematic issue)
    if avg_confidence < 0.65:
        issues.append(
            f"⚠️  Average confidence very low ({avg_confidence:.1%}) - may indicate extraction issues"
        )

    # Check 3: High variance in confidence for duplicate errors
    for error_hash, group in hash_to_classifications.items():
        if len(group) > 1:
            confidences_in_group = [item['confidence'] for item in group]
            confidence_variance = max(confidences_in_group) - min(confidences_in_group)

            if confidence_variance > 0.15:  # More than 15% difference
                issues.append(
                    f"⚠️  High confidence variance for duplicate errors:\n"
                    f"      Jobs: {', '.join([item['job'] for item in group])}\n"
                    f"      Confidences: {', '.join([f'{c:.0%}' for c in confidences_in_group])}"
                )

    # Pattern validation: Check if error text matches category expectations
    pattern_mismatches = []
    for i, (classification, error) in enumerate(zip(classifications, errors)):
        error_text = error.error_text.lower()
        category = classification.primary_category

        # Define expected patterns for each category
        expected_patterns = {
            'infrastructure_error': ['importerror', 'modulenotfounderror', 'error collecting', 'cannot import'],
            'timeout': ['timeout', 'timed out', 'exceeded time limit', 'deadline exceeded'],
            'network_error': ['connection', 'network', 'dns', 'refused', 'unreachable'],
            'dependency_error': ['could not find', 'no matching distribution', 'version conflict'],
            'assertion_failure': ['assertionerror', 'assert ', 'expected', 'actual'],
        }

        # Check if category matches patterns
        if category in expected_patterns:
            patterns = expected_patterns[category]
            if not any(pattern in error_text for pattern in patterns):
                # Category doesn't match error text patterns
                pattern_mismatches.append(
                    f"Pattern mismatch (#{i + 1}): {error.job_name} classified as {category} "
                    f"but error text doesn't match expected patterns"
                )

    # Only report pattern mismatches if confidence is also low
    for i, (classification, error) in enumerate(zip(classifications, errors)):
        if classification.confidence_score < 0.7:
            error_text = error.error_text.lower()
            category = classification.primary_category

            expected_patterns = {
                'infrastructure_error': ['importerror', 'modulenotfounderror', 'error collecting'],
                'timeout': ['timeout', 'timed out', 'exceeded'],
                'network_error': ['connection', 'network', 'dns'],
            }

            if category in expected_patterns:
                if not any(pattern in error_text for pattern in expected_patterns[category]):
                    issues.append(
                        f"Low confidence + pattern mismatch (#{i + 1}): "
                        f"{error.job_name} -> {category} ({classification.confidence_score:.0%})"
                    )

    return issues


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

        # Fetch all jobs from workflow
        print("\n" + "=" * 70)
        print("📡 Fetching workflow jobs...")
        print("=" * 70)

        jobs = fetch_workflow_jobs()

        if not jobs:
            print("\n⚠️  No jobs found in workflow")
            return

        # Filter to FAILED jobs only (don't analyze passing jobs)
        failed_jobs = [
            job for job in jobs
            if job.get("conclusion") == "failure"
        ]

        if not failed_jobs:
            print("\n✅ No failed jobs in workflow")
            return

        print(f"\n🔍 Found {len(failed_jobs)} failed job(s) to analyze:")
        for job in failed_jobs:
            print(f"   - {job.get('name')}")

        # Get GitHub context for workflow
        github_token = os.getenv("GITHUB_TOKEN")
        repo = os.getenv("GITHUB_REPOSITORY")

        workflow_context = {
            "workflow_id": os.getenv("GITHUB_RUN_ID"),
            "workflow_name": os.getenv("GITHUB_WORKFLOW"),
            "repo": repo,
            "branch": os.getenv("GITHUB_REF_NAME"),
            "commit_sha": os.getenv("GITHUB_SHA"),
            "user_alias": os.getenv("GITHUB_ACTOR"),
        }

        # Process failed jobs in parallel
        print("\n" + "=" * 70)
        print("🤖 Analyzing full logs with Claude (parallel processing)...")
        print("=" * 70)

        def analyze_single_job(job, classifier, github_token, repo, workflow_context):
            """Analyze one job's full log."""
            job_id = str(job["id"])
            job_name = job["name"]

            print(f"\n  🤖 Analyzing: {job_name}")

            try:
                # Fetch complete job log
                log_content = fetch_job_logs(job, github_token, repo)
                if not log_content:
                    print(f"    ⚠️  Could not fetch logs for {job_name}")
                    return []

                # Analyze full log and get all errors
                classifications = classifier.classify_job_from_full_log(
                    job_log=log_content,
                    job_name=job_name,
                    job_id=job_id,
                    workflow_context=workflow_context,
                    use_cache=True
                )

                print(f"    ✅ Found {len(classifications)} error(s) in {job_name}")

                # Print summary
                for c in classifications:
                    print(f"       - {c.primary_category} ({c.confidence_score:.0%}): {c.step_name}")

                return classifications

            except Exception as e:
                print(f"    ❌ Failed to analyze {job_name}: {e}")
                import traceback
                traceback.print_exc()
                return []

        # Process up to 5 jobs in parallel
        max_parallel = int(os.getenv("MAX_PARALLEL_JOBS", "5"))
        all_classifications = []
        error_contexts = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            analyze_fn = partial(
                analyze_single_job,
                classifier=classifier,
                github_token=github_token,
                repo=repo,
                workflow_context=workflow_context
            )

            # Submit all jobs
            futures = [executor.submit(analyze_fn, job) for job in failed_jobs]

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    job_classifications = future.result()
                    all_classifications.extend(job_classifications)

                    # Upload to OpenSearch
                    if opensearch_client and config.error_classification_index:
                        for classification in job_classifications:
                            doc = classification.to_opensearch_doc()
                            opensearch_client.index(
                                index=config.error_classification_index,
                                id=doc.get("_id"),
                                body=doc,
                            )

                except Exception as e:
                    print(f"    ⚠️  Job analysis failed: {e}")

        classifications = all_classifications
        print(f"\n📊 Total: {len(classifications)} error(s) across {len(failed_jobs)} failed job(s)")
        print("=" * 70)

        # Validate classifications (skip detailed validation for full log analysis)
        if classifications:
            print("\n" + "=" * 70)
            print("🔍 Validating classifications...")
            print("=" * 70)

            # Basic validation
            confidences = [c.confidence_score for c in classifications]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            print(f"  📊 Confidence statistics:")
            print(f"     Average: {avg_confidence:.1%}")
            print(f"     Min: {min(confidences):.1%}, Max: {max(confidences):.1%}")

            low_confidence = [c for c in classifications if c.confidence_score < 0.6]
            if low_confidence:
                print(f"  ⚠️  {len(low_confidence)} classification(s) with low confidence (<60%)")
            else:
                print("  ✅ All classifications have reasonable confidence")

            # Category distribution
            category_counts = {}
            for c in classifications:
                cat = c.primary_category
                category_counts[cat] = category_counts.get(cat, 0) + 1

            print(f"  📊 Category distribution:")
            for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
                print(f"     - {cat}: {count}")

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

        # Create PR comment with summary
        if classifications:
            # Check if PR comments are enabled
            enable_pr_comments = os.getenv("ENABLE_PR_COMMENTS", "true").lower() == "true"

            if enable_pr_comments:
                print("\n" + "=" * 70)
                print("💬 Creating PR comment summary...")
                print("=" * 70)

                try:
                    success = annotator.create_pr_comment(
                        classifications,
                        error_contexts
                    )

                    if success:
                        print("✅ PR comment created successfully")
                    else:
                        print("ℹ️  PR comment not created (not a PR context or failed)")

                except Exception as e:
                    print(f"⚠️  Failed to create PR comment: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("\nℹ️  PR comments disabled (ENABLE_PR_COMMENTS=false)")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"✅ Analyzed {len(failed_jobs)} failed job(s)")
        print(f"✅ Found {len(classifications)} total error(s)")

        if classifications:
            # Count by category
            categories = {}
            for c in classifications:
                cat = c.primary_category
                categories[cat] = categories.get(cat, 0) + 1

            print(f"\n📊 Errors by category:")
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                print(f"   - {cat}: {count}")

            # Token usage summary
            total_prompt_tokens = sum(c.prompt_tokens for c in classifications)
            total_completion_tokens = sum(c.completion_tokens for c in classifications)
            total_cached_tokens = sum(c.cached_tokens for c in classifications)

            print(f"\n💰 Token usage:")
            print(f"   - Prompt tokens: {total_prompt_tokens:,}")
            print(f"   - Completion tokens: {total_completion_tokens:,}")
            print(f"   - Cached tokens: {total_cached_tokens:,}")

        if opensearch_client:
            print(f"\n💾 Results uploaded to OpenSearch")

        if annotator.is_available():
            print(f"📝 GitHub annotations: {'enabled' if annotator.config.enabled else 'disabled'}")

        enable_pr_comments = os.getenv("ENABLE_PR_COMMENTS", "true").lower() == "true"
        print(f"💬 PR comments: {'enabled' if enable_pr_comments else 'disabled'}")

        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during classification: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail the workflow on classification errors
        sys.exit(0)


if __name__ == "__main__":
    classify_and_annotate_workflow_errors()
