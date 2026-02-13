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

import concurrent.futures
import os
import sys
from functools import partial
from typing import Any, Dict, List, Optional

import requests

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from error_classification import Config, ErrorClassifier, PRCommentator


def find_latest_failed_pr_run(github_token: str, repo: str) -> Optional[str]:
    """
    Find the latest failed PR workflow run.

    Args:
        github_token: GitHub token for authentication
        repo: Repository name (owner/repo)

    Returns:
        Run ID as string, or None if not found
    """
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Fetch recent workflow runs
    runs_url = f"https://api.github.com/repos/{repo}/actions/runs?event=pull_request&status=failure&per_page=10"

    try:
        print("🔍 Searching for latest failed PR workflow run...")
        response = requests.get(runs_url, headers=headers, timeout=10)

        if response.status_code != 200:
            print(f"❌ Failed to fetch runs: HTTP {response.status_code}")
            return None

        runs_data = response.json()
        runs = runs_data.get("workflow_runs", [])

        if not runs:
            print("⚠️  No failed PR runs found")
            return None

        # Get the most recent failed PR run
        latest_run = runs[0]
        run_id = str(latest_run["id"])
        workflow_name = latest_run["name"]
        branch = latest_run["head_branch"]

        print(f"✅ Found latest failed run: {workflow_name} on {branch} (ID: {run_id})")
        return run_id

    except Exception as e:
        print(f"❌ Error finding latest failed run: {e}")
        return None


def fetch_workflow_jobs() -> List[Dict[str, Any]]:
    """
    Fetch all jobs in the current workflow run.

    Respects WORKFLOW_RUN_ID if set (for workflow_dispatch trigger).
    Otherwise uses GITHUB_RUN_ID (for workflow_run trigger).

    Returns:
        List of job objects from GitHub API
    """
    github_token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")

    # Check for explicit run ID (from workflow_dispatch input)
    run_id = os.getenv("WORKFLOW_RUN_ID")

    if not run_id:
        # Fallback to current run ID (for workflow_run trigger)
        run_id = os.getenv("GITHUB_RUN_ID")

    # If still no run_id and we're in workflow_dispatch mode, find latest failed PR run
    if not run_id:
        github_event_name = os.getenv("GITHUB_EVENT_NAME")
        if github_event_name == "workflow_dispatch":
            print("ℹ️  No run_id specified, searching for latest failed PR run...")
            run_id = find_latest_failed_pr_run(github_token, repo)

    if not all([github_token, repo, run_id]):
        print("❌ Missing required GitHub environment variables")
        print(f"   GITHUB_TOKEN: {'set' if github_token else 'missing'}")
        print(f"   GITHUB_REPOSITORY: {repo or 'missing'}")
        print(f"   WORKFLOW_RUN_ID: {os.getenv('WORKFLOW_RUN_ID') or 'not set'}")
        print(f"   GITHUB_RUN_ID: {os.getenv('GITHUB_RUN_ID') or 'not set'}")
        return []

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    jobs_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs"

    try:
        print(f"📡 Fetching workflow jobs from GitHub API (run ID: {run_id})...")
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


def fetch_job_steps(
    job: Dict[str, Any], github_token: str, repo: str
) -> List[Dict[str, Any]]:
    """
    Fetch step details for a specific job.

    Args:
        job: Job object from GitHub API
        github_token: GitHub token for authentication
        repo: Repository name (owner/repo)

    Returns:
        List of step objects with name, conclusion, number
    """
    # The job object already contains steps
    steps = job.get("steps", [])

    if not steps:
        print("  ⚠️  No steps found in job data")
        return []

    # Filter for failed steps, excluding classification steps
    failed_steps = [
        step
        for step in steps
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


def classify_and_annotate_workflow_errors():
    """Main function for workflow-level error classification."""
    # Determine which run ID to use
    run_id = os.getenv("WORKFLOW_RUN_ID") or os.getenv("GITHUB_RUN_ID")
    event_name = os.getenv("GITHUB_EVENT_NAME")

    print("=" * 70)
    print("WORKFLOW ERROR CLASSIFICATION")
    print("=" * 70)
    print(f"Trigger: {event_name}")
    print(f"Workflow: {os.getenv('GITHUB_WORKFLOW')}")
    print(f"Run ID: {run_id}")
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

        # Initialize classifier and PR commentator
        classifier = ErrorClassifier(config)
        # Pass Claude client to PR commentator for intelligent summary generation
        commentator = PRCommentator(claude_client=classifier.claude)

        # Fetch all jobs from workflow
        print("\n" + "=" * 70)
        print("📡 Fetching workflow jobs...")
        print("=" * 70)

        jobs = fetch_workflow_jobs()

        if not jobs:
            print("\n⚠️  No jobs found in workflow")
            return

        # Filter to FAILED jobs only (don't analyze passing jobs)
        failed_jobs = [job for job in jobs if job.get("conclusion") == "failure"]

        if not failed_jobs:
            print("\n✅ No failed jobs in workflow")
            return

        print(f"\n🔍 Found {len(failed_jobs)} failed job(s) to analyze:")
        for job in failed_jobs:
            print(f"   - {job.get('name')}")

        # Get GitHub context for workflow
        github_token = os.getenv("GITHUB_TOKEN")
        repo = os.getenv("GITHUB_REPOSITORY")

        # Use WORKFLOW_RUN_ID if set (workflow_dispatch), otherwise GITHUB_RUN_ID
        target_run_id = os.getenv("WORKFLOW_RUN_ID") or os.getenv("GITHUB_RUN_ID")

        workflow_context = {
            "workflow_id": target_run_id,
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
                    use_cache=True,
                )

                print(f"    ✅ Found {len(classifications)} error(s) in {job_name}")

                # Print summary
                for c in classifications:
                    print(
                        f"       - {c.primary_category} ({c.confidence_score:.0%}): {c.step_name}"
                    )

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

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_parallel
        ) as executor:
            analyze_fn = partial(
                analyze_single_job,
                classifier=classifier,
                github_token=github_token,
                repo=repo,
                workflow_context=workflow_context,
            )

            # Submit all jobs
            futures = [executor.submit(analyze_fn, job) for job in failed_jobs]

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    job_classifications = future.result()
                    all_classifications.extend(job_classifications)

                except Exception as e:
                    print(f"    ⚠️  Job analysis failed: {e}")

        classifications = all_classifications
        print(
            f"\n📊 Total: {len(classifications)} error(s) across {len(failed_jobs)} failed job(s)"
        )
        print("=" * 70)

        # Validate classifications (skip detailed validation for full log analysis)
        if classifications:
            print("\n" + "=" * 70)
            print("🔍 Validating classifications...")
            print("=" * 70)

            # Basic validation
            confidences = [c.confidence_score for c in classifications]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            print("  📊 Confidence statistics:")
            print(f"     Average: {avg_confidence:.1%}")
            print(f"     Min: {min(confidences):.1%}, Max: {max(confidences):.1%}")

            low_confidence = [c for c in classifications if c.confidence_score < 0.6]
            if low_confidence:
                print(
                    f"  ⚠️  {len(low_confidence)} classification(s) with low confidence (<60%)"
                )
            else:
                print("  ✅ All classifications have reasonable confidence")

            # Category distribution
            category_counts = {}
            for c in classifications:
                cat = c.primary_category
                category_counts[cat] = category_counts.get(cat, 0) + 1

            print("  📊 Category distribution:")
            for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
                print(f"     - {cat}: {count}")

        # Create PR comment with summary
        if classifications:
            # Check if PR comments are enabled
            enable_pr_comments = (
                os.getenv("ENABLE_PR_COMMENTS", "true").lower() == "true"
            )

            if enable_pr_comments:
                print("\n" + "=" * 70)
                print("💬 Creating PR comment summary...")
                print("=" * 70)

                try:
                    success = commentator.create_pr_comment(
                        classifications, error_contexts
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

            print("\n📊 Errors by category:")
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                print(f"   - {cat}: {count}")

            # Token usage summary
            total_prompt_tokens = sum(c.prompt_tokens for c in classifications)
            total_completion_tokens = sum(c.completion_tokens for c in classifications)
            total_cached_tokens = sum(c.cached_tokens for c in classifications)

            print("\n💰 Token usage:")
            print(f"   - Prompt tokens: {total_prompt_tokens:,}")
            print(f"   - Completion tokens: {total_completion_tokens:,}")
            print(f"   - Cached tokens: {total_cached_tokens:,}")

        enable_pr_comments = os.getenv("ENABLE_PR_COMMENTS", "true").lower() == "true"
        if commentator.is_available():
            print(f"💬 PR comments: {'enabled' if enable_pr_comments else 'disabled'}")
        else:
            print("💬 PR comments: unavailable (not in GitHub Actions or missing token)")

        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during classification: {e}")
        import traceback

        traceback.print_exc()
        # Don't fail the workflow on classification errors
        sys.exit(0)


if __name__ == "__main__":
    classify_and_annotate_workflow_errors()
