#!/usr/bin/env python3
"""
Flaky Test Detection Script for Post-Merge CI

This script analyzes failed pytest tests to determine if they are flaky (>80% pass rate)
or legitimate failures (≤80% pass rate or new tests). It queries OpenSearch for historical
test data, uses git blame to identify test owners, and sends Slack notifications.

Usage:
    python3 detect_flaky_tests.py

Environment Variables Required:
    OPENSEARCH_ENDPOINT: OpenSearch endpoint URL
    SLACK_WEBHOOK_URL: Slack webhook URL for notifications
    SLACK_OPS_GROUP_ID: Slack user group ID for @mentions
    GITHUB_RUN_ID: GitHub Actions run ID
    GITHUB_REPOSITORY: GitHub repository name
    WORKFLOW_NAME: GitHub workflow name
    FLAKY_THRESHOLD: Pass rate threshold (default: 0.80)
    LOOKBACK_DAYS: Days to look back in history (default: 7)
"""

import argparse
import logging
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from glob import glob
from typing import Dict, List, Optional, Tuple

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT", "")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
SLACK_OPS_GROUP_ID = os.environ.get("SLACK_OPS_GROUP_ID", "")
GITHUB_RUN_ID = os.environ.get("GITHUB_RUN_ID", "")
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "")
WORKFLOW_NAME = os.environ.get("WORKFLOW_NAME", "")
FLAKY_THRESHOLD = float(os.environ.get("FLAKY_THRESHOLD", "0.80"))
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "7"))

# Test results directory
TEST_RESULTS_DIR = "test-results"


def download_and_parse_test_artifacts() -> List[Dict]:
    """
    Parse all JUnit XML files from downloaded test artifacts.

    Returns:
        List of failed test dictionaries with keys:
        - test_name: Test function name
        - test_classname: Test module/class path
        - framework: vllm/trtllm/sglang
        - test_type: unit/integration/e2e/fault-tolerance
        - status: failed/error
    """
    failed_tests = []

    # Find all JUnit XML files
    xml_pattern = os.path.join(TEST_RESULTS_DIR, "**", "*.xml")
    xml_files = glob(xml_pattern, recursive=True)

    if not xml_files:
        logger.warning(f"No XML files found in {TEST_RESULTS_DIR}")
        return failed_tests

    logger.info(f"Found {len(xml_files)} XML files to parse")

    for xml_file in xml_files:
        try:
            logger.info(f"Parsing {xml_file}")

            # Extract metadata from filename
            # Expected format: pytest_test_report_{framework}_{test_type}_{arch}_{run_id}_{job_id}.xml
            filename = os.path.basename(xml_file)
            parts = (
                filename.replace("pytest_test_report_", "")
                .replace(".xml", "")
                .split("_")
            )

            framework = parts[0] if len(parts) > 0 else "unknown"
            test_type = parts[1] if len(parts) > 1 else "unknown"
            arch = parts[2] if len(parts) > 2 else "unknown"
            job_id = parts[4] if len(parts) > 4 else None

            # Build job name and URL
            job_name = f"{framework}-{arch}-{test_type}"
            job_url = None
            if job_id and GITHUB_REPOSITORY and GITHUB_RUN_ID:
                job_url = f"https://github.com/{GITHUB_REPOSITORY}/actions/runs/{GITHUB_RUN_ID}/job/{job_id}"

            # Parse XML
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Find all test cases
            for testcase in root.findall(".//testcase"):
                test_name = testcase.get("name")
                test_classname = testcase.get("classname")

                # Check if test failed or had an error
                failure = testcase.find("failure")
                error = testcase.find("error")

                if failure is not None or error is not None:
                    status = "error" if error is not None else "failed"

                    failed_tests.append(
                        {
                            "test_name": test_name,
                            "test_classname": test_classname,
                            "framework": framework,
                            "test_type": test_type,
                            "job_name": job_name,
                            "job_url": job_url,
                            "status": status,
                        }
                    )

                    logger.info(
                        f"Found failed test: {test_name} ({framework}, {test_type})"
                    )

        except ET.ParseError as e:
            logger.warning(f"Failed to parse XML file {xml_file}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error processing {xml_file}: {e}")
            continue

    logger.info(f"Total failed tests found: {len(failed_tests)}")

    # Deduplicate by (test_name, test_classname, framework) — same test failing in
    # multiple matrix variants (arch × cuda version) should appear once with all jobs listed.
    deduped: Dict[tuple, Dict] = {}
    for test in failed_tests:
        key = (test["test_name"], test["test_classname"], test["framework"])
        if key not in deduped:
            deduped[key] = {
                "test_name": test["test_name"],
                "test_classname": test["test_classname"],
                "framework": test["framework"],
                "test_type": test["test_type"],
                "status": test["status"],
                "jobs": [],
            }
        deduped[key]["jobs"].append(
            {"job_name": test["job_name"], "job_url": test["job_url"]}
        )

    unique_failed = list(deduped.values())
    logger.info(
        f"After deduplication: {len(unique_failed)} unique failing tests "
        f"(from {len(failed_tests)} raw failures across matrix variants)"
    )
    return unique_failed


def find_test_file_and_blame(test_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the test file containing the test function and get the last author.

    Args:
        test_name: Name of the test function

    Returns:
        Tuple of (file_path, author_username) or (None, None) if not found
    """
    try:
        # Search for test function definition
        # Use grep to find "def test_name" in test files
        grep_pattern = f"def {test_name}"
        grep_cmd = ["grep", "-r", grep_pattern, "tests/", "--include=*.py", "-l"]

        result = subprocess.run(grep_cmd, capture_output=True, text=True, timeout=10)

        if result.returncode != 0 or not result.stdout.strip():
            logger.warning(f"Test file not found for {test_name}")
            return None, None

        # Get first matching file
        file_paths = result.stdout.strip().split("\n")
        file_path = file_paths[0]

        if len(file_paths) > 1:
            logger.info(
                f"Multiple files found for {test_name}, using first: {file_path}"
            )

        # Get last author email to extract username
        git_log_cmd = ["git", "log", "-1", "--format=%ae", file_path]

        result = subprocess.run(git_log_cmd, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            logger.warning(f"Git log failed for {file_path}")
            return file_path, None

        author_email = result.stdout.strip()

        # Extract username from email (part before @)
        if author_email and "@" in author_email:
            author_username = author_email.split("@")[0]
        else:
            author_username = author_email if author_email else None

        logger.info(
            f"Found test {test_name} in {file_path}, last modified by {author_username}"
        )
        return file_path, author_username

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout while searching for test {test_name}")
        return None, None
    except Exception as e:
        logger.error(f"Error finding test file for {test_name}: {e}")
        return None, None


def query_opensearch_test_history(
    test_name: str, test_classname: str, framework: str
) -> Dict:
    """
    Query OpenSearch for historical test data over the past 7 days.

    Args:
        test_name: Test function name
        test_classname: Test module/class path
        framework: Test framework (vllm/trtllm/sglang)

    Returns:
        Dictionary with keys:
        - total_runs: Total number of test runs
        - passed_count: Number of passed runs
        - failed_count: Number of failed runs
        - error_count: Number of error runs
    """
    if not OPENSEARCH_ENDPOINT:
        logger.error("OPENSEARCH_ENDPOINT not configured")
        return {"total_runs": 0, "passed_count": 0, "failed_count": 0, "error_count": 0}

    try:
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=LOOKBACK_DAYS)

        # Build OpenSearch query
        query = {
            "size": 0,  # We only need aggregations, not individual documents
            "query": {
                "bool": {
                    "must": [
                        {"term": {"s_test_name": test_name}},
                        {"term": {"s_test_classname": test_classname}},
                        {"term": {"s_framework": framework}},
                        {"term": {"s_branch": "main"}},
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": end_time.isoformat(),
                                }
                            }
                        },
                    ]
                }
            },
            "aggs": {
                "status_counts": {
                    "terms": {"field": "s_test_status", "size": 10}
                }
            },
        }

        logger.info(f"Querying OpenSearch for {test_name} ({framework})")

        # Make request to OpenSearch — index URL needs /_search suffix
        search_url = OPENSEARCH_ENDPOINT.rstrip("/") + "/_search"
        response = requests.post(
            search_url,
            json=query,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        response.raise_for_status()
        data = response.json()

        # Parse aggregation results
        buckets = (
            data.get("aggregations", {}).get("status_counts", {}).get("buckets", [])
        )

        status_counts = {bucket["key"]: bucket["doc_count"] for bucket in buckets}

        passed_count = status_counts.get("passed", 0)
        failed_count = status_counts.get("failed", 0)
        error_count = status_counts.get("error", 0)
        total_runs = passed_count + failed_count + error_count

        logger.info(
            f"Historical data for {test_name}: {total_runs} runs "
            f"({passed_count} passed, {failed_count} failed, {error_count} error)"
        )

        return {
            "total_runs": total_runs,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "error_count": error_count,
        }

    except requests.RequestException as e:
        logger.warning(
            f"OpenSearch query failed for {test_name} (network unreachable): {e}"
        )
        logger.warning(
            "Test will be categorized as legitimate failure due to missing history"
        )
        return {"total_runs": 0, "passed_count": 0, "failed_count": 0, "error_count": 0}
    except Exception as e:
        logger.error(f"Error querying OpenSearch for {test_name}: {e}")
        return {"total_runs": 0, "passed_count": 0, "failed_count": 0, "error_count": 0}


def query_opensearch_run_sequence(
    test_name: str, test_classname: str, framework: str, limit: int = 20
) -> List[Dict]:
    """
    Fetch the last `limit` individual test-run documents for a test, newest-first.

    Returns:
        List of dicts {"status": str, "timestamp": str}, newest first.
        Returns [] on any error so the regression check is simply skipped.
    """
    if not OPENSEARCH_ENDPOINT:
        return []

    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=LOOKBACK_DAYS)

        query = {
            "size": limit,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"s_test_name": test_name}},
                        {"term": {"s_test_classname": test_classname}},
                        {"term": {"s_framework": framework}},
                        {"term": {"s_branch": "main"}},
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": start_time.isoformat(),
                                    "lte": end_time.isoformat(),
                                }
                            }
                        },
                    ]
                }
            },
            "sort": [{"@timestamp": {"order": "desc"}}],
            "_source": ["s_test_status", "@timestamp"],
        }

        search_url = OPENSEARCH_ENDPOINT.rstrip("/") + "/_search"
        response = requests.post(
            search_url,
            json=query,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        hits = data.get("hits", {}).get("hits", [])
        runs = []
        for hit in hits:
            src = hit.get("_source", {})
            runs.append(
                {
                    "status": src.get("s_test_status", "unknown"),
                    "timestamp": src.get("@timestamp", ""),
                }
            )
        return runs

    except Exception as e:
        logger.warning(f"query_opensearch_run_sequence failed for {test_name}: {e}")
        return []


def detect_regression(
    recent_runs: List[Dict], min_streak: int = 2
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Scan recent_runs (newest-first) for a leading streak of failures.

    Returns:
        (is_regression, streak_start_timestamp, last_pass_timestamp)
        streak_start_timestamp: timestamp of the oldest failure in the streak (first failure).
        last_pass_timestamp: timestamp of the most-recent pass before the streak, or None.
    """
    streak_len = 0
    for run in recent_runs:
        if run["status"] in ("failed", "error"):
            streak_len += 1
        else:
            break

    if streak_len < min_streak:
        return False, None, None

    streak_start_timestamp = recent_runs[streak_len - 1]["timestamp"]

    last_pass_timestamp = None
    for run in recent_runs[streak_len:]:
        if run["status"] not in ("failed", "error"):
            last_pass_timestamp = run["timestamp"]
            break

    return True, streak_start_timestamp, last_pass_timestamp


def find_breaking_commit(
    last_pass_timestamp: Optional[str], streak_start_timestamp: Optional[str]
) -> Optional[str]:
    """
    Use git log to find commits that landed between the last pass and first failure.

    Returns a short description string like "abc1234: Commit subject" (with optional
    " (+N more)" suffix), or None if zero commits or an error occurred.
    """
    try:
        cmd = ["git", "log", "main", "--format=%h %s", "--no-merges"]
        if last_pass_timestamp:
            cmd += [f"--after={last_pass_timestamp}"]
        if streak_start_timestamp:
            cmd += [f"--before={streak_start_timestamp}"]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return None

        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return None

        if len(lines) == 1:
            return lines[0]

        if len(lines) <= 5:
            return f"{lines[0]} (+{len(lines) - 1} more)"

        return None

    except Exception as e:
        logger.warning(f"find_breaking_commit failed: {e}")
        return None


def query_opensearch_daily_failures() -> List[Dict]:
    """
    Query OpenSearch for all unique tests that failed on main in the last 24 hours.

    Uses a composite aggregation to enumerate unique (test_name, classname, framework)
    combinations without fetching individual documents.

    Returns:
        List of dicts with keys: test_name, test_classname, framework, test_type, status, jobs
    """
    if not OPENSEARCH_ENDPOINT:
        logger.error("OPENSEARCH_ENDPOINT not configured")
        return []

    try:
        query = {
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        {"terms": {"s_test_status": ["failed", "error"]}},
                        {"term": {"s_branch": "main"}},
                        {"range": {"@timestamp": {"gte": "now-24h"}}},
                    ]
                }
            },
            "aggs": {
                "unique_tests": {
                    "composite": {
                        "size": 1000,
                        "sources": [
                            {"test_name": {"terms": {"field": "s_test_name"}}},
                            {"classname": {"terms": {"field": "s_test_classname"}}},
                            {"framework": {"terms": {"field": "s_framework"}}},
                        ],
                    }
                }
            },
        }

        logger.info("Querying OpenSearch for all failures on main in the last 24h")

        # Index URL needs /_search suffix
        search_url = OPENSEARCH_ENDPOINT.rstrip("/") + "/_search"
        response = requests.post(
            search_url,
            json=query,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        buckets = (
            data.get("aggregations", {})
            .get("unique_tests", {})
            .get("buckets", [])
        )

        unique_tests = []
        for bucket in buckets:
            key = bucket.get("key", {})
            unique_tests.append(
                {
                    "test_name": key.get("test_name", ""),
                    "test_classname": key.get("classname", ""),
                    "framework": key.get("framework", ""),
                    "test_type": "unknown",
                    "status": "failed",
                    "jobs": [],
                }
            )

        logger.info(
            f"Found {len(unique_tests)} unique failing tests on main in the last 24h"
        )
        return unique_tests

    except requests.RequestException as e:
        logger.error(f"Failed to query OpenSearch for daily failures: {e}")
        return []
    except Exception as e:
        logger.error(f"Error querying OpenSearch for daily failures: {e}")
        return []


def categorize_failed_tests(failed_tests: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Categorize failed tests as flaky or legitimate failures.

    Args:
        failed_tests: List of failed test dictionaries

    Returns:
        Tuple of (flaky_tests, legitimate_failures) where each is a list of dicts with:
        - test_name, test_classname, framework, test_type, status (original fields)
        - total_runs, passed_count, failed_count (historical data)
        - pass_rate (calculated)
        - file_path, author (git blame)
        - is_new_test (boolean)
    """
    flaky_tests = []
    legitimate_failures = []

    for test in failed_tests:
        test_name = test["test_name"]
        test_classname = test["test_classname"]
        framework = test["framework"]

        # Query OpenSearch for historical data
        history = query_opensearch_test_history(test_name, test_classname, framework)

        # Find test file and blame
        file_path, author = find_test_file_and_blame(test_name)

        # Calculate pass rate
        total_runs = history["total_runs"]
        passed_count = history["passed_count"]

        is_new_test = total_runs == 0
        pass_rate = passed_count / total_runs if total_runs > 0 else 0.0

        # Create enriched test entry
        enriched_test = {
            **test,
            "total_runs": total_runs,
            "passed_count": passed_count,
            "failed_count": history["failed_count"],
            "error_count": history["error_count"],
            "pass_rate": pass_rate,
            "file_path": file_path,
            "author": author,
            "is_new_test": is_new_test,
        }

        # Categorize
        if is_new_test or pass_rate <= FLAKY_THRESHOLD:
            legitimate_failures.append(enriched_test)
            logger.info(
                f"Categorized {test_name} as LEGITIMATE FAILURE "
                f"(new_test={is_new_test}, pass_rate={pass_rate:.1%})"
            )
        else:
            # Test has high overall pass rate — check for recent failure streak
            recent_runs = query_opensearch_run_sequence(test_name, test_classname, framework)
            is_regression, streak_start_ts, last_pass_ts = detect_regression(recent_runs)
            if is_regression:
                breaking_commit = find_breaking_commit(last_pass_ts, streak_start_ts)
                enriched_test["is_regression"] = True
                enriched_test["breaking_commit"] = breaking_commit
                legitimate_failures.append(enriched_test)
                logger.info(
                    f"Categorized {test_name} as REGRESSION "
                    f"(pass_rate={pass_rate:.1%} overall, breaking_commit={breaking_commit})"
                )
            else:
                flaky_tests.append(enriched_test)
                logger.info(
                    f"Categorized {test_name} as FLAKY "
                    f"(pass_rate={pass_rate:.1%}, {passed_count}/{total_runs} runs)"
                )

    return flaky_tests, legitimate_failures


def group_parameterized_tests(tests: List[Dict]) -> List[Dict]:
    """
    Collapse parameterized test variants with identical pass rates into a single entry.

    Tests like test_foo[engine_args0-True] and test_foo[engine_args1-True] that share
    the same base name and identical pass rates are merged into one line:

        test_foo (N variants) - 98% pass (84/86 runs)

    Variants with different pass rates remain as separate entries.
    Non-parameterized tests (no '[' in name) are always returned unchanged (group size 1).

    Args:
        tests: List of enriched test dicts (output of categorize_failed_tests)

    Returns:
        List of test dicts, with same-rate variants collapsed into grouped entries.
    """
    from collections import defaultdict

    # Group by (base_name, classname, framework, passed_count, total_runs, is_new_test).
    # Using integer counts instead of float pass_rate avoids floating-point comparison issues.
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for test in tests:
        base_name = test["test_name"].split("[")[0]
        key = (
            base_name,
            test.get("test_classname", ""),
            test.get("framework", ""),
            test.get("passed_count", 0),
            test.get("total_runs", 0),
            test.get("is_new_test", False),
        )
        groups[key].append(test)

    result = []
    for (base_name, _classname, _framework, _passed, _total, _new), members in groups.items():
        if len(members) == 1:
            result.append(members[0])
        else:
            # Merge jobs lists from all variants, deduplicating by (name, url).
            merged_jobs = []
            seen_jobs: set = set()
            for m in members:
                for job in m.get("jobs", []):
                    job_key = (job.get("job_name"), job.get("job_url"))
                    if job_key not in seen_jobs:
                        seen_jobs.add(job_key)
                        merged_jobs.append(job)

            representative = members[0]
            merged = {
                **representative,
                "test_name": base_name,
                "is_grouped": True,
                "variant_count": len(members),
                "jobs": merged_jobs,
            }
            result.append(merged)

    return result


def format_slack_mention(github_username: Optional[str]) -> str:
    """
    Format GitHub username for Slack mention.

    Args:
        github_username: GitHub username or None

    Returns:
        Formatted mention string

    Note:
        Currently uses plain text format @username assuming GitHub username = Slack username.
        TODO: Add GitHub->Slack user ID mapping for proper mentions using <@USER_ID> format.
    """
    if not github_username:
        return "unknown"

    # Plain text format - assumes GitHub username = Slack username
    # For proper Slack mentions, we would need a mapping file: GitHub username -> Slack user ID
    # Then format as: <@USER_ID>
    return f"@{github_username}"


def format_test_entry(test: Dict) -> str:
    """
    Format a single test entry for Slack message.

    Args:
        test: Test dictionary with historical and blame data

    Returns:
        Formatted string for Slack
    """
    author_mention = format_slack_mention(test["author"])

    # Name display — grouped variants show "(N variants)" suffix
    if test.get("is_grouped"):
        name_str = f"`{test['test_name']}` ({test['variant_count']} variants)"
    else:
        name_str = f"`{test['test_name']}`"

    # Build job links — one entry may span multiple matrix variants
    jobs = test.get("jobs", [])
    if jobs:
        job_parts = []
        for job in jobs:
            j_name = job.get("job_name", "unknown-job")
            j_url = job.get("job_url")
            job_parts.append(f"<{j_url}|{j_name}>" if j_url else j_name)
        jobs_str = ", ".join(job_parts)
        job_suffix = f" ({jobs_str})"
    else:
        job_suffix = ""

    if test.get("is_regression"):
        pass_rate = test["pass_rate"]
        passed_count = test["passed_count"]
        total_runs = test["total_runs"]
        breaking = test.get("breaking_commit")
        commit_str = f" — broken after `{breaking}`" if breaking else ""
        return (
            f"• {name_str}{job_suffix} - "
            f"*regression*{commit_str} "
            f"({pass_rate:.0%} overall, now failing consistently) - "
            f"last modified by {author_mention}"
        )

    if test["is_new_test"]:
        return (
            f"• {name_str}{job_suffix} - "
            f"*new test, no history* - last modified by {author_mention}"
        )
    else:
        total_runs = test["total_runs"]
        passed_count = test["passed_count"]
        pass_rate = test["pass_rate"]

        return (
            f"• {name_str}{job_suffix} - "
            f"*{pass_rate:.0%} pass rate* ({passed_count}/{total_runs} runs) - "
            f"last modified by {author_mention}"
        )


def send_slack_notification(
    flaky_tests: List[Dict], legitimate_failures: List[Dict]
) -> bool:
    """
    Send Slack notification with categorized test failures.

    Args:
        flaky_tests: List of flaky test dictionaries
        legitimate_failures: List of legitimate failure dictionaries

    Returns:
        True if notification sent successfully, False otherwise
    """
    if not SLACK_WEBHOOK_URL:
        logger.error("SLACK_WEBHOOK_URL not configured")
        return False

    # Don't send notification if there are no failures
    if not flaky_tests and not legitimate_failures:
        logger.info("No test failures to report, skipping Slack notification")
        return True

    try:
        total_failed = len(flaky_tests) + len(legitimate_failures)

        # Build workflow run URL
        run_url = f"https://github.com/{GITHUB_REPOSITORY}/actions/runs/{GITHUB_RUN_ID}"

        # Build Slack Block Kit payload
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🔍 Flaky Test Detection - Post-Merge CI",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Workflow:* {WORKFLOW_NAME}\n"
                        f"*Run:* <{run_url}|#{GITHUB_RUN_ID}>\n"
                        f"*Total Failed Tests:* {total_failed}"
                    ),
                },
            },
        ]

        # Add flaky tests section
        if flaky_tests:
            blocks.append({"type": "divider"})

            flaky_text = "*🎲 Flaky Tests (>80% pass rate):*\n"
            flaky_text += "\n".join(format_test_entry(test) for test in flaky_tests)

            blocks.append(
                {"type": "section", "text": {"type": "mrkdwn", "text": flaky_text}}
            )

        # Add legitimate failures section
        if legitimate_failures:
            blocks.append({"type": "divider"})

            legit_text = "*❌ Legitimate Failures (≤80% pass rate or new test):*\n"
            legit_text += "\n".join(
                format_test_entry(test) for test in legitimate_failures
            )

            blocks.append(
                {"type": "section", "text": {"type": "mrkdwn", "text": legit_text}}
            )

        # Add footer
        blocks.append({"type": "divider"})

        footer_text = (
            "Flaky tests may need retry logic; legitimate failures need investigation"
        )
        if SLACK_OPS_GROUP_ID:
            footer_text = f"<!subteam^{SLACK_OPS_GROUP_ID}> - {footer_text}"

        blocks.append(
            {"type": "context", "elements": [{"type": "mrkdwn", "text": footer_text}]}
        )

        # Build full payload
        payload = {
            "text": f"Flaky Test Detection Results - {total_failed} failed tests",
            "blocks": blocks,
        }

        # Send to Slack
        logger.info("Sending Slack notification")

        response = requests.post(
            SLACK_WEBHOOK_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        response.raise_for_status()

        logger.info("Slack notification sent successfully")
        return True

    except requests.RequestException as e:
        logger.error(f"Failed to send Slack notification: {e}")
        return False
    except Exception as e:
        logger.error(f"Error building Slack notification: {e}")
        return False


def send_digest_slack_notification(
    flaky_tests: List[Dict], legitimate_failures: List[Dict]
) -> bool:
    """
    Send a daily digest Slack notification summarising all failures on main.

    Args:
        flaky_tests: List of flaky test dictionaries
        legitimate_failures: List of legitimate failure dictionaries

    Returns:
        True if notification sent successfully, False otherwise
    """
    if not SLACK_WEBHOOK_URL:
        logger.error("SLACK_WEBHOOK_URL not configured")
        return False

    if not flaky_tests and not legitimate_failures:
        logger.info("No failures to report in the last 24h, skipping digest")
        return True

    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        total_unique = len(flaky_tests) + len(legitimate_failures)

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🔍 Flaky Test Daily Digest — {today}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Main branch failures in the past 24 hours:* "
                        f"{total_unique} unique test(s)"
                    ),
                },
            },
        ]

        if flaky_tests:
            blocks.append({"type": "divider"})
            flaky_text = "*🎲 Flaky Tests (>80% pass rate):*\n"
            flaky_text += "\n".join(format_test_entry(t) for t in flaky_tests)
            blocks.append(
                {"type": "section", "text": {"type": "mrkdwn", "text": flaky_text}}
            )

        if legitimate_failures:
            blocks.append({"type": "divider"})
            legit_text = "*❌ Legitimate Failures (≤80% pass rate or new test):*\n"
            legit_text += "\n".join(
                format_test_entry(t) for t in legitimate_failures
            )
            blocks.append(
                {"type": "section", "text": {"type": "mrkdwn", "text": legit_text}}
            )

        blocks.append({"type": "divider"})
        footer_text = (
            "Flaky tests may need retry logic; legitimate failures need investigation"
        )
        if SLACK_OPS_GROUP_ID:
            footer_text = f"cc <!subteam^{SLACK_OPS_GROUP_ID}> — {footer_text}"
        blocks.append(
            {"type": "context", "elements": [{"type": "mrkdwn", "text": footer_text}]}
        )

        payload = {
            "text": f"Flaky Test Daily Digest — {today}: {total_unique} unique failure(s)",
            "blocks": blocks,
        }

        logger.info("Sending digest Slack notification")
        response = requests.post(
            SLACK_WEBHOOK_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        logger.info("Digest Slack notification sent successfully")
        return True

    except requests.RequestException as e:
        logger.error(f"Failed to send digest Slack notification: {e}")
        return False
    except Exception as e:
        logger.error(f"Error building digest Slack notification: {e}")
        return False


def run_digest_mode() -> int:
    """
    Daily digest mode: query OpenSearch for all failures on main in the last 24h,
    categorise them, and send one Slack notification.
    """
    logger.info("=" * 80)
    logger.info("Starting Flaky Test Daily Digest")
    logger.info("=" * 80)

    failed_tests = query_opensearch_daily_failures()

    if not failed_tests:
        logger.info("No failures found on main in the last 24h — nothing to report")
        return 0

    flaky_tests, legitimate_failures = categorize_failed_tests(failed_tests)
    flaky_tests = group_parameterized_tests(flaky_tests)
    legitimate_failures = group_parameterized_tests(legitimate_failures)

    logger.info("Categorization complete:")
    logger.info(f"  - Flaky tests: {len(flaky_tests)}")
    logger.info(f"  - Legitimate failures: {len(legitimate_failures)}")

    success = send_digest_slack_notification(flaky_tests, legitimate_failures)
    if not success:
        logger.warning("Digest Slack notification failed")

    logger.info("=" * 80)
    logger.info("Flaky Test Daily Digest Complete")
    logger.info("=" * 80)
    return 0


def main():
    """
    Main orchestration function for flaky test detection.
    """
    parser = argparse.ArgumentParser(description="Flaky test detection / digest")
    parser.add_argument(
        "--mode",
        choices=["detect", "digest"],
        default="detect",
        help="detect: parse local JUnit XML artifacts (default); digest: query OpenSearch for last 24h and send daily Slack digest",
    )
    args = parser.parse_args()

    if args.mode == "digest":
        return run_digest_mode()

    logger.info("=" * 80)
    logger.info("Starting Flaky Test Detection")
    logger.info("=" * 80)
    logger.info(f"GitHub Run ID: {GITHUB_RUN_ID}")
    logger.info(f"Repository: {GITHUB_REPOSITORY}")
    logger.info(f"Workflow: {WORKFLOW_NAME}")
    logger.info(f"Flaky Threshold: {FLAKY_THRESHOLD:.1%}")
    logger.info(f"Lookback Days: {LOOKBACK_DAYS}")
    logger.info("=" * 80)

    # Check OpenSearch connectivity
    if OPENSEARCH_ENDPOINT:
        try:
            logger.info(f"Testing OpenSearch connectivity: {OPENSEARCH_ENDPOINT}")
            requests.get(OPENSEARCH_ENDPOINT, timeout=5)
            logger.info("✅ OpenSearch is reachable")
        except requests.RequestException as e:
            logger.warning("⚠️ OpenSearch is NOT reachable from this runner")
            logger.warning(f"   Error: {e}")
            logger.warning(
                "   All tests will be categorized as legitimate failures (no history available)"
            )
    else:
        logger.warning("⚠️ OPENSEARCH_ENDPOINT not configured")

    try:
        # Step 1: Parse test artifacts
        logger.info("Step 1: Parsing test artifacts")
        failed_tests = download_and_parse_test_artifacts()

        if not failed_tests:
            logger.info("No failed tests found, exiting successfully")
            return 0

        # Step 2: Categorize failed tests
        logger.info("Step 2: Categorizing failed tests")
        flaky_tests, legitimate_failures = categorize_failed_tests(failed_tests)
        flaky_tests = group_parameterized_tests(flaky_tests)
        legitimate_failures = group_parameterized_tests(legitimate_failures)

        logger.info("Categorization complete:")
        logger.info(f"  - Flaky tests: {len(flaky_tests)}")
        logger.info(f"  - Legitimate failures: {len(legitimate_failures)}")

        # Step 3: Send Slack notification
        logger.info("Step 3: Sending Slack notification")
        success = send_slack_notification(flaky_tests, legitimate_failures)

        if not success:
            logger.warning("Slack notification failed, but continuing")

        logger.info("=" * 80)
        logger.info("Flaky Test Detection Complete")
        logger.info("=" * 80)

        # Print summary to console
        print("\n" + "=" * 80)
        print("FLAKY TEST DETECTION SUMMARY")
        print("=" * 80)
        print(f"Total Failed Tests: {len(failed_tests)}")
        print(f"Flaky Tests (>80% pass rate): {len(flaky_tests)}")
        print(f"Legitimate Failures (≤80% or new): {len(legitimate_failures)}")
        print("=" * 80 + "\n")

        if flaky_tests:
            print("🎲 FLAKY TESTS:")
            for test in flaky_tests:
                print(f"  - {test['test_name']}")
                for job in test.get("jobs", []):
                    j_name = job.get("job_name", "unknown-job")
                    j_url = job.get("job_url", "")
                    print(f"    Job: {j_name}" + (f" ({j_url})" if j_url else ""))
                print(
                    f"    Pass rate: {test['pass_rate']:.1%} ({test['passed_count']}/{test['total_runs']} runs)"
                )
                print(f"    Last modified by: {test['author'] or 'unknown'}")
            print()

        if legitimate_failures:
            print("❌ LEGITIMATE FAILURES:")
            for test in legitimate_failures:
                print(f"  - {test['test_name']}")
                for job in test.get("jobs", []):
                    j_name = job.get("job_name", "unknown-job")
                    j_url = job.get("job_url", "")
                    print(f"    Job: {j_name}" + (f" ({j_url})" if j_url else ""))
                if test["is_new_test"]:
                    print("    New test (no history)")
                else:
                    print(
                        f"    Pass rate: {test['pass_rate']:.1%} ({test['passed_count']}/{test['total_runs']} runs)"
                    )
                print(f"    Last modified by: {test['author'] or 'unknown'}")
            print()

        return 0

    except Exception as e:
        logger.error(f"Fatal error in flaky test detection: {e}", exc_info=True)

        # Try to send error notification to Slack
        try:
            if SLACK_WEBHOOK_URL:
                error_payload = {
                    "text": "⚠️ Flaky Test Detection encountered an error",
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": "⚠️ Flaky Test Detection Error",
                            },
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": (
                                    f"*Workflow:* {WORKFLOW_NAME}\n"
                                    f"*Run:* <https://github.com/{GITHUB_REPOSITORY}/actions/runs/{GITHUB_RUN_ID}|#{GITHUB_RUN_ID}>\n"
                                    f"*Error:* {str(e)}"
                                ),
                            },
                        },
                    ],
                }
                requests.post(SLACK_WEBHOOK_URL, json=error_payload, timeout=10)
        except Exception as slack_error:
            logger.error(f"Failed to send error notification to Slack: {slack_error}")

        # Return 0 to not block pipeline
        return 0


if __name__ == "__main__":
    sys.exit(main())
