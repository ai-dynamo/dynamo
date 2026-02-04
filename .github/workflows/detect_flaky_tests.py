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
    return failed_tests


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

        # Get last author who modified this file
        git_log_cmd = ["git", "log", "-1", "--format=%an", file_path]

        result = subprocess.run(git_log_cmd, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            logger.warning(f"Git log failed for {file_path}")
            return file_path, None

        author = result.stdout.strip()

        logger.info(f"Found test {test_name} in {file_path}, last modified by {author}")
        return file_path, author

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
                        {"term": {"s_test_name.keyword": test_name}},
                        {"term": {"s_test_classname.keyword": test_classname}},
                        {"term": {"s_framework.keyword": framework}},
                        {"term": {"s_branch.keyword": "main"}},
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
                    "terms": {"field": "s_test_status.keyword", "size": 10}
                }
            },
        }

        logger.info(f"Querying OpenSearch for {test_name} ({framework})")

        # Make request to OpenSearch
        response = requests.post(
            OPENSEARCH_ENDPOINT,
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
        logger.error(f"OpenSearch query failed for {test_name}: {e}")
        return {"total_runs": 0, "passed_count": 0, "failed_count": 0, "error_count": 0}
    except Exception as e:
        logger.error(f"Error querying OpenSearch for {test_name}: {e}")
        return {"total_runs": 0, "passed_count": 0, "failed_count": 0, "error_count": 0}


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
            flaky_tests.append(enriched_test)
            logger.info(
                f"Categorized {test_name} as FLAKY "
                f"(pass_rate={pass_rate:.1%}, {passed_count}/{total_runs} runs)"
            )

    return flaky_tests, legitimate_failures


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
    test_name = test["test_name"]
    framework = test["framework"]
    test_type = test["test_type"]
    author_mention = format_slack_mention(test["author"])

    if test["is_new_test"]:
        return (
            f"• `{test_name}` ({framework}, {test_type}) - "
            f"*new test, no history* - last modified by {author_mention}"
        )
    else:
        total_runs = test["total_runs"]
        passed_count = test["passed_count"]
        pass_rate = test["pass_rate"]

        return (
            f"• `{test_name}` ({framework}, {test_type}) - "
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


def main():
    """
    Main orchestration function for flaky test detection.
    """
    logger.info("=" * 80)
    logger.info("Starting Flaky Test Detection")
    logger.info("=" * 80)
    logger.info(f"GitHub Run ID: {GITHUB_RUN_ID}")
    logger.info(f"Repository: {GITHUB_REPOSITORY}")
    logger.info(f"Workflow: {WORKFLOW_NAME}")
    logger.info(f"Flaky Threshold: {FLAKY_THRESHOLD:.1%}")
    logger.info(f"Lookback Days: {LOOKBACK_DAYS}")
    logger.info("=" * 80)

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
                print(
                    f"  - {test['test_name']} ({test['framework']}, {test['test_type']})"
                )
                print(
                    f"    Pass rate: {test['pass_rate']:.1%} ({test['passed_count']}/{test['total_runs']} runs)"
                )
                print(f"    Last modified by: {test['author'] or 'unknown'}")
            print()

        if legitimate_failures:
            print("❌ LEGITIMATE FAILURES:")
            for test in legitimate_failures:
                print(
                    f"  - {test['test_name']} ({test['framework']}, {test['test_type']})"
                )
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
