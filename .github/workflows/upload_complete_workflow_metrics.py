# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Enhanced script to upload complete GitHub Actions workflow and job metrics.
This version runs as the final job in a workflow and captures metrics for
the entire workflow including all previous jobs.
"""

import glob
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

# FILTERING CONFIGURATION - Process all jobs except excluded ones
EXCLUDED_JOB_NAMES = [
    "Upload Workflow Metrics",  # Avoid infinite loops
    # Add other job names to exclude here as needed
]
FRAMEWORK_IMAGE_BUILD_JOBS = ["vllm", "sglang", "trtllm"]

# NEW STANDARDIZED FIELD SCHEMA - Using consistent prefixes for OpenSearch mapping
# Using prefixes: s_ for strings, l_ for longs, ts_ for timestamps

# Common fields across all tables
FIELD_ID = "_id"
FIELD_USER_ALIAS = "s_user_alias"
FIELD_REPO = "s_repo"
FIELD_WORKFLOW_NAME = "s_workflow_name"
FIELD_GITHUB_EVENT = "s_github_event"
FIELD_BRANCH = "s_branch"
FIELD_PR_ID = "s_pr_id"  # Pull request ID as string ("N/A" if not a PR)
FIELD_STATUS = "s_status"
FIELD_STATUS_NUMBER = "l_status_number"
FIELD_WORKFLOW_ID = "s_workflow_id"
FIELD_COMMIT_SHA = "s_commit_sha"

# Timing fields
FIELD_CREATION_TIME = "ts_creation_time"
FIELD_START_TIME = "ts_start_time"
FIELD_END_TIME = "ts_end_time"
FIELD_QUEUE_TIME = "l_queue_time_sec"  # Integer seconds as long
FIELD_DURATION_SEC = "l_duration_sec"

# Job-specific fields
FIELD_JOB_ID = "s_job_id"
FIELD_JOB_NAME = "s_job_name"
FIELD_RUNNER_ID = "s_runner_id"
FIELD_RUNNER_NAME = "s_runner_name"

# Step-specific fields
FIELD_STEP_ID = "s_step_id"
FIELD_NAME = "s_step_name"
FIELD_STEP_NUMBER = "l_step_number"
FIELD_COMMAND = "s_command"

# Container-specific fields (for CONTAINER_INDEX)
FIELD_BUILD_DURATION_SEC = "l_build_duration_sec"
FIELD_BUILD_START_TIME = "ts_build_start_time"
FIELD_BUILD_END_TIME = "ts_build_end_time"
FIELD_BUILD_TARGET = "s_build_target"
FIELD_BUILD_FRAMEWORK = "s_build_framework"
FIELD_BUILD_SIZE_BYTES = "l_build_size_bytes"

# Test Info
FIELD_FRAMEWORK = "s_framework"
FIELD_ERROR_MESSAGE = "s_error_message"
FIELD_TEST_NAME = "s_test_name"  # Test name (e.g., test_sglang_deployment[aggregated])
FIELD_TEST_CLASSNAME = (
    "s_test_classname"  # Test class name (e.g., tests.serve.test_sglang)
)
FIELD_TEST_DURATION = "l_test_duration_ms"
FIELD_TEST_STATUS = "s_test_status"  # Test status (passed, failed, error, skipped)


class BuildMetricsReader:
    """Reader for build metrics from environment variables and artifacts"""

    @staticmethod
    def _process_artifact_metrics(artifact_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Process and clean up artifact metrics data"""
        # Convert types (same as in get_build_metrics)
        if "build_duration_sec" in artifact_metrics:
            try:
                artifact_metrics["build_duration_sec"] = int(
                    artifact_metrics["build_duration_sec"]
                )
            except (ValueError, TypeError):
                artifact_metrics["build_duration_sec"] = 0

        if "image_size_bytes" in artifact_metrics:
            try:
                artifact_metrics["image_size_bytes"] = int(
                    artifact_metrics["image_size_bytes"]
                )
            except (ValueError, TypeError):
                artifact_metrics["image_size_bytes"] = 0

        # Convert Unix timestamps to ISO format if needed
        for time_field in ["build_start_time", "build_end_time"]:
            if time_field in artifact_metrics and artifact_metrics[time_field]:
                time_value = artifact_metrics[time_field]
                if isinstance(time_value, (int, float)) or (
                    isinstance(time_value, str) and time_value.isdigit()
                ):
                    try:
                        timestamp = float(time_value)
                        artifact_metrics[time_field] = datetime.fromtimestamp(
                            timestamp, tz=timezone.utc
                        ).isoformat()
                    except (ValueError, OSError):
                        pass  # Keep original value if conversion fails

        return artifact_metrics

    @staticmethod
    def get_build_metrics_for_job(job_name: str) -> Optional[Dict[str, Any]]:
        """Get build metrics for a specific job by looking for framework-specific artifacts"""

        # Determine framework from job name
        framework = None
        job_name_lower = job_name.lower()
        if "vllm" in job_name_lower:
            framework = "vllm"
        elif "sglang" in job_name_lower:
            framework = "sglang"
        elif "trtllm" in job_name_lower:
            framework = "trtllm"

        if not framework:
            print(f"⚠️  Could not determine framework from job name: {job_name}")
            return None

        # Determine architecture preference from job name
        preferred_arch = "amd64"  # default
        if "arm64" in job_name_lower:
            preferred_arch = "arm64"

        # Try to read consolidated metrics file first
        consolidated_path = "build-metrics/consolidated-metrics.json"
        if os.path.exists(consolidated_path):
            try:
                with open(consolidated_path, "r") as f:
                    all_metrics = json.load(f)

                # Look for job-specific metrics
                # Try preferred architecture first
                job_key = f"{framework}-{preferred_arch}"
                if job_key in all_metrics:
                    return BuildMetricsReader._process_artifact_metrics(
                        all_metrics[job_key]
                    )

                # Try other architecture
                other_arch = "arm64" if preferred_arch == "amd64" else "amd64"
                job_key = f"{framework}-{other_arch}"
                if job_key in all_metrics:
                    return BuildMetricsReader._process_artifact_metrics(
                        all_metrics[job_key]
                    )

                # Try just framework name (backward compatibility)
                if framework in all_metrics:
                    return BuildMetricsReader._process_artifact_metrics(
                        all_metrics[framework]
                    )

                print(
                    f"⚠️  No metrics found for {framework} in consolidated file. Available keys: {list(all_metrics.keys())}"
                )

            except Exception as e:
                print(f"❌ Error reading consolidated build metrics: {e}")

        # Fallback to individual file approach for backward compatibility
        # Try framework-specific artifact (direct path)
        artifact_path = f"build-metrics/metrics-{framework}-{preferred_arch}.json"
        if not os.path.exists(artifact_path):
            # Try the other architecture (direct path)
            other_arch = "arm64" if preferred_arch == "amd64" else "amd64"
            artifact_path = f"build-metrics/metrics-{framework}-{other_arch}.json"
        if not os.path.exists(artifact_path):
            # Try artifact subdirectory structure (new format)
            artifact_path = f"build-metrics/build-metrics-{framework}-{preferred_arch}/metrics-{framework}-{preferred_arch}.json"
        if not os.path.exists(artifact_path):
            # Try other architecture in subdirectory
            other_arch = "arm64" if preferred_arch == "amd64" else "amd64"
            artifact_path = f"build-metrics/build-metrics-{framework}-{other_arch}/metrics-{framework}-{other_arch}.json"
        if not os.path.exists(artifact_path):
            # Try old naming convention (backward compatibility)
            artifact_path = f"build-metrics/metrics-{framework}.json"
        if not os.path.exists(artifact_path):
            # Try alternative path (old format)
            artifact_path = f"build-metrics/build-metrics-{framework}/metrics.json"

        if os.path.exists(artifact_path):
            try:
                with open(artifact_path, "r") as f:
                    artifact_metrics = json.load(f)
                    return BuildMetricsReader._process_artifact_metrics(
                        artifact_metrics
                    )
            except Exception as e:
                print(
                    f"⚠️  Could not read {framework} build metrics from {artifact_path}: {e}"
                )

        print(f"⚠️  No build metrics artifact found for {framework} at {artifact_path}")
        return None


class TimingProcessor:
    """Centralized processor for all datetime and duration conversions using Python built-ins"""

    @staticmethod
    def _parse_iso(iso_string: str) -> datetime:
        """Parse ISO datetime string using built-in fromisoformat"""
        if not iso_string:
            return None
        try:
            # Handle 'Z' suffix by replacing with '+00:00'
            if iso_string.endswith("Z"):
                iso_string = iso_string[:-1] + "+00:00"
            return datetime.fromisoformat(iso_string)
        except ValueError:
            return None

    @staticmethod
    def calculate_time_diff(start_time: str, end_time: str) -> int:
        """Calculate duration/queue time in integer seconds"""
        if not start_time or not end_time:
            return 0

        start_dt = TimingProcessor._parse_iso(start_time)
        end_dt = TimingProcessor._parse_iso(end_time)

        if not start_dt or not end_dt:
            return 0

        # Return integer seconds directly
        duration = end_dt - start_dt
        return max(0, int(duration.total_seconds()))


def mask_sensitive_urls(error_msg: str, url: str) -> str:
    """Comprehensively mask sensitive URLs and hostnames in error messages"""
    if not url:
        return error_msg

    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        path = parsed_url.path

        # Replace components in order of specificity
        if hostname:
            error_msg = error_msg.replace(hostname, "***HOSTNAME***")
        if url in error_msg:
            error_msg = error_msg.replace(url, "***DATABASE_URL***")
        if path and path in error_msg:
            error_msg = error_msg.replace(path, "***PATH***")

        # Also mask any remaining URL patterns
        if hostname:
            pattern = rf"https?://{re.escape(hostname)}"
            error_msg = re.sub(pattern, "***MASKED_URL***", error_msg)

    except Exception:
        # If URL parsing fails, do basic masking
        if url in error_msg:
            error_msg = error_msg.replace(url, "***DATABASE_URL***")

    return error_msg


class WorkflowMetricsUploader:
    def __init__(self):
        self.headers = {"Content-Type": "application/json", "Accept-Charset": "UTF-8"}
        self.workflow_index = os.getenv("WORKFLOW_INDEX", "")
        self.jobs_index = os.getenv("JOB_INDEX", "")
        self.steps_index = os.getenv("STEPS_INDEX", "")

        # Validate that database URLs are provided
        if not self.workflow_index or not self.jobs_index or not self.steps_index:
            raise ValueError(
                "Database URLs not configured. Please set environment variables:\n"
                "  WORKFLOW_INDEX - URL for workflow metrics\n"
                "  JOB_INDEX - URL for job metrics\n"
                "  STEPS_INDEX - URL for step metrics"
            )

        # Get current workflow information
        self.repo = os.getenv("GITHUB_REPOSITORY")
        self.run_id = os.getenv("GITHUB_RUN_ID")
        self.workflow_name = os.getenv("GITHUB_WORKFLOW")
        self.actor = os.getenv("GITHUB_ACTOR")
        self.event_name = os.getenv("GITHUB_EVENT_NAME")
        self.ref = os.getenv("GITHUB_REF")
        self.ref_name = os.getenv("GITHUB_REF_NAME")
        self.sha = os.getenv("GITHUB_SHA")

        if not self.repo or not self.run_id:
            raise ValueError("Missing required GitHub environment variables")

        print(
            f"Uploading metrics for workflow '{self.workflow_name}' (run {self.run_id}) in {self.repo}"
        )

    def handle_upload_error(self, error: Exception, operation: str) -> str:
        """Centralized error handling with URL masking for all upload operations

        Args:
            error: The exception that occurred
            operation: Description of the operation that failed

        Returns:
            Sanitized error message with URLs masked
        """
        error_msg = str(error)

        # Mask all configured URLs to prevent exposure
        for url in [self.workflow_index, self.jobs_index, self.steps_index]:
            if url:  # Only mask non-empty URLs
                error_msg = mask_sensitive_urls(error_msg, url)

        return f"Error during {operation}: {error_msg}"

    def post_to_db(self, url: str, data: Dict[str, Any]) -> None:
        """Push json data to the database/OpenSearch URL"""
        print(f"Posting metrics to database... with data: {data}")
        try:
            response = requests.post(
                url, data=json.dumps(data), headers=self.headers, timeout=30
            )
            if not (200 <= response.status_code < 300):
                raise ValueError(f"Error posting to DB: HTTP {response.status_code}")
            print(f"Successfully posted metrics with ID: {data.get('_id', 'unknown')}")
        except requests.exceptions.RequestException as e:
            # Use centralized error handling
            sanitized_error = self.handle_upload_error(e, "database upload")
            raise ValueError(sanitized_error)

    def get_github_api_data(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Fetch data from GitHub API"""
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            print(
                "Error: No GitHub token found. Set GITHUB_TOKEN environment variable or repository secret."
            )
            return None

        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }

        try:
            response = requests.get(
                f"https://api.github.com{endpoint}", headers=headers, timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(self.handle_upload_error(e, f"GitHub API GET {endpoint}"))
            return None

    def add_common_context_fields(
        self, db_data: Dict[str, Any], workflow_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add common context fields used across all metric types"""
        db_data[FIELD_USER_ALIAS] = self.actor
        db_data[FIELD_REPO] = self.repo
        db_data[FIELD_WORKFLOW_NAME] = self.workflow_name
        db_data[FIELD_GITHUB_EVENT] = self.event_name
        db_data[FIELD_BRANCH] = self.ref_name
        db_data[FIELD_WORKFLOW_ID] = str(self.run_id)
        db_data[FIELD_COMMIT_SHA] = self.sha

        # Extract PR ID from workflow data if available
        pr_id = "N/A"  # Default to "N/A" for non-PR workflows
        if workflow_data:
            pull_requests = workflow_data.get("pull_requests", [])
            if pull_requests and len(pull_requests) > 0:
                pr_number = pull_requests[0].get("number")
                if pr_number:
                    pr_id = str(pr_number)
        db_data[FIELD_PR_ID] = pr_id

    def add_standardized_timing_fields(
        self,
        db_data: Dict[str, Any],
        creation_time: str,
        start_time: str,
        end_time: str,
        metric_type: str = "workflow",
    ) -> None:
        """Add standardized timing-related fields across all metric types

        Args:
            db_data: Dictionary to add timing fields to
            creation_time: ISO datetime string for creation time
            start_time: ISO datetime string for when execution actually started
            end_time: ISO datetime string for end time
            metric_type: Type of metric ("workflow", "job", "step") for field naming consistency
        """
        # Store original ISO timestamps
        db_data[FIELD_START_TIME] = start_time or ""
        db_data[FIELD_END_TIME] = end_time or ""
        if creation_time:  # Don't add for steps
            db_data[FIELD_CREATION_TIME] = creation_time

        # Duration in integer seconds (using l_ prefix for long type)
        db_data[FIELD_DURATION_SEC] = TimingProcessor.calculate_time_diff(
            start_time, end_time
        )

        # Queue time in integer seconds (using l_ prefix for long type)
        if metric_type != "step":
            db_data[FIELD_QUEUE_TIME] = TimingProcessor.calculate_time_diff(
                creation_time, start_time
            )

        # Use the end_time if available, otherwise use current time
        if end_time:
            # Ensure timestamp is in proper ISO format for OpenSearch date detection
            db_data["@timestamp"] = end_time
        else:
            # Use Z format to match 24h script format
            db_data["@timestamp"] = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

    def post_all_metrics(self) -> None:
        """Upload complete workflow metrics including workflow, jobs, and steps in one operation"""
        print(
            f"Uploading complete metrics for workflow '{self.workflow_name}' (run {self.run_id})"
        )

        max_retries = 1
        retry_delay = 15  # seconds

        for attempt in range(max_retries):
            # Get workflow and jobs data from GitHub API
            workflow_data = self.get_github_api_data(
                f"/repos/{self.repo}/actions/runs/{self.run_id}"
            )
            if not workflow_data:
                print("Could not fetch workflow data from GitHub API")
                return

            jobs_data = self.get_github_api_data(
                f"/repos/{self.repo}/actions/runs/{self.run_id}/jobs"
            )
            if not jobs_data or "jobs" not in jobs_data:
                print("Could not fetch jobs data from GitHub API")
                return

            # Count jobs to process (exclude specified jobs)
            workflow_name = workflow_data.get("name", "")
            jobs_to_process = [
                job
                for job in jobs_data.get("jobs", [])
                if job.get("name") not in EXCLUDED_JOB_NAMES
            ]

            if not jobs_to_process:
                print(
                    f"❌ No jobs to process after excluding jobs: {EXCLUDED_JOB_NAMES}"
                )
                print(
                    f"   Available jobs: {[job.get('name') for job in jobs_data.get('jobs', [])]}"
                )
                return

            print("✅ Processing workflow metrics - proceeding with upload")
            print(f"   Workflow: '{workflow_name}'")
            print(
                f"   Jobs to process: {len(jobs_to_process)} (excluding {EXCLUDED_JOB_NAMES})"
            )
            print(f"   Job names: {[job.get('name') for job in jobs_to_process]}")

            # Check if workflow is completed
            workflow_status = workflow_data.get("status", "")
            workflow_conclusion = workflow_data.get("conclusion")

            if workflow_status == "completed" or workflow_conclusion:
                print(
                    f"Workflow completed with status: {workflow_status}, conclusion: {workflow_conclusion}"
                )
                break
            elif attempt < max_retries - 1:
                print(
                    f"Workflow still {workflow_status}, waiting {retry_delay}s before retry {attempt + 1}/{max_retries}"
                )
                time.sleep(retry_delay)
            else:
                print(
                    f"Workflow still {workflow_status} after {max_retries} attempts, uploading current state"
                )
                break

        # Upload workflow metrics
        try:
            print("Processing workflow metrics...")
            self._upload_workflow_metrics(workflow_data, jobs_data)
            print("Workflow metrics uploaded successfully")
        except Exception as e:
            sanitized_error = self.handle_upload_error(e, "workflow metrics upload")
            print(sanitized_error)

        # Upload all job and step metrics
        try:
            print(f"Processing {len(jobs_data['jobs'])} jobs and their steps...")
            jobs_processed, steps_processed = self._upload_all_job_and_step_metrics(
                jobs_data
            )
            print(
                f"Successfully uploaded {jobs_processed} job metrics and {steps_processed} step metrics"
            )
        except Exception as e:
            sanitized_error = self.handle_upload_error(e, "job/step metrics upload")
            print(sanitized_error)

    def _upload_workflow_metrics(
        self, workflow_data: Dict[str, Any], jobs_data: Dict[str, Any]
    ) -> None:
        """Internal method to upload workflow metrics"""
        db_data = {}
        db_data[FIELD_ID] = f"github-workflow-{self.run_id}"

        # Schema fields
        # Use conclusion for completed workflows, fallback to status
        db_data[FIELD_STATUS] = str(
            workflow_data.get("conclusion") or workflow_data.get("status", "unknown")
        )
        if db_data[FIELD_STATUS] == "success":
            db_data[FIELD_STATUS_NUMBER] = 1
        elif db_data[FIELD_STATUS] == "failure":
            db_data[FIELD_STATUS_NUMBER] = 0
        print(f"Checking branch: {str(workflow_data.get('head_branch'))}")

        # Timing fields
        created_at = workflow_data.get("created_at")
        run_started_at = workflow_data.get("run_started_at")
        end_time = workflow_data.get("completed_at") or workflow_data.get("updated_at")
        self.add_standardized_timing_fields(
            db_data, created_at, run_started_at, end_time, "workflow"
        )

        # Common context fields
        self.add_common_context_fields(db_data, workflow_data)

        # Post to database
        self.post_to_db(self.workflow_index, db_data)

    def _upload_all_job_and_step_metrics(
        self, jobs_data: Dict[str, Any]
    ) -> tuple[int, int]:
        """Internal method to upload all job and step metrics, returns (jobs_processed, steps_processed)"""
        jobs_processed = 0
        steps_processed = 0

        for job in jobs_data["jobs"]:
            try:
                job_name = job.get("name", "")

                # FILTER: Skip excluded jobs to avoid infinite loops and other unwanted jobs
                if job_name in EXCLUDED_JOB_NAMES:
                    print(f"⏭️  Skipping excluded job '{job_name}'")
                    continue

                print(f"📤 Uploading job: '{job_name}'")

                # Upload job metrics
                self._upload_single_job_metrics(job)
                jobs_processed += 1

                # Upload step metrics for this job
                if self.steps_index:
                    step_count = self._upload_job_step_metrics(job)
                    steps_processed += step_count

            except Exception as e:
                print(
                    f"Error uploading metrics for job {job.get('name', 'unknown')}: {e}"
                )
                continue

        return jobs_processed, steps_processed

    def _upload_single_job_metrics(self, job_data: Dict[str, Any]) -> None:
        """Extract and post metrics for a single job"""
        # Extract job metrics using standardized functions
        db_data = {}
        job_id = job_data["id"]
        job_name = job_data["name"]

        db_data[FIELD_ID] = f"github-job-{job_id}"

        # Schema fields
        db_data[FIELD_JOB_ID] = str(job_id)
        # Handle job status - prefer conclusion for completed jobs, fallback to status
        db_data[FIELD_STATUS] = str(
            job_data.get("conclusion") or job_data.get("status") or "unknown"
        )
        if db_data[FIELD_STATUS] == "success":
            db_data[FIELD_STATUS_NUMBER] = 1
        elif db_data[FIELD_STATUS] == "failure":
            db_data[FIELD_STATUS_NUMBER] = 0
        db_data[FIELD_JOB_NAME] = str(job_name)

        # Timing fields
        created_at = job_data.get("created_at")
        started_at = job_data.get("started_at")
        completed_at = job_data.get("completed_at")

        self.add_standardized_timing_fields(
            db_data, created_at, started_at, completed_at, "job"
        )

        # Runner info
        runner_id = job_data.get("runner_id")
        db_data[FIELD_RUNNER_ID] = str(runner_id) if runner_id is not None else ""
        db_data[FIELD_RUNNER_NAME] = str(job_data.get("runner_name", ""))

        # Add common context fields
        self.add_common_context_fields(db_data)
        self.post_to_db(self.jobs_index, db_data)
        print(f"Uploaded metrics for job: {job_name}")

        # Upload container metrics if this is a build job and metrics are available
        # Check if this is one of our framework build jobs
        is_framework_job = any(
            framework in job_name.lower() for framework in FRAMEWORK_IMAGE_BUILD_JOBS
        )

        if is_framework_job:
            self._upload_container_metrics(job_data)
            # Also upload test metrics if available for this framework job
            self._upload_test_metrics(job_data)

    def _upload_job_step_metrics(self, job_data: Dict[str, Any]) -> int:
        """Extract and post metrics for all steps in a job"""
        job_name = job_data["name"]
        steps = job_data.get("steps", [])

        if not steps:
            print(f"No steps found for job {job_name}")
            return 0

        steps_processed = 0
        for step_index, step in enumerate(steps):
            try:
                self._upload_single_step_metrics(step, job_data, step_index)
                steps_processed += 1
            except Exception as e:
                step_name = step.get("name", f"step_{step_index}")
                print(
                    f"Error uploading metrics for step {step_name} in job {job_name}: {e}"
                )
                continue

        print(f"Uploaded metrics for {steps_processed} steps in job {job_name}")
        return steps_processed

    def _upload_single_step_metrics(
        self, step_data: Dict[str, Any], job_data: Dict[str, Any], step_index: int
    ) -> None:
        """Extract and post metrics for a single step"""
        # Extract step metrics using standardized functions
        db_data = {}
        job_id = job_data["id"]
        job_name = job_data["name"]
        step_name = step_data.get("name", f"step_{step_index}")
        step_number = step_data.get("number", step_index + 1)

        # Create unique step ID and use standardized ID generation
        step_id = f"{job_id}_{step_number}"
        db_data[FIELD_ID] = f"github-step-{step_id}"

        # Schema-compliant fields
        db_data[FIELD_STEP_ID] = str(step_id)
        db_data[FIELD_JOB_ID] = str(job_id)
        db_data[FIELD_NAME] = str(step_name)
        db_data[FIELD_STEP_NUMBER] = int(
            step_number
        )  # Using l_ prefix, should be integer
        db_data[FIELD_STATUS] = str(
            step_data.get("conclusion") or step_data.get("status") or "unknown"
        )
        db_data[FIELD_JOB_NAME] = str(job_name)
        if db_data[FIELD_STATUS] == "success":
            db_data[FIELD_STATUS_NUMBER] = 1
        elif db_data[FIELD_STATUS] == "failure":
            db_data[FIELD_STATUS_NUMBER] = 0

        # Timing fields using standardized method - Fix parameter order for steps
        started_at = step_data.get("started_at")
        completed_at = step_data.get("completed_at")

        # For steps: creation_time=None (no queue time), start_time=started_at, end_time=completed_at
        self.add_standardized_timing_fields(
            db_data, None, started_at, completed_at, "step"
        )

        # Command/script executed (GitHub API doesn't always provide this, but we can infer)
        command = ""
        if step_data.get("action"):
            command = f"uses: {step_data['action']}"
        elif "run" in step_name.lower() or "script" in step_name.lower():
            command = (
                "run: <script>"  # GitHub API doesn't expose the actual script content
            )
        db_data[FIELD_COMMAND] = command

        # Add common context fields
        self.add_common_context_fields(db_data)

        # Post to database
        self.post_to_db(self.steps_index, db_data)
        print(f"Uploaded metrics for step: {step_name} (step {step_number})")

    def _upload_container_metrics(
        self, job_data: Dict[str, Any], build_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Upload container-specific metrics to CONTAINER_INDEX"""
        container_index = os.getenv("CONTAINER_INDEX")
        if not container_index:
            print(
                "⚠️  CONTAINER_INDEX not configured, skipping container metrics upload"
            )
            return

        # Get build metrics if not provided
        if build_metrics is None:
            # Try to get framework-specific build metrics based on job name
            job_name = job_data.get("name", "")
            build_metrics = BuildMetricsReader.get_build_metrics_for_job(job_name)

        if not build_metrics:
            print(
                f"⚠️  No build metrics available for container upload for job: {job_data.get('name', 'unknown')}"
            )
            return

        print(f"📦 Uploading container metrics to {container_index}")

        # Create container metrics payload
        container_data = {}

        # Identity & Context - container-specific fields only
        job_id = str(job_data["id"])
        job_name = job_data["name"]
        container_data[
            FIELD_ID
        ] = f"github-container-{job_id}-{build_metrics.get('framework', 'unknown')}"
        container_data[FIELD_JOB_NAME] = str(job_name)
        container_data[FIELD_JOB_ID] = job_id

        # Find the "Build Container" step ID
        build_step_id = None
        steps = job_data.get("steps", [])
        for step in steps:
            if (
                "build" in step.get("name", "").lower()
                and "container" in step.get("name", "").lower()
            ):
                build_step_id = f"{job_id}_{step.get('number', 1)}"
                break

        container_data[FIELD_STEP_ID] = build_step_id or f"{job_id}_build"

        # Status - container-specific
        container_data[FIELD_STATUS] = str(
            job_data.get("conclusion") or job_data.get("status", "unknown")
        )

        # Container Info (only truly container-specific fields)
        container_data[FIELD_BUILD_FRAMEWORK] = build_metrics.get(
            "framework", "unknown"
        )
        container_data[FIELD_BUILD_SIZE_BYTES] = build_metrics.get(
            "image_size_bytes", 0
        )

        # Timing (reusing existing build timing fields)
        if "build_start_time" in build_metrics:
            container_data[FIELD_BUILD_START_TIME] = build_metrics["build_start_time"]
        if "build_end_time" in build_metrics:
            container_data[FIELD_BUILD_END_TIME] = build_metrics["build_end_time"]
        container_data[FIELD_BUILD_DURATION_SEC] = build_metrics.get(
            "build_duration_sec", 0
        )

        # Add @timestamp for time-series data
        container_data["@timestamp"] = build_metrics.get(
            "build_end_time", datetime.now(timezone.utc).isoformat()
        )

        # Add common context fields
        self.add_common_context_fields(container_data)

        # Upload to container index
        try:
            self.post_to_db(container_index, container_data)
            print(
                f"✅ Container metrics uploaded for {build_metrics.get('framework', 'unknown')} framework"
            )
        except Exception as e:
            print(f"❌ Failed to upload container metrics: {e}")

    def _upload_test_metrics(self, job_data: Dict[str, Any]) -> None:
        """Upload individual test metrics by parsing JUnit XML directly from test-results"""
        test_index = os.getenv("TEST_INDEX")
        if not test_index:
            print("⚠️  TEST_INDEX not configured, skipping test metrics upload")
            return

        job_name = job_data.get("name", "")
        job_id = str(job_data["id"])

        # Skip deployment test jobs (No pytest metadata files are created)
        if job_name.lower().startswith("deploy"):
            print(f"⏭️  Skipping test metrics for deployment job '{job_name}'")
            return

        print(f"🧪 Looking for test results for job '{job_name}'")

        # Determine framework from job name to filter metadata files
        framework = None
        job_name_lower = job_name.lower()
        if "vllm" in job_name_lower:
            framework = "vllm"
        elif "sglang" in job_name_lower:
            framework = "sglang"
        elif "trtllm" in job_name_lower:
            framework = "trtllm"

        if not framework:
            print(f"⚠️  Could not determine framework from job name: {job_name}")
            return

        # Determine platform architecture from job name
        # Job names typically look like: "vllm (amd64)" or "sglang (arm64)"
        platform_arch = None
        if "(amd64)" in job_name_lower or "amd64" in job_name_lower:
            platform_arch = "amd64"
        elif "(arm64)" in job_name_lower or "arm64" in job_name_lower:
            platform_arch = "arm64"

        if not platform_arch:
            print(
                f"⚠️  Could not determine platform architecture from job name: {job_name}"
            )
            # Default to amd64 if not specified
            platform_arch = "amd64"
            print(f"   Defaulting to platform_arch: {platform_arch}")

        print(f"📦 Job framework: {framework}, platform_arch: {platform_arch}")

        # Look for test results directory
        test_results_dir = "test-results"
        if not os.path.exists(test_results_dir):
            print(f"⚠️  Test results directory not found: {test_results_dir}")
            return

        # Look for metadata files to get accurate step and framework info
        # Updated pattern to match new unique naming: test_metadata_<framework>_<test_type>_<arch>.json
        # Filter by both framework AND architecture to only process this job's tests
        metadata_files = glob.glob(
            f"{test_results_dir}/test_metadata_{framework}_*_{platform_arch}.json"
        )

        if not metadata_files:
            print(
                f"⚠️  No test metadata files found for framework '{framework}' with arch '{platform_arch}' in {test_results_dir}"
            )
            return

        print(
            f"📄 Found {len(metadata_files)} test metadata files for {framework} ({platform_arch})"
        )

        total_tests_processed = 0

        # Process each metadata file
        for metadata_file in metadata_files:
            try:
                # Read metadata to get accurate step and framework info
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                framework = metadata.get("framework", "unknown")
                test_type = metadata.get("test_type", "unknown")
                step_name = metadata.get("step_name", "Run tests")
                junit_xml_file = metadata.get(
                    "junit_xml_file", "pytest_test_report.xml"
                )

                # Construct step ID from metadata
                test_step_id = f"{job_id}_{step_name.lower().replace(' ', '_')}"

                print("📋 Processing test results:")
                print(f"   Framework: {framework}")
                print(f"   Test Type: {test_type}")
                print(f"   Step Name: {step_name}")
                print(f"   Step ID: {test_step_id}")

                # Find the corresponding XML file
                xml_file = f"{test_results_dir}/{junit_xml_file}"
                if not os.path.exists(xml_file):
                    print(f"⚠️  JUnit XML file not found: {xml_file}")
                    continue

                print(f"📄 Processing JUnit XML: {xml_file}")

                # Parse JUnit XML using xml.etree.ElementTree
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Process each test case
                for testsuite in root.findall(".//testsuite"):
                    for testcase in testsuite.findall("testcase"):
                        # Extract test case information
                        test_classname = testcase.get("classname", "")
                        test_name = testcase.get("name", "")
                        test_time = float(testcase.get("time", 0))
                        test_status = "passed"  # Default status

                        # Create individual test data payload
                        test_data = {}

                        # Identity & Context
                        test_full_name = (
                            f"{test_classname}::{test_name}"
                            if test_classname
                            else test_name
                        )
                        test_data[
                            FIELD_ID
                        ] = f"github-test-{job_id}-{hash(test_full_name) & 0x7FFFFFFF}"  # Use hash for unique ID
                        test_data[FIELD_STEP_ID] = test_step_id
                        test_data[FIELD_JOB_ID] = job_id

                        # Test Info
                        test_data[FIELD_FRAMEWORK] = framework
                        test_data[FIELD_TEST_NAME] = test_name
                        test_data[FIELD_TEST_CLASSNAME] = test_classname
                        test_data[FIELD_TEST_DURATION] = int(
                            test_time * 1000
                        )  # Convert to milliseconds

                        # Check for failure, error, or skipped elements
                        error_msg = ""
                        if testcase.find("failure") is not None:
                            test_status = "failed"
                            failure_elem = testcase.find("failure")
                            error_msg = (
                                failure_elem.get("message", "")
                                if failure_elem is not None
                                else ""
                            )
                            if (
                                not error_msg
                                and failure_elem is not None
                                and failure_elem.text
                            ):
                                error_msg = failure_elem.text
                        elif testcase.find("error") is not None:
                            test_status = "error"
                            error_elem = testcase.find("error")
                            error_msg = (
                                error_elem.get("message", "")
                                if error_elem is not None
                                else ""
                            )
                            if (
                                not error_msg
                                and error_elem is not None
                                and error_elem.text
                            ):
                                error_msg = error_elem.text
                        elif testcase.find("skipped") is not None:
                            test_status = "skipped"
                            skipped_elem = testcase.find("skipped")
                            error_msg = (
                                skipped_elem.get("message", "")
                                if skipped_elem is not None
                                else ""
                            )

                        test_data[FIELD_TEST_STATUS] = test_status
                        test_data[
                            FIELD_STATUS
                        ] = test_status  # Also set general status field

                        if error_msg:
                            test_data[FIELD_ERROR_MESSAGE] = error_msg[
                                :1000
                            ]  # Limit error message length

                        # Add timing (use job completion time as more accurate timestamp)
                        job_completed_at = job_data.get("completed_at")
                        if job_completed_at:
                            test_data["@timestamp"] = job_completed_at
                        else:
                            # Fallback to current time if job completion time not available
                            test_data["@timestamp"] = datetime.now(
                                timezone.utc
                            ).isoformat()

                        # Add common context fields (repo, branch, pr_id, etc.)
                        self.add_common_context_fields(test_data)

                        # Upload individual test
                        try:
                            self.post_to_db(test_index, test_data)
                            print(
                                f"✅ Uploaded test: {test_full_name} ({test_status}, {test_time:.3f}s)"
                            )
                            total_tests_processed += 1
                        except Exception as e:
                            print(f"❌ Failed to upload test {test_full_name}: {e}")

            except Exception as e:
                print(f"❌ Failed to process metadata file {metadata_file}: {e}")

        print(f"📊 Processed {total_tests_processed} individual tests total")
        print("   " + "=" * 50)


def main():
    """Main function to upload complete GitHub Actions workflow metrics"""
    try:
        uploader = WorkflowMetricsUploader()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return

    print(
        f"Processing complete metrics for workflow '{uploader.workflow_name}' (run {uploader.run_id})"
    )

    # Upload all metrics (workflow, jobs, and steps) in one coordinated operation
    uploader.post_all_metrics()


if __name__ == "__main__":
    main()
