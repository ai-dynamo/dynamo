# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Enhanced script to upload complete GitHub Actions workflow and job metrics to Prometheus via OpenTelemetry.
This version runs as the final job in a workflow and captures metrics for 
the entire workflow including all previous jobs.

CONVERSION NOTES:
- Converted from OpenSearch HTTP POST uploads to OpenTelemetry metrics
- Uses OTLP HTTP exporter to send metrics to Prometheus via NVIDIA Observability Service
- Maintains all original metric collection functionality
- Metrics are now recorded as histograms, counters, and gauges with proper labels
- Requires OTEL_AUTH_TOKEN and OTEL_SERVICE_NAME environment variables
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

# Import the working OpenTelemetry client
from otel_metrics_client import OTelMetricsClient
import requests

# OpenTelemetry Configuration - Replace with your actual values
OTLP_ENDPOINT = os.getenv('OTLP_ENDPOINT')
AUTH_TOKEN = os.getenv('OTEL_AUTH_TOKEN') or os.getenv('NVAUTH_TOKEN')  # Use OTEL_AUTH_TOKEN or fallback to NVAUTH_TOKEN
SERVICE_NAME = os.getenv('OTEL_SERVICE_NAME', "Dynamo Metrics")
SERVICE_ID = os.getenv('SERVICE_ID')  # Replace with your Service Registry ID

# FILTERING CONFIGURATION - Only upload data for specific workflow/job combinations
TARGET_WORKFLOW_NAMES = ["Docker Build and Test", "Docs link check"]
TARGET_JOB_NAMES = [
    "vllm",
    "sglang", 
    "trtllm",
    "Check for broken markdown links"
]

# STANDARDIZED FIELD SCHEMA

# Common labels across all metrics
FIELD_USER_ALIAS = "user_alias"
FIELD_REPO = "repo"
FIELD_WORKFLOW_NAME = "workflow_name"
FIELD_GITHUB_EVENT = "github_event"
FIELD_BRANCH = "branch"
FIELD_PR_ID = "pr_id"  # Pull request ID as string ("N/A" if not a PR)
FIELD_PR_TITLE = "pr_title"  # Pull request title
FIELD_PR_FILES_CHANGED = "pr_files_changed"  # Number of files changed in PR
FIELD_PR_FILES_LIST = "pr_files_list"  # List of changed files (first 10)
FIELD_STATUS = "status"
FIELD_STATUS_NUMBER = "status_number"
FIELD_WORKFLOW_ID = "workflow_id"
FIELD_COMMIT_SHA = "commit_sha"

# Timing fields
FIELD_CREATION_TIME = "creation_time"
FIELD_START_TIME = "start_time"
FIELD_END_TIME = "end_time"
FIELD_QUEUE_TIME = "queue_time_sec"
FIELD_DURATION_SEC = "duration_sec"

# Job-specific fields
FIELD_JOB_ID = "job_id"
FIELD_JOB_NAME = "job_name"

# Step-specific fields
FIELD_STEP_ID = "step_id"
FIELD_NAME = "step_name"
FIELD_STEP_NUMBER = "step_number"
FIELD_COMMAND = "command"

# Runner-specific fields
FIELD_RUNNER_PREFIX = "runner_prefix"
FIELD_RUNNER_ID = "runner_id"
FIELD_RUNNER_NAME = "runner_name"

def setup_telemetry():
    """Configure OpenTelemetry using the working OTelMetricsClient"""
    
    # Create the working OpenTelemetry client
    client = OTelMetricsClient(
        endpoint=OTLP_ENDPOINT,
        auth_token=AUTH_TOKEN,
        service_name=SERVICE_NAME,
        service_version="1.0.0",
        meter_name="github.actions.meter"
    )
    
    return client

def status_to_numeric(status: str) -> int:
    """Convert GitHub status string to numeric value for gauge metrics
    
    Returns:
        0: failure
        1: success  
        2: cancelled
        3: skipped
        4: in_progress (or unknown)
    """
    status_lower = status.lower()
    
    if status_lower in ['success', 'completed']:
        return 1
    elif status_lower in ['failure', 'failed']:
        return 0
    elif status_lower in ['cancelled', 'canceled']:
        return 2
    elif status_lower in ['skipped']:
        return 3
    else:  # in_progress, queued, unknown, etc.
        return 4

def extract_runner_prefix(runner_name: str) -> str:
    """Extract runner prefix by removing the final '-*' suffix
    
    Args:
        runner_name: Full runner name (e.g., 'gpu-runner-large-123', 'cpu-runner-01')
        
    Returns:
        Runner prefix (e.g., 'gpu-runner-large', 'cpu-runner')
    """
    if not runner_name:
        return "unknown"
    
    # Split by '-' and remove the last part if it looks like a number/ID
    parts = runner_name.split('-')
    if len(parts) <= 1:
        return runner_name
    
    # Check if the last part looks like a number or ID (digits, alphanumeric, etc.)
    last_part = parts[-1]
    if last_part.isdigit() or (len(last_part) <= 4 and last_part.isalnum()):
        # Remove the last part to get the prefix
        return '-'.join(parts[:-1])
        else:
        # Keep the full name if last part doesn't look like an ID
        return runner_name

class TimingProcessor:
    """Centralized processor for all datetime and duration conversions using Python built-ins"""
    
    @staticmethod
    def _parse_iso(iso_string: str) -> datetime:
        """Parse ISO datetime string using built-in fromisoformat"""
        if not iso_string:
            return None
        try:
            # Handle 'Z' suffix by replacing with '+00:00'
            if iso_string.endswith('Z'):
                iso_string = iso_string[:-1] + '+00:00'
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



class WorkflowMetricsUploader:
    def __init__(self):
        # Setup OpenTelemetry using the working client
        self.client = setup_telemetry()
        
        # Create metric instruments using the working client
        # Performance metrics (for aggregation)
        self.workflow_duration_histogram = self.client.create_histogram(
            name="github_workflow_duration_seconds",
            description="GitHub workflow duration in seconds",
            unit="s"
        )
        
        self.workflow_queue_time_histogram = self.client.create_histogram(
            name="github_workflow_queue_time_seconds", 
            description="GitHub workflow queue time in seconds",
            unit="s"
        )
        
        self.workflow_status_counter = self.client.create_counter(
            name="github_workflow_status",
            description="GitHub workflow status (0=failure, 1=success, 2=cancelled, 3=skipped, 4=in_progress)",
            unit="1"
        )
        
        self.job_duration_histogram = self.client.create_histogram(
            name="github_job_duration_seconds",
            description="GitHub job duration in seconds", 
            unit="s"
        )
        
        self.job_queue_time_histogram = self.client.create_histogram(
            name="github_job_queue_time_seconds",
            description="GitHub job queue time in seconds",
            unit="s"
        )
        
        self.job_status_counter = self.client.create_counter(
            name="github_job_status",
            description="GitHub job status (0=failure, 1=success, 2=cancelled, 3=skipped, 4=in_progress)",
            unit="1"
        )
        
        self.step_duration_histogram = self.client.create_histogram(
            name="github_step_duration_seconds",
            description="GitHub step duration in seconds",
            unit="s"
        )
        
        self.step_status_counter = self.client.create_counter(
            name="github_step_status",
            description="GitHub step status (0=failure, 1=success, 2=cancelled, 3=skipped, 4=in_progress)",
            unit="1"
        )
        
        self.runner_queue_time_histogram = self.client.create_histogram(
            name="github_runner_queue_time_seconds",
            description="GitHub runner queue time in seconds by runner prefix",
            unit="s"
        )
        
        # Info metrics (for metadata storage with rich attributes)
        self.workflow_info_counter = self.client.create_counter(
            name="github_workflow_info",
            description="Workflow metadata information with rich attributes",
            unit="1"
        )
        
        self.job_info_counter = self.client.create_counter(
            name="github_job_info", 
            description="Job metadata information with rich attributes",
            unit="1"
        )
        
        self.step_info_counter = self.client.create_counter(
            name="github_step_info",
            description="Step metadata information with rich attributes", 
            unit="1"
        )
        
        # Get current workflow information
        self.repo = os.getenv('GITHUB_REPOSITORY')
        self.run_id = os.getenv('GITHUB_RUN_ID')
        self.workflow_name = os.getenv('GITHUB_WORKFLOW')
        self.actor = os.getenv('GITHUB_ACTOR')
        self.event_name = os.getenv('GITHUB_EVENT_NAME')
        self.ref = os.getenv('GITHUB_REF')
        self.ref_name = os.getenv('GITHUB_REF_NAME')
        self.sha = os.getenv('GITHUB_SHA')
        
        if not self.repo or not self.run_id:
            raise ValueError("Missing required GitHub environment variables")
        
        print(f"Uploading metrics for workflow '{self.workflow_name}' (run {self.run_id}) in {self.repo}")
        
    def handle_upload_error(self, error: Exception, operation: str) -> str:
        """Centralized error handling for telemetry operations
        
        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
            
        Returns:
            Error message for telemetry operations
        """
        error_msg = str(error)
        return f"Error during {operation}: {error_msg}"

    def record_workflow_metrics(self, data: Dict[str, Any]) -> None:
        """Record workflow metrics using OpenTelemetry with annotations"""
        print(f"Recording workflow metrics via OpenTelemetry: {data.get('_id', 'unknown')}")
        
        # Get clean labels for aggregation (no details to avoid cardinality explosion)
        labels = self.get_common_labels(data)
        
        # Log detailed information for debugging (not as metric labels)
        detailed_info = {
            "workflow_id": data.get(FIELD_WORKFLOW_ID, str(self.run_id)),
            "commit_sha": data.get(FIELD_COMMIT_SHA, "")[:12],
            "pr_id": data.get(FIELD_PR_ID, "N/A"),
            "branch": data.get(FIELD_BRANCH, "")
        }
        print(f"   📋 Workflow details: {detailed_info}")
        
        # Record duration
        duration = data.get(FIELD_DURATION_SEC, 0)
        if duration > 0:
            self.workflow_duration_histogram.record(duration, labels)
            
        # Record queue time
        queue_time = data.get(FIELD_QUEUE_TIME, 0)
        if queue_time > 0:
            self.workflow_queue_time_histogram.record(queue_time, labels)
            
        # Record status as numeric counter
        status_numeric = status_to_numeric(data.get(FIELD_STATUS, "unknown"))
        self.workflow_status_counter.add(status_numeric, labels)
        
        print(f"Successfully recorded workflow metrics for: {data.get('_id', 'unknown')}")
    
    def record_job_metrics(self, data: Dict[str, Any]) -> None:
        """Record job metrics using OpenTelemetry attributes pattern"""
        print(f"Recording job metrics via OpenTelemetry: {data.get('_id', 'unknown')}")
        
        # Performance metrics with minimal labels for aggregation
        perf_labels = {
            FIELD_JOB_NAME: data.get(FIELD_JOB_NAME, ""),
            FIELD_BRANCH: data.get(FIELD_BRANCH, ""),
            FIELD_STATUS: data.get(FIELD_STATUS, "unknown")
        }
        
        # Info metric with rich attributes for metadata
        info_attributes = {
            "workflow_id": data.get(FIELD_WORKFLOW_ID, str(self.run_id)),
            "commit_sha": data.get(FIELD_COMMIT_SHA, "")[:12],
            "pr_id": data.get(FIELD_PR_ID, "N/A"),
            "pr_title": data.get(FIELD_PR_TITLE, "N/A")[:100],
            "job_id": data.get(FIELD_JOB_ID, ""),
            "job_name": data.get(FIELD_JOB_NAME, ""),
            "branch": data.get(FIELD_BRANCH, ""),
            "status": data.get(FIELD_STATUS, "unknown"),
            "repo": data.get(FIELD_REPO, self.repo),
            "workflow_name": data.get(FIELD_WORKFLOW_NAME, self.workflow_name),
            "github_event": data.get(FIELD_GITHUB_EVENT, self.event_name),
            "user_alias": data.get(FIELD_USER_ALIAS, self.actor),
            "runner_name": data.get(FIELD_RUNNER_NAME, ""),
            "runner_prefix": data.get(FIELD_RUNNER_PREFIX, extract_runner_prefix(data.get(FIELD_RUNNER_NAME, ""))),
            "creation_time": data.get(FIELD_CREATION_TIME, ""),
            "start_time": data.get(FIELD_START_TIME, ""),
            "end_time": data.get(FIELD_END_TIME, ""),
            "duration_sec": str(data.get(FIELD_DURATION_SEC, 0)),
            "queue_time_sec": str(data.get(FIELD_QUEUE_TIME, 0))
        }
        
        # Record info metric (always value=1) with rich attributes
        self.job_info_counter.add(1, info_attributes)
        
        # Record performance metrics with minimal labels
        duration = data.get(FIELD_DURATION_SEC, 0)
        if duration > 0:
            self.job_duration_histogram.record(duration, perf_labels)
            
        queue_time = data.get(FIELD_QUEUE_TIME, 0)
        if queue_time > 0:
            self.job_queue_time_histogram.record(queue_time, perf_labels)
            
        # Record status as numeric counter
        status_numeric = status_to_numeric(data.get(FIELD_STATUS, "unknown"))
        self.job_status_counter.add(status_numeric, perf_labels)
        
        print(f"Successfully recorded job metrics for: {data.get(FIELD_JOB_NAME, 'unknown')}")
        print(f"   📊 Performance labels: {perf_labels}")
        print(f"   📋 Info attributes: {len(info_attributes)} fields")
    
    def record_step_metrics(self, data: Dict[str, Any]) -> None:
        """Record step metrics using OpenTelemetry attributes pattern"""
        print(f"Recording step metrics via OpenTelemetry: {data.get('_id', 'unknown')}")
        
        # Performance metrics with minimal labels for aggregation
        perf_labels = {
            FIELD_JOB_NAME: data.get(FIELD_JOB_NAME, ""),
            FIELD_NAME: data.get(FIELD_NAME, ""),  # step_name
            FIELD_BRANCH: data.get(FIELD_BRANCH, ""),
            FIELD_STATUS: data.get(FIELD_STATUS, "unknown")
        }
        
        # Info metric with rich attributes for metadata
        info_attributes = {
            "workflow_id": data.get(FIELD_WORKFLOW_ID, str(self.run_id)),
            "step_id": data.get(FIELD_STEP_ID, ""),
            "step_name": data.get(FIELD_NAME, ""),
            "step_number": str(data.get(FIELD_STEP_NUMBER, 0)),
            "job_id": data.get(FIELD_JOB_ID, ""),
            "job_name": data.get(FIELD_JOB_NAME, ""),
            "branch": data.get(FIELD_BRANCH, ""),
            "status": data.get(FIELD_STATUS, "unknown"),
            "command": data.get(FIELD_COMMAND, "")[:200],  # Truncate long commands
            "repo": data.get(FIELD_REPO, self.repo),
            "workflow_name": data.get(FIELD_WORKFLOW_NAME, self.workflow_name),
            "github_event": data.get(FIELD_GITHUB_EVENT, self.event_name),
            "user_alias": data.get(FIELD_USER_ALIAS, self.actor),
            "commit_sha": data.get(FIELD_COMMIT_SHA, "")[:12],
            "pr_id": data.get(FIELD_PR_ID, "N/A"),
            "creation_time": data.get(FIELD_CREATION_TIME, ""),
            "start_time": data.get(FIELD_START_TIME, ""),
            "end_time": data.get(FIELD_END_TIME, ""),
            "duration_sec": str(data.get(FIELD_DURATION_SEC, 0))
        }
        
        # Record info metric (always value=1) with rich attributes
        self.step_info_counter.add(1, info_attributes)
        
        # Record performance metrics with minimal labels
        duration = data.get(FIELD_DURATION_SEC, 0)
        if duration > 0:
            self.step_duration_histogram.record(duration, perf_labels)
            
        # Record status as numeric counter
        status_numeric = status_to_numeric(data.get(FIELD_STATUS, "unknown"))
        self.step_status_counter.add(status_numeric, perf_labels)
        
        print(f"Successfully recorded step metrics for: {data.get(FIELD_NAME, 'unknown')}")
        print(f"   📊 Performance labels: {perf_labels}")
        print(f"   📋 Info attributes: {len(info_attributes)} fields")
    
    def record_runner_queue_metrics(self, data: Dict[str, Any]) -> None:
        """Record runner queue time metrics using OpenTelemetry with annotations"""
        runner_name = data.get(FIELD_RUNNER_NAME, "")
        queue_time = data.get(FIELD_QUEUE_TIME, 0)
        
        if not runner_name or queue_time <= 0:
            return  # Skip if no runner info or no queue time
        
        print(f"Recording runner queue metrics via OpenTelemetry: runner={runner_name}, queue_time={queue_time}s")
        
        # Extract runner prefix
        runner_prefix = extract_runner_prefix(runner_name)
        
        # Use only most relevant label for runner metrics: runner_prefix
        # NO details annotation to avoid cardinality explosion
        labels = {
            FIELD_RUNNER_PREFIX: runner_prefix
        }
        
        # Log detailed information for debugging (not as metric labels)
        detailed_info = {
            "workflow_id": data.get(FIELD_WORKFLOW_ID, str(self.run_id)),
            "job_name": data.get(FIELD_JOB_NAME, ""),
            "runner_name": runner_name,
            "runner_id": data.get(FIELD_RUNNER_ID, ""),
            "queue_time_sec": queue_time
        }
        print(f"   📋 Runner details: {detailed_info}")
        
        # Record runner queue time
        self.runner_queue_time_histogram.record(queue_time, labels)
        
        print(f"Successfully recorded runner queue metrics for: {runner_prefix} (queue_time={queue_time}s)")

    def get_github_api_data(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Fetch data from GitHub API"""
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            print("Error: No GitHub token found. Set GITHUB_TOKEN environment variable or repository secret.")
            return None
            
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            response = requests.get(f"https://api.github.com{endpoint}", headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching GitHub API data from {endpoint}: {e}")
            return None

    def get_pr_details(self, pr_number: str) -> Dict[str, Any]:
        """Fetch PR title and file changes from GitHub API"""
        if pr_number == "N/A" or not pr_number:
            return {
                "title": "N/A",
                "files_changed": 0,
                "files_list": []
            }
        
        try:
            # Get PR details for title
            pr_data = self.get_github_api_data(f"/repos/{self.repo}/pulls/{pr_number}")
            if not pr_data:
                return {"title": "N/A", "files_changed": 0, "files_list": []}
            
            pr_title = pr_data.get('title', 'N/A')
            
            # Get PR files
            files_data = self.get_github_api_data(f"/repos/{self.repo}/pulls/{pr_number}/files")
            if not files_data:
                return {"title": pr_title, "files_changed": 0, "files_list": []}
            
            files_changed = len(files_data)
            # Get first 10 files to avoid too much data
            files_list = [f.get('filename', '') for f in files_data[:10]]
            
            return {
                "title": pr_title,
                "files_changed": files_changed,
                "files_list": files_list
            }
            
        except Exception as e:
            print(f"Error fetching PR details for PR #{pr_number}: {e}")
            return {"title": "N/A", "files_changed": 0, "files_list": []}

    def add_common_context_fields(self, db_data: Dict[str, Any], workflow_data: Optional[Dict[str, Any]] = None) -> None:
        """Add common context fields used across all metric types"""
        db_data[FIELD_USER_ALIAS] = self.actor
        db_data[FIELD_REPO] = self.repo
        db_data[FIELD_WORKFLOW_NAME] = self.workflow_name
        db_data[FIELD_GITHUB_EVENT] = self.event_name
        db_data[FIELD_BRANCH] = self.ref_name
        db_data[FIELD_WORKFLOW_ID] = str(self.run_id)
        db_data[FIELD_COMMIT_SHA] = self.sha
        
        # Extract PR ID using multiple methods for better accuracy
        pr_id = "N/A"  # Default to "N/A" for non-PR workflows
        
        # Method 1: Check if this is a pull_request event (most reliable)
        if self.event_name == 'pull_request':
            # For pull_request events, try to get PR number from environment
            github_event_path = os.getenv('GITHUB_EVENT_PATH')
            if github_event_path:
                try:
                    with open(github_event_path, 'r') as f:
                        event_data = json.load(f)
                        pr_number = event_data.get('number')
                        if pr_number:
                            pr_id = str(pr_number)
                            print(f"Found PR number from event context: {pr_id}")
                except Exception as e:
                    print(f"Could not read event data: {e}")
        
        # Method 2: Check workflow data pull_requests array (fallback)
        if pr_id == "N/A" and workflow_data:
            pull_requests = workflow_data.get('pull_requests', [])
            if pull_requests and len(pull_requests) > 0:
                pr_number = pull_requests[0].get('number')
                if pr_number:
                    pr_id = str(pr_number)
                    print(f"Found PR number from workflow data: {pr_id}")
        
        # Method 3: Try to find associated PR via commit API (for any event type)
        if pr_id == "N/A" and self.sha:
            try:
                associated_prs = self.get_github_api_data(f"/repos/{self.repo}/commits/{self.sha}/pulls")
                if associated_prs and len(associated_prs) > 0:
                    pr_number = associated_prs[0].get('number')
                    if pr_number:
                        pr_id = str(pr_number)
                        print(f"Found PR number from commit association: {pr_id} (event: {self.event_name})")
            except Exception as e:
                print(f"Could not fetch associated PRs for commit {self.sha}: {e}")
        
        print(f"Final PR ID: {pr_id} (event: {self.event_name})")
        
        db_data[FIELD_PR_ID] = pr_id
        
        # Get PR details if this is a PR workflow
        if pr_id != "N/A":
            pr_details = self.get_pr_details(pr_id)
            db_data[FIELD_PR_TITLE] = pr_details["title"]
            db_data[FIELD_PR_FILES_CHANGED] = pr_details["files_changed"]
            db_data[FIELD_PR_FILES_LIST] = ",".join(pr_details["files_list"][:5])  # First 5 files as comma-separated string
        else:
            db_data[FIELD_PR_TITLE] = "N/A"
            db_data[FIELD_PR_FILES_CHANGED] = 0
            db_data[FIELD_PR_FILES_LIST] = "N/A"

    def get_common_labels(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Get clean common labels for aggregation (minimal cardinality)"""
        return {
            FIELD_REPO: data.get(FIELD_REPO, self.repo),
            FIELD_WORKFLOW_NAME: data.get(FIELD_WORKFLOW_NAME, self.workflow_name),
            FIELD_STATUS: data.get(FIELD_STATUS, "unknown"),
            FIELD_GITHUB_EVENT: data.get(FIELD_GITHUB_EVENT, self.event_name),
            FIELD_USER_ALIAS: data.get(FIELD_USER_ALIAS, self.actor)
            # Removed high-cardinality fields: workflow_id, commit_sha, pr_id, branch
        }
    
    def get_detailed_annotations(self, data: Dict[str, Any]) -> str:
        """Get detailed information as JSON annotation for debugging"""
        import json
        import time
        
        annotations = {
            # Unique identifiers (moved from labels)
            "workflow_id": data.get(FIELD_WORKFLOW_ID, str(self.run_id)),
            "commit_sha": data.get(FIELD_COMMIT_SHA, "")[:12],  # Short SHA
            "branch": data.get(FIELD_BRANCH, ""),
            "pr_id": data.get(FIELD_PR_ID, "N/A"),
            "pr_title": data.get(FIELD_PR_TITLE, "N/A")[:100],  # Truncated
            "pr_files_changed": data.get(FIELD_PR_FILES_CHANGED, 0),
            "pr_files_list": data.get(FIELD_PR_FILES_LIST, "N/A")[:200],  # Truncated
            
            # Timing information
            "creation_time": data.get(FIELD_CREATION_TIME, ""),
            "start_time": data.get(FIELD_START_TIME, ""),
            "end_time": data.get(FIELD_END_TIME, ""),
            "duration_sec": data.get(FIELD_DURATION_SEC, 0),
            "queue_time_sec": data.get(FIELD_QUEUE_TIME, 0),
            
            # Metadata
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }
        
        # Remove empty/null values to keep annotation compact
        annotations = {k: v for k, v in annotations.items() if v and v != "N/A" and v != ""}
        
        return json.dumps(annotations, separators=(',', ':'))

    def add_standardized_timing_fields(self, db_data: Dict[str, Any], creation_time: str, start_time: str, end_time: str, 
                                     metric_type: str = "workflow") -> None:
        """Add standardized timing-related fields across all metric types
        
        Args:
            db_data: Dictionary to add timing fields to
            creation_time: ISO datetime string for creation time
            start_time: ISO datetime string for when execution actually started
            end_time: ISO datetime string for end time  
            metric_type: Type of metric ("workflow", "job", "step") for field naming consistency
        """
        # Store original ISO timestamps
        db_data[FIELD_START_TIME] = start_time or ''
        db_data[FIELD_END_TIME] = end_time or ''
        if creation_time: #Don't add for steps
            db_data[FIELD_CREATION_TIME] = creation_time
        
        # Duration in integer seconds
        db_data[FIELD_DURATION_SEC] = TimingProcessor.calculate_time_diff(start_time, end_time)
        
        # Queue time in integer seconds
        if metric_type != "step":
            db_data[FIELD_QUEUE_TIME] = TimingProcessor.calculate_time_diff(creation_time, start_time)
        
        # Use the end_time if available, otherwise use current time
        if end_time:
            # Ensure timestamp is in proper ISO format for OpenSearch date detection
            db_data['@timestamp'] = end_time
        else:
            # Use Z format to match 24h script format
            db_data['@timestamp'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    def post_all_metrics(self) -> None:
        """Upload complete workflow metrics including workflow, jobs, and steps in one operation"""
        print(f"Uploading complete metrics for workflow '{self.workflow_name}' (run {self.run_id})")
        
        # Wait for workflow to complete before uploading metrics
        import time
        max_retries = 1
        retry_delay = 15  # seconds
        
        for attempt in range(max_retries):
            # Get workflow and jobs data from GitHub API
            workflow_data = self.get_github_api_data(f"/repos/{self.repo}/actions/runs/{self.run_id}")
            if not workflow_data:
                print("Could not fetch workflow data from GitHub API")
                return
                
            jobs_data = self.get_github_api_data(f"/repos/{self.repo}/actions/runs/{self.run_id}/jobs")
            if not jobs_data or 'jobs' not in jobs_data:
                print("Could not fetch jobs data from GitHub API")
                return
            
            # FILTER: Only upload data for specific workflow and job
            workflow_name = workflow_data.get('name', '')
            
            if workflow_name not in TARGET_WORKFLOW_NAMES:
                print(f"❌ Skipping upload - Workflow '{workflow_name}' does not match target '{TARGET_WORKFLOW_NAMES}'")
                return
            
            # Check if any target jobs exist in this workflow
            target_jobs_found = []
            available_jobs = [job.get('name') for job in jobs_data.get('jobs', [])]
            
            for job in jobs_data.get('jobs', []):
                job_name = job.get('name')
                if job_name in TARGET_JOB_NAMES:
                    target_jobs_found.append(job_name)
            
            if not target_jobs_found:
                print(f"❌ Skipping upload - No target jobs found in workflow")
                print(f"   Target jobs: {TARGET_JOB_NAMES}")
                print(f"   Available jobs: {available_jobs}")
                return
            
            print(f"✅ Workflow and job match criteria - proceeding with upload")
            print(f"   Workflow: '{workflow_name}'")
            print(f"   Target jobs found: {target_jobs_found}")
            
            # Check if workflow is completed
            workflow_status = workflow_data.get('status', '')
            workflow_conclusion = workflow_data.get('conclusion')
            
            if workflow_status == 'completed' or workflow_conclusion:
                print(f"Workflow completed with status: {workflow_status}, conclusion: {workflow_conclusion}")
                break
            elif attempt < max_retries - 1:
                print(f"Workflow still {workflow_status}, waiting {retry_delay}s before retry {attempt + 1}/{max_retries}")
                time.sleep(retry_delay)
            else:
                print(f"Workflow still {workflow_status} after {max_retries} attempts, uploading current state")
                break
        
        # Upload workflow metrics
        try:
            print("Processing workflow metrics...")
            self._upload_workflow_metrics(workflow_data, jobs_data)
            print("Workflow metrics uploaded successfully")
        except Exception as e:
            error_msg = self.handle_upload_error(e, "workflow metrics recording")
            print(error_msg)
        
        # Upload all job and step metrics
        try:
            print(f"Processing {len(jobs_data['jobs'])} jobs and their steps...")
            jobs_processed, steps_processed = self._upload_all_job_and_step_metrics(jobs_data)
            print(f"Successfully uploaded {jobs_processed} job metrics and {steps_processed} step metrics")
        except Exception as e:
            error_msg = self.handle_upload_error(e, "job/step metrics recording")
            print(error_msg)

    def _upload_workflow_metrics(self, workflow_data: Dict[str, Any], jobs_data: Dict[str, Any]) -> None:
        """Internal method to upload workflow metrics"""
        db_data = {}
        
        # Schema fields
        # Use conclusion for completed workflows, fallback to status
        db_data[FIELD_STATUS] = str(workflow_data.get('conclusion') or workflow_data.get('status', 'unknown'))
        if db_data[FIELD_STATUS] == "success":
            db_data[FIELD_STATUS_NUMBER] = 1
        elif db_data[FIELD_STATUS] == "failure":
            db_data[FIELD_STATUS_NUMBER] = 0
        print(f"Checking branch: {str(workflow_data.get('head_branch'))}")
        
        # Timing fields
        created_at = workflow_data.get('created_at')
        run_started_at = workflow_data.get('run_started_at')
        end_time = workflow_data.get('completed_at') or workflow_data.get('updated_at')
        self.add_standardized_timing_fields(db_data, created_at, run_started_at, end_time, "workflow")
        
        # Common context fields
        self.add_common_context_fields(db_data, workflow_data)
        
        self.record_workflow_metrics(db_data)

    def _upload_all_job_and_step_metrics(self, jobs_data: Dict[str, Any]) -> tuple[int, int]:
        """Internal method to upload all job and step metrics, returns (jobs_processed, steps_processed)"""
        jobs_processed = 0
        steps_processed = 0
        
        for job in jobs_data['jobs']:
            try:
                job_name = job.get('name', '')
                
                # FILTER: Only upload target jobs
                if job_name not in TARGET_JOB_NAMES:
                    print(f"⏭️  Skipping job '{job_name}' - not in target jobs {TARGET_JOB_NAMES}")
                    continue
                
                print(f"📤 Uploading target job: '{job_name}'")
                
                # Upload job metrics
                self._upload_single_job_metrics(job)
                jobs_processed += 1
                
                # Upload step metrics for this job
                    step_count = self._upload_job_step_metrics(job)
                    steps_processed += step_count
                    
            except Exception as e:
                print(f"Error uploading metrics for job {job.get('name', 'unknown')}: {e}")
                continue
        
        return jobs_processed, steps_processed

    def _upload_single_job_metrics(self, job_data: Dict[str, Any]) -> None:
        """Extract and post metrics for a single job"""
        # Extract job metrics using standardized functions
        db_data = {}
        job_id = job_data['id']
        job_name = job_data['name']
        
        # Schema fields
        db_data[FIELD_JOB_ID] = str(job_id)
        # Handle job status - prefer conclusion for completed jobs, fallback to status
        db_data[FIELD_STATUS] = str(job_data.get('conclusion') or job_data.get('status') or 'unknown')
        if db_data[FIELD_STATUS] == "success":
            db_data[FIELD_STATUS_NUMBER] = 1
        elif db_data[FIELD_STATUS] == "failure":
            db_data[FIELD_STATUS_NUMBER] = 0
        db_data[FIELD_JOB_NAME] = str(job_name)
                
        # Timing fields
        created_at = job_data.get('created_at')
        started_at = job_data.get('started_at')
        completed_at = job_data.get('completed_at')
        
        self.add_standardized_timing_fields(db_data, created_at, started_at, completed_at, "job")
        
        # Runner info
        runner_id = job_data.get('runner_id')
        runner_name = str(job_data.get('runner_name', ''))
        db_data[FIELD_RUNNER_ID] = str(runner_id) if runner_id is not None else ''
        db_data[FIELD_RUNNER_NAME] = runner_name
        db_data[FIELD_RUNNER_PREFIX] = extract_runner_prefix(runner_name)
        
        # Add common context fields
        self.add_common_context_fields(db_data, None)
        
        self.record_job_metrics(db_data)
        
        # Also record runner queue time metrics if we have runner info
        self.record_runner_queue_metrics(db_data)
        
        print(f"Uploaded metrics for job: {job_name}")

    def _upload_job_step_metrics(self, job_data: Dict[str, Any]) -> int:
        """Extract and post metrics for all steps in a job"""
        job_name = job_data['name']
        steps = job_data.get('steps', [])
        
        if not steps:
            print(f"No steps found for job {job_name}")
            return 0
        
        steps_processed = 0
        for step_index, step in enumerate(steps):
            try:
                self._upload_single_step_metrics(step, job_data, step_index)
                steps_processed += 1
            except Exception as e:
                step_name = step.get('name', f'step_{step_index}')
                print(f"Error uploading metrics for step {step_name} in job {job_name}: {e}")
                continue
        
        print(f"Uploaded metrics for {steps_processed} steps in job {job_name}")
        return steps_processed

    def _upload_single_step_metrics(self, step_data: Dict[str, Any], job_data: Dict[str, Any], step_index: int) -> None:
        """Extract and post metrics for a single step"""
        # Extract step metrics using standardized functions
        db_data = {}
        job_id = job_data['id']
        job_name = job_data['name']
        step_name = step_data.get('name', f'step_{step_index}')
        step_number = step_data.get('number', step_index + 1)
        
        # Generate step ID from available data (steps don't always have IDs in GitHub API)
        step_id = step_data.get('id') or f"{job_id}-step-{step_number}"
        
        # Schema-compliant fields
        db_data[FIELD_STEP_ID] = str(step_id)
        db_data[FIELD_JOB_ID] = str(job_id)
        db_data[FIELD_NAME] = str(step_name)
        db_data[FIELD_STEP_NUMBER] = int(step_number)
        db_data[FIELD_STATUS] = str(step_data.get('conclusion') or step_data.get('status') or 'unknown')
        db_data[FIELD_JOB_NAME] = str(job_name)
        if db_data[FIELD_STATUS] == "success":
            db_data[FIELD_STATUS_NUMBER] = 1
        elif db_data[FIELD_STATUS] == "failure":
            db_data[FIELD_STATUS_NUMBER] = 0
        
        # Timing fields using standardized method - Fix parameter order for steps
        started_at = step_data.get('started_at')
        completed_at = step_data.get('completed_at')
        
        # For steps: creation_time=None (no queue time), start_time=started_at, end_time=completed_at
        self.add_standardized_timing_fields(db_data, None, started_at, completed_at, "step")
        
        # Command/script executed (GitHub API doesn't always provide this, but we can infer)
        command = ""
        if step_data.get('action'):
            command = f"uses: {step_data['action']}"
        elif 'run' in step_name.lower() or 'script' in step_name.lower():
            command = "run: <script>"  # GitHub API doesn't expose the actual script content
        db_data[FIELD_COMMAND] = command
        
        # Add common context fields
        self.add_common_context_fields(db_data, None)
        
        self.record_step_metrics(db_data)
        print(f"Uploaded metrics for step: {step_name} (step {step_number})")

def main():
    """Main function to upload complete GitHub Actions workflow metrics"""
    try:
        uploader = WorkflowMetricsUploader()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    
    print(f"Processing complete metrics for workflow '{uploader.workflow_name}' (run {uploader.run_id})")
    
    # Upload all metrics (workflow, jobs, and steps) in one coordinated operation
    uploader.post_all_metrics()
    
    # Force final export of all metrics using the working client
    print("Flushing metrics to ensure delivery...")
    uploader.client.flush()
    print("All metrics have been sent to Prometheus via OpenTelemetry!")

if __name__ == "__main__":
    main()
