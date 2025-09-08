#!/usr/bin/env python3
"""
Enhanced script to upload complete GitHub Actions workflow and job metrics.
This version runs as the final job in a workflow and captures metrics for 
the entire workflow including all previous jobs.
"""

import os
import sys
import json
import requests
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
import re

# Schema Field Constants - Centralized field names for consistency
# Common fields across all metric types
FIELD_ID = "_id"
FIELD_USER_ALIAS = "userAlias"
FIELD_REPO = "repo"
FIELD_WORKFLOW_NAME = "workflowName"
FIELD_GITHUB_EVENT = "githubEvent"
FIELD_BRANCH = "branch"
FIELD_STATUS = "status"
FIELD_EVENT = "event"

# Timing fields
FIELD_CREATION_TIME = "creationTime"
FIELD_START_TIME = "startTime"
FIELD_END_TIME = "endTime"
FIELD_QUEUE_TIME_SEC = "queueTime"
FIELD_DURATION_SEC = "duration"

# Workflow-specific fields
FIELD_WORKFLOW_ID = "workflowId"
FIELD_COMMIT_SHA = "commitSha"
FIELD_JOBS = "jobs"

# Job-specific fields
FIELD_JOB_ID = "jobId"
FIELD_JOB_NAME = "jobName"
FIELD_RUNNER_INFO = "runnerInfo"
FIELD_RUNNER_ID = "runnerId"
FIELD_RUNNER_NAME = "runnerName"
FIELD_WORKFLOW_SOURCE = "workflowSource"
FIELD_LABELS = "labels"
FIELD_STEPS = "steps"

# Step-specific fields
FIELD_STEP_ID = "stepId"
FIELD_NAME = "stepName"
FIELD_ORDER = "order"
FIELD_COMMAND = "command"
FIELD_JOB_LABELS = "jobLabels"

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
    def _seconds_to_hms(total_seconds: float) -> str:
        """Convert seconds to HH:MM:SS using built-in divmod"""
        if total_seconds is None or total_seconds < 0:
            return "00:00:00"
        
        # Use divmod for clean hour/minute/second calculation
        hours, remainder = divmod(int(total_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    @staticmethod
    def to_time_format(iso_string: str) -> str:
        """Convert ISO datetime string to HH:MM:SS format"""
        if not iso_string:
            return None
        dt = TimingProcessor._parse_iso(iso_string)
        return dt.strftime('%H:%M:%S') if dt else None
    
    @staticmethod
    def calculate_time_diff(start_time: str, end_time: str) -> str:
        """Calculate duration/queue time using built-in datetime subtraction"""
        if not start_time or not end_time:
            return "00:00:00"
        
        start_dt = TimingProcessor._parse_iso(start_time)
        end_dt = TimingProcessor._parse_iso(end_time)
        
        if not start_dt or not end_dt:
            return "00:00:00"
        
        # Use built-in timedelta.total_seconds()
        duration = end_dt - start_dt
        return TimingProcessor._seconds_to_hms(duration.total_seconds())


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
        self.workflow_index = os.getenv('WORKFLOW_INDEX', '')
        self.jobs_index = os.getenv('JOB_INDEX', '')
        self.steps_index = os.getenv('STEPS_INDEX', '')
        
        # Validate that database URLs are provided
        if not self.workflow_index or not self.jobs_index or not self.steps_index:
            raise ValueError(
                "Database URLs not configured. Please set environment variables:\n"
                "  WORKFLOW_INDEX - URL for workflow metrics\n"
                "  JOB_INDEX - URL for job metrics\n"
                "  STEPS_INDEX - URL for step metrics"
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
        print(f"Posting metrics to database...")
        try:
            response = requests.post(url, data=json.dumps(data), headers=self.headers, timeout=30)
            if not (200 <= response.status_code < 300):
                raise ValueError(f"Error posting to DB: HTTP {response.status_code}")
            print(f"Successfully posted metrics with ID: {data.get('_id', 'unknown')}")
        except requests.exceptions.RequestException as e:
            # Use centralized error handling
            sanitized_error = self.handle_upload_error(e, "database upload")
            raise ValueError(sanitized_error)

    def get_github_api_data(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Fetch data from GitHub API"""
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            print("Warning: GITHUB_TOKEN not set, skipping API calls")
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


    def add_common_context_fields(self, db_data: Dict[str, Any]) -> None:
        """Add common context fields used across all metric types"""
        db_data[FIELD_USER_ALIAS] = self.actor
        db_data[FIELD_REPO] = self.repo
        db_data[FIELD_WORKFLOW_NAME] = self.workflow_name
        db_data[FIELD_GITHUB_EVENT] = self.event_name
        db_data[FIELD_BRANCH] = self.ref_name

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
        # Creation, Start, End time in HH:MM:SS format
        db_data[FIELD_CREATION_TIME] = TimingProcessor.to_time_format(creation_time)
        db_data[FIELD_START_TIME] = TimingProcessor.to_time_format(start_time)
        db_data[FIELD_END_TIME] = TimingProcessor.to_time_format(end_time)
        
        # Duration in HH:MM:SS format (consistent across all types)
        db_data[FIELD_DURATION_SEC] = TimingProcessor.calculate_time_diff(start_time, end_time)
        
        # Queue time in HH:MM:SS format only for workflows/jobs
        if metric_type != "step":
            db_data[FIELD_QUEUE_TIME_SEC] = TimingProcessor.calculate_time_diff(creation_time, start_time)

    def post_all_metrics(self) -> None:
        """Upload complete workflow metrics including workflow, jobs, and steps in one operation"""
        print(f"Uploading complete metrics for workflow '{self.workflow_name}' (run {self.run_id})")
        
        # Get workflow and jobs data from GitHub API
        workflow_data = self.get_github_api_data(f"/repos/{self.repo}/actions/runs/{self.run_id}")
        if not workflow_data:
            print("Could not fetch workflow data from GitHub API")
            return
            
        jobs_data = self.get_github_api_data(f"/repos/{self.repo}/actions/runs/{self.run_id}/jobs")
        if not jobs_data or 'jobs' not in jobs_data:
            print("Could not fetch jobs data from GitHub API")
            return
        
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
            jobs_processed, steps_processed = self._upload_all_job_and_step_metrics(jobs_data)
            print(f"Successfully uploaded {jobs_processed} job metrics and {steps_processed} step metrics")
        except Exception as e:
            sanitized_error = self.handle_upload_error(e, "job/step metrics upload")
            print(sanitized_error)

    def _upload_workflow_metrics(self, workflow_data: Dict[str, Any], jobs_data: Dict[str, Any]) -> None:
        """Internal method to upload workflow metrics"""
        db_data = {}
        db_data[FIELD_ID] = f"github_{self.run_id}_{self.repo.replace('/', '_')}"
        
        # Schema fields
        db_data[FIELD_WORKFLOW_ID] = str(self.run_id)
        db_data[FIELD_STATUS] = workflow_data.get('status', 'unknown')
        db_data[FIELD_BRANCH] = workflow_data.get('head_branch', self.ref_name)
        db_data[FIELD_COMMIT_SHA] = workflow_data.get('head_sha', self.sha)
        db_data[FIELD_EVENT] = workflow_data.get('event', self.event_name)
        
        # Timing fields
        created_at = workflow_data.get('created_at')
        updated_at = workflow_data.get('updated_at')
        run_started_at = workflow_data.get('run_started_at')
        self.add_standardized_timing_fields(db_data, created_at, updated_at, run_started_at, "workflow")
        
        # Common context fields
        self.add_common_context_fields(db_data)
        
        # Override userAlias with actor from API if available
        actor = workflow_data.get('actor', {})
        if actor and actor.get('login'):
            db_data[FIELD_USER_ALIAS] = actor.get('login')
        
        # Add jobs list
        if jobs_data and 'jobs' in jobs_data:
            job_ids = [str(job['id']) for job in jobs_data['jobs']]
            db_data[FIELD_JOBS] = job_ids
        else:
            db_data[FIELD_JOBS] = []
        
        self.post_to_db(self.workflow_index, db_data)

    def _upload_all_job_and_step_metrics(self, jobs_data: Dict[str, Any]) -> tuple[int, int]:
        """Internal method to upload all job and step metrics, returns (jobs_processed, steps_processed)"""
        jobs_processed = 0
        steps_processed = 0
        
        for job in jobs_data['jobs']:
            try:
                # Upload job metrics
                self._upload_single_job_metrics(job)
                jobs_processed += 1
                
                # Upload step metrics for this job
                if self.steps_index:
                    step_count = self._upload_job_step_metrics(job)
                    steps_processed += step_count
                    
            except Exception as e:
                print(f"Error uploading metrics for job {job.get('name', 'unknown')}: {e}")
                continue
        
        return jobs_processed, steps_processed

    def post_workflow_metrics(self) -> Optional[Dict[str, Any]]:
        """Extract and post workflow metrics"""
        print(f"Uploading workflow metrics for run {self.run_id}")
        
        # Get workflow run data from GitHub API
        workflow_data = self.get_github_api_data(f"/repos/{self.repo}/actions/runs/{self.run_id}")
        print(f"Workflow data: {workflow_data}")
        if not workflow_data:
            print("Could not fetch workflow data from GitHub API")
            return None
            
        # Extract workflow metrics using standardized functions
        db_data = {}
        db_data[FIELD_ID] = f"github_{self.run_id}_{self.repo.replace('/', '_')}"
        
        # Schema fields
        db_data[FIELD_WORKFLOW_ID] = str(self.run_id)
        db_data[FIELD_STATUS] = workflow_data.get('status', 'unknown')
        db_data[FIELD_BRANCH] = workflow_data.get('head_branch', self.ref_name)
        db_data[FIELD_COMMIT_SHA] = workflow_data.get('head_sha', self.sha)
        db_data[FIELD_EVENT] = workflow_data.get('event', self.event_name)
        
        # Timestamps and timing using standardized method
        created_at = workflow_data.get('created_at')
        updated_at = workflow_data.get('updated_at')
        run_started_at = workflow_data.get('run_started_at')
        
        self.add_standardized_timing_fields(db_data, created_at, updated_at, run_started_at, "workflow")
        
        # Add common context fields
        self.add_common_context_fields(db_data)
        
        # Override userAlias with actor from API if available
        actor = workflow_data.get('actor', {})
        if actor and actor.get('login'):
            db_data[FIELD_USER_ALIAS] = actor.get('login')
        
        # Get jobs data from API (will be reused for job metrics)
        jobs_data = self.get_github_api_data(f"/repos/{self.repo}/actions/runs/{self.run_id}/jobs")
        if jobs_data and 'jobs' in jobs_data:
            job_ids = [str(job['id']) for job in jobs_data['jobs']]
            db_data[FIELD_JOBS] = job_ids
        else:
            db_data[FIELD_JOBS] = []
        
        self.post_to_db(self.workflow_index, db_data)
        return jobs_data

    def post_all_job_metrics(self, jobs_data: Optional[Dict[str, Any]] = None) -> None:
        """Extract and post metrics for all jobs in the current workflow"""
        print(f"Uploading job metrics for workflow run {self.run_id}")
        
        # Use provided jobs data or fetch from API if not provided
        if not jobs_data:
            jobs_data = self.get_github_api_data(f"/repos/{self.repo}/actions/runs/{self.run_id}/jobs")
        
        if not jobs_data or 'jobs' not in jobs_data:
            print("Could not fetch jobs data from GitHub API")
            return
            
        # Process all jobs in the workflow (including the current one)
        jobs_processed = 0
        steps_processed = 0
        for job in jobs_data['jobs']:
            try:
                self._upload_single_job_metrics(job)
                jobs_processed += 1
                
                # Also process steps for this job if steps index is configured
                if self.steps_index:
                    step_count = self._upload_job_step_metrics(job)
                    steps_processed += step_count
                    
            except Exception as e:
                print(f"Error uploading metrics for job {job.get('name', 'unknown')}: {e}")
                continue
        
        print(f"Successfully processed {jobs_processed} jobs and {steps_processed} steps")

    def _upload_single_job_metrics(self, job_data: Dict[str, Any]) -> None:
        """Extract and post metrics for a single job"""
        # Extract job metrics using standardized functions
        db_data = {}
        job_id = job_data['id']
        job_name = job_data['name']
        
        db_data[FIELD_ID] = f"github_{job_id}_{self.repo.replace('/', '_')}"
        
        # Schema fields
        db_data[FIELD_JOB_ID] = str(job_id)
        db_data[FIELD_WORKFLOW_ID] = str(self.run_id)
        db_data[FIELD_STATUS] = job_data.get('conclusion', job_data.get('status', 'unknown'))
        db_data[FIELD_BRANCH] = self.ref_name
        db_data[FIELD_RUNNER_INFO] = job_data.get('runner_name', 'unknown')
        
        db_data[FIELD_WORKFLOW_SOURCE] = self.event_name
        db_data[FIELD_JOB_NAME] = job_name
        
        # Timing fields using standardized method
        created_at = job_data.get('created_at')
        completed_at = job_data.get('completed_at')
        started_at = job_data.get('started_at')
        
        self.add_standardized_timing_fields(db_data, created_at, completed_at, started_at, "job")
        
        # Labels
        runner_labels = job_data.get('labels', [])
        db_data[FIELD_LABELS] = runner_labels if runner_labels else ['-']
        
        # Add steps list (get step IDs)
        steps = job_data.get('steps', [])
        if steps:
            step_ids = [f"{job_id}_{step.get('number', i+1)}" for i, step in enumerate(steps)]
            db_data[FIELD_STEPS] = step_ids
        else:
            db_data[FIELD_STEPS] = []
        
        # Runner info
        runner_name = job_data.get('runner_name')
        runner_id = job_data.get('runner_id')
        db_data[FIELD_RUNNER_ID] = str(runner_id) if runner_id else None
        db_data[FIELD_RUNNER_NAME] = runner_name
        
        # Add common context fields
        self.add_common_context_fields(db_data)
        
        self.post_to_db(self.jobs_index, db_data)
        print(f"Uploaded metrics for job: {job_name}")

    def _upload_job_step_metrics(self, job_data: Dict[str, Any]) -> int:
        """Extract and post metrics for all steps in a job"""
        job_id = job_data['id']
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
        
        # Create unique step ID and use standardized ID generation
        step_id = f"{job_id}_{step_number}"
        db_data[FIELD_ID] = f"github_step_{step_id}_{self.repo.replace('/', '_')}"
        
        # Schema-compliant fields
        db_data[FIELD_STEP_ID] = str(step_id)
        db_data[FIELD_JOB_ID] = str(job_id)
        db_data[FIELD_WORKFLOW_ID] = str(self.run_id)
        db_data[FIELD_NAME] = step_name
        db_data[FIELD_ORDER] = step_number
        db_data[FIELD_STATUS] = step_data.get('conclusion', step_data.get('status', 'unknown'))
        db_data[FIELD_JOB_NAME] = job_name
        
        # Timing fields using standardized method
        started_at = step_data.get('started_at')
        completed_at = step_data.get('completed_at')
        
        self.add_standardized_timing_fields(db_data, started_at, completed_at, None, "step")
        
        # Command/script executed (GitHub API doesn't always provide this, but we can infer)
        command = ""
        if step_data.get('action'):
            command = f"uses: {step_data['action']}"
        elif 'run' in step_name.lower() or 'script' in step_name.lower():
            command = "run: <script>"  # GitHub API doesn't expose the actual script content
        db_data[FIELD_COMMAND] = command
        
        # Add common context fields
        self.add_common_context_fields(db_data)
        
        # Job context
        db_data[FIELD_RUNNER_NAME] = job_data.get('runner_name')
        db_data[FIELD_RUNNER_ID] = str(job_data.get('runner_id')) if job_data.get('runner_id') else None
        
        # Job labels (separate from step labels)
        runner_labels = job_data.get('labels', [])
        db_data[FIELD_JOB_LABELS] = runner_labels if runner_labels else ['-']
        
        self.post_to_db(self.steps_index, db_data)
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

if __name__ == "__main__":
    main()
