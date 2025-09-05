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

def parse_iso_datetime(iso_string: str) -> datetime:
    """Parse ISO 8601 datetime string to datetime object. Using this instead of pandas to avoid dependencies."""
    if not iso_string:
        return None
    
    # Remove 'Z' suffix and add UTC timezone info
    if iso_string.endswith('Z'):
        iso_string = iso_string[:-1] + '+00:00'
    
    # Parse the datetime string
    try:
        return datetime.fromisoformat(iso_string)
    except ValueError:
        # Fallback for older Python versions
        if 'T' in iso_string and iso_string.endswith('+00:00'):
            dt_part = iso_string[:-6]  # Remove timezone
            dt = datetime.strptime(dt_part, '%Y-%m-%dT%H:%M:%S')
            return dt.replace(tzinfo=timezone.utc)
        else:
            # Last resort - assume UTC
            dt = datetime.strptime(iso_string, '%Y-%m-%dT%H:%M:%S')
            return dt.replace(tzinfo=timezone.utc)

def datetime_to_timestamp_ms(dt: datetime) -> int:
    """Convert datetime to millisecond timestamp"""
    if not dt:
        return None
    return int(dt.timestamp() * 1000)

def calculate_duration_seconds(start_time: str, end_time: str) -> Optional[float]:
    """Calculate duration in seconds between two ISO datetime strings"""
    if not start_time or not end_time:
        return None
    
    start_dt = parse_iso_datetime(start_time)
    end_dt = parse_iso_datetime(end_time)
    
    if not start_dt or not end_dt:
        return None
    
    return (end_dt - start_dt).total_seconds()

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
        
    def post_to_db(self, url: str, data: Dict[str, Any]) -> None:
        """Push json data to the database/OpenSearch URL"""
        print(f"Posting metrics to database...")
        try:
            response = requests.post(url, data=json.dumps(data), headers=self.headers, timeout=30)
            if not (200 <= response.status_code < 300):
                raise ValueError(f"Error posting to DB: HTTP {response.status_code}")
            print(f"Successfully posted metrics with ID: {data.get('_id', 'unknown')}")
        except requests.exceptions.RequestException as e:
            # Mask the URL in error messages to prevent exposure
            error_msg = str(e)
            error_msg = mask_sensitive_urls(error_msg, url)
            raise ValueError(f"Database connection failed: {error_msg}")

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

    def calculate_queue_time(self, created_at: str, started_at: Optional[str]) -> Optional[float]:
        """Calculate queue time in seconds"""
        if not started_at:
            return None
        try:
            created = parse_iso_datetime(created_at)
            started = parse_iso_datetime(started_at)
            return (started - created).total_seconds()
        except Exception as e:
            print(f"Error calculating queue time: {e}")
            return None

    def add_common_context_fields(self, db_data: Dict[str, Any]) -> None:
        """Add common context fields used across all metric types"""
        db_data["userAlias"] = self.actor
        db_data["repo"] = self.repo
        db_data["workflowName"] = self.workflow_name
        db_data["githubEvent"] = self.event_name
        db_data["branch"] = self.ref_name

    def add_timing_fields(self, db_data: Dict[str, Any], created_at: str, completed_at: str, started_at: Optional[str] = None) -> None:
        """Add timing-related fields (timestamps, duration, queue time)"""
        # Creation time
        if created_at:
            db_data["creationTime"] = datetime_to_timestamp_ms(parse_iso_datetime(created_at))
        
        # Completion time
        if completed_at:
            db_data["finishedTime"] = datetime_to_timestamp_ms(parse_iso_datetime(completed_at))
        
        # Duration
        if created_at and completed_at:
            duration = calculate_duration_seconds(created_at, completed_at)
            db_data["durationSec"] = int(duration) if duration else 0
        
        # Queue time (if started_at is provided)
        if created_at and started_at:
            queue_duration = self.calculate_queue_time(created_at, started_at)
            if queue_duration is not None:
                db_data["queueTimeSec"] = int(queue_duration)

    def categorize_step(self, step_name: str, action: str) -> List[str]:
        """Categorize a step based on its name and action - customized for Dynamo workflows"""
        labels = []
        step_name_lower = step_name.lower()
        action_lower = action.lower()
        
        # Setup and infrastructure
        if any(word in step_name_lower for word in ['setup', 'checkout', 'check out']):
            labels.append('setup')
        elif any(word in action_lower for word in ['checkout', 'setup-python', 'setup-node', 'docker/setup']):
            labels.append('setup')
        elif any(word in step_name_lower for word in ['install', 'pip install', 'apt-get install']):
            labels.append('setup')
        
        # Caching
        elif any(word in step_name_lower for word in ['cache', 'restore']):
            labels.append('cache')
        elif 'cache' in action_lower:
            labels.append('cache')
        
        # Authentication
        elif any(word in step_name_lower for word in ['login', 'auth', 'authenticate']):
            labels.append('auth')
        
        # Build and compilation
        elif any(word in step_name_lower for word in ['build', 'compile', 'make', 'docker build']):
            labels.append('build')
        elif any(word in step_name_lower for word in ['define image tag', 'tag']):
            labels.append('build')
        
        # Testing
        elif any(word in step_name_lower for word in ['test', 'pytest', 'rust checks', 'check']):
            labels.append('test')
        elif any(word in step_name_lower for word in ['lychee', 'link check']):
            labels.append('test')
        
        # Services and infrastructure
        elif any(word in step_name_lower for word in ['start services', 'docker compose', 'cleanup']):
            labels.append('services')
        
        # Artifacts and reporting
        elif any(word in step_name_lower for word in ['copy', 'archive', 'upload', 'artifact']):
            labels.append('artifacts')
        elif 'upload-artifact' in action_lower:
            labels.append('artifacts')
        
        # Linting and formatting
        elif any(word in step_name_lower for word in ['lint', 'format', 'clippy', 'rustfmt']):
            labels.append('lint')
        elif 'pre-commit' in action_lower:
            labels.append('lint')
        
        # Deployment
        elif any(word in step_name_lower for word in ['deploy', 'publish', 'release']):
            labels.append('deploy')
        
        # Default fallback
        else:
            labels.append('custom')
            
        return labels if labels else ['custom']

    def post_workflow_metrics(self) -> None:
        """Extract and post workflow metrics"""
        print(f"Uploading workflow metrics for run {self.run_id}")
        
        # Get workflow run data from GitHub API
        workflow_data = self.get_github_api_data(f"/repos/{self.repo}/actions/runs/{self.run_id}")
        if not workflow_data:
            print("Could not fetch workflow data from GitHub API")
            return
            
        # Extract workflow metrics
        db_data = {}
        db_data["_id"] = f"github_{self.run_id}_{self.repo.replace('/', '_')}"
        
        # Schema fields
        db_data["workflowId"] = str(self.run_id)
        db_data["status"] = workflow_data.get('status', 'unknown')
        db_data["branch"] = workflow_data.get('head_branch', self.ref_name)
        db_data["commitSha"] = workflow_data.get('head_sha', self.sha)
        db_data["event"] = workflow_data.get('event', self.event_name)
        
        # Timestamps and timing
        created_at = workflow_data.get('created_at')
        updated_at = workflow_data.get('updated_at')
        run_started_at = workflow_data.get('run_started_at')
        
        # Add timing fields using helper method
        self.add_timing_fields(db_data, created_at, updated_at, run_started_at)
        
        # Start time (when workflow actually started running)
        if run_started_at:
            db_data["startTime"] = datetime_to_timestamp_ms(parse_iso_datetime(run_started_at))
        
        # End time
        if updated_at and workflow_data.get('status') in ['completed', 'cancelled', 'failure']:
            db_data["endTime"] = datetime_to_timestamp_ms(parse_iso_datetime(updated_at))
        
        # Add common context fields
        self.add_common_context_fields(db_data)
        
        # Override userAlias with actor from API if available
        actor = workflow_data.get('actor', {})
        if actor and actor.get('login'):
            db_data["userAlias"] = actor.get('login')
        
        # Add jobs list (get job IDs from API)
        jobs_data = self.get_github_api_data(f"/repos/{self.repo}/actions/runs/{self.run_id}/jobs")
        if jobs_data and 'jobs' in jobs_data:
            job_ids = [str(job['id']) for job in jobs_data['jobs']]
            db_data["jobs"] = job_ids
        else:
            db_data["jobs"] = []
        
        self.post_to_db(self.workflow_index, db_data)

    def post_all_job_metrics(self) -> None:
        """Extract and post metrics for all jobs in the current workflow"""
        print(f"Uploading job metrics for workflow run {self.run_id}")
        
        # Get jobs data from GitHub API
        jobs_data = self.get_github_api_data(f"/repos/{self.repo}/actions/runs/{self.run_id}/jobs")
        if not jobs_data or 'jobs' not in jobs_data:
            print("Could not fetch jobs data from GitHub API")
            return
            
        # Process all jobs in the workflow (including the current one)
        jobs_processed = 0
        steps_processed = 0
        for job in jobs_data['jobs']:
            try:
                self.post_single_job_metrics(job)
                jobs_processed += 1
                
                # Also process steps for this job if steps index is configured
                if self.steps_index:
                    step_count = self.post_job_step_metrics(job)
                    steps_processed += step_count
                    
            except Exception as e:
                print(f"Error uploading metrics for job {job.get('name', 'unknown')}: {e}")
                continue
        
        print(f"Successfully processed {jobs_processed} jobs and {steps_processed} steps")

    def post_single_job_metrics(self, job_data: Dict[str, Any]) -> None:
        """Extract and post metrics for a single job"""
        # Extract job metrics
        db_data = {}
        job_id = job_data['id']
        job_name = job_data['name']
        
        db_data["_id"] = f"github_{job_id}_{self.repo.replace('/', '_')}"
        
        # Schema fields
        db_data["jobId"] = str(job_id)
        db_data["workflowId"] = str(self.run_id)
        db_data["status"] = job_data.get('conclusion', job_data.get('status', 'unknown'))
        db_data["branch"] = self.ref_name
        db_data["runnerInfo"] = job_data.get('runner_name', 'unknown')
        
        db_data["workflowSource"] = self.event_name
        db_data["jobName"] = job_name
        
        # Timing fields
        created_at = job_data.get('created_at')
        completed_at = job_data.get('completed_at')
        started_at = job_data.get('started_at')
        
        # Add timing fields using helper method
        self.add_timing_fields(db_data, created_at, completed_at, started_at)
        
        # Labels
        runner_labels = job_data.get('labels', [])
        db_data["labels"] = runner_labels if runner_labels else ['-']
        
        # Add steps list (get step IDs)
        steps = job_data.get('steps', [])
        if steps:
            step_ids = [f"{job_id}_{step.get('number', i+1)}" for i, step in enumerate(steps)]
            db_data["steps"] = step_ids
        else:
            db_data["steps"] = []
        
        # Runner info
        runner_name = job_data.get('runner_name')
        runner_id = job_data.get('runner_id')
        db_data["runnerId"] = str(runner_id) if runner_id else None
        db_data["runnerName"] = runner_name
        
        # Add common context fields
        self.add_common_context_fields(db_data)
        
        self.post_to_db(self.jobs_index, db_data)
        print(f"Uploaded metrics for job: {job_name}")

    def post_job_step_metrics(self, job_data: Dict[str, Any]) -> int:
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
                self.post_single_step_metrics(step, job_data, step_index)
                steps_processed += 1
            except Exception as e:
                step_name = step.get('name', f'step_{step_index}')
                print(f"Error uploading metrics for step {step_name} in job {job_name}: {e}")
                continue
        
        print(f"Uploaded metrics for {steps_processed} steps in job {job_name}")
        return steps_processed

    def post_single_step_metrics(self, step_data: Dict[str, Any], job_data: Dict[str, Any], step_index: int) -> None:
        """Extract and post metrics for a single step"""
        # Extract step metrics
        db_data = {}
        job_id = job_data['id']
        job_name = job_data['name']
        step_name = step_data.get('name', f'step_{step_index}')
        step_number = step_data.get('number', step_index + 1)
        
        # Create unique step ID
        step_id = f"{job_id}_{step_number}"
        db_data["_id"] = f"github_step_{step_id}_{self.repo.replace('/', '_')}"
        
        # Schema-compliant fields
        db_data["stepId"] = str(step_id)
        db_data["jobId"] = str(job_id)
        db_data["workflowId"] = str(self.run_id)
        db_data["name"] = step_name
        db_data["order"] = step_number
        db_data["status"] = step_data.get('conclusion', step_data.get('status', 'unknown'))
        db_data["jobName"] = job_name
        
        # Step conclusion (separate from status)
        db_data["conclusion"] = step_data.get('conclusion', step_data.get('status', 'unknown'))
        
        # Timing fields (steps don't have creation time, only start/end)
        started_at = step_data.get('started_at')
        completed_at = step_data.get('completed_at')
        
        # Add step-specific timing fields
        if started_at:
            db_data["startedAt"] = datetime_to_timestamp_ms(parse_iso_datetime(started_at))
        if completed_at:
            db_data["endedAt"] = datetime_to_timestamp_ms(parse_iso_datetime(completed_at))
        if started_at and completed_at:
            duration = calculate_duration_seconds(started_at, completed_at)
            db_data["durationSec"] = int(duration) if duration else 0
        
        # Step details
        db_data["action"] = step_data.get('action', '')  # For action steps (uses: actions/checkout@v4)
        db_data["with"] = json.dumps(step_data.get('with', {})) if step_data.get('with') else ''
        db_data["env"] = json.dumps(step_data.get('env', {})) if step_data.get('env') else ''
        
        # Command/script executed (GitHub API doesn't always provide this, but we can infer)
        command = ""
        if step_data.get('action'):
            command = f"uses: {step_data['action']}"
        elif 'run' in step_name.lower() or 'script' in step_name.lower():
            command = "run: <script>"  # GitHub API doesn't expose the actual script content
        db_data["command"] = command
        
        # Step categorization using helper method
        action = step_data.get('action', '')
        db_data["stepLabels"] = self.categorize_step(step_name, action)
        
        # Add common context fields
        self.add_common_context_fields(db_data)
        
        # Job context
        db_data["runnerName"] = job_data.get('runner_name')
        db_data["runnerId"] = str(job_data.get('runner_id')) if job_data.get('runner_id') else None
        
        # Job labels (separate from step labels)
        runner_labels = job_data.get('labels', [])
        db_data["jobLabels"] = runner_labels if runner_labels else ['-']
        
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
    
    # Upload workflow metrics
    try:
        print("Uploading workflow metrics...")
        uploader.post_workflow_metrics()
        print("Workflow metrics uploaded successfully")
    except Exception as e:
        error_msg = str(e)
        error_msg = mask_sensitive_urls(error_msg, uploader.workflow_index)
        print(f"Error uploading workflow metrics: {error_msg}")
        
    # Upload all job and step metrics
    try:
        print("Uploading job and step metrics for all jobs in workflow...")
        uploader.post_all_job_metrics()
        print("All job and step metrics uploaded successfully")
    except Exception as e:
        error_msg = str(e)
        error_msg = mask_sensitive_urls(error_msg, uploader.jobs_index)
        if uploader.steps_index:
            error_msg = mask_sensitive_urls(error_msg, uploader.steps_index)
        print(f"Error uploading job/step metrics: {error_msg}")

if __name__ == "__main__":
    main()
