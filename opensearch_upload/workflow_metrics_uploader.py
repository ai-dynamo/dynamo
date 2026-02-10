#!/usr/bin/env python3

import json
import os
import sys
import requests
from datetime import datetime, timezone, timedelta
from time import sleep
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import re

# Import error classification modules
try:
    from error_classification.classifier import ErrorClassifier
    from error_classification.error_extractor import ErrorExtractor
    from error_classification.config import Config as ErrorConfig
    ERROR_CLASSIFICATION_AVAILABLE = True
except ImportError:
    ERROR_CLASSIFICATION_AVAILABLE = False
    print("⚠️  Error classification module not available")

# Annotation field constants
FIELD_ANNOTATION_COUNT = "l_annotation_count"
FIELD_ANNOTATION_FAILURE_COUNT = "l_annotation_failure_count"
FIELD_ANNOTATION_WARNING_COUNT = "l_annotation_warning_count"
FIELD_ANNOTATION_NOTICE_COUNT = "l_annotation_notice_count"
FIELD_ANNOTATION_MESSAGES = "s_annotation_messages"

# Runner field constants
FIELD_RUNNER_PREFIX = "s_runner_prefix"

# Retry field constants
FIELD_RUN_ATTEMPT = "l_run_attempt"  # Which attempt this is (1, 2, 3...)
FIELD_RETRY_COUNT = "l_retry_count"  # Number of retries (0, 1, 2...)

# Error classification field constants
FIELD_ERROR_TYPE = "s_error_type"  # Error category from classifier
FIELD_ERROR_SUMMARY = "s_error_summary"  # Root cause summary
FIELD_ERROR_CONFIDENCE = "f_error_confidence"  # Classification confidence score


def process_annotations(
    annotations: list, context_name: str = "", max_messages: int = 10
) -> Dict[str, Any]:
    """
    Process a list of annotations and return structured data
    Args:
        annotations: List of annotation objects from GitHub API
        context_name: Optional context name to prefix messages (e.g., job name)
        max_messages: Maximum number of messages to collect
    Returns:
        Dictionary with annotation counts and messages
    """
    annotation_data = {
        "count": len(annotations),
        "failure_count": 0,
        "warning_count": 0,
        "notice_count": 0,
        "messages": [],
    }

    for annotation in annotations:
        level = annotation.get("annotation_level", "").lower()
        message = annotation.get("message", "")

        # Count by level
        if level == "failure":
            annotation_data["failure_count"] += 1
        elif level == "warning":
            annotation_data["warning_count"] += 1
        elif level == "notice":
            annotation_data["notice_count"] += 1

        # Collect messages (limit to prevent oversized payloads)
        if message and len(annotation_data["messages"]) < max_messages:
            prefix = f"[{context_name}] " if context_name else ""
            annotation_data["messages"].append(f"{prefix}[{level.upper()}] {message}")

    return annotation_data

class WorkflowMetricsUploader:
    def __init__(self):
        self.headers = {"Content-Type": "application/json", "Accept-Charset": "UTF-8"}
        self.workflow_index = os.getenv("WORKFLOW_INDEX")
        self.jobs_index = os.getenv("JOB_INDEX") 
        self.steps_index = os.getenv("STEPS_INDEX")
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo = os.getenv("REPO", "ai-dynamo/dynamo")
        self.hours_back = float(os.getenv("HOURS_BACK", "4"))
        
        if not all([self.workflow_index, self.jobs_index, self.steps_index, self.github_token]):
            raise ValueError("Missing required environment variables")
            
        self.github_headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "workflow-metrics-uploader/1.0"
        }

        # Initialize error classification if enabled and available
        self.enable_error_classification = os.getenv("ENABLE_ERROR_CLASSIFICATION", "false").lower() == "true"
        self.error_classifier = None
        self.error_extractor = None

        if self.enable_error_classification and ERROR_CLASSIFICATION_AVAILABLE:
            try:
                # Check if API key is available
                anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
                if anthropic_api_key:
                    # Create config for error classifier
                    error_config = ErrorConfig(
                        anthropic_api_key=anthropic_api_key,
                        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
                        api_format=os.getenv("API_FORMAT", "anthropic"),
                        api_base_url=os.getenv("API_BASE_URL"),
                        error_classification_index=os.getenv("ERROR_CLASSIFICATION_INDEX"),
                        opensearch_url=os.getenv("OPENSEARCH_URL"),
                        opensearch_username=os.getenv("OPENSEARCH_USERNAME"),
                        opensearch_password=os.getenv("OPENSEARCH_PASSWORD")
                    )
                    # Initialize classifier (OpenSearch client is optional for deduplication)
                    self.error_classifier = ErrorClassifier(
                        config=error_config,
                        opensearch_client=None  # Deduplication handled separately for now
                    )
                    self.error_extractor = ErrorExtractor()
                    print("✅ Error classification enabled")
                else:
                    print("⚠️  ANTHROPIC_API_KEY not set, error classification disabled")
            except Exception as e:
                print(f"⚠️  Error initializing classifier: {e}")
                self.error_classifier = None
        elif self.enable_error_classification:
            print("⚠️  Error classification module not available, classification disabled")

        print(f"🚀 Initialized uploader for {self.repo}")
        print(f"📊 Will fetch workflows from the past {self.hours_back} hours")
    
    def mask_sensitive_urls(self, error_msg: str, url: str) -> str:
        """Mask sensitive URLs in error messages"""
        if not url:
            return error_msg
        try:
            parsed_url = urlparse(url)
            hostname = parsed_url.hostname
            if hostname:
                error_msg = error_msg.replace(hostname, "***HOSTNAME***")
            if url in error_msg:
                error_msg = error_msg.replace(url, "***DATABASE_URL***")
        except Exception:
            if url in error_msg:
                error_msg = error_msg.replace(url, "***DATABASE_URL***")
        return error_msg
    
    def post_to_opensearch(self, url: str, data: Dict[str, Any]) -> bool:
        """Post data to OpenSearch with error handling"""
        try:
            response = requests.post(url, data=json.dumps(data), headers=self.headers, timeout=30)
            if not (200 <= response.status_code < 300):
                print(f"❌ OpenSearch returned HTTP {response.status_code}")
                return False
            print(f"✅ Posted metrics for {data.get('_id', 'unknown')}")
            return True
        except requests.exceptions.RequestException as e:
            masked_error = self.mask_sensitive_urls(str(e), url)
            print(f"❌ Failed to post to OpenSearch: {masked_error}")
            return False
    
    def fetch_workflow_runs_timerange(self, hours: float = 4) -> List[Dict[str, Any]]:
        """Fetch workflow runs with proper pagination"""
        workflow_runs_url = f'https://api.github.com/repos/{self.repo}/actions/runs'
        
        # Calculate time range
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(hours=hours)
        end_time = now
        
        print(f"🔍 Fetching workflow runs from {self.repo} - Last {hours} hours")
        print(f"   Time range: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')} to {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        page = 1
        all_runs = []
        
        # For large time ranges (like a week), use more generous limits
        if hours >= 24:  # 1 day or more
            max_pages = 200  # Allow more pages for large ranges
            consecutive_old_runs_limit = 1000  # Be more patient
            print(f"   🚀 Using extended pagination for large time range ({hours}h)")
        else:
            max_pages = 50  # Original limit for small ranges
            consecutive_old_runs_limit = 200
        
        consecutive_old_runs = 0
        
        while page <= max_pages:
            try:
                response = requests.get(
                    workflow_runs_url,
                    headers=self.github_headers,
                    params={'page': page, 'per_page': 100, 'status': 'completed'},
                    timeout=30
                )
                response.raise_for_status()
                workflow_runs_data = response.json()
                
                if not workflow_runs_data.get('workflow_runs'):
                    print(f"   No more workflow runs available")
                    break
                
                runs = workflow_runs_data['workflow_runs']
                print(f"📄 Page {page}: Found {len(runs)} runs")
                
                # Filter runs from the specified time range
                recent_runs = []
                old_runs_this_page = 0
                
                for run in runs:
                    try:
                        # Check both created_at AND updated_at to capture retries
                        # updated_at changes when a workflow is retried
                        created_at_str = run.get('created_at', '')
                        updated_at_str = run.get('updated_at', '')
                        
                        if not created_at_str:
                            continue
                            
                        created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00')) if updated_at_str else created_at
                        
                        # Include run if EITHER created_at OR updated_at is in range
                        # This catches both new runs AND retries of old runs
                        in_range = (start_time <= created_at <= end_time) or (start_time <= updated_at <= end_time)
                        
                        if in_range:
                            recent_runs.append(run)
                            consecutive_old_runs = 0  # Reset counter
                        else:
                            old_runs_this_page += 1
                            # Only count as old if BOTH timestamps are before start_time
                            if created_at < start_time and updated_at < start_time:
                                consecutive_old_runs += 1
                                
                    except (ValueError, KeyError) as e:
                        print(f"   ⚠️  Skipping run with invalid timestamp: {e}")
                        continue
                
                all_runs.extend(recent_runs)
                print(f"   Added {len(recent_runs)} runs from time range, {old_runs_this_page} outside range")
                print(f"   Total collected: {len(all_runs)} runs")
                
                # Stop conditions
                if len(runs) < 100:
                    print(f"   Reached last page")
                    break
                    
                if consecutive_old_runs >= consecutive_old_runs_limit:
                    print(f"   Stopping: {consecutive_old_runs} consecutive old runs")
                    break
                
                page += 1
                sleep(0.5)  # Rate limiting
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Network error on page {page}: {e}")
                sleep(2)
                continue
            except Exception as e:
                print(f"❌ Error fetching page {page}: {e}")
                break
        
        print(f"✅ Total runs from last {hours}h: {len(all_runs)}")
        return all_runs
    
    def fetch_run_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Fetch detailed information for a specific workflow run including run_attempt"""
        try:
            url = f'https://api.github.com/repos/{self.repo}/actions/runs/{run_id}'
            response = requests.get(
                url,
                headers=self.github_headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error fetching run details for {run_id}: {e}")
            return None
    
    def fetch_run_attempt(self, run_id: str, attempt_number: int) -> Optional[Dict[str, Any]]:
        """Fetch a specific attempt of a workflow run"""
        try:
            url = f'https://api.github.com/repos/{self.repo}/actions/runs/{run_id}/attempts/{attempt_number}'
            response = requests.get(
                url,
                headers=self.github_headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error fetching attempt {attempt_number} for run {run_id}: {e}")
            return None
    
    def fetch_job_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Fetch job details for a workflow run"""
        try:
            response = requests.get(
                f"https://api.github.com/repos/{self.repo}/actions/runs/{run_id}/jobs",
                headers=self.github_headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Failed to fetch jobs for run {run_id}: {e}")
            return None
    
    def calculate_time_diff(self, start_time: str, end_time: str) -> int:
        """Calculate duration in seconds"""
        if not start_time or not end_time:
            return 0
        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            return max(0, int((end_dt - start_dt).total_seconds()))
        except:
            return 0
    
    def get_status_number(self, status: str) -> Optional[int]:
        """Convert status to number"""
        if status == "success":
            return 1
        elif status == "failure":
            return 0
        return None
    
    def extract_pr_id(self, workflow_data: Dict[str, Any]) -> str:
        """Extract PR ID from workflow data"""
        pull_requests = workflow_data.get("pull_requests", [])
        if pull_requests and len(pull_requests) > 0:
            pr_number = pull_requests[0].get("number")
            if pr_number:
                return str(pr_number)
        return "N/A"
    
    def extract_runner_prefix(self, runner_name: str) -> str:
        """
        Extract runner prefix by removing the trailing random suffix.
        
        Examples:
            "gpu-l40-amd64-g6-4xlarge-runners-jh8hc-qdb9l" -> "gpu-l40-amd64-g6-4xlarge-runners"
            "cpu-large-runners-abc123-xyz" -> "cpu-large-runners"
            "fastchecker-05_7d74f3fca8f0" -> "fastchecker"
            "github-ci-runner01" -> "github-ci"
            "GitHub Actions 1004949702" -> "GitHub Actions"
            "" -> "N/A"
        """
        if not runner_name:
            return "N/A"
        
        # First, check if it ends with space + digits (e.g., "GitHub Actions 1004949702")
        if ' ' in runner_name:
            parts = runner_name.rsplit(' ', 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[0]
        
        # For fastchecker runners, keep only "fastchecker"
        # (e.g., "fastchecker-05_7d74f3fca8f0" -> "fastchecker")
        runner_lower = runner_name.lower()
        if runner_lower.startswith('fastchecker'):
            return "fastchecker"
        
        # For github-ci runners, keep only "github-ci"
        # (e.g., "github-ci-runner01" -> "github-ci")
        if runner_lower.startswith('github-ci'):
            return "github-ci"
        
        # For cpu/gpu runners, remove the last TWO hyphen-separated segments
        # (e.g., "gpu-l40-amd64-g6-4xlarge-runners-jh8hc-qdb9l" -> "gpu-l40-amd64-g6-4xlarge-runners")
        if runner_lower.startswith('cpu-') or runner_lower.startswith('gpu-'):
            parts = runner_name.split("-")
            if len(parts) > 2:
                return "-".join(parts[:-2])
        
        return runner_name
    
    def get_github_api_data(self, endpoint: str) -> Optional[Any]:
        """
        Generic method to fetch data from GitHub API
        Args:
            endpoint: API endpoint path (e.g., "/repos/owner/repo/...")
        Returns:
            JSON response data or None on error
        """
        try:
            url = f"https://api.github.com{endpoint}"
            response = requests.get(
                url,
                headers=self.github_headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️  GitHub API request failed for {endpoint}: {e}")
            return None
        except Exception as e:
            print(f"⚠️  Error processing GitHub API response: {e}")
            return None
    
    def add_annotation_fields(
        self,
        db_data: Dict[str, Any],
        annotation_type: str,
        entity_id: str,
        entity_name: str = "",
        max_message_length: int = 1000,
    ) -> None:
        """
        Generic method to fetch annotations and add annotation fields to data payload
        Args:
            db_data: Dictionary to add annotation fields to
            annotation_type: Either "job" or "workflow"
            entity_id: Job ID or workflow run ID
            entity_name: Job name or workflow name for logging
            max_message_length: Maximum length for the messages field
        """
        # Initialize default values
        db_data[FIELD_ANNOTATION_COUNT] = 0
        db_data[FIELD_ANNOTATION_FAILURE_COUNT] = 0
        db_data[FIELD_ANNOTATION_WARNING_COUNT] = 0
        db_data[FIELD_ANNOTATION_NOTICE_COUNT] = 0
        db_data[FIELD_ANNOTATION_MESSAGES] = ""

        # Check if we have a GitHub token
        if not self.github_token:
            return

        try:
            if annotation_type == "job":
                # Fetch annotations for a single job
                annotations_data = self.get_github_api_data(
                    f"/repos/{self.repo}/check-runs/{entity_id}/annotations"
                )
                if annotations_data:
                    annotation_summary = process_annotations(
                        annotations_data, max_messages=10
                    )
                else:
                    return  # No annotations or API error

            elif annotation_type == "workflow":
                # Fetch all jobs first, then aggregate their annotations
                jobs_data = self.get_github_api_data(
                    f"/repos/{self.repo}/actions/runs/{entity_id}/jobs"
                )
                if not jobs_data or "jobs" not in jobs_data:
                    return

                # Aggregate annotations from all jobs
                all_annotations = []
                for job in jobs_data.get("jobs", []):
                    job_id = job.get("id")
                    job_name = job.get("name", "unknown")
                    if job_id:
                        job_annotations_data = self.get_github_api_data(
                            f"/repos/{self.repo}/check-runs/{job_id}/annotations"
                        )
                        if job_annotations_data:
                            # Add job context to each annotation
                            for annotation in job_annotations_data:
                                annotation["_job_context"] = job_name
                            all_annotations.extend(job_annotations_data)

                # Process all aggregated annotations
                annotation_summary = process_annotations(
                    all_annotations, max_messages=20
                )

                # For workflow-level, add job context to messages
                contextualized_messages = []
                for i, annotation in enumerate(all_annotations):
                    if i >= 20:  # Respect max_messages limit
                        break
                    job_context = annotation.get("_job_context", "unknown")
                    level = annotation.get("annotation_level", "").upper()
                    message = annotation.get("message", "")
                    if message:
                        contextualized_messages.append(
                            f"[{job_context}] [{level}] {message}"
                        )

                annotation_summary["messages"] = contextualized_messages
            else:
                print(f"⚠️  Unknown annotation type: {annotation_type}")
                return

            # Add annotation fields to data
            db_data[FIELD_ANNOTATION_COUNT] = annotation_summary["count"]
            db_data[FIELD_ANNOTATION_FAILURE_COUNT] = annotation_summary[
                "failure_count"
            ]
            db_data[FIELD_ANNOTATION_WARNING_COUNT] = annotation_summary[
                "warning_count"
            ]
            db_data[FIELD_ANNOTATION_NOTICE_COUNT] = annotation_summary["notice_count"]

            # Join messages with separator, limit total length
            messages_text = " | ".join(annotation_summary["messages"])
            if len(messages_text) > max_message_length:
                messages_text = messages_text[: max_message_length - 3] + "..."
            db_data[FIELD_ANNOTATION_MESSAGES] = messages_text

            # Log results
            if annotation_summary["count"] > 0:
                print(
                    f"   📊 {annotation_type.title()} annotations: {annotation_summary['count']} total, {annotation_summary['failure_count']} failures, {annotation_summary['warning_count']} warnings"
                )
                # Print the actual annotation messages
                for msg in annotation_summary["messages"]:
                    print(f"      ⚠️  {msg}")

        except Exception as e:
            print(
                f"⚠️  Error fetching {annotation_type} annotations for {entity_name}: {e}"
            )

    def classify_job_errors(
        self,
        job_data: Dict[str, Any],
        workflow_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract and classify errors from a failed job.
        Returns a dict mapping step names to classifications.

        Args:
            job_data: Job information
            workflow_data: Optional workflow information for context

        Returns:
            Dict with 'job_classification' and 'step_classifications' keys
        """
        result = {
            "job_classification": None,
            "step_classifications": {}  # step_name -> classification
        }

        # Skip if error classification is not enabled or available
        if not self.error_classifier or not self.enable_error_classification:
            return result

        # Only classify failed jobs
        status = job_data.get("conclusion") or job_data.get("status", "unknown")
        if status not in ["failure", "failed"]:
            return result

        try:
            job_id = str(job_data["id"])
            job_name = job_data.get("name", "")

            print(f"   🤖 Classifying errors for job: {job_name}")

            # Fetch job logs from GitHub API (once)
            logs_url = f"https://api.github.com/repos/{self.repo}/actions/jobs/{job_id}/logs"
            logs_response = requests.get(logs_url, headers=self.github_headers, timeout=30)

            if logs_response.status_code != 200:
                print(f"   ⚠️  Failed to fetch logs (HTTP {logs_response.status_code})")
                return result

            log_content = logs_response.text

            # Extract errors from all failed steps
            all_errors = self.error_extractor.extract_from_github_job_logs(
                log_content=log_content,
                context={"job_name": job_name}
            )

            if not all_errors:
                print(f"   ⚠️  No errors extracted from job logs")
                return result

            print(f"   📝 Found {len(all_errors)} error(s) in job")

            # Classify each error and map to steps
            for error in all_errors:
                step_name = error.metadata.get("step_name", "") if error.metadata else ""

                # Classify the error
                classification = self.error_classifier.classify_error(
                    error_context=error,
                    use_cache=True,  # Use caching to reduce API costs
                    classification_method="batch"  # Called from batch uploader
                )

                # Store classification for this step
                if step_name:
                    result["step_classifications"][step_name] = {
                        "error_type": classification.primary_category,
                        "error_summary": classification.root_cause_summary,
                        "error_confidence": classification.confidence_score
                    }

                    confidence_pct = int(classification.confidence_score * 100)
                    print(f"      ✅ Step '{step_name}': {classification.primary_category} ({confidence_pct}%)")

                # Use first error as job-level classification
                if result["job_classification"] is None:
                    result["job_classification"] = {
                        "error_type": classification.primary_category,
                        "error_summary": classification.root_cause_summary,
                        "error_confidence": classification.confidence_score
                    }

            return result

        except Exception as e:
            print(f"   ⚠️  Error during classification: {e}")
            return result

    def add_error_classification_fields(
        self,
        db_data: Dict[str, Any],
        classification: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add error classification fields to job or step data.

        Args:
            db_data: The dictionary to add fields to (job_metrics or step_metrics)
            classification: Pre-computed classification dict with error_type, error_summary, error_confidence
        """
        if not classification:
            return

        # Add classification fields to db_data
        db_data[FIELD_ERROR_TYPE] = classification["error_type"]
        db_data[FIELD_ERROR_SUMMARY] = classification["error_summary"]
        db_data[FIELD_ERROR_CONFIDENCE] = classification["error_confidence"]

    def backfill_previous_attempts(self, workflow_data: Dict[str, Any]) -> int:
        """
        When a retry is detected, fetch and upload all previous attempts
        Returns: number of attempts backfilled
        """
        run_id = str(workflow_data["id"])
        run_attempt = workflow_data.get("run_attempt", 1)
        
        if run_attempt <= 1:
            return 0  # No previous attempts to backfill
        
        backfilled = 0
        print(f"   📥 Backfilling {run_attempt - 1} previous attempt(s)...")
        
        for attempt_num in range(1, run_attempt):
            try:
                print(f"      Fetching attempt #{attempt_num}...")
                attempt_data = self.fetch_run_attempt(run_id, attempt_num)
                
                if not attempt_data:
                    print(f"      ⚠️  Could not fetch attempt #{attempt_num}, skipping")
                    continue
                
                # Upload workflow metrics for this attempt
                if self.upload_workflow_metrics(attempt_data):
                    backfilled += 1
                
                # Fetch and upload jobs for this attempt
                jobs_data = self.fetch_job_details(run_id)
                if jobs_data and "jobs" in jobs_data:
                    for job in jobs_data["jobs"]:
                        # Classify errors once for the entire job
                        classifications = self.classify_job_errors(job, attempt_data)
                        job_classification = classifications.get("job_classification")
                        step_classifications = classifications.get("step_classifications", {})

                        self.upload_job_metrics(job, attempt_data, job_classification)

                        # Upload steps for this job with pre-computed classifications
                        for step_index, step in enumerate(job.get("steps", [])):
                            step_name = step.get("name", f"step_{step_index}")
                            step_classification = step_classifications.get(step_name)
                            self.upload_step_metrics(step, job, attempt_data, step_index, step_classification)
                
                sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"      ❌ Error backfilling attempt #{attempt_num}: {e}")
                continue
        
        if backfilled > 0:
            print(f"   ✅ Backfilled {backfilled} previous attempt(s)")
        
        return backfilled
    
    def upload_workflow_metrics(self, workflow_data: Dict[str, Any]) -> bool:
        """Upload workflow metrics"""
        run_id = str(workflow_data["id"])
        workflow_name = workflow_data.get("name", "")
        
        # Fetch detailed run information if run_attempt is not in the data
        # This ensures we always have the retry information
        if "run_attempt" not in workflow_data:
            print(f"   ℹ️  Fetching detailed run info to get run_attempt field...")
            detailed_run = self.fetch_run_details(run_id)
            if detailed_run:
                workflow_data = detailed_run
        
        # Determine final status
        status = workflow_data.get("conclusion") or workflow_data.get("status", "unknown")
        status_number = self.get_status_number(status)
        
        # Calculate timing
        created_at = workflow_data.get("created_at", "")
        run_started_at = workflow_data.get("run_started_at", "")
        completed_at = workflow_data.get("completed_at") or workflow_data.get("updated_at", "")
        
        duration_sec = self.calculate_time_diff(run_started_at, completed_at)
        queue_time_sec = self.calculate_time_diff(created_at, run_started_at)
        
        # Extract retry information from the workflow data
        run_attempt = workflow_data.get("run_attempt", 1)
        retry_count = max(0, run_attempt - 1)  # run_attempt 1 = 0 retries, 2 = 1 retry, etc.
        
        # Log retry information if this is a retry
        if retry_count > 0:
            print(f"   🔄 Workflow retry detected: Attempt #{run_attempt} (Retry #{retry_count})")
        
        # Create workflow data
        # Include attempt number in ID to store each retry separately
        workflow_metrics = {
            "_id": f"github-workflow-{run_id}-attempt-{run_attempt}",
            "s_user_alias": workflow_data.get("actor", {}).get("login", ""),
            "s_repo": self.repo,
            "s_workflow_name": workflow_name,
            "s_github_event": workflow_data.get("event", ""),
            "s_branch": workflow_data.get("head_branch", ""),
            "s_pr_id": self.extract_pr_id(workflow_data),
            "s_status": status,
            "l_status_number": status_number,
            "s_workflow_id": run_id,
            "s_commit_sha": workflow_data.get("head_sha", ""),
            "l_run_attempt": run_attempt,
            "l_retry_count": retry_count,
            "ts_creation_time": created_at,
            "ts_start_time": run_started_at,
            "ts_end_time": completed_at,
            "l_queue_time_sec": queue_time_sec,
            "l_duration_sec": duration_sec,
            "@timestamp": completed_at or datetime.now(timezone.utc).isoformat()
        }
        
        # Add annotation data
        self.add_annotation_fields(
            workflow_metrics,
            annotation_type="workflow",
            entity_id=run_id,
            entity_name=workflow_name
        )
        
        return self.post_to_opensearch(self.workflow_index, workflow_metrics)
    
    def upload_job_metrics(
        self,
        job_data: Dict[str, Any],
        workflow_data: Dict[str, Any],
        error_classification: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Upload job metrics"""
        job_id = str(job_data["id"])
        job_name = job_data.get("name", "")
        
        """
        # Skip excluded jobs
        if job_name in ["Upload Workflow Metrics"]:
            print(f"⏭️  Skipping excluded job: {job_name}")
            return True
        """
        
        # Determine final status
        status = job_data.get("conclusion") or job_data.get("status", "unknown")
        status_number = self.get_status_number(status)
        
        # Calculate timing
        created_at = job_data.get("created_at", "")
        started_at = job_data.get("started_at", "")
        completed_at = job_data.get("completed_at", "")
        
        duration_sec = self.calculate_time_diff(started_at, completed_at)
        queue_time_sec = self.calculate_time_diff(created_at, started_at)
        
        # Extract runner information
        runner_name = job_data.get("runner_name", "")
        runner_prefix = self.extract_runner_prefix(runner_name)
        
        # Extract retry information (from workflow data)
        run_attempt = workflow_data.get("run_attempt", 1)
        retry_count = max(0, run_attempt - 1)
        
        # Log if this job is part of a retry attempt
        if run_attempt > 1:
            print(f"   🔄 Job '{job_name}' - Workflow Attempt #{run_attempt} (Retry #{retry_count})")
        
        # Create job data
        # Include attempt number in ID to store each retry separately
        job_metrics = {
            "_id": f"github-job-{job_id}-attempt-{run_attempt}",
            "s_job_id": job_id,
            "s_job_name": job_name,
            "s_status": status,
            "l_status_number": status_number,
            "s_runner_id": str(job_data.get("runner_id", "")),
            "s_runner_name": runner_name,
            "s_runner_prefix": runner_prefix,
            "s_user_alias": workflow_data.get("actor", {}).get("login", ""),
            "s_repo": self.repo,
            "s_workflow_name": workflow_data.get("name", ""),
            "s_github_event": workflow_data.get("event", ""),
            "s_branch": workflow_data.get("head_branch", ""),
            "s_pr_id": self.extract_pr_id(workflow_data),
            "s_workflow_id": str(workflow_data["id"]),
            "s_commit_sha": workflow_data.get("head_sha", ""),
            "l_run_attempt": run_attempt,
            "l_retry_count": retry_count,
            "ts_creation_time": created_at,
            "ts_start_time": started_at,
            "ts_end_time": completed_at,
            "l_queue_time_sec": queue_time_sec,
            "l_duration_sec": duration_sec,
            "@timestamp": completed_at or datetime.now(timezone.utc).isoformat()
        }
        
        # Add annotation data
        self.add_annotation_fields(
            job_metrics,
            annotation_type="job",
            entity_id=job_id,
            entity_name=job_name
        )

        # Add error classification for failed jobs (pre-computed)
        if error_classification:
            self.add_error_classification_fields(job_metrics, error_classification)

        return self.post_to_opensearch(self.jobs_index, job_metrics)
    
    def upload_step_metrics(
        self,
        step_data: Dict[str, Any],
        job_data: Dict[str, Any],
        workflow_data: Dict[str, Any],
        step_index: int,
        error_classification: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Upload step metrics"""
        job_id = str(job_data["id"])
        step_name = step_data.get("name", f"step_{step_index}")
        step_number = step_data.get("number", step_index + 1)
        step_id = f"{job_id}_{step_number}"
        
        # Determine final status
        status = step_data.get("conclusion") or step_data.get("status", "unknown")
        status_number = self.get_status_number(status)
        
        # Calculate timing
        started_at = step_data.get("started_at", "")
        completed_at = step_data.get("completed_at", "")
        duration_sec = self.calculate_time_diff(started_at, completed_at)
        
        # Determine command
        command = ""
        if step_data.get("action"):
            command = f"uses: {step_data['action']}"
        elif "run" in step_name.lower() or "script" in step_name.lower():
            command = "run: <script>"
        
        # Extract retry information (from workflow data)
        run_attempt = workflow_data.get("run_attempt", 1)
        retry_count = max(0, run_attempt - 1)
        
        # Create step data
        # Include attempt number in ID to store each retry separately
        step_metrics = {
            "_id": f"github-step-{step_id}-attempt-{run_attempt}",
            "s_step_run_id": step_id,  # Link all attempts of this step together
            "s_step_id": step_id,
            "s_job_id": job_id,
            "s_step_name": step_name,
            "l_step_number": step_number,
            "s_status": status,
            "l_status_number": status_number,
            "s_job_name": job_data.get("name", ""),
            "s_command": command,
            "s_user_alias": workflow_data.get("actor", {}).get("login", ""),
            "s_repo": self.repo,
            "s_workflow_name": workflow_data.get("name", ""),
            "s_github_event": workflow_data.get("event", ""),
            "s_branch": workflow_data.get("head_branch", ""),
            "s_pr_id": self.extract_pr_id(workflow_data),
            "s_workflow_id": str(workflow_data["id"]),
            "s_commit_sha": workflow_data.get("head_sha", ""),
            "l_run_attempt": run_attempt,
            "l_retry_count": retry_count,
            "ts_start_time": started_at,
            "ts_end_time": completed_at,
            "l_duration_sec": duration_sec,
            "@timestamp": completed_at or datetime.now(timezone.utc).isoformat()
        }

        # Add error classification for failed steps (pre-computed)
        if error_classification:
            self.add_error_classification_fields(step_metrics, error_classification)

        return self.post_to_opensearch(self.steps_index, step_metrics)
    
    def process_workflows(self):
        """Main processing function"""
        print("🚀 Starting workflow metrics upload process...")
        
        # Fetch workflows
        workflows = self.fetch_workflow_runs_timerange(self.hours_back)
        
        if not workflows:
            print("ℹ️  No workflows found in the specified time range")
            return
        
        workflows_processed = 0
        jobs_processed = 0
        steps_processed = 0
        
        for workflow in workflows:
            try:
                run_id = workflow["id"]
                workflow_name = workflow.get("name", "")
                
                print(f"\n📋 Processing workflow: {workflow_name} (ID: {run_id})")
                
                # Check if we need to backfill previous attempts
                run_attempt = workflow.get("run_attempt", 1)
                if run_attempt > 1:
                    self.backfill_previous_attempts(workflow)
                
                # Upload workflow metrics for current attempt
                if self.upload_workflow_metrics(workflow):
                    workflows_processed += 1
                
                # Fetch and upload job metrics
                jobs_data = self.fetch_job_details(str(run_id))
                if jobs_data and "jobs" in jobs_data:
                    for job in jobs_data["jobs"]:
                        job_name = job.get("name", "")
                        print(f"  📤 Processing job: {job_name}")

                        # Classify errors once for the entire job (if enabled and job failed)
                        classifications = self.classify_job_errors(job, workflow)
                        job_classification = classifications.get("job_classification")
                        step_classifications = classifications.get("step_classifications", {})

                        # Upload job metrics with classification
                        if self.upload_job_metrics(job, workflow, job_classification):
                            jobs_processed += 1

                        # Upload step metrics with pre-computed classifications
                        steps = job.get("steps", [])
                        for step_index, step in enumerate(steps):
                            step_name = step.get("name", f"step_{step_index}")
                            step_classification = step_classifications.get(step_name)
                            if self.upload_step_metrics(step, job, workflow, step_index, step_classification):
                                steps_processed += 1
                
                sleep(0.5)  # Rate limiting between workflows
                
            except Exception as e:
                print(f"❌ Error processing workflow {workflow.get('id', 'unknown')}: {e}")
                continue
        
        print(f"\n✅ Upload completed!")
        print(f"   Workflows: {workflows_processed}")
        print(f"   Jobs: {jobs_processed}")
        print(f"   Steps: {steps_processed}")

def main():
    try:
        uploader = WorkflowMetricsUploader()
        uploader.process_workflows()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
