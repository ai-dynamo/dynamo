"""
Enhanced script to upload complete GitHub Actions workflow and job metrics to Prometheus via OpenTelemetry.
This version runs as the final job in a workflow and captures metrics for 
the entire workflow including all previous jobs.

CONVERSION NOTES:
- Converted from OpenSearch HTTP POST uploads to OpenTelemetry metrics
- Uses OTLP gRPC exporter to send metrics to Prometheus via NVIDIA Observability Service
- Maintains all original metric collection functionality
- Metrics are now recorded as histograms, counters, and gauges with proper labels
- Requires OTLP_ENDPOINT, NVAUTH_TOKEN, and SERVICE_ID environment variables
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
import re

# OpenTelemetry imports
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
import requests

# OpenTelemetry Configuration - Replace with your actual values
OTLP_ENDPOINT = os.getenv('OTLP_ENDPOINT')
AUTH_TOKEN = os.getenv('NVAUTH_TOKEN')  # Replace with your NVAuth token
SERVICE_NAME = "Dynamo Metrics"
SERVICE_ID = os.getenv('SERVICE_ID')  # Replace with your Service Registry ID

# FILTERING CONFIGURATION - Only upload data for specific workflow/job combinations
TARGET_WORKFLOW_NAME = "Docker Build and Test"
TARGET_JOB_NAMES = [
    "vllm",
    "sglang", 
    "trtllm"
]

# NEW STANDARDIZED FIELD SCHEMA - Using consistent prefixes for OpenSearch mapping
# Using prefixes: s_ for strings, l_ for longs, ts_ for timestamps

# Common fields across all metric types
FIELD_ID = "_id"
FIELD_USER_ALIAS = "s_user_alias" #extra you can maybe delete or test
FIELD_REPO = "s_repo"
FIELD_WORKFLOW_NAME = "s_workflow_name"
FIELD_GITHUB_EVENT = "s_github_event"
FIELD_BRANCH = "s_branch" #extra you can maybe delete or test
FIELD_PR_ID = "s_pr_id"  # Pull request ID as string ("N/A" if not a PR)
FIELD_STATUS = "s_status" #duplicate you can maybe consolidate to the common metric adding
FIELD_STATUS_NUMBER = "l_status_number"
FIELD_WORKFLOW_ID = "s_workflow_id"

# Timing fields
FIELD_CREATION_TIME = "ts_creation_time"
FIELD_START_TIME = "ts_start_time"
FIELD_END_TIME = "ts_end_time"
FIELD_QUEUE_TIME = "l_queue_time_sec"  # Integer seconds as long
FIELD_DURATION_SEC = "l_duration_sec"

# Workflow-specific fields
FIELD_COMMIT_SHA = "s_commit_sha"
#FIELD_JOBS = "s_jobs"  # Comma-separated job IDs

# Job-specific fields
FIELD_JOB_ID = "s_job_id"
FIELD_JOB_NAME = "s_job_name"
#FIELD_RUNNER_INFO = "s_runner_info"
FIELD_RUNNER_ID = "s_runner_id"
FIELD_RUNNER_NAME = "s_runner_name"
#FIELD_LABELS = "s_labels"  # Comma-separated labels
#FIELD_STEPS = "s_steps"  # Comma-separated step IDs

# Step-specific fields
FIELD_STEP_ID = "s_step_id"
FIELD_NAME = "s_step_name"
FIELD_STEP_NUMBER = "l_step_number"
FIELD_COMMAND = "s_command"
#FIELD_JOB_LABELS = "s_job_labels"  # Comma-separated labels

# Build-specific fields (Docker image build metrics)
FIELD_BUILD_DURATION_SEC = "l_build_duration_sec"  # Docker build time in seconds
FIELD_IMAGE_SIZE_BYTES = "l_image_size_bytes"      # Docker image size in bytes
FIELD_IMAGE_SIZE_MB = "l_image_size_mb"            # Docker image size in MB
FIELD_BUILD_START_TIME = "ts_build_start_time"     # Build start timestamp
FIELD_BUILD_END_TIME = "ts_build_end_time"         # Build end timestamp
FIELD_BUILD_FRAMEWORK = "s_build_framework"        # Framework (e.g., vllm)
FIELD_BUILD_TARGET = "s_build_target"              # Build target (e.g., runtime)

# Container-specific fields for CONTAINER_INDEX (only truly unique fields)
# All common fields (s_step_id, s_job_id, etc.) and build timing fields are reused
CONTAINER_FIELD_FRAMEWORK = "s_framework"          # Framework (vllm, etc.)
CONTAINER_FIELD_SIZE_MB = "l_size_mb"              # Container size in MB  
CONTAINER_FIELD_CACHE_HIT_RATE = "l_cache_hit_rate" # Cache hit rate percentage

def setup_telemetry():
    """Configure OpenTelemetry with NVIDIA Observability Service"""
    
    # Create resource with required labels
    resource = Resource.create({
        "service.name": SERVICE_NAME,
        "service.version": "1.0.0",
        "Authorization": AUTH_TOKEN,
        "service.id": SERVICE_ID,
        # Optional: customize service name
        "service.name.override": f"{SERVICE_NAME}-github-actions"
    })

    # Configure metrics
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(
            endpoint=OTLP_ENDPOINT,
            headers={"authorization": f"Bearer {AUTH_TOKEN}"}
        ),
        export_interval_millis=5000  # Export every 5 seconds
    )

    metrics.set_meter_provider(MeterProvider(
        resource=resource,
        metric_readers=[metric_reader]
    ))

class BuildMetricsReader:
    """Reader for Docker build metrics from environment variables and artifacts"""
    
    @staticmethod
    def get_build_metrics(framework: str = None) -> Optional[Dict[str, Any]]:
        """Get build metrics from environment variables and/or JSON file for a specific framework"""
        metrics = {}
        
        # If framework is provided, try framework-specific environment variables first
        if framework:
            framework_upper = framework.upper()
            env_metrics = {
                'build_duration_sec': os.getenv(f'{framework_upper}_BUILD_DURATION_SEC'),
                'image_size_bytes': os.getenv(f'{framework_upper}_IMAGE_SIZE_BYTES'),
                'image_size_mb': os.getenv(f'{framework_upper}_IMAGE_SIZE_MB'),
                'build_start_time': os.getenv(f'{framework_upper}_BUILD_START_TIME'),
                'build_end_time': os.getenv(f'{framework_upper}_BUILD_END_TIME'),
                'cache_hit_rate': os.getenv(f'{framework_upper}_CACHE_HIT_RATE'),
                'framework': framework,
                'target': 'runtime'  # Default target
            }
        else:
            # Fallback to generic environment variables
            env_metrics = {
                'build_duration_sec': os.getenv('BUILD_DURATION_SEC'),
                'image_size_bytes': os.getenv('IMAGE_SIZE_BYTES'),
                'image_size_mb': os.getenv('IMAGE_SIZE_MB'),
                'build_start_time': os.getenv('BUILD_START_TIME'),
                'build_end_time': os.getenv('BUILD_END_TIME'),
                'cache_hit_rate': os.getenv('CACHE_HIT_RATE'),
                'framework': os.getenv('BUILD_FRAMEWORK'),
                'target': os.getenv('BUILD_TARGET')
            }
        
        # Filter out None values and convert to appropriate types
        for key, value in env_metrics.items():
            if value is not None:
                if key in ['build_duration_sec', 'image_size_bytes', 'image_size_mb', 'cache_hit_rate']:
                    try:
                        metrics[key] = int(value)
                    except (ValueError, TypeError):
                        print(f"⚠️  Invalid numeric value for {key}: {value}")
                elif key in ['build_start_time', 'build_end_time']:
                    try:
                        # Convert Unix timestamp to ISO format
                        metrics[key] = datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()
                    except (ValueError, TypeError):
                        print(f"⚠️  Invalid timestamp for {key}: {value}")
                else:
                    metrics[key] = str(value)
        
        # Try to read from JSON file as fallback
        json_file_path = "build-metrics/metrics.json"
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r') as f:
                    json_metrics = json.load(f)
                    # Merge JSON data, preferring environment variables
                    for key, value in json_metrics.items():
                        if key not in metrics and value is not None:
                            metrics[key] = value
                print(f"📁 Loaded additional metrics from {json_file_path}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Could not read build metrics JSON: {e}")
        
        if metrics:
            print(f"📊 Build metrics loaded: {list(metrics.keys())}")
            return metrics
        else:
            print("ℹ️  No build metrics available")
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
        # Setup OpenTelemetry
        setup_telemetry()
        self.meter = metrics.get_meter(__name__)
        
        # Create metric instruments
        self.workflow_duration_histogram = self.meter.create_histogram(
            name="github_workflow_duration_seconds",
            description="GitHub workflow duration in seconds",
            unit="s"
        )
        
        self.workflow_queue_time_histogram = self.meter.create_histogram(
            name="github_workflow_queue_time_seconds", 
            description="GitHub workflow queue time in seconds",
            unit="s"
        )
        
        self.workflow_status_counter = self.meter.create_counter(
            name="github_workflow_status_total",
            description="Total GitHub workflow runs by status",
            unit="1"
        )
        
        self.job_duration_histogram = self.meter.create_histogram(
            name="github_job_duration_seconds",
            description="GitHub job duration in seconds", 
            unit="s"
        )
        
        self.job_queue_time_histogram = self.meter.create_histogram(
            name="github_job_queue_time_seconds",
            description="GitHub job queue time in seconds",
            unit="s"
        )
        
        self.job_status_counter = self.meter.create_counter(
            name="github_job_status_total",
            description="Total GitHub job runs by status",
            unit="1"
        )
        
        self.step_duration_histogram = self.meter.create_histogram(
            name="github_step_duration_seconds",
            description="GitHub step duration in seconds",
            unit="s"
        )
        
        self.step_status_counter = self.meter.create_counter(
            name="github_step_status_total", 
            description="Total GitHub step runs by status",
            unit="1"
        )
        
        self.build_duration_histogram = self.meter.create_histogram(
            name="github_build_duration_seconds",
            description="Docker build duration in seconds",
            unit="s"
        )
        
        self.image_size_gauge = self.meter.create_up_down_counter(
            name="github_image_size_mb",
            description="Docker image size in MB",
            unit="MB"
        )
        
        self.cache_hit_rate_gauge = self.meter.create_up_down_counter(
            name="github_cache_hit_rate_percent",
            description="Docker build cache hit rate percentage",
            unit="%"
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
        """Record workflow metrics using OpenTelemetry"""
        print(f"Recording workflow metrics via OpenTelemetry: {data.get('_id', 'unknown')}")
        
        # Create labels for metrics
        labels = {
            "repo": self.repo,
            "workflow_name": self.workflow_name,
            "branch": data.get(FIELD_BRANCH, ""),
            "event": self.event_name,
            "status": data.get(FIELD_STATUS, "unknown"),
            "actor": self.actor
        }
        
        # Record duration
        duration = data.get(FIELD_DURATION_SEC, 0)
        if duration > 0:
            self.workflow_duration_histogram.record(duration, labels)
            
        # Record queue time
        queue_time = data.get(FIELD_QUEUE_TIME, 0)
        if queue_time > 0:
            self.workflow_queue_time_histogram.record(queue_time, labels)
            
        # Record status counter
        self.workflow_status_counter.add(1, labels)
        
        print(f"Successfully recorded workflow metrics for: {data.get('_id', 'unknown')}")
    
    def record_job_metrics(self, data: Dict[str, Any]) -> None:
        """Record job metrics using OpenTelemetry"""
        print(f"Recording job metrics via OpenTelemetry: {data.get('_id', 'unknown')}")
        
        # Create labels for metrics
        labels = {
            "repo": self.repo,
            "workflow_name": self.workflow_name,
            "job_name": data.get(FIELD_JOB_NAME, ""),
            "branch": data.get(FIELD_BRANCH, ""),
            "event": self.event_name,
            "status": data.get(FIELD_STATUS, "unknown"),
            "actor": self.actor
        }
        
        # Record duration
        duration = data.get(FIELD_DURATION_SEC, 0)
        if duration > 0:
            self.job_duration_histogram.record(duration, labels)
            
        # Record queue time
        queue_time = data.get(FIELD_QUEUE_TIME, 0)
        if queue_time > 0:
            self.job_queue_time_histogram.record(queue_time, labels)
            
        # Record status counter
        self.job_status_counter.add(1, labels)
        
        print(f"Successfully recorded job metrics for: {data.get(FIELD_JOB_NAME, 'unknown')}")
    
    def record_step_metrics(self, data: Dict[str, Any]) -> None:
        """Record step metrics using OpenTelemetry"""
        print(f"Recording step metrics via OpenTelemetry: {data.get('_id', 'unknown')}")
        
        # Create labels for metrics
        labels = {
            "repo": self.repo,
            "workflow_name": self.workflow_name,
            "job_name": data.get(FIELD_JOB_NAME, ""),
            "step_name": data.get(FIELD_NAME, ""),
            "branch": data.get(FIELD_BRANCH, ""),
            "event": self.event_name,
            "status": data.get(FIELD_STATUS, "unknown"),
            "actor": self.actor
        }
        
        # Record duration
        duration = data.get(FIELD_DURATION_SEC, 0)
        if duration > 0:
            self.step_duration_histogram.record(duration, labels)
            
        # Record status counter
        self.step_status_counter.add(1, labels)
        
        print(f"Successfully recorded step metrics for: {data.get(FIELD_NAME, 'unknown')}")
    
    def record_build_metrics(self, data: Dict[str, Any]) -> None:
        """Record build/container metrics using OpenTelemetry"""
        print(f"Recording build metrics via OpenTelemetry: {data.get('_id', 'unknown')}")
        
        # Create labels for metrics
        labels = {
            "repo": self.repo,
            "workflow_name": self.workflow_name,
            "framework": data.get(CONTAINER_FIELD_FRAMEWORK, "unknown"),
            "branch": data.get(FIELD_BRANCH, ""),
            "event": self.event_name,
            "status": data.get(FIELD_STATUS, "unknown"),
            "actor": self.actor
        }
        
        # Record build duration
        build_duration = data.get(FIELD_BUILD_DURATION_SEC, 0)
        if build_duration > 0:
            self.build_duration_histogram.record(build_duration, labels)
            
        # Record image size
        image_size = data.get(CONTAINER_FIELD_SIZE_MB, 0)
        if image_size > 0:
            self.image_size_gauge.add(image_size, labels)
            
        # Record cache hit rate
        cache_hit_rate = data.get(CONTAINER_FIELD_CACHE_HIT_RATE, 0)
        if cache_hit_rate > 0:
            self.cache_hit_rate_gauge.add(cache_hit_rate, labels)
        
        print(f"Successfully recorded build metrics for: {data.get(CONTAINER_FIELD_FRAMEWORK, 'unknown')}")

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


    def add_common_context_fields(self, db_data: Dict[str, Any], workflow_data: Dict[str, Any] = None) -> None:
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
            pull_requests = workflow_data.get('pull_requests', [])
            if pull_requests and len(pull_requests) > 0:
                pr_number = pull_requests[0].get('number')
                if pr_number:
                    pr_id = str(pr_number)
        db_data[FIELD_PR_ID] = pr_id

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
        
        # Duration in integer seconds (using l_ prefix for long type)
        db_data[FIELD_DURATION_SEC] = TimingProcessor.calculate_time_diff(start_time, end_time)
        
        # Queue time in integer seconds (using l_ prefix for long type)
        if metric_type != "step":
            db_data[FIELD_QUEUE_TIME] = TimingProcessor.calculate_time_diff(creation_time, start_time)
        
        # Add @timestamp field for Grafana/OpenSearch indexing (CRITICAL FIX!)
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
            print(f"Workflow data: {workflow_data}")
            if not workflow_data:
                print("Could not fetch workflow data from GitHub API")
                return
                
            jobs_data = self.get_github_api_data(f"/repos/{self.repo}/actions/runs/{self.run_id}/jobs")
            print(f"Jobs data: {jobs_data}")
            if not jobs_data or 'jobs' not in jobs_data:
                print("Could not fetch jobs data from GitHub API")
                return
            
            # FILTER: Only upload data for specific workflow and job
            workflow_name = workflow_data.get('name', '')
            
            if workflow_name != TARGET_WORKFLOW_NAME:
                print(f"❌ Skipping upload - Workflow '{workflow_name}' does not match target '{TARGET_WORKFLOW_NAME}'")
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
        db_data[FIELD_ID] = f"github-workflow-{self.run_id}"
        
        
        # Schema fields
        # Use conclusion for completed workflows, fallback to status
        db_data[FIELD_STATUS] = str(workflow_data.get('conclusion') or workflow_data.get('status', 'unknown'))
        if db_data[FIELD_STATUS] is "success":
            db_data[FIELD_STATUS_NUMBER] = 1
        elif db_data[FIELD_STATUS] is "failure":
            db_data[FIELD_STATUS_NUMBER] = 0
        #db_data[FIELD_BRANCH] = str(workflow_data.get('head_branch', self.ref_name))
        print(f"Checking branch: {str(workflow_data.get('head_branch'))}")
        #db_data[FIELD_COMMIT_SHA] = str(workflow_data.get('head_sha', self.sha))        
        # Timing fields - Fix parameter order for correct duration/queue time calculation
        created_at = workflow_data.get('created_at')
        run_started_at = workflow_data.get('run_started_at')
        # Use completed_at if available, otherwise updated_at
        end_time = workflow_data.get('completed_at') or workflow_data.get('updated_at')
        self.add_standardized_timing_fields(db_data, created_at, run_started_at, end_time, "workflow")
        
        # Common context fields
        self.add_common_context_fields(db_data, workflow_data)
        
        # Override userAlias with actor from API if available
        """
        actor = workflow_data.get('actor', {})
        if actor and actor.get('login'):
            db_data[FIELD_USER_ALIAS] = actor.get('login')
        """
        actor = workflow_data.get('actor', {})
        print(f"Checking actor: {actor.get('login')}")
        
        # Add jobs list as comma-separated string (using s_ prefix)
        """
        if jobs_data and 'jobs' in jobs_data:
            job_ids = [str(job['id']) for job in jobs_data['jobs']]
            db_data[FIELD_JOBS] = ','.join(job_ids)
        else:
            db_data[FIELD_JOBS] = ''
        """
        
        self.record_workflow_metrics(db_data)

    def _upload_all_job_and_step_metrics(self, jobs_data: Dict[str, Any]) -> tuple[int, int]:
        """Internal method to upload all job and step metrics, returns (jobs_processed, steps_processed)"""
        jobs_processed = 0
        steps_processed = 0
        
        # Use the configured target job name
        
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
                if self.steps_index:
                    step_count = self._upload_job_step_metrics(job)
                    steps_processed += step_count
                    
            except Exception as e:
                print(f"Error uploading metrics for job {job.get('name', 'unknown')}: {e}")
                continue
        
        return jobs_processed, steps_processed

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
        
        db_data[FIELD_ID] = f"github-job-{job_id}"
        
        # Schema fields
        db_data[FIELD_JOB_ID] = str(job_id)
        # Handle job status - prefer conclusion for completed jobs, fallback to status
        job_status = str(job_data.get('conclusion') or job_data.get('status', 'unknown'))
        # Don't upload jobs with null/None status as they cause Grafana filtering issues
        if job_status is None:
            job_status = 'in_progress'
        db_data[FIELD_STATUS] = str(job_status)
        if db_data[FIELD_STATUS] is "success":
            db_data[FIELD_STATUS_NUMBER] = 1
        elif db_data[FIELD_STATUS] is "failure":
            db_data[FIELD_STATUS_NUMBER] = 0
        #db_data[FIELD_RUNNER_INFO] = str(job_data.get('runner_name', 'unknown'))
        
        db_data[FIELD_JOB_NAME] = str(job_name)
                
        # Timing fields using standardized method - Fix parameter order
        created_at = job_data.get('created_at')
        started_at = job_data.get('started_at')
        completed_at = job_data.get('completed_at')
        
        self.add_standardized_timing_fields(db_data, created_at, started_at, completed_at, "job")
        
        # Labels - Convert array to comma-separated string to avoid indexing issues
        """
        runner_labels = job_data.get('labels', [])
        if runner_labels:
            db_data[FIELD_LABELS] = ','.join(runner_labels)
        else:
            db_data[FIELD_LABELS] = 'unknown'
        """
        
        # Add steps list (get step IDs) - Convert to string to avoid array issues
        """
        steps = job_data.get('steps', [])
        if steps:
            step_ids = [f"{job_id}_{step.get('number', i+1)}" for i, step in enumerate(steps)]
            db_data[FIELD_STEPS] = ','.join(step_ids)  # Convert array to comma-separated string
        else:
            db_data[FIELD_STEPS] = ''
        """
        
        # Runner info
        runner_id = job_data.get('runner_id')
        db_data[FIELD_RUNNER_ID] = str(runner_id) if runner_id is not None else ''
        db_data[FIELD_RUNNER_NAME] = str(job_data.get('runner_name', ''))
        
        # Add common context fields
        self.add_common_context_fields(db_data, None)
        
        self.record_job_metrics(db_data)
        print(f"Uploaded metrics for job: {job_name}")
        
        # Upload container metrics if this is a target job and metrics are available
        if job_name in TARGET_JOB_NAMES:
            self._upload_container_metrics(job_data, job_name)

    def _upload_container_metrics(self, job_data: Dict[str, Any], framework: str, build_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Upload container-specific metrics via OpenTelemetry"""
        
        # Get build metrics if not provided
        if build_metrics is None:
            build_metrics = BuildMetricsReader.get_build_metrics(framework)
        
        if not build_metrics:
            print("⚠️  No build metrics available for container upload")
            return
        
        print(f"📦 Recording container metrics via OpenTelemetry")
        
        # Create container metrics payload
        container_data = {}
        
        # Identity & Context - using common field names
        job_id = str(job_data['id'])
        container_data[FIELD_ID] = f"github-container-{job_id}-{build_metrics.get('framework', 'unknown')}"
        container_data[FIELD_JOB_ID] = job_id
        container_data[FIELD_WORKFLOW_ID] = str(self.run_id)
        container_data[FIELD_REPO] = self.repo
        container_data[FIELD_WORKFLOW_NAME] = self.workflow_name
        container_data[FIELD_BRANCH] = self.ref_name
        
        # Find the "Build image" step ID
        build_step_id = None
        steps = job_data.get('steps', [])
        for step in steps:
            if 'build' in step.get('name', '').lower() and 'image' in step.get('name', '').lower():
                build_step_id = f"{job_id}_{step.get('number', 1)}"
                break
                
        # Status & Events - using common field names
        container_data[FIELD_STATUS] = str(job_data.get('conclusion') or job_data.get('status', 'unknown'))
        container_data[FIELD_GITHUB_EVENT] = self.event_name
        if container_data[FIELD_STATUS] is "success":
            container_data[FIELD_STATUS_NUMBER] = 1
        elif container_data[FIELD_STATUS] is "failure":
            container_data[FIELD_STATUS_NUMBER] = 0
        
        # Container Info (only truly container-specific fields)
        container_data[CONTAINER_FIELD_FRAMEWORK] = build_metrics.get('framework', 'unknown')
        container_data[CONTAINER_FIELD_SIZE_MB] = build_metrics.get('image_size_mb', 0)
        container_data[CONTAINER_FIELD_CACHE_HIT_RATE] = build_metrics.get('cache_hit_rate', 0)
        
        # Timing (reusing existing build timing fields)
        if 'build_start_time' in build_metrics:
            container_data[FIELD_BUILD_START_TIME] = build_metrics['build_start_time']
        if 'build_end_time' in build_metrics:
            container_data[FIELD_BUILD_END_TIME] = build_metrics['build_end_time']
        container_data[FIELD_BUILD_DURATION_SEC] = build_metrics.get('build_duration_sec', 0)
        
        # Add @timestamp for time-series data
        container_data['@timestamp'] = build_metrics.get('build_end_time', datetime.now(timezone.utc).isoformat())
        
        # Record container metrics via OpenTelemetry
        try:
            print(f"🔍 Debug: Container data being recorded:")
            print(f"   Data: {container_data}")
            
            self.record_build_metrics(container_data)
            print(f"✅ Container metrics recorded successfully")
            print(f"   Framework: {build_metrics.get('framework', 'N/A')}")
            print(f"   Size: {build_metrics.get('image_size_mb', 'N/A')} MB")
            print(f"   Cache Hit Rate: {build_metrics.get('cache_hit_rate', 'N/A')}%")
            print(f"   Build Duration: {build_metrics.get('build_duration_sec', 'N/A')} seconds")
        except Exception as e:
            print(f"❌ Failed to record container metrics: {e}")
            print(f"🔍 Debug: Container data that failed: {container_data}")

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
        db_data[FIELD_ID] = f"github-step-{step_id}"
        
        # Schema-compliant fields
        db_data[FIELD_STEP_ID] = str(step_id)
        db_data[FIELD_JOB_ID] = str(job_id)
        db_data[FIELD_NAME] = str(step_name)
        db_data[FIELD_STEP_NUMBER] = int(step_number)  # Using l_ prefix, should be integer
        db_data[FIELD_STATUS] = str(step_data.get('conclusion', step_data.get('status', 'unknown')))
        db_data[FIELD_JOB_NAME] = str(job_name)
        if db_data[FIELD_STATUS] is "success":
            db_data[FIELD_STATUS_NUMBER] = 1
        elif db_data[FIELD_STATUS] is "failure":
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
        
        # Job context - Ensure all fields are strings
        #db_data[FIELD_RUNNER_NAME] = str(job_data.get('runner_name', ''))
        #db_data[FIELD_RUNNER_ID] = str(job_data.get('runner_id')) if job_data.get('runner_id') is not None else ''
        
        # Job labels (separate from step labels) - Convert array to string
        """
        runner_labels = job_data.get('labels', [])
        if runner_labels:
            db_data[FIELD_JOB_LABELS] = ','.join(runner_labels)
        else:
            db_data[FIELD_JOB_LABELS] = 'unknown'
        """
        
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
    
    # Force final export of all metrics
    print("Flushing metrics to ensure delivery...")
    metrics.get_meter_provider().force_flush(30_000)  # 30 second timeout
    print("All metrics have been sent to Prometheus via OpenTelemetry!")

if __name__ == "__main__":
    main()
