#!/usr/bin/env python3
"""
GitHub Actions Artifact Lister - Detailed Build Metrics
Lists artifacts and uploads detailed container, stage, and layer metrics to OpenSearch
Handles nested JSON format with build stages and individual layers
"""

import os
import sys
import json
import argparse
import tempfile
import zipfile
from datetime import datetime, timezone, timedelta
from time import sleep
from typing import Dict, Any, List, Optional
from pathlib import Path
import requests


# Container-level field constants
FIELD_JOB_NAME = "s_job_name"
FIELD_JOB_ID = "s_job_id"
FIELD_STEP_ID = "s_step_id"
FIELD_STATUS = "s_status"
FIELD_BUILD_DURATION_SEC = "l_build_duration_sec"
FIELD_BUILD_START_TIME = "ts_build_start_time"
FIELD_BUILD_END_TIME = "ts_build_end_time"
FIELD_BUILD_FRAMEWORK = "s_build_framework"
FIELD_BUILD_TARGET = "s_build_target"
FIELD_BUILD_PLATFORM = "s_build_platform"
FIELD_BUILD_SIZE_BYTES = "l_build_size_bytes"
FIELD_TOTAL_SIZE_TRANSFERRED = "l_total_size_transferred_bytes"
FIELD_TOTAL_STEPS = "l_total_steps"
FIELD_CACHED_STEPS = "l_cached_steps"
FIELD_BUILT_STEPS = "l_built_steps"
FIELD_CACHE_HIT_RATE = "f_cache_hit_rate"

# Stage-level field constants
FIELD_STAGE_NAME = "s_stage_name"
FIELD_STAGE_TOTAL_STEPS = "l_stage_total_steps"
FIELD_STAGE_CACHED_STEPS = "l_stage_cached_steps"
FIELD_STAGE_BUILT_STEPS = "l_stage_built_steps"
FIELD_STAGE_DURATION_SEC = "f_stage_duration_sec"
FIELD_STAGE_CACHE_HIT_RATE = "f_stage_cache_hit_rate"

# Layer-level field constants
FIELD_LAYER_STEP_NUMBER = "l_step_number"
FIELD_LAYER_STEP_NAME = "s_step_name"
FIELD_LAYER_COMMAND = "s_command"
FIELD_LAYER_STATUS = "s_layer_status"
FIELD_LAYER_CACHED = "b_cached"
FIELD_LAYER_DURATION_SEC = "f_duration_sec"
FIELD_LAYER_SIZE_TRANSFERRED = "l_size_transferred"
FIELD_LAYER_STAGE = "s_stage"

# Common context fields
FIELD_USER_ALIAS = "s_user_alias"
FIELD_REPO = "s_repo"
FIELD_WORKFLOW_NAME = "s_workflow_name"
FIELD_GITHUB_EVENT = "s_github_event"
FIELD_BRANCH = "s_branch"
FIELD_WORKFLOW_ID = "s_workflow_id"
FIELD_COMMIT_SHA = "s_commit_sha"
FIELD_PR_ID = "s_pr_id"
FIELD_RUN_ID = "s_run_id"


class DetailedArtifactLister:
    """Lists artifacts and uploads detailed build metrics to OpenSearch"""

    def __init__(self, repo: str, github_token: str, upload_to_opensearch: bool = True):
        self.repo = repo
        self.github_token = github_token
        self.github_headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
        }
        self.base_url = "https://api.github.com"
        self.upload_to_opensearch = upload_to_opensearch

    def get_github_api_data(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Generic method to fetch data from GitHub API with error handling"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, headers=self.github_headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(
                    f"⚠️  GitHub API returned status {response.status_code} for {endpoint}"
                )
                return None
        except Exception as e:
            print(f"❌ Error fetching from GitHub API {endpoint}: {e}")
            return None

    def fetch_workflow_runs_timerange(self, hours_back: float) -> List[Dict[str, Any]]:
        """Fetch workflow runs within the specified time range"""
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(hours=hours_back)

        print(f"🔍 Fetching workflow runs from {self.repo} - Last {hours_back} hours")
        print(f"   Time range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC to {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")

        all_runs = []
        page = 1
        per_page = 100
        consecutive_old_runs = 0
        max_consecutive_old = 200

        while True:
            params = {
                "per_page": per_page,
                "page": page,
            }

            data = self.get_github_api_data(
                f"/repos/{self.repo}/actions/runs", params=params
            )

            if not data or "workflow_runs" not in data:
                break

            runs = data["workflow_runs"]
            print(f"📄 Page {page}: Found {len(runs)} runs")

            if not runs:
                break

            runs_in_range = 0
            runs_outside_range = 0

            for run in runs:
                created_at = run.get("created_at", "")
                updated_at = run.get("updated_at", "")

                try:
                    created_dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                        tzinfo=timezone.utc
                    )
                    updated_dt = datetime.strptime(updated_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                        tzinfo=timezone.utc
                    )

                    # Include if created OR updated in time range
                    if created_dt >= start_time or updated_dt >= start_time:
                        all_runs.append(run)
                        runs_in_range += 1
                        consecutive_old_runs = 0
                    else:
                        runs_outside_range += 1
                        consecutive_old_runs += 1

                except Exception as e:
                    print(f"⚠️  Error parsing timestamp: {e}")
                    continue

            print(
                f"   Added {runs_in_range} runs from time range, {runs_outside_range} outside range"
            )
            print(f"   Total collected: {len(all_runs)} runs")

            # Stop if we've seen too many old runs consecutively
            if consecutive_old_runs >= max_consecutive_old:
                print(f"   Stopping: {consecutive_old_runs} consecutive old runs")
                break

            # Stop if we got fewer results than requested (last page)
            if len(runs) < per_page:
                break

            page += 1
            sleep(0.5)  # Rate limiting

        print(f"✅ Total runs from last {hours_back}h: {len(all_runs)}")
        return all_runs

    def fetch_artifacts_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Fetch artifacts for a specific workflow run"""
        data = self.get_github_api_data(
            f"/repos/{self.repo}/actions/runs/{run_id}/artifacts"
        )

        if data and "artifacts" in data:
            return data["artifacts"]
        return []

    def fetch_job_details(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fetch details for a specific job"""
        data = self.get_github_api_data(
            f"/repos/{self.repo}/actions/jobs/{job_id}"
        )
        return data

    def fetch_jobs_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Fetch all jobs for a specific workflow run"""
        data = self.get_github_api_data(
            f"/repos/{self.repo}/actions/runs/{run_id}/jobs"
        )
        
        if data and "jobs" in data:
            return data["jobs"]
        return []

    def extract_id_from_artifact_name(self, artifact_name: str) -> Optional[Dict[str, str]]:
        """
        Extract workflow ID and job ID from artifact name
        
        Format: build-metrics-vllm-arm64-19517275889-122222
          - workflow_id: 19517275889
          - job_id: 122222
        
        Returns dict with 'workflow_id' and/or 'job_id' keys, or None if no IDs found
        """
        if not artifact_name.startswith("build-metrics"):
            return None
        
        # Split by hyphen and check if last parts are numeric
        parts = artifact_name.split("-")
        
        # Check for format with two IDs at the end
        if len(parts) >= 6:  # build, metrics, framework, arch, id1, id2
            last_part = parts[-1]
            second_last_part = parts[-2]
            
            if last_part.isdigit() and second_last_part.isdigit():
                # Format: workflow_id and job_id
                if len(second_last_part) >= 10 and len(last_part) >= 5:
                    return {
                        'workflow_id': second_last_part,
                        'job_id': last_part
                    }
        
        # Check for old format with single ID at the end
        if len(parts) >= 5:  # build, metrics, framework, arch, id
            last_part = parts[-1]
            if last_part.isdigit() and len(last_part) >= 10:
                return {
                    'id': last_part
                }
        
        return None

    def download_artifact(self, artifact_url: str, download_path: str) -> bool:
        """Download an artifact from GitHub"""
        try:
            response = requests.get(artifact_url, headers=self.github_headers, stream=True)
            if response.status_code == 200:
                with open(download_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                print(f"⚠️  Failed to download artifact: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error downloading artifact: {e}")
            return False

    def extract_build_metrics(self, zip_path: str) -> Optional[Dict[str, Any]]:
        """Extract and parse build metrics from artifact zip file"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                print(f"      📂 Files in artifact: {file_list if file_list else 'EMPTY'}")
                
                # Look for build metrics JSON file - various naming patterns
                for file_name in file_list:
                    # Match: build_metrics.json, metrics-*.json, *-metrics.json, or build-*.json
                    if (file_name.endswith('build_metrics.json') or 
                        file_name == 'build_metrics.json' or
                        (file_name.startswith('metrics-') and file_name.endswith('.json')) or
                        file_name.endswith('-metrics.json') or
                        (file_name.startswith('build-') and file_name.endswith('.json'))):
                        print(f"      📄 Found metrics file: {file_name}")
                        with zip_ref.open(file_name) as f:
                            return json.load(f)
            return None
        except Exception as e:
            print(f"⚠️  Error extracting build metrics: {e}")
            return None

    def post_to_db(self, index_url: str, data: Dict[str, Any]) -> bool:
        """Post data to OpenSearch"""
        try:
            response = requests.post(
                index_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(data),
                timeout=10
            )
            if response.status_code in [200, 201]:
                return True
            else:
                print(f"⚠️  OpenSearch returned status {response.status_code}: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error posting to OpenSearch: {e}")
            return False

    def add_common_context_fields(
        self, 
        db_data: Dict[str, Any], 
        job_data: Dict[str, Any],
        workflow_data: Dict[str, Any]
    ) -> None:
        """Add common context fields used across all metric types"""
        db_data[FIELD_REPO] = self.repo
        db_data[FIELD_JOB_ID] = str(job_data["id"])
        db_data[FIELD_JOB_NAME] = str(job_data["name"])
        db_data[FIELD_RUN_ID] = str(job_data.get("run_id", workflow_data.get("id", "unknown")))
        db_data[FIELD_WORKFLOW_ID] = str(workflow_data.get("id", "unknown"))
        
        # Extract PR ID from workflow data if available
        pr_id = "N/A"  # Default to "N/A" for non-PR workflows
        pull_requests = workflow_data.get("pull_requests", [])
        if pull_requests and len(pull_requests) > 0:
            pr_number = pull_requests[0].get("number")
            if pr_number:
                pr_id = str(pr_number)
        
        # Add other workflow context
        db_data[FIELD_USER_ALIAS] = workflow_data.get("actor", {}).get("login", "unknown")
        db_data[FIELD_WORKFLOW_NAME] = workflow_data.get("name", "unknown")
        db_data[FIELD_GITHUB_EVENT] = workflow_data.get("event", "unknown")
        db_data[FIELD_BRANCH] = workflow_data.get("head_branch", "unknown")
        db_data[FIELD_COMMIT_SHA] = workflow_data.get("head_sha", "unknown")
        db_data[FIELD_PR_ID] = pr_id

    def upload_container_metrics(
        self, 
        job_data: Dict[str, Any], 
        build_metrics: Dict[str, Any],
        workflow_data: Dict[str, Any]
    ) -> bool:
        """Upload container-level metrics to OpenSearch"""
        container_index = os.getenv("CONTAINER_INDEX")
        if not container_index:
            print("⚠️  CONTAINER_INDEX not configured, skipping container metrics upload")
            print("      (Set CONTAINER_INDEX env var or pass --upload-detailed-metrics to shell script)")
            return False

        container_data = build_metrics.get("container", {})
        if not container_data:
            print("⚠️  No container data found in build metrics")
            print(f"      build_metrics keys: {list(build_metrics.keys())}")
            return False

        print(f"📦 Uploading container metrics to {container_index}")

        # Create container metrics payload
        payload = {}

        # Identity & Context
        job_id = str(job_data["id"])
        framework = container_data.get("framework", "unknown")
        
        # Set document ID for OpenSearch
        payload["_id"] = f"github-container-{job_id}-{framework}"
        
        # Add common context fields (includes job_id, workflow_id, etc.)
        self.add_common_context_fields(payload, job_data, workflow_data)
        
        # Container-specific fields
        payload[FIELD_STATUS] = str(job_data.get("conclusion") or job_data.get("status", "unknown"))
        payload[FIELD_BUILD_FRAMEWORK] = container_data.get("framework", "unknown")
        payload[FIELD_BUILD_TARGET] = container_data.get("target", "unknown")
        payload[FIELD_BUILD_PLATFORM] = container_data.get("platform", "unknown")
        payload[FIELD_BUILD_SIZE_BYTES] = container_data.get("image_size_bytes", 0)
        payload[FIELD_TOTAL_SIZE_TRANSFERRED] = container_data.get("total_size_transferred_bytes", 0)
        payload[FIELD_TOTAL_STEPS] = container_data.get("total_steps", 0)
        payload[FIELD_CACHED_STEPS] = container_data.get("cached_steps", 0)
        payload[FIELD_BUILT_STEPS] = container_data.get("built_steps", 0)
        payload[FIELD_CACHE_HIT_RATE] = container_data.get("overall_cache_hit_rate", 0.0)
        payload[FIELD_BUILD_DURATION_SEC] = container_data.get("build_duration_sec", 0)
        
        # Timing
        if "build_start_time" in container_data:
            payload[FIELD_BUILD_START_TIME] = container_data["build_start_time"]
        if "build_end_time" in container_data:
            payload[FIELD_BUILD_END_TIME] = container_data["build_end_time"]
            payload["@timestamp"] = container_data["build_end_time"]
        else:
            payload["@timestamp"] = datetime.now(timezone.utc).isoformat()

        # Upload to container index
        if self.post_to_db(container_index, payload):
            print(f"✅ Container metrics uploaded for {framework} framework")
            return True
        else:
            print(f"❌ Failed to upload container metrics")
            return False

    def upload_stage_metrics(
        self, 
        job_data: Dict[str, Any], 
        build_metrics: Dict[str, Any],
        workflow_data: Dict[str, Any]
    ) -> bool:
        """Upload stage-level metrics to OpenSearch"""
        stage_index = os.getenv("STAGE_INDEX")
        if not stage_index:
            print("⚠️  STAGE_INDEX not configured, skipping stage metrics upload")
            return False

        stages = build_metrics.get("stages", [])
        if not stages:
            print("⚠️  No stage data found in build metrics")
            return False

        print(f"📊 Uploading {len(stages)} stage metrics to {stage_index}")

        container_data = build_metrics.get("container", {})
        job_id = str(job_data["id"])
        framework = container_data.get("framework", "unknown")
        
        success_count = 0
        for stage in stages:
            stage_name = stage.get("stage_name", "unknown")
            
            # Create stage metrics payload
            payload = {}
            
            # Set document ID for OpenSearch
            payload["_id"] = f"github-stage-{job_id}-{framework}-{stage_name}"
            
            # Add common context fields
            self.add_common_context_fields(payload, job_data, workflow_data)
            
            # Stage-specific fields
            payload[FIELD_BUILD_FRAMEWORK] = framework
            payload[FIELD_STAGE_NAME] = stage_name
            payload[FIELD_STAGE_TOTAL_STEPS] = stage.get("total_steps", 0)
            payload[FIELD_STAGE_CACHED_STEPS] = stage.get("cached_steps", 0)
            payload[FIELD_STAGE_BUILT_STEPS] = stage.get("built_steps", 0)
            payload[FIELD_STAGE_DURATION_SEC] = stage.get("build_duration_sec", 0.0)
            payload[FIELD_STAGE_CACHE_HIT_RATE] = stage.get("cache_hit_rate", 0.0)
            
            # Use container timestamp
            if "build_end_time" in container_data:
                payload["@timestamp"] = container_data["build_end_time"]
            else:
                payload["@timestamp"] = datetime.now(timezone.utc).isoformat()
            
            # Upload to stage index
            if self.post_to_db(stage_index, payload):
                success_count += 1
            else:
                print(f"❌ Failed to upload stage metrics for {stage_name}")

        print(f"✅ Uploaded {success_count}/{len(stages)} stage metrics")
        return success_count == len(stages)

    def upload_layer_metrics(
        self, 
        job_data: Dict[str, Any], 
        build_metrics: Dict[str, Any],
        workflow_data: Dict[str, Any]
    ) -> bool:
        """Upload layer-level metrics to OpenSearch"""
        layer_index = os.getenv("LAYER_INDEX")
        if not layer_index:
            print("⚠️  LAYER_INDEX not configured, skipping layer metrics upload")
            return False

        layers = build_metrics.get("layers", [])
        if not layers:
            print("⚠️  No layer data found in build metrics")
            return False

        print(f"🔧 Uploading {len(layers)} layer metrics to {layer_index}")

        container_data = build_metrics.get("container", {})
        job_id = str(job_data["id"])
        framework = container_data.get("framework", "unknown")
        
        success_count = 0
        for layer in layers:
            step_number = layer.get("step_number", 0)
            stage_name = layer.get("stage", "unknown")
            
            # Create layer metrics payload
            payload = {}
            
            # Set document ID for OpenSearch
            payload["_id"] = f"github-layer-{job_id}-{framework}-{stage_name}-step{step_number}"
            
            # Add common context fields
            self.add_common_context_fields(payload, job_data, workflow_data)
            
            # Layer-specific fields
            payload[FIELD_BUILD_FRAMEWORK] = framework
            payload[FIELD_LAYER_STAGE] = stage_name
            payload[FIELD_LAYER_STEP_NUMBER] = step_number
            payload[FIELD_LAYER_STEP_NAME] = layer.get("step_name", "unknown")
            payload[FIELD_LAYER_COMMAND] = layer.get("command", "")
            payload[FIELD_LAYER_STATUS] = layer.get("status", "unknown")
            payload[FIELD_LAYER_CACHED] = layer.get("cached", False)
            payload[FIELD_LAYER_DURATION_SEC] = layer.get("duration_sec", 0.0)
            payload[FIELD_LAYER_SIZE_TRANSFERRED] = layer.get("size_transferred", 0)
            
            # Use container timestamp
            if "build_end_time" in container_data:
                payload["@timestamp"] = container_data["build_end_time"]
            else:
                payload["@timestamp"] = datetime.now(timezone.utc).isoformat()
            
            # Upload to layer index
            if self.post_to_db(layer_index, payload):
                success_count += 1

        print(f"✅ Uploaded {success_count}/{len(layers)} layer metrics")
        return success_count == len(layers)

    def process_artifact(
        self,
        artifact: Dict[str, Any],
        extracted_ids: Dict[str, str],
        workflow_data: Dict[str, Any]
    ) -> None:
        """Process a single artifact and upload all metrics"""
        artifact_name = artifact.get("name", "Unknown")
        artifact_id = artifact.get("id", "?")
        expired = artifact.get("expired", False)
        
        if expired:
            print(f"      ⚠️  Artifact expired, skipping")
            return
        
        if 'job_id' not in extracted_ids or 'workflow_id' not in extracted_ids:
            print(f"      ⚠️  Missing job_id or workflow_id, skipping")
            return
        
        job_id = extracted_ids['job_id']
        workflow_id = extracted_ids['workflow_id']
        
        # Fetch job details
        job_info = self.fetch_job_details(job_id)
        if not job_info:
            print(f"      ⚠️  Could not fetch job info for Job ID {job_id}")
            return
        
        print(f"      📋 Job Info:")
        print(f"         Job ID: {job_id}")
        print(f"         Name: {job_info.get('name', 'Unknown')}")
        print(f"         Status: {job_info.get('status', 'unknown')}, Conclusion: {job_info.get('conclusion', 'N/A')}")
        print(f"         Runner: {job_info.get('runner_name', 'N/A')}")
        
        # Download artifact
        if not self.upload_to_opensearch:
            print(f"      ℹ️  Upload to OpenSearch disabled (pass --upload-detailed-metrics to enable)")
            return
        
        print(f"      📥 Downloading artifact for detailed metrics upload...")
        
        artifact_download_url = artifact.get("archive_download_url")
        if not artifact_download_url:
            print(f"      ⚠️  No download URL available for artifact")
            return
        
        # Create temp directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, f"artifact_{artifact_id}.zip")
            
            # Download artifact
            if not self.download_artifact(artifact_download_url, zip_path):
                print(f"      ❌ Failed to download artifact")
                return
            
            print(f"      ✅ Artifact downloaded")
            
            # Extract build metrics
            build_metrics = self.extract_build_metrics(zip_path)
            if not build_metrics:
                print(f"      ⚠️  No build metrics found in artifact")
                return
            
            print(f"      📊 Build metrics extracted")
            
            # Check for nested structure
            if "container" not in build_metrics:
                print(f"      ⚠️  Metrics do not have expected nested structure (missing 'container' field)")
                return
            
            # Success! Found the new nested format
            has_stages = "stages" in build_metrics
            has_layers = "layers" in build_metrics
            print(f"      ✅ SUCCESS: Artifact matches NEW NESTED FORMAT!")
            print(f"         - Contains 'container' data: ✓")
            print(f"         - Contains 'stages' array: {'✓' if has_stages else '✗'} ({len(build_metrics.get('stages', []))} stages)")
            print(f"         - Contains 'layers' array: {'✓' if has_layers else '✗'} ({len(build_metrics.get('layers', []))} layers)")
            
            # Upload all metrics
            print(f"      📤 Uploading detailed metrics...")
            
            # Upload container metrics
            self.upload_container_metrics(job_info, build_metrics, workflow_data)
            
            # Upload stage metrics
            if "stages" in build_metrics:
                self.upload_stage_metrics(job_info, build_metrics, workflow_data)
            
            # Upload layer metrics
            if "layers" in build_metrics:
                self.upload_layer_metrics(job_info, build_metrics, workflow_data)

    def list_all_artifacts(self, hours_back: float) -> None:
        """List all artifacts from workflow runs in the specified time range"""
        print(f"\n🚀 Listing artifacts from the past {hours_back} hours\n")

        # Fetch workflow runs
        workflow_runs = self.fetch_workflow_runs_timerange(hours_back)

        if not workflow_runs:
            print("❌ No workflow runs found in the specified time range")
            return

        # Track statistics
        total_artifacts = 0
        total_size_bytes = 0
        artifacts_by_name = {}
        artifacts_by_workflow = {}

        # Process each workflow run
        for workflow in workflow_runs:
            run_id = str(workflow["id"])
            workflow_name = workflow.get("name", "Unknown")
            run_number = workflow.get("run_number", "?")
            
            # Fetch artifacts for this run
            artifacts = self.fetch_artifacts_for_run(run_id)

            if artifacts:
                # First pass: filter for relevant artifacts only
                relevant_artifacts = []
                for artifact in artifacts:
                    artifact_name = artifact.get("name", "Unknown")
                    extracted_ids = self.extract_id_from_artifact_name(artifact_name)
                    if extracted_ids:
                        relevant_artifacts.append((artifact, extracted_ids))
                
                # Only print workflow header if we have relevant artifacts
                if relevant_artifacts:
                    print(f"\n📦 Workflow: {workflow_name} (Run #{run_number}, ID: {run_id})")
                
                for artifact, extracted_ids in relevant_artifacts:
                    artifact_name = artifact.get("name", "Unknown")
                    artifact_size = artifact.get("size_in_bytes", 0)
                    artifact_id = artifact.get("id", "?")
                    created_at = artifact.get("created_at", "")
                    expired = artifact.get("expired", False)
                    
                    # Print artifact details
                    status = "❌ EXPIRED" if expired else "✅"
                    size_mb = artifact_size / (1024 * 1024)
                    print(f"   {status} {artifact_name}")
                    print(f"      ID: {artifact_id}, Size: {size_mb:.2f} MB, Created: {created_at}")
                    
                    # Process the artifact
                    if 'job_id' in extracted_ids and 'workflow_id' in extracted_ids:
                        workflow_id = extracted_ids['workflow_id']
                        job_id = extracted_ids['job_id']
                        print(f"      🔍 Detected IDs - Workflow: {workflow_id}, Job: {job_id}")
                        
                        # Process artifact and upload metrics
                        self.process_artifact(artifact, extracted_ids, workflow)
                    
                    # Update statistics (only for relevant artifacts)
                    if not expired:
                        total_artifacts += 1
                        total_size_bytes += artifact_size
                        
                        # Track by name
                        if artifact_name not in artifacts_by_name:
                            artifacts_by_name[artifact_name] = 0
                        artifacts_by_name[artifact_name] += 1
                        
                        # Track by workflow
                        if workflow_name not in artifacts_by_workflow:
                            artifacts_by_workflow[workflow_name] = 0
                        artifacts_by_workflow[workflow_name] += 1

            sleep(0.3)  # Rate limiting

        # Print summary
        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)
        print(f"Total workflow runs: {len(workflow_runs)}")
        print(f"Total artifacts (not expired): {total_artifacts}")
        print(f"Total size: {total_size_bytes / (1024 * 1024 * 1024):.2f} GB")

        if artifacts_by_name:
            print("\n📋 Artifacts by name:")
            for name, count in sorted(artifacts_by_name.items(), key=lambda x: x[1], reverse=True)[:20]:
                print(f"   {name}: {count} occurrences")

        if artifacts_by_workflow:
            print("\n📋 Artifacts by workflow:")
            for workflow, count in sorted(artifacts_by_workflow.items(), key=lambda x: x[1], reverse=True):
                print(f"   {workflow}: {count} artifacts")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="List GitHub Actions artifacts and upload detailed build metrics"
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=1.0,
        help="Number of hours to look back (default: 1.0)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=os.getenv("GITHUB_REPOSITORY", ""),
        help="Repository in format owner/repo (default: from GITHUB_REPOSITORY env)",
    )
    parser.add_argument(
        "--upload-detailed-metrics",
        action="store_true",
        default=True,
        help="Download artifacts and upload detailed metrics to OpenSearch (default: enabled)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Disable uploading metrics to OpenSearch",
    )

    args = parser.parse_args()

    # Get GitHub token
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        token_file = os.path.expanduser("~/.github-token")
        if os.path.exists(token_file):
            with open(token_file, "r") as f:
                github_token = f.read().strip()

    if not github_token:
        print("❌ GitHub token not found. Set GITHUB_TOKEN or create ~/.github-token")
        sys.exit(1)

    if not args.repo:
        print("❌ Repository not specified. Use --repo or set GITHUB_REPOSITORY")
        sys.exit(1)

    # Determine upload setting (--no-upload overrides --upload-detailed-metrics)
    upload_enabled = args.upload_detailed_metrics and not args.no_upload
    
    # Create lister and run
    print(f"\n🔧 Configuration:")
    print(f"   Upload to OpenSearch: {upload_enabled}")
    print(f"   CONTAINER_INDEX: {os.getenv('CONTAINER_INDEX', 'NOT SET')}")
    print(f"   STAGE_INDEX: {os.getenv('STAGE_INDEX', 'NOT SET')}")
    print(f"   LAYER_INDEX: {os.getenv('LAYER_INDEX', 'NOT SET')}")
    
    lister = DetailedArtifactLister(args.repo, github_token, upload_enabled)
    lister.list_all_artifacts(args.hours)


if __name__ == "__main__":
    main()


