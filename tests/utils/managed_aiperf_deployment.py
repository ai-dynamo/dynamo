# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import kr8s
from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import exceptions

# LogStreamManager removed - using PVC-based log collection only


@dataclass
class LoadConfig:
    """Configuration for AI Performance load testing"""

    concurrency: int = 4
    input_tokens_mean: int = 512
    input_tokens_stddev: int = 0
    output_tokens_mean: int = 64
    output_tokens_stddev: int = 0
    request_count: int = 100
    warmup_requests: Optional[int] = None
    request_rate: Optional[float] = None  # requests per second
    duration_minutes: Optional[int] = None  # Alternative to request_count

    # Model and endpoint configuration
    model: str = "Qwen/Qwen3-0.6B"
    endpoint_url: str = "http://localhost:8000"
    endpoint_path: str = "/v1/chat/completions"

    # Additional configuration
    streaming: bool = True
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    ignore_eos: bool = True
    extra_inputs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.warmup_requests is None:
            self.warmup_requests = min(self.concurrency, 10)


def _get_workspace_dir() -> str:
    """Get workspace directory without depending on dynamo.common package."""
    # Start from this file's location and walk up to find workspace root
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):  # Stop at filesystem root
        # Workspace root has pyproject.toml
        if os.path.exists(os.path.join(current, "pyproject.toml")):
            return current
        current = os.path.dirname(current)

    # Fallback: assume workspace is 3 levels up from tests/utils/
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ManagedAIPerfDeployment:
    """Managed AI Perf deployment that runs load tests in Kubernetes"""

    log_dir: str
    load_config: LoadConfig
    namespace: str
    job_name: str = "ai-perf-load-test"
    container_log_dir: str = "/tmp/aiperf"  # Directory inside container

    # Internal state
    _core_api: Optional[client.CoreV1Api] = None
    _batch_api: Optional[client.BatchV1Api] = None
    _logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    _job_created: bool = False
    _terminated: bool = False

    def __post_init__(self):
        # Ensure unique job name to avoid conflicts
        self.job_name = f"{self.job_name}-{secrets.token_hex(4)}"

        # Set up local output directory (following ManagedDeployment pattern)
        self.local_output_dir = os.path.join(self.log_dir, "aiperf")
        os.makedirs(self.local_output_dir, exist_ok=True)
        self._logger.info(f"AI Perf results will be saved to: {self.local_output_dir}")

    async def _init_kubernetes(self):
        """Initialize kubernetes client"""
        try:
            # Try in-cluster config first (for pods with service accounts)
            config.load_incluster_config()
        except Exception:
            # Fallback to kube config file (for local development)
            await config.load_kube_config()

        k8s_client = client.ApiClient()
        self._core_api = client.CoreV1Api(k8s_client)
        self._batch_api = client.BatchV1Api(k8s_client)

    def _generate_job_spec(self) -> dict:
        """Generate Kubernetes Job specification for AI Perf load test"""

        # Determine if this is a duration-based test (continuous load)
        use_duration = self.load_config.duration_minutes is not None

        # Calculate timeout based on test type (following client.py pattern)
        if use_duration:
            # Duration-based test: duration + buffer time
            _timeout_seconds = (
                self.load_config.duration_minutes * 60
            ) + 60  # 1 minute buffer  # noqa: F841
        else:
            # Request count-based test: adaptive timeout based on request count
            _timeout_seconds = max(
                self.load_config.request_count * 2 + 60, 300
            )  # At least 5 minutes  # noqa: F841

        # Build aiperf command (following client.py pattern)
        aiperf_args = [
            "profile",
            "--artifact-dir",
            self.container_log_dir,
            "--model",
            self.load_config.model,
            "--tokenizer",
            self.load_config.model,
            "--endpoint-type",
            "chat",
            "--endpoint",
            self.load_config.endpoint_path,
            "--url",
            self.load_config.endpoint_url,
            "--synthetic-input-tokens-mean",
            str(self.load_config.input_tokens_mean),
            "--synthetic-input-tokens-stddev",
            str(self.load_config.input_tokens_stddev),
            "--output-tokens-mean",
            str(self.load_config.output_tokens_mean),
            "--output-tokens-stddev",
            str(self.load_config.output_tokens_stddev),
            "--concurrency",
            str(self.load_config.concurrency),
            "--warmup-request-count",
            str(self.load_config.warmup_requests),
            "--num-dataset-entries",
            "12800",
            "--random-seed",
            "100",
            "--workers-max",
            "252",
            "--record-processors",
            "32",
            "--ui",
            "simple",
            "--verbose",
        ]

        # Add duration or request count based on configuration (following client.py pattern)
        if use_duration:
            # Use benchmark duration for continuous load
            duration_seconds = self.load_config.duration_minutes * 60
            aiperf_args.extend(["--benchmark-duration", str(duration_seconds)])
        else:
            # Use request count for fixed load
            aiperf_args.extend(["--request-count", str(self.load_config.request_count)])

        # Add optional parameters
        if self.load_config.streaming:
            aiperf_args.append("--streaming")

        if self.load_config.request_rate:
            aiperf_args.extend(["--request-rate", str(self.load_config.request_rate)])

        # Add extra inputs
        extra_inputs = {
            "max_tokens": self.load_config.output_tokens_mean,
            "min_tokens": self.load_config.output_tokens_mean,
            "temperature": self.load_config.temperature,
            "repetition_penalty": self.load_config.repetition_penalty,
            **self.load_config.extra_inputs,
        }

        if self.load_config.ignore_eos:
            extra_inputs["ignore_eos"] = True

        for key, value in extra_inputs.items():
            aiperf_args.extend(["--extra-inputs", f"{key}:{value}"])

        # Build the complete shell script that keeps container alive after completion
        script = f"""#!/bin/bash
set -e

# Setup environment
apt-get update && apt-get install -y curl jq procps git && apt-get clean
pip install aiperf
echo "aiperf installation completed"

# Configure networking
sysctl -w net.ipv4.ip_local_port_range="1024 65000"
export COLUMNS=200
export PYTHONUNBUFFERED=1

# Create log directory
mkdir -p {self.container_log_dir}

# Create completion marker directory
mkdir -p {self.container_log_dir}/status

# Wait for model to be ready
echo "Waiting for model '{self.load_config.model}' at {self.load_config.endpoint_url}/v1/models"
while ! curl -s "{self.load_config.endpoint_url}/v1/models" | jq -e --arg model "{self.load_config.model}" '.data[]? | select(.id == $$model)' ; do
    echo "[$(date '+%H:%M:%S')] Model not ready yet, sleeping 5s..."
    sleep 5
done
echo "✅ Model '{self.load_config.model}' is now available!"

# Run AI Perf
echo "Starting AI Perf load test..."
if [ "{use_duration}" = "True" ]; then
    echo "Configuration: duration-based test, concurrency={self.load_config.concurrency}, input_tokens={self.load_config.input_tokens_mean}, output_tokens={self.load_config.output_tokens_mean}, duration={self.load_config.duration_minutes}min"
else
    echo "Configuration: request-count test, concurrency={self.load_config.concurrency}, input_tokens={self.load_config.input_tokens_mean}, output_tokens={self.load_config.output_tokens_mean}, requests={self.load_config.request_count}"
fi

# Mark test as started
echo "$(date)" > {self.container_log_dir}/status/started

# Define signal handlers to keep container alive when aiperf is terminated
cleanup_handler() {{
    echo "Container script received signal, but keeping container alive for artifact extraction..."
    if [ -n "$AIPERF_PID" ]; then
        echo "Waiting for aiperf process $AIPERF_PID to finish..."
        wait $AIPERF_PID
        AIPERF_EXIT_CODE=$?
        echo "aiperf process finished with exit code: $AIPERF_EXIT_CODE"
        echo "$AIPERF_EXIT_CODE" > {self.container_log_dir}/status/exit_code
    fi
}}

# Set up signal handlers (but don't exit - let container stay alive)
trap cleanup_handler SIGTERM SIGINT

# Run aiperf in background and capture its PID
echo "Starting aiperf in background..."
set +e
aiperf {' '.join(aiperf_args)} &
AIPERF_PID=$!
echo "aiperf started with PID: $AIPERF_PID"

# Wait for aiperf to complete (or be terminated)
wait $AIPERF_PID
AIPERF_EXIT_CODE=$?
set -e

# Mark test as completed with exit code
echo "$(date)" > {self.container_log_dir}/status/completed
echo "$AIPERF_EXIT_CODE" > {self.container_log_dir}/status/exit_code

echo "AI Perf load test completed with exit code: $AIPERF_EXIT_CODE"
echo "Results:"
ls -la {self.container_log_dir}

# Extract and display summary if available
if [ -f "{self.container_log_dir}/profile_export_aiperf.json" ]; then
    echo "=== Performance Summary ==="
    jq . "{self.container_log_dir}/profile_export_aiperf.json"

    # Check for errors
    if jq -e '.error_stats' "{self.container_log_dir}/profile_export_aiperf.json" >/dev/null 2>&1; then
        echo "🚨 ERRORS DETECTED:"
        jq '.error_stats' "{self.container_log_dir}/profile_export_aiperf.json"

        # Specifically check for 500 errors (DEP-709 issue)
        if jq -e '.error_stats | has("500")' "{self.container_log_dir}/profile_export_aiperf.json" >/dev/null 2>&1; then
            echo "🚨 500 ERRORS DETECTED - This indicates DEP-709 issue!"
            jq '.error_stats["500"]' "{self.container_log_dir}/profile_export_aiperf.json"
        fi
    else
        echo "✅ No errors detected"
    fi
fi

# Prepare compressed archive for efficient extraction
echo "=== Creating compressed archive for efficient extraction ==="
ARCHIVE_PATH="{self.container_log_dir}/aiperf_artifacts.tar.gz"
echo "Creating archive: $ARCHIVE_PATH"

# Brief delay to ensure all file writes have completed
sleep 2

# Create compressed tar archive of all artifacts
cd {self.container_log_dir}
echo "Current directory contents:"
ls -la
# Create a list of files to archive (excluding the archive itself and ensuring stable file state)
# Include only stable output files, exclude logs and status files that might still be changing
find . -type f \\( -name "*.json" -o -name "*.jsonl" -o -name "*.csv" \\) | grep -v "aiperf_artifacts.tar.gz" | grep -v "/status/" | sort > /tmp/file_list.txt
echo "Files to archive:"
cat /tmp/file_list.txt
echo "File count: $(wc -l < /tmp/file_list.txt)"

# Create the archive with error handling for "file changed" warnings
echo "Creating tar archive..."
if tar -czf aiperf_artifacts.tar.gz -T /tmp/file_list.txt 2>/tmp/tar_errors.log; then
    ARCHIVE_SIZE=$(ls -la aiperf_artifacts.tar.gz | awk '{{print $5}}')
    echo "Archive created successfully, size: $ARCHIVE_SIZE bytes"

    # Check for warnings in tar output
    if [ -s /tmp/tar_errors.log ]; then
        echo "Tar warnings/errors detected:"
        cat /tmp/tar_errors.log
    fi
else
    echo "Archive creation failed:"
    cat /tmp/tar_errors.log
    echo "$(date)" > status/archive_failed
    exit 1
fi

# Verify archive integrity
if tar -tzf aiperf_artifacts.tar.gz >/dev/null 2>&1; then
    echo "✅ Archive integrity verified"
    echo "$(date)" > status/archive_ready
else
    echo "🚨 Archive integrity check failed"
    echo "$(date)" > status/archive_failed
fi

# Keep container alive indefinitely for result extraction
echo "=== AI Perf test completed, keeping container alive for result extraction ==="
echo "Container will stay alive until explicitly terminated by test cleanup..."

# Check if aiperf was terminated externally (by pkill)
if [ "$AIPERF_EXIT_CODE" -eq 143 ] || [ "$AIPERF_EXIT_CODE" -eq 130 ] || [ -f "{self.container_log_dir}/status/terminated" ]; then
    echo "aiperf was terminated externally (exit code: $AIPERF_EXIT_CODE)"
    echo "$(date)" > {self.container_log_dir}/status/terminated_by_signal
else
    echo "aiperf completed normally (exit code: $AIPERF_EXIT_CODE)"
fi

# Create a marker that extraction can check
touch {self.container_log_dir}/status/ready_for_extraction

# Keep container alive indefinitely - test will clean up explicitly
echo "=== AI Perf test completed, keeping container alive for result extraction ==="
echo "Container will stay alive until explicitly terminated by test cleanup..."
echo "Termination flag: $([ -f "{self.container_log_dir}/status/terminated" ] && echo "YES" || echo "NO")"
echo "Ready for extraction: YES"

while true; do
    echo "[$(date '+%H:%M:%S')] Container alive, waiting for test cleanup..."
    sleep 60
done
"""

        job_spec = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": self.job_name,
                "namespace": self.namespace,
                "labels": {
                    "app": "ai-perf-load-test",
                    "managed-by": "managed-aiperf-deployment",
                },
            },
            "spec": {
                "backoffLimit": 1,
                "completions": 1,
                "parallelism": 1,
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "ai-perf-load-test",
                            "job-name": self.job_name,
                        }
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "aiperf",
                                "image": "python:3.12-slim",
                                "imagePullPolicy": "IfNotPresent",
                                "command": ["/bin/bash", "-c", script],
                                "env": [
                                    {"name": "PYTHONUNBUFFERED", "value": "1"},
                                    {
                                        "name": "AIPERF_HTTP_CONNECTION_LIMIT",
                                        "value": "200",
                                    },
                                ],
                                "securityContext": {"privileged": True},
                                "volumeMounts": [
                                    {
                                        "name": "perf-results",
                                        "mountPath": self.container_log_dir,
                                    }
                                ],
                                "workingDir": "/workspace",
                            }
                        ],
                        "volumes": [{"name": "perf-results", "emptyDir": {}}],
                    },
                },
            },
        }

        return job_spec

    async def _create_job(self):
        """Create the AI Perf job in Kubernetes"""
        job_spec = self._generate_job_spec()

        self._logger.info(f"Creating AI Perf job: {self.job_name}")

        try:
            assert self._batch_api is not None, "Kubernetes API not initialized"
            await self._batch_api.create_namespaced_job(
                namespace=self.namespace, body=job_spec
            )
            self._job_created = True
            self._logger.info(f"AI Perf job created: {self.job_name}")
        except exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                self._logger.warning(f"Job {self.job_name} already exists")
                self._job_created = True
            else:
                self._logger.error(f"Failed to create job {self.job_name}: {e}")
                raise

    async def _wait_for_status_marker(
        self, marker_file: str, marker_description: str, timeout: Optional[int] = None
    ) -> bool:
        """Wait for a specific status marker file to appear in the pod

        Args:
            marker_file: Path to the marker file to wait for
            marker_description: Human-readable description for logging
            timeout: Maximum time to wait in seconds

        Returns:
            True if marker found, False if job failed or terminated

        Raises:
            TimeoutError: If timeout is reached without finding marker
        """
        start_time = time.time()

        self._logger.info(
            f"Waiting for {marker_description} in job {self.job_name} (timeout: {timeout}s)"
        )

        while (time.time() - start_time) < timeout:
            try:
                # Check if termination was requested
                if self._terminated:
                    self._logger.info(
                        f"{marker_description} wait terminated by request"
                    )
                    return False

                # Find the pod for this job
                pods = []
                pod_generator = kr8s.get(
                    "pods",
                    namespace=self.namespace,
                    label_selector=f"job-name={self.job_name}",
                )
                for pod in pod_generator:
                    pods.append(pod)

                if pods:
                    pod = pods[0]

                    # Check if the status marker exists
                    try:
                        result = await asyncio.wait_for(
                            asyncio.create_task(
                                asyncio.to_thread(pod.exec, ["test", "-f", marker_file])
                            ),
                            timeout=10.0,
                        )

                        if result.returncode == 0:
                            # Marker found
                            self._logger.info(
                                f"{marker_description} marker found in job {self.job_name}"
                            )
                            return True

                    except (asyncio.TimeoutError, Exception) as marker_e:
                        # Marker not ready yet, continue waiting
                        # Only log detailed errors for non-timeout exceptions
                        if not isinstance(marker_e, asyncio.TimeoutError):
                            self._logger.debug(
                                f"Marker check failed (will retry): {type(marker_e).__name__}: {marker_e}"
                            )
                        pass

                    # Also check for job failure
                    assert self._batch_api is not None, "Kubernetes API not initialized"
                    job = await self._batch_api.read_namespaced_job(
                        name=self.job_name, namespace=self.namespace
                    )

                    if job.status.failed:
                        self._logger.error(
                            f"AI Perf job {self.job_name} failed while waiting for {marker_description}"
                        )
                        return False

                # Log progress periodically
                elapsed = time.time() - start_time
                if int(elapsed) % 60 == 0:  # Every minute
                    self._logger.info(
                        f"Still waiting for {marker_description}... ({elapsed:.1f}s elapsed)"
                    )

            except exceptions.ApiException as e:
                self._logger.warning(f"Error checking {marker_description} status: {e}")

            await asyncio.sleep(5)

        if self._terminated:
            self._logger.info(f"{marker_description} wait was terminated")
            return False
        else:
            raise TimeoutError(f"{marker_description} did not appear within {timeout}s")

    async def _wait_for_started(self, timeout: int = 300) -> bool:
        """Wait for the AI Perf test to start (5 minute timeout by default)"""
        marker_file = f"{self.container_log_dir}/status/started"
        return await self._wait_for_status_marker(
            marker_file, "AI Perf test start", timeout
        )

    async def _wait_for_completion(self, timeout: Optional[int] = None) -> bool:
        """Wait for the AI Perf test to complete (not the pod/job)"""

        # Calculate timeout based on test configuration if not provided
        if timeout is None:
            if self.load_config.duration_minutes:
                # Duration-based test: duration + buffer time (following client.py pattern)
                timeout = (
                    self.load_config.duration_minutes * 60
                ) + 60  # 1 minute buffer
            else:
                # Request count-based test: adaptive timeout
                timeout = max(
                    self.load_config.request_count * 2 + 60, 300
                )  # At least 5 minutes

        marker_file = f"{self.container_log_dir}/status/ready_for_extraction"
        return await self._wait_for_status_marker(
            marker_file, "AI Perf test completion", timeout
        )

    async def get_logs(self) -> str:
        """Get logs from the AI Perf job pod and save locally"""
        try:
            # Find the pod for this job
            assert self._core_api is not None, "Kubernetes API not initialized"
            pods = await self._core_api.list_namespaced_pod(
                namespace=self.namespace, label_selector=f"job-name={self.job_name}"
            )

            if not pods.items:
                self._logger.warning(f"No pods found for job {self.job_name}")
                return ""

            pod = pods.items[0]
            logs = await self._core_api.read_namespaced_pod_log(
                name=pod.metadata.name, namespace=self.namespace
            )

            # Save logs locally if output directory is configured
            if self.local_output_dir and logs:
                log_file = os.path.join(self.local_output_dir, "aiperf_job.log")
                with open(log_file, "w") as f:
                    f.write(logs)
                self._logger.info(f"Saved job logs to {log_file}")

            return logs

        except Exception as e:
            self._logger.error(f"Failed to get logs for job {self.job_name}: {e}")
            return ""

    async def get_results(self) -> Dict[str, Any]:
        """Extract performance results from the live pod"""
        try:
            # Extract results from live pod
            results = await self._extract_results_from_pod()

            # If pod extraction didn't work, try parsing logs as fallback
            if not results:
                logs = await self.get_logs()
                results = self._parse_results_from_logs(logs)

                if results and self.local_output_dir:
                    # Save parsed results locally
                    summary_file = os.path.join(
                        self.local_output_dir, "profile_summary.json"
                    )
                    with open(summary_file, "w") as f:
                        json.dump(results, f, indent=2)
                    self._logger.info(f"Saved performance summary to {summary_file}")

            return results

        except Exception as e:
            self._logger.error(f"Failed to get results for job {self.job_name}: {e}")
            return {}

    def _parse_results_from_logs(self, logs: str) -> Dict[str, Any]:
        """Parse performance results from the job logs"""
        try:
            # Look for JSON output in the logs - AI Perf should print the summary
            lines = logs.split("\n")

            # Look for the performance summary JSON block
            json_start = -1
            _json_end = -1  # noqa: F841

            for i, line in enumerate(lines):
                if "=== Performance Summary ===" in line:
                    # Look for JSON block after this line
                    for j in range(i + 1, len(lines)):
                        stripped = lines[j].strip()
                        if stripped.startswith("{"):
                            json_start = j
                            break
                    break

            if json_start >= 0:
                # Find the end of the JSON block
                brace_count = 0
                json_lines = []

                for i in range(json_start, len(lines)):
                    line = lines[i].strip()
                    if not line:
                        continue

                    json_lines.append(line)

                    # Count braces to find end of JSON
                    brace_count += line.count("{") - line.count("}")

                    if brace_count == 0 and line.endswith("}"):
                        break

                if json_lines:
                    json_text = " ".join(json_lines)
                    try:
                        results = json.loads(json_text)
                        print(
                            f"DEBUG: Successfully parsed results from logs: {len(results)} keys"
                        )
                        return results
                    except json.JSONDecodeError as je:
                        print(f"DEBUG: Failed to parse JSON from logs: {je}")
                        print(f"DEBUG: JSON text was: {json_text[:200]}...")

            print("DEBUG: No performance summary found in logs")
            return {}

        except Exception as e:
            print(f"DEBUG: Error parsing logs: {e}")
            return {}

    async def _extract_results_from_pod(self) -> Dict[str, Any]:
        """Extract results directly from live pod"""
        try:
            # Get the pod to access the volume
            pods = []
            pod_generator = kr8s.get(
                "pods",
                namespace=self.namespace,
                label_selector=f"job-name={self.job_name}",
            )
            for pod in pod_generator:
                pods.append(pod)

            if not pods:
                raise Exception(f"No pods found for job {self.job_name}")

            pod = pods[0]
            results = {}

            # Extract the summary file - AI Perf generates profile_export_aiperf.json
            try:
                result = await asyncio.wait_for(
                    asyncio.create_task(
                        asyncio.to_thread(
                            pod.exec,
                            [
                                "cat",
                                f"{self.container_log_dir}/profile_export_aiperf.json",
                            ],
                        )
                    ),
                    timeout=30.0,
                )

                if result.returncode == 0:
                    results = json.loads(result.stdout.decode())

                    # Also extract AI Perf exit code
                    try:
                        exit_code_result = await asyncio.wait_for(
                            asyncio.create_task(
                                asyncio.to_thread(
                                    pod.exec,
                                    [
                                        "cat",
                                        f"{self.container_log_dir}/status/exit_code",
                                    ],
                                )
                            ),
                            timeout=10.0,
                        )
                        if exit_code_result.returncode == 0:
                            results["aiperf_exit_code"] = int(
                                exit_code_result.stdout.decode().strip()
                            )
                    except Exception:
                        pass

            except Exception as e:
                self._logger.warning(f"Could not extract results from pod: {e}")

            # Extract all artifacts if we have results and local output directory is configured
            if self.local_output_dir and results:
                await self._extract_all_artifacts(pod)

            return results

        except Exception as e:
            import traceback

            error_details = f"Exception type: {type(e).__name__}, Message: '{str(e)}', Traceback: {traceback.format_exc()}"
            self._logger.warning(f"Failed to extract from pod: {error_details}")
            return {}

    async def _extract_all_artifacts(self, pod):
        """Extract all AI Perf artifacts using compressed archive transfer"""
        try:
            self._logger.info(f"Starting artifact extraction from pod {pod.name}")

            # Wait for archive to be ready
            archive_ready = False
            for attempt in range(30):  # Wait up to 30 seconds for archive
                try:
                    check_result = await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(
                                pod.exec,
                                [
                                    "ls",
                                    "-la",
                                    f"{self.container_log_dir}/status/archive_ready",
                                ],
                            )
                        ),
                        timeout=5.0,
                    )
                    if check_result.returncode == 0:
                        archive_ready = True
                        self._logger.info("Archive ready marker found")
                        break
                except Exception:
                    pass

                self._logger.info(
                    f"Waiting for archive to be ready... (attempt {attempt + 1}/30)"
                )
                await asyncio.sleep(1)

            if not archive_ready:
                self._logger.warning(
                    "Archive ready marker not found, proceeding anyway..."
                )

            # Check if the compressed archive exists and get its info
            archive_path = f"{self.container_log_dir}/aiperf_artifacts.tar.gz"
            try:
                check_result = await asyncio.wait_for(
                    asyncio.create_task(
                        asyncio.to_thread(pod.exec, ["ls", "-la", archive_path])
                    ),
                    timeout=10.0,
                )
                self._logger.info(
                    f"Archive info: {check_result.stdout.decode().strip() if check_result.stdout else 'No stdout'}"
                )
                if check_result.returncode != 0:
                    raise Exception(f"Archive file not found at {archive_path}")
            except Exception as check_e:
                self._logger.warning(
                    f"Could not check archive file: {type(check_e).__name__}: {check_e}"
                )
                raise Exception(f"Archive file not accessible: {check_e}")

            # Use cat to transfer the compressed file (faster and more reliable than tar streaming)
            self._logger.info(f"Transferring compressed archive: {archive_path}")
            cat_result = await asyncio.wait_for(
                asyncio.create_task(asyncio.to_thread(pod.exec, ["cat", archive_path])),
                timeout=60.0,  # Reduced timeout since we're transferring a single compressed file
            )

            self._logger.info(
                f"Archive transfer completed with return code: {cat_result.returncode}"
            )

            if cat_result.returncode != 0:
                stderr_output = (
                    cat_result.stderr.decode()
                    if cat_result.stderr
                    else "No stderr output"
                )
                raise Exception(
                    f"Archive transfer failed with return code {cat_result.returncode}. Stderr: {stderr_output}"
                )

            # Check the size of the archive
            archive_size = len(cat_result.stdout) if cat_result.stdout else 0
            self._logger.info(f"Archive size: {archive_size} bytes")

            if archive_size == 0:
                raise Exception("Archive transfer succeeded but produced no data")

            # Save the compressed archive locally
            local_archive = os.path.join(
                self.local_output_dir, "aiperf_artifacts.tar.gz"
            )
            self._logger.info(f"Saving archive to {local_archive}")
            with open(local_archive, "wb") as f:
                f.write(cat_result.stdout)

            # Verify the archive was saved correctly
            if not os.path.exists(local_archive):
                raise Exception(
                    f"Local archive file was not created at {local_archive}"
                )

            local_archive_size = os.path.getsize(local_archive)
            self._logger.info(
                f"Local archive created with size: {local_archive_size} bytes"
            )

            if local_archive_size == 0:
                raise Exception("Local archive file is empty")

            # Extract the compressed archive locally
            import tarfile

            self._logger.info("Extracting compressed archive locally...")
            with tarfile.open(local_archive, "r:gz") as tar:
                tar.extractall(path=self.local_output_dir)

            self._logger.info("Archive extraction completed")

            # Remove the temporary archive file
            os.remove(local_archive)
            self._logger.info("Temporary archive file removed")

            # Count extracted files
            extracted_files = []
            for root, dirs, files in os.walk(self.local_output_dir):
                for file in files:
                    if file == "test_metadata.json":  # Skip our metadata file
                        continue
                    rel_path = os.path.relpath(
                        os.path.join(root, file), self.local_output_dir
                    )
                    extracted_files.append(rel_path)

            self._logger.info(
                f"Extracted {len(extracted_files)} artifacts to {self.local_output_dir}"
            )

            # Save metadata about the test run
            metadata = {
                "job_name": self.job_name,
                "namespace": self.namespace,
                "timestamp": int(time.time()),
                "extraction_method": "compressed_archive_transfer",
                "load_config": {
                    "concurrency": self.load_config.concurrency,
                    "input_tokens_mean": self.load_config.input_tokens_mean,
                    "output_tokens_mean": self.load_config.output_tokens_mean,
                    "request_count": self.load_config.request_count,
                    "duration_minutes": self.load_config.duration_minutes,
                    "model": self.load_config.model,
                    "endpoint_url": self.load_config.endpoint_url,
                },
            }

            metadata_file = os.path.join(self.local_output_dir, "test_metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            self._logger.info(
                f"Artifact extraction completed successfully. Extracted {len(extracted_files)} files"
            )

        except Exception as e:
            import traceback

            error_details = f"Exception type: {type(e).__name__}, Message: '{str(e)}', Traceback: {traceback.format_exc()}"
            self._logger.error(f"Failed to extract artifacts: {error_details}")
            # Don't re-raise, just return empty list so we can see what happened
            return []

    async def _cleanup(self):
        """Clean up the AI Perf job and associated pods"""
        if self._job_created:
            # Always try to extract logs and artifacts before cleanup
            self._logger.info(
                f"Extracting logs and artifacts before cleanup for job {self.job_name}"
            )
            try:
                # Extract logs first (always, regardless of test outcome)
                await self.get_logs()

                # Try to extract any available results/artifacts
                try:
                    await self.get_results()
                except Exception as result_e:
                    import traceback

                    error_details = f"Exception type: {type(result_e).__name__}, Message: '{result_e}'"
                    self._logger.warning(
                        f"Could not extract results during cleanup: {error_details}"
                    )

            except Exception as extract_e:
                import traceback

                error_details = f"Exception type: {type(extract_e).__name__}, Message: '{extract_e}', Traceback: {traceback.format_exc()}"
                self._logger.warning(
                    f"Error during log/artifact extraction in cleanup: {error_details}"
                )

            # Now proceed with cleanup
            if self._batch_api:
                try:
                    # Delete the job with foreground propagation to cascade to pods
                    from kubernetes_asyncio.client.models import V1DeleteOptions

                    delete_options = V1DeleteOptions(propagation_policy="Foreground")

                    await self._batch_api.delete_namespaced_job(
                        name=self.job_name,
                        namespace=self.namespace,
                        body=delete_options,
                    )
                    self._logger.info(
                        f"AI Perf job {self.job_name} deleted with cascade to pods"
                    )

                    # Additional explicit pod cleanup as fallback
                    if self._core_api:
                        try:
                            # Wait a moment for cascade deletion to start
                            await asyncio.sleep(2)

                            # Check for any remaining pods and delete them explicitly
                            pods = await self._core_api.list_namespaced_pod(
                                namespace=self.namespace,
                                label_selector=f"job-name={self.job_name}",
                            )

                            for pod in pods.items:
                                try:
                                    await self._core_api.delete_namespaced_pod(
                                        name=pod.metadata.name,
                                        namespace=self.namespace,
                                        body=delete_options,
                                    )
                                    self._logger.info(
                                        f"Explicitly deleted pod {pod.metadata.name}"
                                    )
                                except exceptions.ApiException as pod_e:
                                    if pod_e.status != 404:  # Ignore if already deleted
                                        self._logger.warning(
                                            f"Failed to delete pod {pod.metadata.name}: {pod_e}"
                                        )

                        except Exception as pod_cleanup_e:
                            self._logger.warning(
                                f"Failed during pod cleanup: {pod_cleanup_e}"
                            )

                except exceptions.ApiException as e:
                    if e.status != 404:  # Ignore if already deleted
                        self._logger.warning(
                            f"Failed to delete job {self.job_name}: {e}"
                        )

    async def run(self, wait_for_completion: bool = True) -> Dict[str, Any]:
        """Run the AI Perf load test"""
        await self._init_kubernetes()

        try:
            await self._create_job()

            if wait_for_completion:
                success = await self._wait_for_completion()
                if success:
                    self._logger.info("AI Perf load test completed successfully")
                    return {"success": True, "job_name": self.job_name}
                else:
                    self._logger.error("AI Perf load test failed")
                    return {"success": False, "job_name": self.job_name}
            else:
                self._logger.info(f"AI Perf job {self.job_name} started")
                return {"success": True, "job_name": self.job_name}

        except Exception as e:
            self._logger.error(f"AI Perf load test failed: {e}")
            raise

    async def __aenter__(self):
        await self._init_kubernetes()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Always wait for completion before cleanup, regardless of run() parameter
        if self._job_created:
            try:
                self._logger.info(
                    "Waiting for AI Perf test completion before cleanup..."
                )
                success = await self._wait_for_completion()
                if success:
                    self._logger.info("AI Perf test completed successfully")
                else:
                    self._logger.warning("AI Perf test failed or was terminated")
            except Exception as e:
                self._logger.error(f"Error during completion wait: {e}")

        # _cleanup() will now handle extracting logs and results automatically
        await self._cleanup()

    async def terminate(self):
        """Gracefully terminate the running AI Perf test (similar to test_deployment terminate clients)"""
        self._logger.info(f"Terminating AI Perf test in job {self.job_name}")
        self._terminated = True

        if not self._job_created:
            self._logger.warning("No job created to terminate")
            return

        try:
            # Find and signal the pod to terminate gracefully (similar to client.py signal handling)
            pods = []
            pod_generator = kr8s.get(
                "pods",
                namespace=self.namespace,
                label_selector=f"job-name={self.job_name}",
            )
            for pod in pod_generator:
                pods.append(pod)

            if pods:
                pod = pods[0]
                self._logger.info(f"Sending termination signal to pod {pod.name}")

                # Try to send SIGINT first for graceful shutdown (following client.py pattern)
                try:
                    result = await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(pod.exec, ["pkill", "-SIGINT", "aiperf"])
                        ),
                        timeout=10.0,
                    )
                    if result.returncode == 0:
                        self._logger.info("SIGINT sent to aiperf process")
                    else:
                        self._logger.warning(
                            "SIGINT failed, process may already be stopped"
                        )
                except Exception as e:
                    self._logger.warning(f"Failed to send SIGINT: {e}")

                # Wait a moment for graceful shutdown
                await asyncio.sleep(5)

                # If still running, send SIGTERM
                try:
                    result = await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(pod.exec, ["pkill", "-SIGTERM", "aiperf"])
                        ),
                        timeout=10.0,
                    )
                    if result.returncode == 0:
                        self._logger.info("SIGTERM sent to aiperf process")
                    await asyncio.sleep(2)
                except Exception as e:
                    self._logger.warning(f"Failed to send SIGTERM: {e}")

                # Mark termination complete in the container
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(
                                pod.exec,
                                [
                                    "touch",
                                    f"{self.container_log_dir}/status/terminated",
                                ],
                            )
                        ),
                        timeout=5.0,
                    )
                except Exception:
                    pass  # Best effort

                self._logger.info("AI Perf test termination completed")
            else:
                self._logger.warning("No pods found to terminate")

        except Exception as e:
            self._logger.error(f"Error during termination: {e}")
            # Continue with cleanup even if termination failed

    async def is_running(self) -> bool:
        """Check if the AI Perf test is currently running"""
        if not self._job_created or self._terminated:
            return False

        try:
            # Check if job still exists and is active
            assert self._batch_api is not None, "Kubernetes API not initialized"
            job = await self._batch_api.read_namespaced_job(
                name=self.job_name, namespace=self.namespace
            )

            # Job is running if it's not completed and not failed
            is_active = job.status.completion_time is None and (
                job.status.failed is None or job.status.failed == 0
            )

            if is_active:
                # Double check by looking for running pods
                pods = []
                pod_generator = kr8s.get(
                    "pods",
                    namespace=self.namespace,
                    label_selector=f"job-name={self.job_name}",
                )
                for pod in pod_generator:
                    if pod.status.phase in ["Running", "Pending"]:
                        pods.append(pod)
                return len(pods) > 0
            else:
                return False

        except exceptions.ApiException as e:
            if e.status == 404:  # Job not found
                return False
            else:
                self._logger.warning(f"Error checking job status: {e}")
                return False
        except Exception as e:
            self._logger.warning(f"Error checking if running: {e}")
            return False


async def main():
    """Example usage of ManagedAIPerfDeployment"""
    logging.basicConfig(level=logging.INFO)

    # Configure load test
    load_config = LoadConfig(
        concurrency=4,
        input_tokens_mean=512,
        output_tokens_mean=64,
        request_count=100,
        model="Qwen/Qwen3-0.6B",
        endpoint_url="http://trtllm-agg-frontend:8000",
        streaming=True,
        temperature=0.0,
    )

    # Set up output directory (following the ManagedDeployment pattern)
    workspace_dir = _get_workspace_dir()
    log_dir = os.path.join(workspace_dir, "test_rolling_restart")

    # Run load test
    async with ManagedAIPerfDeployment(
        log_dir=log_dir,
        load_config=load_config,
        namespace="default",
        job_name="example-load-test",
    ) as aiperf:
        results = await aiperf.run(wait_for_completion=True)

        print("AI Perf Results:")
        print(f"Success: {results['success']}")
        print(f"Job Name: {results['job_name']}")

        if aiperf.local_output_dir:
            print(f"Results saved to: {aiperf.local_output_dir}")

        if results["results"]:
            print("Performance Summary:")
            print(json.dumps(results["results"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
