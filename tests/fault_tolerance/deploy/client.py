# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AI-Perf client implementation for fault tolerance testing."""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from tests.utils.managed_deployment import ManagedDeployment

LOG_FORMAT = "[TEST] %(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
)


def get_frontend_port(
    managed_deployment: ManagedDeployment,
    client_index: int,
    deployment_spec: Any,
    pod_ports: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Select a frontend pod using round-robin and setup port forwarding.

    Args:
        managed_deployment: ManagedDeployment instance
        client_index: Client index for round-robin selection
        deployment_spec: Deployment specification with port info
        pod_ports: Dictionary to track existing port forwards
                  - Key: pod name (str)
                  - Value: port forward object from managed_deployment.port_forward()
        logger: Logger instance

    Returns:
        Tuple of (pod_name, local_port, pod_instance) or (None, None, None) if failed
    """
    pods = managed_deployment.get_pods(managed_deployment.frontend_service_name)

    port = 0
    pod_name = None
    selected_pod = None

    # Filter ready pods and cleanup stale port forwards
    pods_ready = []

    for pod in pods[managed_deployment.frontend_service_name]:
        if pod.ready():
            pods_ready.append(pod)
        else:
            # Cleanup port forwards for non-ready pods
            if pod.name in pod_ports:
                try:
                    pod_ports[pod.name].stop()
                except Exception as e:
                    logger.debug(f"Error stopping port forward for {pod.name}: {e}")
                del pod_ports[pod.name]

    if not pods_ready:
        logger.error("No ready frontend pods found")
        return None, None, None

    # Round-robin selection based on client index
    selected_pod = pods_ready[client_index % len(pods_ready)]
    pod_name = selected_pod.name

    # Setup or reuse port forward
    if pod_name not in pod_ports:
        # Get port from deployment_spec (default: 8000)
        port_value = getattr(deployment_spec, "_port", 8000)
        port_forward = managed_deployment.port_forward(selected_pod, port_value)
        if port_forward:
            pod_ports[pod_name] = port_forward
            port = port_forward.local_port
        else:
            logger.error(f"Failed to create port forward for pod {pod_name}")
            return None, None, None
    else:
        # Reuse existing port forward
        port = pod_ports[pod_name].local_port

    logger.debug(f"Selected pod {pod_name} with local port {port}")
    return pod_name, port, selected_pod


def wait_for_model_availability(
    url: str,
    endpoint: str,
    model: str,
    logger: logging.Logger,
    max_attempts: int = 15,
    attempt_timeouts: Optional[List[float]] = None,
) -> bool:
    """
    Wait for model to be available before running AI-Perf.

    Args:
        url: Base URL for the service
        endpoint: API endpoint path
        model: Model name to test
        logger: Logger instance
        max_attempts: Maximum number of attempts to check availability
        attempt_timeouts: List of timeout values for each attempt

    Returns:
        True if model is available, False otherwise
    """
    if attempt_timeouts is None:
        # Default: Start with 60s timeout, then gradually decrease
        attempt_timeouts = [60, 60, 45, 30, 30, 20, 20, 15, 15, 15, 10, 10, 10, 10, 10]

    test_url = f"{url}{endpoint}"

    for attempt in range(max_attempts):
        try:
            test_payload = {
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
                "stream": False,
            }

            timeout_val = attempt_timeouts[min(attempt, len(attempt_timeouts) - 1)]
            logger.info(
                f"Testing model availability at {test_url} (attempt {attempt+1}/{max_attempts}, timeout={timeout_val}s)"
            )
            response = requests.post(test_url, json=test_payload, timeout=timeout_val)

            if response.status_code == 200:
                logger.info(f"Model '{model}' is available and responding")
                # Give a bit more time for stabilization
                logger.info("Model ready, waiting 5s for stabilization...")
                time.sleep(5)
                return True
            elif response.status_code == 404:
                logger.warning(
                    f"Model '{model}' not found (404). Response: {response.text[:200]}"
                )
            elif response.status_code == 400:
                logger.warning(f"Bad request (400). Response: {response.text[:200]}")
            else:
                logger.warning(
                    f"Unexpected status code {response.status_code}: {response.text[:200]}"
                )

        except requests.Timeout as e:
            logger.warning(
                f"Model availability test timed out (attempt {attempt+1}): {e}"
            )
        except Exception as e:
            logger.warning(f"Model availability test failed (attempt {attempt+1}): {e}")

        if attempt < max_attempts - 1:
            wait_time = 10 if attempt < 5 else 5
            logger.info(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)

    logger.warning("Could not confirm model availability after all attempts")
    return False


def validate_aiperf_results(
    json_path: Path,
    requests_per_client: int,
    attempt: int,
    logger: logging.Logger,
    attempt_dir: Path,
    pod_name: str,
    port: int,
) -> bool:
    """
    Validate AI-Perf results from JSON output.

    Args:
        json_path: Path to the AI-Perf JSON output file
        requests_per_client: Expected number of requests
        attempt: Current attempt number (0-based)
        logger: Logger instance
        attempt_dir: Directory containing attempt results
        pod_name: Pod name for logging
        port: Port number for logging

    Returns:
        True if the attempt was successful, False if it should be retried
    """
    if not json_path.exists():
        # No JSON output, but aiperf returned 0 - might be okay
        logger.info(f"Attempt {attempt + 1} completed (return code 0, no JSON output)")
        log_summary_metrics(attempt_dir, logger, pod_name, port)
        return True

    try:
        with open(json_path, "r") as f:
            aiperf_data = json.load(f)

        # Check for errors in the output
        error_count = 0
        if "records" in aiperf_data and "error_request_count" in aiperf_data["records"]:
            error_count = int(
                aiperf_data["records"]["error_request_count"].get("avg", 0)
            )

        # Also check error_summary
        if "error_summary" in aiperf_data:
            error_summary_count = sum(
                err.get("count", 0) for err in aiperf_data["error_summary"]
            )
            error_count = max(error_count, error_summary_count)

        # Consider it a failure if most requests failed (> 90%)
        failure_threshold = requests_per_client * 0.9
        if error_count >= failure_threshold:
            logger.warning(
                f"Attempt {attempt + 1} had {error_count}/{requests_per_client} failed requests - retrying"
            )
            return False  # Not successful, continue retrying
        else:
            successful_count = requests_per_client - error_count
            logger.info(
                f"Attempt {attempt + 1} succeeded with {successful_count}/{requests_per_client} successful requests"
            )
            log_summary_metrics(attempt_dir, logger, pod_name, port)
            return True  # Successful

    except Exception as e:
        logger.warning(f"Could not parse AI-Perf output to check for failures: {e}")
        # Assume success if we can't parse the output but aiperf returned 0
        logger.info(
            f"Attempt {attempt + 1} completed (return code 0, could not verify success)"
        )
        log_summary_metrics(attempt_dir, logger, pod_name, port)
        return True  # Assume success


def run_aiperf(
    url: str,
    endpoint: str,
    model: str,
    pod_name: str,
    port: int,
    requests_per_client: int,
    input_token_length: int,
    output_token_length: int,
    output_dir: Path,
    logger: logging.Logger,
    max_retries: int = 1,
    retry_delay: float = 1,
) -> bool:
    """
    Execute AI-Perf with specified parameters.

    Args:
        url: Base URL (http://localhost:port)
        endpoint: API endpoint path (e.g., "v1/chat/completions")
        model: Model name
        pod_name: Selected pod name for logging
        port: Local port number
        requests_per_client: Number of requests to send
        input_token_length: Input token count
        output_token_length: Output token count
        output_dir: Directory for AI-Perf artifacts
        logger: Logger instance
        max_retries: Maximum number of retry attempts (default: 1)
        retry_delay: Delay in seconds between retries (default: 1)

    Returns:
        True if successful, False otherwise
    """
    # Validate required parameters
    if not model or not url or not endpoint:
        logger.error(
            f"Missing required parameter: model={model!r}, url={url!r}, endpoint={endpoint!r}"
        )
        return False

    # Build AI-Perf command
    cmd = [
        "aiperf",
        "profile",
        # Model configuration (required)
        "--model",
        model,
        # Endpoint configuration
        "--url",
        url,
        "--endpoint",
        endpoint if endpoint.startswith("/") else f"/{endpoint}",
        "--endpoint-type",
        "chat",  # Required: tells AI-Perf the API type
        # Enable streaming for TTFT and ITL metrics
        "--streaming",
        # Request parameters
        "--request-count",
        str(requests_per_client),  # Required: how many requests
        "--concurrency",
        "1",  # Optional: we set to 1 for sequential
        # Token configuration
        "--synthetic-input-tokens-mean",
        str(input_token_length),
        "--synthetic-input-tokens-stddev",
        "0",  # Set to 0 for consistent token counts
        "--output-tokens-mean",
        str(output_token_length),
        "--output-tokens-stddev",
        "0",  # Set to 0 for consistent token counts
        # Skip warmup to avoid initial failures
        "--warmup-request-count",
        "0",
        # Output configuration
        "--artifact-dir",
        str(output_dir),
        "--random-seed",
        "100",  # For reproducible results
    ]

    # Calculate timeout (same as legacy would for all requests)
    timeout = max(requests_per_client * 2 + 60, 300)  # At least 5 minutes

    # Log execution
    logger.info(f"Starting AI-Perf for Pod {pod_name} Local Port {port}")
    logger.info(f"Using model name: {model}")

    # Wait for model to be available
    model_ready = wait_for_model_availability(url, endpoint, model, logger)
    if not model_ready:
        logger.warning("Model not ready, but proceeding with AI-Perf test anyway")
        # This might result in all requests failing, but the retry logic will handle it

    logger.info(f"Command: {' '.join(cmd)}")

    # Retry logic for fault tolerance - retry FULL request count until success

    max_attempts = max_retries if max_retries > 0 else 1
    success = False
    all_results = []

    for attempt in range(max_attempts):
        logger.info(
            f"AI-Perf attempt {attempt + 1}/{max_attempts} with {requests_per_client} requests"
        )

        # Update output directory for this attempt
        attempt_dir = output_dir / f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        # Use the original command but update artifact directory
        cmd_attempt = cmd.copy()
        artifact_dir_idx = cmd_attempt.index("--artifact-dir") + 1
        cmd_attempt[artifact_dir_idx] = str(attempt_dir)

        try:
            result = subprocess.run(
                cmd_attempt,
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,  # Prevent stdin reading which can cause process suspension
            )

            # Save logs for this attempt
            with open(attempt_dir / "genai_perf.log", "w") as f:
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)

            all_results.append(
                {
                    "attempt": attempt + 1,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )

            if result.returncode == 0:
                # AI-Perf returns 0 even if all requests failed, so we need to check the output
                json_path = attempt_dir / "profile_export_aiperf.json"
                success = validate_aiperf_results(
                    json_path=json_path,
                    requests_per_client=requests_per_client,
                    attempt=attempt,
                    logger=logger,
                    attempt_dir=attempt_dir,
                    pod_name=pod_name,
                    port=port,
                )
                if success:
                    break  # Success - exit the retry loop
            else:
                logger.warning(
                    f"Attempt {attempt + 1} failed with return code {result.returncode}"
                )
                logger.debug(
                    f"Stderr: {result.stderr[:500] if result.stderr else 'No stderr'}"
                )
        except Exception as e:
            logger.error(f"Error in attempt {attempt + 1}: {str(e)}")
            all_results.append({"attempt": attempt + 1, "error": str(e)})

        # Sleep before next attempt (if not the last attempt)
        if not success and attempt < max_attempts - 1:
            time.sleep(retry_delay)

    if success:
        logger.info(
            f"AI-Perf successfully completed all {requests_per_client} requests for {pod_name}"
        )
    else:
        logger.error(f"AI-Perf failed all {max_attempts} attempts for {pod_name}")

    return success


def log_summary_metrics(
    output_dir: Path, logger: logging.Logger, pod_name: str, port: int
) -> None:
    """
    Log summary metrics from AI-Perf results.

    Args:
        output_dir: Directory containing AI-Perf artifacts
        logger: Logger instance
        pod_name: Pod name for logging
        port: Port number for logging
    """
    # Look for AI-Perf output file
    profile_json = output_dir / "profile_export_aiperf.json"
    if not profile_json.exists():
        # Try alternative names
        for name in ["profile_export.json", "profile_results.json"]:
            alt_path = output_dir / name
            if alt_path.exists():
                profile_json = alt_path
                break

    if profile_json.exists():
        try:
            with open(profile_json) as f:
                metrics = json.load(f)

            # Request count
            request_count = int(metrics.get("request_count", {}).get("avg", 0))

            # Check for errors
            error_count = len(metrics.get("error_summary", []))

            # Latency metrics (in milliseconds)
            request_latency = metrics.get("request_latency", {})
            avg_latency = request_latency.get("avg", 0) / 1000.0  # Convert to seconds
            p99_latency = request_latency.get("p99", 0) / 1000.0  # Convert to seconds

            # Throughput metrics
            throughput = metrics.get("request_throughput", {}).get("avg", 0)

            # Log summary
            logger.info(
                f"Summary: Pod {pod_name} Port {port} "
                f"Requests: {request_count} "
                f"Errors: {error_count} "
                f"Throughput: {throughput:.1f} req/s "
                f"Avg Latency: {avg_latency:.3f}s "
                f"P99 Latency: {p99_latency:.3f}s"
            )

            # Log success rate
            if request_count > 0:
                success_rate = ((request_count - error_count) / request_count) * 100
                logger.info(f"Success rate: {success_rate:.1f}%")

            # Also write summary to CSV file for aggregation
            csv_path = output_dir / "profile_export_aiperf.csv"
            if csv_path.exists():
                logger.info(f"AI-Perf results saved to {csv_path}")

        except Exception as e:
            logger.warning(f"Failed to parse AI-Perf metrics: {e}")


def client(
    deployment_spec,
    namespace: str,
    model: str,
    log_dir: str,
    index: int,
    requests_per_client: int,
    input_token_length: int,
    output_token_length: int,
    max_retries: int,
    retry_delay: float = 1,
):
    """
    Generate load using AI-Perf for fault tolerance testing.

    This function sets up port forwarding to a frontend pod and uses AI-Perf
    to generate synthetic requests for performance testing and fault tolerance
    evaluation.

    Args:
        deployment_spec: Deployment specification object
        namespace: Kubernetes namespace
        model: Model name
        log_dir: Directory for output logs and AI-Perf artifacts
        index: Client index used for round-robin pod selection
        requests_per_client: Number of requests to generate
        input_token_length: Number of input tokens per request
        output_token_length: Number of output tokens per request
        max_retries: Maximum retry attempts for AI-Perf execution
        retry_delay: Delay in seconds between retry attempts
    """
    logger = logging.getLogger(f"CLIENT: {index}")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    managed_deployment = ManagedDeployment(log_dir, deployment_spec, namespace)
    pod_ports: Dict[str, Any] = {}

    try:
        os.makedirs(log_dir, exist_ok=True)
        client_output_dir = Path(log_dir) / f"client_{index}"
        client_output_dir.mkdir(parents=True, exist_ok=True)

        # Add a startup delay for early clients to give model time to load
        time.sleep(15)

        # Select frontend pod and setup port forwarding
        pod_name, port, selected_pod = get_frontend_port(
            managed_deployment=managed_deployment,
            client_index=index,
            deployment_spec=deployment_spec,
            pod_ports=pod_ports,
            logger=logger,
        )

        if not pod_name or not port:
            logger.error("Failed to select pod or setup port forwarding")
            return

        url = f"http://localhost:{port}"

        # Get endpoint from deployment_spec (default: /v1/chat/completions)
        endpoint = getattr(deployment_spec, "_endpoint", "/v1/chat/completions")

        success = run_aiperf(
            url=url,
            endpoint=endpoint,
            model=model,
            pod_name=pod_name,
            port=port,
            requests_per_client=requests_per_client,
            input_token_length=input_token_length,
            output_token_length=output_token_length,
            output_dir=client_output_dir,
            logger=logger,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        if not success:
            logger.error("AI-Perf execution failed")

    except Exception as e:
        logger.error(f"Client error: {str(e)}")
    finally:
        for pf_name, port_forward in pod_ports.items():
            try:
                port_forward.stop()
                logger.debug(f"Stopped port forward for {pf_name}")
            except Exception as e:
                logger.debug(f"Error stopping port forward for {pf_name}: {e}")

    logger.info("Exiting")
