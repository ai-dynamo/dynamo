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
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
        port_forward = managed_deployment.port_forward(
            selected_pod, deployment_spec.port
        )
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


def run_aiperf(
    url: str,
    endpoint: str,
    model: str,
    pod_name: str,
    port: int,
    requests_per_client: int,
    input_token_length: int,
    output_token_length: int,
    max_request_rate: float,
    output_dir: Path,
    logger: logging.Logger,
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
        max_request_rate: Maximum request rate (0 for unlimited)
        output_dir: Directory for AI-Perf artifacts
        logger: Logger instance

    Returns:
        True if successful, False otherwise
    """
    # Build AI-Perf command
    cmd = [
        "genai-perf",
        "profile",
        # Model configuration (required)
        "--model",
        model,
        "--tokenizer",
        model,
        # Endpoint configuration
        "--url",
        url,
        "--endpoint",
        f"/{endpoint}",  # Optional: defaults to /v1/chat/completions
        "--endpoint-type",
        "chat",  # Required: tells AI-Perf the API type
        # Request parameters
        "--request-count",
        str(requests_per_client),  # Required: how many requests
        "--concurrency",
        "1",  # Optional: we set to 1 for sequential (legacy behavior)
        # Token configuration
        "--synthetic-input-tokens-mean",
        str(input_token_length),
        "--synthetic-input-tokens-stddev",
        "0",  # Set to 0 for consistent token counts
        "--output-tokens-mean",
        str(output_token_length),
        "--output-tokens-stddev",
        "0",  # Set to 0 for consistent token counts
        # Enforce exact output length (for test consistency)
        "--extra-inputs",
        f"max_tokens:{output_token_length}",
        "--extra-inputs",
        f"min_tokens:{output_token_length}",
        "--extra-inputs",
        "ignore_eos:true",  # Force exact length
        "--extra-inputs",
        '{"nvext":{"ignore_eos":true}}',
        # Rate limiting
        "--request-rate",
        str(max_request_rate) if max_request_rate > 0 else "inf",
        # Warmup
        "--warmup-request-count",
        "3",
        # Output configuration
        "--artifact-dir",
        str(output_dir),
        # Fixed parameters
        "--streaming",
        "false",  # Non-streaming mode
        "--random-seed",
        "100",  # For reproducible results
        # Additional parameters for reliability
        "--",
        "-v",
        "--max-threads",
        str(max(int(requests_per_client * 0.1), 10)),
        "-H",
        "Authorization: Bearer NOT USED",
        "-H",
        "Accept: text/event-stream",
    ]

    # Calculate timeout (same as legacy would for all requests)
    timeout = max(requests_per_client * 2 + 60, 300)  # At least 5 minutes

    # Log execution
    logger.info(f"Starting AI-Perf for Pod {pod_name} Local Port {port}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(output_dir)
        )

        # Save both stdout and stderr to a single log file
        with open(output_dir / "genai_perf.log", "w") as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n\n=== STDERR ===\n")
            f.write(result.stderr)

        if result.returncode == 0:
            logger.info(f"AI-Perf completed successfully for {pod_name}")
            log_summary_metrics(output_dir, logger, pod_name, port)
            return True
        else:
            logger.error(f"AI-Perf failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"stderr: {result.stderr[:500]}...")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"AI-Perf timed out after {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"AI-Perf execution error: {e}")
        return False


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
    profile_json = output_dir / "profile_export.json"
    if not profile_json.exists():
        # Try alternative name
        profile_json = output_dir / "profile_results.json"

    if profile_json.exists():
        try:
            with open(profile_json) as f:
                metrics = json.load(f)

            # Extract key metrics safely
            request_count = metrics.get("request_count", 0)
            error_count = metrics.get("error_count", 0)

            request_latencies = metrics.get("request_latencies", {})
            avg_latency = request_latencies.get("mean", 0)
            p99_latency = request_latencies.get("p99", 0)

            throughput = metrics.get("request_throughput", {}).get("value", 0)

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
                success_rate = (request_count - error_count) / request_count * 100
                logger.info(f"Success rate: {success_rate:.1f}%")

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
    max_request_rate: float,
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
        max_retries: Maximum retry attempts (AI-Perf handles retries internally)
        max_request_rate: Maximum requests per second (0 for unlimited)
        retry_delay: Delay between retries (AI-Perf handles retries internally)
    """
    logger = logging.getLogger(f"CLIENT: {index}")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    managed_deployment = ManagedDeployment(log_dir, deployment_spec, namespace)
    pod_ports: Dict[str, Any] = {}

    try:
        os.makedirs(log_dir, exist_ok=True)
        client_output_dir = Path(log_dir) / f"client_{index}"
        client_output_dir.mkdir(parents=True, exist_ok=True)

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

        success = run_aiperf(
            url=url,
            endpoint=deployment_spec.endpoint,
            model=model,
            pod_name=pod_name,
            port=port,
            requests_per_client=requests_per_client,
            input_token_length=input_token_length,
            output_token_length=output_token_length,
            max_request_rate=max_request_rate,
            output_dir=client_output_dir,
            logger=logger,
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
