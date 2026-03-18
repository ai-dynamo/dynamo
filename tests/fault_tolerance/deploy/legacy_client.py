# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Legacy custom client implementation for fault tolerance testing.

This is the original client implementation that was used before migrating to AI-Perf.
It sends direct HTTP requests and logs results in JSONL format.
"""

import json
import logging
import os
import random
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict

import requests

from tests.utils.managed_deployment import ManagedDeployment

LOG_FORMAT = "[TEST] %(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Base payload template for chat completions
payload = {
    "model": "",
    "messages": [
        {
            "role": "user",
            "content": "",
        }
    ],
    "max_tokens": 0,
    "temperature": 0.1,
    "ignore_eos": True,
    "min_tokens": 0,
    "stream": False,
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
)


def _get_random_prompt(length):
    """Generate a random prompt of specified token length.

    Args:
        length: Approximate number of tokens to generate

    Returns:
        String containing random words
    """
    word_list = [f"{i}" for i in range(10)]
    return " ".join(random.choices(word_list, k=length))


def _single_request(
    url,
    pod,
    payload,
    model,
    logger,
    retry_attempts=1,
    input_token_length=100,
    output_token_length=100,
    timeout=30,
    retry_delay=1,
):
    """Execute a single HTTP request with retry logic.

    Args:
        url: Full URL to send request to
        pod: Pod name for logging
        payload: Base payload template
        model: Model name to use
        logger: Logger instance
        retry_attempts: Number of retry attempts on failure
        input_token_length: Number of input tokens
        output_token_length: Number of output tokens
        timeout: Request timeout in seconds
        retry_delay: Delay between retries in seconds

    Returns:
        Dictionary containing request results and timing
    """
    prompt = _get_random_prompt(input_token_length)
    payload_copy = deepcopy(payload)
    payload_copy["messages"][0]["content"] = prompt
    payload_copy["max_tokens"] = output_token_length
    payload_copy["min_tokens"] = output_token_length
    payload_copy["model"] = model
    response = None
    end_time = None
    start_time = time.time()
    results = []

    # Convert retries to total attempts (1 initial attempt + N retries)
    attempts_remaining = 1 + max(0, retry_attempts)

    while attempts_remaining:
        start_request_time = time.time()
        response = None

        try:
            response = requests.post(
                url,
                json=payload_copy,
                timeout=timeout,
            )
            end_time = time.time()

            content = None
            if response.status_code == 200:
                try:
                    content = response.json()
                except ValueError:
                    pass

            results.append(
                {
                    "status": response.status_code,
                    "result": content,
                    "request_elapsed_time": end_time - start_request_time,
                    "url": url,
                    "pod": pod,
                }
            )

            # Success - exit immediately
            if response.status_code == 200:
                break

            # Failure - retry if we have attempts left
            attempts_remaining -= 1
            if attempts_remaining == 0:
                break
            time.sleep(retry_delay)

        except (requests.RequestException, requests.Timeout) as e:
            results.append(
                {
                    "status": str(e),
                    "result": None,
                    "request_elapsed_time": time.time() - start_request_time,
                    "url": url,
                    "pod": pod,
                }
            )

            # Exception - retry if we have attempts left
            attempts_remaining -= 1
            if attempts_remaining == 0:
                break
            time.sleep(retry_delay)

    return {
        "time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
        "total_time": time.time() - start_time,
        "url": url,
        "pod": pod,
    }


def client(
    deployment_spec,
    namespace,
    model,
    log_dir,
    index,
    requests_per_client,
    input_token_length,
    output_token_length,
    max_retries,
    max_request_rate,
    retry_delay=1,
    continuous_load=False,
):
    """Legacy custom client for fault tolerance testing.

    This client sends individual HTTP requests with rate limiting and logs
    results in JSONL format. Each client runs independently and logs to
    its own file.

    Args:
        deployment_spec: Deployment specification object
        namespace: Kubernetes namespace
        model: Model name to test
        log_dir: Directory for output logs
        index: Client index for identification
        requests_per_client: Number of requests to send
        input_token_length: Number of input tokens per request
        output_token_length: Number of output tokens per request
        max_retries: Maximum retry attempts per request
        max_request_rate: Maximum requests per second (for rate limiting)
        retry_delay: Delay in seconds between retries
        continuous_load: If True, use continuous load instead of fixed request count
    """
    if continuous_load:
        raise ValueError("Continuous load is not supported for legacy client")

    logger = logging.getLogger(f"CLIENT: {index}")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    managed_deployment = ManagedDeployment(log_dir, deployment_spec, namespace)
    pod_ports: Dict[str, Any] = {}

    min_elapsed_time = (1 / max_request_rate) if max_request_rate > 0 else 0.0

    try:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"client_{index}.log.txt")

        with open(log_path, "w") as log:
            for i in range(requests_per_client):
                # Get available pods
                pods = managed_deployment.get_pods(
                    [managed_deployment.frontend_service_name]
                )
                port = 0
                pod_name = None

                pods_ready = []

                # Filter ready pods and cleanup stale port forwards
                for pod in pods[managed_deployment.frontend_service_name]:
                    if pod.ready():
                        pods_ready.append(pod)
                    else:
                        if pod.name in pod_ports:
                            pod_ports[pod.name].stop()
                            del pod_ports[pod.name]

                # Setup port forwarding for selected pod
                if pods_ready:
                    pod = pods_ready[i % len(pods_ready)]
                    if pod.name not in pod_ports:
                        port_forward = managed_deployment.port_forward(
                            pod, deployment_spec.port
                        )
                        if port_forward:
                            pod_ports[pod.name] = port_forward
                    if pod.name in pod_ports:
                        port = pod_ports[pod.name].local_port
                        pod_name = pod.name

                # Construct URL
                url = f"http://localhost:{port}/{deployment_spec.endpoint}"

                # Execute request
                result = _single_request(
                    url,
                    pod_name,
                    payload,
                    model,
                    logger,
                    max_retries,
                    input_token_length=input_token_length,
                    output_token_length=output_token_length,
                    retry_delay=retry_delay,
                )

                # Log result
                logger.debug(
                    f"Request: {i} Pod {pod_name} Local Port {port} "
                    f"Status: {result['results'][-1]['status']} "
                    f"Latency: {result['results'][-1]['request_elapsed_time']}"
                )
                # Write to JSONL log file
                log.write(json.dumps(result) + "\n")
                log.flush()

                # Rate limiting
                if result["total_time"] < min_elapsed_time:
                    time.sleep(min_elapsed_time - result["total_time"])

    except Exception as e:
        logger.error(str(e))
    finally:
        # Cleanup port forwards
        for pf_name, port_forward in pod_ports.items():
            try:
                port_forward.stop()
            except Exception:
                pass

    logger.info("Exiting")
