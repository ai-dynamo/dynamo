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
    #        "seed": 10,
    "ignore_eos": True,
    "min_tokens": 0,
    "stream": False,
}


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,  # ISO 8601 UTC format
)


def _get_random_prompt(length):
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

    while retry_attempts:
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

            if response.status_code != 200:
                time.sleep(retry_delay)
                retry_attempts -= 1
                continue
            else:
                break

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
            time.sleep(retry_delay)
            retry_attempts -= 1
            continue

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
):
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
                pods = managed_deployment.get_pods(
                    managed_deployment.frontend_service_name
                )
                port = 0
                pod_name = None

                pods_ready = []

                for pod in pods[managed_deployment.frontend_service_name]:
                    if pod.ready():
                        pods_ready.append(pod)
                    else:
                        if pod.name in pod_ports:
                            pod_ports[pod.name].stop()
                            del pod_ports[pod.name]

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

                url = f"http://localhost:{port}/{deployment_spec.endpoint}"

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
                logger.info(
                    f"Request: {i} Pod {pod_name} Local Port {port} Status: {result['results'][-1]['status']} Latency: {result['results'][-1]['request_elapsed_time']}"
                )

                log.write(json.dumps(result) + "\n")
                log.flush()
                if result["total_time"] < min_elapsed_time:
                    time.sleep(min_elapsed_time - result["total_time"])

    except Exception as e:
        logger.error(str(e))
    logger.info("Exiting")
