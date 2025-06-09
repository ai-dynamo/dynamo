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


import logging
import time
from multiprocessing import Process, Queue

import psutil
import pytest
import requests

from tests.fault_tolerance.utils.circus_controller import CircusController
from tests.serve.test_dynamo_serve import DynamoServeProcess
from tests.utils.deployment_graph import (
    DeploymentGraph,
    Payload,
    completions_response_handler,
)

text_prompt = "Tell me a short joke about AI."

text_payload = Payload(
    payload={
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "messages": [
            {
                "role": "user",
                "content": text_prompt,  # Shorter prompt
            }
        ],
        "max_tokens": 150,  # Reduced from 500
        "temperature": 0.1,
        "seed": 0,
    },
    expected_log=[],
    expected_response=["AI"],
)

deployment_graphs = {
    "agg": (
        DeploymentGraph(
            module="graphs.mock_agg:Frontend",
            config="configs/mock_agg.yaml",
            directory="/workspace/examples/llm",
            endpoint="v1/chat/completions",
            response_handler=completions_response_handler,
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
}


@pytest.fixture(
    params=[
        pytest.param("agg", marks=[pytest.mark.vllm, pytest.mark.gpu_1]),
    ]
)
def deployment_graph_test(request):
    """
    Fixture that provides different deployment graph test configurations.
    """
    return deployment_graphs[request.param]


def _terminate_process(process):
    try:
        logging.info("Terminating %s", process)
        process.terminate()
    except psutil.AccessDenied:
        logging.warning("Access denied for PID %s", process.pid)
    except psutil.NoSuchProcess:
        logging.warning("PID %s no longer exists", process.pid)
    except psutil.TimeoutExpired:
        logging.warning("PID %s did not terminate before timeout, killing", process.pid)
        process.kill()


def _terminate_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            _terminate_process(child)
        _terminate_process(parent)
    except psutil.NoSuchProcess:
        # Process already terminated
        pass


def client(
    deployment_graph, server_process, payload, queue, index, logger=logging.getLogger()
):
    url = f"http://localhost:{server_process.port}/{deployment_graph.endpoint}"
    start_time = time.time()
    retry_delay = 5
    elapsed = 0.0
    while time.time() - start_time < deployment_graph.timeout:
        elapsed = time.time() - start_time
        try:
            response = requests.post(
                url,
                json=payload.payload,
                timeout=deployment_graph.timeout - elapsed,
            )
        except (requests.RequestException, requests.Timeout) as e:
            logger.warning("Retrying due to Request failed: %s", e)
            time.sleep(retry_delay)
            continue
        logger.info("Response%r", response)
        if response.status_code == 500:
            error = response.json().get("error", "")
            if "no instances" in error:
                logger.warning("Retrying due to no instances available")
                time.sleep(retry_delay)
                continue
        if response.status_code == 404:
            error = response.json().get("error", "")
            if "Model not found" in error:
                logger.warning("Retrying due to model not found")
                time.sleep(retry_delay)
                continue
        # Process the response
        if response.status_code != 200:
            logger.error(
                "Service returned status code %s: %s",
                response.status_code,
                response.text,
            )
            pytest.fail(
                "Service returned status code %s: %s"
                % (response.status_code, response.text)
            )
        else:
            break
    else:
        logger.error(
            "Service did not return a successful response within %s s",
            deployment_graph.timeout,
        )
        pytest.fail(
            "Service did not return a successful response within %s s"
            % deployment_graph.timeout
        )

    content = deployment_graph.response_handler(response)

    logger.info("Client:%s Received Content: %s", index, content)

    # Check for expected responses
    assert content, "Empty response content"

    for expected in payload.expected_response:
        assert expected in content, "Expected '%s' not found in response" % expected

    queue.put((index, content))


def _wait_until_ready(
    deployment_graph, server_process, payload, logger=logging.getLogger()
):
    url = f"http://localhost:{server_process.port}/{deployment_graph.endpoint}"
    start_time = time.time()
    retry_delay = 5
    elapsed = 0.0
    while time.time() - start_time < deployment_graph.timeout:
        elapsed = time.time() - start_time
        try:
            response = requests.post(
                url,
                json=payload.payload,
                timeout=deployment_graph.timeout - elapsed,
            )
        except (requests.RequestException, requests.Timeout) as e:
            logger.warning("Retrying due to Request failed: %s", e)
            time.sleep(retry_delay)
            continue
        logger.info("Response%r", response)
        if response.status_code == 500:
            error = response.json().get("error", "")
            if "no instances" in error:
                logger.warning("Retrying due to no instances available")
                time.sleep(retry_delay)
                continue
        if response.status_code == 404:
            error = response.json().get("error", "")
            if "Model not found" in error:
                logger.warning("Retrying due to model not found")
                time.sleep(retry_delay)
                continue
        # Process the response
        if response.status_code != 200:
            logger.error(
                "Service returned status code %s: %s",
                response.status_code,
                response.text,
            )
            pytest.fail(
                "Service returned status code %s: %s"
                % (response.status_code, response.text)
            )
        else:
            break
    else:
        logger.error(
            "Service did not return a successful response within %s s",
            deployment_graph.timeout,
        )
        pytest.fail(
            "Service did not return a successful response within %s s"
            % deployment_graph.timeout
        )

    content = deployment_graph.response_handler(response)

    logger.info("Received Content: %s", content)

    # Check for expected responses
    assert content, "Empty response content"

    for expected in payload.expected_response:
        assert expected in content, "Expected '%s' not found in response" % expected


@pytest.mark.e2e
@pytest.mark.slow
async def test_worker_failure(deployment_graph_test, request, runtime_services):
    """
    Test dynamo serve deployments with different graph configurations.
    """

    # runtime_services is used to start nats and etcd

    logger = logging.getLogger(request.node.name)
    logger.info("Starting test_deployment")
    deployment_graph, payload = deployment_graph_test

    with DynamoServeProcess(deployment_graph, request) as server_process:
        _wait_until_ready(deployment_graph, server_process, payload)
        circus_controller = CircusController.from_state_file("dynamo")

        procs = []
        queue = Queue()
        for i in range(10):
            procs.append(
                Process(
                    target=client,
                    args=(
                        deployment_graph,
                        server_process,
                        payload,
                        queue,
                        i,
                    ),
                )
            )
            procs[-1].start()

        for x in await circus_controller._list_watchers():
            print(x)
            result = circus_controller.client.call(
                {"command": "list", "properties": {"name": f"{x}"}}
            )

            print(result["pids"])

            if x == "dynamo_mockvllmworker":
                _terminate_process_tree(result["pids"][0])
                break

        for proc in procs:
            proc.join()

        while not queue.empty():
            print(queue.get())

        circus_controller.close()
