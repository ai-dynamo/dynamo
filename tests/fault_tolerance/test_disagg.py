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


import asyncio
import logging
import random
import time

import psutil
import pytest
import requests

from dynamo.llm import KvMetricsAggregator
from dynamo.runtime import dynamo_worker
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
        "ignore_eos": "1",
        "min_tokens": 150,
    },
    expected_log=[],
    expected_response=["AI"],
)

deployment_graphs = {
    "disagg": (
        DeploymentGraph(
            module="graphs.mock_disagg:Frontend",
            config="configs/mock_disagg.yaml",
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
        pytest.param("disagg", marks=[pytest.mark.vllm, pytest.mark.gpu_1]),
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


def _get_random_prompt(length):
    word_list = [f"{i}" for i in range(10)]
    return " ".join(random.choices(word_list, k=length))


def _single_request(
    url,
    payload,
    logger,
    retry_attempts=1,
    input_token_length=100,
    output_token_length=100,
    timeout=60,
):
    prompt = _get_random_prompt(input_token_length)
    payload["messages"][0]["content"] = prompt
    payload["max_tokens"] = output_token_length
    retry_delay = 1
    response = None
    end_time = None
    print(prompt)
    start_time = time.time()

    # How many attempts it took
    # How much time it took total
    # How much time the request took
    # Results
    results = []

    while retry_attempts:
        start_request_time = time.time()
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
            )
            end_time = time.time()

            results.append((response, end_time - start_request_time))

            if response.status_code != 200:
                time.sleep(retry_delay)
                retry_attempts -= 1
                continue
            else:
                break

        except (requests.RequestException, requests.Timeout) as e:
            results.append((e, end_time - start_request_time))
            logger.warning("Retrying due to Request failed: %s", e)
            time.sleep(retry_delay)
            retry_attempts -= 1
            continue

    return results, time.time() - start_time


def client(
    deployment_graph, server_process, payload, queue, index, requests_per_client=5
):
    logger = logging.getLogger(f"CLIENT: {index}")

    url = f"http://localhost:{server_process.port}/{deployment_graph.endpoint}"
    start_time = time.time()
    retry_delay = 5
    elapsed = 0.0
    results = []

    for _ in range(requests_per_client):
        results.append(_single_request(url, payload.payload, logger))

    queue.put((index, results))


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


def run_metrics_process():
    asyncio.run(get_metrics())


@dynamo_worker()
async def get_metrics(runtime):
    # Log # processes
    # Log # metrics per vllm worker

    kv_listener = runtime.namespace("dynamo").component("VllmWorker")
    await kv_listener.create_service()

    logging.info("HERE!")

    pipeline = (
        await runtime.namespace("dynamo")
        .component("VllmWorker")
        .endpoint("load_metrics")
        .client()
    )

    while True:
        try:
            await asyncio.sleep(1)

            logging.info("HERE 22222!")

            print(pipeline.instance_ids(), flush=True)

            logging.info("HERE eeeee!")
            async for char in await pipeline.generate(None):
                print("got value!!!!!")
                print(char)

            metrics_aggregator = KvMetricsAggregator(kv_listener)

            logging.info("HERE!")

            endpoints = await metrics_aggregator.get_metrics()
            logging.info("HERE!")

            print(endpoints.endpoints)

            for endpoint in endpoints.endpoints:
                logging.info("Worker ID: %s", endpoint.worker_id)
                logging.info("GPU Cache Usage: %s", endpoint.gpu_cache_usage_perc)
                print("Number of Requests Waiting: ", endpoint.num_requests_waiting)
                print("GPU Prefix Cache Hit Rate: ", endpoint.gpu_prefix_cache_hit_rate)
                print("***")
        except:
            pass


@pytest.fixture
def num_clients(request):
    return request.config.getoption("--clients")


@pytest.fixture
def requests_per_client(request):
    return request.config.getoption("--requests-per-client")


@pytest.mark.e2e
@pytest.mark.slow
async def test_worker_failure(
    deployment_graph_test, request, runtime_services, num_clients, requests_per_client
):
    """
    Test dynamo serve deployments with different graph configurations.
    """

    # runtime_services is used to start nats and etcd

    logger = logging.getLogger(request.node.name)
    logger.info("Starting test_deployment")
    deployment_graph, payload = deployment_graph_test

    with DynamoServeProcess(
        deployment_graph, request, args={"--MockVllmWorker.max_num_seqs": "2"}
    ) as server_process:
        _wait_until_ready(deployment_graph, server_process, payload)
