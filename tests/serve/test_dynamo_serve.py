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
import os
import time

import pytest
import requests

from tests.utils.deployment_graph import (
    DeploymentGraph,
    Payload,
    completions_response_handler,
)
from tests.utils.managed_process import ManagedProcess

text_prompt = "Tell me a short joke about AI."

multimodal_payload = Payload(
    payload={
        "model": "llava-hf/llava-1.5-7b-hf",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "http://images.cocodataset.org/test2017/000000155781.jpg"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 300,  # Reduced from 500
        "stream": False,
    },
    expected_log=[],
    expected_response=["bus"],
)

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
            module="graphs.agg:Frontend",
            config="configs/agg.yaml",
            directory="/workspace/examples/llm",
            endpoint="v1/chat/completions",
            response_handler=completions_response_handler,
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "sglang_agg": (
        DeploymentGraph(
            module="graphs.agg:Frontend",
            config="configs/agg.yaml",
            directory="/workspace/examples/sglang",
            endpoint="v1/chat/completions",
            response_handler=completions_response_handler,
            marks=[pytest.mark.gpu_1, pytest.mark.sglang],
        ),
        text_payload,
    ),
    "disagg": (
        DeploymentGraph(
            module="graphs.disagg:Frontend",
            config="configs/disagg.yaml",
            directory="/workspace/examples/llm",
            endpoint="v1/chat/completions",
            response_handler=completions_response_handler,
            marks=[pytest.mark.gpu_2, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "agg_router": (
        DeploymentGraph(
            module="graphs.agg_router:Frontend",
            config="configs/agg_router.yaml",
            directory="/workspace/examples/llm",
            endpoint="v1/chat/completions",
            response_handler=completions_response_handler,
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "disagg_router": (
        DeploymentGraph(
            module="graphs.disagg_router:Frontend",
            config="configs/disagg_router.yaml",
            directory="/workspace/examples/llm",
            endpoint="v1/chat/completions",
            response_handler=completions_response_handler,
            marks=[pytest.mark.gpu_2, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "multimodal_agg": (
        DeploymentGraph(
            module="graphs.agg:Frontend",
            config="configs/agg.yaml",
            directory="/workspace/examples/multimodal",
            endpoint="v1/chat/completions",
            response_handler=completions_response_handler,
            marks=[pytest.mark.gpu_2, pytest.mark.vllm],
        ),
        multimodal_payload,
    ),
    "vllm_v1_agg": (
        DeploymentGraph(
            module="graphs.agg:Frontend",
            config="configs/agg.yaml",
            directory="/workspace/examples/vllm_v1",
            endpoint="v1/chat/completions",
            response_handler=completions_response_handler,
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
}


class DynamoServeProcess(ManagedProcess):
    def __init__(self, graph: DeploymentGraph, request, port=8000, timeout=900):
        command = ["dynamo", "serve", graph.module]

        if graph.config:
            command.extend(["-f", os.path.join(graph.directory, graph.config)])

        command.extend(["--Frontend.port", str(port)])

        health_check_urls = [(f"http://localhost:{port}/v1/models", self._check_model)]

        if "multimodal" in graph.directory:
            health_check_urls = []

        self.port = port

        super().__init__(
            command=command,
            timeout=timeout,
            display_output=True,
            working_dir=graph.directory,
            health_check_ports=[port],
            health_check_urls=health_check_urls,
            stragglers=["http"],
            log_dir=request.node.name,
        )

    def _check_model(self, response):
        try:
            data = response.json()
        except ValueError:
            return False
        if data.get("data") and len(data["data"]) > 0:
            return True
        return False


@pytest.fixture(
    params=[
        pytest.param("agg", marks=[pytest.mark.vllm, pytest.mark.gpu_1]),
        pytest.param("agg_router", marks=[pytest.mark.vllm, pytest.mark.gpu_1]),
        pytest.param("disagg", marks=[pytest.mark.vllm, pytest.mark.gpu_2]),
        pytest.param("disagg_router", marks=[pytest.mark.vllm, pytest.mark.gpu_2]),
        pytest.param("multimodal_agg", marks=[pytest.mark.vllm, pytest.mark.gpu_2]),
        #        pytest.param("sglang", marks=[pytest.mark.sglang, pytest.mark.gpu_2]),
    ]
)
def deployment_graph_test(request):
    """
    Fixture that provides different deployment graph test configurations.
    """
    return deployment_graphs[request.param]


@pytest.mark.e2e
@pytest.mark.slow
def test_serve_deployment(deployment_graph_test, request, runtime_services):
    """
    Test dynamo serve deployments with different graph configurations.
    """

    # runtime_services is used to start nats and etcd

    logger = logging.getLogger(request.node.name)
    logger.info("Starting test_deployment")

    deployment_graph, payload = deployment_graph_test

    with DynamoServeProcess(deployment_graph, request) as server_process:
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
