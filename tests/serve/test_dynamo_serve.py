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
            "graphs.agg:Frontend",
            "configs/agg.yaml",
            "/workspace/examples/llm",
            "v1/chat/completions",
            completions_response_handler,
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "sglang_agg": (
        DeploymentGraph(
            "graphs.agg:Frontend",
            "configs/agg.yaml",
            "/workspace/examples/sglang",
            "v1/chat/completions",
            completions_response_handler,
            marks=[pytest.mark.gpu_1, pytest.mark.sglang],
        ),
        text_payload,
    ),
    "disagg": (
        DeploymentGraph(
            "graphs.disagg:Frontend",
            "configs/disagg.yaml",
            "/workspace/examples/llm",
            "v1/chat/completions",
            completions_response_handler,
            marks=[pytest.mark.gpu_2, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "agg_router": (
        DeploymentGraph(
            "graphs.agg_router:Frontend",
            "configs/agg_router.yaml",
            "/workspace/examples/llm",
            "v1/chat/completions",
            completions_response_handler,
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "disagg_router": (
        DeploymentGraph(
            "graphs.disagg_router:Frontend",
            "configs/disagg_router.yaml",
            "/workspace/examples/llm",
            "v1/chat/completions",
            completions_response_handler,
            marks=[pytest.mark.gpu_2, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "multimodal_agg": (
        DeploymentGraph(
            "graphs.agg:Frontend",
            "configs/agg.yaml",
            "/workspace/examples/multimodal",
            "v1/chat/completions",
            completions_response_handler,
            marks=[pytest.mark.gpu_2, pytest.mark.vllm],
        ),
        multimodal_payload,
    ),
}


class DynamoServeProcess(ManagedProcess):
    def __init__(self, graph: DeploymentGraph, request, port=8000, timeout=900):
        command = ["dynamo", "serve", graph.module]

        if graph.config:
            command.extend(["-f", os.path.join(graph.directory, graph.config)])

        command.extend(["--Frontend.port", str(port)])

        health_check_urls = [("http://localhost:8000/v1/models", self._check_model)]

        if "multimodal" in graph.directory:
            health_check_urls = []

        print(request)
        print(request.node.name)
        super().__init__(
            command=command,
            timeout=timeout,
            display_output=True,
            working_dir=graph.directory,
            health_check_ports=[8000],
            health_check_urls=health_check_urls,
            stragglers=["http"],
            log_dir=request.node.name,
        )

    def _check_model(self, response):
        data = response.json()
        if data.get("data") and len(data["data"]) > 0:
            return True
        return False


@pytest.fixture(
    params=[
        pytest.param("agg", marks=[pytest.mark.vllm, pytest.mark.gpu]),
        pytest.param("agg_router", marks=[pytest.mark.vllm, pytest.mark.gpu]),
        pytest.param("disagg", marks=[pytest.mark.vllm, pytest.mark.gpu_2]),
        pytest.param("disagg_router", marks=[pytest.mark.vllm, pytest.mark.gpu_2]),
        pytest.param("multimodal_agg", marks=[pytest.mark.vllm, pytest.mark.gpu_2]),
        pytest.param("sglang", marks=[pytest.mark.sglang, pytest.mark.gpu_2]),
    ]
)
def deployment_graph_test(request):
    """
    Fixture that provides different deployment graph test configurations.
    """
    global deployment_graphs
    return deployment_graphs[request.param]


@pytest.mark.e2e
@pytest.mark.slow
def test_serve_deployment(deployment_graph_test, request, runtime_services):
    """
    Test dynamo serve deployments with different graph configurations.
    """

    logging.getLogger("test_serve_deployment")
    logging.info("Starting test_deployment")

    deployment_graph, payload = deployment_graph_test

    with DynamoServeProcess(deployment_graph, request):
        url = f"http://localhost:8000/{deployment_graph.endpoint}"
        start_time = time.time()

        while time.time() - start_time < deployment_graph.timeout:
            try:
                response = requests.post(url, json=payload.payload, timeout=300)
            except Exception as e:
                # pytest.fail(f"Request failed: {str(e)}")
                print(e)
                time.sleep(5)
                continue
            logging.info(f"Response{response}")
            if (
                response.status_code == 500
                and "no instances" in response.json()["error"]
            ):
                time.sleep(5)
                continue

            # Process the response
            if response.status_code != 200:
                pytest.fail(
                    f"Service returned status code {response.status_code}: {response.text}"
                )
            else:
                break

        content = deployment_graph.response_handler(response)

        # Check for expected responses
        assert content, "Empty response content"
        for expected in payload.expected_response:
            assert expected in content, f"Expected '{expected}' not found in response"
