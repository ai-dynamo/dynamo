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
    chat_completions_response_handler,
    completions_response_handler,
)
from tests.utils.managed_process import ManagedProcess

text_prompt = "Tell me a short joke about AI."

multimodal_payload = Payload(
    payload_chat={
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
    repeat_count=1,
    expected_log=[],
    expected_response=["bus"],
)

text_payload = Payload(
    payload_chat={
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "messages": [
            {
                "role": "user",
                "content": text_prompt,  # Shorter prompt
            }
        ],
        "max_tokens": 150,  # Reduced from 500
        "temperature": 0.1,
        # "seed": 0,
    },
    payload_completions={
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "prompt": text_prompt,
        "max_tokens": 150,
        "temperature": 0.1,
        # "seed": 0,
    },
    repeat_count=10,
    expected_log=[],
    expected_response=["AI"],
)

deployment_graphs = {
    "agg": (
        DeploymentGraph(
            module="graphs.agg:Frontend",
            config="configs/agg.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[
                chat_completions_response_handler,
            ],
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "sglang_agg": (
        DeploymentGraph(
            module="graphs.agg:Frontend",
            config="configs/agg.yaml",
            directory="/workspace/examples/sglang",
            endpoints=["v1/chat/completions", "v1/completions"],
            response_handlers=[
                chat_completions_response_handler,
                completions_response_handler,
            ],
            marks=[pytest.mark.gpu_1, pytest.mark.sglang],
        ),
        text_payload,
    ),
    "disagg": (
        DeploymentGraph(
            module="graphs.disagg:Frontend",
            config="configs/disagg.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[
                chat_completions_response_handler,
            ],
            marks=[pytest.mark.gpu_2, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "agg_router": (
        DeploymentGraph(
            module="graphs.agg_router:Frontend",
            config="configs/agg_router.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[
                chat_completions_response_handler,
            ],
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
            # FIXME: This is a hack to allow deployments to start before sending any requests.
            # When using KV-router, if all the endpoints are not registered, the service
            # enters a non-recoverable state.
            delayed_start=120,
        ),
        text_payload,
    ),
    "disagg_router": (
        DeploymentGraph(
            module="graphs.disagg_router:Frontend",
            config="configs/disagg_router.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[
                chat_completions_response_handler,
            ],
            marks=[pytest.mark.gpu_2, pytest.mark.vllm],
            # FIXME: This is a hack to allow deployments to start before sending any requests.
            # When using KV-router, if all the endpoints are not registered, the service
            # enters a non-recoverable state.
            delayed_start=120,
        ),
        text_payload,
    ),
    "multimodal_agg": (
        DeploymentGraph(
            module="graphs.agg:Frontend",
            config="configs/agg-llava.yaml",
            directory="/workspace/examples/multimodal",
            endpoints=["v1/chat/completions"],
            response_handlers=[
                chat_completions_response_handler,
            ],
            marks=[pytest.mark.gpu_2, pytest.mark.vllm],
        ),
        multimodal_payload,
    ),
    "vllm_v1_agg": (
        DeploymentGraph(
            module="graphs.agg:Frontend",
            config="configs/agg.yaml",
            directory="/workspace/examples/vllm_v1",
            endpoints=["v1/chat/completions", "v1/completions"],
            response_handlers=[
                chat_completions_response_handler,
                completions_response_handler,
            ],
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "trtllm_agg": (
        DeploymentGraph(
            module="graphs.agg:Frontend",
            config="configs/agg.yaml",
            directory="/workspace/examples/tensorrt_llm",
            endpoints=["v1/chat/completions", "v1/completions"],
            response_handlers=[
                chat_completions_response_handler,
                completions_response_handler,
            ],
            marks=[pytest.mark.gpu_1, pytest.mark.tensorrtllm],
        ),
        text_payload,
    ),
    "trtllm_agg_router": (
        DeploymentGraph(
            module="graphs.agg:Frontend",
            config="configs/agg_router.yaml",
            directory="/workspace/examples/tensorrt_llm",
            endpoints=["v1/chat/completions", "v1/completions"],
            response_handlers=[
                chat_completions_response_handler,
                completions_response_handler,
            ],
            marks=[pytest.mark.gpu_1, pytest.mark.tensorrtllm],
            # FIXME: This is a hack to allow deployments to start before sending any requests.
            # When using KV-router, if all the endpoints are not registered, the service
            # enters a non-recoverable state.
            delayed_start=120,
        ),
        text_payload,
    ),
    "trtllm_disagg": (
        DeploymentGraph(
            module="graphs.disagg:Frontend",
            config="configs/disagg.yaml",
            directory="/workspace/examples/tensorrt_llm",
            endpoints=["v1/chat/completions", "v1/completions"],
            response_handlers=[
                chat_completions_response_handler,
                completions_response_handler,
            ],
            marks=[pytest.mark.gpu_2, pytest.mark.tensorrtllm],
        ),
        text_payload,
    ),
    "trtllm_disagg_router": (
        DeploymentGraph(
            module="graphs.disagg:Frontend",
            config="configs/disagg_router.yaml",
            directory="/workspace/examples/tensorrt_llm",
            endpoints=["v1/chat/completions", "v1/completions"],
            response_handlers=[
                chat_completions_response_handler,
                completions_response_handler,
            ],
            marks=[pytest.mark.gpu_2, pytest.mark.tensorrtllm],
            # FIXME: This is a hack to allow deployments to start before sending any requests.
            # When using KV-router, if all the endpoints are not registered, the service
            # enters a non-recoverable state.
            delayed_start=120,
        ),
        text_payload,
    ),
}


class DynamoServeProcess(ManagedProcess):
    def __init__(
        self,
        graph: DeploymentGraph,
        request,
        port=8000,
        timeout=900,
        display_output=True,
        args=None,
    ):
        command = ["dynamo", "serve", graph.module]

        if graph.config:
            command.extend(["-f", os.path.join(graph.directory, graph.config)])

        command.extend(["--Frontend.port", str(port)])

        if args:
            for k, v in args.items():
                command.extend([f"{k}", f"{v}"])

        health_check_urls = []
        health_check_ports = []

        # Handle multimodal deployments differently
        if "multimodal" in graph.directory:
            # Set DYNAMO_PORT environment variable for multimodal
            env = os.environ.copy()
            env["DYNAMO_PORT"] = str(port)
            # Don't add health check on port since multimodal uses DYNAMO_PORT
        else:
            # Regular LLM deployments
            command.extend(["--Frontend.port", str(port)])
            health_check_urls = [
                (f"http://localhost:{port}/v1/models", self._check_model)
            ]
            health_check_ports = [port]
            env = None

        self.port = port
        self.graph = graph

        super().__init__(
            command=command,
            timeout=timeout,
            display_output=display_output,
            working_dir=graph.directory,
            health_check_ports=health_check_ports,
            health_check_urls=health_check_urls,
            delayed_start=graph.delayed_start,
            stragglers=["http"],
            straggler_commands=[
                "dynamo.sdk.cli.serve_dynamo",
                "from multiprocessing.resource_tracker",
                "from multiprocessing.spawn",
            ],
            log_dir=request.node.name,
            env=env,  # Pass the environment variables
        )

    def _check_model(self, response):
        try:
            data = response.json()
        except ValueError:
            return False
        if data.get("data") and len(data["data"]) > 0:
            return True
        return False

    def check_response(
        self, payload, response, response_handler, logger=logging.getLogger()
    ):
        assert response.status_code == 200, "Response Error"
        content = response_handler(response)
        logger.info("Received Content: %s", content)
        # Check for expected responses
        assert content, "Empty response content"
        for expected in payload.expected_response:
            assert expected in content, "Expected '%s' not found in response" % expected

    def wait_for_ready(self, payload, logger=logging.getLogger()):
        url = f"http://localhost:{self.port}/{self.graph.endpoints[0]}"
        start_time = time.time()
        retry_delay = 5
        elapsed = 0.0
        logger.info("Waiting for Deployment Ready")
        json_payload = (
            payload.payload_chat
            if self.graph.endpoints[0] == "v1/chat/completions"
            else payload.payload_completions
        )

        while time.time() - start_time < self.graph.timeout:
            elapsed = time.time() - start_time
            try:
                response = requests.post(
                    url,
                    json=json_payload,
                    timeout=self.graph.timeout - elapsed,
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
                self.graph.timeout,
            )
            pytest.fail(
                "Service did not return a successful response within %s s"
                % self.graph.timeout
            )

        self.check_response(payload, response, self.graph.response_handlers[0], logger)

        logger.info("Deployment Ready")


@pytest.fixture(
    params=[
        pytest.param("agg", marks=[pytest.mark.vllm, pytest.mark.gpu_1]),
        pytest.param("agg_router", marks=[pytest.mark.vllm, pytest.mark.gpu_1]),
        pytest.param("disagg", marks=[pytest.mark.vllm, pytest.mark.gpu_2]),
        pytest.param("disagg_router", marks=[pytest.mark.vllm, pytest.mark.gpu_2]),
        pytest.param("multimodal_agg", marks=[pytest.mark.vllm, pytest.mark.gpu_2]),
        pytest.param("trtllm_agg", marks=[pytest.mark.tensorrtllm, pytest.mark.gpu_1]),
        pytest.param(
            "trtllm_agg_router", marks=[pytest.mark.tensorrtllm, pytest.mark.gpu_1]
        ),
        pytest.param(
            "trtllm_disagg", marks=[pytest.mark.tensorrtllm, pytest.mark.gpu_2]
        ),
        pytest.param(
            "trtllm_disagg_router", marks=[pytest.mark.tensorrtllm, pytest.mark.gpu_2]
        ),
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
        server_process.wait_for_ready(payload, logger)

        for endpoint, response_handler in zip(
            deployment_graph.endpoints, deployment_graph.response_handlers
        ):
            url = f"http://localhost:{server_process.port}/{endpoint}"
            start_time = time.time()
            elapsed = 0.0

            request_body = (
                payload.payload_chat
                if endpoint == "v1/chat/completions"
                else payload.payload_completions
            )

            for _ in range(payload.repeat_count):
                elapsed = time.time() - start_time

                response = requests.post(
                    url,
                    json=request_body,
                    timeout=deployment_graph.timeout - elapsed,
                )
                server_process.check_response(
                    payload, response, response_handler, logger
                )
