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

import pytest

from tests.utils.deployment_graph import (
    DeploymentGraph,
    Payload,
    chat_completions_response_handler,
)

# Initial payload used for testing
# initial deployment readiness.

text_prompt = "Tell me a short joke about AI."

text_payload = Payload(
    payload_chat={
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "messages": [
            {
                "role": "user",
                "content": text_prompt,  # Shorter prompt
            }
        ],
        "max_tokens": 150,
        "temperature": 0.1,
        #        "seed": 10,
        "ignore_eos": True,
        "min_tokens": 150,
        "stream": False,
    },
    expected_log=[],
    expected_response=["AI"],
)

# Each Deployment Graph contains
# the dynamo serve module and configuration as well
# as the endpoint for interaction

deployment_graphs = {
    "agg-tp-1-dp-1": (
        DeploymentGraph(
            module="graphs.agg:Frontend",
            config="/workspace/tests/fault_tolerance/configs/agg_tp_1_dp_1.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[chat_completions_response_handler],
            marks=[pytest.mark.gpu_1, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "agg-tp-1-dp-8": (
        DeploymentGraph(
            module="graphs.agg:Frontend",
            config="/workspace/tests/fault_tolerance/configs/agg_tp_1_dp_8.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[chat_completions_response_handler],
            marks=[pytest.mark.gpu_8, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "agg-tp-2-dp-4": (
        DeploymentGraph(
            module="graphs.agg:Frontend",
            config="/workspace/tests/fault_tolerance/configs/agg_tp_2_dp_4.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[chat_completions_response_handler],
            marks=[pytest.mark.gpu_8, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "disagg-p-tp-1-dp-1-d-tp-1-dp-1": (
        DeploymentGraph(
            module="graphs.disagg:Frontend",
            config="/workspace/tests/fault_tolerance/configs/disagg_p_tp_1_dp_1_d_tp_1_dp_1.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[chat_completions_response_handler],
            marks=[pytest.mark.gpu_2, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "disagg-p-tp-1-dp-4-d-tp-4-dp-1": (
        DeploymentGraph(
            module="graphs.disagg:Frontend",
            config="/workspace/tests/fault_tolerance/configs/disagg_p_tp_1_dp_4_d_tp_4_dp_1.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[chat_completions_response_handler],
            marks=[pytest.mark.gpu_8, pytest.mark.vllm],
        ),
        text_payload,
    ),
    "disagg-p-tp-2-dp-2-d-tp-4-dp-1": (
        DeploymentGraph(
            module="graphs.disagg:Frontend",
            config="/workspace/tests/fault_tolerance/configs/disagg_p_tp_2_dp_2_d_tp_4_dp_1.yaml",
            directory="/workspace/examples/llm",
            endpoints=["v1/chat/completions"],
            response_handlers=[chat_completions_response_handler],
            marks=[pytest.mark.gpu_8, pytest.mark.vllm],
        ),
        text_payload,
    ),
}

# Each failure scenaro contains a list of failure injections
# Each failure injection has a time in seconds after the pervious injection and
# a list of failures to inject including the number of failures for each type.
# Failures are currently process termination.
#
# Example:
#
#   "prefill_worker": [[30, [("dynamo_prefillworker", 1)]]],
#
# terminates 1 prefill worker after 30 seconds

failure_scenarios = {
    "decode_worker": [[30, [("dynamo_vllmworker", 1)]]],
    "prefill_worker": [[30, [("dynamo_prefillworker", 1)]]],
    "frontend": [[30, [("dynamo_frontend", 1)]]],
    "processor": [[30, [("dynamo_processor", 1)]]],
    "vllm_worker": [[30, [("vllm_worker", 1)]]],
    "none": [],
}


@pytest.fixture(params=list(failure_scenarios.keys()))
def failures(request):
    return failure_scenarios[request.param]


@pytest.fixture(params=list(deployment_graphs.keys()))
def deployment_graph_test(request):
    """
    Fixture that provides different deployment graph test configurations.
    """
    return deployment_graphs[request.param]
