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

from dataclasses import dataclass

import pytest

from tests.utils.managed_deployment import DeploymentSpec

# Each Deployment Spec contains
# the dynamo deployment configuration

deployment_specs = {
    "agg-tp-1-dp-1": (
        DeploymentSpec("/workspace/components/backends/vllm/deploy/agg.yaml")
    ),
    "disagg-tp-1-dp-1": (
        DeploymentSpec("/workspace/components/backends/vllm/deploy/disagg.yaml")
    ),
}

deployment_specs["agg-tp-1-dp-2"] = DeploymentSpec(
    "/workspace/components/backends/vllm/deploy/agg.yaml"
)
deployment_specs["agg-tp-1-dp-2"]["Frontend"].replicas = 2
deployment_specs["agg-tp-1-dp-2"]["VllmDecodeWorker"].replicas = 2

deployment_specs["disagg-tp-1-dp-2"] = DeploymentSpec(
    "/workspace/components/backends/vllm/deploy/disagg.yaml"
)
deployment_specs["disagg-tp-1-dp-2"]["Frontend"].replicas = 2
deployment_specs["disagg-tp-1-dp-2"]["VllmDecodeWorker"].replicas = 2
deployment_specs["disagg-tp-1-dp-2"]["VllmPrefillWorker"].replicas = 2


@dataclass
class Failure:
    time: int
    pod_name: str
    command: str
    signal: str = "SIGINT"
    replicas: int = 1


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
    "frontend": [Failure(10, "Frontend", "dynamo.frontend")],
    "frontend_pod": [Failure(10, "Frontend", "delete_pod")],
    "decode_worker": [Failure(10, "VllmDecodeWorker", "dynamo.vllm")],
    "decode_worker_pod": [Failure(10, "VllmDecodeWorker", "delete_pod")],
    "prefill_worker": [Failure(10, "VllmPrefillWorker", "dynamo.vllm")],
    "prefill_worker_pod": [Failure(10, "VllmPrefillWorker", "delete_pod")],
    "vllm_decode_engine_core": [
        Failure(10, "VllmDecodeWorker", "VLLM::EngineCore", "SIGKILL")
    ],
    "vllm_prefill_engine_core": [
        Failure(10, "VllmPrefillWorker", "VLLM::EngineCore", "SIGKILL")
    ],
    "none": [],
}


@pytest.fixture(params=failure_scenarios.keys())
def failures(request):
    return failure_scenarios[request.param]


@pytest.fixture(params=list(deployment_specs.keys()))
def deployment_spec(request):
    """
    Fixture that provides different deployment graph test configurations.
    """
    return deployment_specs[request.param]
