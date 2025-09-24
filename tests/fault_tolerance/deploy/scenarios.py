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
from typing import Optional

from tests.utils.managed_deployment import DeploymentSpec


@dataclass
class Load:
    clients: int = 10
    requests_per_client: int = 150
    input_token_length: int = 100
    output_token_length: int = 100
    max_retries: int = 1
    max_request_rate: float = 1
    sla: Optional[float] = None


@dataclass
class Failure:
    time: int
    pod_name: str
    command: str
    signal: str = "SIGINT"
    replicas: int = 1


@dataclass
class Scenario:
    deployment: DeploymentSpec
    load: Load
    failures: list[Failure]
    model: Optional[str] = None


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

# TP-2 scenarios
deployment_specs["agg-tp-2-dp-1"] = DeploymentSpec(
    "/workspace/components/backends/vllm/deploy/agg.yaml"
)
deployment_specs["agg-tp-2-dp-1"].set_tensor_parallel(2, ["VllmDecodeWorker"])

deployment_specs["disagg-prefill-tp-2-decode-tp-2-dp-1"] = DeploymentSpec(
    "/workspace/components/backends/vllm/deploy/disagg.yaml"
)
deployment_specs["disagg-prefill-tp-2-decode-tp-2-dp-1"][
    "VllmPrefillWorker"
].tensor_parallel_size = 2
deployment_specs["disagg-prefill-tp-2-decode-tp-2-dp-1"][
    "VllmDecodeWorker"
].tensor_parallel_size = 2

# TP-4 scenarios
deployment_specs["agg-tp-4-dp-1"] = DeploymentSpec(
    "/workspace/components/backends/vllm/deploy/agg.yaml"
)
deployment_specs["agg-tp-4-dp-1"].set_tensor_parallel(4, ["VllmDecodeWorker"])

deployment_specs["disagg-prefill-tp-4-decode-tp-4-dp-1"] = DeploymentSpec(
    "/workspace/components/backends/vllm/deploy/disagg.yaml"
)
deployment_specs["disagg-prefill-tp-4-decode-tp-4-dp-1"][
    "VllmPrefillWorker"
].tensor_parallel_size = 4
deployment_specs["disagg-prefill-tp-4-decode-tp-4-dp-1"][
    "VllmDecodeWorker"
].tensor_parallel_size = 4

# Derivative Specs With Incremented Replicats

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


# Each failure scenaro contains a list of failure injections
# Each failure injection has a time in seconds after the pervious injection and
# a list of failures to inject including the number of failures for each type.
# Failures are currently process termination or pod deletion
#
# Example:
#
#   "prefill_worker": [[30, [("dynamo_prefillworker", 1)]]],
#
# terminates 1 prefill worker after 30 seconds

failures = {
    "frontend": [Failure(30, "Frontend", "dynamo.frontend")],
    "frontend_pod": [Failure(30, "Frontend", "delete_pod")],
    "decode_worker": [Failure(30, "VllmDecodeWorker", "dynamo.vllm", "SIGKILL")],
    "decode_worker_pod": [Failure(30, "VllmDecodeWorker", "delete_pod")],
    "prefill_worker": [Failure(30, "VllmPrefillWorker", "dynamo.vllm", "SIGKILL")],
    "prefill_worker_pod": [Failure(30, "VllmPrefillWorker", "delete_pod")],
    "vllm_decode_engine_core": [
        Failure(30, "VllmDecodeWorker", "VLLM::EngineCore", "SIGKILL")
    ],
    "vllm_prefill_engine_core": [
        Failure(30, "VllmPrefillWorker", "VLLM::EngineCore", "SIGKILL")
    ],
    "none": [],
}

load = Load()

# model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

model = None

# Populate Scenarios

scenarios = {}

for deployment_name, deployment_spec in deployment_specs.items():
    for failure_name, failure in failures.items():
        if "prefill" in failure_name and "disagg" not in deployment_name:
            continue
        scenarios[f"{deployment_name}-{failure_name}"] = Scenario(
            deployment=deployment_spec, load=load, failures=failure, model=model
        )
