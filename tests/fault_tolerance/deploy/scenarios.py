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
    backend: str = "vllm"  # Backend type for tracking


# Helper functions to create deployment specs
def _create_deployment_spec(backend, deploy_type, yaml_path):
    """Create a deployment spec with backend information."""
    return {"spec": DeploymentSpec(yaml_path), "backend": backend}


def _set_replicas(deployment_spec, backend, deploy_type, replicas):
    """Set replicas for all components in a deployment based on backend type."""
    spec = deployment_spec["spec"]

    # Frontend is common for all backends
    spec["Frontend"].replicas = replicas

    if backend == "vllm":
        if deploy_type == "agg":
            spec["VllmDecodeWorker"].replicas = replicas
        elif deploy_type == "disagg":
            spec["VllmDecodeWorker"].replicas = replicas
            spec["VllmPrefillWorker"].replicas = replicas
    elif backend == "sglang":
        if deploy_type == "agg":
            spec["decode"].replicas = replicas
        elif deploy_type == "disagg":
            spec["decode"].replicas = replicas
            spec["prefill"].replicas = replicas


def _set_tensor_parallel(deployment_spec, backend, deploy_type, tp_size):
    """Set tensor parallel size for worker components."""
    spec = deployment_spec["spec"]

    if backend == "vllm":
        if deploy_type == "agg":
            spec.set_tensor_parallel(tp_size, ["VllmDecodeWorker"])
        elif deploy_type == "disagg":
            spec["VllmPrefillWorker"].tensor_parallel_size = tp_size
            spec["VllmDecodeWorker"].tensor_parallel_size = tp_size
    elif backend == "sglang":
        if deploy_type == "agg":
            # SGLang might use different method, adjust as needed
            if hasattr(spec, "set_tensor_parallel"):
                spec.set_tensor_parallel(tp_size, ["decode"])
            else:
                spec["decode"].tensor_parallel_size = tp_size
        elif deploy_type == "disagg":
            spec["prefill"].tensor_parallel_size = tp_size
            spec["decode"].tensor_parallel_size = tp_size


def _create_vllm_deployments():
    """Create all vLLM deployment specifications."""
    vllm_deployments = {}

    # Base deployments
    vllm_deployments["vllm-agg-tp-1-dp-1"] = _create_deployment_spec(
        "vllm", "agg", "components/backends/vllm/deploy/agg.yaml"
    )
    vllm_deployments["vllm-disagg-tp-1-dp-1"] = _create_deployment_spec(
        "vllm", "disagg", "components/backends/vllm/deploy/disagg.yaml"
    )

    # TP-2 scenarios
    vllm_deployments["vllm-agg-tp-2-dp-1"] = _create_deployment_spec(
        "vllm", "agg", "components/backends/vllm/deploy/agg.yaml"
    )
    _set_tensor_parallel(vllm_deployments["vllm-agg-tp-2-dp-1"], "vllm", "agg", 2)

    vllm_deployments[
        "vllm-disagg-prefill-tp-2-decode-tp-2-dp-1"
    ] = _create_deployment_spec(
        "vllm", "disagg", "components/backends/vllm/deploy/disagg.yaml"
    )
    _set_tensor_parallel(
        vllm_deployments["vllm-disagg-prefill-tp-2-decode-tp-2-dp-1"],
        "vllm",
        "disagg",
        2,
    )

    # TP-4 scenarios
    vllm_deployments["vllm-agg-tp-4-dp-1"] = _create_deployment_spec(
        "vllm", "agg", "components/backends/vllm/deploy/agg.yaml"
    )
    _set_tensor_parallel(vllm_deployments["vllm-agg-tp-4-dp-1"], "vllm", "agg", 4)

    vllm_deployments[
        "vllm-disagg-prefill-tp-4-decode-tp-4-dp-1"
    ] = _create_deployment_spec(
        "vllm", "disagg", "components/backends/vllm/deploy/disagg.yaml"
    )
    _set_tensor_parallel(
        vllm_deployments["vllm-disagg-prefill-tp-4-decode-tp-4-dp-1"],
        "vllm",
        "disagg",
        4,
    )

    # DP-2 scenarios (increased replicas)
    vllm_deployments["vllm-agg-tp-1-dp-2"] = _create_deployment_spec(
        "vllm", "agg", "components/backends/vllm/deploy/agg.yaml"
    )
    _set_replicas(vllm_deployments["vllm-agg-tp-1-dp-2"], "vllm", "agg", 2)

    vllm_deployments["vllm-disagg-tp-1-dp-2"] = _create_deployment_spec(
        "vllm", "disagg", "components/backends/vllm/deploy/disagg.yaml"
    )
    _set_replicas(vllm_deployments["vllm-disagg-tp-1-dp-2"], "vllm", "disagg", 2)

    return vllm_deployments


def _create_sglang_deployments():
    """Create all SGLang deployment specifications."""
    sglang_deployments = {}

    # Base deployments
    sglang_deployments["sglang-agg-tp-1-dp-1"] = _create_deployment_spec(
        "sglang", "agg", "components/backends/sglang/deploy/agg-debug.yaml"
    )
    sglang_deployments["sglang-disagg-tp-1-dp-1"] = _create_deployment_spec(
        "sglang", "disagg", "components/backends/sglang/deploy/disagg.yaml"
    )

    # DP-2 scenarios (increased replicas)
    sglang_deployments["sglang-agg-tp-1-dp-2"] = _create_deployment_spec(
        "sglang", "agg", "components/backends/sglang/deploy/agg.yaml"
    )
    _set_replicas(sglang_deployments["sglang-agg-tp-1-dp-2"], "sglang", "agg", 2)

    sglang_deployments["sglang-disagg-tp-1-dp-2"] = _create_deployment_spec(
        "sglang", "disagg", "components/backends/sglang/deploy/disagg.yaml"
    )
    _set_replicas(sglang_deployments["sglang-disagg-tp-1-dp-2"], "sglang", "disagg", 2)

    return sglang_deployments


# Create all deployment specifications
deployment_specs = {}
deployment_specs.update(_create_vllm_deployments())
deployment_specs.update(_create_sglang_deployments())


# Each failure scenaro contains a list of failure injections
# Each failure injection has a time in seconds after the pervious injection and
# a list of failures to inject including the number of failures for each type.
# Failures are currently process termination or pod deletion
#
# Example:
#
#   "prefill_worker": [Failure(30, "VllmPrefillWorker", "dynamo.vllm", "SIGKILL")],
#
# terminates 1 prefill worker after 30 seconds

# vLLM-specific failures
vllm_failures = {
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

# SGLang-specific failures
sglang_failures = {
    "frontend": [Failure(30, "Frontend", "dynamo.frontend")],
    "frontend_pod": [Failure(30, "Frontend", "delete_pod")],
    "decode_worker": [Failure(30, "decode", "dynamo.sglang", "SIGKILL")],
    "decode_worker_pod": [Failure(30, "decode", "delete_pod")],
    "prefill_worker": [Failure(30, "prefill", "dynamo.sglang", "SIGKILL")],
    "prefill_worker_pod": [Failure(30, "prefill", "delete_pod")],
    "sglang_decode_scheduler": [Failure(30, "decode", "sglang::scheduler", "SIGKILL")],
    "sglang_decode_detokenizer": [
        Failure(30, "decode", "sglang::detokenizer", "SIGKILL")
    ],
    "sglang_prefill_scheduler": [
        Failure(30, "prefill", "sglang::scheduler", "SIGKILL")
    ],
    "sglang_prefill_detokenizer": [
        Failure(30, "prefill", "sglang::detokenizer", "SIGKILL")
    ],
    "none": [],
}

load = Load()

# model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

model = None

# Populate Scenarios

scenarios = {}

# Map of backend to failure definitions
backend_failure_map = {
    "vllm": vllm_failures,
    "sglang": sglang_failures,
}

for deployment_name, deployment_info in deployment_specs.items():
    backend = deployment_info["backend"]

    # Validate backend
    if backend not in backend_failure_map:
        raise ValueError(
            f"Unsupported backend: {backend}. Supported backends are: {list(backend_failure_map.keys())}"
        )

    # Get the appropriate failure set for this backend
    failure_set = backend_failure_map[backend]

    for failure_name, failure in failure_set.items():
        # Skip prefill failures for aggregated deployments
        if "prefill" in failure_name and "disagg" not in deployment_name:
            continue

        scenario_name = f"{deployment_name}-{failure_name}"
        scenarios[scenario_name] = Scenario(
            deployment=deployment_info["spec"],
            load=load,
            failures=failure,
            model=model,
            backend=backend,
        )
