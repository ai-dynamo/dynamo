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

import re
from dataclasses import dataclass
from typing import Dict, Optional, Pattern

from tests.utils.managed_deployment import DeploymentSpec

# Worker name mapping for different backends
WORKER_MAP = {
    "vllm": {
        "decode": "VllmDecodeWorker",
        "prefill": "VllmPrefillWorker",
    },
    "sglang": {
        "decode": "decode",
        "prefill": "prefill",
    },
    "trtllm": {
        "decode": "TRTLLMDecodeWorker",
        "decode_agg": "TRTLLMWorker",  # Aggregated uses different name
        "prefill": "TRTLLMPrefillWorker",
    },
}

# Process ready patterns for recovery detection
WORKER_READY_PATTERNS: Dict[str, Pattern] = {
    # Frontend
    "Frontend": re.compile(r"added model"),
    # vLLM workers
    "VllmDecodeWorker": re.compile(
        r"VllmWorker for (?P<model_name>.*?) has been initialized"
    ),
    "VllmPrefillWorker": re.compile(
        r"VllmWorker for (?P<model_name>.*?) has been initialized"
    ),
    # SGLang workers - look for their specific initialization messages
    "decode": re.compile(
        r"Model registration succeeded|Decode worker handler initialized|Worker handler initialized"
    ),
    "prefill": re.compile(
        r"Model registration succeeded|Prefill worker handler initialized|Worker handler initialized"
    ),
    # TensorRT-LLM workers
    "TRTLLMWorker": re.compile(
        r"TrtllmWorker for (?P<model_name>.*?) has been initialized|Model registration succeeded"
    ),
    "TRTLLMDecodeWorker": re.compile(
        r"TrtllmWorker for (?P<model_name>.*?) has been initialized|Model registration succeeded"
    ),
    "TRTLLMPrefillWorker": re.compile(
        r"TrtllmWorker for (?P<model_name>.*?) has been initialized|Model registration succeeded"
    ),
}


def get_all_worker_types() -> list[str]:
    """Get all worker type names for both vLLM and SGLang."""
    worker_types = ["Frontend"]
    for backend in WORKER_MAP.values():
        worker_types.extend(backend.values())
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for x in worker_types:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def get_worker_ready_pattern(worker_name: str) -> Optional[Pattern]:
    """Get the ready pattern for a specific worker type."""
    return WORKER_READY_PATTERNS.get(worker_name)


def get_backend_workers(backend: str) -> Dict[str, str]:
    """Get worker mapping for a specific backend."""
    return WORKER_MAP.get(backend, {})


@dataclass
class Load:
    clients: int = 10
    requests_per_client: int = 150
    input_token_length: int = 100
    output_token_length: int = 100
    max_retries: int = 3  # Increased for fault tolerance
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

    if backend in WORKER_MAP:
        # For trtllm agg deployments, use different worker name
        if backend == "trtllm" and deploy_type == "agg":
            decode_worker = WORKER_MAP[backend]["decode_agg"]
        else:
            decode_worker = WORKER_MAP[backend]["decode"]

        # always scale decode
        spec[decode_worker].replicas = replicas
        # scale prefill only for disagg
        if deploy_type == "disagg":
            spec[WORKER_MAP[backend]["prefill"]].replicas = replicas


def _set_tensor_parallel(deployment_spec, backend, deploy_type, tp_size):
    """Set tensor parallel size for worker components."""
    spec = deployment_spec["spec"]

    if backend in WORKER_MAP:
        # For trtllm agg deployments, use different worker name
        if backend == "trtllm" and deploy_type == "agg":
            decode_worker = WORKER_MAP[backend]["decode_agg"]
        else:
            decode_worker = WORKER_MAP[backend]["decode"]
        prefill_worker = WORKER_MAP[backend]["prefill"]

        if deploy_type == "agg":
            if hasattr(spec, "set_tensor_parallel"):
                spec.set_tensor_parallel(tp_size, [decode_worker])
            else:
                spec[decode_worker].tensor_parallel_size = tp_size
        elif deploy_type == "disagg":
            spec[prefill_worker].tensor_parallel_size = tp_size
            spec[decode_worker].tensor_parallel_size = tp_size


def _create_deployments_for_backend(backend):
    """Create all deployment specifications for a given backend."""
    deployments = {}

    # Define the yaml files for agg and disagg deployments
    yaml_files = {
        "agg": f"components/backends/{backend}/deploy/agg.yaml",
        "disagg": f"components/backends/{backend}/deploy/disagg.yaml",
    }

    # Define the different configurations to test
    configurations = [
        {"tp": 1, "dp": 1},
        {"tp": 1, "dp": 2},
        {"tp": 2, "dp": 1},
        {"tp": 4, "dp": 1},
    ]

    for deploy_type in ["agg", "disagg"]:
        for config in configurations:
            tp_size = config["tp"]
            dp_replicas = config["dp"]
            # Skip creating disagg scenarios for TP > 1 if DP is also > 1 (uncommon case)
            if deploy_type == "disagg" and tp_size > 1 and dp_replicas > 1:
                continue

            # Construct the scenario name
            name_parts = [backend, deploy_type]

            if deploy_type == "agg":
                name_parts.append(f"tp-{tp_size}")
            elif deploy_type == "disagg":
                name_parts.append(f"prefill-tp-{tp_size}-decode-tp-{tp_size}")

            name_parts.append(f"dp-{dp_replicas}")

            scenario_name = "-".join(name_parts)

            # Create and configure the deployment
            deployment = _create_deployment_spec(
                backend, deploy_type, yaml_files[deploy_type]
            )
            if tp_size > 1:
                _set_tensor_parallel(deployment, backend, deploy_type, tp_size)
            if dp_replicas > 1:
                _set_replicas(deployment, backend, deploy_type, dp_replicas)

            deployments[scenario_name] = deployment

    return deployments


# Create all deployment specifications
deployment_specs = {}
deployment_specs.update(_create_deployments_for_backend("vllm"))
deployment_specs.update(_create_deployments_for_backend("sglang"))
deployment_specs.update(_create_deployments_for_backend("trtllm"))


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
def _create_backend_failures(backend, deploy_type="disagg"):
    """Generate backend-specific failure scenarios.

    Args:
        backend: Backend type (vllm, sglang, trtllm)
        deploy_type: Deployment type (agg or disagg)
    """
    workers = WORKER_MAP[backend]

    # Use correct worker name based on deployment type
    if backend == "trtllm" and deploy_type == "agg":
        decode_worker = workers["decode_agg"]
    else:
        decode_worker = workers["decode"]

    prefill_worker = workers["prefill"]
    process_name = f"dynamo.{backend}"

    failures = {
        "frontend": [Failure(30, "Frontend", "dynamo.frontend")],
        "frontend_pod": [Failure(30, "Frontend", "delete_pod")],
        "decode_worker": [Failure(30, decode_worker, process_name, "SIGKILL")],
        "decode_worker_pod": [Failure(30, decode_worker, "delete_pod")],
        "prefill_worker": [Failure(30, prefill_worker, process_name, "SIGKILL")],
        "prefill_worker_pod": [Failure(30, prefill_worker, "delete_pod")],
        "none": [],
    }

    if backend == "vllm":
        failures["vllm_decode_engine_core"] = [
            Failure(30, decode_worker, "VLLM::EngineCore", "SIGKILL")
        ]
        failures["vllm_prefill_engine_core"] = [
            Failure(30, prefill_worker, "VLLM::EngineCore", "SIGKILL")
        ]
    elif backend == "sglang":
        failures["sglang_decode_scheduler"] = [
            Failure(30, decode_worker, "sglang::scheduler", "SIGKILL")
        ]
        failures["sglang_decode_detokenizer"] = [
            Failure(30, decode_worker, "sglang::detokenizer", "SIGKILL")
        ]
        failures["sglang_prefill_scheduler"] = [
            Failure(30, prefill_worker, "sglang::scheduler", "SIGKILL")
        ]
        failures["sglang_prefill_detokenizer"] = [
            Failure(30, prefill_worker, "sglang::detokenizer", "SIGKILL")
        ]
    elif backend == "trtllm":
        failures["trtllm_decode_engine_core"] = [
            Failure(30, decode_worker, "TRTLLM::EngineCore", "SIGKILL")
        ]
        failures["trtllm_prefill_engine_core"] = [
            Failure(30, prefill_worker, "TRTLLM::EngineCore", "SIGKILL")
        ]

    return failures


load = Load()

# model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

model = None

# Populate Scenarios

scenarios = {}

# Map of backend+deploy_type to failure definitions
backend_failure_map = {}
for backend in ["vllm", "sglang", "trtllm"]:
    backend_failure_map[f"{backend}_agg"] = _create_backend_failures(backend, "agg")
    backend_failure_map[f"{backend}_disagg"] = _create_backend_failures(
        backend, "disagg"
    )

for deployment_name, deployment_info in deployment_specs.items():
    backend = deployment_info["backend"]

    # Determine deployment type from deployment name
    deploy_type = (
        "agg"
        if ("agg" in deployment_name and "disagg" not in deployment_name)
        else "disagg"
    )

    # Get the appropriate failure set for this backend+deploy_type
    failure_map_key = f"{backend}_{deploy_type}"
    if failure_map_key not in backend_failure_map:
        raise ValueError(
            f"Unsupported backend+deploy_type: {failure_map_key}. Available: {list(backend_failure_map.keys())}"
        )

    failure_set = backend_failure_map[failure_map_key]

    for failure_name, failure in failure_set.items():
        # Skip prefill failures for aggregated deployments
        if "prefill" in failure_name and deploy_type == "agg":
            continue

        scenario_name = f"{deployment_name}-{failure_name}"
        scenarios[scenario_name] = Scenario(
            deployment=deployment_info["spec"],
            load=load,
            failures=failure,
            model=model,
            backend=backend,
        )
