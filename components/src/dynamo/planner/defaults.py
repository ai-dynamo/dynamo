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
import shlex
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from dynamo.planner.kube import get_current_k8s_namespace
from dynamo.planner.utils.exceptions import (
    DuplicateSubComponentError,
    SubComponentNotFoundError,
)
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def _get_prometheus_port_from_env():
    """
    Get prometheus port from environment variables if set.
    Otherwise, return 0, which means not reporting metrics using prometheus.
    """
    return os.environ.get("PLANNER_PROMETHEUS_PORT", 0)


# Source of truth for planner defaults
class BasePlannerDefaults:
    namespace = "dynamo"
    environment = "kubernetes"
    backend = "vllm"
    no_operation = False
    log_dir = None
    adjustment_interval = 180  # in seconds
    max_gpu_budget = 8
    min_endpoint = 1  # applies to both decode and prefill
    decode_engine_num_gpu = 1
    prefill_engine_num_gpu = 1
    prometheus_port = _get_prometheus_port_from_env()


class LoadPlannerDefaults(BasePlannerDefaults):
    metric_pulling_interval = 10  # in seconds
    decode_kv_scale_up_threshold = 0.9
    decode_kv_scale_down_threshold = 0.5
    prefill_queue_scale_up_threshold = 5.0
    prefill_queue_scale_down_threshold = 0.2


def _get_default_prometheus_endpoint(port: str, namespace: str):
    """Compute default prometheus endpoint using environment variables and Kubernetes service discovery"""
    prometheus_endpoint = os.environ.get("PROMETHEUS_ENDPOINT", "").strip()
    if prometheus_endpoint:
        logger.debug("Using PROMETHEUS_ENDPOINT override: %s", prometheus_endpoint)
        return prometheus_endpoint

    k8s_namespace = get_current_k8s_namespace()
    if k8s_namespace and k8s_namespace != "default":
        prometheus_service = f"{namespace}-prometheus"
        return f"http://{prometheus_service}.{k8s_namespace}.svc.cluster.local:{port}"
    else:
        logger.warning(
            f"Cannot determine Prometheus endpoint. Running in namespace '{k8s_namespace}'. "
            "Ensure the planner is deployed in a Kubernetes cluster with proper namespace configuration."
        )
        return f"{namespace}-prometheus"


class SLAPlannerDefaults(BasePlannerDefaults):
    port = os.environ.get("PROMETHEUS_PORT", "9090")
    namespace = os.environ.get("DYN_NAMESPACE", "vllm-disagg-planner")
    prometheus_endpoint = _get_default_prometheus_endpoint(port, namespace)
    profile_results_dir = "profiling_results"
    isl = 3000  # in number of tokens
    osl = 150  # in number of tokens
    ttft = 0.5  # in seconds
    itl = 0.05  # in seconds
    load_predictor = "arima"  # ["constant", "arima", "prophet"]
    load_prediction_window_size = 50  # predict load using how many recent load samples
    no_correction = False  # disable correction factor, might be useful under some conditions like long cold start time


class VllmComponentName:
    prefill_worker_k8s_name = "VllmPrefillWorker"
    prefill_worker_component_name = "prefill"
    prefill_worker_endpoint = "generate"
    decode_worker_k8s_name = "VllmDecodeWorker"
    decode_worker_component_name = "backend"
    decode_worker_endpoint = "generate"


class SGLangComponentName:
    prefill_worker_k8s_name = (
        "prefill"  # use short name to stay within k8s limits with grove
    )
    prefill_worker_component_name = "prefill"
    prefill_worker_endpoint = "generate"
    decode_worker_k8s_name = (
        "decode"  # use short name to stay within k8s limits with grove
    )
    decode_worker_component_name = "backend"
    decode_worker_endpoint = "generate"


class TrtllmComponentName:
    # Note: Planner only supports DECODE_FIRST strategy in TRT-LLM:
    # - Decode worker is the first worker (tensorrt_llm)
    # - Prefill worker is the next worker (tensorrt_llm_next)
    prefill_worker_k8s_name = "TRTLLMPrefillWorker"
    prefill_worker_component_name = (
        "tensorrt_llm_next"  # Prefill is "next" with DECODE_FIRST
    )
    prefill_worker_endpoint = "generate"
    decode_worker_k8s_name = "TRTLLMDecodeWorker"
    decode_worker_component_name = "tensorrt_llm"  # Decode is "first" with DECODE_FIRST
    decode_worker_endpoint = "generate"


WORKER_COMPONENT_NAMES = {
    "vllm": VllmComponentName,
    "sglang": SGLangComponentName,
    "trtllm": TrtllmComponentName,
}


class SubComponentType(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


def break_arguments(args: list[str] | None) -> list[str]:
    ans: list[str] = []
    if args is None:
        return ans
    if isinstance(args, str):
        # Use shlex.split to properly handle quoted arguments and JSON values
        ans = shlex.split(args)
    else:
        for arg in args:
            if arg is not None:
                # Use shlex.split to properly handle quoted arguments
                ans.extend(shlex.split(arg))
    return ans


class Service(BaseModel):
    name: str
    service: dict

    def number_replicas(self) -> int:
        return self.service.get("replicas", 0)

    def get_model_name(self) -> Optional[str]:
        args = (
            self.service.get("extraPodSpec", {})
            .get("mainContainer", {})
            .get("args", [])
        )

        args = break_arguments(args)
        if (
            "--served-model-name" in args
            and len(args) > args.index("--served-model-name") + 1
        ):
            return args[args.index("--served-model-name") + 1]
        if "--model" in args and len(args) > args.index("--model") + 1:
            return args[args.index("--model") + 1]

        return None


# TODO: still supporting framework component names for backwards compatibility
# Should be deprecated in favor of service subComponentType
def get_service_from_sub_component_type_or_name(
    deployment: dict,
    sub_component_type: SubComponentType,
    component_name: Optional[str] = None,
) -> Service:
    """
    Get the current replicas for a component in a graph deployment

    Returns: Service object

    Raises:
        SubComponentNotFoundError: If no service with the specified subComponentType is found
        DuplicateSubComponentError: If multiple services with the same subComponentType are found
    """
    services = deployment.get("spec", {}).get("services", {})

    # Collect all available subComponentTypes for better error messages
    available_types = []
    matching_services = []

    for curr_name, curr_service in services.items():
        service_sub_type = curr_service.get("subComponentType", "")
        if service_sub_type:
            available_types.append(service_sub_type)

        if service_sub_type == sub_component_type.value:
            matching_services.append((curr_name, curr_service))

    # Check for duplicates
    if len(matching_services) > 1:
        service_names = [name for name, _ in matching_services]
        raise DuplicateSubComponentError(sub_component_type.value, service_names)

    # If no service found with subCompontType and fallback component_name is not provided or not found,
    # or if the fallback component has a non-empty subComponentType, raise error
    if not matching_services and (
        not component_name
        or component_name not in services
        or services[component_name].get("subComponentType", "") != ""
    ):
        raise SubComponentNotFoundError(sub_component_type.value)
    # If fallback component_name is provided and exists within services, add to matching_services
    elif not matching_services and component_name in services:
        matching_services.append((component_name, services[component_name]))

    name, service = matching_services[0]
    return Service(name=name, service=service)
