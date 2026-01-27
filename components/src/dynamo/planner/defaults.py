# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dynamo.planner.utils.exceptions import (
    DuplicateSubComponentError,
    SubComponentNotFoundError,
)
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


# Source of truth for planner defaults
class BasePlannerDefaults:
    # Namespace from DYN_NAMESPACE env var (injected by operator as "{k8s_namespace}-{dgd_name}")
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    environment = "kubernetes"
    backend = "vllm"
    no_operation = False
    log_dir = None
    adjustment_interval = 180  # in seconds
    max_gpu_budget = 8
    min_endpoint = 1  # applies to both decode and prefill
    decode_engine_num_gpu = 1
    prefill_engine_num_gpu = 1
    # Port for exposing planner's own metrics (0 means disabled)
    metric_reporting_prometheus_port = int(os.environ.get("PLANNER_PROMETHEUS_PORT", 0))


class LoadPlannerDefaults(BasePlannerDefaults):
    metric_pulling_interval = 10  # in seconds
    decode_kv_scale_up_threshold = 0.9
    decode_kv_scale_down_threshold = 0.5
    prefill_queue_scale_up_threshold = 5.0
    prefill_queue_scale_down_threshold = 0.2


class SLAPlannerDefaults(BasePlannerDefaults):
    # Prometheus endpoint URL for pulling/querying metrics
    metric_pulling_prometheus_endpoint = os.environ.get(
        "PROMETHEUS_ENDPOINT",
        "http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090",
    )
    profile_results_dir = "profiling_results"

    isl = 3000  # in number of tokens
    osl = 150  # in number of tokens
    ttft = 500.0  # in milliseconds
    itl = 50.0  # in milliseconds

    # for load predictor
    load_predictor = "arima"  # ["constant", "arima", "kalman", "prophet"]
    prophet_window_size = 50
    load_predictor_log1p = False
    kalman_q_level = 1.0
    kalman_q_trend = 0.1
    kalman_r = 10.0
    kalman_min_points = 5

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
    # Unified frontend architecture (consistent with vLLM/SGLang):
    # - Prefill workers use "prefill" component
    # - Decode workers use "tensorrt_llm" component
    prefill_worker_k8s_name = "TRTLLMPrefillWorker"
    prefill_worker_component_name = "prefill"
    prefill_worker_endpoint = "generate"
    decode_worker_k8s_name = "TRTLLMDecodeWorker"
    decode_worker_component_name = "tensorrt_llm"
    decode_worker_endpoint = "generate"


class MockerComponentName:
    # Mocker backend for testing/simulation purposes
    prefill_worker_k8s_name = "prefill"
    prefill_worker_component_name = "prefill"
    prefill_worker_endpoint = "generate"
    decode_worker_k8s_name = "decode"
    decode_worker_component_name = "backend"
    decode_worker_endpoint = "generate"


WORKER_COMPONENT_NAMES = {
    "vllm": VllmComponentName,
    "sglang": SGLangComponentName,
    "trtllm": TrtllmComponentName,
    "mocker": MockerComponentName,
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
        if (
            "--model-name" in args and len(args) > args.index("--model-name") + 1
        ):  # mocker use --model-name
            return args[args.index("--model-name") + 1]
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
