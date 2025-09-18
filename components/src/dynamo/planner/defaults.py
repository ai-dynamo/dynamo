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

from dynamo.planner.kube import get_current_k8s_namespace
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
    namespace = os.environ.get("DYNAMO_NAMESPACE", "vllm-disagg-planner")
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
    prefill_worker_k8s_name = "SGLangPrefillWorker"
    prefill_worker_component_name = "worker"
    prefill_worker_endpoint = "generate"
    decode_worker_k8s_name = "SGLangDecodeWorker"
    decode_worker_component_name = "decode"
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
