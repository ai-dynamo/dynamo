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

import os
import socket


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


class LoadPlannerDefaults(BasePlannerDefaults):
    metric_pulling_interval = 10  # in seconds
    decode_kv_scale_up_threshold = 0.9
    decode_kv_scale_down_threshold = 0.5
    prefill_queue_scale_up_threshold = 5.0
    prefill_queue_scale_down_threshold = 0.2


def _get_dynamo_namespace_from_k8s() -> str:
    """Get the dynamo namespace from current pod's Kubernetes labels"""
    try:
        from kubernetes import client

        from dynamo.planner.kube import KubernetesAPI

        k8s_api = KubernetesAPI()
        v1 = client.CoreV1Api()

        # Get current pod name from hostname
        hostname = socket.gethostname()

        # Get current pod to read its labels
        pod = v1.read_namespaced_pod(name=hostname, namespace=k8s_api.current_namespace)
        labels = pod.metadata.labels or {}

        # Extract dynamo namespace from labels
        dynamo_namespace = labels.get("nvidia.com/dynamo-namespace")
        if not dynamo_namespace:
            raise RuntimeError(
                "Failed to determine the dynamo namespace from Kubernetes pod labels"
            )
        return dynamo_namespace

    except Exception as e:
        raise RuntimeError(
            "Failed to determine the dynamo namespace from Kubernetes pod labels"
        ) from e


def _get_default_prometheus_endpoint(port: str):
    """Compute default prometheus endpoint using Kubernetes service discovery"""

    # Try to get current namespace and deployment name from Kubernetes
    try:
        from dynamo.planner.kube import KubernetesAPI

        k8s_api = KubernetesAPI()
        k8s_namespace = k8s_api.current_namespace

        if k8s_namespace and k8s_namespace != "default":
            dynamo_namespace = _get_dynamo_namespace_from_k8s()
            prometheus_service = f"{dynamo_namespace}-prometheus"
            return (
                f"http://{prometheus_service}.{k8s_namespace}.svc.cluster.local:{port}"
            )
    except Exception as e:
        raise RuntimeError(
            "Failed to determine the prometheus endpoint from Kubernetes service discovery"
        ) from e


class SLAPlannerDefaults(BasePlannerDefaults):
    port = os.environ.get("DYNAMO_PORT", "9090")
    prometheus_endpoint = _get_default_prometheus_endpoint(port)
    profile_results_dir = "profiling_results"
    isl = 3000  # in number of tokens
    osl = 150  # in number of tokens
    ttft = 0.5  # in seconds
    itl = 0.05  # in seconds
    load_predictor = "arima"  # ["constant", "arima", "prophet"]
    load_prediction_window_size = 50  # predict load using how many recent load samples


class VllmComponentName:
    prefill_worker = "prefill"
    prefill_worker_endpoint = "generate"
    decode_worker = "backend"
    decode_worker_endpoint = "generate"


WORKER_COMPONENT_NAMES = {
    "vllm": VllmComponentName,
}
