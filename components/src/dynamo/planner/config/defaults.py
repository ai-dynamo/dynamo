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

import os
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel


class BasePlannerDefaults:
    # Namespace from DYN_NAMESPACE env var (injected by operator as "{k8s_namespace}-{dgd_name}")
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    environment: Literal["kubernetes", "virtual", "global-planner"] = "kubernetes"
    backend: Literal["vllm", "sglang", "trtllm", "mocker"] = "vllm"
    log_dir = None
    throughput_adjustment_interval = 180  # in seconds
    max_gpu_budget = 8
    min_endpoint = 1  # applies to both decode and prefill
    decode_engine_num_gpu = 1
    prefill_engine_num_gpu = 1
    # Port for exposing planner's own metrics (0 means disabled)
    metric_reporting_prometheus_port = int(os.environ.get("PLANNER_PROMETHEUS_PORT", 0))


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

    mode: Literal["disagg", "prefill", "decode", "agg"] = "disagg"

    throughput_metrics_source: Literal["frontend", "router"] = "frontend"

    # Scaling mode flags
    enable_throughput_scaling = True
    enable_load_scaling = False

    # Load-based scaling settings
    load_adjustment_interval = 5  # in seconds; also controls FPM regression update frequency for throughput scaling
    max_num_fpm_samples = 64  # max retained FPM observations for regression
    fpm_sample_bucket_size = (
        16  # must be a perfect square; total buckets across input axes
    )
    load_scaling_down_sensitivity = 80  # 0-100
    load_metric_samples = 10  # number of samples per interval
    load_min_observations = 5  # cold start threshold

    # Advisory mode: compute and log decisions without executing scaling
    advisory = False


class SubComponentType(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class TargetReplica(BaseModel):
    sub_component_type: SubComponentType
    component_name: Optional[str] = None
    desired_replicas: int
