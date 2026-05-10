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
    throughput_adjustment_interval_seconds = 180
    max_gpu_budget = 8
    # GPU floor for the local planner (per-DGD scope). -1 disables.
    # When set alongside max_gpu_budget (with min == max), pins the total
    # and the planner only redistributes replicas between pools.
    # See dynamo.planner.core.budget.proportional_clamp_pair for the
    # tolerance band semantics.
    min_gpu_budget = -1
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
    # Optional bearer token sent as `Authorization: Bearer <token>` on every
    # PromQL request. Useful for hardened monitoring stacks that require
    # token auth (OpenShift thanos-querier, anything fronted by an OAuth
    # proxy). When set, the token is read once at startup; for SA tokens
    # that rotate, use a token_file knob instead (separate config field
    # planned as follow-up).
    metric_pulling_prometheus_token = os.environ.get("PROMETHEUS_TOKEN")
    # Verify the upstream Prometheus TLS certificate. Default False preserves
    # the previous PrometheusConnect(disable_ssl=True) behavior so existing
    # deployments are unaffected. Set to True (or env PROMETHEUS_SSL_VERIFY=1)
    # for hardened monitoring stacks where you want the request to fail
    # closed on a bad cert. Pair with PROMETHEUS_CA_BUNDLE if the upstream
    # uses a private CA.
    metric_pulling_prometheus_ssl_verify = os.environ.get(
        "PROMETHEUS_SSL_VERIFY", "false"
    ).lower() in ("1", "true", "yes")
    profile_results_dir = "profiling_results"

    isl = 3000  # in number of tokens
    osl = 150  # in number of tokens
    ttft_ms = 500.0
    itl_ms = 50.0

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
    load_adjustment_interval_seconds = (
        5  # also controls FPM regression update frequency for throughput scaling
    )
    max_num_fpm_samples = 64  # max retained FPM observations for regression
    fpm_sample_bucket_size = (
        16  # must be a perfect square; total buckets across input axes
    )
    load_scaling_down_sensitivity = 80  # 0-100
    load_metric_samples = 10  # number of samples per interval
    load_min_observations = 5  # cold start threshold
    prefill_scale_up_queue_tokens = None
    prefill_scale_down_queue_tokens = None
    decode_scale_up_kv_rate = None
    decode_scale_down_kv_rate = None

    # Advisory mode: compute and log decisions without executing scaling
    advisory = False

    # Power-aware scaling (Phase 1 + 2, disabled by default for backward compat).
    # total_gpu_power_limit and power_agent_safe_default_watts are required when
    # enable_power_awareness=True (validator enforces). Defaults are None — not
    # placeholder integers — so the type itself signals "operator must set".
    enable_power_awareness: bool = False
    total_gpu_power_limit: Optional[int] = None
    prefill_engine_gpu_power_limit: int = 300  # watts per GPU, prefill replicas
    decode_engine_gpu_power_limit: int = 300   # watts per GPU, decode replicas
    power_agent_safe_default_watts: Optional[int] = None

    # AIC optimizer (Phase 3+, disabled by default)
    enable_aic_optimizer: bool = False
    aic_reoptimize_interval: int = 300          # seconds between AIC sweeps
    aic_drift_relative_threshold: float = 0.15  # fraction before re-sweep triggers
    aic_drift_consecutive_ticks: int = 3        # ticks of sustained signal needed
    # Cold-start values for per-component power coefficients (Phase 3+).
    aic_initial_c_power_prefill: float = 1.0
    aic_initial_c_power_decode: float = 1.0
    aic_initial_c_power_agg: float = 1.0
    aic_initial_c_ttft: float = 1.0
    aic_initial_c_itl: float = 1.0
    aic_max_consecutive_failures: int = 5
    aic_throughput_regression_warn_threshold: float = 0.10


class SubComponentType(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class TargetReplica(BaseModel):
    sub_component_type: SubComponentType
    component_name: Optional[str] = None
    desired_replicas: int
