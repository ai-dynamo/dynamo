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

import argparse

from dynamo.planner.defaults import SLAPlannerDefaults


def create_sla_planner_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for SLA Planner.

    Returns:
        argparse.ArgumentParser: Configured argument parser for SLA Planner
    """
    parser = argparse.ArgumentParser(description="SLA Planner")
    parser.add_argument(
        "--environment",
        default=SLAPlannerDefaults.environment,
        choices=["kubernetes", "virtual"],
        help="Environment type",
    )
    parser.add_argument(
        "--namespace",
        default=SLAPlannerDefaults.namespace,
        help="Dynamo namespace",
    )
    parser.add_argument(
        "--backend",
        default=SLAPlannerDefaults.backend,
        choices=["vllm", "sglang", "trtllm", "mocker"],
        help="Backend type",
    )
    parser.add_argument(
        "--no-operation",
        action="store_true",
        default=SLAPlannerDefaults.no_operation,
        help="Enable no-operation mode",
    )
    parser.add_argument(
        "--log-dir", default=SLAPlannerDefaults.log_dir, help="Log directory path"
    )
    parser.add_argument(
        "--adjustment-interval",
        type=int,
        default=SLAPlannerDefaults.adjustment_interval,
        help="Adjustment interval in seconds",
    )
    parser.add_argument(
        "--max-gpu-budget",
        type=int,
        default=SLAPlannerDefaults.max_gpu_budget,
        help="Maximum GPU budget",
    )
    parser.add_argument(
        "--min-endpoint",
        type=int,
        default=SLAPlannerDefaults.min_endpoint,
        help="Minimum number of endpoints",
    )
    parser.add_argument(
        "--decode-engine-num-gpu",
        type=int,
        default=SLAPlannerDefaults.decode_engine_num_gpu,
        help="Number of GPUs for decode engine",
    )
    parser.add_argument(
        "--prefill-engine-num-gpu",
        type=int,
        default=SLAPlannerDefaults.prefill_engine_num_gpu,
        help="Number of GPUs for prefill engine",
    )
    parser.add_argument(
        "--profile-results-dir",
        default=SLAPlannerDefaults.profile_results_dir,
        help="Profile results directory or 'use-pre-swept-results:<gpu_type>:<framework>:<model>:<tp>:<dp>:<pp>:<block_size>:<max_batch_size>:<gpu_count>' to use pre-swept results from pre_swept_results directory",
    )
    parser.add_argument(
        "--ttft",
        type=float,
        default=SLAPlannerDefaults.ttft,
        help="Time to first token (float, in milliseconds)",
    )
    parser.add_argument(
        "--itl",
        type=float,
        default=SLAPlannerDefaults.itl,
        help="Inter-token latency (float, in milliseconds)",
    )
    parser.add_argument(
        "--load-predictor",
        default=SLAPlannerDefaults.load_predictor,
        help="Load predictor type (constant, arima, kalman, prophet)",
    )
    parser.add_argument(
        "--load-predictor-log1p",
        action="store_true",
        default=SLAPlannerDefaults.load_predictor_log1p,
        help="Model log1p(y) instead of y in the selected load predictor (ARIMA/Kalman/Prophet)",
    )
    parser.add_argument(
        "--prophet-window-size",
        type=int,
        default=SLAPlannerDefaults.prophet_window_size,
        help="Prophet history window size",
    )
    parser.add_argument(
        "--load-predictor-warmup-trace",
        type=str,
        default=None,
        help="Optional path to a mooncake-style JSONL trace file used to warm up load predictors before observing live traffic",
    )
    parser.add_argument(
        "--kalman-q-level",
        type=float,
        default=SLAPlannerDefaults.kalman_q_level,
        help="Kalman process noise for level (higher = more responsive)",
    )
    parser.add_argument(
        "--kalman-q-trend",
        type=float,
        default=SLAPlannerDefaults.kalman_q_trend,
        help="Kalman process noise for trend (higher = faster trend changes)",
    )
    parser.add_argument(
        "--kalman-r",
        type=float,
        default=SLAPlannerDefaults.kalman_r,
        help="Kalman measurement noise (lower = remember less / react more to new measurements)",
    )
    parser.add_argument(
        "--kalman-min-points",
        type=int,
        default=SLAPlannerDefaults.kalman_min_points,
        help="Minimum number of points before Kalman predictor returns forecasts",
    )
    parser.add_argument(
        "--metric-pulling-prometheus-endpoint",
        type=str,
        default=SLAPlannerDefaults.metric_pulling_prometheus_endpoint,
        help="Prometheus endpoint URL for pulling dynamo deployment metrics",
    )
    parser.add_argument(
        "--metric-reporting-prometheus-port",
        type=int,
        default=SLAPlannerDefaults.metric_reporting_prometheus_port,
        help="Port for exposing planner's own metrics to Prometheus",
    )
    parser.add_argument(
        "--no-correction",
        action="store_true",
        default=SLAPlannerDefaults.no_correction,
        help="Disable correction factor",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name of deployment (only required for virtual environment)",
    )
    return parser
