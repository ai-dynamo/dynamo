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
        help="Namespace",
    )
    parser.add_argument(
        "--backend",
        default=SLAPlannerDefaults.backend,
        choices=["vllm", "sglang", "trtllm"],
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
        help="Profile results directory",
    )
    parser.add_argument(
        "--ttft",
        type=float,
        default=SLAPlannerDefaults.ttft,
        help="Time to first token",
    )
    parser.add_argument(
        "--itl", type=float, default=SLAPlannerDefaults.itl, help="Inter-token latency"
    )
    parser.add_argument(
        "--load-predictor",
        default=SLAPlannerDefaults.load_predictor,
        help="Load predictor type",
    )
    parser.add_argument(
        "--load-prediction-window-size",
        type=int,
        default=SLAPlannerDefaults.load_prediction_window_size,
        help="Load prediction window size",
    )
    parser.add_argument(
        "--prometheus-port",
        type=int,
        default=SLAPlannerDefaults.prometheus_port,
        help="Prometheus port",
    )
    parser.add_argument(
        "--no-correction",
        action="store_true",
        default=SLAPlannerDefaults.no_correction,
        help="Disable correction factor",
    )
    return parser
