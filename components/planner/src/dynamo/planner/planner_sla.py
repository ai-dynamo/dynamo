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
import asyncio
import logging

from pydantic import BaseModel

from dynamo.planner.defaults import SLAPlannerDefaults
from dynamo.planner.utils.planner_core import start_sla_planner
from dynamo.runtime import DistributedRuntime, dynamo_worker

logger = logging.getLogger(__name__)

# start planner 30 seconds after the other components to make sure planner can see them
# TODO: remove this delay
INIT_PLANNER_START_DELAY = 30


class RequestType(BaseModel):
    text: str


@dynamo_worker(static=False)
async def init_planner(runtime: DistributedRuntime, args):
    await asyncio.sleep(INIT_PLANNER_START_DELAY)

    await start_sla_planner(runtime, args)

    component = runtime.namespace(SLAPlannerDefaults.namespace).component("Planner")
    await component.create_service()

    async def generate(request: RequestType):
        """Dummy endpoint to satisfy that each component has an endpoint"""
        yield "mock endpoint"

    generate_endpoint = component.endpoint("generate")
    await generate_endpoint.serve_endpoint(generate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLA Planner")
    parser.add_argument(
        "--environment",
        default=SLAPlannerDefaults.environment,
        choices=["kubernetes"],
        help="Environment type",
    )
    parser.add_argument(
        "--backend",
        default=SLAPlannerDefaults.backend,
        choices=["vllm", "sglang"],
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
        "--isl", type=int, default=SLAPlannerDefaults.isl, help="Input sequence length"
    )
    parser.add_argument(
        "--osl", type=int, default=SLAPlannerDefaults.osl, help="Output sequence length"
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

    args = parser.parse_args()
    asyncio.run(init_planner(args))
