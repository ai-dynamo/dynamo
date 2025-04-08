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
import json
import logging
import os
import time

import numpy as np
from rich.console import Console
from rich.table import Table
from utils.prefill_queue import PrefillQueue

from dynamo.llm import KvMetricsAggregator
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_logger

configure_logger()
logger = logging.getLogger(__name__)


class Planner:
    def __init__(self, runtime: DistributedRuntime, args: argparse.Namespace):
        self.runtime = runtime
        self.args = args
        self.namespace = args.namespace

        self._prefill_queue_nats_server = os.getenv(
            "NATS_SERVER", "nats://localhost:4222"
        )
        self._prefill_queue_stream_name = self.args.served_model_name

        self.init_time = time.time()

    async def set_metric_aggregator(self):
        # TODO: separate KV metrics and prefill metrics
        kv_listener = self.runtime.namespace(self.namespace).component("VllmWorker")
        await kv_listener.create_service()
        self.metrics_aggregator = KvMetricsAggregator(kv_listener)

    async def get_workers_info(self):
        try:
            prefill_client = (
                await self.runtime.namespace(self.namespace)
                .component("PrefillWorker")
                .endpoint("mock")
                .client()
            )
            p_endpoints = prefill_client.endpoint_ids()
        except Exception:
            p_endpoints = []
            logger.info("No prefill workers found, operating in aggregated mode")
        try:
            workers_client = (
                await self.runtime.namespace(self.namespace)
                .component("VllmWorker")
                .endpoint("generate")
                .client()
            )
            d_endpoints = workers_client.endpoint_ids()
        except Exception as e:
            raise RuntimeError(f"Failed to get decode worker endpoints: {e}")
        return p_endpoints, d_endpoints

    async def reset_adjustment_interval(self):
        logger.info(
            f"Reset metrics for new adjustment interval at t={time.time() - self.init_time:.1f}s"
        )

        self.p_endpoints, self.d_endpoints = await self.get_workers_info()
        logger.info(
            f"Number of prefill workers: {len(self.p_endpoints)}, number of decode workers: {len(self.d_endpoints)}"
        )

        self.metrics_collection_time = []
        self.prefill_queue_load = []
        self.kv_load = []

        self.last_adjustment_time = time.time()

    async def collect_metrics(self):
        logger.info(f"Collecting metrics at t={time.time() - self.init_time:.1f}s")

        # collect prefill queue load
        async with PrefillQueue.get_instance(
            nats_server=self._prefill_queue_nats_server,
            stream_name=self._prefill_queue_stream_name,
        ) as prefill_queue:
            prefill_queue_size = await prefill_queue.get_queue_size()
        self.prefill_queue_load.append(prefill_queue_size > 0)
        logger.info(
            f"Collected prefill queue size at t={time.time() - self.init_time:.1f}s: {int(prefill_queue_size)}"
        )

        # collect kv load
        metrics = await self.metrics_aggregator.get_metrics()
        try:
            prev_kv_load_len = len(self.kv_load)
            for endpoint in metrics.endpoints:
                kv_load = getattr(endpoint, "gpu_cache_usage_perc", 0.0)
                num_requests_waiting = getattr(endpoint, "num_requests_waiting", 0.0)
                if num_requests_waiting > 0:
                    # if requests are waiting, we assume the needed kv is higher
                    kv_load = 1.2
                self.kv_load.append(kv_load)
            logger.info(
                f"Collected kv load at t={time.time() - self.init_time:.1f}s: {self.kv_load[prev_kv_load_len:]}"
            )
        except Exception as e:
            logger.warning(f"Failed to collect kv load metrics: {e}")

        self.metrics_collection_time.append(time.time())

    async def make_adjustments(self):
        # Note: all adjustments are blocking. Non-blocking adjustment and metric pulling
        # make the optimization problem too complex and should not be needed in most cases.
        logger.info(f"Making adjustments at t={time.time() - self.init_time:.1f}s")

        # check if decode/prefill workers is still the same
        # note that we only check length as endpoint ids might change
        new_p_endpoints, new_d_endpoints = await self.get_workers_info()
        if len(new_p_endpoints) != len(self.p_endpoints) or len(new_d_endpoints) != len(
            self.d_endpoints
        ):
            logger.warning(
                "Decode/prefill workers changed, no adjustments will be made"
            )
            return

        # compute current gpu usage
        curr_gpu_usage = (
            len(self.p_endpoints) * self.args.prefill_engine_num_gpu
            + len(self.d_endpoints) * self.args.decode_engine_num_gpu
        )
        logger.info(f"Current engines use {curr_gpu_usage} GPUs")

        # check if we need to scale up/down decode workers
        avg_kv_load = np.mean(self.kv_load)
        if (
            avg_kv_load < self.args.decode_kv_scale_down_threshold
            and len(self.d_endpoints) > self.args.min_gpu_budget
        ):
            logger.info(
                f"Average kv load ({avg_kv_load:.2f}) is below threshold ({self.args.decode_kv_scale_down_threshold:.2f}), scaling down decode workers"
            )
            # TODO: scale down one decode worker
            curr_gpu_usage -= self.args.decode_engine_num_gpu
        elif (
            avg_kv_load > self.args.decode_kv_scale_up_threshold
            and curr_gpu_usage + self.args.decode_engine_num_gpu
            <= self.args.max_gpu_budget
        ):
            logger.info(
                f"Average kv load ({avg_kv_load:.2f}) is above threshold ({self.args.decode_kv_scale_up_threshold:.2f}), scaling up decode workers"
            )
            # TODO: scale up one decode worker
            curr_gpu_usage += self.args.decode_engine_num_gpu
        else:
            logger.info(
                f"kv load ({avg_kv_load:.2f}) is within threshold, no decode worker scaling needed"
            )

        # check if we need to scale up/down prefill workers
        avg_prefill_queue_load = np.mean(self.prefill_queue_load)
        if (
            avg_prefill_queue_load < self.args.prefill_queue_scale_down_threshold
            and len(self.p_endpoints) > self.args.min_gpu_budget
        ):
            logger.info(
                f"Average prefill queue load ({avg_prefill_queue_load:.2f}) is below threshold ({self.args.prefill_queue_scale_down_threshold:.2f}), scaling down prefill workers"
            )
            # TODO: scale down one prefill worker
            curr_gpu_usage -= self.args.prefill_engine_num_gpu
        elif (
            avg_prefill_queue_load > self.args.prefill_queue_scale_up_threshold
            and curr_gpu_usage + self.args.prefill_engine_num_gpu
            <= self.args.max_gpu_budget
        ):
            logger.info(
                f"Average prefill queue load ({avg_prefill_queue_load:.2f}) is above threshold ({self.args.prefill_queue_scale_up_threshold:.2f}), scaling up prefill workers"
            )
            # TODO: scale up one prefill worker
            curr_gpu_usage += self.args.prefill_engine_num_gpu
        else:
            logger.info(
                f"prefill queue load ({avg_prefill_queue_load:.2f}) is within threshold, no prefill worker scaling needed"
            )

        logger.info(f"Engines after adjustment use {curr_gpu_usage} GPUs")

    async def run(self):
        """Main loop for the planner"""

        await self.set_metric_aggregator()
        await self.reset_adjustment_interval()

        while True:
            current_time = time.time()

            # Collect metrics at each metric pulling interval
            if (
                len(self.metrics_collection_time) == 0
                or current_time - self.metrics_collection_time[-1]
                >= self.args.metric_pulling_interval
            ):
                await self.collect_metrics()

            # Check if it's time for adjustment
            if (
                current_time - self.last_adjustment_time
                >= self.args.adjustment_interval
            ):
                await self.make_adjustments()
                await self.reset_adjustment_interval()

            # Sleep to avoid busy waiting
            await asyncio.sleep(self.args.metric_pulling_interval / 10)


@dynamo_worker()
async def start_planner(runtime: DistributedRuntime, args: argparse.Namespace):
    planner = Planner(runtime, args)
    logger.info(f"Components present in namespace: {args.namespace}")
    console = Console()
    table = Table()
    table.add_column("Component", style="cyan")
    table.add_column("Endpoint", style="green")

    components = await runtime.etcd_client().kv_get_prefix(args.namespace)
    for component in components:
        try:
            # Parse the byte string as JSON and extract component name
            data = json.loads(component["value"].decode("utf-8"))
            if "component" in data:
                name = data["component"]
                endpoint = data["endpoint"]
                table.add_row(name, endpoint)
        except Exception:
            # Some entries may not be valid JSON or might be binary data
            pass

    # Print the table before running the planner
    console.print(table)

    await planner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--namespace",
        type=str,
        required=True,
        help="Namespace planner will look at",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default="vllm",
        help="Model name that is being served (used for prefill queue name)",
    )
    parser.add_argument(
        "--adjustment-interval",
        type=int,
        default=300,
        help="Interval in seconds between scaling adjustments",
    )
    parser.add_argument(
        "--metric-pulling-interval",
        type=int,
        default=10,
        help="Interval in seconds between metric pulls",
    )
    parser.add_argument(
        "--max-gpu-budget",
        type=int,
        default=8,
        help="Maximum number of GPUs to use",
    )
    parser.add_argument(
        "--min-gpu-budget",
        type=int,
        default=1,
        help="Minimum number of GPUs to use for both prefill and decode",
    )
    parser.add_argument(
        "--decode-kv-scale-up-threshold",
        type=float,
        default=0.9,
        help="KV cache utilization threshold to scale up decode workers",
    )
    parser.add_argument(
        "--decode-kv-scale-down-threshold",
        type=float,
        default=0.5,
        help="KV cache utilization threshold to scale down decode workers",
    )
    parser.add_argument(
        "--prefill-queue-scale-up-threshold",
        type=float,
        default=0.5,
        help="Queue utilization threshold to scale up prefill workers",
    )
    parser.add_argument(
        "--prefill-queue-scale-down-threshold",
        type=float,
        default=0.2,
        help="Queue utilization threshold to scale down prefill workers",
    )
    parser.add_argument(
        "--decode-engine-num-gpu",
        type=int,
        default=1,
        help="Number of GPUs per decode engine",
    )
    parser.add_argument(
        "--prefill-engine-num-gpu",
        type=int,
        default=1,
        help="Number of GPUs per prefill engine",
    )
    args = parser.parse_args()
    asyncio.run(start_planner(args))
