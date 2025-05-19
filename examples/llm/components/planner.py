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
from datetime import datetime
from typing import Any, List

import numpy as np
from rich.console import Console
from rich.table import Table
from tensorboardX import SummaryWriter
from utils.prefill_queue import PrefillQueue

from dynamo.llm import KvMetricsAggregator
from dynamo.planner import KubernetesConnector, LocalConnector
from dynamo.planner.defaults import PlannerDefaults
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

# will not decrease decode worker number within 3 adjustment interval after a new decode worker
# is added. this is to leave time for the new decode worker to populate its kv cache.
NEW_DECODE_WORKER_GRACE_PERIOD = 3

# we do not scale up prefill worker if the prefill queue size is estimated to reduce within
# --prefill-queue-scale-up-threshold within the next NEW_PREFILL_WORKER_QUEUE_BUFFER_PERIOD
# adjustment intervals following the trend observed in the current adjustment interval.
# this is to account for the time for prefill workers to start.
NEW_PREFILL_WORKER_QUEUE_BUFFER_PERIOD = 3


class Planner:
    def __init__(self, runtime: DistributedRuntime, args: argparse.Namespace):
        self.runtime = runtime
        self.args = args
        self.namespace = args.namespace
        if args.environment == "local":
            self.connector = LocalConnector(args.namespace, runtime)
        elif args.environment == "kubernetes":
            self.connector = KubernetesConnector(args.namespace)
        else:
            raise ValueError(f"Invalid environment: {args.environment}")

        self._prefill_queue_nats_server = os.getenv(
            "NATS_SERVER", "nats://localhost:4222"
        )
        self._prefill_queue_stream_name = self.args.served_model_name

        self.prefill_client: Any | None = None
        self.workers_client: Any | None = None
        self.p_endpoints: List[int] = []
        self.d_endpoints: List[int] = []
        self.decode_worker_remaining_grace_period = 0

        if args.log_dir is None:
            args.log_dir = f"logs/{datetime.now().strftime('%m%d_%H%M%S')}"
        self.writer = SummaryWriter(args.log_dir)

        logger.info(f"Components present in namespace: {args.namespace}")

        self.init_time = time.time()
        # Set the appropriate logger function for repeated metric logging
        self._repeating_log_func = logger.debug if args.no_operation else logger.info

    async def set_metric_aggregator(self):
        # TODO: separate KV metrics and prefill metrics
        kv_listener = self.runtime.namespace(self.namespace).component("VllmWorker")
        await kv_listener.create_service()
        self.metrics_aggregator = KvMetricsAggregator(kv_listener)

    async def get_workers_info(self):
        try:
            if self.prefill_client is None:
                self.prefill_client = (
                    await self.runtime.namespace(self.namespace)
                    .component("PrefillWorker")
                    .endpoint("mock")
                    .client()
                )
                # TODO: remove this sleep after rust client() is blocking until watching state
                await asyncio.sleep(0.1)
            # TODO: use etcd events instead of pulling endpoints_ids
            p_endpoints = self.prefill_client.endpoint_ids()
        except Exception:
            p_endpoints = []
            self._repeating_log_func(
                "No prefill workers found, operating in aggregated mode"
            )
        try:
            if self.workers_client is None:
                self.workers_client = (
                    await self.runtime.namespace(self.namespace)
                    .component("VllmWorker")
                    .endpoint("generate")
                    .client()
                )
                # TODO: remove this sleep after rust client() is blocking until watching state
                await asyncio.sleep(0.1)
            # TODO: use etcd events instead of pulling endpoints_ids
            d_endpoints = self.workers_client.endpoint_ids()
        except Exception as e:
            raise RuntimeError(f"Failed to get decode worker endpoints: {e}")
        return p_endpoints, d_endpoints

    async def reset_adjustment_interval(self):
        self._repeating_log_func(
            f"Reset metrics for new adjustment interval at t={time.time() - self.init_time:.1f}s"
        )

        self.p_endpoints, self.d_endpoints = await self.get_workers_info()

        self._repeating_log_func(
            f"Number of prefill workers: {len(self.p_endpoints)}, number of decode workers: {len(self.d_endpoints)}"
        )

        self.metrics_collection_time = []
        self.prefill_queue_load = []
        self.kv_load = []

        self.last_adjustment_time = time.time()

    async def collect_metrics(self):
        self._repeating_log_func(
            f"Collecting metrics at t={time.time() - self.init_time:.1f}s"
        )

        # collect prefill queue load
        try:
            async with PrefillQueue.get_instance(
                nats_server=self._prefill_queue_nats_server,
                stream_name=self._prefill_queue_stream_name,
            ) as prefill_queue:
                prefill_queue_size = await prefill_queue.get_queue_size()
                measure_time = time.time() - self.init_time
            self.prefill_queue_load.append(prefill_queue_size)
            self._repeating_log_func(
                f"Collected prefill queue size at t={measure_time:.1f}s: {int(prefill_queue_size)}"
            )
            self.writer.add_scalar(
                "prefill_queue_size", prefill_queue_size, measure_time
            )
        except Exception as e:
            self._repeating_log_func(
                f"Failed to collect prefill queue size metrics: {e}"
            )

        # collect kv load
        total_active_requests: int = 0
        total_queued_requests: int = 0
        metrics = await self.metrics_aggregator.get_metrics()
        try:
            prev_kv_load_len = len(self.kv_load)
            for endpoint in metrics.endpoints:
                kv_load = getattr(endpoint, "gpu_cache_usage_perc", 0.0)
                num_requests_waiting = getattr(endpoint, "num_requests_waiting", 0)
                total_queued_requests += num_requests_waiting
                request_active_slots = getattr(endpoint, "request_active_slots", None)
                if request_active_slots:
                    total_active_requests += request_active_slots
                    if num_requests_waiting > 0:
                        # estimate kv load after waiting requests are scheduled based on current isl/osl
                        # TODO: use actual isl/osl estimation after the request_active_slot bug in disaggg is fixed
                        # Currently, we assume each request uses 0.02 kv cache
                        # kv_load = kv_load * (request_active_slots + num_requests_waiting) / request_active_slots
                        kv_load = kv_load + 0.02 * num_requests_waiting
                self.kv_load.append(kv_load)
            measure_time = time.time() - self.init_time
            self._repeating_log_func(
                f"Collected kv load at t={measure_time:.1f}s: {self.kv_load[prev_kv_load_len:]} (act/pnd req: {total_active_requests}/{total_queued_requests})"
            )
            average_kv_load = np.mean(self.kv_load[prev_kv_load_len:])
            self.writer.add_scalar("average_kv_load", average_kv_load, measure_time)
            self.writer.add_scalar(
                "total_queued_requests", total_queued_requests, measure_time
            )
        except Exception as e:
            self._repeating_log_func(f"Failed to collect kv load metrics: {e}")

        p_endpoints, d_endpoints = await self.get_workers_info()
        self.writer.add_scalar(
            "num_prefill_workers", len(p_endpoints), time.time() - self.init_time
        )
        self.writer.add_scalar(
            "num_decode_workers", len(d_endpoints), time.time() - self.init_time
        )
        curr_gpu_usage = (
            len(p_endpoints) * self.args.prefill_engine_num_gpu
            + len(d_endpoints) * self.args.decode_engine_num_gpu
        )
        self.writer.add_scalar("num_gpu", curr_gpu_usage, time.time() - self.init_time)

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
            logger.info("Decode/prefill workers changed, no adjustments will be made")
            return

        # compute current gpu usage
        curr_gpu_usage = (
            len(self.p_endpoints) * self.args.prefill_engine_num_gpu
            + len(self.d_endpoints) * self.args.decode_engine_num_gpu
        )
        logger.info(f"Current engines use {curr_gpu_usage} GPUs")

        avg_prefill_queue_load = np.mean(self.prefill_queue_load)
        avg_kv_load = np.mean(self.kv_load)
        # first check if we need to scale down any workers
        if (
            avg_prefill_queue_load < self.args.prefill_queue_scale_down_threshold
            and len(self.p_endpoints) > self.args.min_endpoint
        ):
            logger.info(
                f"Average prefill queue load ({avg_prefill_queue_load:.2f}) is below threshold ({self.args.prefill_queue_scale_down_threshold:.2f}), scaling down prefill workers"
            )
            success = await self.connector.remove_component("PrefillWorker")
            if success:
                curr_gpu_usage -= self.args.prefill_engine_num_gpu
            else:
                logger.info("Failed to scale down prefill worker")
        if (
            avg_kv_load < self.args.decode_kv_scale_down_threshold
            and len(self.d_endpoints) > self.args.min_endpoint
        ):
            if self.decode_worker_remaining_grace_period > 0:
                logger.info(
                    f"Decode worker remaining grace period is {self.decode_worker_remaining_grace_period}, skipping scale down"
                )
            else:
                logger.info(
                    f"Average kv load ({avg_kv_load:.2f}) is below threshold ({self.args.decode_kv_scale_down_threshold:.2f}), scaling down decode workers"
                )
                success = await self.connector.remove_component("VllmWorker")
                if success:
                    curr_gpu_usage -= self.args.decode_engine_num_gpu
                else:
                    logger.info("Failed to scale down decode worker")

        # check if we need to scale up workers
        # we first check for prefill worker because prefill queueing can also lead
        # to high kv load on decode workers
        if (
            avg_prefill_queue_load > self.args.prefill_queue_scale_up_threshold
            and curr_gpu_usage + self.args.prefill_engine_num_gpu
            <= self.args.max_gpu_budget
        ):
            logger.info(
                f"Average prefill queue load ({avg_prefill_queue_load:.2f}) is above threshold ({self.args.prefill_queue_scale_up_threshold:.2f})"
            )
            # check prefill queue size trend:
            prefill_queue_size_change = (
                self.prefill_queue_load[-1] - self.prefill_queue_load[0]
            )
            predicted_prefill_future_queue_size = (
                self.prefill_queue_load[-1]
                + prefill_queue_size_change * NEW_PREFILL_WORKER_QUEUE_BUFFER_PERIOD
            )
            if (
                predicted_prefill_future_queue_size
                > self.args.prefill_queue_scale_up_threshold
            ):
                logger.info(
                    f"Predicted future prefill queue size ({predicted_prefill_future_queue_size:.2f}) is also above threshold ({self.args.prefill_queue_scale_up_threshold:.2f}), scaling up prefill workers"
                )
                success = await self.connector.add_component("PrefillWorker")
                if success:
                    curr_gpu_usage += self.args.prefill_engine_num_gpu
                else:
                    logger.info("Failed to scale up prefill worker")
            else:
                logger.info(
                    f"Predicted future prefill queue size ({predicted_prefill_future_queue_size:.2f}) is below threshold ({self.args.prefill_queue_scale_up_threshold:.2f}), skipping prefill worker scaling"
                )
        if (
            avg_kv_load > self.args.decode_kv_scale_up_threshold
            and curr_gpu_usage + self.args.decode_engine_num_gpu
            <= self.args.max_gpu_budget
        ):
            logger.info(
                f"Average kv load ({avg_kv_load:.2f}) is above threshold ({self.args.decode_kv_scale_up_threshold:.2f}), scaling up decode workers"
            )
            success = await self.connector.add_component("VllmWorker")
            if success:
                curr_gpu_usage += self.args.decode_engine_num_gpu
                self.decode_worker_remaining_grace_period = (
                    NEW_DECODE_WORKER_GRACE_PERIOD
                )
            else:
                logger.info("Failed to scale up decode worker")

        # no adjustment needed, just log the current metrics
        if (
            avg_prefill_queue_load > self.args.prefill_queue_scale_down_threshold
            and avg_prefill_queue_load < self.args.prefill_queue_scale_up_threshold
        ):
            logger.info(
                f"Average prefill queue load ({avg_prefill_queue_load:.2f}) is within threshold, no prefill worker scaling needed"
            )
        if (
            avg_kv_load > self.args.decode_kv_scale_down_threshold
            and avg_kv_load < self.args.decode_kv_scale_up_threshold
        ):
            logger.info(
                f"Average kv load ({avg_kv_load:.2f}) is within threshold, no decode worker scaling needed"
            )

        logger.info(f"Engines after adjustment use {curr_gpu_usage} GPUs")

        if self.decode_worker_remaining_grace_period > 0:
            self.decode_worker_remaining_grace_period -= 1

    async def run(self):
        """Main loop for the planner"""

        await self.set_metric_aggregator()

        if self._repeating_log_func == logger.debug:
            logger.info(
                "Running in no-operation mode - detailed metrics will be logged at DEBUG level"
            )

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
                if not self.args.no_operation:
                    # blockingly make adjustments to avoid overcompensation
                    await self.make_adjustments()
                await self.reset_adjustment_interval()

            # Sleep to avoid busy waiting
            await asyncio.sleep(self.args.metric_pulling_interval / 10)


# @dynamo_worker()
# TODO: let's make it such that planner still works via CLI invokation
async def start_planner(runtime: DistributedRuntime, args: argparse.Namespace):
    planner = Planner(runtime, args)
    console = Console()
    table = Table()
    table.add_column("Component", style="cyan")
    table.add_column("Endpoint", style="green")

    components = await runtime.etcd_client().kv_get_prefix(args.namespace)
    for component in components:
        try:
            data = json.loads(component["value"].decode("utf-8"))
            if "component" in data:
                name = data["component"]
                endpoint = data["endpoint"]
                table.add_row(name, endpoint)
        except Exception:
            # Some entries may not be valid JSON or might be binary data
            pass

    console.print(table)

    await planner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--namespace",
        type=str,
        default=PlannerDefaults.namespace,
        help="Namespace planner will look at",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=PlannerDefaults.served_model_name,
        help="Model name that is being served (used for prefill queue name)",
    )
    parser.add_argument(
        "--no-operation",
        action="store_true",
        default=PlannerDefaults.no_operation,
        help="Do not make any adjustments, just observe the metrics",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=PlannerDefaults.log_dir,
        help="Tensorboard logging directory",
    )
    parser.add_argument(
        "--adjustment-interval",
        type=int,
        default=PlannerDefaults.adjustment_interval,
        help="Interval in seconds between scaling adjustments",
    )
    parser.add_argument(
        "--metric-pulling-interval",
        type=int,
        default=PlannerDefaults.metric_pulling_interval,
        help="Interval in seconds between metric pulls",
    )
    parser.add_argument(
        "--max-gpu-budget",
        type=int,
        default=PlannerDefaults.max_gpu_budget,
        help="Maximum number of GPUs to use",
    )
    parser.add_argument(
        "--min-endpoint",
        type=int,
        default=PlannerDefaults.min_endpoint,
        help="Minimum number of endpoints to keep for prefill/decode workers",
    )
    parser.add_argument(
        "--decode-kv-scale-up-threshold",
        type=float,
        default=PlannerDefaults.decode_kv_scale_up_threshold,
        help="KV cache utilization threshold to scale up decode workers",
    )
    parser.add_argument(
        "--decode-kv-scale-down-threshold",
        type=float,
        default=PlannerDefaults.decode_kv_scale_down_threshold,
        help="KV cache utilization threshold to scale down decode workers",
    )
    parser.add_argument(
        "--prefill-queue-scale-up-threshold",
        type=float,
        default=PlannerDefaults.prefill_queue_scale_up_threshold,
        help="Queue utilization threshold to scale up prefill workers",
    )
    parser.add_argument(
        "--prefill-queue-scale-down-threshold",
        type=float,
        default=PlannerDefaults.prefill_queue_scale_down_threshold,
        help="Queue utilization threshold to scale down prefill workers",
    )
    parser.add_argument(
        "--decode-engine-num-gpu",
        type=int,
        default=PlannerDefaults.decode_engine_num_gpu,
        help="Number of GPUs per decode engine",
    )
    parser.add_argument(
        "--prefill-engine-num-gpu",
        type=int,
        default=PlannerDefaults.prefill_engine_num_gpu,
        help="Number of GPUs per prefill engine",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default=PlannerDefaults.environment,
        help="Environment to run the planner in (local, kubernetes)",
    )
    args = parser.parse_args()
    asyncio.run(dynamo_worker()(start_planner)(args))
