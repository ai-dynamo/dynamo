# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from prometheus_client import Gauge, start_http_server

from dynamo.planner import KubernetesConnector
from dynamo.planner.defaults import WORKER_COMPONENT_NAMES, SLAPlannerDefaults
from dynamo.planner.utils.load_predictor import LOAD_PREDICTORS
from dynamo.planner.utils.perf_interpolation import (
    DecodeInterpolator,
    PrefillInterpolator,
)
from dynamo.planner.utils.prometheus import PrometheusAPIClient
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    ttft: Optional[float] = None
    itl: Optional[float] = None
    num_req: Optional[float] = None
    isl: Optional[float] = None
    osl: Optional[float] = None
    request_duration: Optional[float] = None
    p_load: Optional[float] = None
    d_load: Optional[float] = None

    def is_valid(self) -> bool:
        """Check if all metrics are valid (not None and not NaN)."""
        return (
            self.ttft is not None
            and self.itl is not None
            and self.isl is not None
            and self.osl is not None
            and not math.isnan(self.ttft)
            and not math.isnan(self.itl)
            and not math.isnan(self.isl)
            and not math.isnan(self.osl)
        )


class Planner:
    def __init__(self, runtime: DistributedRuntime, args: argparse.Namespace):
        self.runtime = runtime
        self.args = args
        self.namespace = SLAPlannerDefaults.namespace

        if not args.no_operation:
            if args.environment == "kubernetes":
                self.connector = KubernetesConnector(self.namespace)
            else:
                raise ValueError(f"Invalid environment: {args.environment}")

        self.prometheus_api_client = PrometheusAPIClient(
            SLAPlannerDefaults.prometheus_endpoint
        )

        self.num_req_predictor = LOAD_PREDICTORS[args.load_predictor](
            window_size=args.load_prediction_window_size,
        )
        self.isl_predictor = LOAD_PREDICTORS[args.load_predictor](
            window_size=args.load_prediction_window_size,
        )
        self.osl_predictor = LOAD_PREDICTORS[args.load_predictor](
            window_size=args.load_prediction_window_size,
        )

        self.prefill_interpolator = PrefillInterpolator(args.profile_results_dir)
        self.decode_interpolator = DecodeInterpolator(args.profile_results_dir)

        self.prefill_client = None
        self.workers_client = None
        self.p_endpoints = []  # type: ignore
        self.d_endpoints = []  # type: ignore

        self.last_adjustment_time = time.time()
        self.last_metrics = Metrics()

        self.p_correction_factor = 1.0
        self.d_correction_factor = 1.0
        self.no_correction = args.no_correction

        self.prometheus_port = args.prometheus_port

        # Initialize Prometheus metrics
        # TODO: use proper naming
        self.num_p_workers_gauge = Gauge("num_p_workers", "Number of prefill workers")
        self.num_d_workers_gauge = Gauge("num_d_workers", "Number of decode workers")

        # Start Prometheus HTTP server if port is specified
        if self.prometheus_port != 0:
            try:
                start_http_server(self.prometheus_port)
                logger.info(
                    f"Started Prometheus metrics server on port {self.prometheus_port}"
                )
            except Exception as e:
                logger.error(f"Failed to start Prometheus metrics server: {e}")

    async def get_workers_info(self):
        try:
            if self.prefill_client is None:
                self.prefill_client = (
                    await self.runtime.namespace(self.namespace)
                    .component(
                        WORKER_COMPONENT_NAMES[
                            self.args.backend
                        ].prefill_worker_component_name
                    )
                    .endpoint(
                        WORKER_COMPONENT_NAMES[
                            self.args.backend
                        ].prefill_worker_endpoint
                    )
                    .client()
                )
                # TODO: remove this sleep after rust client() is blocking until watching state
                await asyncio.sleep(0.1)
            # TODO: use etcd events instead of pulling instance_ids
            p_endpoints = self.prefill_client.instance_ids()  # type: ignore
        except Exception:
            p_endpoints = []
            logger.warning(
                "No prefill workers found, aggregated mode is not supported yet"
            )
        try:
            if self.workers_client is None:
                self.workers_client = (
                    await self.runtime.namespace(self.namespace)
                    .component(
                        WORKER_COMPONENT_NAMES[
                            self.args.backend
                        ].decode_worker_component_name
                    )
                    .endpoint(
                        WORKER_COMPONENT_NAMES[self.args.backend].decode_worker_endpoint
                    )
                    .client()
                )
                # TODO: remove this sleep after rust client() is blocking until watching state
                await asyncio.sleep(0.1)
            # TODO: use etcd events instead of pulling instance_ids
            d_endpoints = self.workers_client.instance_ids()  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to get decode worker endpoints: {e}")
        return p_endpoints, d_endpoints

    async def observe_metrics(self):
        self.p_endpoints, self.d_endpoints = await self.get_workers_info()
        logger.debug(
            f"Number of prefill workers: {len(self.p_endpoints)}, number of decode workers: {len(self.d_endpoints)}"
        )

        # Update Prometheus metrics if server is running
        if self.prometheus_port != 0:
            self.num_p_workers_gauge.set(len(self.p_endpoints))
            self.num_d_workers_gauge.set(len(self.d_endpoints))

        self.last_metrics.ttft = self.prometheus_api_client.get_avg_time_to_first_token(
            f"{self.args.adjustment_interval}s"
        )
        self.last_metrics.itl = self.prometheus_api_client.get_avg_inter_token_latency(
            f"{self.args.adjustment_interval}s"
        )
        self.last_metrics.num_req = self.prometheus_api_client.get_avg_request_count(
            f"{self.args.adjustment_interval}s"
        )
        self.last_metrics.request_duration = (
            self.prometheus_api_client.get_avg_request_duration(
                f"{self.args.adjustment_interval}s"
            )
        )
        self.last_metrics.isl = (
            self.prometheus_api_client.get_avg_input_sequence_tokens(
                f"{self.args.adjustment_interval}s"
            )
        )
        self.last_metrics.osl = (
            self.prometheus_api_client.get_avg_output_sequence_tokens(
                f"{self.args.adjustment_interval}s"
            )
        )

        logger.info(
            f"Observed num_req: {self.last_metrics.num_req:.2f} isl: {self.last_metrics.isl:.2f} osl: {self.last_metrics.osl:.2f}"
        )
        logger.info(
            f"Observed ttft: {self.last_metrics.ttft:.3f}s itl: {self.last_metrics.itl:.3f}s"
        )

        self.num_req_predictor.add_data_point(self.last_metrics.num_req)
        self.isl_predictor.add_data_point(self.last_metrics.isl)
        self.osl_predictor.add_data_point(self.last_metrics.osl)

    async def make_adjustments(self):
        if not self.no_correction:
            try:
                # Skip adjustment if no traffic
                if not self.last_metrics.is_valid():
                    logger.info(
                        "Metrics contain None or NaN values (no active requests), skipping adjustment"
                    )
                    return

                self.p_endpoints, self.d_endpoints = await self.get_workers_info()
                logger.info(
                    f"Number of prefill workers: {len(self.p_endpoints)}, number of decode workers: {len(self.d_endpoints)}"
                )

                # first correct the prediction correction factor
                # for TTFT, we expect the correction factor to be << 1 due to queuing delay
                expect_ttft = self.prefill_interpolator.interpolate_ttft(
                    self.last_metrics.isl
                )
                self.p_correction_factor = self.last_metrics.ttft / expect_ttft
                # for ITL, we expect the correction factor to be close to 1
                expect_itl = self.decode_interpolator.interpolate_itl(
                    concurrency=self.last_metrics.num_req  # type: ignore
                    / len(self.d_endpoints)
                    * self.last_metrics.request_duration  # type: ignore
                    / self.args.adjustment_interval,
                    context_length=self.last_metrics.isl + self.last_metrics.osl / 2,  # type: ignore
                )
                self.d_correction_factor = self.last_metrics.itl / expect_itl
                logger.info(
                    f"Correction factors: TTFT: {self.p_correction_factor:.3f}, ITL: {self.d_correction_factor:.3f}"
                )
            except Exception as e:
                logger.error(f"Failed to correct prediction factors: {e}")
                return

        try:
            # predict the next load
            next_num_req = self.num_req_predictor.predict_next()
            next_isl = self.isl_predictor.predict_next()
            next_osl = self.osl_predictor.predict_next()
            logger.info(
                f"Predicted load: num_req={next_num_req:.2f}, isl={next_isl:.2f}, osl={next_osl:.2f}"
            )
        except Exception as e:
            logger.error(f"Failed to predict load: {e}")
            return

        try:
            # compute how many replicas are needed for prefill
            # here we assume the prefill bias is purely due to request queueing
            # and we increase the number of prefill replicas linearly to account for the queueing delay
            pred_prefill_load_per_gpu = (
                next_num_req
                * next_isl
                / self.args.adjustment_interval
                * min(1, self.p_correction_factor)
            )
            next_num_p = math.ceil(
                pred_prefill_load_per_gpu
                / self.prefill_interpolator.interpolate_thpt_per_gpu(next_isl)
                / self.args.prefill_engine_num_gpu
            )

            # compute how many replicas are needed for decode
            # 1. apply d_correction_factor to the ITL SLA
            # Prevent divide by zero when d_correction_factor is 0 (no metrics yet)
            if self.d_correction_factor <= 0:
                logger.warning(
                    f"d_correction_factor is {self.d_correction_factor}, using default value of 1.0"
                )
                corrected_itl = self.args.itl
            else:
                corrected_itl = self.args.itl / self.d_correction_factor
            # 2. reversely find out what is best throughput/gpu that can achieve corrected_itl under the predicted context length
            (
                pred_decode_thpt_per_gpu,
                _,
                _,
            ) = self.decode_interpolator.find_best_throughput_per_gpu(
                itl=corrected_itl, context_length=next_isl + next_osl / 2
            )
            # 3. compute number of decode replicas needed
            next_num_d = math.ceil(
                next_num_req
                * next_osl
                / self.args.adjustment_interval
                / pred_decode_thpt_per_gpu
                / self.args.decode_engine_num_gpu
            )

            # correct num_p and num_d based on the gpu budget
            next_num_p = max(next_num_p, self.args.min_endpoint)
            next_num_d = max(next_num_d, self.args.min_endpoint)
            logger.info(
                f"Predicted number of engine replicas: prefill={next_num_p}, decode={next_num_d}"
            )

            total_gpu_required = (
                next_num_p * self.args.prefill_engine_num_gpu
                + next_num_d * self.args.decode_engine_num_gpu
            )
            if total_gpu_required > self.args.max_gpu_budget:
                scale = self.args.max_gpu_budget / total_gpu_required
                next_num_p = max(self.args.min_endpoint, round(next_num_p * scale))
                next_num_d = max(
                    self.args.min_endpoint,
                    round(
                        (
                            self.args.max_gpu_budget
                            - next_num_p * self.args.prefill_engine_num_gpu
                        )
                        / self.args.decode_engine_num_gpu
                    ),
                )
                logger.warning(
                    f"Total number of GPUs required ({total_gpu_required}) exceeds the max GPU budget ({self.args.max_gpu_budget}), scaling down to {next_num_p} prefill and {next_num_d} decode replicas"
                )
        except Exception as e:
            logger.error(f"Failed to compute number of replicas: {e}")
            return

        if not self.args.no_operation:
            target_replicas = {
                WORKER_COMPONENT_NAMES[
                    self.args.backend
                ].prefill_worker_k8s_name: next_num_p,
                WORKER_COMPONENT_NAMES[
                    self.args.backend
                ].decode_worker_k8s_name: next_num_d,
            }
            await self.connector.set_component_replicas(target_replicas, blocking=False)

    async def run(self):
        """Main loop for the planner"""

        self.last_adjustment_time = time.time()

        while True:
            current_time = time.time()

            if (
                current_time - self.last_adjustment_time
                >= self.args.adjustment_interval
            ):
                self.last_adjustment_time = time.time()
                logger.info("New adjustment interval started!")
                await self.observe_metrics()
                await self.make_adjustments()

            # sleep for a while to avoid busy-waiting but not too long to miss the next adjustment
            await asyncio.sleep(self.args.adjustment_interval / 10)


async def start_sla_planner(runtime: DistributedRuntime, args: argparse.Namespace):
    planner = Planner(runtime, args)
    await planner.run()
