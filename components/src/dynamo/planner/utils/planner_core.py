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

from dynamo.planner import KubernetesConnector, VirtualConnector
from dynamo.planner.defaults import WORKER_COMPONENT_NAMES, SLAPlannerDefaults
from dynamo.planner.utils.load_predictor import LOAD_PREDICTORS
from dynamo.planner.utils.perf_interpolation import (
    DecodeInterpolator,
    PrefillInterpolator,
)
from dynamo.planner.utils.prometheus import PrometheusAPIClient
from dynamo.planner.utils.trace_data_extractor import extract_metrics_from_mooncake
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
    def __init__(
        self,
        runtime: Optional[DistributedRuntime],
        args: argparse.Namespace,
        dryrun: bool = False,
    ):
        self.args = args
        self.dryrun = dryrun

        if not self.dryrun:
            self.runtime = runtime
            self.namespace = args.namespace

            if not args.no_operation:
                if args.environment == "kubernetes":
                    self.connector = KubernetesConnector(self.namespace)
                elif args.environment == "virtual":
                    self.connector = VirtualConnector(
                        runtime, self.namespace, args.backend
                    )
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

        if not self.dryrun:
            self.prefill_client = None
            self.workers_client = None
            self.p_endpoints = []  # type: ignore
            self.d_endpoints = []  # type: ignore

            self.last_adjustment_time = time.time()
            self.last_metrics = Metrics()

            self.prometheus_port = args.prometheus_port

            # Initialize Prometheus metrics
            # TODO: use proper naming
            self.num_p_workers_gauge = Gauge(
                "num_p_workers", "Number of prefill workers"
            )
            self.num_d_workers_gauge = Gauge(
                "num_d_workers", "Number of decode workers"
            )

            # Start Prometheus HTTP server if port is specified
            if self.prometheus_port != 0:
                try:
                    start_http_server(self.prometheus_port)
                    logger.info(
                        f"Started Prometheus metrics server on port {self.prometheus_port}"
                    )
                except Exception as e:
                    logger.error(f"Failed to start Prometheus metrics server: {e}")

        self.p_correction_factor = 1.0
        self.d_correction_factor = 1.0
        if self.dryrun:
            self.no_correction = True
        else:
            self.no_correction = args.no_correction

    async def _async_init(self):
        """Async initialization for components that need it"""
        if (
            not self.dryrun
            and hasattr(self, "connector")
            and hasattr(self.connector, "_async_init")
        ):
            await self.connector._async_init()

    async def get_workers_info(self):
        if self.runtime is None:
            raise RuntimeError("Runtime is not initialized")

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

    def predict_load(self):
        try:
            # predict the next load
            next_num_req = self.num_req_predictor.predict_next()
            next_isl = self.isl_predictor.predict_next()
            next_osl = self.osl_predictor.predict_next()
            logger.info(
                f"Predicted load: num_req={next_num_req:.2f}, isl={next_isl:.2f}, osl={next_osl:.2f}"
            )
            return next_num_req, next_isl, next_osl
        except Exception as e:
            logger.error(f"Failed to predict load: {e}")
            return None, None, None

    def dryrun_observe_metrics(self, num_req: int, isl_avg: float, osl_avg: float):
        self.num_req_predictor.add_data_point(num_req)
        self.isl_predictor.add_data_point(isl_avg)
        self.osl_predictor.add_data_point(osl_avg)

    def _compute_replica_requirements(
        self, next_num_req: float, next_isl: float, next_osl: float
    ) -> tuple[int, int]:
        """Compute the number of prefill and decode replicas needed based on predicted load.

        Args:
            next_num_req: Predicted number of requests
            next_isl: Predicted input sequence length
            next_osl: Predicted output sequence length

        Returns:
            tuple[int, int]: Number of prefill and decode replicas needed
        """
        # compute how many replicas are needed for prefill
        # here we assume the prefill bias is purely due to request queueing
        # and we increase the number of prefill replicas linearly to account for the queueing delay
        pred_prefill_throughput = (
            next_num_req
            * next_isl
            / self.args.adjustment_interval
            * min(1, self.p_correction_factor)
        )
        next_num_p = math.ceil(
            pred_prefill_throughput
            / self.prefill_interpolator.interpolate_thpt_per_gpu(next_isl)
            / self.args.prefill_engine_num_gpu
        )

        logger.info(
            f"Prefill calculation: {pred_prefill_throughput:.2f}(p_thpt) / "
            f"{self.prefill_interpolator.interpolate_thpt_per_gpu(next_isl) * self.args.prefill_engine_num_gpu:.2f}(p_engine_cap) = "
            f"{next_num_p}(num_p)"
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
        pred_decode_throughput = next_num_req * next_osl / self.args.adjustment_interval
        next_num_d = math.ceil(
            pred_decode_throughput
            / pred_decode_thpt_per_gpu
            / self.args.decode_engine_num_gpu
        )

        logger.info(
            f"Decode calculation: {pred_decode_throughput:.2f}(d_thpt) / "
            f"{pred_decode_thpt_per_gpu * self.args.decode_engine_num_gpu:.2f}(d_engine_cap) = "
            f"{next_num_d}(num_d)"
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

        return next_num_p, next_num_d

    async def make_adjustments(self):
        # Skip adjustment if no traffic
        if not self.last_metrics.is_valid():
            logger.info(
                "Metrics contain None or NaN values (no active requests), skipping adjustment"
            )
            return

        if not self.no_correction:
            try:
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

        next_num_req, next_isl, next_osl = self.predict_load()

        if next_num_req is not None and next_isl is not None and next_osl is not None:
            try:
                next_num_p, next_num_d = self._compute_replica_requirements(
                    next_num_req, next_isl, next_osl
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

    def dryrun_run(self):
        """Run planner in dry-run mode with dataset"""
        metrics = extract_metrics_from_mooncake(
            self.args.dataset, self.args.adjustment_interval
        )

        def compute_safe_p_thpt(num_p: int, isl: float, ttft: float):
            """safe throughput is maximum throughput that the engine can handle given the TTFT SLA"""
            actual_ttft = self.prefill_interpolator.interpolate_ttft(isl)
            if actual_ttft > ttft:
                return 0
            else:
                return num_p * self.prefill_interpolator.interpolate_thpt_per_gpu(isl)

        def compute_safe_d_thpt(num_d: int, isl: float, osl: float, itl: float):
            """safe throughput is maximum throughput that the engine can handle given the ITL SLA"""
            (
                pred_decode_thpt_per_gpu,
                actual_itl,
                _,
            ) = self.decode_interpolator.find_best_throughput_per_gpu(
                itl=itl, context_length=isl + osl / 2
            )
            if actual_itl > itl:
                return 0
            else:
                return num_d * pred_decode_thpt_per_gpu

        time = [0]
        rr = [metrics[0]["request_count"]]
        est_rr = [metrics[0]["request_count"]]
        isl = [metrics[0]["avg_isl"]]
        est_isl = [metrics[0]["avg_isl"]]
        osl = [metrics[0]["avg_osl"]]
        est_osl = [metrics[0]["avg_osl"]]
        num_p = [self.args.start_num_p]
        p_thpt = [metrics[0]["request_count"] * metrics[0]["avg_isl"]]
        safe_p_thpt = [
            compute_safe_p_thpt(
                self.args.start_num_p, metrics[0]["avg_isl"], self.args.ttft
            )
            * self.args.adjustment_interval
        ]
        num_d = [self.args.start_num_d]
        d_thpt = [metrics[0]["request_count"] * metrics[0]["avg_osl"]]
        safe_d_thpt = [
            compute_safe_d_thpt(
                self.args.start_num_d,
                metrics[0]["avg_isl"],
                metrics[0]["avg_osl"],
                self.args.itl,
            )
            * self.args.adjustment_interval
        ]
        self.dryrun_observe_metrics(
            metrics[0]["request_count"], metrics[0]["avg_isl"], metrics[0]["avg_osl"]
        )

        for metric in metrics[1:]:
            # update time
            time.append(time[-1] + self.args.adjustment_interval)

            # load prediction
            _est_rr, _est_isl, _est_osl = self.predict_load()
            est_rr.append(_est_rr)
            est_isl.append(_est_isl)
            est_osl.append(_est_osl)

            # compute num_p and num_d
            _num_p, _num_d = self._compute_replica_requirements(
                _est_rr, _est_isl, _est_osl
            )
            num_p.append(_num_p)
            num_d.append(_num_d)

            # update load predictor
            self.dryrun_observe_metrics(
                metric["request_count"], metric["avg_isl"], metric["avg_osl"]
            )

            # fill in ground truth
            rr.append(metric["request_count"])
            isl.append(metric["avg_isl"])
            osl.append(metric["avg_osl"])

            p_thpt.append(rr[-1] * isl[-1])
            d_thpt.append(rr[-1] * osl[-1])

            safe_p_thpt.append(
                compute_safe_p_thpt(num_p[-1], isl[-1], self.args.ttft)
                * self.args.adjustment_interval
            )
            safe_d_thpt.append(
                compute_safe_d_thpt(num_d[-1], isl[-1], osl[-1], self.args.itl)
                * self.args.adjustment_interval
            )

        # plot the results
        from dynamo.planner.utils.dryrun_plot_utils import create_dryrun_plot

        create_dryrun_plot(
            time=time,
            rr=rr,
            est_rr=est_rr,
            isl=isl,
            est_isl=est_isl,
            osl=osl,
            est_osl=est_osl,
            num_p=num_p,
            p_thpt=p_thpt,
            safe_p_thpt=safe_p_thpt,
            num_d=num_d,
            d_thpt=d_thpt,
            safe_d_thpt=safe_d_thpt,
            output_path=self.args.output_plot,
        )


async def start_sla_planner(runtime: DistributedRuntime, args: argparse.Namespace):
    planner = Planner(runtime, args)
    await planner._async_init()
    await planner.run()
