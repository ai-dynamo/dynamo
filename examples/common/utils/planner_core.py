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
import math
import asyncio
import logging
import time

from utils.load_predictor import LOAD_PREDICTORS
from utils.prometheus import PrometheusAPIClient
from utils.perf_interpolation import PrefillInterpolator, DecodeInterpolator

from dynamo.planner import KubernetesConnector, LocalConnector
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

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
        
        self.prometheus_api_client = PrometheusAPIClient(args.prometheus_endpoint)

        self.num_req_predictor = LOAD_PREDICTORS[args.load_predictor](
            window_size=args.load_prediction_window_size,
            step_size=args.load_prediction_step_size,
        )
        self.isl_predictor = LOAD_PREDICTORS[args.load_predictor](
            window_size=args.load_prediction_window_size,
            step_size=args.load_prediction_step_size,
        )
        self.osl_predictor = LOAD_PREDICTORS[args.load_predictor](
            window_size=args.load_prediction_window_size,
            step_size=args.load_prediction_step_size,
        )

        self.prefill_interpolator = PrefillInterpolator(args.profile_results_dir)
        self.decode_interpolator = DecodeInterpolator(args.profile_results_dir)
        
        self.prefill_client = None
        self.workers_client = None
        self.p_endpoints = []
        self.d_endpoints = []

        self.last_adjustment_time = time.time()
        self.last_ttft = None
        self.last_itl = None
        self.last_num_req = None
        self.last_isl = None
        self.last_osl = None
        self.last_request_duration = None
        self.last_p_load = None
        self.last_d_load = None

        self.p_correction_factor = 1.0
        self.d_correction_factor = 1.0

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
            # TODO: use etcd events instead of pulling instance_ids
            p_endpoints = self.prefill_client.instance_ids()
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
            # TODO: use etcd events instead of pulling instance_ids
            d_endpoints = self.workers_client.instance_ids()
        except Exception as e:
            raise RuntimeError(f"Failed to get decode worker endpoints: {e}")
        return p_endpoints, d_endpoints

    def observe_metrics(self):
        self.last_ttft = self.prometheus_api_client.get_avg_time_to_first_token(f"{self.args.adjustment_interval}s")
        self.last_itl = self.prometheus_api_client.get_avg_inter_token_latency(f"{self.args.adjustment_interval}s")
        self.last_num_req = self.prometheus_api_client.get_avg_request_count(f"{self.args.adjustment_interval}s")
        self.last_request_duration = self.prometheus_api_client.get_avg_request_duration(f"{self.args.adjustment_interval}s")
        self.last_isl = self.prometheus_api_client.get_avg_input_sequence_tokens(f"{self.args.adjustment_interval}s")
        self.last_osl = self.prometheus_api_client.get_avg_output_sequence_tokens(f"{self.args.adjustment_interval}s")

        self.num_req_predictor.add_data_point(self.last_num_req)
        self.isl_predictor.add_data_point(self.last_isl)
        self.osl_predictor.add_data_point(self.last_osl)

    async def make_adjustments(self):
        try:
            self.p_endpoints, self.d_endpoints = await self.get_workers_info()
            logger.info(
                f"Number of prefill workers: {len(self.p_endpoints)}, number of decode workers: {len(self.d_endpoints)}"
            )
            
            # first correct the prediction correction factor
            # for TTFT, we expect the correction factor to be << 1 due to queuing delay
            expect_ttft = self.prefill_interpolator.interpolate_ttft(self.last_isl)
            self.p_correction_factor = self.last_ttft / expect_ttft
            # for ITL, we expect the correction factor to be close to 1
            expect_itl = self.decode_interpolator.interpolate_itl(
                concurrency=self.last_num_req / len(self.d_endpoints) * self.last_request_duration / self.args.adjustment_interval,
                context_length=self.last_isl + self.last_osl / 2
            )
            self.d_correction_factor = self.last_itl / expect_itl
            logger.info(f"Correction factors: TTFT: {self.p_correction_factor}, ITL: {self.d_correction_factor}")

            # predict the next load
            next_num_req = self.num_req_predictor.predict_next()
            next_isl = self.isl_predictor.predict_next()
            next_osl = self.osl_predictor.predict_next()
            logger.info(f"Predicted load: num_req={next_num_req}, isl={next_isl}, osl={next_osl}")

            # compute how many replicas are needed for prefill
            # here we assume the prefill bias is purely due to request queueing
            # and we increase the number of prefill replicas linearly to account for the queueing delay
            pred_prefill_load_per_gpu = next_num_req * next_isl / self.args.adjustment_interval * min(1, self.p_correction_factor)
            next_num_p = math.ceil(pred_prefill_load_per_gpu / self.prefill_interpolator.interpolate_thpt_per_gpu(next_isl) / self.args.prefill_engine_num_gpu)

            # compute how many replicas are needed for decode
            # 1. apply d_correction_factor to the ITL SLA
            corrected_itl = self.args.itl / self.d_correction_factor
            # 2. reversely find out what is best throughput/gpu that can achieve corrected_itl under the predicted context length
            pred_decode_thpt_per_gpu = self.decode_interpolator.find_best_throughput_per_gpu(
                itl=corrected_itl,
                context_length=next_isl + next_osl / 2
            )
            # 3. compute number of decode replicas needed
            next_num_d = math.ceil(next_num_req * next_osl / self.args.adjustment_interval / pred_decode_thpt_per_gpu / self.args.decode_engine_num_gpu)

            # correct num_p and num_d based on the gpu budget
            next_num_p = max(next_num_p, self.args.min_endpoint)
            next_num_d = max(next_num_d, self.args.min_endpoint)
            logger.info(f"Predicted number of engine replicas: prefill={next_num_p}, decode={next_num_d}")

            if next_num_p * self.args.prefill_engine_num_gpu + next_num_d * self.args.decode_engine_num_gpu > self.args.max_gpu_budget:
                scale = self.args.max_gpu_budget / (next_num_p * self.args.prefill_engine_num_gpu + next_num_d * self.args.decode_engine_num_gpu)
                next_num_p = round(next_num_p * scale)
                next_num_d = (self.args.max_gpu_budget - next_num_p * self.args.prefill_engine_num_gpu) // self.args.decode_engine_num_gpu
                logger.warning(f"Total number of GPUs required ({next_num_p * self.args.prefill_engine_num_gpu + next_num_d * self.args.decode_engine_num_gpu}) exceeds the max GPU budget ({self.args.max_gpu_budget}), scaling down to {next_num_p} prefill and {next_num_d} decode replicas")

        except Exception as e:
            logger.error(f"Failed to make adjustments: {e}")
            return
        
        if not self.args.no_operation:
            # TODO: scale up/down the number of prefill/decode non-blockingly
            pass

    async def run(self):
        """Main loop for the planner"""

        self.last_adjustment_time = time.time()

        while True:
            current_time = time.time()

            if current_time - self.last_adjustment_time >= self.args.adjustment_interval:
                self.observe_metrics()
                await self.make_adjustments()

            await asyncio.sleep(self.args.metric_pulling_interval / 10)

async def start_sla_planner(runtime: DistributedRuntime, args: argparse.Namespace):
    planner = Planner(runtime, args)
    await planner.run()
