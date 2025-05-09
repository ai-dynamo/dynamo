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
import random
import traceback
from argparse import Namespace
from typing import AsyncIterator

from common.protocol import Tokens
from components.worker import TensorRTLLMWorker

from dynamo.llm import AggregatedMetrics, KvIndexer, KvMetricsAggregator, OverlapScores
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)

WorkerId = str


def parse_args(service_name, prefix) -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-workers",
        type=int,
        default=1,
        help="Minimum number of workers required before proceeding",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model that is being served",
    )
    # TODO: Read block size
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="KV block size",
    )
    parser.add_argument(
        "--custom-router",
        type=bool,
        default=False,
        help="Whether to use custom router or not",
    )
    config = ServiceConfig.get_instance()
    config_args = config.as_args(service_name, prefix=prefix)
    args = parser.parse_args(config_args)
    return args


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class Router:
    worker = depends(TensorRTLLMWorker)

    def __init__(self):
        logger.info("Initializing KV router.")
        class_name = self.__class__.__name__
        self.args = parse_args(class_name, "")

    @async_on_start
    async def async_init(self):
        self.runtime = dynamo_context["runtime"]
        self.workers_client = (
            await self.runtime.namespace("dynamo")
            .component("TensorRTLLMWorker")
            .endpoint("generate")
            .client()
        )
        while len(self.workers_client.endpoint_ids()) < self.args.min_workers:
            logger.info(
                f"Waiting for more workers to be ready.\n"
                f" Current: {len(self.workers_client.endpoint_ids())},"
                f" Required: {self.args.min_workers}"
            )
            await asyncio.sleep(30)

        kv_listener = self.runtime.namespace("dynamo").component("TensorRTLLMWorker")
        await kv_listener.create_service()
        self.indexer = KvIndexer(kv_listener, self.args.block_size)
        self.metrics_aggregator = KvMetricsAggregator(kv_listener)
        logger.info("KV Router initialized")

    def _cost_function(
        self,
        scores: OverlapScores | None,
        metrics: AggregatedMetrics | None,
        token_length: int,
    ):
        worker_scores = {}
        if scores:
            for worker_id, score in scores.scores.items():
                # score is number of matching blocks we multiply by block_size to get tokens
                # and compare to token_length. The larger the cache hit the better
                worker_scores[worker_id] = (
                    score * self.indexer.block_size() / token_length
                )

        logger.debug(f"Worker scores: {worker_scores}")
        worker_metrics = {}
        # pull metrics for each worker
        max_waiting = 0.0
        if metrics:
            for endpoint in metrics.endpoints:
                worker_id = endpoint.worker_id
            worker_metrics[worker_id] = {
                "gpu_cache_usage_perc": endpoint.gpu_cache_usage_perc
                if hasattr(endpoint, "gpu_cache_usage_perc")
                else 0.0,
                "num_requests_waiting": endpoint.num_requests_waiting
                if hasattr(endpoint, "num_requests_waiting")
                else 0.0,
                "gpu_prefix_cache_hit_rate": endpoint.gpu_prefix_cache_hit_rate
                if hasattr(endpoint, "gpu_prefix_cache_hit_rate")
                else 0.0,
            }
            max_waiting = max(
                max_waiting, worker_metrics[worker_id]["num_requests_waiting"]
            )
        logger.debug(f"Worker metrics: {worker_metrics}")

        # Get all worker IDs from the client. This is needed because scores / metrics may not have values for all workers
        # and we want all workers to be considered in the logit calculation
        worker_ids = self.workers_client.endpoint_ids()

        worker_logits = {}
        for worker_id in worker_ids:
            # Use default values if worker not in scores or metrics
            score = worker_scores.get(worker_id, 0.0)
            metrics_dict = worker_metrics.get(
                worker_id,
                {
                    "gpu_cache_usage_perc": 0.0,
                    "num_requests_waiting": 0.0,
                    "gpu_prefix_cache_hit_rate": 0.0,
                },
            )

            normalized_waiting = (
                metrics_dict["num_requests_waiting"] / max_waiting
                if max_waiting > 0
                else 0.0
            )

            # Have 1 metric that weights towards cache hit
            # 2 metrics that penalize overloaded worker and queuing
            worker_logits[worker_id] = (
                2 * score - metrics_dict["gpu_cache_usage_perc"] - normalized_waiting
            )
            logger.debug(
                f"Formula for {worker_id}: {worker_logits[worker_id]:.3f} = 2.0 * {score:.3f} - {metrics_dict['gpu_cache_usage_perc']:.3f} - {normalized_waiting:.3f}"
            )

        if not worker_logits or all(logit == 0 for logit in worker_logits.values()):
            return ""

        # Select the worker with the highest logit
        if worker_logits:
            max_logit = max(worker_logits.values())
            best_workers = [
                wid for wid, logit in worker_logits.items() if logit == max_logit
            ]
            best_worker_id = random.choice(best_workers)
        else:
            best_worker_id = ""

        # Log the metrics for the selected worker
        if best_worker_id:
            logger.debug(
                f"Selected worker: {best_worker_id}, logit: {worker_logits[best_worker_id]:.3f}"
            )
            logger.debug(
                f"Score: {scores.scores.get(best_worker_id, 0.0) if scores else 0.0:.3f}"
            )

            metrics_dict = worker_metrics.get(best_worker_id, {})
            logger.debug(
                f"GPU Cache Hit Rate: {metrics_dict.get('gpu_prefix_cache_hit_rate', 0.0):.3f}"
            )
            logger.debug(
                f"GPU Cache Usage: {metrics_dict.get('gpu_cache_usage_perc', 0.0):.3f}"
            )
            logger.debug(
                f"Requests Waiting: {metrics_dict.get('num_requests_waiting', 0.0) / max_waiting if max_waiting > 0 else 0.0:.3f}"
            )

        return best_worker_id, worker_scores.get(best_worker_id, 0.0)

    @dynamo_endpoint()
    async def generate(self, request: Tokens) -> AsyncIterator[WorkerId]:
        if self.indexer is None or self.metrics_aggregator is None:
            yield "_0.0"

        lora_id = 0
        worker_id = ""
        try:
            scores = await self.indexer.find_matches_for_request(
                request.tokens, lora_id
            )
            token_length = len(request.tokens)
            metrics = await self.metrics_aggregator.get_metrics()
            schedule_result = self._cost_function(scores, metrics, token_length)
        except Exception:
            schedule_result = ""
            logger.warning(f"Error during worker selection: {traceback.format_exc()}")

        if schedule_result == "":
            worker_id = ""
            prefix_hit_rate = 0.0
        else:
            worker_id, prefix_hit_rate = schedule_result

        yield f"{worker_id}_{prefix_hit_rate}"
