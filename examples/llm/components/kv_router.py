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
import logging
import random
from argparse import Namespace
from typing import AsyncIterator, Tuple

from components.worker import VllmWorker
from utils.check_worker import check_required_workers
from utils.protocol import Tokens
from utils.vllm import RouterType

from dynamo.llm import AggregatedMetrics, KvIndexer, KvMetricsAggregator, OverlapScores
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

WorkerId = str
fallback_msg = "Will fallback to random routing."

logger = logging.getLogger(__name__)


def parse_args(service_name, prefix) -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-workers",
        type=int,
        default=1,
        help="Minimum number of workers required before proceeding",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model that is being served",
    )
    # TODO: Read block size
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="KV block size",
    )
    parser.add_argument(
        "--custom-router",
        type=bool,
        default=False,
        help="Whether to use custom router or not",
    )
    parser.add_argument(
        "--router",
        type=str,
        default="kv",
        help="The router type",
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
    """
    Request handler for the generate endpoint
    """

    worker = depends(VllmWorker)

    def __init__(self):
        logger.info("Initializing Custom Router")
        self.args = parse_args(self.__class__.__name__, "")

        self.default_metrics = {
            "gpu_cache_usage_perc": 0.0,
            "num_requests_waiting": 0.0,
            "gpu_prefix_cache_hit_rate": 0.0,
        }

    @async_on_start
    async def async_init(self):
        self.runtime = dynamo_context["runtime"]
        self.workers_client = (
            await self.runtime.namespace("dynamo")
            .component("VllmWorker")
            .endpoint("generate")
            .client()
        )

        self.router_type = self.args.router

        await check_required_workers(self.workers_client, self.args.min_workers)

        kv_listener = self.runtime.namespace("dynamo").component("VllmWorker")
        await kv_listener.create_service()
        if self.router_type == RouterType.KV:
            self.indexer = KvIndexer(kv_listener, self.args.block_size)
        self.metrics_aggregator = KvMetricsAggregator(kv_listener)
        logger.info("KV Router initialized")

    def _cost_function(
        self,
        scores: OverlapScores | None,
        metrics: AggregatedMetrics | None,
        token_length: int,
    ):
        """The cost function for deciding the best worker to route a request to.
        If there are multiple workers sharing the same optimal cost, then
        one of them is randomly selected.

        Args:
            scores (OverlapScores | None): The number of matching blocks between
                the request and the prefix cache of each worker.
            metrics (AggregatedMetrics | None): Several worker metrics polled
                by the `KvMetricsAggregator`, currently including the
                GPU cache usage, number of waiting requests, and the
                GPU prefix cache hit rate.
            token_length (int): The number of tokens in the request.

        Returns:
            (str, float): The best worker id and the corresponding score.
        """

        worker_scores = {}
        if scores:
            for worker_id, score in scores.scores.items():
                # score is number of matching blocks we multiply by block_size to get tokens
                # and compare to token_length. The larger the cache hit the better
                worker_scores[worker_id] = (
                    score * self.indexer.block_size() / token_length
                )
        else:
            logger.warning("Cannot get KV scores")

        worker_metrics = {}
        max_waiting = 0.0
        if metrics:
            for endpoint in metrics.endpoints:
                worker_id = endpoint.worker_id
                worker_metrics[worker_id] = {
                    key: getattr(endpoint, key, self.default_metrics[key])
                    for key in self.default_metrics.keys()
                }
                max_waiting = max(
                    max_waiting, worker_metrics[worker_id]["num_requests_waiting"]
                )
        else:
            logger.warning("Cannot get metrics")

        # Get all worker IDs from the client. This is needed because scores / metrics may not have values for all workers
        # and we want all workers to be considered in the logit calculation
        worker_ids = self.workers_client.endpoint_ids()

        worker_logits = {}
        for worker_id in worker_ids:
            # Use default values if worker not in scores or metrics
            score = worker_scores.get(worker_id, 0.0)
            metrics_dict = worker_metrics.get(worker_id, self.default_metrics)
            gpu_cache_usage = metrics_dict["gpu_cache_usage_perc"]

            normalized_waiting = (
                metrics_dict["num_requests_waiting"] / max_waiting
                if max_waiting > 0
                else 0.0
            )

            # Have 1 metric that weights towards cache hit
            # 2 metrics that penalize overloaded worker and queuing
            worker_logits[worker_id] = 2 * score - gpu_cache_usage - normalized_waiting
            logger.info(
                f"Formula for {worker_id}: {worker_logits[worker_id]:.3f} = 2.0 * {score:.3f} - {gpu_cache_usage:.3f} - {normalized_waiting:.3f}"
            )

        if not worker_logits or not any(worker_logits.values()):
            logger.warning(f"All worker logits are zero. {fallback_msg}.")
            return "", 0.0

        # Select the worker with the highest logit
        max_logit = max(worker_logits.values())
        best_workers = [
            wid for wid, logit in worker_logits.items() if logit == max_logit
        ]
        best_worker_id = random.choice(best_workers)

        # Log the metrics for the selected worker
        if best_worker_id:
            metrics_dict = worker_metrics.get(best_worker_id, self.default_metrics)

            # Create log messages
            log_messages = [
                f"Selected worker: {best_worker_id}, logit: {worker_logits[best_worker_id]:.3f}",
                f"Score: {scores.scores.get(best_worker_id, 0.0) if scores else 0.0:.3f}",
                f"GPU Cache Hit Rate: {metrics_dict['gpu_prefix_cache_hit_rate']:.3f}",
                f"GPU Cache Usage: {metrics_dict['gpu_cache_usage_perc']:.3f}",
                f"Requests Waiting: {metrics_dict['num_requests_waiting']}",
            ]

            # Log to vllm_logger
            for message in log_messages:
                logger.info(message)

        return best_worker_id, worker_scores.get(best_worker_id, 0.0)

    def _get_underloaded_worker(self, metrics: AggregatedMetrics | None):
        if not metrics:
            logger.warning(f"Cannot get metrics. {fallback_msg}")
            return "", 0.0

        kv_load = {
            endpoint.worker_id: getattr(endpoint, "gpu_cache_usage_perc", 0.0)
            for endpoint in metrics.endpoints
        }

        if not kv_load or not any(kv_load.values()):
            logger.warning(f"All KV loads are zero. {fallback_msg}")
            return "", 0.0

        min_load = min(kv_load.values())
        min_load_workers = [
            worker_id for worker_id, load in kv_load.items() if load == min_load
        ]
        best_worker_id = random.choice(min_load_workers)

        logger.info(
            f"Selected worker: {best_worker_id}, KV load: {kv_load[best_worker_id]:.3f}"
        )
        return best_worker_id, kv_load[best_worker_id]

    @dynamo_endpoint()
    async def generate(self, request: Tokens) -> AsyncIterator[Tuple[WorkerId, float]]:
        metrics = await self.metrics_aggregator.get_metrics()

        # Quick return for KV_LOAD mode
        if self.router_type == RouterType.KV_LOAD:
            try:
                yield self._get_underloaded_worker(metrics)
            except Exception as e:
                logger.exception(
                    f"Error finding underloaded worker: {e}. {fallback_msg}"
                )
                yield "", 0.0
            return

        # Existing KV routing logic
        lora_id = 0
        try:
            scores = await self.indexer.find_matches_for_request(
                request.tokens, lora_id
            )
        except Exception as e:
            scores = {}
            logger.exception(f"Error finding matches: {e}. {fallback_msg}")
            yield "", 0.0
            return

        worker_id, prefix_hit_rate = self._cost_function(
            scores, metrics, len(request.tokens)
        )

        if worker_id:
            logger.info(
                f"Scheduling to worker_id: {worker_id} with estimated prefix hit rate: {prefix_hit_rate}"
            )

        yield worker_id, prefix_hit_rate
