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


import asyncio
import random
from argparse import Namespace
from typing import AsyncIterator

import uvloop
from utils.protocol import Tokens
from vllm.logger import logger as vllm_logger

from dynamo.llm import AggregatedMetrics, KvIndexer, KvMetricsAggregator, OverlapScores
from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker

WorkerId = str


class CustomRouter:
    """
    Request handler for the generate endpoint
    """

    def __init__(
        self,
        workers_client,
        indexer: KvIndexer,
        metrics_aggregator: KvMetricsAggregator,
    ):
        vllm_logger.info("Initializing Custom Router")
        self.indexer = indexer
        self.metrics_aggregator = metrics_aggregator
        self.workers_client = workers_client

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
            vllm_logger.info(
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
            vllm_logger.info(
                f"Selected worker: {best_worker_id}, logit: {worker_logits[best_worker_id]:.3f}"
            )
            vllm_logger.info(
                f"Score: {scores.scores.get(best_worker_id, 0.0) if scores else 0.0:.3f}"
            )

            metrics_dict = worker_metrics.get(best_worker_id, {})
            vllm_logger.info(
                f"GPU Cache Hit Rate: {metrics_dict.get('gpu_prefix_cache_hit_rate', 0.0):.3f}"
            )
            vllm_logger.info(
                f"GPU Cache Usage: {metrics_dict.get('gpu_cache_usage_perc', 0.0):.3f}"
            )
            vllm_logger.info(
                f"Requests Waiting: {metrics_dict.get('num_requests_waiting', 0.0) / max_waiting if max_waiting > 0 else 0.0:.3f}"
            )

        return best_worker_id, worker_scores.get(best_worker_id, 0.0)

    @dynamo_endpoint(Tokens, WorkerId)
    async def generate(self, request) -> AsyncIterator[WorkerId]:
        lora_id = 0
        worker_id = ""
        try:
            scores = await self.indexer.find_matches_for_request(
                request.tokens, lora_id
            )
        except Exception as e:
            scores = {}
            vllm_logger.exception(f"Error finding matches: {e}")

        token_length = len(request.tokens)
        metrics = await self.metrics_aggregator.get_metrics()
        schedule_result = self._cost_function(scores, metrics, token_length)
        if schedule_result == "":
            worker_id = ""
            prefix_hit_rate = 0.0
        else:
            worker_id, prefix_hit_rate = schedule_result

        vllm_logger.info(
            f"Scheduling to worker_id: {worker_id} with estimated prefix hit rate: {prefix_hit_rate}"
        )

        yield f"{worker_id}_{prefix_hit_rate}"


@dynamo_worker()
async def worker(runtime: DistributedRuntime, args: Namespace):
    """
    Set up the worker clients.
    Serve the dynamo-init.router.generate endpoint.
    """
    workers_client = (
        await runtime.namespace("dynamo-init")
        .component("vllm")
        .endpoint("generate")
        .client()
    )

    while len(workers_client.endpoint_ids()) < args.min_workers:
        vllm_logger.info(
            f"Waiting for more workers... Current: {len(workers_client.endpoint_ids())}, Required: {args.min_workers}"
        )
        await asyncio.sleep(5)

    vllm_logger.info(
        f"Required number of workers ({args.min_workers}) are ready:\n"
        + "\n".join(f"id: {id}" for id in workers_client.endpoint_ids())
    )

    kv_listener = runtime.namespace("dynamo-init").component("vllm")
    await kv_listener.create_service()

    router_component = runtime.namespace("dynamo-init").component("router")
    await router_component.create_service()

    endpoint = router_component.endpoint("generate")

    indexer = KvIndexer(kv_listener, args.block_size)
    metrics_aggregator = KvMetricsAggregator(kv_listener)
    await endpoint.serve_endpoint(
        CustomRouter(workers_client, indexer, metrics_aggregator).generate
    )


if __name__ == "__main__":
    uvloop.install()

    import argparse

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
        default=64,
        help="KV block size",
    )
    parser.add_argument(
        "--custom-router",
        type=bool,
        default=False,
        help="Whether to use custom router or not",
    )
    args = parser.parse_args()

    asyncio.run(worker(args))
