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

"""KV-based router for SGLang workers.

This implementation mirrors the KV router logic used in the vLLM example, but
routes requests to ``SGLangWorker`` instances instead.  The key idea is to
compute a *cost* (or *logit*) for each candidate worker using

1. Cache-overlap scores obtained from a ``KvIndexer`` (shared library from
   ``dynamo.llm``).
2. Runtime metrics (GPU cache usage, prefix-cache hit-rate, request queue
   length) aggregated by ``KvMetricsAggregator``.

The router selects the worker with the highest score and streams back the
selected ``worker_id`` together with the estimated prefix-hit ratio so that the
caller can attach this information to the subsequent request.

Both the indexer and the aggregator communicate with workers over ZMQ, which
matches the existing SGLang deployment pattern.
"""

from __future__ import annotations

import argparse
import logging
import random
from argparse import Namespace
from typing import AsyncIterator, Tuple

from components.worker import SGLangWorker
from utils.protocol import Tokens

from dynamo.llm import (
    AggregatedMetrics,
    KvIndexer,
    KvMetricsAggregator,
    OverlapScores,
)
from dynamo.sdk import async_on_start, depends, dynamo_context, endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

class RouterType:
    RANDOM = "random"
    ROUND_ROBIN = "round-robin"
    KV = "kv"
    KV_LOAD = "kv-load"

# Type alias for clarity
WorkerId = str

fallback_msg = "Will fallback to random routing."
logger = logging.getLogger(__name__)


def parse_args(service_name: str, prefix: str) -> Namespace:
    """Parse router-specific CLI/ServiceConfig arguments."""
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
        default=RouterType.KV,
        help="The router type",
    )

    # Merge with service-level configuration
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
    """Router that decides which ``SGLangWorker`` should execute a request."""

    # A dependency is declared so that the runtime ensures workers are launched
    worker = depends(SGLangWorker)

    def __init__(self):
        logger.info("Initializing Custom KV Router for SGLang")
        self.args = parse_args(self.__class__.__name__, "")

        # Default metric values when data is unavailable
        self.default_metrics = {
            "gpu_cache_usage_perc": 0.0,
            "num_requests_waiting": 0.0,
            "gpu_prefix_cache_hit_rate": 0.0,
        }

    # ---------------------------------------------------------------------
    # Lifecycle hooks
    # ---------------------------------------------------------------------

    @async_on_start
    async def async_init(self):
        """Connect to worker endpoints and initialise ZMQ helpers."""
        logger.info("=== starting async init")
        self.runtime = dynamo_context["runtime"]

        # Client used to fetch current list of worker instance IDs
        self.workers_client = (
            await self.runtime.namespace("dynamo")
            .component("SGLangWorker")
            .endpoint("generate")
            .client()
        )

        self.router_type = self.args.router


        kv_listener = self.runtime.namespace("dynamo").component("SGLangWorker")
        await kv_listener.create_service()

        self.indexer = KvIndexer(kv_listener, self.args.block_size)

        logger.info("=== created kvIndexer with block size: %s", self.args.block_size)

        self.metrics_aggregator = KvMetricsAggregator(kv_listener)
        logger.info("KV Router initialised (type=%s)", self.router_type)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cost_function(
        self,
        scores: OverlapScores | None,
        metrics: AggregatedMetrics | None,
        token_length: int,
    ) -> Tuple[WorkerId, float]:
        """Compute the best worker given scores/metrics.

        Returns a tuple of ``(worker_id, estimated_prefix_hit_rate)``.
        """

        worker_scores: dict[WorkerId, float] = {}
        if scores:
            for worker_id, score in scores.scores.items():
                # score is in *blocks*; convert to *tokens* hit-rate
                worker_scores[worker_id] = (
                    score * self.indexer.block_size() / token_length
                )
        else:
            logger.warning("Cannot get KV scores")

        worker_metrics: dict[WorkerId, dict[str, float]] = {}
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

        # Consider *all* workers, even if metrics/scores missing
        worker_ids = self.workers_client.instance_ids()

        worker_logits: dict[WorkerId, float] = {}
        for worker_id in worker_ids:
            score = worker_scores.get(worker_id, 0.0)
            metrics_dict = worker_metrics.get(worker_id, self.default_metrics)
            gpu_cache_usage = metrics_dict["gpu_cache_usage_perc"]
            normalized_waiting = (
                metrics_dict["num_requests_waiting"] / max_waiting
                if max_waiting > 0 else 0.0
            )

            # Simple linear combination (higher is better)
            worker_logits[worker_id] = 2 * score - gpu_cache_usage - normalized_waiting
            logger.info(
                "Formula for %s: %.3f = 2.0 * %.3f - %.3f - %.3f",
                worker_id,
                worker_logits[worker_id],
                score,
                gpu_cache_usage,
                normalized_waiting,
            )

        if not worker_logits or not any(worker_logits.values()):
            logger.warning("All worker logits are zero. %s", fallback_msg)
            return "", 0.0

        max_logit = max(worker_logits.values())
        best_workers = [wid for wid, logit in worker_logits.items() if logit == max_logit]
        best_worker_id = random.choice(best_workers)

        if best_worker_id:
            metrics_dict = worker_metrics.get(best_worker_id, self.default_metrics)
            logger.info(
                "Selected worker: %s, logit: %.3f, Score: %.3f, GPU Cache Hit Rate: %.3f, GPU Cache Usage: %.3f, Requests Waiting: %s",
                best_worker_id,
                worker_logits[best_worker_id],
                worker_scores.get(best_worker_id, 0.0),
                metrics_dict["gpu_prefix_cache_hit_rate"],
                metrics_dict["gpu_cache_usage_perc"],
                metrics_dict["num_requests_waiting"],
            )

        return best_worker_id, worker_scores.get(best_worker_id, 0.0)

    def _get_underloaded_worker(self, metrics: AggregatedMetrics | None) -> Tuple[WorkerId, float]:
        """Pick worker with lowest GPU cache utilisation (KV_LOAD mode)."""
        if not metrics:
            logger.warning("Cannot get metrics. %s", fallback_msg)
            return "", 0.0

        kv_load = {
            endpoint.worker_id: getattr(endpoint, "gpu_cache_usage_perc", 0.0)
            for endpoint in metrics.endpoints
        }

        if not kv_load or not any(kv_load.values()):
            logger.warning("All KV loads are zero. %s", fallback_msg)
            return "", 0.0

        min_load = min(kv_load.values())
        min_load_workers = [wid for wid, load in kv_load.items() if load == min_load]
        best_worker_id = random.choice(min_load_workers)

        logger.info("Selected worker: %s, KV load: %.3f", best_worker_id, kv_load[best_worker_id])
        return best_worker_id, kv_load[best_worker_id]

    # ------------------------------------------------------------------
    # Public endpoint
    # ------------------------------------------------------------------

    @endpoint()
    async def generate(self, request: Tokens) -> AsyncIterator[Tuple[WorkerId, float]]:
        """Determine the best worker for the given request tokens."""
        # Pull latest metrics snapshot
        metrics = await self.metrics_aggregator.get_metrics()

        # Fast path for simple load-balancing
        if self.router_type == RouterType.KV_LOAD:
            try:
                yield self._get_underloaded_worker(metrics)
            except Exception as e:
                logger.exception("Error finding underloaded worker: %s. %s", e, fallback_msg)
                yield "", 0.0
            return

        # Full KV routing logic
        lora_id = 0  # LoRA not used in this example
        try:
            scores = await self.indexer.find_matches_for_request(request.tokens, lora_id)
        except Exception as e:
            logger.exception("Error finding matches: %s. %s", e, fallback_msg)
            yield "", 0.0
            return

        worker_id, prefix_hit_rate = self._cost_function(scores, metrics, len(request.tokens))

        if worker_id:
            logger.info(
                "Scheduling to worker_id: %s with estimated prefix hit rate: %s",
                worker_id,
                prefix_hit_rate,
            )

        yield worker_id, prefix_hit_rate
