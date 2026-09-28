# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone KV Router Service

Usage: python -m dynamo.router --endpoint <namespace.component.endpoint> [args]

This service provides a standalone KV-aware router for any set of workers
in a Dynamo deployment. It can be used for disaggregated serving (e.g., routing
to prefill workers) or any other scenario requiring intelligent KV cache-aware
routing decisions.
"""

import asyncio
import logging
from typing import Optional

import uvloop

from dynamo.llm import AicPerfConfig, KvRouter, KvRouterConfig
from dynamo.router.args import (
    DynamoRouterConfig,
    build_aic_perf_config,
    build_kv_router_config,
)
from dynamo.router.args import parse_args as parse_router_args
from dynamo.runtime import Client, DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class StandaloneRouterHandler:
    """Handles routing requests to workers using KV-aware routing."""

    def __init__(
        self,
        runtime: DistributedRuntime,
        worker_endpoint_path: str,
        block_size: int,
        kv_router_config: KvRouterConfig,
        aic_perf_config: Optional[AicPerfConfig],
    ):
        self.runtime = runtime
        self.worker_endpoint_path = worker_endpoint_path
        self.block_size = block_size
        self.kv_router_config = kv_router_config
        self.aic_perf_config = aic_perf_config
        self.kv_router: Optional[KvRouter] = None
        self.worker_client: Optional[Client] = None

    async def initialize(self):
        """Initialize the KV router for workers."""
        try:
            # Parse endpoint path (format: namespace.component.endpoint)
            parts = self.worker_endpoint_path.split(".")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid endpoint path format: {self.worker_endpoint_path}. "
                    "Expected format: namespace.component.endpoint"
                )
            namespace, component, endpoint = parts

            # Get worker endpoint
            worker_endpoint = self.runtime.endpoint(
                f"{namespace}.{component}.{endpoint}"
            )
            self.worker_client = await worker_endpoint.client()

            self.kv_router = KvRouter(
                endpoint=worker_endpoint,
                block_size=self.block_size,
                kv_router_config=self.kv_router_config,
                aic_perf_config=self.aic_perf_config,
            )

        except Exception as e:
            logger.error(f"Failed to initialize KvRouter: {e}")
            raise

    async def generate(self, request):
        """
        Generate tokens using the KV-aware router.

        Routes a PreprocessedRequest-shaped request to the best worker and streams back
        its LLMEngineOutput responses unchanged, so fields this router does not inspect
        (multimodal data, KV hints, agent context, ...) survive the hop.
        """
        if self.kv_router is None:
            logger.error("KvRouter not initialized - cannot process request")
            raise RuntimeError("Router not initialized")

        preprocessed_request = dict(request)
        preprocessed_request.setdefault("model", "unknown")
        # Legacy callers send a top-level dp_rank instead of routing hints.
        dp_rank = preprocessed_request.get("dp_rank")
        if preprocessed_request.get("routing") is None and dp_rank is not None:
            preprocessed_request["routing"] = {"dp_rank": dp_rank}

        async for worker_output in await self.kv_router.generate_from_request(
            preprocessed_request
        ):
            yield worker_output

    async def best_worker_id(
        self, token_ids, router_config_override=None, cache_namespace=None
    ):
        """
        Get the best worker ID for a given set of tokens without actually routing.

        This method returns the worker ID that would be selected based on KV cache
        overlap, but does NOT actually route the request or update router states.
        It's useful for debugging, monitoring, or implementing custom routing logic.
        """
        if self.kv_router is None:
            logger.error("KvRouter not initialized - cannot get best worker")
            raise RuntimeError("Router not initialized")

        (worker_id, _dp_rank, _overlap_blocks) = await self.kv_router.best_worker(
            token_ids,
            router_config_override,
            cache_namespace=cache_namespace,
        )

        yield worker_id

    async def get_overlap_scores(self, request):
        """
        Get per-worker KV overlap by storage tier without routing the request.

        This endpoint returns matched blocks for each worker_id/dp_rank pair.
        Shared-cache hits are request-global and are also reported per row as
        blocks beyond that rank's device-local prefix.
        """
        if self.kv_router is None:
            logger.error("KvRouter not initialized - cannot get overlap scores")
            raise RuntimeError("Router not initialized")

        scores = await self.kv_router.get_overlap_scores(
            request["token_ids"],
            request.get("router_config_override"),
            request.get("block_mm_infos"),
            request.get("lora_name"),
            request.get("include_shared", True),
            request.get("cache_namespace"),
        )

        yield scores


def parse_args(argv=None) -> DynamoRouterConfig:
    """Parse router CLI arguments (compatibility shim delegating to args.parse_args)."""
    return parse_router_args(argv)


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    """Main worker function for the standalone router service."""

    config = parse_args()

    logger.info("Starting Standalone Router Service")
    logger.debug(
        "Configuration: endpoint=%s, router_block_size=%s, "
        "overlap_score_credit=%s, overlap_score_credit_decay=%s, "
        "prefill_load_scale=%s, decode_active_request_weight=%s, "
        "router_temperature=%s, use_kv_events=%s, router_replica_sync=%s, "
        "router_track_active_blocks=%s, router_track_output_blocks=%s, "
        "router_assume_kv_reuse=%s, router_track_prefill_tokens=%s, "
        "router_ttl_secs=%s, router_approximate_cache_policy=%s",
        config.endpoint,
        config.router_block_size,
        config.overlap_score_credit,
        config.overlap_score_credit_decay,
        config.prefill_load_scale,
        config.decode_active_request_weight,
        config.router_temperature,
        config.use_kv_events,
        config.router_replica_sync,
        config.router_track_active_blocks,
        config.router_track_output_blocks,
        config.router_assume_kv_reuse,
        config.router_track_prefill_tokens,
        config.router_ttl_secs,
        config.router_approximate_cache_policy,
    )

    kv_router_config = build_kv_router_config(config)
    aic_perf_config = build_aic_perf_config(config)

    # Create handler
    handler = StandaloneRouterHandler(
        runtime,
        config.endpoint,
        config.router_block_size,
        kv_router_config,
        aic_perf_config,
    )
    await handler.initialize()

    # Create endpoints
    generate_endpoint = runtime.endpoint(f"{config.namespace}.router.generate")
    best_worker_endpoint = runtime.endpoint(f"{config.namespace}.router.best_worker_id")
    overlap_scores_endpoint = runtime.endpoint(
        f"{config.namespace}.router.get_overlap_scores"
    )

    logger.debug("Starting to serve endpoints...")

    # Serve both endpoints concurrently
    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=[("service", "router")],
            ),
            best_worker_endpoint.serve_endpoint(
                handler.best_worker_id,
                graceful_shutdown=True,
                metrics_labels=[("service", "router")],
            ),
            overlap_scores_endpoint.serve_endpoint(
                handler.get_overlap_scores,
                graceful_shutdown=True,
                metrics_labels=[("service", "router")],
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoint: {e}")
        raise
    finally:
        logger.info("Standalone Router Service shutting down")


def main():
    """Entry point for the standalone router service."""
    uvloop.run(worker())


if __name__ == "__main__":
    main()
