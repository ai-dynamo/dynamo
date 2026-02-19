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

from dynamo.llm import KvRouter, KvRouterConfig
from dynamo.router.args import build_kv_router_config
from dynamo.router.args import parse_args as parse_router_args
from dynamo.router.backend_args import DynamoRouterConfig
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
    ):
        self.runtime = runtime
        self.worker_endpoint_path = worker_endpoint_path
        self.block_size = block_size
        self.kv_router_config = kv_router_config
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
            )

        except Exception as e:
            logger.error(f"Failed to initialize KvRouter: {e}")
            raise

    async def generate(self, request):
        """
        Generate tokens using the KV-aware router.

        This endpoint routes the request to the best worker and streams back results.
        Wraps the request into PreprocessedRequest format and wraps worker responses
        into LLMEngineOutput format.
        """
        if self.kv_router is None:
            logger.error("KvRouter not initialized - cannot process request")
            raise RuntimeError("Router not initialized")

        # Wrap incoming request into PreprocessedRequest format for KvRouter
        # The request should already have most fields, but we ensure it has the structure
        # Build routing hints from request (supports both nested routing object and legacy dp_rank)
        routing = request.get("routing")
        dp_rank = request.get("dp_rank")
        if routing is None and dp_rank is not None:
            routing = {"dp_rank": dp_rank}

        preprocessed_request = {
            "model": request.get("model", "unknown"),
            "token_ids": request["token_ids"],
            "stop_conditions": request.get("stop_conditions", {}),
            "sampling_options": request.get("sampling_options", {}),
            "output_options": request.get("output_options", {}),
            "eos_token_ids": request.get("eos_token_ids", []),
            "annotations": request.get("annotations", []),
            "routing": routing,
            "router_config_override": request.get("router_config_override"),
            "prefill_result": request.get("prefill_result"),
            "bootstrap_info": request.get("bootstrap_info"),
            "extra_args": request.get("extra_args"),
        }

        async for worker_output in await self.kv_router.generate_from_request(
            preprocessed_request
        ):
            # Wrap worker output into LLMEngineOutput format
            # Worker should return dict with at minimum kv_transfer_params in extra_args
            llm_engine_output = {
                "token_ids": worker_output.get("token_ids", []),
                "tokens": worker_output.get("tokens"),
                "text": worker_output.get("text"),
                "cum_log_probs": worker_output.get("cum_log_probs"),
                "log_probs": worker_output.get("log_probs"),
                "top_logprobs": worker_output.get("top_logprobs"),
                "finish_reason": worker_output.get("finish_reason"),
                "stop_reason": worker_output.get("stop_reason"),
                "index": worker_output.get("index"),
                "disaggregated_params": worker_output.get("disaggregated_params"),
                "extra_args": worker_output.get("extra_args"),
                "completion_usage": worker_output.get("completion_usage"),
            }
            yield llm_engine_output

    async def best_worker_id(self, token_ids, router_config_override=None):
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
            token_ids, router_config_override
        )

        yield worker_id


def parse_args(argv=None) -> DynamoRouterConfig:
    """Parse router CLI arguments (compatibility shim delegating to args.parse_args)."""
    return parse_router_args(argv)


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    """Main worker function for the standalone router service."""

    config = parse_args()

    logger.info("Starting Standalone Router Service")
    logger.debug(
        f"Configuration: endpoint={config.endpoint}, router_block_size={config.router_block_size}, "
        f"overlap_score_weight={config.router_kv_overlap_score_weight}, "
        f"router_temperature={config.router_temperature}, "
        f"router_use_kv_events={config.router_use_kv_events}, "
        f"router_durable_kv_events={config.router_durable_kv_events}, "
        f"router_replica_sync={config.router_replica_sync}, "
        f"router_reset_states={config.router_reset_states}, "
        f"router_track_active_blocks={config.router_track_active_blocks}, "
        f"router_track_output_blocks={config.router_track_output_blocks}, "
        f"router_assume_kv_reuse={config.router_assume_kv_reuse}, "
        f"router_ttl_secs={config.router_ttl_secs}, "
        f"router_max_tree_size={config.router_max_tree_size}, "
        f"router_prune_target_ratio={config.router_prune_target_ratio}"
    )

    kv_router_config = build_kv_router_config(config)

    # Create handler
    handler = StandaloneRouterHandler(
        runtime, config.endpoint, config.router_block_size, kv_router_config
    )
    await handler.initialize()

    # Create endpoints
    generate_endpoint = runtime.endpoint(f"{config.namespace}.router.generate")
    best_worker_endpoint = runtime.endpoint(f"{config.namespace}.router.best_worker_id")

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
