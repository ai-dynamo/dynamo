# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone KV Router Service

Usage: python -m dynamo.router --endpoint <namespace.component.endpoint> [args]

This service provides a standalone KV-aware router for any set of workers
in a Dynamo deployment. It can be used for disaggregated serving (e.g., routing
to prefill workers) or any other scenario requiring intelligent KV cache-aware
routing decisions.
"""

import argparse
import asyncio
import logging
from typing import Optional

import uvloop

from dynamo.llm import KvPushRouter, KvRouterConfig
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
        self.kv_push_router: Optional[KvPushRouter] = None
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
            worker_endpoint = (
                self.runtime.namespace(namespace)
                .component(component)
                .endpoint(endpoint)
            )

            self.worker_client = await worker_endpoint.client()

            # Create KvPushRouter with specified configuration
            self.kv_push_router = KvPushRouter(
                endpoint=worker_endpoint,
                block_size=self.block_size,
                kv_router_config=self.kv_router_config,
            )

        except Exception as e:
            logger.error(f"Failed to initialize KvPushRouter: {e}")
            raise

    async def generate(self, request):
        """
        Generate tokens using the KV-aware router.

        This endpoint routes the request to the best worker and streams back results.
        Wraps the request into PreprocessedRequest format and wraps worker responses
        into LLMEngineOutput format.
        """
        if self.kv_push_router is None:
            logger.error("KvPushRouter not initialized - cannot process request")
            raise RuntimeError("Router not initialized")

        # Wrap incoming request into PreprocessedRequest format for KvPushRouter
        # The request should already have most fields, but we ensure it has the structure
        preprocessed_request = {
            "model": request.get("model", "unknown"),
            "token_ids": request["token_ids"],
            "stop_conditions": request.get("stop_conditions", {}),
            "sampling_options": request.get("sampling_options", {}),
            "output_options": request.get("output_options", {}),
            "eos_token_ids": request.get("eos_token_ids", []),
            "annotations": request.get("annotations", []),
            "disaggregated_params": request.get("disaggregated_params"),
            "dp_rank": request.get("dp_rank"),
            "extra_args": request.get("extra_args", {}),
        }

        # Route and process through KvPushRouter
        async for worker_output in await self.kv_push_router.generate_from_request(
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
        if self.kv_push_router is None:
            logger.error("KvPushRouter not initialized - cannot get best worker")
            raise RuntimeError("Router not initialized")

        result = await self.kv_push_router.best_worker_id(
            token_ids, router_config_override
        )

        yield result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamo Standalone Router Service: Configurable KV-aware routing for any worker endpoint",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        required=True,
        help=(
            "Full endpoint path for workers in the format namespace.component.endpoint\n"
            "(e.g., dynamo.prefill.generate for prefill workers)"
        ),
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="KV cache block size for routing decisions (default: 128)",
    )

    parser.add_argument(
        "--kv-overlap-score-weight",
        type=float,
        default=1.0,
        help="KV Router: Weight for overlap score in worker selection. Higher values prioritize KV cache reuse (default: 1.0)",
    )

    parser.add_argument(
        "--router-temperature",
        type=float,
        default=0.0,
        help="KV Router: Temperature for worker sampling via softmax. Higher values promote more randomness, and 0 fallbacks to deterministic (default: 0.0)",
    )

    parser.add_argument(
        "--no-kv-events",
        action="store_false",
        dest="use_kv_events",
        default=True,
        help="KV Router: Disable KV events. When set, uses ApproxKvRouter for predicting block creation/deletion based only on incoming requests. By default, KV events are enabled.",
    )

    parser.add_argument(
        "--router-replica-sync",
        action="store_true",
        default=False,
        help="KV Router: Enable replica synchronization across multiple router instances. When true, routers will publish and subscribe to events to maintain consistent state (default: False)",
    )

    parser.add_argument(
        "--router-snapshot-threshold",
        type=int,
        default=1000000,
        help="KV Router: Number of messages in stream before triggering a snapshot (default: 1000000)",
    )

    parser.add_argument(
        "--router-reset-states",
        action="store_true",
        dest="router_reset_states",
        default=False,
        help="KV Router: Reset router state on startup, purging stream and object store. By default, states are persisted. WARNING: This can affect existing router replicas (default: False)",
    )

    parser.add_argument(
        "--no-track-active-blocks",
        action="store_false",
        dest="router_track_active_blocks",
        default=True,
        help="KV Router: Disable tracking of active blocks (blocks being used for ongoing generation). By default, active blocks are tracked for load balancing (default: True)",
    )

    return parser.parse_args()


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    """Main worker function for the standalone router service."""

    args = parse_args()

    # Parse endpoint path to get namespace for service registration
    endpoint_parts = args.endpoint.split(".")
    if len(endpoint_parts) != 3:
        raise ValueError(
            f"Invalid endpoint path format: {args.endpoint}. "
            "Expected format: namespace.component.endpoint"
        )
    namespace = endpoint_parts[0]

    logger.info("Starting Standalone Router Service")
    logger.debug(
        f"Configuration: endpoint={args.endpoint}, block_size={args.block_size}, "
        f"overlap_score_weight={args.kv_overlap_score_weight}, "
        f"router_temperature={args.router_temperature}, "
        f"use_kv_events={args.use_kv_events}, "
        f"router_replica_sync={args.router_replica_sync}, "
        f"router_reset_states={args.router_reset_states}, "
        f"router_track_active_blocks={args.router_track_active_blocks}"
    )

    # Create KvRouter configuration
    kv_router_config = KvRouterConfig(
        overlap_score_weight=args.kv_overlap_score_weight,
        router_temperature=args.router_temperature,
        use_kv_events=args.use_kv_events,
        router_replica_sync=args.router_replica_sync,
        router_snapshot_threshold=args.router_snapshot_threshold,
        router_reset_states=args.router_reset_states,
        router_track_active_blocks=args.router_track_active_blocks,
    )

    # Create service component - use "router" as component name
    component = runtime.namespace(namespace).component("router")

    # Create handler
    handler = StandaloneRouterHandler(
        runtime, args.endpoint, args.block_size, kv_router_config
    )
    await handler.initialize()

    # Expose endpoints
    generate_endpoint = component.endpoint("generate")
    best_worker_endpoint = component.endpoint("best_worker_id")

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
