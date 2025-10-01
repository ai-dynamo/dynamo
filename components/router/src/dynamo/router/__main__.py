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

from dynamo.llm import KvRouter, KvRouterConfig
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
            worker_endpoint = (
                self.runtime.namespace(namespace)
                .component(component)
                .endpoint(endpoint)
            )

            self.worker_client = await worker_endpoint.client()

            # Create KvRouter with specified configuration
            self.kv_router = KvRouter(
                endpoint=worker_endpoint,
                block_size=self.block_size,
                kv_router_config=self.kv_router_config,
            )

        except Exception as e:
            logger.error(f"Failed to initialize KvRouter: {e}")
            raise

    async def find_best_worker(self, request):
        """
        Find the best worker based on KV cache state.

        This endpoint is called by clients to determine which worker
        should handle a request.
        """
        if self.kv_router is None:
            # Fallback to round-robin if router not initialized
            logger.warning("KvRouter not initialized, falling back to round-robin")
            yield {
                "status": "fallback",
                "message": "Router not initialized",
            }
            return

        try:
            # Get current workers
            if self.worker_client is None:
                yield {
                    "status": "error",
                    "message": "Worker client not initialized",
                }
                return

            instance_ids = self.worker_client.instance_ids()
            if not instance_ids:
                yield {
                    "status": "error",
                    "message": "No workers available",
                }
                return

            logger.debug(f"Routing request with {len(instance_ids)} available workers")

            # Validate required fields
            if "token_ids" not in request:
                raise ValueError("Missing required field 'token_ids' in request")
            if "request_id" not in request:
                raise ValueError("Missing required field 'request_id' in request")

            token_ids = request["token_ids"]
            request_id = request["request_id"]

            # Use KvRouter to find the best worker with state updates
            best_worker_id, overlap_blocks = await self.kv_router.find_best_match(
                request_id=request_id,
                tokens=token_ids,
                update_states=True,  # Always update states for routing
            )

            logger.debug(
                f"Selected worker {best_worker_id} with {overlap_blocks} overlap blocks for request {request_id}"
            )

            yield {
                "worker_id": best_worker_id,
                "overlap_blocks": overlap_blocks,
            }

        except Exception as e:
            logger.error(f"Error finding best worker: {e}")
            yield {
                "status": "error",
                "message": str(e),
            }

    async def free(self, request):
        """
        Free resources associated with a request.

        This endpoint is called when a request is completed to clean up
        router state.
        """
        if self.kv_router is None:
            logger.warning("KvRouter not initialized")
            yield {
                "status": "error",
                "message": "Router not initialized",
            }
            return

        try:
            if "request_id" not in request:
                raise ValueError("Missing required field 'request_id' in request")

            request_id = request["request_id"]

            # Free the request from the router
            await self.kv_router.free(request_id=request_id)

            logger.debug(f"Freed resources for request {request_id}")

            yield {
                "status": "success",
                "message": f"Request {request_id} freed successfully",
            }

        except Exception as e:
            logger.error(f"Error freeing request: {e}")
            yield {
                "status": "error",
                "message": str(e),
            }


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


@dynamo_worker(static=False)
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
    await component.create_service()

    # Create handler
    handler = StandaloneRouterHandler(
        runtime, args.endpoint, args.block_size, kv_router_config
    )
    await handler.initialize()

    # Expose endpoints
    find_best_worker_endpoint = component.endpoint("find_best_worker")
    free_endpoint = component.endpoint("free")

    logger.debug("Starting to serve find_best_worker and free endpoints...")

    try:
        await asyncio.gather(
            find_best_worker_endpoint.serve_endpoint(
                handler.find_best_worker,
                graceful_shutdown=True,
                metrics_labels=[("service", "router")],
            ),
            free_endpoint.serve_endpoint(
                handler.free,
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
