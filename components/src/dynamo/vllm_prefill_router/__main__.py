# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Centralized Prefill Router Service

Usage: python -m dynamo.vllm_prefill_router [args]

This service provides a single KV-aware router for all prefill workers in a
disaggregated vLLM deployment. Instead of each decode worker maintaining its own
round-robin client to prefill workers, this service uses KvRouter to make
intelligent routing decisions based on KV cache state.
"""

import argparse
import asyncio
import logging
import os
from typing import Optional

import uvloop

from dynamo.llm import KvRouter, KvRouterConfig
from dynamo.runtime import Client, DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class PrefillRouterHandler:
    """Handles routing requests to prefill workers using KV-aware routing."""

    def __init__(self, runtime: DistributedRuntime, namespace: str, block_size: int):
        self.runtime = runtime
        self.namespace = namespace
        self.block_size = block_size
        self.kv_router: Optional[KvRouter] = None
        self.prefill_client: Optional[Client] = None

    async def initialize(self):
        """Initialize the KV router for prefill workers."""
        try:
            # Get prefill endpoint
            prefill_endpoint = (
                self.runtime.namespace(self.namespace)
                .component("prefill")
                .endpoint("generate")
            )

            self.prefill_client = await prefill_endpoint.client()

            # Create KvRouter with specified configuration
            kv_router_config = KvRouterConfig(
                router_track_active_blocks=False,  # this won't matter for prefill workers
                router_reset_states=True,  # reset for now
            )

            self.kv_router = KvRouter(
                endpoint=prefill_endpoint,
                block_size=self.block_size,
                kv_router_config=kv_router_config,
            )

            logger.info(
                f"KvRouter initialized for prefill workers with block_size={self.block_size}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize KvRouter: {e}")
            raise

    async def find_best_worker(self, request):
        """
        Find the best prefill worker based on KV cache state.

        This endpoint is called by decode workers to determine which prefill
        worker should handle a request.
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
            # Get current prefill workers
            if self.prefill_client is None:
                yield {
                    "status": "error",
                    "message": "Prefill client not initialized",
                }
                return

            instance_ids = self.prefill_client.instance_ids()
            if not instance_ids:
                yield {
                    "status": "error",
                    "message": "No prefill workers available",
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
                update_states=True,  # Always update states for prefill routing
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
        description="Dynamo Prefill Router Service: Centralized KV-aware routing for prefill workers",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--namespace",
        type=str,
        default=os.environ.get("DYN_NAMESPACE", "dynamo"),
        help="Dynamo namespace for discovering prefill workers (default: dynamo or DYN_NAMESPACE env var)",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="KV cache block size for routing decisions (default: 128)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    """Main worker function for the prefill router service."""

    args = parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info(f"Starting Prefill Router Service for namespace: {args.namespace}")
    logger.debug(f"Configuration: block_size={args.block_size}")

    # Create service component
    component = runtime.namespace(args.namespace).component("prefill_router")
    await component.create_service()

    # Create handler
    handler = PrefillRouterHandler(runtime, args.namespace, args.block_size)
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
                metrics_labels=[("service", "prefill_router")],
            ),
            free_endpoint.serve_endpoint(
                handler.free,
                graceful_shutdown=True,
                metrics_labels=[("service", "prefill_router")],
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoint: {e}")
        raise
    finally:
        logger.info("Prefill Router Service shutting down")


def main():
    """Entry point for the prefill router service."""
    uvloop.run(worker())


if __name__ == "__main__":
    main()
