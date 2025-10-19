# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Companion server main entry point using Dynamo DistributedRuntime."""

import argparse
import asyncio
import logging
import os
import signal

import uvloop

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

from .handler import CompanionHandler

configure_dynamo_logging()
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Companion server for CUDA IPC weight sharing")
    parser.add_argument("--device-id", type=int, required=True, help="GPU device ID")
    parser.add_argument(
        "--companion-master-port",
        type=int,
        default=29700,
        help=f"Port for companion master process coordination (default: {29700})"
    )
    return parser.parse_args()


async def graceful_shutdown(runtime):
    """Shutdown dynamo distributed runtime."""
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    """Main worker function for companion server."""
    args = parse_args()

    # Get namespace from environment or default to "dynamo"
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")

    # Default component name based on device ID
    component_name = f"companion-gpu{args.device_id}"

    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logger.info(f"Starting companion server for GPU {args.device_id}")
    logger.info(f"Using companion master port: {args.companion_master_port}")

    # Create component and endpoint
    component = runtime.namespace(namespace).component(component_name)
    await component.create_service()

    load_model_endpoint = component.endpoint("load_model")

    # Create handler
    handler = CompanionHandler(
        device_id=args.device_id,
        companion_master_port=args.companion_master_port
    )

    logger.info(f"Serving companion endpoint at {namespace}.{component_name}.load_model")

    try:
        await load_model_endpoint.serve_endpoint(
            handler.load_model,
            graceful_shutdown=True
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoint: {e}")
        raise

    logger.info("Companion server shutdown complete")


def main():
    """Main entry point."""
    uvloop.run(worker())


if __name__ == "__main__":
    main()
