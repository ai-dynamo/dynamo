# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service allocation server component for Dynamo.

This component wraps the GMSRPCServer from gpu_memory_service to manage
GPU memory allocations with connection-based RW/RO locking.

Workers connect via the socket path, which should be passed to vLLM/SGLang via:
    --load-format gpu_memory_service
    --model-loader-extra-config '{"gpu_memory_service_socket_path": "/tmp/gpu_memory_service_{device}.sock"}'

Usage:
    python -m dynamo.gpu_memory_service --device 0
    python -m dynamo.gpu_memory_service --device 0 --socket-path /tmp/gpu_memory_service_{device}.sock
"""

import asyncio
import logging
import signal

import uvloop
from gpu_memory_service.server import GMSRPCServer

from .args import parse_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def worker() -> None:
    """Main async worker function."""
    config = parse_args()

    # Configure logging level
    if config.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("dynamo.gpu_memory_service").setLevel(logging.DEBUG)

    logger.info(f"Starting GPU Memory Service Server for device {config.device}")
    logger.info(f"Socket path: {config.socket_path}")

    server = GMSRPCServer(config.socket_path, device=config.device)

    # Set up shutdown handling
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    await server.start()

    logger.info("GPU Memory Service Server ready, waiting for connections...")
    logger.info(
        f"To connect vLLM workers, use: --load-format gpu_memory_service "
        f'--model-loader-extra-config \'{{"gpu_memory_service_socket_path": "{config.socket_path}"}}\''
    )

    # Wait for shutdown signal
    try:
        await shutdown_event.wait()
    finally:
        logger.info("Shutting down GPU Memory Service Server...")
        await server.stop()
        logger.info("GPU Memory Service Server shutdown complete")


def main() -> None:
    """Entry point for GPU Memory Service server."""
    uvloop.install()
    asyncio.run(worker())


if __name__ == "__main__":
    main()
