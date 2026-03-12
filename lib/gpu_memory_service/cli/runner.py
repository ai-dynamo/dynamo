# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service allocation server runner.

This module provides the CLI runner for the GPU Memory Service server,
which manages GPU memory allocations with connection-based RW/RO locking.

Usage:
    python -m gpu_memory_service --device 0
    python -m gpu_memory_service --device 0 --socket-path /tmp/gpu_memory_service_{device}.sock
"""

import asyncio
import logging

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
        logging.getLogger("gpu_memory_service").setLevel(logging.DEBUG)

    logger.info(f"Starting GPU Memory Service Server for device {config.device}")
    logger.info(f"GMS scope: {config.scope}")
    logger.info(f"Socket path: {config.socket_path}")
    logger.info(
        "Allocation retry config: interval=%ss timeout=%s",
        config.alloc_retry_interval,
        (
            f"{config.alloc_retry_timeout}s"
            if config.alloc_retry_timeout is not None
            else "none"
        ),
    )

    server = GMSRPCServer(
        config.socket_path,
        device=config.device,
        allocation_retry_interval=config.alloc_retry_interval,
        allocation_retry_timeout=config.alloc_retry_timeout,
    )

    logger.info("GPU Memory Service Server ready, waiting for connections...")
    logger.info(f"Clients can connect via socket: {config.socket_path}")
    await server.serve()


def main() -> None:
    """Entry point for GPU Memory Service server."""
    uvloop.install()
    asyncio.run(worker())


if __name__ == "__main__":
    main()
