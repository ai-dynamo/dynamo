# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service allocation server runner.

This module provides the CLI runner for the GPU Memory Service server,
which manages GPU memory allocations with connection-based RW/RO locking.
By default one process serves every production GMS tag on independent
sockets; the instances share only process-wide CUDA initialization. Pass
--tag to serve a subset.

Usage:
    python -m gpu_memory_service --device 0
    python -m gpu_memory_service --device 0 --tag weights
    python -m gpu_memory_service --device 0 --tag weights --socket-path /tmp/gms.sock

Set DYN_GMS_USE_V1=true to run the CUDA-only V1 server.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
from collections.abc import Sequence

import uvloop
from gpu_memory_service.common.vmm import init_vmm
from gpu_memory_service.server.rpc import GMSRPCServer

from .args import Config, parse_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_servers(servers: Sequence[GMSRPCServer]) -> None:
    """Serve all instances; when any stops, cancel the rest and raise."""
    tasks = [asyncio.create_task(server.serve()) for server in servers]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        raise RuntimeError("GMS server stopped unexpectedly")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def serve_configs(configs: Sequence[Config]) -> None:
    """Construct and serve independent GMS instances in one process."""
    if any(config.verbose for config in configs):
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("gpu_memory_service").setLevel(logging.DEBUG)

    init_vmm(configs[0].device_type)
    servers = []
    for config in configs:
        logger.info("Starting GPU Memory Service Server for device %d", config.device)
        logger.info("GMS tag: %s", config.tag)
        logger.info("Socket path: %s", config.socket_path)
        logger.info("VMM device type: %s", config.device_type.value)
        logger.info(
            "Allocation retry config: interval=%ss timeout=%s",
            config.alloc_retry_interval,
            (
                f"{config.alloc_retry_timeout}s"
                if config.alloc_retry_timeout is not None
                else "none"
            ),
        )
        servers.append(
            GMSRPCServer(
                config.socket_path,
                device=config.device,
                allocation_retry_interval=config.alloc_retry_interval,
                allocation_retry_timeout=config.alloc_retry_timeout,
            )
        )

    await run_servers(servers)


def main(argv: list[str] | None = None) -> None:
    """Entry point for GPU Memory Service server."""
    if os.environ.get("DYN_GMS_USE_V1") == "true":
        v1_cli = importlib.import_module("gpu_memory_service.v1.cli")
        v1_cli.main(argv)
        return

    uvloop.install()
    asyncio.run(serve_configs(parse_args(argv)))


if __name__ == "__main__":
    main()
