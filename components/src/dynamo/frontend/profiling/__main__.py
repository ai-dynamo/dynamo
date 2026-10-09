# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional standalone host for the same callable coordinator."""

import argparse
import asyncio
import logging
import signal
import socket

from dynamo.profiling import Config, Coordinator, RuntimeSnapshot
from dynamo.profiling.collectors import process_epoch

from .hosting import host, load_config


async def serve(config: Config, pid: int, epoch: str) -> None:
    runtime = RuntimeSnapshot(pid, epoch, socket.gethostname(), config.workers)
    coordinator = Coordinator(config, runtime)
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, shutdown.set)

    async def watch_parent() -> None:
        while not shutdown.is_set():
            try:
                if process_epoch(pid) != epoch:
                    shutdown.set()
            except (OSError, IndexError):
                shutdown.set()
            try:
                await asyncio.wait_for(shutdown.wait(), 0.5)
            except asyncio.TimeoutError:
                pass

    watcher = asyncio.create_task(watch_parent())
    try:
        async with host(coordinator):
            await shutdown.wait()
    finally:
        shutdown.set()
        await watcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--frontend-pid", type=int, required=True)
    parser.add_argument("--frontend-epoch", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(load_config(args.config), args.frontend_pid, args.frontend_epoch))


if __name__ == "__main__":
    main()
