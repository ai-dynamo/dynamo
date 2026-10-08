# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run outside the frontend's event loop so capture deadlines survive stalls."""

import argparse
import asyncio
import logging
import os
import signal
from pathlib import Path

from aiohttp import web

from .collectors import process_epoch
from .models import Config
from .service import ProfilingService
from .supervisor import load_config


def private_directory(path: Path) -> None:
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    stat = path.stat()
    if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        raise ValueError(
            f"Profiling directory must be owned by this user with mode 0700: {path}"
        )


async def serve(config: Config, pid: int, epoch: str) -> None:
    os.umask(0o077)
    socket = Path(config.socket_path)
    private_directory(socket.parent)
    private_directory(Path(config.output_dir))
    if socket.exists():
        raise ValueError("Profiling socket already exists; refusing to replace it")
    service = ProfilingService(config, pid, epoch)
    runner = web.AppRunner(service.application())
    await runner.setup()
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
        await web.UnixSite(runner, str(socket)).start()
        await shutdown.wait()
        # Stop accepting new leases before waiting for trace serialization.
        await runner.cleanup()
        await service.close()
    finally:
        shutdown.set()
        await watcher
        await runner.cleanup()
        socket.unlink(missing_ok=True)


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
