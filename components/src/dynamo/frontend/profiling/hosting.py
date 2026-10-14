# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend hosting adapter; the coordinator itself has no process/HTTP owner."""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec
from aiohttp import web

from dynamo.profiling import Config, RuntimeSnapshot
from dynamo.profiling.collectors import process_epoch

if TYPE_CHECKING:
    from dynamo.profiling import Coordinator


def load_config(path: str) -> Config:
    return msgspec.json.decode(Path(path).read_bytes(), type=Config)


def private_directory(path: Path) -> None:
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    stat = path.stat()
    if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        raise ValueError(
            f"Profiling directory must be owned by this user with mode 0700: {path}"
        )


@asynccontextmanager
async def host(coordinator: Coordinator) -> AsyncIterator[Coordinator]:
    """Own only the private admin listener and drain capture on shutdown."""
    from .http import application

    endpoint = Path(coordinator.config.socket_path)
    private_directory(endpoint.parent)
    private_directory(Path(coordinator.config.output_dir))
    if endpoint.exists() or endpoint.is_symlink():
        if not endpoint.is_socket():
            raise ValueError("Profiling socket path exists and is not a socket")
        # A crashed host leaves its socket behind; only a live listener owns it.
        with socket.socket(socket.AF_UNIX) as probe:
            try:
                probe.connect(str(endpoint))
            except ConnectionRefusedError:
                endpoint.unlink()
            else:
                raise ValueError("Profiling socket is owned by a running host")
    runner = web.AppRunner(application(coordinator))
    await runner.setup()
    try:
        await web.UnixSite(runner, str(endpoint)).start()
        endpoint.chmod(0o600)
        yield coordinator
    finally:
        # Stop accepting leases first, then finish any owned profiler exports.
        await runner.cleanup()
        try:
            await coordinator.close()
        finally:
            endpoint.unlink(missing_ok=True)


@asynccontextmanager
async def profiling_coordinator(
    config_path: str | None, *, runtime_context: dict | None = None
) -> AsyncIterator[Coordinator | None]:
    """Frontend gathers its runtime snapshot and calls the reusable module."""
    if config_path is None:
        yield None
        return
    from dynamo.profiling import Coordinator

    config = load_config(config_path)
    pid = os.getpid()
    runtime = RuntimeSnapshot(
        pid,
        process_epoch(pid),
        socket.gethostname(),
        config.workers,
        runtime_context or {},
    )
    coordinator = Coordinator(config, runtime)
    async with host(coordinator):
        yield coordinator
