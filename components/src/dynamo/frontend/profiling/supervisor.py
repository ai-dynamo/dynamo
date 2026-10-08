# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit FrontendConfig-to-helper process boundary."""

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import msgspec

from .collectors import process_epoch
from .models import Config


def load_config(path: str) -> Config:
    return msgspec.json.decode(Path(path).read_bytes(), type=Config)


@asynccontextmanager
async def profiling_helper(config_path: str | None) -> AsyncIterator[None]:
    """Start the opt-in helper and request orderly finalization on shutdown."""
    if config_path is None:
        yield
        return
    config = load_config(config_path)
    if Path(config.socket_path).exists():
        raise ValueError("Profiling socket already exists; refusing to reuse it")
    pid = os.getpid()
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "dynamo.frontend.profiling",
        "--config",
        config_path,
        "--frontend-pid",
        str(pid),
        "--frontend-epoch",
        process_epoch(pid),
    )
    try:
        async with aiohttp.ClientSession(
            connector=aiohttp.UnixConnector(path=config.socket_path),
            timeout=aiohttp.ClientTimeout(total=0.5),
        ) as client:

            async def ready() -> None:
                while True:
                    if child.returncode is not None:
                        raise RuntimeError("Profiling helper exited during startup")
                    try:
                        async with client.get(
                            "http://localhost/admin/profiling/v1/targets"
                        ) as response:
                            response.raise_for_status()
                            return
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        await asyncio.sleep(0.05)

            await asyncio.wait_for(ready(), 10.0)
        yield
    finally:
        if child.returncode is None:
            child.terminate()
        # Never force-kill a helper while it might own a worker profiler.
        # Backend flushing can be long; its configured timeout governs cleanup.
        await child.wait()
