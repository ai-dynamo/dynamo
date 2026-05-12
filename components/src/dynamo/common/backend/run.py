# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common entry point for unified backends.

Each backend's ``unified_main.py`` calls :func:`run` with its
``LLMEngine`` subclass.  Example::

    from dynamo.common.backend.run import run
    from dynamo.vllm.llm_engine import VllmLLMEngine

    def main():
        run(VllmLLMEngine)
"""

import asyncio
import logging
import signal

import uvloop

from .engine import LLMEngine
from .worker import Worker

logger = logging.getLogger(__name__)


async def _start(engine_cls: type[LLMEngine], argv: list[str] | None = None):
    engine, worker_config = await engine_cls.from_args(argv)
    w = Worker(engine, worker_config)
    start_succeeded = False
    try:
        await w.run()
        start_succeeded = True
    finally:
        if start_succeeded:
            loop = asyncio.get_running_loop()
            try:
                await asyncio.wait_for(engine.cleanup(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Engine cleanup timed out after 10s — force exiting")
            except Exception as e:
                logger.warning("Engine cleanup raised: %s — force exiting", e)

            # Stop the loop so uvloop.run() returns; Phase 3 (engine cleanup)
            # has already completed via the Rust side before Worker.run()
            # returns, so stopping the loop here is safe.
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.remove_signal_handler(sig)
                except (ValueError, OSError):
                    pass
            loop.stop()


def run(engine_cls: type[LLMEngine], argv: list[str] | None = None):
    """Entry point for per-backend unified_main.py files."""
    uvloop.run(_start(engine_cls, argv))
