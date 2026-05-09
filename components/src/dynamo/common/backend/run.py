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

import os
import threading
import time

import uvloop

from .engine import LLMEngine
from .worker import Worker


async def _start(engine_cls: type[LLMEngine], argv: list[str] | None = None):
    engine, worker_config = await engine_cls.from_args(argv)
    w = Worker(engine, worker_config)
    await w.run()


def _shutdown_watchdog() -> None:
    time.sleep(2)
    # 911 mirrors the Rust orchestrator's deadline backstop in
    # lib/backend-common/src/worker.rs so monitoring sees a forced
    # termination rather than a clean exit.
    os._exit(911)


def run(engine_cls: type[LLMEngine], argv: list[str] | None = None):
    """Entry point for per-backend unified_main.py files."""
    uvloop.run(_start(engine_cls, argv))
    # Engines leak non-daemon threads that block exit after Worker.run
    # cleaned up; force-exit backstop. Issue #9343.
    threading.Thread(
        target=_shutdown_watchdog,
        daemon=True,
        name="dynamo-backend-shutdown-watchdog",
    ).start()
