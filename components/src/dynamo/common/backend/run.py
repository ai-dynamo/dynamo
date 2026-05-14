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

import uvloop

from .engine import LLMEngine
from .worker import Worker


async def _start(engine_cls: type[LLMEngine], argv: list[str] | None = None):
    engine, worker_config = await engine_cls.from_args(argv)
    w = Worker(engine, worker_config)
    await w.run()


def run(engine_cls: type[LLMEngine], argv: list[str] | None = None):
    """Entry point for per-backend unified_main.py files."""
    uvloop.run(_start(engine_cls, argv))
