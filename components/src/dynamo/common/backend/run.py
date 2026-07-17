# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common entry point for backend workers.

A backend's ``main`` calls :func:`run` with its ``BaseEngine`` subclass.
Example::

    from dynamo.common.backend.run import run
    from dynamo.common.backend.sample_engine import SampleLLMEngine

    def main():
        run(SampleLLMEngine)
"""

import uvloop

from .engine import BaseEngine
from .worker import Worker


async def _start(engine_cls: type[BaseEngine], argv: list[str] | None = None):
    engine, worker_config = await engine_cls.from_args(argv)
    w = Worker(engine, worker_config)
    await w.run()


def run(engine_cls: type[BaseEngine], argv: list[str] | None = None):
    """Entry point for per-backend unified_main.py files.

    ``engine_cls`` may be an :class:`LLMEngine` or a :class:`DiffusionEngine`
    subclass; both share the ``from_args -> (engine, WorkerConfig)`` contract.
    """
    uvloop.run(_start(engine_cls, argv))
