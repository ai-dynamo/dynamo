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

from collections.abc import Awaitable, Callable

import uvloop

from .engine import BaseEngine
from .worker import Worker, WorkerConfig

EngineFactory = Callable[[list[str] | None], Awaitable[tuple[BaseEngine, WorkerConfig]]]


async def _start(
    engine_cls: type[BaseEngine],
    argv: list[str] | None = None,
    engine_factory: EngineFactory | None = None,
):
    from_args = engine_factory or engine_cls.from_args
    engine, worker_config = await from_args(argv)
    w = Worker(engine, worker_config)
    await w.run()


def run(
    engine_cls: type[BaseEngine],
    argv: list[str] | None = None,
    engine_factory: EngineFactory | None = None,
):
    """Entry point for per-backend unified_main.py files.

    ``engine_cls`` may be an :class:`LLMEngine` or a :class:`DiffusionEngine`
    subclass; both share the ``from_args -> (engine, WorkerConfig)`` contract.

    ``engine_factory`` lets an entry point that already parsed args construct
    its engine without parsing again. The default remains
    ``engine_cls.from_args(argv)`` so backend-specific arguments do not leak
    into the shared :class:`BaseEngine` contract.
    """
    uvloop.run(_start(engine_cls, argv, engine_factory))
