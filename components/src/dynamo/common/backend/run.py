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

from .engine import BaseEngine
from .worker import Worker


async def _start(
    engine_cls: type[BaseEngine], argv: list[str] | None = None, config=None
):
    # Forward a pre-parsed config only when supplied, so engines whose
    # from_args doesn't accept `config` keep their original call shape.
    if config is not None:
        engine, worker_config = await engine_cls.from_args(argv, config=config)
    else:
        engine, worker_config = await engine_cls.from_args(argv)
    w = Worker(engine, worker_config)
    await w.run()


def run(engine_cls: type[BaseEngine], argv: list[str] | None = None, config=None):
    """Entry point for per-backend unified_main.py files.

    ``engine_cls`` may be an :class:`LLMEngine` or a :class:`DiffusionEngine`
    subclass; both share the ``from_args -> (engine, WorkerConfig)`` contract.

    ``config`` lets an entry point that already parsed args (e.g. the vLLM
    headless check in ``unified_main``) thread the parsed config through so
    ``from_args`` does not re-parse; engines whose ``from_args`` does not
    accept ``config`` are only ever called without it.
    """
    uvloop.run(_start(engine_cls, argv, config))
