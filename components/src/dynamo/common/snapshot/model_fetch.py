# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-safe model prefetch helpers."""

import asyncio
import logging
import multiprocessing
import os
import sys

from dynamo.common.snapshot.lifecycle import SENTINEL_POLL_INTERVAL_SEC

logger = logging.getLogger(__name__)


def _fetch_model_process_main(remote_name: str, ignore_weights: bool) -> None:
    from dynamo.llm import fetch_model

    async def fetch_model_async() -> None:
        await fetch_model(remote_name, ignore_weights)

    asyncio.run(fetch_model_async())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


async def fetch_model_in_subprocess(
    remote_name: str,
    ignore_weights: bool = False,
) -> None:
    """Fetch a model in a short-lived process before checkpointing."""
    logger.info("Fetching model %s in a subprocess", remote_name)
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_fetch_model_process_main,
        args=(remote_name, ignore_weights),
        name="dynamo-model-fetch",
    )
    proc.start()
    try:
        while proc.is_alive():
            await asyncio.sleep(SENTINEL_POLL_INTERVAL_SEC)
        proc.join()
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join()

    if proc.exitcode != 0:
        raise RuntimeError(
            f"Model fetch subprocess failed for {remote_name!r} "
            f"with exit code {proc.exitcode}"
        )
