# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async fetch loop with per-request latency capture.

Sets ``DYN_HTTP_BACKEND`` for the chosen backend, calls
``close_http_client()`` before+after to defend against stale singletons
across runs in the same process. Returns raw samples; aggregation is in
``stats.py``.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass

from dynamo.common.http import close_http_client, fetch_bytes


@dataclass
class RunResult:
    backend: str
    n: int
    wall_s: float
    samples: list[tuple[float, str]]


def gen_urls(seeds: list[str], n: int) -> list[str]:
    return [f"{seeds[i % len(seeds)]}?cb={uuid.uuid4().hex}" for i in range(n)]


async def _timed_fetch(url: str, timeout: float) -> tuple[float, str]:
    start = time.perf_counter()
    try:
        await fetch_bytes(url, timeout=timeout)
        return (time.perf_counter() - start, "success")
    except Exception as e:
        return (time.perf_counter() - start, type(e).__name__)


async def run_one(
    backend: str, urls: list[str], timeout: float, request_rate: float
) -> RunResult:
    os.environ["DYN_HTTP_BACKEND"] = backend
    await close_http_client()
    interval = 1.0 / request_rate
    t0 = time.perf_counter()
    tasks: list[asyncio.Task[tuple[float, str]]] = []
    for i, u in enumerate(urls):
        delay = (t0 + i * interval) - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(_timed_fetch(u, timeout)))
    samples = await asyncio.gather(*tasks)
    wall_s = time.perf_counter() - t0
    await close_http_client()
    return RunResult(backend=backend, n=len(urls), wall_s=wall_s, samples=samples)
