# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async fetch loop that records the latency of each request.

Each request goes through ``dynamo.common.http.fetch_bytes``, which uses
aiohttp. ``run_one`` calls ``close_http_client()`` before and after each run,
so that a run does not reuse the warm connection pool of the previous run.
This module returns raw samples, and ``stats.py`` aggregates them.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

from dynamo.common.http import close_http_client, fetch_bytes


@dataclass
class RunResult:
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


async def run_one(urls: list[str], timeout: float, request_rate: float) -> RunResult:
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
    return RunResult(n=len(urls), wall_s=wall_s, samples=samples)
