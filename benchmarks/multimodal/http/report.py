# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain-text formatting for sweep benchmark output."""

from __future__ import annotations

from .stats import Summary

_DISPLAY_ORDER = {"httpx": 0, "aiohttp": 1}


def print_batch_header(*, request_rate: float, requests: int) -> None:
    print(f"== request_rate={request_rate:g} rps  requests={requests} ==")


def print_iteration(mean_ms: float, summaries: list[Summary]) -> None:
    if not summaries:
        return
    n = summaries[0].n
    print(f"=== mean_ms={mean_ms:g}  n={n} ===")
    print(
        f"{'backend':<8} {'wall(s)':>7} {'avg':>7} {'p50':>7} {'p90':>7} {'p99':>7}"
        f"  {'success':>7} {'Timeout':>7} {'Status':>6} {'Conn':>4}"
    )
    for s in sorted(summaries, key=lambda x: _DISPLAY_ORDER.get(x.backend, 99)):
        print(
            f"{s.backend:<8} {s.wall_s:>7.1f} {s.avg_ms:>7.1f} {s.p50_ms:>7.1f} "
            f"{s.p90_ms:>7.1f} {s.p99_ms:>7.1f}  "
            f"{s.outcomes.get('success', 0):>7} "
            f"{s.outcomes.get('HttpTimeoutError', 0):>7} "
            f"{s.outcomes.get('HttpStatusError', 0):>6} "
            f"{s.outcomes.get('HttpConnectionError', 0):>4}"
        )
    print()


def print_grid(rows: list[tuple[float, Summary, Summary]]) -> None:
    if not rows:
        return
    print()
    print(
        f"{'mean_ms':>7} | "
        f"{'aiohttp wall / p50 / p99 / err':>34} | "
        f"{'httpx wall / p50 / p99 / err':>34}"
    )
    print(f"{'-' * 7}-+-{'-' * 34}-+-{'-' * 34}")
    for mean_ms, aio, htx in rows:
        aio_err = aio.n - aio.outcomes.get("success", 0)
        htx_err = htx.n - htx.outcomes.get("success", 0)
        print(
            f"{mean_ms:>7.0f} | "
            f"{aio.wall_s:>6.1f} / {aio.p50_ms:>6.1f} / {aio.p99_ms:>7.1f} / {aio_err:>4} | "
            f"{htx.wall_s:>6.1f} / {htx.p50_ms:>6.1f} / {htx.p99_ms:>7.1f} / {htx_err:>4}"
        )
