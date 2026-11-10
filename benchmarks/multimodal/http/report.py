# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain-text formatting for sweep benchmark output."""

from __future__ import annotations

from .stats import Summary


def print_batch_header(*, request_rate: float, requests: int) -> None:
    print(f"== request_rate={request_rate:g} rps  requests={requests} ==")


def print_iteration(mean_ms: float, summary: Summary) -> None:
    print(f"=== mean_ms={mean_ms:g}  n={summary.n} ===")
    print(
        f"{'wall(s)':>7} {'avg':>7} {'p50':>7} {'p90':>7} {'p99':>7}"
        f"  {'success':>7} {'Timeout':>7} {'Status':>6} {'Conn':>4}"
    )
    print(
        f"{summary.wall_s:>7.1f} {summary.avg_ms:>7.1f} {summary.p50_ms:>7.1f} "
        f"{summary.p90_ms:>7.1f} {summary.p99_ms:>7.1f}  "
        f"{summary.outcomes.get('success', 0):>7} "
        f"{summary.outcomes.get('HttpTimeoutError', 0):>7} "
        f"{summary.outcomes.get('HttpStatusError', 0):>6} "
        f"{summary.outcomes.get('HttpConnectionError', 0):>4}"
    )
    print()


def print_grid(rows: list[tuple[float, Summary]]) -> None:
    if not rows:
        return
    print()
    print(f"{'mean_ms':>7} | {'wall / p50 / p99 / err':>34}")
    print(f"{'-' * 7}-+-{'-' * 34}")
    for mean_ms, s in rows:
        err = s.n - s.outcomes.get("success", 0)
        print(
            f"{mean_ms:>7.0f} | "
            f"{s.wall_s:>6.1f} / {s.p50_ms:>6.1f} / {s.p99_ms:>7.1f} / {err:>4}"
        )
