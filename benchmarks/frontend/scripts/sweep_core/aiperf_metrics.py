# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode AIPerf ``profile_export_aiperf.json`` metrics into sweep result fields."""

from __future__ import annotations

from typing import Any, Iterator


def iter_aiperf_metrics(data: dict) -> Iterator[tuple[str, Any]]:
    """Yield ``(RunResult field, value)`` pairs from decoded AIPerf JSON, in order.

    Each value is computed only when the next pair is requested, so a caller that
    assigns as it goes keeps the fields set before a mid-sequence error.
    Throughput averages are passed through raw; callers normalize if they want.
    Latency percentiles are omitted when the latency entry is not a dict.
    """
    yield "req_per_sec", data.get("request_throughput", {}).get("avg", 0)
    yield "output_tok_per_sec", data.get("output_token_throughput", {}).get("avg", 0)
    ttft = data.get("time_to_first_token", data.get("ttft", {}))
    if isinstance(ttft, dict):
        yield "ttft_p50_ms", ttft.get("p50", 0) or 0
        yield "ttft_p99_ms", ttft.get("p99", 0) or 0
    itl = data.get("inter_token_latency", data.get("itl", {}))
    if isinstance(itl, dict):
        yield "itl_p50_ms", itl.get("p50", 0) or 0
        yield "itl_p99_ms", itl.get("p99", 0) or 0
    bd = data.get("benchmark_duration", 0)
    yield "duration_sec", bd.get("avg", 0) if isinstance(bd, dict) else (bd or 0)
