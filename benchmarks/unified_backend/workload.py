# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workload definitions for the unified-backend bridge microbenchmark.

A :class:`Workload` is one request shape; the harness runs each workload at
every concurrency level in the sweep. Concurrency is the GIL-contention knob:
the per-token ``spawn_blocking`` + ``with_gil`` hop in the bridge serializes
across in-flight requests, so the bridge/floor gap should widen as concurrency
rises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Concurrency sweep — the primary axis. Pure throughput at 1 isolates
# per-request bridge cost; the higher rungs expose GIL contention.
DEFAULT_CONCURRENCY = [1, 4, 16, 64, 128]


@dataclass(frozen=True)
class Workload:
    """One request shape driven through both engines.

    ``per_token_delay_ms`` is applied identically on both sides (the Python
    engine's ``asyncio.sleep`` and the Rust floor's ``tokio::sleep``). Default
    ``0.0`` exposes pure bridge + GIL overhead with no pacing; set it to model
    a realistically-paced engine where the delta shrinks relative to compute.
    """

    name: str
    prompt_len: int = 256
    max_tokens: int = 128
    logprobs_k: Optional[int] = None
    per_token_delay_ms: float = 0.0
    total_requests: int = 512


def default_sweep() -> list[Workload]:
    """Workloads chosen to stress the bridge along different axes."""
    return [
        # Baseline: short prompt, moderate decode, no logprobs.
        Workload(name="base", prompt_len=256, max_tokens=128),
        # Logprobs inflate per-chunk payload → more depythonize work per token.
        Workload(name="logprobs_k5", prompt_len=256, max_tokens=128, logprobs_k=5),
        # Long decode: amortizes per-request cost, isolates per-token hop cost.
        Workload(name="long_decode", prompt_len=512, max_tokens=512),
    ]
