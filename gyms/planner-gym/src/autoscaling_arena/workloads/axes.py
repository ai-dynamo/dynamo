# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The three orthogonal workload axes.

Each axis maps to exactly one field of the Mooncake-JSONL trace record, so a
workload composes by picking one value per axis independently:

* **arrival**  -> ``timestamp``                     : the rate schedule λ(t).
* **shape**    -> ``input_length`` / ``output_length`` : the per-request size sampler.
* **prefix**   -> ``hash_ids``                       : block-level cache-sharing structure.

Everything here is deterministic given a seeded ``random.Random``.
"""

from __future__ import annotations

import math
from typing import Callable, List

# An arrival process is a rate function λ(t_seconds) -> requests/second.
RateFn = Callable[[float], float]
# A shape sampler draws one (input_length, output_length) pair.
ShapeFn = Callable[["object"], "tuple[int, int]"]  # rng -> (isl, osl)


# --------------------------------------------------------------------------
# Arrival axis — rate schedules λ(t) and a Poisson sampler
# --------------------------------------------------------------------------


def flat(rate: float) -> RateFn:
    """Constant load."""
    return lambda t: rate


def staircase(base: float, step: float, step_interval_s: float) -> RateFn:
    """Load that steps up by ``step`` req/s every ``step_interval_s``."""
    return lambda t: base + step * math.floor(t / step_interval_s)


def square_wave(low: float, high: float, period_s: float) -> RateFn:
    """Alternates ``high`` and ``low`` each half-period (on/off bursts)."""
    return lambda t: high if (t % period_s) < (period_s / 2.0) else low


def flash_crowd(base: float, spike: float, spike_at_s: float, width_s: float) -> RateFn:
    """Flat ``base`` with a single sharp spike to ``spike`` over a short window."""
    return lambda t: spike if spike_at_s <= t < (spike_at_s + width_s) else base


def diurnal(mean: float, amplitude: float, period_s: float) -> RateFn:
    """Sinusoidal load (a compressed day/night cycle)."""
    return lambda t: max(0.0, mean + amplitude * math.sin(2.0 * math.pi * t / period_s))


def _poisson(rng, lam: float) -> int:
    """Knuth's algorithm — Poisson draw for small/moderate λ."""
    if lam <= 0.0:
        return 0
    target = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= target:
            return k - 1


def poisson_arrivals(
    rate_fn: RateFn, duration_s: float, rng, bin_s: float = 1.0
) -> List[float]:
    """Sample arrival timestamps (ms, sorted) from a time-varying Poisson process.

    Each ``bin_s`` window draws a Poisson count from λ(t)·bin_s and scatters
    those arrivals uniformly within the bin.
    """
    out: List[float] = []
    t = 0.0
    while t < duration_s:
        n = _poisson(rng, max(0.0, rate_fn(t)) * bin_s)
        for _ in range(n):
            out.append((t + rng.random() * bin_s) * 1000.0)
        t += bin_s
    out.sort()
    return out


# --------------------------------------------------------------------------
# Shape axis — per-request (ISL, OSL) samplers
# --------------------------------------------------------------------------


def _lognormal_int(rng, median: float, sigma: float, lo: int = 1) -> int:
    """Lognormal draw rounded to an int >= ``lo`` (median = exp(mu))."""
    v = math.exp(rng.normalvariate(math.log(median), sigma))
    return max(lo, int(round(v)))


def prefill_heavy(rng) -> tuple[int, int]:
    """Long context, short output (Mooncake-like): stresses prefill / TTFT."""
    return _lognormal_int(rng, 6000, 0.8), _lognormal_int(rng, 34, 0.5)


def decode_heavy(rng) -> tuple[int, int]:
    """Short context, long output: stresses decode / ITL and decode capacity."""
    return _lognormal_int(rng, 200, 0.5), _lognormal_int(rng, 600, 0.5)


def balanced(rng) -> tuple[int, int]:
    """Moderate context and output."""
    return _lognormal_int(rng, 1000, 0.5), _lognormal_int(rng, 200, 0.5)


# --------------------------------------------------------------------------
# Prefix axis — block-level hash_ids (cache-sharing structure)
# --------------------------------------------------------------------------


def _n_blocks(isl: int, block_size: int) -> int:
    return max(1, math.ceil(isl / block_size))


class NoSharing:
    """Every request gets unique block hashes — no cross-request cache hits."""

    def __init__(self) -> None:
        self._next = 0

    def assign(self, rng, isl: int, block_size: int) -> List[int]:
        ids = list(range(self._next, self._next + _n_blocks(isl, block_size)))
        self._next += len(ids)
        return ids


class SharedPrefix:
    """A fixed shared prefix of ``shared_blocks`` blocks across all requests.

    Models RAG / shared-system-prompt reuse: the leading blocks hit the prefix
    cache, the tail is unique. ``shared_blocks`` reserves the low id range; unique
    tail blocks are drawn from above it.
    """

    def __init__(self, shared_blocks: int) -> None:
        self._shared = list(range(shared_blocks))
        self._next = shared_blocks

    def assign(self, rng, isl: int, block_size: int) -> List[int]:
        total = _n_blocks(isl, block_size)
        # A hash denotes an exact token block. A partial terminal block cannot
        # reuse a full block (or a differently sized partial block) in AIPerf.
        shared = self._shared[: min(len(self._shared), isl // block_size)]
        tail_n = total - len(shared)
        tail = list(range(self._next, self._next + tail_n))
        self._next += tail_n
        return shared + tail
