# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GC pause mitigation for FPM self-benchmarking.

CPython gen2 (full) collections walk every tracked object and trigger on
the 25% long-lived-growth rule, so their pauses grow with the heap: over a
30k-point benchmark sweep we measured pauses rising from ~0.4s to ~9s at
geometrically spaced iteration indices (ratio ~1.23), inside the vLLM
worker processes. Under TP/DP lockstep a single worker's pause stalls the
whole forward pass, and with the scheduler's host-side wall_time it lands
in the measured latency as a 3-200x outlier.

``gc.freeze()`` moves currently tracked objects into the permanent
generation, excluding them from future collections: after a periodic
freeze, gen2 only walks objects allocated since the previous tick, so the
pause is bounded by the allocation rate instead of total heap size.
The policy's only goal is keeping GC pauses out of measured wall_time:
automatic gen2 collections are disabled entirely and the periodic tick
only freezes (no traversal, no pause). Cyclic garbage is therefore NOT
reclaimed while the policy is active - acceptable for benchmark sweeps;
long-running processes can reclaim it by calling :func:`gc_maintain`
from an untimed window.

The policy is env-gated and off by default:

  DYN_FPM_GC_POLICY=freeze        enable the periodic freeze loop
  DYN_FPM_GC_FREEZE_INTERVAL_S=60 tick interval in seconds

Coverage:
  * worker processes  - pass ``--worker-extension-cls
    dynamo.vllm.gc_policy.FpmGcWorkerExtension``; resolving the class
    imports this module inside every worker, which auto-starts the
    policy (no collective_rpc required). The RPC methods remain
    available for explicit control.
  * engine-core process - ``InstrumentedScheduler`` imports this module
    and calls :func:`start_gc_policy` when benchmarking starts.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_started = False

WORKER_EXTENSION_CLS = "dynamo.vllm.gc_policy.FpmGcWorkerExtension"


def _policy() -> str:
    return os.environ.get("DYN_FPM_GC_POLICY", "").strip().lower()


def _interval_seconds() -> float:
    raw = os.environ.get("DYN_FPM_GC_FREEZE_INTERVAL_S", "60")
    try:
        value = float(raw)
    except ValueError:
        logger.warning("invalid DYN_FPM_GC_FREEZE_INTERVAL_S=%r, using 60", raw)
        return 60.0
    return max(1.0, value)


def gc_maintain() -> int:
    """Collect unfrozen cycles, then freeze survivors. Returns frozen count."""
    gc.collect()
    gc.freeze()
    return gc.get_freeze_count()


def _freeze_loop(interval: float) -> None:
    while True:
        time.sleep(interval)
        try:
            # freeze only: O(1) generation-list splice, no heap traversal,
            # no pause. Never collect on the timer (a periodic collect walks
            # the unfrozen heap) and never call gc.get_freeze_count() here
            # (it walks the whole permanent generation: measured 0.4-4s
            # pauses once millions of objects are frozen).
            gc.freeze()
        except Exception:  # pragma: no cover - defensive
            logger.exception("fpm gc tick failed")


def start_gc_policy() -> bool:
    """Idempotently start the env-gated freeze loop in this process."""
    global _started
    if _policy() != "freeze":
        return False
    with _lock:
        if _started:
            return True
        # One-time setup. Automatic full (gen2) collections are the pause
        # source: they walk every tracked object and grow with the heap
        # (measured 0.4s -> 9s over a sweep). Disable them outright; gen0/1
        # stay enabled and only walk objects younger than the last freeze.
        t0, t1, _t2 = gc.get_threshold()
        gc.set_threshold(t0, t1, 1 << 30)
        gc.freeze()
        interval = _interval_seconds()
        thread = threading.Thread(
            target=_freeze_loop, args=(interval,), name="fpm-gc-freeze", daemon=True
        )
        thread.start()
        _started = True
    logger.info(
        "FPM GC policy active: auto-gen2 disabled, gc.freeze() every %.0fs (pid=%d)",
        interval,
        os.getpid(),
    )
    return True


class FpmGcWorkerExtension:
    """vLLM worker extension exposing GC control inside worker processes."""

    def fpm_gc_start(self) -> bool:
        return start_gc_policy()

    def fpm_gc_maintain(self) -> int:
        return gc_maintain()


# Auto-start in any process that imports this module (worker processes
# import it while resolving worker_extension_cls; the engine-core process
# via InstrumentedScheduler). No-op unless DYN_FPM_GC_POLICY is set.
start_gc_policy()
