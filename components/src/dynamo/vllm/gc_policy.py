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
  * engine-core process - ``InstrumentedScheduler`` imports this module,
    calls :func:`start_gc_policy` when benchmarking starts, and
    :func:`stop_gc_policy` when it deactivates.
"""

from __future__ import annotations

import gc
import logging
import os
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_started = False
_stop_event: threading.Event | None = None
_freeze_thread: threading.Thread | None = None
_saved_gen2_threshold: int | None = None


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
    """Reclaim cyclic garbage from an untimed window, then re-freeze.

    Unfreeze first: the periodic ticks freeze unreachable-but-uncollected
    cycles into the permanent generation, which ``gc.collect()`` never
    scans. The full-heap walk this costs is why callers must be outside
    any measured window.
    """
    gc.unfreeze()
    gc.collect()
    gc.freeze()
    return gc.get_freeze_count()


def _freeze_loop(interval: float, stop: threading.Event) -> None:
    while not stop.wait(interval):
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
    global _started, _stop_event, _freeze_thread, _saved_gen2_threshold
    if _policy() != "freeze":
        return False
    with _lock:
        if _started:
            return True
        # One-time setup. Automatic full (gen2) collections are the pause
        # source: they walk every tracked object and grow with the heap
        # (measured 0.4s -> 9s over a sweep). Disable them outright; gen0/1
        # stay enabled and only walk objects younger than the last freeze.
        t0, t1, _saved_gen2_threshold = gc.get_threshold()
        gc.set_threshold(t0, t1, 1 << 30)
        gc.freeze()
        interval = _interval_seconds()
        _stop_event = threading.Event()
        _freeze_thread = threading.Thread(
            target=_freeze_loop,
            args=(interval, _stop_event),
            name="fpm-gc-freeze",
            daemon=True,
        )
        _freeze_thread.start()
        _started = True
    logger.info(
        "FPM GC policy active: auto-gen2 disabled, gc.freeze() every %.0fs (pid=%d)",
        interval,
        os.getpid(),
    )
    return True


def stop_gc_policy() -> None:
    """Restore normal GC once benchmarking ends: stop the freeze loop,
    re-enable automatic gen2 collections, and reclaim everything frozen so
    far. Idempotent; no-op if the policy never started."""
    global _started, _stop_event, _freeze_thread, _saved_gen2_threshold
    with _lock:
        if not _started:
            return
        _stop_event.set()
        thread = _freeze_thread
        t0, t1, _ = gc.get_threshold()
        gc.set_threshold(t0, t1, _saved_gen2_threshold)
        _started = False
        _stop_event = None
        _freeze_thread = None
        _saved_gen2_threshold = None
    thread.join(timeout=5.0)
    gc.unfreeze()
    gc.collect()
    logger.info("FPM GC policy stopped: auto-gen2 restored (pid=%d)", os.getpid())


class FpmGcWorkerExtension:
    """vLLM worker extension exposing GC control inside worker processes."""

    def fpm_gc_start(self) -> bool:
        return start_gc_policy()

    def fpm_gc_stop(self) -> None:
        return stop_gc_policy()

    def fpm_gc_maintain(self) -> int:
        return gc_maintain()


# Auto-start when a worker process imports this module while resolving
# worker_extension_cls (args.py injects the class path as a literal for the
# same reason: importing this module starts the policy in the importing
# process). The engine-core process instead imports lazily and starts from
# InstrumentedScheduler._bench_init, so serving without an active benchmark
# never runs the policy. No-op unless DYN_FPM_GC_POLICY is set.
start_gc_policy()
