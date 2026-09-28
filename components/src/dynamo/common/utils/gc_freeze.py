#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""CPython GC pause mitigation for long-lived Dynamo Python processes.

CPython's cyclic collector runs a full (gen2) collection once the number of
long-lived objects pending examination exceeds 25% of the long-lived total.
Every full collection walks every tracked object while holding the GIL, so
its pause scales with heap size and its frequency with allocation rate.

Both the ``dynamo.frontend`` process (with ``--dyn-chat-processor
vllm|sglang``) and the ``dynamo.vllm`` worker parent hold an engine's Python
imports, a tokenizer, parser registries and config objects: on the order of
a million tracked objects that live for the process lifetime. Per-request
work then pushes the pending count over the 25% threshold every few seconds
under load, and each resulting gen2 pass stalls the main thread (and with it
every in-flight request, since the event loop and any helper threads all
need the GIL) for hundreds of milliseconds to seconds. Measured on one
A100 with a 4B embedding model: the worker parent stalled 1.1-1.7 s at
320 RPS (EngineCore idle, starved of work); freezing its heap removed every
such stall and raised the completed-RPS ceiling ~10%, after which the only
remaining >300 ms outliers were 0.2-0.7 s stalls of the same shape in the
frontend process.

``gc.freeze()`` moves everything currently tracked into the permanent
generation, which no collection ever scans again; later gen2 passes walk only
objects allocated since the freeze, so their cost is bounded by the working
set rather than the static heap. vLLM applies the same fix in three places
(``gc.collect(); gc.freeze()`` in its API-server lifespan, and
``freeze_gc_heap()`` in ``EngineCore.__init__`` and each GPU worker after
warmup), but none of those run in Dynamo's own processes. The trade-off is
that a startup-era object which later dies is never reclaimed; the static
heap this targets does not die, so that cost is ~0.

Two entry points:

* :func:`freeze_gc_heap` -- collect every generation, then freeze. Call it
  once the static heap is built (a model is registered, a processor is
  constructed). Calling it again after more static state is built is fine:
  each call pins whatever is alive at that moment.
* :func:`install_gc_pause_logger` -- opt-in ``gc.callbacks`` hook that logs
  any collection whose wall time is at or above a threshold, so the freeze
  can be verified (or a remaining stall attributed) from the process log
  rather than inferred from per-thread CPU samples.

Not to be confused with ``dynamo.vllm.gc_policy``, which is the FPM
self-benchmark's periodic-freeze loop for the vLLM GPU worker processes and
is off unless ``DYN_FPM_GC_POLICY`` is set.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# `gc.callbacks` entries are process-global; guard installs so a second
# call (a re-run factory, a test) cannot double-log every collection.
_pause_logger_lock = threading.Lock()
_pause_logger: GcPauseLogger | None = None


def freeze_gc_heap(*, context: str) -> int:
    """Collect every generation, then freeze the surviving heap.

    Mirrors ``vllm.utils.gc_utils.freeze_gc_heap``. The collection first
    reclaims any cyclic garbage that would otherwise be pinned forever and
    promotes every survivor to the oldest generation; the freeze then moves
    that whole set into the permanent generation. Returns the number of
    frozen objects (``gc.get_freeze_count()``) for the log line the A/B
    verification greps for.

    Runs synchronously and holds the GIL for the duration of one full
    collection (hundreds of ms on a ~1M-object heap). Call it from startup
    paths only, never from a request.
    """
    t0 = time.perf_counter()
    gc.collect()
    gc.freeze()
    frozen = gc.get_freeze_count()
    logger.info(
        "GC heap frozen after %s: gc.get_freeze_count()=%d "
        "(collect+freeze %.0f ms, pid=%d)",
        context,
        frozen,
        (time.perf_counter() - t0) * 1000.0,
        os.getpid(),
    )
    return frozen


def maybe_freeze_gc_heap(enabled: bool, *, context: str) -> int | None:
    """:func:`freeze_gc_heap` when ``enabled``; otherwise a logged no-op.

    Returns the frozen count, or ``None`` when disabled so callers and tests
    can tell "skipped" from "froze nothing".
    """
    if not enabled:
        logger.debug("GC heap freeze disabled; skipping after %s", context)
        return None
    return freeze_gc_heap(context=context)


class GcPauseLogger:
    """``gc.callbacks`` hook that logs collections at or above a threshold.

    CPython invokes callbacks with ``("start", info)`` before and
    ``("stop", info)`` after each collection, on the thread that triggered it
    and with the GIL held, where ``info`` carries ``generation``,
    ``collected`` and ``uncollectable``. Collections of different generations
    cannot interleave within one interpreter, so a per-generation start time
    is enough to pair them; the dict is written only under the GIL.
    """

    def __init__(self, threshold_ms: float, log: logging.Logger = logger) -> None:
        self.threshold_ms = threshold_ms
        self._log = log
        self._starts: dict[int, float] = {}
        # Cheap counters for tests and for an operator reading the log tail.
        self.pauses_logged = 0
        self.max_pause_ms = 0.0

    def __call__(self, phase: str, info: dict[str, Any]) -> None:
        generation = info.get("generation", -1)
        if phase == "start":
            self._starts[generation] = time.perf_counter()
            return
        if phase != "stop":
            return
        started = self._starts.pop(generation, None)
        if started is None:
            return
        pause_ms = (time.perf_counter() - started) * 1000.0
        if pause_ms > self.max_pause_ms:
            self.max_pause_ms = pause_ms
        if pause_ms < self.threshold_ms:
            return
        self.pauses_logged += 1
        # Deliberately no gc.get_freeze_count()/gc.get_objects() here: both
        # walk object lists and would add their own pause to every collection.
        self._log.warning(
            "GC gen%d pause %.1f ms (collected=%d, uncollectable=%d, "
            "thresholds=%s, pid=%d)",
            generation,
            pause_ms,
            info.get("collected", 0),
            info.get("uncollectable", 0),
            gc.get_threshold(),
            os.getpid(),
        )


def install_gc_pause_logger(threshold_ms: float) -> GcPauseLogger | None:
    """Register a :class:`GcPauseLogger` once per process.

    A non-positive ``threshold_ms`` leaves ``gc.callbacks`` untouched and
    returns ``None``: the hook costs two ``perf_counter`` calls per
    collection, which is negligible, but there is no reason to pay it when
    nothing would be logged. A second call returns the already-installed
    logger (with its original threshold) rather than stacking hooks.
    """
    global _pause_logger
    if threshold_ms <= 0:
        return None
    with _pause_logger_lock:
        if _pause_logger is not None:
            return _pause_logger
        pause_logger = GcPauseLogger(threshold_ms)
        gc.callbacks.append(pause_logger)
        _pause_logger = pause_logger
    logger.info(
        "GC pause logging enabled: collections >= %.1f ms are logged (pid=%d)",
        threshold_ms,
        os.getpid(),
    )
    return pause_logger


def uninstall_gc_pause_logger() -> None:
    """Remove the process-wide pause logger, if any. Intended for tests."""
    global _pause_logger
    with _pause_logger_lock:
        pause_logger = _pause_logger
        _pause_logger = None
    if pause_logger is not None:
        try:
            gc.callbacks.remove(pause_logger)
        except ValueError:
            pass
