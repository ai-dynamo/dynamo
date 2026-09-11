# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tighten the NCCL/collective watchdog timeout once a replica is serving traffic.

During startup, model loading, warmup, CUDA-graph capture, the first cross-node
all-reduce, and load spikes can legitimately take many seconds. The process-group
timeout must therefore remain generous during initialization.

Once the replica is serving steady traffic, individual TP/PP collectives are sub-second,
so a much lower timeout (default 5s) gives fast HANG detection without false positives.
This module lowers the timeout via ``_set_pg_timeout`` once the engine is past warmup.

Pairs with the ZMQ rank-liveness watcher (dynamo.common.rank_liveness): that catches
*crashes* in ~one heartbeat; this catches *hangs* within the serving timeout.

Config:
  DYN_GMS_SERVING_NCCL_TIMEOUT_S        serving timeout in seconds (default 5; <=0 disables)

NOTE on failover latency: the serving timeout governs DETECTION. How fast the process then
*dies* (releasing the failover flock) is governed by TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC —
the monitoring thread waits that long for the flight-recorder dump to coordinate before
aborting. Default 60000ms; measured ~69s exit at the default vs ~5.5s with it set to 1000.
Set TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC=1000 (and TORCH_NCCL_ASYNC_ERROR_HANDLING=1) for
prompt failover. This module warns once if it is left high.
"""

from __future__ import annotations

import datetime
import logging
import math
import os

logger = logging.getLogger(__name__)

_applied = False
_warned = False
# Ids of process groups already tightened ("default" sentinel for the world
# group). Tracking per-PG -- rather than a single latch -- lets a later call
# tighten PGs created after the first application (e.g. TP/PP sub-groups built
# during a deferred warmup) instead of leaving them on the startup timeout.
_applied_pg_ids: set = set()


def serving_timeout_s() -> float:
    raw = os.environ.get("DYN_GMS_SERVING_NCCL_TIMEOUT_S", "5")
    try:
        seconds = float(raw)
        if not math.isfinite(seconds):
            raise ValueError("nonfinite timeout")
        # Reject values outside timedelta range or below its microsecond resolution.
        # Nonpositive values disable the feature and need no duration conversion.
        if seconds > 0 and datetime.timedelta(seconds=seconds) <= datetime.timedelta(0):
            raise ValueError("timeout rounds to zero")
    except (ValueError, OverflowError):
        logger.warning("Invalid DYN_GMS_SERVING_NCCL_TIMEOUT_S=%r; using 5", raw)
        return 5.0
    return seconds


def enabled() -> bool:
    return serving_timeout_s() > 0


def _warn_if_slow_teardown() -> None:
    global _warned
    if _warned:
        return
    _warned = True
    try:
        dump_wait = int(os.environ.get("TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC", "60000"))
    except ValueError:
        dump_wait = 60000
    if dump_wait > 5000:
        logger.warning(
            "[GMS serving-timeout] TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC=%dms: after a "
            "%.1fs hang detection the process may wait this long before aborting, "
            "delaying failover (flock release). Set it to ~1000 for prompt failover.",
            dump_wait,
            serving_timeout_s(),
        )


def apply_serving_collective_timeout(seconds: float | None = None) -> bool:
    """Lower the watchdog timeout on the default PG and every tracked PG. Returns
    True if applied to at least one group. No-op (returns False) if torch.distributed
    is unavailable / not yet initialized, so it is safe to call early or repeatedly."""
    global _applied
    seconds = serving_timeout_s() if seconds is None else seconds
    if not math.isfinite(seconds) or seconds <= 0:
        return False
    try:
        td = datetime.timedelta(seconds=seconds)
    except OverflowError:
        return False
    if td <= datetime.timedelta(0):
        return False
    try:
        import torch.distributed as dist
        from torch.distributed import distributed_c10d as c10d
    except Exception:
        return False
    if not dist.is_available() or not dist.is_initialized():
        return False

    set_fn = getattr(c10d, "_set_pg_timeout", None)
    if set_fn is None:
        logger.warning(
            "[GMS serving-timeout] this torch has no _set_pg_timeout; cannot tighten"
        )
        return False

    applied = 0
    if "default" not in _applied_pg_ids:
        try:
            set_fn(td, None)  # default / world group
            applied += 1
            _applied_pg_ids.add("default")
        except Exception:
            logger.debug(
                "[GMS serving-timeout] set on default group failed", exc_info=True
            )
    # Cover TP/PP sub-groups: each forward-pass collective runs on its own PG whose
    # ProcessGroupNCCL watchdog uses that PG's own timeout. Skip PGs already
    # tightened so a repeat call only handles newly-created groups.
    try:
        world = getattr(c10d, "_world", None)
        pg_map = getattr(world, "pg_map", {}) if world is not None else {}
        for pg in list(pg_map.keys()):
            if id(pg) in _applied_pg_ids:
                continue
            try:
                set_fn(td, pg)
                applied += 1
                _applied_pg_ids.add(id(pg))
            except Exception:
                logger.debug(
                    "[GMS serving-timeout] set on a sub-PG failed", exc_info=True
                )
    except Exception:
        logger.debug("[GMS serving-timeout] iterating PGs failed", exc_info=True)

    if applied:
        _applied = True
        _warn_if_slow_teardown()
        logger.info(
            "[GMS serving-timeout] tightened collective watchdog to %.1fs on %d PG(s)",
            seconds,
            applied,
        )
    return applied > 0


def tighten_now() -> bool:
    """Apply the serving timeout immediately. Called from each engine's precise
    post-warmup hook (vLLM GMSWorker / SGLang Scheduler.run_event_loop)."""
    return apply_serving_collective_timeout()
