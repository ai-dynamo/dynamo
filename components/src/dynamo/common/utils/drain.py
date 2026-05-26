# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prefill-drain helper for disaggregated workers.

Prefill workers must hold GPU memory until in-flight KV transfers to
decode peers complete; otherwise decode hits a use-after-free during
teardown. Each backend's :meth:`LLMEngine.drain` wraps its
engine-specific body in :func:`prefill_drain_context`; cadence and
logging are shared, the body is what differs.

Environment:

* ``DYN_PREFILL_DRAIN_TIMEOUT_S`` (default 30) -- max time spent in
  drain before proceeding to cleanup.
* ``DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT`` (Rust ``Worker``, default
  10) must be raised above
  ``DYN_PREFILL_DRAIN_TIMEOUT_S + DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS``
  so the outer budget does not force-exit the process mid-drain.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

DRAIN_TIMEOUT_ENV = "DYN_PREFILL_DRAIN_TIMEOUT_S"
DEFAULT_DRAIN_TIMEOUT_S = 30.0
HEARTBEAT_INTERVAL_S = 5.0


def _resolve_timeout_s() -> float:
    raw = os.getenv(DRAIN_TIMEOUT_ENV)
    if not raw:
        return DEFAULT_DRAIN_TIMEOUT_S
    try:
        parsed = float(raw)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Invalid %s=%r; using default %.1fs",
            DRAIN_TIMEOUT_ENV,
            raw,
            DEFAULT_DRAIN_TIMEOUT_S,
        )
        return DEFAULT_DRAIN_TIMEOUT_S
    if not math.isfinite(parsed):
        logging.getLogger(__name__).warning(
            "Non-finite %s=%r; using default %.1fs",
            DRAIN_TIMEOUT_ENV,
            raw,
            DEFAULT_DRAIN_TIMEOUT_S,
        )
        return DEFAULT_DRAIN_TIMEOUT_S
    if parsed < 0:
        logging.getLogger(__name__).warning(
            "Negative %s=%r; clamping to 0", DRAIN_TIMEOUT_ENV, raw
        )
        return 0.0
    return parsed


class DrainCtx:
    """Drain-loop state: deadline tracking plus throttled heartbeat."""

    def __init__(
        self,
        logger: logging.Logger,
        started: float,
        timeout_s: float,
        heartbeat_s: float,
    ) -> None:
        self._logger = logger
        self._started = started
        self._timeout_s = timeout_s
        self._heartbeat_s = heartbeat_s
        self._last_hb = started

    @property
    def heartbeat_s(self) -> float:
        return self._heartbeat_s

    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    def remaining_s(self) -> float:
        return max(0.0, self._started + self._timeout_s - self._now())

    def expired(self) -> bool:
        return self._now() >= self._started + self._timeout_s

    def heartbeat(self, **extra: object) -> None:
        """Log a heartbeat if at least ``heartbeat_s`` has passed since
        the last one. ``extra`` kwargs append as ``k=v`` pairs."""
        now = self._now()
        if now - self._last_hb < self._heartbeat_s:
            return
        elapsed = now - self._started
        if extra:
            extras = ", ".join(f"{k}={v}" for k, v in extra.items())
            self._logger.info("drain: heartbeat (elapsed=%.1fs, %s)", elapsed, extras)
        else:
            self._logger.info("drain: heartbeat (elapsed=%.1fs)", elapsed)
        self._last_hb = now


@asynccontextmanager
async def prefill_drain_context(
    logger: logging.Logger,
) -> AsyncIterator[DrainCtx]:
    """Bounded drain scope. The caller's body runs the backend-specific
    signal inside the ``async with``; this owns entry/exit logging and
    timing."""
    timeout_s = _resolve_timeout_s()
    loop = asyncio.get_running_loop()
    started = loop.time()
    logger.info("drain: entered (disagg=PREFILL, timeout=%.1fs)", timeout_s)
    try:
        yield DrainCtx(logger, started, timeout_s, HEARTBEAT_INTERVAL_S)
    finally:
        logger.info("drain: exited (elapsed=%.1fs)", loop.time() - started)
