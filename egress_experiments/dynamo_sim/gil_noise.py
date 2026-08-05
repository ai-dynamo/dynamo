# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The other GIL-capable threads in the app interpreter.

``ASYNCIO_GIL_PATH.md`` locates the pull path's cost precisely::

    On the decode worker -- 45 GIL-capable threads in the app interpreter
    versus trtllm-serve's 3, and a GIL wait/hold ratio of 23.4 versus serve's
    0.3 -- those cross-thread acquisitions are the expensive ones.

A bare simulation has three: the event loop, ``proxy_dispatch_result_thread``,
and the tokio stand-in. At that thread count a cross-thread GIL acquisition is
nearly free, so the *structural* claims (batch ratio, deque entries, loop load)
reproduce faithfully while the pull-vs-push *latency* difference does not --
its entire mechanism is contention that is absent.

This module supplies the missing threads. Each one does what the real worker's
background threads do: wake periodically, hold the GIL briefly, sleep. With
``sys.getswitchinterval()`` at its 5 ms default, every waiting thread is a
chance for the loop to be forced to drop the GIL mid-stage, and every
cross-thread acquisition the pull path adds is another queue to join.

Off by default (``threads=0``), because it is a knob for reproducing a regime,
not part of the baseline model. Set it explicitly and say so when reporting.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from egress_experiments.costs import spin


@dataclass
class GilNoiseConfig:
    #: How many extra GIL-capable threads. The decode worker had ~45 total.
    threads: int = 0
    #: GIL held per wake-up.
    hold_us: float = 20.0
    #: Sleep between wake-ups (GIL released).
    period_us: float = 200.0


class GilNoise:
    """Start/stop a pool of background GIL contenders."""

    def __init__(self, config: GilNoiseConfig) -> None:
        self.config = config
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        #: Total wake-ups, so a report can state the contention actually applied.
        self.wakeups = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        cfg = self.config
        if cfg.threads <= 0:
            return
        sleep_s = cfg.period_us / 1e6

        def worker() -> None:
            local = 0
            while not self._stop.is_set():
                spin(cfg.hold_us)
                local += 1
                time.sleep(sleep_s)
            with self._lock:
                self.wakeups += local

        for i in range(cfg.threads):
            thread = threading.Thread(target=worker, name=f"gil-noise-{i}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()

    def __enter__(self) -> "GilNoise":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
