# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Counts items at the moment they leave the asyncio loop.

This is the definition of the benchmark's score, so it is worth being exact
about where the tick goes.

Measuring at the *system* exit -- after the tokio-side consumer has drained
them -- is wrong, and wrong in a way that silently understates a good
architecture. Under saturation that consumer can itself fall behind, and then
the loop's work is divided by however many items happened to make it all the
way out. A run measured that way reported 219 us of loop work per item against
a modelled 85: the loop had processed ~83,000 items and only ~32,000 had
reached the far end.

So the tick goes at the last point the loop touches an item:

* push -- ``ResponseSender.send``, which runs on the loop under the GIL the
  handler already holds,
* pull -- the return of ``anext_call``, which completes on the loop.

**Every architecture must tick exactly once per item, on the loop thread.** An
architecture that forgets scores ~0; one that ticks off-loop is caught by
:func:`report`, which records the ticking thread and the benchmark prints it.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List

_perf = time.perf_counter_ns

#: Append-only, and ``list.append`` is atomic under the GIL, so no lock is
#: needed on the hot path. One entry per item.
_TIMES: List[int] = []
_THREADS: Dict[str, int] = {}


def reset() -> None:
    del _TIMES[:]
    _THREADS.clear()


def item() -> None:
    """One item finished on the loop. Call exactly once, from the loop."""
    _TIMES.append(_perf())
    name = threading.current_thread().name
    _THREADS[name] = _THREADS.get(name, 0) + 1


def count() -> int:
    """Items the loop has finished, O(1).

    Used for backpressure accounting on a 10 Hz sampler, where copying the
    timestamp list would itself become the cost.
    """
    return len(_TIMES)


def timestamps() -> List[int]:
    return list(_TIMES)


def report() -> Dict[str, int]:
    """Ticks by thread name. Should be a single entry: the loop thread."""
    return dict(_THREADS)
