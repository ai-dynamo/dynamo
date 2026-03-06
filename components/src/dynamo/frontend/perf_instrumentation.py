#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Performance instrumentation for diagnosing frontend preprocessing bottlenecks.

Activated by passing --dyn-debug-perf to dynamo.frontend.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concurrency gauge
# ---------------------------------------------------------------------------

_active_requests = 0
_peak_requests = 0


def enter_generator() -> int:
    """Increment active request count. Returns current count.

    Safe without a lock: only called while the GIL is held (all callers are
    in Python code), so the read-modify-write on the global int is atomic
    with respect to other Python threads.
    """
    global _active_requests, _peak_requests
    _active_requests += 1
    count = _active_requests
    if count > _peak_requests:
        _peak_requests = count
    return count


def exit_generator() -> int:
    """Decrement active request count. Returns current count."""
    global _active_requests
    _active_requests -= 1
    return _active_requests


def get_active_requests() -> int:
    return _active_requests


def get_peak_requests() -> int:
    return _peak_requests
