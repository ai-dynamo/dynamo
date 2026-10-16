#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""
Performance instrumentation for diagnosing frontend preprocessing bottlenecks.

Activated by passing --dyn-debug-perf to dynamo.frontend.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_active_requests = 0


def enter_generator() -> int:
    """Increment and return the active request count.

    Safe without a lock: both enter and exit are only called from the
    asyncio event-loop thread (one task at a time between awaits), so
    the read-modify-write cannot interleave.
    """
    global _active_requests
    _active_requests += 1
    return _active_requests


def exit_generator() -> int:
    """Decrement and return the active request count."""
    global _active_requests
    _active_requests -= 1
    return _active_requests
