# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nonnegative fixed-window average shared by the AIS planner perf model."""

from __future__ import annotations

from collections import deque


class NonnegativeMovingAverage:
    """Fixed-window average that starts at the first positive sample.

    The window holds ``max(1, max_len)`` samples. Samples before the first
    value ``> 0`` are ignored; later samples are clamped to ``>= 0``. ``value``
    is ``0.0`` while the window is empty.

    This differs from the regression ``_MovingAverage`` in ``base.py``, which
    starts at any nonzero sample and keeps negative values.
    """

    def __init__(self, max_len: int) -> None:
        self._values: deque[float] = deque()
        self._sum = 0.0
        self._max_len = max(1, max_len)
        self._seen_nonzero = False

    def add_after_first_nonzero(self, value: float) -> None:
        """Add ``value`` once a positive sample has been seen, clamped to ``>= 0``."""
        if value > 0:
            self._seen_nonzero = True
        if self._seen_nonzero:
            self._add(max(0.0, value))

    def _add(self, value: float) -> None:
        self._values.append(value)
        self._sum += value
        while len(self._values) > self._max_len:
            self._sum -= self._values.popleft()

    @property
    def value(self) -> float:
        """Mean of the samples in the window, or ``0.0`` if it is empty."""
        if not self._values:
            return 0.0
        return self._sum / len(self._values)
