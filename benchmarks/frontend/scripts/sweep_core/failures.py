# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Consecutive-failure skip policy for sweep runs."""

from __future__ import annotations

from typing import Dict, Tuple


class FailureTracker:
    """Track consecutive failures per (backend, concurrency, workers) tuple.

    After max_consecutive_fails consecutive failures at a given key,
    subsequent runs with the same key are skipped.
    """

    def __init__(self, max_consecutive_fails: int = 2):
        self.max_consecutive_fails = max_consecutive_fails
        self._counts: Dict[Tuple[str, int, int], int] = {}

    def should_skip(self, backend: str, concurrency: int, workers: int) -> bool:
        """Check if a run should be skipped due to prior consecutive failures."""
        key = (backend, concurrency, workers)
        return self._counts.get(key, 0) >= self.max_consecutive_fails

    def record_success(self, backend: str, concurrency: int, workers: int) -> None:
        """Record a successful run, resetting the failure count."""
        key = (backend, concurrency, workers)
        self._counts[key] = 0

    def record_failure(self, backend: str, concurrency: int, workers: int) -> int:
        """Record a failed run. Returns the new consecutive failure count."""
        key = (backend, concurrency, workers)
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def get_count(self, backend: str, concurrency: int, workers: int) -> int:
        """Get the current consecutive failure count for a key."""
        key = (backend, concurrency, workers)
        return self._counts.get(key, 0)
