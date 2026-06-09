# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import pytest
from gms_kv_ring.engines.handle import GMSKvRing


class _Counters:
    def __init__(self) -> None:
        self.failed: list[tuple[int, int]] = []

    def try_reserve_completed_slot(self) -> tuple[int, int]:
        return (7, 11)

    def mark_slot_failed(self, slot: int, target: int) -> None:
        self.failed.append((slot, target))


class _FailingWriter:
    def push(self, **kwargs):
        raise RuntimeError("writer failed")


def _handle_with_failing_writers() -> tuple[GMSKvRing, _Counters]:
    handle = GMSKvRing.__new__(GMSKvRing)
    handle._evict_lock = threading.Lock()
    handle._restore_lock = threading.Lock()
    handle.evict_writer = _FailingWriter()
    handle.restore_writer = _FailingWriter()
    handle.counters = _Counters()
    return handle, handle.counters


def test_record_evict_marks_reserved_counter_slot_failed_on_push_exception():
    handle, counters = _handle_with_failing_writers()

    with pytest.raises(RuntimeError, match="writer failed"):
        handle.record_evict(block_id=1, ranges=[(0, 128, 0)])

    assert counters.failed == [(7, 11)]


def test_record_restore_marks_reserved_counter_slot_failed_on_push_exception():
    handle, counters = _handle_with_failing_writers()

    with pytest.raises(RuntimeError, match="writer failed"):
        handle.record_restore(src_engine_id="engine-a", block_pairs=[(1, 2)])

    assert counters.failed == [(7, 11)]
