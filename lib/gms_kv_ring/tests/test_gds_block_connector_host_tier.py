# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from gms_kv_ring.engines.gds_block_connector import BlockGdsConnector


class _Counters:
    def __init__(self):
        self.values = {}

    def read_slot(self, slot):
        return self.values.get(int(slot), 0)


class _Handle:
    def __init__(self):
        self.counters = _Counters()
        self.evict_calls = []
        self.restore_calls = []
        self.next_slot = 1
        self.evict_ok = True
        self.restore_ok = True

    def gpu_direct_storage_available(self):
        return False

    def _target(self):
        slot = self.next_slot
        self.next_slot += 1
        target = slot + 100
        self.counters.values[slot] = target
        return slot, target

    def record_evict(self, block_id, ranges, generation=0):
        self.evict_calls.append((int(block_id), list(ranges), int(generation)))
        return self._target()

    def evict_succeeded(self, slot, target):
        return self.evict_ok and self.counters.read_slot(slot) == int(target)

    def record_restore(self, src_engine_id, block_pairs, *, flags=0):
        self.restore_calls.append((str(src_engine_id), list(block_pairs), int(flags)))
        return self._target()

    def restore_succeeded(self, slot, target):
        return self.restore_ok and self.counters.read_slot(slot) == int(target)


def _layout(block_id):
    return [(0, int(block_id) * 64, 64)]


def test_evict_blocks_to_host_waits_for_ack_and_reports_success():
    handle = _Handle()
    conn = BlockGdsConnector(handle, _layout)

    result = conn.evict_blocks_to_host([3, 4])

    assert result.succeeded == 2
    assert result.failed == 0
    assert handle.evict_calls == [
        (3, [(0, 64, 3 * 64)], 0),
        (4, [(0, 64, 4 * 64)], 0),
    ]


def test_restore_blocks_remap_from_host_uses_source_engine_id():
    handle = _Handle()
    conn = BlockGdsConnector(handle, _layout)

    out = conn.restore_blocks_remap_from_host("primary", [(3, 30, 7), (4, 40, 8)])

    assert out == {30: True, 40: True}
    assert handle.restore_calls == [
        ("primary", [(3, 30), (4, 40)], 0),
    ]


class _SyncRestoreHandle(_Handle):
    def __init__(self):
        super().__init__()
        self.sync_restore_calls = []

    def restore_host_blocks_sync(self, src_engine_id, triples):
        self.sync_restore_calls.append((str(src_engine_id), list(triples)))
        return True


def test_restore_blocks_remap_from_host_preserves_expected_generations():
    handle = _SyncRestoreHandle()
    conn = BlockGdsConnector(handle, _layout)

    out = conn.restore_blocks_remap_from_host("primary", [(3, 30, 7), (4, 40, 8)])

    assert out == {30: True, 40: True}
    assert handle.sync_restore_calls == [
        ("primary", [(3, 30, 7), (4, 40, 8)]),
    ]
