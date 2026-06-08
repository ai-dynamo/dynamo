# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for common/metrics.py."""

from __future__ import annotations

import os
import uuid


def test_metrics_create_increment_read(tmp_path):
    from gpu_memory_service.common import metrics as m

    path = f"/dev/shm/gms-test-metrics-{uuid.uuid4().hex}.bin"
    try:
        arr = m.MetricsArray.open(path)
        # Initial state — all zero.
        assert arr.get(m.RING_SPILL_PUSHED) == 0
        # Increment several slots.
        arr.inc(m.RING_SPILL_PUSHED, 5)
        arr.inc(m.RING_RESTORE_PUSHED, 3)
        arr.inc(m.SPILL_RPC_CALLS, 1)
        arr.inc(m.SPILL_RPC_CALLS, 1)
        # Re-open from a different handle (sim cross-process attach).
        arr2 = m.MetricsArray.open(path)
        assert arr2.get(m.RING_SPILL_PUSHED) == 5
        assert arr2.get(m.RING_RESTORE_PUSHED) == 3
        assert arr2.get(m.SPILL_RPC_CALLS) == 2
        # snapshot dict
        snap = arr2.snapshot()
        assert snap["ring_spill_pushed"] == 5
        assert snap["spill_rpc_calls"] == 2
        # Prometheus format
        prom = arr2.prometheus_text()
        assert "# TYPE gms_ring_spill_pushed counter" in prom
        assert "gms_ring_spill_pushed 5" in prom
        arr.close()
        arr2.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_metrics_global_handle_lazy(monkeypatch):
    from gpu_memory_service.common import metrics as m

    path = f"/dev/shm/gms-test-metrics-global-{uuid.uuid4().hex}.bin"
    monkeypatch.setenv("GMS_METRICS_FILE", path)

    # Reset the module-level lazy cache.
    m._LAZY = None

    try:
        m.inc(m.CHUNK_TABLE_HITS, 7)
        m.inc(m.CHUNK_TABLE_HITS, 3)
        arr = m.get_global()
        assert arr.get(m.CHUNK_TABLE_HITS) == 10
    finally:
        if m._LAZY is not None:
            m._LAZY.close()
            m._LAZY = None
        if os.path.exists(path):
            os.unlink(path)


def test_metrics_never_break_hot_path(tmp_path, monkeypatch):
    """inc() must be a no-op on failure — never raises."""
    from gpu_memory_service.common import metrics as m

    # Force an invalid path (a directory).
    monkeypatch.setenv("GMS_METRICS_FILE", str(tmp_path))
    m._LAZY = None

    # Should not raise even though the file create would fail.
    m.inc(m.RING_SPILL_PUSHED, 1)
    # Reset.
    m._LAZY = None
