# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

import pytest
from gpu_memory_service.integrations.common.kv_lease_client import (
    SharedMemoryKVLeaseClient,
    set_kv_lease_reservation,
)
from gpu_memory_service.integrations.common.transition_reclaim import (
    start_kv_transition_reclaim_watcher,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


pytest.importorskip("gms_rust_ring")


def test_transition_reclaim_watcher_releases_until_shadow_reserve_has_headroom(
    tmp_path, monkeypatch
):
    shm_path = tmp_path / "lease.shm"
    reserve_path = tmp_path / "lease.reserve"
    monkeypatch.setenv("GMS_KV_LEASES", "1")
    monkeypatch.setenv("GMS_VLLM_KV_LEASE_NAMESPACE", "transition-test")
    monkeypatch.setenv("GMS_VLLM_KV_LEASE_SHM_PATH", str(shm_path))
    monkeypatch.setenv("GMS_VLLM_KV_LEASE_RESERVATION_PATH", str(reserve_path))
    monkeypatch.setenv("GMS_VLLM_KV_LEASE_OWNER_ID", "primary")
    monkeypatch.setenv("GMS_VLLM_TRANSITION_RECLAIM_POLL_S", "0.05")

    primary = SharedMemoryKVLeaseClient(
        str(shm_path),
        namespace="transition-test",
        owner_id="primary",
        total_blocks=8,
        reservation_path=str(reserve_path),
    )
    leases = primary.acquire(7)
    calls: list[int] = []

    def reclaim(target_blocks: int) -> int:
        calls.append(int(target_blocks))
        to_release = leases[: int(target_blocks)]
        del leases[: int(target_blocks)]
        primary.release(to_release)
        return len(to_release)

    watcher = start_kv_transition_reclaim_watcher(
        "vllm", reclaim, device=0, poll_s=0.05
    )
    assert watcher is not None
    try:
        set_kv_lease_reservation(
            "vllm", 0, reserved_blocks=3, reserved_for_owner="shadow"
        )
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and primary.raw_free_count() < 3:
            time.sleep(0.02)
        assert primary.raw_free_count() >= 3
        assert calls
        assert calls[0] == 2
    finally:
        watcher.stop()
        primary.release(leases)
        primary.close()
