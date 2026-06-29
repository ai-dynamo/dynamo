# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import mmap
import multiprocessing as mp
import os
import struct
import time

import pytest
from gpu_memory_service.integrations.common.kv_lease_client import (
    SharedMemoryKVLeaseClient,
    read_any_kv_lease_namespace_total_blocks,
    read_kv_lease_namespace_total_blocks,
    read_kv_lease_reservation,
    resolve_kv_lease_namespace_total_blocks,
    set_kv_lease_reservation,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


pytest.importorskip("gms_rust_ring")


_LEASE_FREE_COUNT_OFFSET = 16
_LEASE_ACTIVE_MUTATIONS_OFFSET = 24
_LEASE_RECORD_OFFSET = 64
_LEASE_STATE_TRANSITION = 4


def _crash_in_lease_transition(path: str, operation: str, phase: int) -> None:
    """Model process death after one atomic step of acquire or release."""
    fd = os.open(path, os.O_RDWR)
    buf = mmap.mmap(fd, 0)
    try:
        # A real native mutation registers itself before changing its record.
        struct.pack_into("<Q", buf, _LEASE_ACTIVE_MUTATIONS_OFFSET, 1)
        struct.pack_into("<I", buf, _LEASE_RECORD_OFFSET, _LEASE_STATE_TRANSITION)
        if phase >= 1:
            if operation == "acquire":
                generation = struct.unpack_from("<I", buf, _LEASE_RECORD_OFFSET + 4)[0]
                struct.pack_into("<I", buf, _LEASE_RECORD_OFFSET + 4, generation + 1)
            else:
                struct.pack_into("<Q", buf, _LEASE_RECORD_OFFSET + 8, 0)
        if phase >= 2:
            if operation == "acquire":
                struct.pack_into("<Q", buf, _LEASE_RECORD_OFFSET + 8, 0xBAD)
            else:
                free = struct.unpack_from("<Q", buf, _LEASE_FREE_COUNT_OFFSET)[0]
                struct.pack_into("<Q", buf, _LEASE_FREE_COUNT_OFFSET, free + 1)
        if phase >= 3:
            assert operation == "acquire"
            free = struct.unpack_from("<Q", buf, _LEASE_FREE_COUNT_OFFSET)[0]
            struct.pack_into("<Q", buf, _LEASE_FREE_COUNT_OFFSET, free - 1)
    finally:
        # Deliberately bypass close/finally behavior, as SIGKILL would.
        os._exit(99)


def test_shared_memory_lease_client_coordinates_two_clients(tmp_path):
    path = tmp_path / "leases.shm"
    first = SharedMemoryKVLeaseClient(
        str(path),
        namespace="test",
        owner_id="first",
        total_blocks=8,
        reserved_blocks=[0],
    )
    second = SharedMemoryKVLeaseClient(
        str(path),
        namespace="test",
        owner_id="second",
        total_blocks=8,
        reserved_blocks=[0],
    )
    try:
        assert first.free_count() == 7
        assert second.free_count() == 7

        first_leases = first.acquire(
            3, preferred_blocks=[1, 2, 3], strict_preferred=True
        )
        second_leases = second.acquire(
            4, preferred_blocks=[1, 2, 3, 4, 5, 6, 7], strict_preferred=True
        )

        assert [lease.block_id for lease in first_leases] == [1, 2, 3]
        assert [lease.block_id for lease in second_leases] == [4, 5, 6, 7]
        assert first.free_count() == 0
        assert second.free_count() == 0

        first.seal([first_leases[0]])
        first.release([first_leases[0]])
        reacquired = second.acquire(
            1, preferred_blocks=[first_leases[0].block_id], strict_preferred=True
        )
        assert reacquired[0].block_id == first_leases[0].block_id
        assert reacquired[0].generation == first_leases[0].generation + 1
    finally:
        first.close()
        second.close()


def test_shared_memory_lease_reclaim_foreign_preserves_current_owner(tmp_path):
    path = tmp_path / "leases-reclaim.shm"
    first = SharedMemoryKVLeaseClient(
        str(path), namespace="reclaim", owner_id="old-primary", total_blocks=8
    )
    second = SharedMemoryKVLeaseClient(
        str(path), namespace="reclaim", owner_id="shadow", total_blocks=8
    )
    try:
        old_leases = first.acquire(4)
        first.seal(old_leases[:2])
        shadow_leases = second.acquire(2)

        reclaimed = second.reclaim_foreign()

        assert reclaimed == 4
        assert second.raw_free_count() == 6
        reacquired = second.acquire(4)
        assert len(reacquired) == 4
        second.release(reacquired)
        second.release(shadow_leases)
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize(
    ("operation", "phase"),
    [
        ("acquire", 0),  # after FREE -> TRANSITION
        ("acquire", 1),  # after generation bump
        ("acquire", 2),  # after owner publication
        ("acquire", 3),  # after free-count decrement
        ("release", 0),  # after LEASED -> TRANSITION
        ("release", 1),  # after owner clear
        ("release", 2),  # after free-count increment
    ],
)
def test_post_fence_reclaim_recovers_process_death_at_every_transition_step(
    tmp_path, operation, phase
):
    path = str(tmp_path / f"crash-{operation}-{phase}.shm")
    old = SharedMemoryKVLeaseClient(
        path, namespace="crash-recovery", owner_id="old", total_blocks=4
    )
    stale = []
    if operation == "release":
        stale = old.acquire(1, preferred_blocks=[0], strict_preferred=True)
    old.close()

    proc = mp.get_context("spawn").Process(
        target=_crash_in_lease_transition, args=(path, operation, phase)
    )
    proc.start()
    proc.join(timeout=10)
    assert not proc.is_alive()
    assert proc.exitcode == 99

    shadow = SharedMemoryKVLeaseClient(
        path, namespace="crash-recovery", owner_id="shadow", total_blocks=4
    )
    try:
        assert shadow.reclaim_foreign() == 1
        assert shadow.raw_free_count() == 4
        with open(path, "rb") as lease_file:
            lease_file.seek(_LEASE_ACTIVE_MUTATIONS_OFFSET)
            assert struct.unpack("<Q", lease_file.read(8))[0] == 0

        current = shadow.acquire(4)
        assert len(current) == 4
        assert shadow.raw_free_count() == 0
        if stale:
            shadow.release(stale)
            assert shadow.raw_free_count() == 0
        shadow.release(current)
        assert shadow.raw_free_count() == 4
    finally:
        shadow.close()


def test_reclaim_foreign_kv_leases_in_shm_dir_scans_rank_local_files(tmp_path):
    from gpu_memory_service.integrations.common.kv_lease_client import (
        reclaim_foreign_kv_leases_in_shm_dir,
    )

    shm_dir = tmp_path / "rank"
    shm_dir.mkdir()
    client = SharedMemoryKVLeaseClient(
        str(shm_dir / "gms-kv-lease-test.shm"),
        namespace="scan-reclaim",
        owner_id="old-primary",
        total_blocks=5,
    )
    shadow = SharedMemoryKVLeaseClient(
        str(shm_dir / "gms-kv-lease-test.shm"),
        namespace="scan-reclaim",
        owner_id="shadow",
        total_blocks=5,
    )
    try:
        client.acquire(3)
        result = reclaim_foreign_kv_leases_in_shm_dir(
            "vllm", 0, owner_id="shadow", shm_dir=str(shm_dir)
        )

        assert result.files == 1
        assert result.reclaimed_blocks == 3
        assert result.errors == 0
        assert shadow.raw_free_count() == 5
    finally:
        client.close()
        shadow.close()


def test_read_kv_lease_namespace_total_blocks_is_read_only(tmp_path, monkeypatch):
    path = tmp_path / "geometry.shm"
    monkeypatch.setenv("GMS_VLLM_KV_LEASE_NAMESPACE", "geometry")
    monkeypatch.setenv("GMS_VLLM_KV_LEASE_SHM_PATH", str(path))

    namespace, total = read_kv_lease_namespace_total_blocks("vllm", 0)
    assert namespace == "geometry"
    assert total is None
    assert not path.exists()

    client = SharedMemoryKVLeaseClient(
        str(path), namespace="geometry", owner_id="writer", total_blocks=37
    )
    client.close()

    namespace, total = read_kv_lease_namespace_total_blocks("vllm", 0)
    assert namespace == "geometry"
    assert total == 37


def test_read_any_kv_lease_namespace_total_blocks_finds_rank0_geometry(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GMS_VLLM_KV_LEASE_SHM_DIR", str(tmp_path))
    client = SharedMemoryKVLeaseClient.from_env(
        "vllm", 0, total_blocks=41, namespace_suffix="block-pool"
    )
    client.close()

    path, total = read_any_kv_lease_namespace_total_blocks("vllm")

    assert str(tmp_path) in path
    assert total == 41


def test_resolve_kv_lease_namespace_adopts_existing_geometry(tmp_path, monkeypatch):
    path = tmp_path / "geometry-race.shm"
    monkeypatch.setenv("GMS_SGLANG_KV_LEASE_NAMESPACE", "geometry-race")
    monkeypatch.setenv("GMS_SGLANG_KV_LEASE_SHM_PATH", str(path))

    namespace, total = resolve_kv_lease_namespace_total_blocks(
        "sglang", 0, total_blocks=107103
    )
    assert namespace == "geometry-race"
    assert total == 107103

    namespace, total = resolve_kv_lease_namespace_total_blocks(
        "sglang", 0, total_blocks=106576
    )
    assert namespace == "geometry-race"
    assert total == 107103

    namespace, read_total = read_kv_lease_namespace_total_blocks("sglang", 0)
    assert namespace == "geometry-race"
    assert read_total == 107103


def test_shared_memory_lease_client_keeps_strict_geometry_check(tmp_path):
    path = tmp_path / "strict-geometry.shm"
    writer = SharedMemoryKVLeaseClient(
        str(path), namespace="strict-geometry", owner_id="writer", total_blocks=8
    )
    writer.close()

    with pytest.raises(RuntimeError, match="size mismatch"):
        SharedMemoryKVLeaseClient(
            str(path), namespace="strict-geometry", owner_id="reader", total_blocks=7
        )


def test_resolve_lease_device_uses_engine_env_before_rank(monkeypatch):
    from gpu_memory_service.integrations.common.kv_lease_client import (
        resolve_lease_device,
    )

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("GMS_VLLM_KV_LEASE_DEVICE", "1")

    assert resolve_lease_device("GMS_VLLM_KV_LEASE_DEVICE") == 1


def test_shared_memory_lease_acquire_uses_lockless_fast_path_without_reservation(
    tmp_path, monkeypatch
):
    path = tmp_path / "fast-path.shm"
    client = SharedMemoryKVLeaseClient(
        str(path),
        namespace="fast-path",
        owner_id="primary",
        total_blocks=4,
        reserved_blocks=[0],
    )
    try:
        from gpu_memory_service.integrations.common import kv_lease_client

        def fail_open_reservation_file(_path: str) -> int:
            raise AssertionError("reservation lock opened on no-reservation fast path")

        monkeypatch.setattr(
            kv_lease_client,
            "_open_reservation_lock_file",
            fail_open_reservation_file,
        )

        leases = client.acquire(2, preferred_blocks=[1, 2], strict_preferred=True)
        assert [lease.block_id for lease in leases] == [1, 2]
        client.release(leases)
        assert client.free_count() == 3
    finally:
        client.close()


def test_shared_memory_lease_reservation_is_owner_aware(tmp_path, monkeypatch):
    path = tmp_path / "reserved.shm"
    reserve_path = tmp_path / "reserved.json"
    monkeypatch.setenv("GMS_VLLM_KV_LEASE_NAMESPACE", "reserved")
    monkeypatch.setenv("GMS_VLLM_KV_LEASE_SHM_PATH", str(path))
    monkeypatch.setenv("GMS_VLLM_KV_LEASE_RESERVATION_PATH", str(reserve_path))

    primary = SharedMemoryKVLeaseClient(
        str(path),
        namespace="reserved",
        owner_id="primary",
        total_blocks=8,
        reservation_path=str(reserve_path),
    )
    shadow = SharedMemoryKVLeaseClient(
        str(path),
        namespace="reserved",
        owner_id="shadow",
        total_blocks=8,
        reservation_path=str(reserve_path),
    )
    try:
        namespace, reservation = set_kv_lease_reservation(
            "vllm", 0, reserved_blocks=3, reserved_for_owner="shadow"
        )
        assert namespace == "reserved"
        assert reservation.reserved_blocks == 3
        assert read_kv_lease_reservation("vllm", 0)[1] == reservation
        assert primary.raw_free_count() == 8
        assert primary.free_count() == 5
        assert shadow.free_count() == 8

        leases = primary.acquire(5)
        assert len(leases) == 5
        assert primary.raw_free_count() == 3
        assert primary.free_count() == 0
        with pytest.raises(RuntimeError, match="reserved"):
            primary.acquire(1)
        assert len(shadow.acquire(3)) == 3
    finally:
        primary.close()
        shadow.close()


def _lease_worker(path: str, active, lock, errors, loops: int) -> None:
    client = SharedMemoryKVLeaseClient(
        path,
        namespace="stress",
        owner_id=f"worker-{os.getpid()}",
        total_blocks=65,
        reserved_blocks=[0],
    )
    try:
        for _ in range(loops):
            leases = client.acquire(3, strict_preferred=False)
            block_ids = [lease.block_id for lease in leases]
            if len(block_ids) != len(set(block_ids)):
                errors.append(f"duplicate lease in one acquire: {block_ids}")
            with lock:
                for block_id in block_ids:
                    if block_id in active:
                        errors.append(
                            f"duplicate active block {block_id}: {active[block_id]} and {os.getpid()}"
                        )
                    active[block_id] = os.getpid()
            time.sleep(0.0005)
            with lock:
                for block_id in block_ids:
                    active.pop(block_id, None)
            client.release(leases)
    except Exception as exc:  # noqa: BLE001
        errors.append(repr(exc))
    finally:
        client.close()


def _reservation_primary_worker(
    path: str,
    reserve_path: str,
    active,
    lock,
    errors,
    start,
    stop,
    loops: int,
) -> None:
    owner = f"primary-{os.getpid()}"
    client = SharedMemoryKVLeaseClient(
        path,
        namespace="reservation-stress",
        owner_id=owner,
        total_blocks=9,
        reserved_blocks=[0],
        reservation_path=reserve_path,
    )
    try:
        if not start.wait(10):
            errors.append(f"{owner} timed out waiting for start")
            return
        completed = 0
        while completed < loops and not stop.is_set():
            try:
                leases = client.acquire(1)
            except RuntimeError:
                time.sleep(0.0005)
                continue
            block_id = leases[0].block_id
            try:
                with lock:
                    previous = active.get(block_id)
                    if previous is not None:
                        errors.append(
                            f"duplicate active block {block_id}: {previous} and {owner}"
                        )
                    active[block_id] = owner
                time.sleep(0.001)
            finally:
                with lock:
                    if active.get(block_id) == owner:
                        active.pop(block_id, None)
                client.release(leases)
            completed += 1
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{owner}: {exc!r}")
    finally:
        client.close()


def _reservation_shadow_worker(
    path: str,
    reserve_path: str,
    active,
    lock,
    errors,
    start,
    reservation_set,
    shadow_done,
) -> None:
    owner = f"shadow-{os.getpid()}"
    client = SharedMemoryKVLeaseClient(
        path,
        namespace="reservation-stress",
        owner_id="shadow",
        total_blocks=9,
        reserved_blocks=[0],
        reservation_path=reserve_path,
    )
    leases = []
    try:
        if not start.wait(10):
            errors.append(f"{owner} timed out waiting for start")
            return
        if not reservation_set.wait(10):
            errors.append(f"{owner} timed out waiting for reservation")
            return
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                leases = client.acquire(4)
                break
            except RuntimeError:
                time.sleep(0.001)
        if len(leases) != 4:
            errors.append(f"{owner} acquired {len(leases)} reserved blocks, want 4")
            return
        with lock:
            for lease in leases:
                previous = active.get(lease.block_id)
                if previous is not None:
                    errors.append(
                        f"shadow duplicate active block {lease.block_id}: {previous} and {owner}"
                    )
                active[lease.block_id] = owner
        time.sleep(0.05)
        with lock:
            for lease in leases:
                if active.get(lease.block_id) == owner:
                    active.pop(lease.block_id, None)
        shadow_done.set()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{owner}: {exc!r}")
    finally:
        if leases:
            client.release(leases)
        client.close()


def test_shared_memory_lease_client_cross_process_stress(tmp_path):
    path = str(tmp_path / "stress.shm")
    client = SharedMemoryKVLeaseClient(
        path,
        namespace="stress",
        owner_id="parent",
        total_blocks=65,
        reserved_blocks=[0],
    )
    client.close()

    ctx = mp.get_context("fork")
    with ctx.Manager() as manager:
        active = manager.dict()
        lock = manager.Lock()
        errors = manager.list()
        procs = [
            ctx.Process(target=_lease_worker, args=(path, active, lock, errors, 80))
            for _ in range(4)
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(timeout=20)
        for proc in procs:
            assert not proc.is_alive()
            assert proc.exitcode == 0
        assert list(errors) == []

    final = SharedMemoryKVLeaseClient(
        path, namespace="stress", owner_id="final", total_blocks=65, reserved_blocks=[0]
    )
    try:
        assert final.free_count() == 64
    finally:
        final.close()


def test_shared_memory_lease_transition_reservation_cross_process_stress(
    tmp_path, monkeypatch
):
    shm_path = str(tmp_path / "reservation-stress.shm")
    reserve_path = str(tmp_path / "reservation-stress.reserve")
    monkeypatch.setenv("GMS_VLLM_KV_LEASE_NAMESPACE", "reservation-stress")
    monkeypatch.setenv("GMS_VLLM_KV_LEASE_SHM_PATH", shm_path)
    monkeypatch.setenv("GMS_VLLM_KV_LEASE_RESERVATION_PATH", reserve_path)

    client = SharedMemoryKVLeaseClient(
        shm_path,
        namespace="reservation-stress",
        owner_id="parent",
        total_blocks=9,
        reserved_blocks=[0],
        reservation_path=reserve_path,
    )
    client.close()

    ctx = mp.get_context("fork")
    start = ctx.Event()
    stop = ctx.Event()
    reservation_set = ctx.Event()
    shadow_done = ctx.Event()
    with ctx.Manager() as manager:
        active = manager.dict()
        lock = manager.Lock()
        errors = manager.list()
        procs = [
            ctx.Process(
                target=_reservation_primary_worker,
                args=(shm_path, reserve_path, active, lock, errors, start, stop, 200),
            )
            for _ in range(8)
        ]
        shadow = ctx.Process(
            target=_reservation_shadow_worker,
            args=(
                shm_path,
                reserve_path,
                active,
                lock,
                errors,
                start,
                reservation_set,
                shadow_done,
            ),
        )
        procs.append(shadow)
        for proc in procs:
            proc.start()
        start.set()
        time.sleep(0.05)

        namespace, reservation = set_kv_lease_reservation(
            "vllm", 0, reserved_blocks=4, reserved_for_owner="shadow"
        )
        assert namespace == "reservation-stress"
        assert reservation.reserved_blocks == 4
        reservation_set.set()

        deadline = time.monotonic() + 12
        while time.monotonic() < deadline and not shadow_done.is_set():
            time.sleep(0.02)
        stop.set()
        for proc in procs:
            proc.join(timeout=20)
        for proc in procs:
            assert not proc.is_alive()
            assert proc.exitcode == 0
        assert shadow_done.is_set(), "shadow never acquired reserved KV headroom"
        assert list(errors) == []

    set_kv_lease_reservation("vllm", 0, reserved_blocks=0)
    final = SharedMemoryKVLeaseClient(
        shm_path,
        namespace="reservation-stress",
        owner_id="final",
        total_blocks=9,
        reserved_blocks=[0],
        reservation_path=reserve_path,
    )
    try:
        assert final.raw_free_count() == 8
        assert final.free_count() == 8
    finally:
        final.close()


def test_shared_memory_lease_release_ignores_stale_generation(tmp_path):
    path = str(tmp_path / "stale-generation.shm")
    client = SharedMemoryKVLeaseClient(
        path,
        namespace="stale-generation",
        owner_id="primary",
        total_blocks=2,
        reserved_blocks=[0],
    )
    shadow = SharedMemoryKVLeaseClient(
        path,
        namespace="stale-generation",
        owner_id="shadow",
        total_blocks=2,
        reserved_blocks=[0],
    )
    try:
        old = client.acquire(1, preferred_blocks=[1], strict_preferred=True)
        assert client.raw_free_count() == 0
        client.release(old)

        new = shadow.acquire(1, preferred_blocks=[1], strict_preferred=True)
        assert new[0].block_id == old[0].block_id
        assert new[0].generation == old[0].generation + 1
        assert shadow.raw_free_count() == 0

        client.release(old)
        assert shadow.raw_free_count() == 0

        with pytest.raises(RuntimeError):
            client.acquire(1, allow_partial=False)
        shadow.release(new)
        assert client.raw_free_count() == 1
    finally:
        client.close()
        shadow.close()
