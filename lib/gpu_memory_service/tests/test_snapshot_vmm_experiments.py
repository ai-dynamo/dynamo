# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only coverage for optional snapshot VMM/host-registration experiments."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from _fake_vmm import FakeVMM
from gpu_memory_service.cli.snapshot import loader
from gpu_memory_service.common.snapshot_profile import SnapshotProfile
from gpu_memory_service.common.vmm import cuda_utils
from gpu_memory_service.snapshot.backends import pinned_host

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_mapping_coordinator_releases_waiters_on_failure():
    coordinator = loader._MappingCoordinator(2)
    participants = [loader._MappingParticipant(coordinator) for _ in range(2)]
    participants[0].start()
    participants[0].complete()
    participants[1].fail(ValueError("device failed"))

    with pytest.raises(RuntimeError, match="another device failed"):
        coordinator.wait()


def test_loader_experiment_defaults_and_environment(monkeypatch):
    monkeypatch.delenv(loader.SHARDED_SSD_CUDA_MODE_ENV, raising=False)
    monkeypatch.delenv(loader.MAPPING_FIRST_ENV, raising=False)
    monkeypatch.delenv(loader.PINNED_REGISTRATION_GROUPS_ENV, raising=False)
    args = loader._build_parser().parse_args([])
    assert args.sharded_ssd_cuda_mode == loader.CUDA_MODE_RUNTIME
    assert args.mapping_first is False
    assert args.pinned_registration_groups == 0

    monkeypatch.setenv(loader.SHARDED_SSD_CUDA_MODE_ENV, "driver")
    monkeypatch.setenv(loader.MAPPING_FIRST_ENV, "1")
    monkeypatch.setenv(loader.PINNED_REGISTRATION_GROUPS_ENV, "2")
    args = loader._build_parser().parse_args([])
    assert args.sharded_ssd_cuda_mode == loader.CUDA_MODE_DRIVER
    assert args.mapping_first is True
    assert args.pinned_registration_groups == 2


def test_invalid_mapping_environment_fails_closed(monkeypatch):
    monkeypatch.setenv(loader.MAPPING_FIRST_ENV, "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        loader._build_parser()


def test_pinned_arena_groups_registration_and_requires_slot_release(monkeypatch):
    operations = FakeVMM()
    registrations = []
    unregistrations = []
    monkeypatch.setattr(
        operations,
        "host_register",
        lambda ptr, size: registrations.append((ptr, size)),
    )
    monkeypatch.setattr(
        operations,
        "host_unregister",
        lambda ptr: unregistrations.append(ptr),
    )
    monkeypatch.setattr(
        pinned_host,
        "_allocate_aligned_buffer",
        lambda size: (memoryview(bytearray(size)), bytearray(size), 0x1000),
    )
    monkeypatch.setattr(
        pinned_host,
        "_free_aligned_buffer",
        lambda _view, _ptr: None,
    )

    arena = pinned_host.PinnedCopyArena(
        operations,
        slot_count=4,
        registration_groups=2,
        slot_size=4096,
        profile=SnapshotProfile("loader", enabled=False),
    )
    assert registrations == [(0x1000, 8192), (0x3000, 8192)]
    view, _, ptr = arena.claim_slot(1, 4096)
    with pytest.raises(RuntimeError, match="logical slots remain active"):
        arena.close()
    arena.release_slot(ptr, view)
    arena.close()
    assert unregistrations == [0x3000, 0x1000]


def test_mapping_participant_completion_is_idempotent():
    coordinator = loader._MappingCoordinator(1)
    participant = loader._MappingParticipant(coordinator)
    participant.start()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: participant.complete(), range(2)))
    coordinator.wait()
    wall_start, wall_end, duration = coordinator.envelope()
    assert wall_end >= wall_start
    assert duration >= 0


def test_driver_transfer_operations_use_driver_api(monkeypatch):
    calls = []

    class FakeResult:
        CUDA_SUCCESS = 0

    class FakeStreamFlags:
        CU_STREAM_NON_BLOCKING = 1

    class FakeEventFlags:
        CU_EVENT_DEFAULT = 0

    class FakeCuda:
        CUresult = FakeResult
        CUstream_flags = FakeStreamFlags
        CUevent_flags = FakeEventFlags

        @staticmethod
        def cuInit(flags):
            calls.append(("init", flags))
            return (0,)

        @staticmethod
        def cuDeviceGet(ordinal):
            calls.append(("device_get", ordinal))
            return 0, f"device-{ordinal}"

        @staticmethod
        def cuDevicePrimaryCtxRetain(device):
            calls.append(("retain", device))
            return 0, 0xCAFE

        @staticmethod
        def cuCtxSetCurrent(context):
            calls.append(("set_current", context))
            return (0,)

        @staticmethod
        def cuStreamCreate(flags):
            calls.append(("stream_create", flags))
            return 0, "stream"

        @staticmethod
        def cuMemcpyHtoDAsync(dst, src, size, stream):
            calls.append(("h2d", dst, src, size, stream))
            return (0,)

        @staticmethod
        def cuStreamSynchronize(stream):
            calls.append(("stream_sync", stream))
            return (0,)

        @staticmethod
        def cuStreamDestroy(stream):
            calls.append(("stream_destroy", stream))
            return (0,)

        @staticmethod
        def cuDevicePrimaryCtxRelease(device):
            calls.append(("release", device))
            return (0,)

    monkeypatch.setattr(cuda_utils, "cuda", FakeCuda)
    process = cuda_utils.DriverCudaProcess()
    process.initialize()
    device = process.device_get(0)
    process.primary_context_retain(0, device)
    operations = process.operations(
        0,
        SnapshotProfile("loader", enabled=False),
    )
    operations.set_current_device(0)
    stream = operations.stream_create_nonblocking()
    operations.memcpy_h2d_async(0x1000, 0x2000, 4096, stream)
    operations.stream_synchronize(stream)
    operations.stream_destroy(stream)
    process.close()

    assert calls == [
        ("init", 0),
        ("device_get", 0),
        ("retain", "device-0"),
        ("set_current", 0xCAFE),
        ("stream_create", 1),
        ("h2d", 0x1000, 0x2000, 4096, "stream"),
        ("stream_sync", "stream"),
        ("stream_destroy", "stream"),
        ("release", "device-0"),
    ]
