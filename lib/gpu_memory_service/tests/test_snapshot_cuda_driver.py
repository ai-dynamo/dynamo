# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Driver-only sharded-SSD snapshot loading."""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

try:
    from gpu_memory_service.cli.snapshot import loader
    from gpu_memory_service.common import cuda_utils
    from gpu_memory_service.common.snapshot_profile import (
        SNAPSHOT_PROFILE_ENV,
        SnapshotProfile,
    )
    from gpu_memory_service.snapshot import storage_client
    from gpu_memory_service.snapshot.backends import nixl_staging, pinned_host
except ModuleNotFoundError:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


class _FakeDriver:
    CUresult = SimpleNamespace(CUDA_SUCCESS=0)
    CUstream_flags = SimpleNamespace(CU_STREAM_NON_BLOCKING=1)
    CUevent_flags = SimpleNamespace(CU_EVENT_DEFAULT=0)

    def __init__(self):
        self.calls = []
        self._next_event = 0

    def cuInit(self, flags):
        self.calls.append(("cuInit", flags))
        return (0,)

    def cuDeviceGet(self, ordinal):
        self.calls.append(("cuDeviceGet", ordinal))
        return 0, ordinal + 10

    def cuDevicePrimaryCtxRetain(self, device):
        self.calls.append(("cuDevicePrimaryCtxRetain", device))
        return 0, device + 100

    def cuCtxSetCurrent(self, context):
        self.calls.append(("cuCtxSetCurrent", threading.get_ident(), context))
        return (0,)

    def cuDevicePrimaryCtxRelease(self, device):
        self.calls.append(("cuDevicePrimaryCtxRelease", device))
        return (0,)

    def cuMemHostRegister(self, ptr, size, flags):
        self.calls.append(("cuMemHostRegister", ptr, size, flags))
        return (0,)

    def cuMemHostUnregister(self, ptr):
        self.calls.append(("cuMemHostUnregister", ptr))
        return (0,)

    def cuStreamCreate(self, flags):
        self.calls.append(("cuStreamCreate", flags))
        return 0, 77

    def cuStreamDestroy(self, stream):
        self.calls.append(("cuStreamDestroy", stream))
        return (0,)

    def cuStreamSynchronize(self, stream):
        self.calls.append(("cuStreamSynchronize", stream))
        return (0,)

    def cuEventCreate(self, flags):
        self._next_event += 1
        event = f"event-{self._next_event}"
        self.calls.append(("cuEventCreate", flags, event))
        return 0, event

    def cuEventRecord(self, event, stream):
        self.calls.append(("cuEventRecord", event, stream))
        return (0,)

    def cuEventElapsedTime(self, start_event, end_event):
        self.calls.append(("cuEventElapsedTime", start_event, end_event))
        return 0, 1.25

    def cuEventDestroy(self, event):
        self.calls.append(("cuEventDestroy", event))
        return (0,)

    def cuMemcpyHtoDAsync(self, dst_ptr, src_ptr, size, stream):
        self.calls.append(("cuMemcpyHtoDAsync", dst_ptr, src_ptr, size, stream))
        return (0,)

    def cuMemcpyDtoHAsync(self, dst_ptr, src_ptr, size, stream):
        self.calls.append(("cuMemcpyDtoHAsync", dst_ptr, src_ptr, size, stream))
        return (0,)

    def cuGetErrorString(self, result):
        return 0, f"driver failure {result}".encode()


def _profile_payloads(caplog):
    return [
        json.loads(record.getMessage().removeprefix("GMS_SNAPSHOT_PROFILE "))
        for record in caplog.records
        if record.getMessage().startswith("GMS_SNAPSHOT_PROFILE ")
    ]


def test_driver_result_failure_is_recoverable_and_names_api(monkeypatch):
    fake = _FakeDriver()
    monkeypatch.setattr(cuda_utils, "cuda", fake)

    with pytest.raises(
        RuntimeError,
        match="CUDA Driver error in cuDeviceGet\\(7\\): driver failure 3",
    ):
        cuda_utils.cuda_driver_check_result(3, "cuDeviceGet(7)")


def test_driver_process_retains_primary_contexts_across_worker_threads(
    monkeypatch,
):
    fake = _FakeDriver()
    monkeypatch.setattr(cuda_utils, "cuda", fake)
    process = cuda_utils.DriverCudaProcess()

    process.initialize()
    process.initialize()
    for ordinal in (0, 1):
        device = process.device_get(ordinal)
        process.primary_context_retain(ordinal, device)

    worker_done = []

    def use_context(ordinal):
        operations = process.operations(
            ordinal,
            SnapshotProfile("loader", enabled=False),
        )
        operations.set_current_device(ordinal)
        worker_done.append(ordinal)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(use_context, (0, 1)))

    assert sorted(worker_done) == [0, 1]
    assert [call for call in fake.calls if call[0] == "cuInit"] == [("cuInit", 0)]
    set_current_calls = [call for call in fake.calls if call[0] == "cuCtxSetCurrent"]
    assert sorted(call[2] for call in set_current_calls) == [110, 111]
    assert not any(call[0] == "cuDevicePrimaryCtxRelease" for call in fake.calls)

    process.close()

    assert [call for call in fake.calls if call[0] == "cuDevicePrimaryCtxRelease"] == [
        ("cuDevicePrimaryCtxRelease", 10),
        ("cuDevicePrimaryCtxRelease", 11),
    ]


def test_driver_pinned_slot_uses_driver_for_copy_timing_and_cleanup(
    monkeypatch,
    caplog,
):
    fake = _FakeDriver()
    monkeypatch.setattr(cuda_utils, "cuda", fake)
    raw = bytearray(16)
    monkeypatch.setattr(
        pinned_host,
        "_allocate_aligned_buffer",
        lambda _size: (memoryview(raw), raw, 123),
    )
    freed = []
    monkeypatch.setattr(
        pinned_host,
        "_free_aligned_buffer",
        lambda _view, ptr: freed.append(ptr),
    )
    process = cuda_utils.DriverCudaProcess()
    process.initialize()
    device = process.device_get(0)
    process.primary_context_retain(0, device)
    operations = process.operations(
        0,
        SnapshotProfile("loader", logger=pinned_host._LOGGER, enabled=True),
    )
    operations.set_current_device(0)
    profile = SnapshotProfile(
        "loader",
        logger=pinned_host._LOGGER,
        enabled=True,
    )

    slot = pinned_host.PinnedCopySlot(
        size=16,
        profile=profile,
        cuda_operations=operations,
    )
    slot.copy_to_device_async(456, 8)
    slot.wait()
    slot.close()

    assert freed == [123]
    assert ("cuMemHostRegister", 123, 16, 0) in fake.calls
    assert ("cuMemcpyHtoDAsync", 456, 123, 8, 77) in fake.calls
    assert ("cuStreamSynchronize", 77) in fake.calls
    assert ("cuMemHostUnregister", 123) in fake.calls
    assert ("cuStreamDestroy", 77) in fake.calls
    assert [call[1] for call in fake.calls if call[0] == "cuEventDestroy"] == [
        "event-1",
        "event-2",
    ]

    with caplog.at_level(logging.INFO):
        profile.emit_aggregates()
    payloads = _profile_payloads(caplog)
    driver_phases = {
        payload["phase"] for payload in payloads if payload.get("cuda_api") == "driver"
    }
    assert {
        "cuda_stream_create",
        "cuda_event_create",
        "cuda_host_register",
        "cuda_host_unregister",
        "cuda_event_destroy",
        "cuda_stream_destroy",
    } <= driver_phases


@pytest.mark.parametrize(
    ("slot_count", "registration_groups", "expected_group_slots"),
    [
        (28, 28, [1] * 28),
        (28, 14, [2] * 14),
        (28, 7, [4] * 7),
        (28, 4, [7] * 4),
        (28, 2, [14] * 2),
        (28, 1, [28]),
    ],
)
def test_pinned_arena_registration_curve_offsets_and_cleanup(
    monkeypatch,
    slot_count,
    registration_groups,
    expected_group_slots,
):
    slot_size = 4096
    total_size = slot_count * slot_size
    raw = bytearray(total_size)
    freed = []
    calls = []

    class FakeCuda:
        api = "runtime"

        @staticmethod
        def host_register(ptr, size):
            calls.append(("register", ptr, size))

        @staticmethod
        def host_unregister(ptr):
            calls.append(("unregister", ptr))

    monkeypatch.setattr(
        pinned_host,
        "_allocate_aligned_buffer",
        lambda size: (memoryview(raw), raw, 0x100000) if size == total_size else None,
    )
    monkeypatch.setattr(
        pinned_host,
        "_free_aligned_buffer",
        lambda view, ptr: (view.release(), freed.append(ptr)),
    )
    arena = pinned_host.PinnedCopyArena(
        slot_count,
        registration_groups,
        slot_size=slot_size,
        cuda_operations=FakeCuda(),
    )
    claimed = [arena.claim_slot(slot, slot_size) for slot in range(slot_count)]

    pointers = [ptr for _view, _raw, ptr in claimed]
    assert pointers == [0x100000 + slot * slot_size for slot in range(slot_count)]
    assert len(set(pointers)) == slot_count
    assert [size // slot_size for op, _ptr, size in calls if op == "register"] == (
        expected_group_slots
    )
    with pytest.raises(RuntimeError, match="logical slots remain active"):
        arena.close()

    for view, _raw, ptr in claimed:
        arena.release_slot(ptr, view)
    arena.close()

    assert [ptr for op, ptr, *_rest in calls if op == "unregister"] == [
        ptr for op, ptr, *_rest in reversed(calls) if op == "register"
    ]
    assert freed == [0x100000]


def test_pinned_arena_slot_cleanup_precedes_arena_unregister(monkeypatch):
    raw = bytearray(8192)
    calls = []

    class FakeCuda:
        api = "runtime"

        @staticmethod
        def host_register(ptr, size):
            calls.append(("register", ptr, size))

        @staticmethod
        def host_unregister(ptr):
            calls.append(("unregister", ptr))

        @staticmethod
        def stream_create_nonblocking():
            calls.append(("stream_create",))
            return 7

        @staticmethod
        def stream_destroy(stream):
            calls.append(("stream_destroy", stream))

    monkeypatch.setattr(
        pinned_host,
        "_allocate_aligned_buffer",
        lambda _size: (memoryview(raw), raw, 0x200000),
    )
    monkeypatch.setattr(
        pinned_host,
        "_free_aligned_buffer",
        lambda view, ptr: (view.release(), calls.append(("free", ptr))),
    )
    arena = pinned_host.PinnedCopyArena(
        2,
        1,
        slot_size=4096,
        cuda_operations=FakeCuda(),
    )
    slots = [
        pinned_host.PinnedCopySlot(
            size=4096,
            profile=SnapshotProfile("loader", enabled=False),
            cuda_operations=FakeCuda(),
            arena=arena,
            arena_slot=slot,
        )
        for slot in range(2)
    ]
    for slot in slots:
        slot.close()
    arena.close()

    assert calls.index(("stream_destroy", 7)) < calls.index(("unregister", 0x200000))
    assert calls[-1] == ("free", 0x200000)


def test_staging_default_keeps_independent_pinned_slots(monkeypatch):
    captured = []
    session = nixl_staging._NixlPosixStagingTransferSession.__new__(
        nixl_staging._NixlPosixStagingTransferSession
    )
    session._pinned_registration_groups = 0
    session._worker_count = 14
    session._arena = None
    session._arena_lock = threading.Lock()
    monkeypatch.setattr(
        nixl_staging,
        "make_pinned_copy_slots",
        lambda count, **kwargs: captured.append((count, kwargs)) or [],
    )

    assert session._get_arena() is None
    nixl_staging.make_pinned_copy_slots(2, worker=0)
    assert captured == [(2, {"worker": 0})]


@pytest.mark.parametrize(
    "runtime_call",
    [
        lambda: cuda_utils.cuda_runtime_set_device(0),
        lambda: cuda_utils.cuda_host_register(1, 2),
        lambda: cuda_utils.cuda_host_unregister(1),
        cuda_utils.cuda_stream_create_nonblocking,
        lambda: cuda_utils.cuda_stream_destroy(1),
        lambda: cuda_utils.cuda_stream_synchronize(1),
        cuda_utils.cuda_event_create,
        lambda: cuda_utils.cuda_event_record(1, 2),
        lambda: cuda_utils.cuda_event_elapsed_ns(1, 2),
        lambda: cuda_utils.cuda_event_destroy(1),
        lambda: cuda_utils.cuda_memcpy_h2d_async(1, 2, 3, 4),
        lambda: cuda_utils.cuda_memcpy_d2h_async(1, 2, 3, 4),
    ],
)
def test_driver_mode_runtime_guard_fails_before_cudart_call(
    monkeypatch,
    runtime_call,
):
    monkeypatch.setattr(cuda_utils, "_cuda_runtime_calls_forbidden", False)
    cuda_utils.forbid_cuda_runtime_calls()

    with pytest.raises(RuntimeError, match="forbidden in Driver-only loader mode"):
        runtime_call()


def test_driver_load_binds_context_and_passes_driver_operations(monkeypatch):
    calls = []

    class FakeOperations:
        api = "driver"

        def set_current_device(self, device):
            calls.append(("cuCtxSetCurrent", device))
            return 19

    operations = FakeOperations()

    class FakeProcess:
        def device_get(self, device):
            calls.append(("cuDeviceGet", device))
            return device + 10

        def primary_context_retain(self, device, cuda_device):
            calls.append(("retain", device, cuda_device))

        def operations(self, device, _profile):
            calls.append(("operations", device))
            return operations

    class FakeStorageClient:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs["cuda_operations"]))

        def load_to_gms(self, *_args, **_kwargs):
            calls.append("load")

    monkeypatch.delenv(SNAPSHOT_PROFILE_ENV, raising=False)
    monkeypatch.setattr(loader, "GMSStorageClient", FakeStorageClient)
    monkeypatch.setattr(loader, "get_socket_path", lambda _device: "/tmp/gms")

    loader._load_device(
        "/checkpoints/run/versions/1",
        3,
        16,
        "sharded-ssd",
        ["/ssd"],
        2,
        lambda: calls.append("initialization_complete"),
        sharded_ssd_cuda_mode=loader.CUDA_MODE_DRIVER,
        primary_context_retain_complete=lambda: calls.append("retain_complete"),
        driver_process=FakeProcess(),
    )

    assert calls == [
        ("cuDeviceGet", 3),
        ("retain", 3, 13),
        "retain_complete",
        ("operations", 3),
        ("cuCtxSetCurrent", 3),
        "initialization_complete",
        ("client", operations),
        "load",
    ]


def test_driver_main_profiles_initialization_and_releases_after_loads(
    tmp_path,
    monkeypatch,
    caplog,
):
    for device in (0, 1):
        (tmp_path / f"device-{device}").mkdir()
    calls = []

    class StopLoader(Exception):
        pass

    class FakeProcess:
        def initialize(self):
            calls.append("cuInit")

        def close(self):
            calls.append("release")

    process = FakeProcess()

    def fake_load_device(
        _checkpoint_dir,
        device,
        _max_workers,
        _transfer_backend,
        _roots,
        _queues,
        initialization_complete,
        *,
        sharded_ssd_cuda_mode,
        primary_context_retain_complete,
        driver_process,
        mapping_participant,
        **_kwargs,
    ):
        assert sharded_ssd_cuda_mode == loader.CUDA_MODE_DRIVER
        assert driver_process is process
        calls.append(("load", device))
        primary_context_retain_complete()
        initialization_complete()
        mapping_participant.start()
        mapping_participant.complete()
        calls.append(("loaded", device))

    monkeypatch.setenv(SNAPSHOT_PROFILE_ENV, "1")
    monkeypatch.setattr(loader.cuda_utils, "list_devices", lambda: [0, 1])
    monkeypatch.setattr(loader.cuda_utils, "forbid_cuda_runtime_calls", lambda: None)
    monkeypatch.setattr(
        loader.cuda_utils,
        "DriverCudaProcess",
        lambda: process,
    )
    monkeypatch.setattr(loader, "_load_device", fake_load_device)
    monkeypatch.setattr(
        loader.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(StopLoader()),
    )

    with caplog.at_level(logging.INFO), pytest.raises(StopLoader):
        loader.main(
            [
                "--checkpoint-dir",
                str(tmp_path),
                "--transfer-backend",
                "sharded-ssd",
                "--sharded-ssd-cuda-mode",
                "driver",
            ]
        )

    assert calls[0] == "cuInit"
    assert sorted(call for call in calls if isinstance(call, tuple)) == [
        ("load", 0),
        ("load", 1),
        ("loaded", 0),
        ("loaded", 1),
    ]
    assert calls[-1] == "release"
    phases = {payload["phase"] for payload in _profile_payloads(caplog)}
    assert {
        "loader_cu_init",
        "all_device_primary_context_retain_envelope",
        "all_device_cu_ctx_set_current_envelope",
        "all_device_driver_initialization",
        "loader_cuda_initialization_total",
        "primary_context_release",
    } <= phases


def test_loader_cuda_mode_defaults_to_runtime_and_supports_environment(
    monkeypatch,
):
    monkeypatch.delenv(loader.SHARDED_SSD_CUDA_MODE_ENV, raising=False)
    assert (
        loader._build_parser().parse_args([]).sharded_ssd_cuda_mode
        == loader.CUDA_MODE_RUNTIME
    )

    monkeypatch.setenv(
        loader.SHARDED_SSD_CUDA_MODE_ENV,
        loader.CUDA_MODE_DRIVER,
    )
    assert (
        loader._build_parser().parse_args([]).sharded_ssd_cuda_mode
        == loader.CUDA_MODE_DRIVER
    )


def test_loader_rejects_invalid_cuda_mode_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(loader.SHARDED_SSD_CUDA_MODE_ENV, "invalid")

    with pytest.raises(SystemExit):
        loader.main(
            [
                "--checkpoint-dir",
                str(tmp_path),
                "--transfer-backend",
                "sharded-ssd",
            ]
        )


def test_storage_client_skips_redundant_cu_init_only_for_injected_operations(
    monkeypatch,
):
    constructor_args = []

    class FakeMemoryManager:
        def __init__(self, _socket_path, **kwargs):
            constructor_args.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def connect(self, *_args, **_kwargs):
            raise RuntimeError("stop after construction")

    class FakeBackend:
        def start_restore(self, _sources):
            return SimpleNamespace(close=lambda: None)

        def close(self):
            pass

    manifest = SimpleNamespace(allocations=[])
    monkeypatch.setattr(storage_client, "GMSClientMemoryManager", FakeMemoryManager)
    monkeypatch.setattr(
        storage_client,
        "create_transfer_backend",
        lambda *_args, **_kwargs: FakeBackend(),
    )
    monkeypatch.setattr(
        storage_client,
        "load_manifest_and_metadata",
        lambda *_args, **_kwargs: (manifest, {}),
    )

    for operations in (None, SimpleNamespace(api="driver")):
        client = storage_client.GMSStorageClient(
            socket_path="/tmp/gms",
            cuda_operations=operations,
        )
        with pytest.raises(RuntimeError, match="stop after construction"):
            client.load_to_gms("/checkpoint")

    assert constructor_args[0]["cuda_initialized"] is False
    assert constructor_args[1]["cuda_initialized"] is True
