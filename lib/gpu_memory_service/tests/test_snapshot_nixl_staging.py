# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NIXL/POSIX staging restore planning."""

import re
import threading
from pathlib import Path

import pytest

try:
    from gpu_memory_service.snapshot.backends.nixl_common import (
        NixlFileGroup,
        NixlWorkGroup,
        split_work_groups,
    )
    from gpu_memory_service.snapshot.backends.nixl_staging import (
        _NixlPosixStagingTransferSession,
    )
    from gpu_memory_service.snapshot.transfer import FileTransferSource
except ModuleNotFoundError:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from _fake_vmm import FakeVMM
from gpu_memory_service.common.vmm import VMMDeviceType
from gpu_memory_service.snapshot.backends import pinned_host
from gpu_memory_service.snapshot.backends.pinned_host import PinnedCopySlot

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def _work_group(name: str, byte_count: int) -> NixlWorkGroup:
    source = FileTransferSource(
        allocation_id=name,
        file_path=f"/checkpoint/{name}.bin",
        file_offset=0,
        byte_count=byte_count,
    )
    file_group: NixlFileGroup = (source.file_path, [source])
    return name, [file_group]


def test_split_work_groups_limits_bucket_count():
    work_groups = [_work_group(f"shard-{idx}", 1) for idx in range(142)]

    buckets = split_work_groups(work_groups, worker_count=8)

    assert len(buckets) == 8
    assert sum(len(file_groups) for _name, file_groups in buckets) == 142
    expected = sorted(f"shard-{idx}" for idx in range(142))
    actual = sorted(
        source.allocation_id
        for _bucket_name, file_groups in buckets
        for _file_path, sources in file_groups
        for source in sources
    )
    assert actual == expected


def test_split_work_groups_balances_by_bytes():
    work_groups = [
        _work_group("large", 10),
        _work_group("medium-a", 6),
        _work_group("medium-b", 6),
        _work_group("small-a", 1),
        _work_group("small-b", 1),
    ]

    buckets = split_work_groups(work_groups, worker_count=2)
    bucket_sizes = sorted(
        sum(source.byte_count for _path, sources in file_groups for source in sources)
        for _name, file_groups in buckets
    )

    assert bucket_sizes == [12, 12]


def test_staging_prep_starts_before_restore(monkeypatch):
    from gpu_memory_service.snapshot.backends import nixl_staging

    source = FileTransferSource(
        allocation_id="alloc-0",
        file_path="/checkpoint/shard.bin",
        file_offset=0,
        byte_count=4096,
    )
    prep_started = threading.Event()
    allow_finish = threading.Event()

    class FakeApi:
        @staticmethod
        def agent_config_type(*, backends):
            return {"backends": backends}

        @staticmethod
        def agent_type(_agent_name, _config):
            return FakeAgent()

    class FakeAgent:
        def create_backend(self, _backend_name, backend_params=None):
            self.backend_params = backend_params

    def group_sources(_sources):
        return {"file": [(source.file_path, [source])]}

    def fake_load_nixl_api():
        prep_started.set()
        assert allow_finish.wait(timeout=1.0)
        return FakeApi()

    fake_vmm = FakeVMM()
    monkeypatch.setattr(nixl_staging, "get_vmm", lambda: fake_vmm)
    monkeypatch.setattr(nixl_staging, "load_nixl_api", fake_load_nixl_api)
    monkeypatch.setattr(nixl_staging, "make_pinned_copy_slots", lambda _vmm, _count: [])

    session = _NixlPosixStagingTransferSession(
        backend_name="test-backend",
        device=0,
        max_workers=1,
        group_sources=group_sources,
        group_kind="file",
        warn_under_parallelized=False,
        posix_backend_params={"ios_pool_size": "64", "kernel_queue_size": "16"},
        sources=[source],
    )
    try:
        assert prep_started.wait(timeout=1.0)
    finally:
        allow_finish.set()
        session.close()


_SYCL_UNSUPPORTED_REGISTER_MESSAGES = (
    "_sycl_vmm: register_host_memory not available (oneAPI too old)",
    "_sycl_vmm: device does not support host memory registration "
    "(aspect::ext_oneapi_register_host_memory not advertised by this device).",
)


class _HostRegisterVMM(FakeVMM):
    """Records host-register/stream calls; optionally fails host_register."""

    def __init__(self, register_error=None):
        super().__init__()
        self.register_error = register_error
        self.registered = []
        self.unregistered = []
        self.destroyed_streams = []

    def host_register(self, ptr, size):
        if self.register_error is not None:
            raise self.register_error
        self.registered.append((ptr, size))

    def host_unregister(self, ptr):
        self.unregistered.append(ptr)

    def stream_destroy(self, stream):
        self.destroyed_streams.append(stream)


@pytest.fixture
def freed_buffers(monkeypatch):
    freed = []
    free = pinned_host._free_aligned_buffer

    def record_free(view, ptr):
        freed.append(ptr)
        free(view, ptr)

    monkeypatch.setattr(pinned_host, "_free_aligned_buffer", record_free)
    return freed


def _use_device_type(monkeypatch, device_type):
    monkeypatch.setattr(pinned_host, "get_vmm_device_type", lambda: device_type)


def test_pinned_copy_slot_registers_on_cuda(monkeypatch, freed_buffers):
    _use_device_type(monkeypatch, VMMDeviceType.CUDA)
    vmm = _HostRegisterVMM()

    slot = PinnedCopySlot(vmm, size=8192)
    assert slot._registered is True
    assert vmm.registered == [(slot.ptr, 8192)]

    slot.close()
    assert vmm.unregistered == [slot.ptr]
    assert vmm.destroyed_streams == ["fake_stream"]
    assert freed_buffers == [slot.ptr]


@pytest.mark.parametrize("message", _SYCL_UNSUPPORTED_REGISTER_MESSAGES)
def test_pinned_copy_slot_skips_unsupported_host_register_on_xpu(
    monkeypatch, freed_buffers, message
):
    _use_device_type(monkeypatch, VMMDeviceType.XPU)
    vmm = _HostRegisterVMM(RuntimeError(message))

    slot = PinnedCopySlot(vmm, size=8192)
    assert slot._registered is False
    assert freed_buffers == []

    slot.close()
    assert vmm.unregistered == []
    assert vmm.destroyed_streams == ["fake_stream"]
    assert freed_buffers == [slot.ptr]


@pytest.mark.parametrize(
    "device_type, register_error",
    [
        (VMMDeviceType.CUDA, RuntimeError(_SYCL_UNSUPPORTED_REGISTER_MESSAGES[0])),
        (VMMDeviceType.XPU, RuntimeError("_sycl_vmm: register_host_memory failed")),
        (VMMDeviceType.XPU, ValueError(_SYCL_UNSUPPORTED_REGISTER_MESSAGES[0])),
    ],
    ids=["cuda-matching-message", "xpu-other-runtime-error", "xpu-not-runtime-error"],
)
def test_pinned_copy_slot_cleans_up_and_fails_on_other_register_errors(
    monkeypatch, freed_buffers, device_type, register_error
):
    _use_device_type(monkeypatch, device_type)
    vmm = _HostRegisterVMM(register_error)

    with pytest.raises(type(register_error)) as raised:
        PinnedCopySlot(vmm, size=8192)

    assert raised.value is register_error
    assert vmm.unregistered == []
    assert vmm.destroyed_streams == ["fake_stream"]
    assert len(freed_buffers) == 1


def test_unsupported_host_register_messages_match_sycl_vmm():
    """The XPU fallback matches on _sycl_vmm's error text; catch drift in CI."""
    source = (
        Path(__file__).resolve().parents[1]
        / "common"
        / "vmm"
        / "_sycl_vmm"
        / "sycl_vmm.cpp"
    )
    if not source.is_file():
        pytest.skip(f"{source} is not available in this test image")

    # Adjacent C++ string literals are concatenated, so join them first.
    text = re.sub(r'"\s*"', "", source.read_text()).lower()
    for message in pinned_host._UNSUPPORTED_HOST_REGISTER_MESSAGES:
        assert message in text, f"{message!r} no longer raised by sycl_vmm.cpp"
