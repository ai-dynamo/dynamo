# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GMS snapshot saver CLI."""

import json
import logging

import pytest

try:
    from gpu_memory_service.cli.snapshot import saver
    from gpu_memory_service.common.snapshot_profile import SnapshotProfile
    from gpu_memory_service.snapshot.backends import pinned_host
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


def test_save_device_sets_cuda_context_before_storage_client(monkeypatch):
    calls = []

    class FakeVMM:
        def ensure_initialized(self):
            calls.append(("ensure_initialized",))

        def runtime_set_device(self, device):
            calls.append(("set_device", device))

    class FakeStorageClient:
        def __init__(self, output_dir, **kwargs):
            calls.append(("init", output_dir, kwargs))

        def save(self, *, max_workers):
            calls.append(("save", {"max_workers": max_workers}))

    monkeypatch.setattr(saver, "get_socket_path", lambda device: f"/tmp/gms-{device}")
    monkeypatch.setattr(saver, "GMSStorageClient", FakeStorageClient)
    monkeypatch.setattr(saver, "get_vmm", lambda: FakeVMM())

    saver._save_device(
        "/checkpoints/run/versions/1",
        3,
        8,
        60_000,
        4 * 1024**3,
        [],
    )

    assert calls[0] == ("ensure_initialized",)
    assert calls[1] == ("set_device", 3)
    assert calls[2][0] == "init"
    assert calls[2][1] == "/checkpoints/run/versions/1/device-3"
    assert calls[2][2]["socket_path"] == "/tmp/gms-3"
    assert calls[2][2]["device"] == 3
    assert calls[3] == ("save", {"max_workers": 8})


def test_pinned_slot_profiles_cuda_event_after_existing_wait(monkeypatch, caplog):
    calls = []
    events = iter(["start", "end"])

    def allocate(size):
        raw = bytearray(size)
        return memoryview(raw), raw, 123

    monkeypatch.setattr(pinned_host, "_allocate_aligned_buffer", allocate)
    monkeypatch.setattr(
        pinned_host,
        "_free_aligned_buffer",
        lambda _view, _ptr: calls.append("free"),
    )

    class FakeCopyVMM:
        def stream_create_nonblocking(self):
            return "stream"

        def event_create(self):
            return next(events)

        def host_register(self, _ptr, _size):
            calls.append("register")

        def event_record(self, event, _stream):
            calls.append(("record", event))

        def memcpy_d2h_async(self, *_args):
            calls.append("copy")

        def stream_synchronize(self, _stream):
            calls.append("synchronize")

        def event_elapsed_ns(self, _start, _end):
            calls.append("elapsed")
            return 77

        def host_unregister(self, _ptr):
            calls.append("unregister")

        def event_destroy(self, event):
            calls.append(("destroy", event))

        def stream_destroy(self, _stream):
            calls.append("destroy_stream")

    profile = SnapshotProfile("saver", logger=saver.logger, enabled=True)
    slot = pinned_host.PinnedCopySlot(FakeCopyVMM(), size=16, profile=profile)
    slot.copy_from_device_async(456, 8)
    assert "synchronize" not in calls
    slot.wait()
    assert calls.index("synchronize") < calls.index("elapsed")
    assert calls.count("synchronize") == 1
    slot.close()

    with caplog.at_level(logging.INFO):
        profile.emit_aggregates()
    payloads = [
        json.loads(record.getMessage().removeprefix("GMS_SNAPSHOT_PROFILE "))
        for record in caplog.records
        if record.getMessage().startswith("GMS_SNAPSHOT_PROFILE ")
    ]
    device = next(payload for payload in payloads if payload["phase"] == "d2h_device")
    assert device["duration_ns"] == 77
    assert device["count"] == 1
    assert device["bytes"] == 8
    assert ("destroy", "start") in calls
    assert ("destroy", "end") in calls
