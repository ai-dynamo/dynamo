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

    class FakeStorageClient:
        def __init__(self, output_dir, **kwargs):
            calls.append(("init", output_dir, kwargs))

        def save(self, *, max_workers):
            calls.append(("save", {"max_workers": max_workers}))

    monkeypatch.setattr(saver, "get_socket_path", lambda device: f"/tmp/gms-{device}")
    monkeypatch.setattr(saver, "GMSStorageClient", FakeStorageClient)
    monkeypatch.setattr(
        saver.cuda_utils,
        "cuda_runtime_set_device",
        lambda device: calls.append(("set_device", device)),
    )

    saver._save_device(
        "/checkpoints/run/versions/1",
        3,
        8,
        60_000,
        4 * 1024**3,
        [],
    )

    assert calls[0] == ("set_device", 3)
    assert calls[1][0] == "init"
    assert calls[1][1] == "/checkpoints/run/versions/1/device-3"
    assert calls[1][2]["socket_path"] == "/tmp/gms-3"
    assert calls[1][2]["device"] == 3
    assert calls[2] == ("save", {"max_workers": 8})


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
    monkeypatch.setattr(
        pinned_host.cuda_utils,
        "cuda_stream_create_nonblocking",
        lambda: "stream",
    )
    monkeypatch.setattr(
        pinned_host.cuda_utils,
        "cuda_event_create",
        lambda: next(events),
    )
    monkeypatch.setattr(
        pinned_host.cuda_utils,
        "cuda_host_register",
        lambda _ptr, _size: calls.append("register"),
    )
    monkeypatch.setattr(
        pinned_host.cuda_utils,
        "cuda_event_record",
        lambda event, _stream: calls.append(("record", event)),
    )
    monkeypatch.setattr(
        pinned_host.cuda_utils,
        "cuda_memcpy_d2h_async",
        lambda *_args: calls.append("copy"),
    )
    monkeypatch.setattr(
        pinned_host.cuda_utils,
        "cuda_stream_synchronize",
        lambda _stream: calls.append("synchronize"),
    )
    monkeypatch.setattr(
        pinned_host.cuda_utils,
        "cuda_event_elapsed_ns",
        lambda _start, _end: calls.append("elapsed") or 77,
    )
    monkeypatch.setattr(
        pinned_host.cuda_utils,
        "cuda_host_unregister",
        lambda _ptr: calls.append("unregister"),
    )
    monkeypatch.setattr(
        pinned_host.cuda_utils,
        "cuda_event_destroy",
        lambda event: calls.append(("destroy", event)),
    )
    monkeypatch.setattr(
        pinned_host.cuda_utils,
        "cuda_stream_destroy",
        lambda _stream: calls.append("destroy_stream"),
    )

    profile = SnapshotProfile("saver", logger=saver.logger, enabled=True)
    slot = pinned_host.PinnedCopySlot(size=16, profile=profile)
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
