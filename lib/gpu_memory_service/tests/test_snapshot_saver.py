# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GMS snapshot saver CLI."""

import pytest

try:
    from gpu_memory_service.cli.snapshot import saver
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
