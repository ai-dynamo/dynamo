# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GMS storage client CLI."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

try:
    from gpu_memory_service.cli import storage_runner
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


def test_run_save_sets_cuda_context_before_storage_client(monkeypatch):
    calls = []

    class FakeStorageClient:
        def __init__(self, output_dir, **kwargs):
            calls.append(("init", output_dir, kwargs))

        def save(self, *, max_workers):
            calls.append(("save", {"max_workers": max_workers}))
            return SimpleNamespace(
                allocations=[SimpleNamespace(tensor_file="shard_0000.bin")],
                layout_hash="abc123",
            )

    fake_storage_client = ModuleType("gpu_memory_service.snapshot.storage_client")
    fake_storage_client.GMSStorageClient = FakeStorageClient
    monkeypatch.setitem(
        sys.modules,
        "gpu_memory_service.snapshot.storage_client",
        fake_storage_client,
    )
    monkeypatch.setattr(storage_runner, "_resolve_socket", lambda *_: "/tmp/gms-3")
    monkeypatch.setattr(
        storage_runner.cuda_utils,
        "cuda_runtime_set_device",
        lambda device: calls.append(("set_device", device)),
    )

    storage_runner._run_save(
        SimpleNamespace(
            verbose=False,
            device=3,
            socket_path=None,
            output_dir="/checkpoints",
            save_workers=8,
            sharded_ssd_roots="",
            timeout_ms=60_000,
            shard_size_bytes=4 * 1024**3,
        )
    )

    assert calls[0] == ("set_device", 3)
    assert calls[1][0] == "init"
    assert calls[1][1] == "/checkpoints"
    assert calls[1][2]["socket_path"] == "/tmp/gms-3"
    assert calls[1][2]["device"] == 3
    assert calls[2] == ("save", {"max_workers": 8})


def test_run_load_sets_cuda_context_before_storage_client(monkeypatch):
    calls = []

    class FakeStorageClient:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def load_to_gms(self, input_dir, *, max_workers, clear_existing):
            calls.append(
                (
                    "load_to_gms",
                    {
                        "input_dir": input_dir,
                        "max_workers": max_workers,
                        "clear_existing": clear_existing,
                    },
                )
            )
            return {"old": "new"}

    fake_storage_client = ModuleType("gpu_memory_service.snapshot.storage_client")
    fake_storage_client.GMSStorageClient = FakeStorageClient
    monkeypatch.setitem(
        sys.modules,
        "gpu_memory_service.snapshot.storage_client",
        fake_storage_client,
    )
    monkeypatch.setattr(storage_runner, "_resolve_socket", lambda *_: "/tmp/gms-3")
    monkeypatch.setattr(
        storage_runner.cuda_utils,
        "cuda_runtime_set_device",
        lambda device: calls.append(("set_device", device)),
    )

    storage_runner._run_load(
        SimpleNamespace(
            verbose=False,
            device=3,
            socket_path=None,
            input_dir="/checkpoints",
            no_clear=False,
            transfer_backend="nixl",
            timeout_ms=60_000,
            sharded_ssd_roots="",
            sharded_ssd_queues_per_root=2,
            workers=16,
        )
    )

    assert calls[0] == ("set_device", 3)
    assert calls[1][0] == "init"
    assert calls[1][1]["socket_path"] == "/tmp/gms-3"
    assert calls[1][1]["device"] == 3
    assert calls[2] == (
        "load_to_gms",
        {
            "input_dir": "/checkpoints",
            "max_workers": 16,
            "clear_existing": True,
        },
    )
