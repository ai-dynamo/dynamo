# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GMS snapshot saver CLI."""

import pytest

try:
    from gpu_memory_service.cli.snapshot import saver
    from gpu_memory_service.common.locks import GrantedLockType
    from gpu_memory_service.snapshot import storage_client
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


def test_save_device_holds_same_ro_lease_through_save(monkeypatch):
    calls = []

    class FakeSession:
        def __init__(self, socket_path, lock_type, timeout_ms):
            calls.append(("connect", socket_path, lock_type, timeout_ms))

        def close(self):
            calls.append(("disconnect",))

    class FakeStorageClient:
        def __init__(self, output_dir, **kwargs):
            calls.append(("init", output_dir, kwargs))

        def save(self, *, max_workers, session):
            calls.append(("save", {"max_workers": max_workers}, session))
            # GMSStorageClient.save() adopts the session and closes it.
            session.close()

    monkeypatch.setattr(saver, "get_socket_path", lambda device: f"/tmp/gms-{device}")
    monkeypatch.setattr(saver, "GMSClientSession", FakeSession)
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

    assert calls[0] == (
        "connect",
        "/tmp/gms-3",
        saver.RequestedLockType.RO,
        60_000,
    )
    assert calls[1] == ("set_device", 3)
    assert calls[2][0] == "init"
    assert calls[2][1] == "/checkpoints/run/versions/1/device-3"
    assert calls[2][2]["socket_path"] == "/tmp/gms-3"
    assert calls[2][2]["device"] == 3
    assert calls[3][0:2] == ("save", {"max_workers": 8})
    assert isinstance(calls[3][2], FakeSession)
    assert calls[4] == ("disconnect",)
    assert calls.count(("disconnect",)) == 1


def test_save_device_closes_session_when_cuda_setup_fails(monkeypatch):
    close_calls = []

    class FakeSession:
        def __init__(self, *_args):
            pass

        def close(self):
            close_calls.append("close")

    def fail_set_device(_device):
        raise RuntimeError("CUDA setup failed")

    monkeypatch.setattr(saver, "get_socket_path", lambda _device: "/tmp/gms")
    monkeypatch.setattr(saver, "GMSClientSession", FakeSession)
    monkeypatch.setattr(saver.cuda_utils, "cuda_runtime_set_device", fail_set_device)

    with pytest.raises(RuntimeError, match="CUDA setup failed"):
        saver._save_device("/checkpoints", 0, 1, 100, 1024, [])

    assert close_calls == ["close"]


def test_storage_save_closes_owned_session_on_output_validation_failure():
    close_calls = []

    class FakeSession:
        def close(self):
            close_calls.append("close")

    client = storage_client.GMSStorageClient(
        output_dir=None,
        socket_path="/tmp/gms",
    )

    with pytest.raises(ValueError, match="output_dir must be set"):
        client.save(session=FakeSession())

    assert close_calls == ["close"]


def test_storage_save_closes_adopted_session_on_shard_failure(monkeypatch):
    calls = []

    class FakeSession:
        lock_type = GrantedLockType.RO

        def close(self):
            calls.append("session_close")

    class FakeMemoryManager:
        def __init__(self, *_args, **_kwargs):
            self.session = None

        def adopt_session(self, session):
            calls.append("adopt")
            self.session = session

        def get_memory_layout_hash(self):
            return "layout-hash"

        def list_handles(self):
            return []

        def close(self):
            calls.append("manager_close")
            self.session.close()

    client = storage_client.GMSStorageClient(
        output_dir="/snapshot",
        socket_path="/tmp/gms",
    )
    monkeypatch.setattr(storage_client, "GMSClientMemoryManager", FakeMemoryManager)
    monkeypatch.setattr(
        client,
        "_prepare_output_dir",
        lambda: ("/snapshot", ["/snapshot/shards"], False),
    )
    monkeypatch.setattr(
        client,
        "_write_shards",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("shard failed")),
    )

    with pytest.raises(RuntimeError, match="shard failed"):
        client.save(session=FakeSession())

    assert calls == ["adopt", "manager_close", "session_close"]
