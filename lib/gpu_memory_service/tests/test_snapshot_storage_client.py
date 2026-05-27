# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GMS snapshot storage-client restore sequencing."""

from types import SimpleNamespace

import pytest

try:
    from gpu_memory_service.snapshot import storage_client
    from gpu_memory_service.snapshot.model import AllocationEntry, SaveManifest
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


def test_load_to_gms_streams_targets_in_manifest_order(monkeypatch):
    events = []

    manifest = SaveManifest(
        version="1.0",
        timestamp=1.0,
        layout_hash="hash",
        device=0,
        allocations=[
            AllocationEntry(
                allocation_id="old-a",
                size=4096,
                aligned_size=4096,
                tag="weight",
                tensor_file="shard_0000.bin",
            ),
            AllocationEntry(
                allocation_id="old-b",
                size=8192,
                aligned_size=8192,
                tag="weight",
                tensor_file="shard_0001.bin",
            ),
        ],
    )
    metadata = {
        "tensor": {
            "allocation_id": "old-b",
            "offset_bytes": 128,
            "value": b"payload",
        }
    }

    class FakeSession:
        def restore(self, _targets):
            events.append(("restore",))

        def submit_targets(self, targets):
            events.append(("submit", tuple(targets)))

        def finish_restore(self):
            events.append(("finish_restore",))

        def close(self):
            events.append(("session_close",))

    session = FakeSession()

    class FakeBackend:
        name = "fake-streaming"

        def start_restore(self, sources):
            events.append(
                ("start_restore", [source.allocation_id for source in sources])
            )
            return session

        def close(self):
            events.append(("backend_close",))

    class FakeMM:
        def __init__(self, _socket_path, *, device):
            self.device = device
            self.mappings = {}
            self._next_va = 0x1000

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            events.append(("mm_exit",))

        def connect(self, lock_type, *, timeout_ms):
            events.append(("connect", lock_type, timeout_ms))

        def create_mapping(self, *, size, tag):
            va = self._next_va
            self._next_va += 0x1000
            self.mappings[va] = SimpleNamespace(
                allocation_id=f"new-{len(self.mappings)}"
            )
            events.append(("create_mapping", size, tag))
            return va

        def metadata_put(self, key, allocation_id, offset_bytes, value):
            events.append(("metadata_put", key, allocation_id, offset_bytes, value))
            return True

        def commit(self):
            events.append(("commit",))
            return True

    monkeypatch.setattr(storage_client, "_GMS_CORE_IMPORTS_AVAILABLE", True)
    monkeypatch.setattr(storage_client, "GMSClientMemoryManager", FakeMM)
    monkeypatch.setattr(
        storage_client,
        "RequestedLockType",
        SimpleNamespace(RW="rw"),
    )
    monkeypatch.setattr(
        storage_client,
        "_load_manifest_and_metadata",
        lambda _input_dir: (manifest, metadata),
    )
    monkeypatch.setattr(
        storage_client,
        "create_transfer_backend",
        lambda _name, _config: FakeBackend(),
    )

    client = storage_client.GMSStorageClient(socket_path="/tmp/gms.sock", device=0)

    id_map = client.load_to_gms("/checkpoint/device-0")

    assert id_map == {"old-a": "new-0", "old-b": "new-1"}
    assert events[:6] == [
        ("start_restore", ["old-a", "old-b"]),
        ("connect", "rw", None),
        ("create_mapping", 4096, "weight"),
        ("submit", ("old-a",)),
        ("create_mapping", 8192, "weight"),
        ("submit", ("old-b",)),
    ]
    assert ("finish_restore",) in events
    assert ("restore",) not in events
    assert ("metadata_put", "tensor", "new-1", 128, b"payload") in events
    assert ("commit",) in events
