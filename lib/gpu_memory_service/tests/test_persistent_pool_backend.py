# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from gpu_memory_service.client.persistent_pool import V0PersistentPoolBackend
from gpu_memory_service.client.rpc import GMS_ERR_CLAIM_CONFLICT, GmsRemoteError
from gpu_memory_service.common.persistent_pool import PersistentPoolKey


class _Session:
    def __init__(self) -> None:
        self.calls = []

    def claim_persistent(self, **kwargs):
        self.calls.append(("claim", kwargs))
        return SimpleNamespace(
            allocation_id="allocation-1",
            size=6000,
            aligned_size=8192,
            reattached=True,
        )

    def export_persistent(self, **kwargs):
        self.calls.append(("export", kwargs))
        return SimpleNamespace(allocation_id="allocation-1"), 42

    def list_persistent(self, **kwargs):
        self.calls.append(("inventory", kwargs))
        return [
            SimpleNamespace(
                engine_id="engine-a",
                tag="kv_pool",
                allocation_id="allocation-1",
                size=6000,
                aligned_size=8192,
                claimed=False,
            )
        ]

    def release_persistent(self, **kwargs):
        self.calls.append(("destroy", kwargs))
        return True


def test_v0_backend_preserves_claim_and_inventory_semantics():
    session = _Session()
    backend = V0PersistentPoolBackend(session)
    key = PersistentPoolKey("engine-a", "kv_pool")

    claimed = backend.claim(key, 8192, shared=True)
    inventory = backend.inventory("engine-a", include_unclaimed=True)

    assert claimed.key == key
    assert claimed.engine_id == "engine-a"
    assert claimed.tag == "kv_pool"
    assert claimed.allocation_id == "allocation-1"
    assert claimed.size == 6000
    assert claimed.aligned_size == 8192
    assert claimed.reattached is True
    assert claimed.claimed is True
    assert inventory == [
        claimed.__class__(
            key=key,
            allocation_id="allocation-1",
            size=6000,
            aligned_size=8192,
            claimed=False,
        ),
    ]
    assert session.calls == [
        (
            "claim",
            {
                "engine_id": "engine-a",
                "tag": "kv_pool",
                "size": 8192,
                "shared": True,
            },
        ),
        (
            "inventory",
            {"engine_id": "engine-a", "include_unclaimed": True},
        ),
    ]


def test_v0_backend_exports_and_destroys_by_stable_key():
    session = _Session()
    backend = V0PersistentPoolBackend(session)
    key = PersistentPoolKey("engine-a", "kv_pool")

    assert backend.export(key) == 42
    assert backend.destroy(key) is True
    assert session.calls == [
        ("export", {"engine_id": "engine-a", "tag": "kv_pool"}),
        ("destroy", {"engine_id": "engine-a", "tag": "kv_pool"}),
    ]


def test_v0_backend_retries_only_claim_conflicts(monkeypatch):
    class ConflictThenSuccessSession(_Session):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def claim_persistent(self, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise GmsRemoteError("still owned", GMS_ERR_CLAIM_CONFLICT)
            return super().claim_persistent(**kwargs)

    monkeypatch.setenv("GMS_PERSISTENT_CLAIM_RETRY_SECS", "1")
    session = ConflictThenSuccessSession()

    allocation = V0PersistentPoolBackend(session).claim(
        PersistentPoolKey("engine-a", "kv_pool"), 8192
    )

    assert allocation.allocation_id == "allocation-1"
    assert session.attempts == 2


def test_v0_backend_does_not_retry_other_remote_errors(monkeypatch):
    session = _Session()
    session.claim_persistent = lambda **_kwargs: (_ for _ in ()).throw(
        GmsRemoteError("fatal", 99)
    )
    monkeypatch.setenv("GMS_PERSISTENT_CLAIM_RETRY_SECS", "1")

    with pytest.raises(GmsRemoteError, match="fatal"):
        V0PersistentPoolBackend(session).claim(
            PersistentPoolKey("engine-a", "kv_pool"), 8192
        )
