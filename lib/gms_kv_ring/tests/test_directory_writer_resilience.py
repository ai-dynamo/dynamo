# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from gms_kv_ring.common.content_directory import ContentDirectory
from gms_kv_ring.daemon.rpc_directory import (
    SERVER_CONNECTION_ID,
    handle_directory_ensure_hbm_capacity,
    handle_directory_lookup_claim,
    release_directory_connection_claims,
)

pytestmark = pytest.mark.pre_merge


def test_publish_worker_survives_a_failing_mutation():
    directory = ContentDirectory(
        "/tmp/gms-directory-writer-resilience.sock",
        engine="test",
        block_size=16,
        mode="shadow",
    )
    try:
        calls: list[list[dict]] = []

        def flaky_publish(items):
            calls.append(items)
            if len(calls) == 1:
                raise RuntimeError("simulated transient publish failure")
            return len(items)

        directory.publish = flaky_publish  # type: ignore[method-assign]
        directory._defer_mutation("publish", [{"a": 1}])
        directory._defer_mutation("publish", [{"b": 2}])

        assert directory.flush_deferred(timeout=5.0) is True
        assert len(calls) == 2, "worker died instead of continuing past the failure"
        assert directory._mutation_failed == 1
        assert (
            directory._mutation_error is None
        ), "per-mutation failure must not be fatal"
        assert (
            directory._mutation_thread is not None
            and directory._mutation_thread.is_alive()
        )
    finally:
        directory.close()


def test_zero_capacity_request_preserves_ready_hbm_entry():
    content_hash = b"h" * 32
    key = ("manifest", content_hash)
    entry = {
        "tier": "hbm",
        "state": "ready",
        "engine_id": "engine",
        "slot_ids": [7],
        "generations": [3],
        "_claim_count": 0,
    }
    daemon = SimpleNamespace(
        _content_hash_lock=threading.Condition(),
        _content_directory_writer_id="writer",
        _content_directory_epoch=4,
        _content_directory={key: entry},
    )

    response = handle_directory_ensure_hbm_capacity(
        daemon,
        {
            "manifest_id": "manifest",
            "writer_id": "writer",
            "expected_epoch": 4,
            "required_blocks": 0,
        },
    )

    assert response == {
        "ok": True,
        "victims": [],
        "freed_blocks": 0,
        "rejected_stale_writer": False,
    }
    assert daemon._content_directory == {key: entry}


@pytest.mark.parametrize("eligible", [[], [8]])
def test_capacity_retires_only_native_free_eligible_slots(monkeypatch, eligible):
    from gms_kv_ring.daemon import rpc_directory

    entries = {
        ("manifest", bytes([page]) * 32): {
            "tier": "hbm",
            "state": "ready",
            "engine_id": "engine",
            "slot_ids": [page],
            "generations": [3],
            "_claim_count": 0,
            "_last_access_seq": page,
        }
        for page in (7, 8)
    }
    daemon = SimpleNamespace(
        _content_hash_lock=threading.Condition(),
        _content_directory_writer_id="writer",
        _content_directory_epoch=4,
        _content_directory=entries,
    )
    monkeypatch.setattr(
        rpc_directory,
        "_directory_remove_locked",
        lambda daemon, key: daemon._content_directory.pop(key),
    )
    response = handle_directory_ensure_hbm_capacity(
        daemon,
        {
            "manifest_id": "manifest",
            "writer_id": "writer",
            "expected_epoch": 4,
            "required_blocks": 1,
            "eligible_slot_ids": eligible,
        },
    )
    assert response["ok"] is True
    assert [victim["slot_ids"][0] for victim in response["victims"]] == eligible
    # Page 7 is older but still native-resident: generic LRU must not retire it.
    assert ("manifest", bytes([7]) * 32) in entries


@pytest.mark.parametrize(
    ("pending_generations", "expected_hit"),
    [(None, False), ([4], True)],
)
def test_active_hbm_is_claimable_only_during_adoption(
    pending_generations, expected_hit
):
    content_hash = b"h" * 32
    key = ("manifest", content_hash)
    entry = {
        "tier": "hbm",
        "state": "active",
        "engine_id": "engine",
        "slot_ids": [7],
        "generations": [3],
        "_claim_count": 0,
        "_owner_writer": "writer",
    }
    if pending_generations is not None:
        entry["_pending_generations"] = pending_generations
    daemon = SimpleNamespace(
        _content_hash_lock=threading.Condition(),
        _content_directory_writer_id="writer",
        _content_directory_epoch=4,
        _content_directory={key: entry},
        _content_directory_claims={},
        _content_directory_access_seq=0,
    )

    response = handle_directory_lookup_claim(
        daemon,
        {
            "manifest_id": "manifest",
            "writer_id": "writer",
            "expected_epoch": 4,
            "hashes": [content_hash.hex()],
        },
    )

    assert (response["entries"][0] is not None) is expected_hit
    assert (response["claim_token"] is not None) is expected_hit
    assert entry["_claim_count"] == int(expected_hit)


def test_async_view_does_not_hide_tp_adoption_pending_hbm(monkeypatch):
    content_hash = b"h" * 32
    directory = ContentDirectory(
        "/tmp/gms-directory-tp-adoption.sock",
        engine="test",
        block_size=16,
        mode="shadow",
    )
    directory._view = {
        content_hash: {
            "tier": "hbm",
            "state": "active",
            "slot_ids": [7],
            "generations": [3],
            "_pending_generations": [4],
        }
    }
    directory._view_ready.set()
    directory.start_async_read = lambda: None  # type: ignore[method-assign]
    seen = []

    class Client:
        def directory_lookup_claim(self, manifest, writer, epoch, hashes):
            seen.append((manifest, writer, epoch, hashes))
            return ([{"tier": "hbm", "state": "active"}], "claim", False, epoch)

    directory._writer_epoch = 4
    monkeypatch.setattr(
        directory,
        "_call",
        lambda operation, **_kwargs: operation(Client()),
    )

    entries, token = directory.lookup_and_claim([content_hash])

    assert entries == [{"tier": "hbm", "state": "active"}]
    assert token == "claim"
    assert seen and seen[0][3] == [content_hash]


def test_authoritative_lookup_bypasses_async_view(monkeypatch):
    content_hash = b"h" * 32
    stale = {"state": "ready", "tier": "hbm", "slot_ids": [1]}
    committed = {"state": "ready", "tier": "hbm", "slot_ids": [2]}
    directory = ContentDirectory(
        "/tmp/gms-directory-authoritative-read.sock",
        engine="test",
        block_size=16,
        mode="shadow",
    )
    directory._view = {content_hash: stale}
    directory._view_ready.set()
    directory.start_async_read = lambda: None  # type: ignore[method-assign]
    calls = []

    class Client:
        def directory_lookup(self, manifest, hashes):
            calls.append((manifest, hashes))
            return [committed], 4, "writer"

    monkeypatch.setattr(
        directory,
        "_call",
        lambda operation, **_kwargs: operation(Client()),
    )

    assert directory.lookup([content_hash]) == [stale]
    assert directory.lookup_authoritative([content_hash]) == [committed]
    assert calls == [(directory.manifest_id, [content_hash])]


def test_disconnect_releases_only_that_connections_claims():
    hashes = (b"a" * 32, b"b" * 32)
    entries = {}
    for slot_id, content_hash in enumerate(hashes, start=7):
        entries[("manifest", content_hash)] = {
            "tier": "hbm",
            "state": "ready",
            "engine_id": "engine",
            "slot_ids": [slot_id],
            "generations": [3],
            "_claim_count": 0,
        }
    daemon = SimpleNamespace(
        _content_hash_lock=threading.Condition(),
        _content_directory_writer_id="writer",
        _content_directory_epoch=4,
        _content_directory=entries,
        _content_directory_claims={},
        _content_directory_access_seq=0,
    )
    tokens = []
    for connection_id, content_hash in zip(("lost", "live"), hashes):
        response = handle_directory_lookup_claim(
            daemon,
            {
                "manifest_id": "manifest",
                "writer_id": "writer",
                "expected_epoch": 4,
                "hashes": [content_hash.hex()],
                SERVER_CONNECTION_ID: connection_id,
            },
        )
        tokens.append(response["claim_token"])

    assert release_directory_connection_claims(daemon, "lost") == 1
    assert tokens[0] not in daemon._content_directory_claims
    assert tokens[1] in daemon._content_directory_claims
    assert entries[("manifest", hashes[0])]["_claim_count"] == 0
    assert entries[("manifest", hashes[1])]["_claim_count"] == 1
