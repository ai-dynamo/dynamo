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
    handle_directory_promote,
    release_directory_connection_claims,
)

pytestmark = pytest.mark.pre_merge


def test_freeze_current_writer_view_preserves_epoch_and_publication_pipeline():
    directory = ContentDirectory(
        "/tmp/gms-directory-freeze-writer.sock",
        engine="test",
        block_size=16,
        mode="shadow",
    )
    directory._view_ready.set()
    directory._view_current_writer = True
    directory._view_epoch = 7
    directory._view_revision = 11

    assert directory.freeze_current_writer_view() is True
    assert directory.async_read_enabled is False
    assert directory.read_view_is_current_writer is True
    assert directory.read_view_cursor == (7, 11)
    assert directory._pipeline_stop is False


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
        # Queue both writes atomically so this test covers a failed combined
        # batch instead of depending on whether the worker wins a scheduling
        # race after the first enqueue.
        with directory._mutation_condition:
            directory._mutation_sequence = 2
            directory._mutations.extend(
                [
                    (
                        1,
                        "publish",
                        [
                            {
                                "content_hash": b"a",
                                "engine_id": "engine",
                                "slot_ids": [1],
                            }
                        ],
                    ),
                    (
                        2,
                        "publish",
                        [
                            {
                                "content_hash": b"b",
                                "engine_id": "engine",
                                "slot_ids": [2],
                            }
                        ],
                    ),
                ]
            )
            directory._start_mutation_worker_locked()
            directory._mutation_condition.notify()
        with pytest.raises(RuntimeError, match="mutations failed"):
            directory.flush_deferred(timeout=5.0)

        # A failed batch is a safe miss for every item in it. The worker must
        # remain available for the next independent mutation.
        directory._defer_mutation(
            "publish", [{"content_hash": b"c", "engine_id": "engine", "slot_ids": [3]}]
        )
        with pytest.raises(RuntimeError, match="mutations failed"):
            directory.flush_deferred(timeout=5.0)
        assert calls == [
            [
                {"content_hash": b"a", "engine_id": "engine", "slot_ids": [1]},
                {"content_hash": b"b", "engine_id": "engine", "slot_ids": [2]},
            ],
            [{"content_hash": b"c", "engine_id": "engine", "slot_ids": [3]}],
        ]
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


def test_publish_worker_batches_adjacent_mutations_in_order():
    directory = ContentDirectory(
        "/tmp/gms-directory-writer-batching.sock",
        engine="test",
        block_size=16,
        mode="shadow",
    )
    calls = []
    directory.publish = lambda items: calls.append(items) or len(items)  # type: ignore[method-assign]
    with directory._mutation_condition:
        directory._mutation_sequence = 2
        directory._mutations.extend(
            [
                (
                    1,
                    "publish",
                    [{"content_hash": b"a", "engine_id": "engine", "slot_ids": [1]}],
                ),
                (
                    2,
                    "publish",
                    [{"content_hash": b"b", "engine_id": "engine", "slot_ids": [2]}],
                ),
            ]
        )
        directory._mutation_stop = True

    directory._mutation_loop()

    assert calls == [
        [
            {"content_hash": b"a", "engine_id": "engine", "slot_ids": [1]},
            {"content_hash": b"b", "engine_id": "engine", "slot_ids": [2]},
        ]
    ]
    assert directory._mutation_committed == 2


@pytest.mark.parametrize("collision", ["hash", "slot", "legacy-slot"])
def test_adjacent_publications_preserve_sequential_daemon_semantics(
    tmp_path, collision
):
    from gms_kv_ring.daemon.directory_server import DirectoryState
    from gms_kv_ring.daemon.rpc_directory import handle_directory_publish_batch

    daemon = DirectoryState()
    daemon._content_directory_writer_id = "writer"
    directory = ContentDirectory(
        str(tmp_path / "unused.sock"), engine="test", block_size=16, mode="shadow"
    )
    first = {
        "content_hash": b"a" * 32,
        "engine_id": "e",
        "slot_ids": [1],
        "generations": [1],
        "tier": "hbm",
    }
    second = dict(first, content_hash=b"b" * 32, generations=[2])
    if collision == "hash":
        second.update(content_hash=first["content_hash"], slot_ids=[2])

    if collision == "legacy-slot":
        for item in (first, second):
            item["slot_id"] = item.pop("slot_ids")[0]

    def publish(items):
        response = handle_directory_publish_batch(
            daemon,
            {
                "writer_id": "writer",
                "expected_epoch": 1,
                "manifest_id": "manifest",
                "items": [
                    dict(item, content_hash=item["content_hash"].hex())
                    for item in items
                ],
            },
        )
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["published"]

    directory.publish = publish
    directory._mutations.extend([(1, "publish", [first]), (2, "publish", [second])])
    directory._mutation_sequence = 2
    directory._mutation_stop = True
    directory._mutation_loop()
    assert directory._mutation_failed == 0
    recovered = daemon._content_directory[("manifest", second["content_hash"])]
    assert recovered["slot_ids"] == second.get("slot_ids", [second.get("slot_id")])
    assert recovered["generations"] == [2]


def test_deferred_publish_owns_nested_metadata(tmp_path, monkeypatch):
    directory = ContentDirectory(
        str(tmp_path / "unused.sock"), engine="test", block_size=16, mode="shadow"
    )
    directory._async_publish = True
    directory._pipeline_enabled = False
    monkeypatch.setattr(directory, "_start_mutation_worker_locked", lambda: None)
    item = {
        "content_hash": b"h",
        "engine_id": "e",
        "slot_ids": [1],
        "generations": [2],
        "ranges": [[0, 0, 16]],
    }
    assert directory.publish_deferred([item]) == 1
    item["slot_ids"][0] = 9
    item["generations"][0] = 10
    item["ranges"][0][2] = 32
    queued = directory._mutations[0][2][0]
    assert queued["slot_ids"] == [1]
    assert queued["generations"] == [2]
    assert queued["ranges"] == [[0, 0, 16]]


def test_pipelined_publish_copies_before_return_and_validates_ack(tmp_path):
    directory = ContentDirectory(
        str(tmp_path / "unused.sock"),
        engine="test",
        block_size=16,
        mode="shadow",
    )
    sent = []
    responses = []
    ready = threading.Condition()

    class Client:
        def send_request(self, message):
            with ready:
                sent.append(message)
                responses.append(
                    {
                        "ok": True,
                        "published": len(message["items"]),
                        "rejected_stale_writer": False,
                    }
                )
                ready.notify_all()

        def receive_response(self):
            with ready:
                while not responses:
                    ready.wait()
                return responses.pop(0)

        def close(self):
            return None

    directory._pipeline_client = Client()
    directory._view_epoch = 4
    directory._view_current_writer = True
    item = {
        "content_hash": b"h",
        "engine_id": "e",
        "slot_ids": [1],
        "generations": [2],
        "tier": "hbm",
    }
    try:
        assert directory.publish_deferred([item]) == 1
        item["slot_ids"][0] = 9
        assert directory.flush_deferred(timeout=1.0)
        assert sent[0]["items"][0]["slot_ids"] == [1]
        assert directory._pipeline_committed == 1
        assert directory._pipeline_pending_items == 0
    finally:
        directory.close()


def test_pipelined_publish_never_waits_for_socket_send(tmp_path):
    directory = ContentDirectory(
        str(tmp_path / "unused.sock"), engine="test", block_size=16, mode="shadow"
    )
    release_send = threading.Event()
    entered_send = threading.Event()

    class Client:
        def send_request(self, _message):
            entered_send.set()
            assert release_send.wait(2.0)

        def receive_response(self):
            return {"ok": True, "published": 1, "rejected_stale_writer": False}

        def close(self):
            release_send.set()

    directory._pipeline_client = Client()
    directory._view_epoch = 4
    directory._view_current_writer = True
    item = {
        "content_hash": b"h",
        "engine_id": "e",
        "slot_ids": [1],
        "generations": [2],
        "tier": "hbm",
    }
    try:
        started = __import__("time").monotonic()
        assert directory.publish_deferred([item]) == 1
        assert __import__("time").monotonic() - started < 0.05
        assert entered_send.wait(1.0)
        release_send.set()
        assert directory.flush_deferred(timeout=1.0)
    finally:
        release_send.set()
        directory.close()


def test_pipelined_publish_rejects_queue_overflow_without_blocking(tmp_path):
    directory = ContentDirectory(
        str(tmp_path / "unused.sock"), engine="test", block_size=16, mode="shadow"
    )
    directory._max_pending_items = 1
    directory._view_epoch = 4
    directory._view_current_writer = True
    directory._pipeline_expected.append((1, {}, 1))
    directory._pipeline_pending_items = 1
    item = {
        "content_hash": b"h",
        "engine_id": "e",
        "slot_ids": [1],
        "generations": [2],
        "tier": "hbm",
    }
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        directory._pipeline_publish([item])


def test_pipelined_mixed_publish_uses_accepted_count(tmp_path):
    directory = ContentDirectory(
        str(tmp_path / "unused.sock"),
        engine="test",
        block_size=16,
        mode="shadow",
    )
    responses = []
    ready = threading.Condition()

    class Client:
        def send_request(self, message):
            with ready:
                responses.append(
                    {
                        "ok": True,
                        "accepted": len(message["items"]),
                        "published": 1,
                        "removed": 1,
                        "rejected_stale_writer": False,
                    }
                )
                ready.notify_all()

        def receive_response(self):
            with ready:
                while not responses:
                    ready.wait()
                return responses.pop(0)

        def close(self):
            return None

    directory._pipeline_client = Client()
    directory._view_epoch = 4
    directory._view_current_writer = True
    items = [
        {
            "content_hash": b"new",
            "engine_id": "e",
            "slot_ids": [2],
            "generations": [3],
            "tier": "hbm",
        },
        {
            "content_hash": b"old",
            "engine_id": "e",
            "slot_ids": [1],
            "generations": [2],
            "tier": "hbm",
            "sealed": False,
        },
    ]
    try:
        assert directory.publish_deferred(items) == 2
        assert directory.flush_deferred(timeout=1.0)
        assert directory._pipeline_committed == 1
    finally:
        directory.close()


def test_pipelined_publish_fails_closed_after_stale_writer_ack(tmp_path):
    directory = ContentDirectory(
        str(tmp_path / "unused.sock"),
        engine="test",
        block_size=16,
        mode="shadow",
    )
    response_ready = threading.Event()

    class Client:
        def send_request(self, _message):
            response_ready.set()

        def receive_response(self):
            assert response_ready.wait(1.0)
            return {
                "ok": True,
                "published": 0,
                "rejected_stale_writer": True,
            }

        def close(self):
            return None

    directory._pipeline_client = Client()
    directory._view_epoch = 4
    directory._view_current_writer = True
    item = {
        "content_hash": b"h",
        "engine_id": "e",
        "slot_ids": [1],
        "generations": [2],
        "tier": "hbm",
    }
    try:
        assert directory.publish_deferred([item]) == 1
        with pytest.raises(RuntimeError, match="pipeline failed"):
            directory.flush_deferred(timeout=1.0)
        assert directory._pipeline_pending_items == 0
        with pytest.raises(RuntimeError, match="pipeline failed"):
            directory.publish_deferred([item])
    finally:
        directory.close()


def test_pipelined_publish_fails_closed_after_send_error(tmp_path):
    directory = ContentDirectory(
        str(tmp_path / "unused.sock"),
        engine="test",
        block_size=16,
        mode="shadow",
    )

    class Client:
        closed = False

        def send_request(self, _message):
            raise BrokenPipeError("partial write")

        def receive_response(self):
            raise AssertionError("reader must not run after send failure")

        def close(self):
            self.closed = True

    client = Client()
    directory._pipeline_client = client
    directory._view_epoch = 4
    directory._view_current_writer = True
    item = {
        "content_hash": b"h",
        "engine_id": "e",
        "slot_ids": [1],
        "generations": [2],
        "tier": "hbm",
    }
    try:
        assert directory.publish_deferred([item]) == 1
        with pytest.raises(RuntimeError, match="pipeline failed") as failure:
            directory.flush_deferred(timeout=1.0)
        assert isinstance(failure.value.__cause__, BrokenPipeError)
        assert client.closed
        assert directory._pipeline_client is None
        assert directory._pipeline_stop
        with pytest.raises(RuntimeError, match="pipeline failed"):
            directory.publish_deferred([item])
    finally:
        directory.close()


def test_malformed_publication_does_not_stop_writer(tmp_path):
    directory = ContentDirectory(
        str(tmp_path / "unused.sock"), engine="test", block_size=16, mode="shadow"
    )
    valid = {"content_hash": b"h", "engine_id": "e", "slot_ids": [1]}
    calls = []

    def publish(items):
        calls.append(items)
        if items == [{}]:
            raise ValueError("malformed item")
        return len(items)

    directory.publish = publish
    directory._mutations.extend([(1, "publish", [{}]), (2, "publish", [valid])])
    directory._mutation_sequence = 2
    directory._mutation_stop = True
    directory._mutation_loop()
    assert calls == [[{}], [valid]]
    assert directory._mutation_committed == 2
    assert directory._mutation_failed == 1


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


def test_hbm_candidate_probe_is_conservative_until_view_is_ready():
    content_hash = b"h" * 32
    directory = ContentDirectory(
        "/tmp/gms-directory-candidate-probe.sock",
        engine="test",
        block_size=16,
        mode="shadow",
    )

    assert directory.may_have_hbm_candidate([content_hash]) is True
    directory._view_ready.set()
    assert directory.may_have_hbm_candidate([content_hash]) is True
    directory._view_caught_up = True
    assert directory.may_have_hbm_candidate([content_hash]) is False


@pytest.mark.parametrize("state", ["ready", "active"])
def test_hbm_candidate_probe_includes_claimable_states(state):
    content_hash = b"h" * 32
    directory = ContentDirectory(
        "/tmp/gms-directory-candidate-state.sock",
        engine="test",
        block_size=16,
        mode="shadow",
    )
    directory._view = {content_hash: {"tier": "hbm", "state": state}}
    directory._view_caught_up = True
    directory._view_ready.set()

    assert directory.may_have_hbm_candidate([content_hash]) is True


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

    reader = handle_directory_lookup_claim(
        daemon,
        {
            "manifest_id": "manifest",
            "reader_only": True,
            "hashes": [hashes[0].hex()],
            SERVER_CONNECTION_ID: "reader",
        },
    )
    assert reader["claim_token"] is not None
    assert release_directory_connection_claims(daemon, "reader") == 0
    assert entries[("manifest", hashes[0])]["_claim_count"] == 1

    promoted = handle_directory_promote(
        daemon,
        {"writer_id": "successor", "expected_epoch": 4},
    )
    assert promoted["promoted"] is True
    assert entries[("manifest", hashes[0])]["_claim_count"] == 0
    assert entries[("manifest", hashes[1])]["_claim_count"] == 0
