# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import hashlib
import secrets
import sqlite3
import threading
import time

import pytest

pytest.importorskip("opendal")

from dynamo.artifacts.operations import (  # noqa: E402
    ArtifactOperationStore,
    ArtifactService,
)
from dynamo.artifacts.storage import ArtifactStorageError  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.timeout(15),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("times_out", [False, True])
async def test_delivery_without_python311_timeout_api(tmp_path, monkeypatch, times_out):
    monkeypatch.delattr(asyncio, "timeout", raising=False)
    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))

    async def transfer(data, target):
        if times_out:
            await asyncio.Event().wait()
        return {"actual_bytes": len(data), "object_id": "bound-object"}

    async def verify(data, target, receipt):
        return {**receipt, "sha256": hashlib.sha256(data).hexdigest()}

    service = ArtifactService(
        store, transfer, verify, timeout_seconds=0.01 if times_out else 60
    )
    pending = await service.submit(b"owned", "owner", [], {"url": "SENTINEL"})
    await service.drain()
    result = store.get(pending["operation_id"], "owner")
    assert result["state"] == ("failed" if times_out else "ready")
    if times_out:
        assert result["error_code"] == "commit_unknown"
    assert service.source_bytes == 0
    assert service._pending_count == 0


@pytest.mark.asyncio
async def test_pending_operation_count_is_bounded(tmp_path):
    release = asyncio.Event()

    async def transfer(data, target):
        await release.wait()
        raise RuntimeError("stop")

    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))
    service = ArtifactService(store, transfer, None, max_pending_operations=1)
    await service.submit(b"x", "owner", [], {})
    with pytest.raises(ValueError, match="capacity"):
        await service.submit(b"y", "owner", [], {})
    release.set()
    await service.drain()
    await service.submit(b"z", "owner", [], {})
    await service.drain()
    assert service.source_bytes == 0


@pytest.mark.asyncio
async def test_queue_wait_does_not_consume_transfer_timeout(tmp_path):
    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def transfer(data, target):
        if data == b"first":
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
                raise
        return {"actual_bytes": len(data), "object_id": data.hex()}

    async def verify(data, target, receipt):
        return {**receipt, "sha256": hashlib.sha256(data).hexdigest()}

    service = ArtifactService(
        store, transfer, verify, concurrency=1, timeout_seconds=0.1
    )
    first = await service.submit(b"first", "owner", [], {})
    await entered.wait()
    second = await service.submit(b"second", "owner", [], {})
    await asyncio.sleep(0.15)
    release.set()
    await service.drain()
    assert store.get(second["operation_id"], "owner")["state"] == "ready"
    assert store.get(first["operation_id"], "owner")["state"] == "failed"


@pytest.mark.asyncio
async def test_ambiguous_timeout_gets_fresh_readback_budget(tmp_path):
    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))
    attempts = []

    async def transfer(data, target):
        attempts.append(data)
        await asyncio.Event().wait()

    async def verify(data, target, receipt):
        await asyncio.sleep(0.03)
        return {
            "actual_bytes": len(data),
            "object_id": "committed-object",
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    service = ArtifactService(store, transfer, verify, timeout_seconds=0.3)
    pending = await service.submit(b"opaque", "owner", [], {})
    await service.drain()
    assert len(attempts) == 1
    assert store.get(pending["operation_id"], "owner")["state"] == "ready"


@pytest.mark.asyncio
async def test_unresolved_ambiguous_write_records_commit_unknown(tmp_path):
    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))

    async def transfer(data, target):
        raise ArtifactStorageError("commit_unknown", "ack lost", outcome="ambiguous")

    async def verify(data, target, receipt):
        await asyncio.Event().wait()

    service = ArtifactService(store, transfer, verify, timeout_seconds=0.01)
    pending = await service.submit(b"opaque", "owner", [], {})
    await service.drain()
    result = store.get(pending["operation_id"], "owner")
    assert result["state"] == "failed"
    assert result["error_code"] == "commit_unknown"


@pytest.mark.asyncio
async def test_unlisted_transfer_exception_records_terminal_failure(tmp_path):
    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))

    async def transfer(data, target):
        raise KeyError("provider failure")

    service = ArtifactService(store, transfer, None)
    pending = await service.submit(b"opaque", "owner", [], {})
    await service.drain()
    result = store.get(pending["operation_id"], "owner")
    assert result["state"] == "failed"
    assert result["error_code"] == "delivery_failed"
    assert service.source_bytes == 0


@pytest.mark.asyncio
async def test_unlisted_verification_exception_preserves_ambiguous_outcome(tmp_path):
    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))

    async def transfer(data, target):
        raise ArtifactStorageError("commit_unknown", "ack lost", outcome="ambiguous")

    async def verify(data, target, receipt):
        raise KeyError("provider readback failed")

    service = ArtifactService(store, transfer, verify)
    pending = await service.submit(b"opaque", "owner", [], {})
    await service.drain()
    result = store.get(pending["operation_id"], "owner")
    assert result["state"] == "failed"
    assert result["error_code"] == "commit_unknown"


@pytest.fixture
def blocking_writer():
    releases = []

    def block(method):
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        releases.append(release)

        def writer(*args, **kwargs):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(5):
                raise TimeoutError("test writer was not released")
            return method(*args, **kwargs)

        return writer, entered, release

    yield block
    for release in releases:
        release.set()


def test_state_is_durable_without_persisting_grants(tmp_path):
    path = tmp_path / "operations.db"
    key = secrets.token_bytes(32)
    store = ArtifactOperationStore(path, key)
    pending = store.create("owner", ["moe_routes"], {"url": "SENTINEL_SECRET"})
    assert pending["state"] == "pending"
    assert "SENTINEL_SECRET" not in repr(pending)
    assert b"SENTINEL_SECRET" not in path.read_bytes()
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(operations)")}
        assert "grant" not in columns
    with pytest.raises(KeyError):
        store.get(pending["operation_id"], "other-owner")
    store.finish(pending["operation_id"], "failed", error_code="source_lost")
    reopened = ArtifactOperationStore(path, key)
    assert reopened.get(pending["operation_id"], "owner")["state"] == "failed"
    with pytest.raises(ValueError):
        reopened.finish(pending["operation_id"], "ready", actual_bytes=1)


def test_legacy_grants_are_removed_on_migration(tmp_path):
    path = tmp_path / "operations.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE operations (id TEXT PRIMARY KEY, owner TEXT NOT NULL, "
            "result TEXT NOT NULL, grant BLOB, created REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO operations VALUES (?, ?, ?, ?, ?)",
            (
                "legacy",
                "owner",
                '{"state":"pending","revision":1}',
                b"SENTINEL_SECRET",
                time.time(),
            ),
        )
    store = ArtifactOperationStore(path, secrets.token_bytes(32))
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(operations)")}
        assert "grant" not in columns
    assert b"SENTINEL_SECRET" not in path.read_bytes()
    assert store.recover() == 1
    assert store.get("legacy", "owner")["error_code"] == "source_lost"


@pytest.mark.asyncio
async def test_pending_does_not_wait_for_transfer(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def transfer(data, target):
        entered.set()
        await release.wait()
        return {"object_id": "immutable-object", "actual_bytes": len(data)}

    async def verify(data, target, receipt):
        return {**receipt, "sha256": hashlib.sha256(data).hexdigest()}

    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))
    service = ArtifactService(store, transfer, verify)
    result = await service.submit(b"opaque", "owner", [], {"kind": "managed_store"})
    assert result["state"] == "pending"
    await entered.wait()
    assert store.get(result["operation_id"], "owner")["state"] == "pending"
    release.set()
    await service.drain()
    ready = store.get(result["operation_id"], "owner")
    assert ready["state"] == "ready"
    assert ready["sha256"] == hashlib.sha256(b"opaque").hexdigest()


@pytest.mark.asyncio
async def test_failed_verification_never_becomes_ready(tmp_path):
    async def transfer(data, target):
        return {"actual_bytes": len(data), "object_id": "object"}

    async def verify(data, target, receipt):
        return {**receipt, "sha256": "incorrect"}

    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))
    service = ArtifactService(store, transfer, verify)
    pending = await service.submit(b"opaque", "owner", [], {})
    await service.drain()
    failed = store.get(pending["operation_id"], "owner")
    assert failed["state"] == "failed"
    assert failed["error_code"] == "verification_failed"
    assert "sha256" not in failed


def test_recovery_fails_lost_ephemeral_sources(tmp_path):
    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))
    pending = store.create("owner", [], {})
    assert store.recover() == 1
    assert store.get(pending["operation_id"], "owner")["error_code"] == "source_lost"


def test_recovery_queries_indexed_pending_state(tmp_path):
    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))
    pending = store.create("owner", [], {})
    terminal = store.create("owner", [], {})
    store.finish(terminal["operation_id"], "failed", error_code="source_lost")
    with store._connect() as conn:
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(operations)")}
        assert "operations_state_idx" in indexes
    assert store.recover() == 1
    assert store.get(pending["operation_id"], "owner")["state"] == "failed"


def test_terminal_receipts_expire_without_deleting_pending_rows(tmp_path, monkeypatch):
    clock = [time.time()]
    monkeypatch.setattr("dynamo.artifacts.operations.time.time", lambda: clock[0])
    store = ArtifactOperationStore(
        tmp_path / "ops.db", secrets.token_bytes(32), terminal_ttl_seconds=10
    )
    expired = store.create("owner", [], {})
    pending = store.create("owner", [], {})
    store.finish(expired["operation_id"], "failed", error_code="source_lost")
    clock[0] += 11
    with pytest.raises(KeyError):
        store.get(expired["operation_id"], "owner")
    store.create("owner", [], {})
    with store._connect() as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM operations WHERE id=?",
                (expired["operation_id"],),
            ).fetchone()[0]
            == 0
        )
    assert store.get(pending["operation_id"], "owner")["state"] == "pending"


@pytest.mark.asyncio
async def test_queue_budget_releases_after_failure(tmp_path):
    release = asyncio.Event()

    async def transfer(data, target):
        await release.wait()
        raise RuntimeError("SENTINEL_SECRET")

    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))
    service = ArtifactService(store, transfer, None, max_source_bytes=6)
    first = await service.submit(b"opaque", "owner", [], {})
    with pytest.raises(ValueError, match="capacity"):
        await service.submit(b"x", "owner", [], {})
    release.set()
    await service.drain()
    failed = store.get(first["operation_id"], "owner")
    assert "SENTINEL_SECRET" not in repr(failed)
    assert service.source_bytes == 0


@pytest.mark.asyncio
async def test_ambiguous_commit_is_verified_without_a_second_put(tmp_path):
    attempts = []

    async def transfer(data, target):
        attempts.append(data)
        raise ArtifactStorageError("commit_unknown", "ack lost", outcome="ambiguous")

    async def verify(data, target, receipt):
        return {
            "actual_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "object_id": "bound-object",
        }

    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))
    service = ArtifactService(store, transfer, verify)
    pending = await service.submit(b"captured", "owner", [], {"url": "SENTINEL"})
    await service.drain()
    assert store.get(pending["operation_id"], "owner")["state"] == "ready"
    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_caller_cannot_mutate_a_handed_off_target(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def transfer(data, target):
        entered.set()
        await release.wait()
        assert target["required_headers"]["signed"] == "original"
        raise RuntimeError("stop")

    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))
    service = ArtifactService(store, transfer, None)
    target = {"required_headers": {"signed": "original"}}
    await service.submit(b"bytes", "owner", [], target)
    await entered.wait()
    target["required_headers"]["signed"] = "changed"
    release.set()
    await service.drain()
    assert service.source_bytes == 0


@pytest.mark.asyncio
async def test_repeated_create_cancellation_keeps_source_until_failed_commit(
    tmp_path, monkeypatch, blocking_writer
):
    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))
    created = []
    original_create = store.create

    def create(*args, **kwargs):
        result = original_create(*args, **kwargs)
        created.append(result["operation_id"])
        return result

    writer, entered, release = blocking_writer(create)
    monkeypatch.setattr(store, "create", writer)

    async def transfer(data, target):
        pytest.fail("cancelled submission must not start transfer")

    service = ArtifactService(store, transfer, None)
    submission = asyncio.create_task(service.submit(b"owned", "owner", [], {}))
    await entered.wait()
    submission.cancel()
    await asyncio.sleep(0)
    submission.cancel()
    await asyncio.sleep(0)
    assert not submission.done()
    assert service.source_bytes == len(b"owned")
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await submission
    assert service.source_bytes == 0
    assert store.get(created[0], "owner")["state"] == "failed"


@pytest.mark.asyncio
async def test_cancelled_ready_writer_commits_ready_once(
    tmp_path, monkeypatch, blocking_writer
):
    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))
    writes = []
    original_finish = store.finish

    def finish(operation_id, state, **fields):
        writes.append(state)
        return original_finish(operation_id, state, **fields)

    writer, entered, release = blocking_writer(finish)
    monkeypatch.setattr(store, "finish", writer)

    async def transfer(data, target):
        return {"actual_bytes": len(data), "object_id": "bound-object"}

    async def verify(data, target, receipt):
        return {**receipt, "sha256": hashlib.sha256(data).hexdigest()}

    service = ArtifactService(store, transfer, verify)
    pending = await service.submit(b"owned", "owner", [], {})
    await entered.wait()
    delivery = next(iter(service._tasks))
    delivery.cancel()
    await asyncio.sleep(0)
    delivery.cancel()
    await asyncio.sleep(0)
    assert not delivery.done()
    assert service.source_bytes == len(b"owned")
    assert store.get(pending["operation_id"], "owner")["state"] == "pending"
    release.set()
    await delivery
    ready = store.get(pending["operation_id"], "owner")
    assert ready["state"] == "ready"
    assert ready["revision"] == 2
    assert writes == ["ready"]
    assert service.source_bytes == 0


@pytest.mark.asyncio
async def test_repeated_delivery_cancellation_settles_failed_writer(
    tmp_path, monkeypatch, blocking_writer
):
    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))
    writer, writer_entered, release = blocking_writer(store.finish)
    monkeypatch.setattr(store, "finish", writer)
    transfer_entered = asyncio.Event()

    async def transfer(data, target):
        transfer_entered.set()
        await asyncio.Event().wait()

    service = ArtifactService(store, transfer, None)
    pending = await service.submit(b"owned", "owner", [], {})
    await transfer_entered.wait()
    delivery = next(iter(service._tasks))
    delivery.cancel()
    await writer_entered.wait()
    delivery.cancel()
    await asyncio.sleep(0)
    assert not delivery.done()
    assert service.source_bytes == len(b"owned")
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await delivery
    assert store.get(pending["operation_id"], "owner")["state"] == "failed"
    assert service.source_bytes == 0


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_transfer_cleanup_before_release(tmp_path):
    entered = asyncio.Event()
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))

    async def transfer(data, target):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_entered.set()
            await release_cleanup.wait()

    service = ArtifactService(store, transfer, None)
    pending = await service.submit(b"owned", "owner", [], {})
    await entered.wait()
    delivery = next(iter(service._tasks))
    delivery.cancel()
    await cleanup_entered.wait()
    delivery.cancel()
    draining = asyncio.create_task(service.drain())
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(delivery), timeout=0.05)
        assert not draining.done()
        assert service.source_bytes == len(b"owned")
        assert store.get(pending["operation_id"], "owner")["state"] == "pending"
    finally:
        release_cleanup.set()
        await draining
    assert store.get(pending["operation_id"], "owner")["state"] == "failed"
    assert service.source_bytes == 0


@pytest.mark.asyncio
async def test_background_writer_exception_is_observed_without_secret_logs(
    tmp_path, monkeypatch, caplog
):
    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))
    incidents = []
    completed = asyncio.Event()
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "call_exception_handler", incidents.append)

    def unavailable(*args, **kwargs):
        loop.call_soon_threadsafe(completed.set)
        raise sqlite3.OperationalError("SENTINEL_SECRET")

    monkeypatch.setattr(store, "finish", unavailable)

    async def transfer(data, target):
        return {"actual_bytes": len(data), "object_id": "bound-object"}

    async def verify(data, target, receipt):
        return {**receipt, "sha256": hashlib.sha256(data).hexdigest()}

    service = ArtifactService(store, transfer, verify)
    await service.submit(b"owned", "owner", [], {})
    await completed.wait()
    while service._tasks:
        await asyncio.sleep(0)
    gc.collect()
    assert not incidents
    assert "OperationalError" in caplog.text
    assert "SENTINEL_SECRET" not in caplog.text
    assert service.source_bytes == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_drain", [False, True])
async def test_drain_owns_admission_through_delivery_handoff(
    tmp_path, monkeypatch, blocking_writer, cancel_drain
):
    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))
    writer, creation_entered, release_creation = blocking_writer(store.create)
    monkeypatch.setattr(store, "create", writer)
    transfer_entered = asyncio.Event()
    release_transfer = asyncio.Event()

    async def transfer(data, target):
        transfer_entered.set()
        await release_transfer.wait()
        return {"actual_bytes": len(data), "object_id": "bound-object"}

    async def verify(data, target, receipt):
        return {**receipt, "sha256": hashlib.sha256(data).hexdigest()}

    service = ArtifactService(store, transfer, verify)
    submission = asyncio.create_task(service.submit(b"owned", "owner", [], {}))
    await creation_entered.wait()
    draining = asyncio.create_task(service.drain())
    await asyncio.sleep(0)
    if cancel_drain:
        draining.cancel()
        await asyncio.sleep(0)
        draining.cancel()
        await asyncio.sleep(0)
    assert not draining.done()
    assert service.source_bytes == len(b"owned")
    release_creation.set()
    pending = await submission
    await transfer_entered.wait()
    assert not draining.done()
    assert service.source_bytes == len(b"owned")
    release_transfer.set()
    if cancel_drain:
        with pytest.raises(asyncio.CancelledError):
            await draining
    else:
        await draining
    assert store.get(pending["operation_id"], "owner")["state"] == "ready"
    assert service.source_bytes == 0


@pytest.mark.asyncio
async def test_mutable_sources_are_rejected_before_admission(tmp_path):
    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))

    async def transfer(data, target):
        pytest.fail("mutable source must not reach storage")

    service = ArtifactService(store, transfer, None)
    with pytest.raises(ValueError, match="immutable bytes"):
        await service.submit(bytearray(b"mutable"), "owner", [], {})
    assert service.source_bytes == 0


@pytest.mark.asyncio
async def test_cancel_before_delivery_starts_settles_failed_without_budget_leak(
    tmp_path, monkeypatch, blocking_writer
):
    store = ArtifactOperationStore(tmp_path / "ops.db", secrets.token_bytes(32))
    writer, entered, release = blocking_writer(store.finish)
    monkeypatch.setattr(store, "finish", writer)

    async def transfer(data, target):
        pytest.fail("cancelled queued delivery must not invoke transfer")

    service = ArtifactService(store, transfer, None)
    pending = await service.submit(b"owned", "owner", [], {})
    delivery = next(iter(service._tasks))
    delivery.cancel()
    draining = asyncio.create_task(service.drain())
    await entered.wait()
    assert not draining.done()
    assert service.source_bytes == len(b"owned")
    assert store.get(pending["operation_id"], "owner")["state"] == "pending"
    release.set()
    await draining
    assert store.get(pending["operation_id"], "owner")["state"] == "failed"
    assert service.source_bytes == 0
