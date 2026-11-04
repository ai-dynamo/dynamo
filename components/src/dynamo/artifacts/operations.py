# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable receipts and bounded ownership for a single delivery authority."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import secrets
import sqlite3
import threading
import time
import traceback
from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

from dynamo.artifacts.storage import ArtifactStorageError

logger = logging.getLogger(__name__)


async def _settle(task: asyncio.Future):
    """Keep an owned operation alive until it settles, recording cancellation."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    return task.result(), cancelled


class ArtifactOperationStore:
    """SQLite authority with bounded receipts and fenced terminal writes.

    One worker owns a database. Cross-worker recovery requires an external lease
    authority; callers must not run recovery against a live worker's database.
    """

    def __init__(
        self,
        path: Path,
        capability_key: bytes | None = None,
        *,
        terminal_ttl_seconds=86400,
    ):
        if (
            isinstance(terminal_ttl_seconds, bool)
            or not isinstance(terminal_ttl_seconds, (int, float))
            or not math.isfinite(terminal_ttl_seconds)
            or terminal_ttl_seconds <= 0
        ):
            raise ValueError("invalid terminal receipt retention")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal_ttl_seconds = terminal_ttl_seconds
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS operations ("
                "id TEXT PRIMARY KEY, owner TEXT NOT NULL, result TEXT NOT NULL, "
                "state TEXT NOT NULL, created REAL NOT NULL, terminal_at REAL)"
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(operations)")}
            if "grant" in columns or "state" not in columns:
                conn.execute("PRAGMA secure_delete=ON")
                conn.execute("BEGIN IMMEDIATE")
                self._migrate_legacy(conn)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS operations_state_idx ON operations(state)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS operations_terminal_idx "
                "ON operations(terminal_at) WHERE terminal_at IS NOT NULL"
            )
        self.path.chmod(0o600)

    @staticmethod
    def _migrate_legacy(conn):
        conn.execute(
            "CREATE TABLE operations_migrated ("
            "id TEXT PRIMARY KEY, owner TEXT NOT NULL, result TEXT NOT NULL, "
            "state TEXT NOT NULL, created REAL NOT NULL, terminal_at REAL)"
        )
        for operation_id, owner, result, created in conn.execute(
            "SELECT id, owner, result, created FROM operations"
        ):
            state = json.loads(result)["state"]
            conn.execute(
                "INSERT INTO operations_migrated VALUES (?, ?, ?, ?, ?, ?)",
                (
                    operation_id,
                    owner,
                    result,
                    state,
                    created,
                    time.time() if state != "pending" else None,
                ),
            )
        conn.execute("DROP TABLE operations")
        conn.execute("ALTER TABLE operations_migrated RENAME TO operations")

    def _prune(self, conn):
        conn.execute(
            "DELETE FROM operations WHERE terminal_at IS NOT NULL AND terminal_at <= ?",
            (time.time() - self.terminal_ttl_seconds,),
        )

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=5)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create(self, owner: str, contents: list[str], target: Mapping[str, Any]):
        operation_id = "ga_" + secrets.token_urlsafe(24)
        result = {
            "kind": "generation_artifact",
            "mode": "object_store",
            "format": "generation_artifact_v1",
            "contents": list(contents),
            "operation_id": operation_id,
            "state": "pending",
            "revision": 1,
        }
        with self._connect() as conn:
            self._prune(conn)
            conn.execute(
                "INSERT INTO operations VALUES (?, ?, ?, ?, ?, NULL)",
                (operation_id, owner, json.dumps(result), "pending", time.time()),
            )
        return result

    def get(self, operation_id: str, owner: str):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result FROM operations WHERE id=? AND owner=? "
                "AND (terminal_at IS NULL OR terminal_at>?)",
                (operation_id, owner, time.time() - self.terminal_ttl_seconds),
            ).fetchone()
        if row is None:
            raise KeyError("operation not found")
        return json.loads(row[0])

    def finish(self, operation_id: str, state: str, **fields):
        if state not in {"ready", "failed"}:
            raise ValueError("invalid terminal state")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT result FROM operations WHERE id=?", (operation_id,)
            ).fetchone()
            if row is None:
                raise KeyError("operation not found")
            previous = json.loads(row[0])
            if previous["state"] != "pending":
                raise ValueError("operation is already terminal")
            allowed = (
                {"actual_bytes", "sha256", "object_id", "provider_identity"}
                if state == "ready"
                else {"error_code"}
            )
            if fields.keys() - allowed:
                raise ValueError("invalid terminal fields")
            if (
                state == "ready"
                and not {"actual_bytes", "sha256", "object_id"} <= fields.keys()
            ):
                raise ValueError(
                    "ready operation requires verified identity and digest"
                )
            result = {
                **previous,
                "state": state,
                "revision": previous["revision"] + 1,
                **fields,
            }
            conn.execute(
                "UPDATE operations SET result=?, state=?, terminal_at=? WHERE id=?",
                (json.dumps(result), state, time.time(), operation_id),
            )
        return result

    def recover(self):
        """Fail pending records after this authority has lost all volatile sources."""
        with self._connect() as conn:
            self._prune(conn)
            pending = [
                row[0]
                for row in conn.execute(
                    "SELECT id FROM operations WHERE state='pending'"
                )
            ]
        for key in pending:
            self.finish(key, "failed", error_code="source_lost")
        return len(pending)


Transfer = Callable[[bytes, Mapping[str, Any]], Awaitable[Mapping[str, Any]]]


class ArtifactService:
    """Own bounded sources independently of the generation response iterator."""

    def __init__(
        self,
        store: ArtifactOperationStore,
        transfer: Transfer,
        verify: Callable | None,
        *,
        max_source_bytes: int = 256 << 20,
        concurrency: int = 2,
        timeout_seconds: float = 60,
        max_pending_operations: int = 128,
    ):
        if (
            max_source_bytes <= 0
            or concurrency <= 0
            or timeout_seconds <= 0
            or type(max_pending_operations) is not int
            or max_pending_operations <= 0
        ):
            raise ValueError("invalid delivery limits")
        self.store = store
        self.transfer = transfer
        self.verify = verify
        self.max_source_bytes = max_source_bytes
        self.timeout_seconds = timeout_seconds
        self.max_pending_operations = max_pending_operations
        self._lock = threading.Lock()
        self._source_bytes = 0
        self._pending_count = 0
        self._semaphore = asyncio.Semaphore(concurrency)
        self._tasks: set[asyncio.Future] = set()
        self._admissions: set[asyncio.Future] = set()

    @property
    def source_bytes(self):
        with self._lock:
            return self._source_bytes

    async def submit(
        self, data: bytes, owner: str, contents: list[str], target: Mapping[str, Any]
    ):
        loop = asyncio.get_running_loop()
        if type(data) is not bytes:
            raise ValueError("artifact source must be immutable bytes")
        sealed_target = json.loads(json.dumps(dict(target)))
        sealed_contents = list(contents)
        with self._lock:
            if (
                self._source_bytes + len(data) > self.max_source_bytes
                or self._pending_count >= self.max_pending_operations
            ):
                raise ValueError("artifact delivery capacity exhausted")
            self._source_bytes += len(data)
            self._pending_count += 1
        admission = loop.create_future()
        self._admissions.add(admission)
        try:
            creating = loop.run_in_executor(
                None, partial(self.store.create, owner, sealed_contents, sealed_target)
            )
            pending, cancelled = await _settle(creating)
            if cancelled:
                await self._finish(
                    pending["operation_id"],
                    "failed",
                    error_code="source_lost",
                )
                raise asyncio.CancelledError
            started = asyncio.Event()
            task = loop.create_task(
                self._deliver(data, sealed_target, pending["operation_id"], started)
            )
            self._tasks.add(task)
            task.add_done_callback(
                partial(
                    self._delivery_done, pending["operation_id"], len(data), started
                )
            )
            return pending
        except BaseException:
            self._release_source(len(data))
            raise
        finally:
            self._admissions.discard(admission)
            admission.set_result(None)

    async def _finish(self, operation_id, state, **fields):
        writing = asyncio.get_running_loop().run_in_executor(
            None, partial(self.store.finish, operation_id, state, **fields)
        )
        # A terminal commit, once begun, wins caller cancellation.
        result, _ = await _settle(writing)
        return result

    def _delivery_done(self, operation_id, source_size, started, task):
        self._tasks.discard(task)
        if not started.is_set():
            # Cancellation before a coroutine's first step skips its finally.
            writing = asyncio.get_running_loop().run_in_executor(
                None,
                partial(
                    self.store.finish, operation_id, "failed", error_code="source_lost"
                ),
            )
            self._tasks.add(writing)
            writing.add_done_callback(partial(self._abandoned_done, source_size))
        self._observe(task)

    def _abandoned_done(self, source_size, writing):
        self._tasks.discard(writing)
        self._release_source(source_size)
        self._observe(writing)

    def _release_source(self, source_size):
        with self._lock:
            self._source_bytes -= source_size
            self._pending_count -= 1

    @staticmethod
    def _observe(task):
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error("Artifact delivery task failed: %s", type(error).__name__)
                seen = set()
                while error is not None and id(error) not in seen:
                    seen.add(id(error))
                    # Completed traceback frames must not retain uncharged sources.
                    traceback.clear_frames(error.__traceback__)
                    error = error.__cause__ or error.__context__

    async def _deliver(self, data, target, operation_id, started):
        started.set()
        try:
            fields = await self._bounded_transfer(data, target)
            # Keep the fenced commit owned by this task through cancellation.
            await self._finish(operation_id, "ready", **fields)
        except asyncio.CancelledError:
            await self._finish(operation_id, "failed", error_code="source_lost")
            raise
        except Exception as exc:
            code = (
                "commit_unknown"
                if isinstance(exc, ArtifactStorageError) and exc.outcome == "ambiguous"
                else "verification_failed"
                if isinstance(exc, ValueError)
                else "delivery_failed"
            )
            await self._finish(operation_id, "failed", error_code=code)
            if not isinstance(
                exc,
                (
                    ArtifactStorageError,
                    ValueError,
                    RuntimeError,
                    TypeError,
                    OSError,
                    asyncio.TimeoutError,
                ),
            ):
                raise
        finally:
            self._release_source(len(data))

    async def _bounded_transfer(self, data, target):
        child = asyncio.create_task(self._transfer_and_verify(data, target))
        try:
            return await asyncio.shield(child)
        except BaseException:
            child.cancel()
            # Repeated caller cancellation cannot detach transfer cleanup.
            await _settle(asyncio.gather(child, return_exceptions=True))
            raise

    async def _transfer_and_verify(self, data, target):
        async with self._semaphore:
            ambiguous = False
            try:
                receipt = await asyncio.wait_for(
                    self.transfer(data, target), self.timeout_seconds
                )
            except ArtifactStorageError as exc:
                if exc.outcome != "ambiguous":
                    raise
                # Read back once; never retry an uncertain PUT.
                ambiguous = True
                receipt = {}
            except asyncio.TimeoutError:
                ambiguous = True
                receipt = {}
            if self.verify is None:
                if ambiguous:
                    raise ArtifactStorageError(
                        "commit_unknown",
                        "write outcome cannot be verified",
                        outcome="ambiguous",
                    )
                raise ValueError("verification unavailable")
            try:
                verified = await asyncio.wait_for(
                    self.verify(data, target, receipt), self.timeout_seconds
                )
                if (
                    verified.get("sha256") != hashlib.sha256(data).hexdigest()
                    or verified.get("actual_bytes") != len(data)
                    or not verified.get("object_id")
                ):
                    raise ValueError("verification failed")
            except Exception:
                if ambiguous:
                    raise ArtifactStorageError(
                        "commit_unknown",
                        "write outcome cannot be verified",
                        outcome="ambiguous",
                    ) from None
                raise
            return {
                key: verified[key]
                for key in ("actual_bytes", "sha256", "object_id", "provider_identity")
                if key in verified
            }

    async def drain(self):
        cancelled = False
        while self._tasks or self._admissions:
            pending = asyncio.gather(
                *tuple(self._admissions), *tuple(self._tasks), return_exceptions=True
            )
            _, interrupted = await _settle(pending)
            cancelled = cancelled or interrupted
            # Done callbacks may hand a never-started task to a SQLite writer.
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError
