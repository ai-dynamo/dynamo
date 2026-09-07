# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small engine-facing client for the GMS-owned KV content directory.

The directory is deliberately narrower than a cache controller: it records
only durable READY residencies and writer ownership. Inference engines keep
their native HBM scheduling and can run this adapter in comparison-only
(``shadow``) mode before making directory lookup authoritative.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Callable, Optional, TypeVar

from gms_kv_ring.daemon.client import DaemonClient

logger = logging.getLogger(__name__)

DIRECTORY_MODES = frozenset(("off", "shadow", "authoritative"))
_T = TypeVar("_T")


def resolve_directory_mode(value: Optional[str] = None) -> str:
    mode = (value or os.environ.get("GMS_KV_DIRECTORY_MODE", "off")).strip().lower()
    if mode not in DIRECTORY_MODES:
        logger.warning("unknown GMS_KV_DIRECTORY_MODE=%r; disabling", mode)
        return "off"
    return mode


def resolve_manifest_id(engine: str, block_size: int) -> str:
    """Return the explicit compatibility manifest or a safe POC fallback.

    Production deployments should set ``GMS_KV_DIRECTORY_MANIFEST`` from the
    model/layout metadata. The fallback intentionally includes engine, model,
    block size and cache salt so accidental incompatible reuse is unlikely.
    """
    explicit = os.environ.get("GMS_KV_DIRECTORY_MANIFEST", "").strip()
    if explicit:
        return explicit
    model = (
        os.environ.get("DYN_MODEL_PATH")
        or os.environ.get("DYN_MODEL")
        or os.environ.get("MODEL_PATH")
        or "unknown-model"
    )
    salt = os.environ.get("GMS_KVR_CROSS_NODE_SALT", "")
    return f"poc-v1|{engine}|{model}|block={int(block_size)}|salt={salt}"


def resolve_writer_id(engine_id: Optional[str] = None) -> str:
    raw = str(os.environ.get("ENGINE_ID") or engine_id or "0")
    return raw if raw.startswith("engine-") else f"engine-{raw}"


class ContentDirectory:
    """Thread-safe reconnecting facade over directory daemon RPCs."""

    def __init__(
        self,
        socket_path: Optional[str],
        *,
        engine: str,
        block_size: int,
        engine_id: Optional[str] = None,
        mode: Optional[str] = None,
        keyspace: Optional[str] = None,
        standby: Optional[bool] = None,
    ) -> None:
        self.mode = resolve_directory_mode(mode)
        self.engine = str(engine)
        self.socket_path = str(socket_path or "")
        self.manifest_id = resolve_manifest_id(engine, block_size)
        if keyspace:
            self.manifest_id = f"{self.manifest_id}|keyspace={str(keyspace).strip()}"
        self.writer_id = resolve_writer_id(engine_id)
        if standby is None:
            standby = os.environ.get("GMS_KV_DIRECTORY_STANDBY", "0").lower() not in (
                "0",
                "false",
                "no",
                "off",
                "",
            )
        self._standby = bool(standby)
        self._has_owned = False
        self._client: Optional[DaemonClient] = None
        self._writer_epoch: Optional[int] = None
        self._lock = threading.Lock()
        async_value = os.environ.get("GMS_KV_DIRECTORY_ASYNC_READ", "1")
        self._async_read = async_value.lower() not in (
            "0",
            "false",
            "no",
            "off",
            "",
        )
        try:
            poll_ms = float(os.environ.get("GMS_KV_DIRECTORY_POLL_MS", "250"))
        except ValueError:
            poll_ms = 250.0
        self._poll_seconds = max(0.001, poll_ms / 1_000)
        self._view_lock = threading.Lock()
        self._view: dict[bytes, dict] = {}
        self._view_revision = 0
        self._view_epoch: Optional[int] = None
        self._view_writer: Optional[str] = None
        self._view_ready = threading.Event()
        self._view_caught_up = False
        self._view_stop = threading.Event()
        self._view_thread: Optional[threading.Thread] = None
        # Publishing a newly completed block synchronously delays the first
        # token even though no request consumes the directory reply. Keep one
        # ordered writer per engine and make request finalization/close the
        # durability boundary. A process failure before the queued mutation
        # commits leaves the entry undiscoverable (safe recompute); it can
        # never expose an uncommitted or out-of-order residency.
        async_publish = os.environ.get("GMS_KV_DIRECTORY_ASYNC_PUBLISH", "1")
        self._async_publish = async_publish.lower() not in (
            "0",
            "false",
            "no",
            "off",
            "",
        )
        self._mutation_condition = threading.Condition()
        self._mutations = deque()
        self._mutation_thread: Optional[threading.Thread] = None
        self._mutation_stop = False
        self._mutation_sequence = 0
        self._mutation_committed = 0
        self._mutation_error: Optional[BaseException] = None
        # Best-effort failure accounting for the async publish worker. A single
        # mutation failure must not kill the worker (see _mutation_loop); these
        # give operators visibility into dropped publications.
        self._mutation_failed = 0
        self._mutation_last_error: Optional[BaseException] = None
        if self.mode != "off" and not self.socket_path:
            logger.warning("GMS KV directory requested without daemon socket")
            self.mode = "off"

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def authoritative(self) -> bool:
        return self.mode == "authoritative"

    @property
    def async_read_enabled(self) -> bool:
        return self.enabled and self._async_read

    def _close_locked(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    @property
    def async_publish_enabled(self) -> bool:
        return self.enabled and self._async_publish

    def _start_mutation_worker_locked(self) -> None:
        if self._mutation_thread is not None and self._mutation_thread.is_alive():
            return
        self._mutation_thread = threading.Thread(
            target=self._mutation_loop,
            name=f"gms-directory-writer-{self.engine}-{self.writer_id}",
            daemon=True,
        )
        self._mutation_thread.start()

    def _defer_mutation(self, kind: str, payload) -> int:
        with self._mutation_condition:
            if self._mutation_error is not None:
                raise RuntimeError("GMS directory mutation worker failed") from (
                    self._mutation_error
                )
            if self._mutation_stop:
                raise RuntimeError("GMS directory mutation worker is closed")
            self._mutation_sequence += 1
            sequence = int(self._mutation_sequence)
            self._mutations.append((sequence, kind, payload))
            self._start_mutation_worker_locked()
            self._mutation_condition.notify()
            return sequence

    def publish_deferred(self, items: list[dict]) -> int:
        """Enqueue an ordered publication; return once ownership is copied.

        The commit point remains the daemon mutation. Until then a crash may
        lose this cache entry, but can never expose a stale slot or wrong
        generation. Call flush_deferred at an explicit durability boundary.
        """
        if not items:
            return 0
        if not self.async_publish_enabled:
            return self.publish(items)
        self._defer_mutation("publish", [dict(item) for item in items])
        return len(items)

    def mark_hbm_dormant_deferred(self, content_hashes: list[bytes]) -> int:
        if not content_hashes:
            return 0
        if not self.async_publish_enabled:
            return self.mark_hbm_dormant(content_hashes)
        self._defer_mutation("dormant", [bytes(value) for value in content_hashes])
        return len(content_hashes)

    def flush_deferred(self, timeout: Optional[float] = None) -> bool:
        """Wait for mutations accepted before this call to commit in order."""
        with self._mutation_condition:
            target = int(self._mutation_sequence)
            if target <= self._mutation_committed:
                return True
            deadline = None if timeout is None else time.monotonic() + timeout
            while self._mutation_committed < target:
                if self._mutation_error is not None:
                    raise RuntimeError("GMS directory mutation worker failed") from (
                        self._mutation_error
                    )
                remaining = (
                    None if deadline is None else deadline - time.monotonic()
                )
                if remaining is not None and remaining <= 0:
                    return False
                self._mutation_condition.wait(remaining)
            return True

    def _mutation_loop(self) -> None:
        while True:
            with self._mutation_condition:
                while not self._mutations and not self._mutation_stop:
                    self._mutation_condition.wait()
                if not self._mutations:
                    return
                sequence, kind, payload = self._mutations.popleft()
            try:
                if kind == "publish":
                    self.publish(payload)
                elif kind == "dormant":
                    self.mark_hbm_dormant(payload)
                else:
                    raise AssertionError(f"unknown directory mutation {kind!r}")
            except BaseException as exc:  # noqa: BLE001
                # A single mutation failure -- a transient daemon error, or this
                # engine being fenced/demoted mid-drain -- must NOT permanently
                # kill the writer and silently stop ALL future publications. An
                # unpublished entry is a safe cache miss (recompute), never a
                # stale hit, so skip this one and keep serving the queue: a
                # transient error self-heals on the next mutation, and a fenced
                # writer simply drops its now-rejected publications. Record the
                # failure for observability and advance the committed sequence so
                # flush_deferred does not block on the skipped mutation.
                logger.warning(
                    "GMS deferred directory mutation (%s) failed; skipping "
                    "(published entry will be a safe cache miss)",
                    kind,
                    exc_info=True,
                )
                with self._mutation_condition:
                    self._mutation_failed += 1
                    self._mutation_last_error = exc
            with self._mutation_condition:
                self._mutation_committed = int(sequence)
                self._mutation_condition.notify_all()

    def close(self) -> None:
        try:
            self.flush_deferred(timeout=2.0)
        except Exception:  # noqa: BLE001
            logger.warning("GMS directory close could not flush mutations", exc_info=True)
        with self._mutation_condition:
            self._mutation_stop = True
            self._mutation_condition.notify_all()
        mutation_thread = self._mutation_thread
        if (
            mutation_thread is not None
            and mutation_thread is not threading.current_thread()
        ):
            mutation_thread.join(timeout=1.0)
        self._view_stop.set()
        thread = self._view_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        with self._lock:
            self._close_locked()

    @property
    def read_view_ready(self) -> bool:
        return self._view_ready.is_set()

    @property
    def read_view_cursor(self) -> tuple[Optional[int], int]:
        with self._view_lock:
            return self._view_epoch, int(self._view_revision)

    def start_async_read(self) -> bool:
        """Start snapshot/delta synchronization without blocking the caller."""
        if not self.enabled or not self._async_read or self._view_stop.is_set():
            return False
        with self._lock:
            if self._view_thread is not None and self._view_thread.is_alive():
                return True
            self._view_thread = threading.Thread(
                target=self._read_view_loop,
                name=f"gms-directory-{self.engine}-{self.writer_id}",
                daemon=True,
            )
            self._view_thread.start()
        return True

    def wait_until_synced(self, timeout: Optional[float] = None) -> bool:
        self.start_async_read()
        return self._view_ready.wait(timeout)

    @property
    def read_view_is_current_writer(self) -> bool:
        if not self._view_ready.is_set():
            return False
        with self._view_lock:
            return self._view_caught_up and self._view_writer == self.writer_id

    def _invalidate_read_view(self) -> None:
        self._view_ready.clear()
        with self._view_lock:
            self._view = {}
            self._view_revision = 0
            self._view_epoch = None
            self._view_writer = None
            self._view_caught_up = False

    def _install_snapshot(
        self,
        entries: dict[bytes, dict],
        revision: int,
        epoch: int,
        writer_id: Optional[str],
    ) -> None:
        with self._view_lock:
            self._view = entries
            self._view_revision = int(revision)
            self._view_epoch = int(epoch)
            self._view_writer = writer_id
            self._view_caught_up = True
        self._view_ready.set()

    def _apply_changes(self, response: dict) -> None:
        if response["reset_required"]:
            self._invalidate_read_view()
            return
        with self._view_lock:
            for change in response["changes"]:
                content_hash = change["content_hash"]
                entry = change["entry"]
                if entry is None:
                    self._view.pop(content_hash, None)
                else:
                    self._view[content_hash] = entry
            self._view_revision = int(response["next_revision"])
            self._view_epoch = int(response["directory_epoch"])
            writer = response.get("writer_id")
            self._view_writer = None if writer is None else str(writer)
            self._view_caught_up = not bool(response["has_more"])
        self._view_ready.set()

    def _read_view_loop(self) -> None:
        client: Optional[DaemonClient] = None
        while not self._view_stop.is_set():
            try:
                if client is None:
                    client = DaemonClient(
                        self.socket_path,
                        connect_timeout=0.5,
                        op_timeout=2.0,
                    )
                if not self._view_ready.is_set():
                    snapshot = client.directory_snapshot(
                        self.manifest_id,
                        scope=self.engine,
                    )
                    entries, revision, epoch, writer = snapshot
                    self._install_snapshot(entries, revision, epoch, writer)
                    continue
                _epoch, revision = self.read_view_cursor
                response = client.directory_changes(
                    self.manifest_id,
                    revision,
                    scope=self.engine,
                    wait_ms=max(1, int(self._poll_seconds * 1_000)),
                )
                if response["reset_required"]:
                    self._invalidate_read_view()
                    continue
                self._apply_changes(response)
                if response["has_more"]:
                    continue
                # The daemon already held this request for ``wait_ms``. Reissue
                # the long poll immediately: sleeping here adds a second blind
                # window in which a committed publication cannot wake us.
                continue
            except Exception:  # noqa: BLE001
                self._invalidate_read_view()
                if client is not None:
                    try:
                        client.close()
                    except Exception:  # noqa: BLE001
                        pass
                    client = None
                self._view_stop.wait(min(0.1, self._poll_seconds * 2))
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass

    def read_view_items(
        self,
        *,
        tier: Optional[str] = None,
        state: str = "ready",
        limit: Optional[int] = None,
    ) -> list[tuple[bytes, dict]]:
        """Return a stable copy of matching entries without daemon IPC."""
        if not self._view_ready.is_set():
            return []
        with self._view_lock:
            result = []
            for content_hash, entry in self._view.items():
                if state and entry.get("state") != state:
                    continue
                if tier is not None and entry.get("tier") != tier:
                    continue
                result.append((bytes(content_hash), dict(entry)))
                if limit is not None and len(result) >= limit:
                    break
            return result

    def _read_view_lookup(
        self, content_hashes: list[bytes]
    ) -> list[Optional[dict]]:
        if not self._view_ready.is_set():
            return [None] * len(content_hashes)
        with self._view_lock:
            return [
                (
                    dict(entry)
                    if (entry := self._view.get(content_hash)) is not None
                    and entry.get("state") == "ready"
                    else None
                )
                for content_hash in content_hashes
            ]

    def _call(
        self, operation: Callable[[DaemonClient], _T], *, retryable: bool = True
    ) -> _T:
        # ``retryable`` must be False for non-idempotent, destructive ops
        # (ensure_hbm_capacity evicts entries; adopt_claim consumes a token). If
        # the reply is lost after the daemon executed such an op, a blind retry
        # would double-evict or falsely report a stale writer. Only retry ops
        # that are safe to run twice.
        with self._lock:
            attempts = 2 if retryable else 1
            for attempt in range(attempts):
                try:
                    if self._client is None:
                        self._client = DaemonClient(
                            self.socket_path,
                            connect_timeout=0.5,
                            op_timeout=2.0,
                        )
                    return operation(self._client)
                except Exception:
                    self._close_locked()
                    if attempt + 1 >= attempts:
                        raise
            raise AssertionError("unreachable")

    def status(self) -> tuple[int, Optional[str]]:
        _entries, epoch, writer_id = self._call(
            lambda client: client.directory_lookup(self.manifest_id, [])
        )
        self._writer_epoch = int(epoch)
        return epoch, writer_id

    def _writer_call(
        self,
        operation: Callable[[DaemonClient, int], tuple[_T, bool]],
        passive_value: _T,
        *,
        retryable: bool = True,
    ) -> _T:
        """Run one fenced mutation, refreshing after external promotion."""
        for attempt in range(2):
            if self._writer_epoch is None:
                self.status()
            assert self._writer_epoch is not None
            value, rejected = self._call(
                lambda client: operation(client, int(self._writer_epoch)),
                retryable=retryable,
            )
            if not rejected:
                self._has_owned = True
                return value
            _epoch, active = self.status()
            if active is None:
                if self._standby and not self._has_owned:
                    return passive_value
                self.promote()
            elif active == self.writer_id:
                self._has_owned = True
            elif self._standby and not self._has_owned:
                return passive_value
            else:
                break
            if attempt:
                break
        raise RuntimeError(f"GMS KV directory rejected stale writer {self.writer_id!r}")

    def promote(self) -> int:
        """Claim writer ownership after the external failover lock is held."""
        epoch, active = self.status()
        if active == self.writer_id:
            self._writer_epoch = int(epoch)
            self._has_owned = True
            return epoch
        promoted, observed_epoch, observed = self._call(
            lambda client: client.directory_promote(epoch, self.writer_id)
        )
        if not promoted or observed != self.writer_id:
            raise RuntimeError(
                "GMS KV directory promotion lost: "
                f"expected={epoch} observed={observed_epoch} writer={observed!r}"
            )
        self._writer_epoch = int(observed_epoch)
        self._has_owned = True
        if self._async_read:
            self._invalidate_read_view()
            self.start_async_read()
        return observed_epoch

    def lookup(self, content_hashes: list[bytes]) -> list[Optional[dict]]:
        if self._async_read:
            self.start_async_read()
            return self._read_view_lookup(content_hashes)
        entries, _epoch, _writer = self._call(
            lambda client: client.directory_lookup(self.manifest_id, content_hashes)
        )
        return entries

    def publish(self, items: list[dict]) -> int:
        if not items:
            return 0

        def request(client, epoch):
            response = client.directory_publish_batch(
                self.manifest_id,
                self.writer_id,
                items,
                expected_epoch=epoch,
                scope=self.engine,
            )
            return response, bool(response["rejected_stale_writer"])

        result = self._writer_call(request, None)
        if result is None:
            return 0
        self._writer_epoch = int(result["directory_epoch"])
        return int(result["published"])

    def lookup_and_claim(
        self, content_hashes: list[bytes]
    ) -> tuple[list[Optional[dict]], Optional[str]]:
        if self._async_read:
            self.start_async_read()
            local = self._read_view_lookup(content_hashes)
            # Misses and host/storage hits are read-only. Only HBM adoption
            # needs the daemon claim that fences eviction and slot reuse.
            if not any(
                entry is not None and entry.get("tier") == "hbm"
                for entry in local
            ):
                return local, None

        def request(client, epoch):
            entries, token, rejected, _observed = client.directory_lookup_claim(
                self.manifest_id, self.writer_id, epoch, content_hashes
            )
            return (entries, token), rejected

        return self._writer_call(
            request,
            ([None] * len(content_hashes), None),
        )

    def release_claim(self, claim_token: Optional[str]) -> bool:
        if not claim_token:
            return False
        return self._call(lambda client: client.directory_release_claim(claim_token))

    def adopt_claim(
        self,
        claim_token: str,
        items: list[dict],
    ) -> int:
        return self._writer_call(
            lambda client, epoch: client.directory_adopt_claim(
                self.manifest_id, self.writer_id, epoch, claim_token, items
            ),
            0,
            retryable=False,
        )

    def ensure_hbm_capacity(self, required_blocks: int) -> list[dict]:
        if required_blocks <= 0:
            return []
        return self._writer_call(
            lambda client, epoch: client.directory_ensure_hbm_capacity(
                self.manifest_id, self.writer_id, epoch, int(required_blocks)
            ),
            [],
            retryable=False,
        )

    def mark_hbm_dormant(self, content_hashes: list[bytes]) -> int:
        if not content_hashes:
            return 0
        return self._writer_call(
            lambda client, epoch: client.directory_mark_hbm_dormant(
                self.manifest_id, self.writer_id, epoch, content_hashes
            ),
            0,
        )

    def hbm_inventory(self) -> dict[str, list[int]]:
        return self._writer_call(
            lambda client, epoch: client.directory_hbm_inventory(
                self.writer_id, epoch, scope=self.engine
            ),
            {},
        )
