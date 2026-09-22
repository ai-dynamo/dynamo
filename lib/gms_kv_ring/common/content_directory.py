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
from copy import deepcopy
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


def resolve_manifest_id(
    engine: str, block_size: int, *, keyspace: Optional[str] = None
) -> str:
    """Return the explicit compatibility manifest or a safe POC fallback.

    Production deployments should set ``GMS_KV_DIRECTORY_MANIFEST`` from the
    model/layout metadata. The fallback intentionally includes engine, model,
    block size and cache salt so accidental incompatible reuse is unlikely.
    """
    explicit = os.environ.get("GMS_KV_DIRECTORY_MANIFEST", "").strip()
    if explicit:
        return explicit
        # The explicit value is the complete compatibility identity shared by
        # engine adapters and failover orchestration. Appending adapter-local
        # fields here made those two clients silently address different
        # manifests during promotion.
    model = (
        os.environ.get("DYN_MODEL_PATH")
        or os.environ.get("DYN_MODEL")
        or os.environ.get("MODEL_PATH")
        or "unknown-model"
    )
    salt = os.environ.get("GMS_KVR_CROSS_NODE_SALT", "")
    manifest = f"poc-v1|{engine}|{model}|block={int(block_size)}|salt={salt}"
    if keyspace:
        manifest = f"{manifest}|keyspace={str(keyspace).strip()}"
    return manifest


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
        self.manifest_id = resolve_manifest_id(engine, block_size, keyspace=keyspace)
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
        # Published last by the single view-reader thread. CPython object
        # reference assignment is atomic under the GIL, so hot-path readers do
        # not need the view mutex merely to consume this derived status bit.
        self._view_current_writer = False
        self._view_ready = threading.Event()
        self._view_caught_up = False
        self._view_stop = threading.Event()
        self._view_thread: Optional[threading.Thread] = None
        # Publishing a newly completed block synchronously delays the first
        # token even though no request consumes the directory reply. The
        # fallback worker preserves the older acknowledged path during
        # startup; steady-state READY publications use the daemon-owned
        # pipeline below.
        async_publish = os.environ.get("GMS_KV_DIRECTORY_ASYNC_PUBLISH", "1")
        self._async_publish = async_publish.lower() not in (
            "0",
            "false",
            "no",
            "off",
            "",
        )
        self._mutation_submit_lock = threading.Lock()
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
        # Steady-state READY publications use a dedicated ordered connection:
        # the scheduler copies a complete frame into the Unix socket and
        # returns, while this reader validates daemon acknowledgements. The
        # daemon owns the queued frame even if the engine then exits.
        pipeline_value = os.environ.get("GMS_KV_DIRECTORY_PIPELINED_PUBLISH", "1")
        self._pipeline_enabled = pipeline_value.lower() not in (
            "0",
            "false",
            "no",
            "off",
            "",
        )
        self._pipeline_condition = threading.Condition()
        self._pipeline_client: Optional[DaemonClient] = None
        self._pipeline_expected = deque()
        self._pipeline_thread: Optional[threading.Thread] = None
        self._pipeline_stop = False
        self._pipeline_sequence = 0
        self._pipeline_committed = 0
        self._pipeline_error: Optional[BaseException] = None
        if self.mode != "off" and not self.socket_path:
            logger.warning("GMS KV directory requested without daemon socket")
            self.mode = "off"

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def authoritative(self) -> bool:
        return self.mode == "authoritative" or (
            self.mode == "shadow" and self.read_view_is_current_writer
        )

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

    def _start_pipeline_locked(self) -> None:
        if self._pipeline_client is None:
            self._pipeline_client = DaemonClient(
                self.socket_path,
                connect_timeout=0.5,
                op_timeout=2.0,
            )
        if self._pipeline_thread is not None and self._pipeline_thread.is_alive():
            return
        self._pipeline_thread = threading.Thread(
            target=self._pipeline_loop,
            name=f"gms-directory-pipeline-{self.engine}-{self.writer_id}",
            daemon=True,
        )
        self._pipeline_thread.start()

    def _pipeline_loop(self) -> None:
        while True:
            with self._pipeline_condition:
                while not self._pipeline_expected and not self._pipeline_stop:
                    self._pipeline_condition.wait()
                if not self._pipeline_expected:
                    return
                sequence, expected = self._pipeline_expected[0]
                client = self._pipeline_client
            assert client is not None
            try:
                response = client.receive_response()
                if not response.get("ok"):
                    raise RuntimeError(
                        f"GMS pipelined publication failed: {response.get('error')}"
                    )
                if response.get("rejected_stale_writer"):
                    raise RuntimeError(
                        "GMS pipelined publication rejected stale writer"
                    )
                accepted = int(response.get("accepted", response.get("published", 0)))
                if accepted != int(expected):
                    raise RuntimeError(
                        "GMS pipelined publication committed "
                        f"{accepted}/{expected} entries"
                    )
            except BaseException as exc:  # noqa: BLE001
                with self._pipeline_condition:
                    self._pipeline_error = exc
                    self._pipeline_stop = True
                    self._pipeline_expected.clear()
                    self._pipeline_condition.notify_all()
                return
            with self._pipeline_condition:
                if (
                    not self._pipeline_expected
                    or self._pipeline_expected[0][0] != sequence
                ):
                    self._pipeline_error = RuntimeError(
                        "GMS publication acknowledgement sequence diverged"
                    )
                    self._pipeline_stop = True
                    self._pipeline_expected.clear()
                    self._pipeline_condition.notify_all()
                    return
                self._pipeline_expected.popleft()
                self._pipeline_committed = int(sequence)
                self._pipeline_condition.notify_all()

    def _pipeline_publish(self, items: list[dict]) -> bool:
        if not self._pipeline_enabled:
            return False
        with self._view_lock:
            if not self._view_current_writer or self._view_epoch is None:
                return False
            epoch = int(self._view_epoch)
        message = DaemonClient.directory_publish_batch_message(
            self.manifest_id,
            self.writer_id,
            items,
            epoch,
            self.engine,
        )
        with self._pipeline_condition:
            if self._pipeline_error is not None:
                raise RuntimeError("GMS publication pipeline failed") from (
                    self._pipeline_error
                )
            if self._pipeline_stop:
                raise RuntimeError("GMS publication pipeline is closed")
            self._start_pipeline_locked()
            assert self._pipeline_client is not None
            self._pipeline_sequence += 1
            sequence = int(self._pipeline_sequence)
            self._pipeline_expected.append((sequence, len(items)))
            try:
                self._pipeline_client.send_request(message)
            except BaseException as exc:
                self._pipeline_expected.pop()
                self._pipeline_error = exc
                self._pipeline_stop = True
                if self._pipeline_client is not None:
                    try:
                        self._pipeline_client.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self._pipeline_client = None
                self._pipeline_condition.notify_all()
                raise
            self._pipeline_condition.notify_all()
        return True

    def _flush_pipeline(self, timeout: Optional[float]) -> bool:
        with self._pipeline_condition:
            target = int(self._pipeline_sequence)
            if target <= self._pipeline_committed:
                return True
            deadline = None if timeout is None else time.monotonic() + timeout
            while self._pipeline_committed < target:
                if self._pipeline_error is not None:
                    raise RuntimeError("GMS publication pipeline failed") from (
                        self._pipeline_error
                    )
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._pipeline_condition.wait(remaining)
            return True

    def _defer_mutation(self, kind: str, payload) -> int:
        # Keep direct durability commits ordered with concurrent producers.
        with self._mutation_submit_lock, self._mutation_condition:
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

        In steady state the complete frame is copied to the daemon connection
        before returning; the engine may exit while the daemon commits it.
        During startup the legacy worker queue remains the safe fallback.
        """
        if not items:
            return 0
        if not self.async_publish_enabled:
            return self.publish(items)
        if self._pipeline_enabled:
            if self._pipeline_publish(items):
                return len(items)
            # Before the replicated read view confirms this writer epoch, keep
            # the original acknowledged commit semantics. A userspace queue
            # would be lost if the engine crashed during that transition.
            return self.publish(items)
        self._defer_mutation("publish", deepcopy(items))
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
        started = time.monotonic()
        with self._mutation_condition:
            target = int(self._mutation_sequence)
            if target > self._mutation_committed:
                deadline = None if timeout is None else time.monotonic() + timeout
                while self._mutation_committed < target:
                    if self._mutation_error is not None:
                        raise RuntimeError(
                            "GMS directory mutation worker failed"
                        ) from self._mutation_error
                    remaining = (
                        None if deadline is None else deadline - time.monotonic()
                    )
                    if remaining is not None and remaining <= 0:
                        return False
                    self._mutation_condition.wait(remaining)
        remaining = (
            None
            if timeout is None
            else max(0.0, timeout - (time.monotonic() - started))
        )
        return self._flush_pipeline(remaining)

    def commit_hbm_dormant(
        self, content_hashes: list[bytes], timeout: Optional[float] = None
    ) -> bool:
        """Commit completed HBM blocks without a queue-and-wake round trip.

        Earlier active publications remain ordered through the mutation worker.
        Once they drain, this scheduler-thread call writes the READY transition
        directly to the daemon and returns only after its writer fence commits.
        """
        if not content_hashes:
            return True
        with self._mutation_submit_lock:
            if self.async_publish_enabled and not self.flush_deferred(timeout):
                return False
            self.mark_hbm_dormant(content_hashes)
            return True

    @staticmethod
    def _publication_keys(items: list[dict]) -> Optional[set[tuple]]:
        # The daemon validates each batch atomically: repeated content hashes
        # or reused physical slots must remain separate, ordered publications.
        # Malformed items stay isolated so normal RPC error handling can skip
        # them without terminating the background worker during coalescing.
        try:
            keys = set()
            for item in items:
                keys.add(("hash", bytes(item["content_hash"])))
                slots = item.get("slot_ids")
                if slots is None:
                    slots = [item["slot_id"]]
                keys.update(
                    ("slot", str(item["engine_id"]), int(slot)) for slot in slots
                )
            return keys
        except (KeyError, TypeError, ValueError):
            return None

    def _mutation_loop(self) -> None:
        while True:
            with self._mutation_condition:
                while not self._mutations and not self._mutation_stop:
                    self._mutation_condition.wait()
                if not self._mutations:
                    return
                sequence, kind, payload = self._mutations.popleft()
                payload = list(payload)
                # Coalesce disjoint publications only. Turning two sequential
                # updates of one hash/slot into an atomic batch makes the daemon
                # reject both. Bound coalescing to avoid monopolizing the queue.
                keys = self._publication_keys(payload) if kind == "publish" else set()
                while self._mutations and self._mutations[0][1] == kind:
                    following = self._mutations[0][2]
                    if len(payload) + len(following) > 4096:
                        break
                    following_keys = (
                        self._publication_keys(following)
                        if kind == "publish"
                        else set()
                    )
                    if keys is None or following_keys is None or keys & following_keys:
                        break
                    sequence, _same_kind, following = self._mutations.popleft()
                    payload.extend(following)
                    keys.update(following_keys)
            try:
                if kind == "publish":
                    self.publish(payload)
                elif kind == "dormant":
                    self.mark_hbm_dormant(payload)
                else:
                    raise AssertionError(f"unknown directory mutation {kind!r}")
            except BaseException as exc:  # noqa: BLE001
                stale_writer = isinstance(
                    exc, RuntimeError
                ) and "rejected stale writer" in str(exc)
                if stale_writer:
                    logger.error(
                        "GMS deferred directory mutation rejected a stale writer; "
                        "stopping the writer to preserve the generation fence",
                        exc_info=True,
                    )
                    with self._mutation_condition:
                        self._mutation_error = exc
                        self._mutations.clear()
                        self._mutation_condition.notify_all()
                    return

                # Other publication failures are safe cache misses. Keep the
                # ordered writer alive so a transient daemon error can heal on
                # the next mutation, and advance the committed sequence so an
                # explicit flush cannot strand forever on the skipped entry.
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
            logger.warning(
                "GMS directory close could not flush mutations", exc_info=True
            )
        with self._mutation_condition:
            self._mutation_stop = True
            self._mutation_condition.notify_all()
        mutation_thread = self._mutation_thread
        if (
            mutation_thread is not None
            and mutation_thread is not threading.current_thread()
        ):
            mutation_thread.join(timeout=1.0)
        with self._pipeline_condition:
            self._pipeline_stop = True
            self._pipeline_condition.notify_all()
        pipeline_client = self._pipeline_client
        if pipeline_client is not None:
            pipeline_client.close()
        pipeline_thread = self._pipeline_thread
        if (
            pipeline_thread is not None
            and pipeline_thread is not threading.current_thread()
        ):
            pipeline_thread.join(timeout=1.0)
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

    def freeze_current_writer_view(self) -> bool:
        """Stop consuming self-published deltas after writer handoff completes.

        A promoted engine needs one synchronized snapshot to discover the
        predecessor's recoverable HBM entries and to prove that its writer epoch
        is current. After that inventory has been copied into engine-local
        metadata, replaying every publication back into the same process only
        contends with the scheduler. The ordered publication connection remains
        open and continues to validate every daemon acknowledgement, including
        stale-writer rejection.

        The frozen ownership bit is valid for the process's writer epoch. Loss
        of the external failover lock fences the process; it is not expected to
        become a standby again in place.
        """
        if not self.read_view_is_current_writer or not self._view_ready.is_set():
            return False
        self._async_read = False
        self._view_stop.set()
        thread = self._view_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self._poll_seconds * 2))
        return thread is None or not thread.is_alive()

    def wait_until_synced(self, timeout: Optional[float] = None) -> bool:
        self.start_async_read()
        return self._view_ready.wait(timeout)

    @property
    def read_view_is_current_writer(self) -> bool:
        return bool(self._view_current_writer)

    def _invalidate_read_view(self) -> None:
        # Revoke the lock-free status before clearing any view state.
        self._view_current_writer = False
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
            self._view_current_writer = writer_id == self.writer_id
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
            self._view_current_writer = (
                self._view_caught_up and self._view_writer == self.writer_id
            )
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

    def may_have_hbm_candidate(self, content_hashes: list[bytes]) -> bool:
        """Cheaply test whether an authoritative HBM claim may be useful.

        An unavailable or not-yet-synchronized local view must return true so
        callers fall back to the authoritative path. Once synchronized, a
        definite miss can stay entirely on the local read path.
        """
        if not self._async_read or not self._view_ready.is_set():
            return True
        with self._view_lock:
            if not self._view_caught_up:
                return True
            return any(
                entry is not None
                and entry.get("tier") == "hbm"
                and entry.get("state") in ("ready", "active")
                for content_hash in content_hashes
                if (entry := self._view.get(content_hash)) is not None
            )

    def _read_view_lookup(self, content_hashes: list[bytes]) -> list[Optional[dict]]:
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
        # Destructive mutations on the regular RPC connection must observe all
        # READY publications already accepted on the pipelined connection.
        if threading.current_thread() is not self._mutation_thread:
            if not self._flush_pipeline(timeout=2.0):
                raise RuntimeError(
                    "timed out ordering a directory mutation after publication"
                )
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

    def promote(self, *, force_new_epoch: bool = False) -> int:
        """Claim writer ownership after the external failover lock is held.

        Force a new epoch only at a proven external-fence boundary. Normal
        calls remain idempotent for clients in the same live engine cohort.
        """
        epoch, active = self.status()
        if active == self.writer_id and not force_new_epoch:
            self._writer_epoch = int(epoch)
            self._has_owned = True
            return epoch
        promoted, observed_epoch, observed = self._call(
            lambda client: client.directory_promote(
                epoch,
                self.writer_id,
                force_new_epoch=force_new_epoch,
            )
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
        return self.lookup_authoritative(content_hashes)

    def lookup_authoritative(self, content_hashes: list[bytes]) -> list[Optional[dict]]:
        """Read READY entries directly from the daemon.

        Normal request lookups deliberately use the asynchronous local view.
        Transaction cleanup instead needs a read-after-write check against the
        daemon commit point, without waiting for that view to catch up.
        """
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
        return int(result.get("accepted", result["published"]))

    def lookup_and_claim(
        self, content_hashes: list[bytes]
    ) -> tuple[list[Optional[dict]], Optional[str]]:
        if self._async_read:
            self.start_async_read()
            # The public read view intentionally exposes only READY entries,
            # but a TP peer may have changed a shared HBM entry to ACTIVE while
            # staging the same successor generation. That ACTIVE candidate
            # still requires an authoritative daemon claim: short-circuiting it
            # as a miss lets TP ranks choose different prefix lengths and hang
            # in their next collective.
            with self._view_lock:
                raw = [self._view.get(content_hash) for content_hash in content_hashes]
                local = [
                    (
                        dict(entry)
                        if entry is not None and entry.get("state") == "ready"
                        else None
                    )
                    for entry in raw
                ]
            # Misses and host/storage hits are read-only. Only HBM adoption
            # needs the daemon claim that fences eviction and slot reuse.
            if not any(
                entry is not None
                and entry.get("tier") == "hbm"
                and entry.get("state") in ("ready", "active")
                for entry in raw
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

    def lookup_and_read_claim(
        self, content_hashes: list[bytes]
    ) -> tuple[list[Optional[dict]], Optional[str]]:
        """Pin READY HBM entries for a non-writer consuming their bytes."""
        return self._call(
            lambda client: client.directory_lookup_read_claim(
                self.manifest_id, content_hashes
            )
        )

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

    def ensure_hbm_capacity(
        self, required_blocks: int, *, eligible_slot_ids: list[int] | None = None
    ) -> list[dict]:
        if required_blocks <= 0:
            return []
        return self._writer_call(
            lambda client, epoch: client.directory_ensure_hbm_capacity(
                self.manifest_id,
                self.writer_id,
                epoch,
                int(required_blocks),
                **(
                    {"eligible_slot_ids": eligible_slot_ids}
                    if eligible_slot_ids is not None
                    else {}
                ),
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
        protected, _leases = self.hbm_lease_inventory()
        return protected

    def hbm_lease_inventory(
        self,
    ) -> tuple[dict[str, list[int]], dict[str, list[tuple[int, int]]] | None,]:
        def request(client, epoch):
            protected, leases, rejected = client.directory_hbm_lease_inventory(
                self.writer_id, epoch, scope=self.engine
            )
            return (protected, leases), rejected

        return self._writer_call(
            request,
            ({}, None),
        )

    def compare_prefix(
        self,
        content_hashes: list[bytes],
        legacy_count: int,
    ) -> tuple[list[Optional[dict]], bool]:
        """Lookup once and report whether its contiguous prefix agrees."""
        entries = self.lookup(content_hashes)
        directory_count = 0
        for entry in entries:
            if entry is None:
                break
            directory_count += 1
        return entries, directory_count == int(legacy_count)
