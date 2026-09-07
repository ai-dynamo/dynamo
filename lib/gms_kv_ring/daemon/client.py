# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sync client for the daemon's control socket.

Used by engine hooks to:
  1. attach the engine's KV pool (one-time at startup)
  2. attach the evict + restore rings (one-time at startup)
  3. detach on engine shutdown

No hot-path methods here. The rings are the hot path.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from typing import Optional


class DaemonError(RuntimeError):
    pass


class DaemonClient:
    def __init__(
        self,
        socket_path: str,
        *,
        connect_timeout: float = 5.0,
        op_timeout: float = 30.0,
    ) -> None:
        self.socket_path = socket_path
        # Last-seen daemon epoch from any RPC response. None until the
        # first response arrives. The connector reads this via
        # `current_daemon_epoch()` after each RPC and invalidates its
        # _PrefixIndex if it changes between two reads. Crash-restart
        # detection without an extra round trip.
        self._daemon_epoch: Optional[int] = None
        # Serialize concurrent _call() invocations: the single socket
        # carries length-prefixed request/response framing, so two
        # threads sending into the same socket would interleave their
        # request bodies and read each other's responses.
        self._call_lock = threading.Lock()
        deadline = time.monotonic() + connect_timeout
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.connect(socket_path)
                s.settimeout(op_timeout)
                self._sock = s
                if not self._call({"op": "ping"}).get("ok"):
                    raise DaemonError("ping failed")
                return
            except (FileNotFoundError, ConnectionRefusedError) as exc:
                last_err = exc
                s.close()
                time.sleep(0.05)
        raise DaemonError(
            f"could not connect to daemon at {socket_path}: {last_err}",
        )

    def current_daemon_epoch(self) -> Optional[int]:
        """Last `daemon_epoch` value seen on any RPC response, or
        None if no RPC has completed yet. A change between two
        successive reads means the daemon restarted (its in-memory
        state was zeroed)."""
        return self._daemon_epoch

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> "DaemonClient":
        return self

    def __exit__(self, *_a) -> None:
        self.close()

    def _call(self, msg: dict) -> dict:
        body = json.dumps(msg).encode("utf-8")
        with self._call_lock:
            self._sock.sendall(struct.pack("<I", len(body)) + body)
            header = b""
            while len(header) < 4:
                chunk = self._sock.recv(4 - len(header))
                if not chunk:
                    raise DaemonError("daemon closed connection")
                header += chunk
            n = struct.unpack("<I", header)[0]
            body = b""
            while len(body) < n:
                chunk = self._sock.recv(n - len(body))
                if not chunk:
                    raise DaemonError("daemon closed mid-response")
                body += chunk
        resp = json.loads(body.decode("utf-8"))
        # Capture the daemon epoch on every response. Older daemons
        # omit the field — leave the last-seen value unchanged in
        # that case (graceful rolling upgrade).
        ep = resp.get("daemon_epoch") if isinstance(resp, dict) else None
        if ep is not None:
            self._daemon_epoch = int(ep)
        return resp

    def _ok(self, msg: dict) -> dict:
        resp = self._call(msg)
        if not resp.get("ok"):
            raise DaemonError(
                f"daemon op {msg.get('op')!r} failed: {resp.get('error')}",
            )
        return resp

    # ---- API ----

    def directory_lookup(
        self,
        manifest_id: str,
        content_hashes: list[bytes],
    ) -> tuple[list[Optional[dict]], int, Optional[str]]:
        """Lookup READY GMS residencies in request order.

        The manifest scopes hashes to a compatible model, KV layout and
        block size. Missing entries are represented by ``None`` so callers
        can stop at the first gap without rebuilding a hash-to-position map.
        """
        resp = self._ok(
            {
                "op": "directory_lookup",
                "manifest_id": str(manifest_id),
                "hashes": [content_hash.hex() for content_hash in content_hashes],
            }
        )
        entries = [
            None if entry is None else dict(entry)
            for entry in (resp.get("entries", []) or [])
        ]
        epoch = int(resp.get("directory_epoch", 0))
        self._directory_epoch = epoch
        writer_id = resp.get("writer_id")
        return (
            entries,
            epoch,
            None if writer_id is None else str(writer_id),
        )

    def directory_snapshot(
        self,
        manifest_id: str,
        *,
        scope: str = "",
    ) -> tuple[dict[bytes, dict], int, int, Optional[str]]:
        """Read one atomic inventory image and its revision cursor."""
        resp = self._ok(
            {
                "op": "directory_snapshot",
                "manifest_id": str(manifest_id),
                "scope": str(scope),
            }
        )
        entries = {
            bytes.fromhex(str(item["content_hash"])): dict(item["entry"])
            for item in (resp.get("items") or [])
        }
        epoch = int(resp.get("directory_epoch", 0))
        revision = int(resp.get("directory_revision", 0))
        self._directory_epoch = epoch
        writer_id = resp.get("writer_id")
        return entries, revision, epoch, (None if writer_id is None else str(writer_id))

    def directory_changes(
        self,
        manifest_id: str,
        after_revision: int,
        *,
        scope: str = "",
        limit: int = 4096,
        wait_ms: int = 0,
    ) -> dict:
        """Read committed upserts/deletes following a snapshot cursor."""
        resp = self._ok(
            {
                "op": "directory_changes",
                "manifest_id": str(manifest_id),
                "scope": str(scope),
                "after_revision": int(after_revision),
                "limit": int(limit),
                "wait_ms": int(wait_ms),
            }
        )
        epoch = int(resp.get("directory_epoch", 0))
        self._directory_epoch = epoch
        return {
            "changes": [
                {
                    "revision": int(change["revision"]),
                    "content_hash": bytes.fromhex(str(change["content_hash"])),
                    "entry": (
                        None if change.get("entry") is None else dict(change["entry"])
                    ),
                }
                for change in (resp.get("changes") or [])
            ],
            "next_revision": int(resp.get("next_revision", after_revision)),
            "directory_revision": int(resp.get("directory_revision", after_revision)),
            "directory_epoch": epoch,
            "writer_id": resp.get("writer_id"),
            "has_more": bool(resp.get("has_more", False)),
            "reset_required": bool(resp.get("reset_required", False)),
        }

    def directory_promote(
        self,
        expected_epoch: int,
        writer_id: str,
    ) -> tuple[bool, int, Optional[str]]:
        """CAS-promote ``writer_id`` and return its resulting epoch."""
        resp = self._ok(
            {
                "op": "directory_promote",
                "expected_epoch": int(expected_epoch),
                "writer_id": str(writer_id),
            }
        )
        active = resp.get("writer_id")
        epoch = int(resp.get("directory_epoch", 0))
        self._directory_epoch = epoch
        return (
            bool(resp.get("promoted", False)),
            epoch,
            None if active is None else str(active),
        )

    def directory_publish_batch(
        self,
        manifest_id: str,
        writer_id: str,
        items: "list[dict]",
        expected_epoch: Optional[int] = None,
        scope: str = "",
    ) -> dict:
        """Publish sealed daemon-owned KV locations for the active writer."""
        payload_items = []
        for item in items:
            slot_ids = item.get("slot_ids")
            if slot_ids is None:
                slot_ids = [item["slot_id"]]
            generations = item.get("generations")
            if generations is None:
                generations = [item.get("generation", 0)] * len(slot_ids)
            payload_items.append(
                {
                    "content_hash": item["content_hash"].hex(),
                    "engine_id": str(item["engine_id"]),
                    "slot_ids": [int(slot_id) for slot_id in slot_ids],
                    "generations": [int(generation) for generation in generations],
                    "ranges": [
                        {
                            "layer": int(layer),
                            "offset": int(offset),
                            "size": int(size),
                        }
                        for layer, offset, size in item.get("ranges", [])
                    ],
                    "tier": str(item.get("tier", "")),
                    "sealed": bool(item.get("sealed", True)),
                    "active": bool(item.get("active", False)),
                }
            )
        resp = self._ok(
            {
                "op": "directory_publish_batch",
                "manifest_id": str(manifest_id),
                "writer_id": str(writer_id),
                "scope": str(scope),
                "expected_epoch": int(
                    expected_epoch
                    if expected_epoch is not None
                    else getattr(self, "_directory_epoch", 0)
                ),
                "items": payload_items,
            }
        )
        self._directory_epoch = int(resp.get("directory_epoch", 0))
        return {
            "published": int(resp.get("published", 0)),
            "removed": int(resp.get("removed", 0)),
            "rejected_stale_writer": bool(resp.get("rejected_stale_writer", False)),
            "directory_epoch": int(resp.get("directory_epoch", 0)),
            "writer_id": resp.get("writer_id"),
        }

    def directory_lookup_claim(
        self,
        manifest_id: str,
        writer_id: str,
        expected_epoch: int,
        content_hashes: list[bytes],
    ) -> tuple[list[Optional[dict]], Optional[str], bool, int]:
        resp = self._ok(
            {
                "op": "directory_lookup_claim",
                "manifest_id": str(manifest_id),
                "writer_id": str(writer_id),
                "expected_epoch": int(expected_epoch),
                "hashes": [value.hex() for value in content_hashes],
            }
        )
        epoch = int(resp.get("directory_epoch", 0))
        self._directory_epoch = epoch
        token = resp.get("claim_token")
        return (
            [
                None if entry is None else dict(entry)
                for entry in (resp.get("entries") or [])
            ],
            None if token is None else str(token),
            bool(resp.get("rejected_stale_writer", False)),
            epoch,
        )

    def directory_release_claim(self, claim_token: str) -> bool:
        resp = self._ok(
            {
                "op": "directory_release_claim",
                "claim_token": str(claim_token),
            }
        )
        return bool(resp.get("released", False))

    def directory_adopt_claim(
        self,
        manifest_id: str,
        writer_id: str,
        expected_epoch: int,
        claim_token: str,
        items: list[dict],
    ) -> tuple[int, bool]:
        resp = self._ok(
            {
                "op": "directory_adopt_claim",
                "manifest_id": str(manifest_id),
                "writer_id": str(writer_id),
                "expected_epoch": int(expected_epoch),
                "claim_token": str(claim_token),
                "items": [
                    {
                        "content_hash": item["content_hash"].hex(),
                        "generations": [int(value) for value in item["generations"]],
                    }
                    for item in items
                ],
            }
        )
        return (
            int(resp.get("adopted", 0)),
            bool(resp.get("rejected_stale_writer", False)),
        )

    def directory_ensure_hbm_capacity(
        self,
        manifest_id: str,
        writer_id: str,
        expected_epoch: int,
        required_blocks: int,
    ) -> tuple[list[dict], bool]:
        resp = self._ok(
            {
                "op": "directory_ensure_hbm_capacity",
                "manifest_id": str(manifest_id),
                "writer_id": str(writer_id),
                "expected_epoch": int(expected_epoch),
                "required_blocks": int(required_blocks),
            }
        )
        victims = []
        for item in resp.get("victims") or []:
            victims.append(
                {
                    "content_hash": bytes.fromhex(str(item["content_hash"])),
                    "engine_id": str(item["engine_id"]),
                    "slot_ids": [int(value) for value in item["slot_ids"]],
                    "generations": [int(value) for value in item["generations"]],
                }
            )
        return victims, bool(resp.get("rejected_stale_writer", False))

    def directory_mark_hbm_dormant(
        self,
        manifest_id: str,
        writer_id: str,
        expected_epoch: int,
        content_hashes: list[bytes],
    ) -> tuple[int, bool]:
        resp = self._ok(
            {
                "op": "directory_mark_hbm_dormant",
                "manifest_id": str(manifest_id),
                "writer_id": str(writer_id),
                "expected_epoch": int(expected_epoch),
                "hashes": [value.hex() for value in content_hashes],
            }
        )
        return (
            int(resp.get("updated", 0)),
            bool(resp.get("rejected_stale_writer", False)),
        )

    def directory_hbm_inventory(
        self,
        writer_id: str,
        expected_epoch: int,
        scope: str = "",
    ) -> tuple[dict[str, list[int]], bool]:
        resp = self._ok(
            {
                "op": "directory_hbm_inventory",
                "writer_id": str(writer_id),
                "expected_epoch": int(expected_epoch),
                "scope": str(scope),
            }
        )
        protected = {
            str(engine_id): [int(slot_id) for slot_id in slot_ids]
            for engine_id, slot_ids in (resp.get("protected") or {}).items()
        }
        return protected, bool(resp.get("rejected_stale_writer", False))
