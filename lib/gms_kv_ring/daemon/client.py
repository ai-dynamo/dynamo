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

    def attach_engine_pool(
        self,
        engine_id: str,
        layers: list[dict],
    ) -> None:
        """`layers`: list of {layer_idx, va, size, stride} dicts."""
        self._ok(
            {
                "op": "attach_engine_pool",
                "engine_id": engine_id,
                "layers": layers,
            }
        )

    def detach_engine_pool(self, engine_id: str) -> bool:
        resp = self._ok(
            {
                "op": "detach_engine_pool",
                "engine_id": engine_id,
            }
        )
        return bool(resp.get("found", False))

    def attach_evict_ring(
        self,
        engine_id: str,
        ring_path: str,
        counter_host_addr: int = 0,
        num_counters: int = 0,
        counter_path: str = "",
    ) -> None:
        """Enable evict-ack. Two modes:
        - same-process: pass `counter_host_addr` (engine's pinned VA)
        - cross-process: pass `counter_path` (filesystem path)
        Both need `num_counters`. Default = legacy fire-and-forget
        (unsafe with HBM-slot-reuse)."""
        self._ok(
            {
                "op": "attach_evict_ring",
                "engine_id": engine_id,
                "ring_path": ring_path,
                "counter_host_addr": int(counter_host_addr),
                "num_counters": int(num_counters),
                "counter_path": counter_path,
            }
        )

    def demote_to_storage(
        self,
        engine_id: str,
        layer: int,
        offset: int,
    ) -> bool:
        """Move a host_tier slot to the storage tier. Returns True iff
        the daemon found a ready host slot and wrote a CRC-stamped
        file. The host slot is freed on success."""
        resp = self._ok(
            {
                "op": "demote_to_storage",
                "engine_id": engine_id,
                "layer": int(layer),
                "offset": int(offset),
            }
        )
        return bool(resp.get("demoted", False))

    def promote_from_storage(
        self,
        engine_id: str,
        layer: int,
        offset: int,
    ) -> bool:
        """Load a storage_tier slot back into host_tier. Returns True
        on successful CRC verify; False if file missing, malformed,
        or CRC-mismatched (at-rest corruption)."""
        resp = self._ok(
            {
                "op": "promote_from_storage",
                "engine_id": engine_id,
                "layer": int(layer),
                "offset": int(offset),
            }
        )
        return bool(resp.get("promoted", False))

    def demote_hbm_to_storage(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        size: int,
        *,
        generation: int = 0,
    ) -> bool:
        """HBM_FRESH -> AT_REST_STORAGE in one hop. Requires the
        storage backend to support GPU-direct (currently: NixlBackend
        with plugin in {GDS, GDS_MT}). Skips host_tier allocation
        entirely; the bytes go from HBM to durable storage via
        GPUDirect Storage.

        `generation`: caller-supplied generation tag for the slot's
        block_id (offset // stride). Recorded daemon-side so the
        restore consumer can detect cross-step async-restore-vs-
        evict races and refuse to read post-overwrite bytes.
        `0` means "don't care" (legacy callers)."""
        resp = self._ok(
            {
                "op": "demote_hbm_to_storage",
                "engine_id": engine_id,
                "layer": int(layer),
                "offset": int(offset),
                "size": int(size),
                "generation": int(generation),
            }
        )
        return bool(resp.get("demoted", False))

    def capabilities(self) -> dict:
        """Snapshot of the daemon's runtime feature map. Used by
        engine adapters to decide whether to enable GPU-direct
        paths at attach time. Plain dict — caller should `.get`
        with defaults for forward compat."""
        resp = self._ok({"op": "capabilities"})
        return dict(resp.get("capabilities", {}))

    def promote_storage_to_hbm(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        size: int,
        dest_offset: Optional[int] = None,
        expected_generation: int = 0,
    ) -> bool:
        """AT_REST_STORAGE -> HBM_CACHED in one hop. Symmetric to
        `demote_hbm_to_storage`: storage backend reads the file
        directly into the engine's HBM region via GPUDirect Storage,
        verifies CRC, returns True on success / False on CRC
        mismatch or any other verify failure.

        `offset` keys the storage slot. `dest_offset` (default: same
        as `offset`) is the byte offset inside the engine's pool to
        write the restored bytes to. They differ when the engine's
        cache hit assigned a new block_id for the same content —
        the connector's restore-on-hit path needs cross-block-id
        remap (src_offset = src_block_id * block_bytes; dest_offset
        = dst_block_id * block_bytes)."""
        if dest_offset is None:
            dest_offset = offset
        resp = self._ok(
            {
                "op": "promote_storage_to_hbm",
                "engine_id": engine_id,
                "layer": int(layer),
                "offset": int(offset),
                "size": int(size),
                "dest_offset": int(dest_offset),
                "expected_generation": int(expected_generation),
            }
        )
        return bool(resp.get("promoted", False))

    def release_engine_storage(self, engine_id: str) -> int:
        """Drop every storage-tier slot for an engine. Use after the
        engine is permanently retired. Returns the count released."""
        resp = self._ok(
            {
                "op": "release_engine_storage",
                "engine_id": engine_id,
            }
        )
        return int(resp.get("released", 0))

    def prune_storage(
        self,
        max_age_seconds: Optional[float] = None,
        max_bytes: Optional[int] = None,
        max_bytes_per_engine: Optional[int] = None,
    ) -> int:
        """Operator-driven storage cleanup. Pass one or more of:
          - max_age_seconds: TTL prune
          - max_bytes_per_engine: per-engine LRU cap (prevents one
            noisy engine from starving others on shared disk)
          - max_bytes: global LRU eviction until total <= max_bytes
        Returns total slots evicted."""
        payload = {"op": "prune_storage"}
        if max_age_seconds is not None:
            payload["max_age_seconds"] = float(max_age_seconds)
        if max_bytes is not None:
            payload["max_bytes"] = int(max_bytes)
        if max_bytes_per_engine is not None:
            payload["max_bytes_per_engine"] = int(max_bytes_per_engine)
        resp = self._ok(payload)
        return int(resp.get("evicted", 0))

    def storage_stats(self) -> dict:
        """Snapshot of storage-tier resident state. Keys:
        n_slots, total_bytes, oldest_mtime, newest_mtime."""
        resp = self._ok({"op": "storage_stats"})
        return dict(resp.get("stats", {}))

    def transport_info(self) -> dict:
        """Discover the daemon's NIXL transport listen endpoint.
        Returns `{agent_name, listen_port, receive_buffer_bytes}`.
        Used by the router to learn each daemon's transport identity
        before issuing `transport_add_peer` calls. Empty values mean
        the daemon was started without cross-node transport."""
        resp = self._ok({"op": "transport_info"})
        return {
            "agent_name": str(resp.get("agent_name", "")),
            "listen_port": int(resp.get("listen_port", 0)),
            "receive_buffer_bytes": int(resp.get("receive_buffer_bytes", 0)),
        }

    def transport_add_peer(
        self,
        nixl_name: str,
        ip_addr: str,
        port: int,
        label: str = "",
    ) -> None:
        """Tell this daemon about a peer NIXL agent it should be able
        to send to. Triggers a NIXL metadata exchange. The router
        calls this on each pair of daemons that may need to transfer
        between each other (mesh setup)."""
        self._ok(
            {
                "op": "transport_add_peer",
                "nixl_name": str(nixl_name),
                "ip_addr": str(ip_addr),
                "port": int(port),
                "label": str(label),
            }
        )

    def staging_reserve(
        self,
        content_hash: bytes,
        size: int,
        source_daemon: str = "",
    ) -> dict:
        """Receiver-side RPC: reserve a staging slot + receive buffer
        offset for an inbound cross-node transfer. Router calls this
        on the destination daemon before commanding the source daemon
        to send.

        Returns a dict with `outcome` in `{reserved, already_ready,
        coalesced}`:
          - reserved: `{reservation_id, remote_ptr}` — sender uses
            remote_ptr in `initialize_xfer(..., remote_descs)` and
            reservation_id in the notif payload.
          - already_ready: `{bytes_size, generation}` — the receiver
            already has this hash; sender skips.
          - coalesced: another transfer for this hash is in flight;
            sender skips and retries later.
        Raises DaemonError on hard failure (capacity exhausted, etc.)."""
        resp = self._ok(
            {
                "op": "staging_reserve",
                "content_hash": content_hash.hex(),
                "size": int(size),
                "source_daemon": str(source_daemon),
            }
        )
        return {
            "outcome": str(resp.get("outcome", "")),
            "reservation_id": str(resp.get("reservation_id", "")),
            "remote_ptr": int(resp.get("remote_ptr", 0)),
            "bytes_size": int(resp.get("bytes_size", 0)),
            "generation": int(resp.get("generation", 0)),
        }

    def register_content_address(
        self,
        content_hash: bytes,
        engine_id: str,
        ranges: list[tuple[int, int, int]],
        *,
        generation: int | None = None,
        sealed: bool = True,
        metadata: dict | None = None,
    ) -> int:
        """Connector-side: advertise `(content_hash → host_tier
        addresses)` after a successful spill. Each tuple is
        `(layer, offset, size)`. Daemon stores the mapping and
        publishes a Stored event (tier=host_pinned). Returns total
        bytes registered. Multi-range because one logical block
        spans N layers."""
        body = {
            "op": "register_content_address",
            "content_hash": content_hash.hex(),
            "engine_id": str(engine_id),
            "ranges": [
                {"layer": int(L), "offset": int(o), "size": int(s)}
                for (L, o, s) in ranges
            ],
            "sealed": bool(sealed),
        }
        if generation is not None:
            body["generation"] = int(generation)
        if metadata is not None:
            body["metadata"] = metadata
        resp = self._ok(body)
        return int(resp.get("total_size", 0))

    def register_content_addresses_batch(
        self,
        items: "list[dict]",
    ) -> tuple[int, bool]:
        """Batched register_content_address. Each item is a dict with
        `content_hash` (bytes), `engine_id` (str), and `ranges`
        (list[tuple[layer, offset, size]]). Returns `(total_bytes, skipped)`.

        `skipped=True` means the daemon has no cross-node transport
        enabled; the RPC is a no-op and the caller can skip future
        batches without performance cost. Use this to short-circuit
        in standalone GMS deployments."""
        payload_items = []
        for it in items:
            body = {
                "content_hash": it["content_hash"].hex(),
                "engine_id": str(it["engine_id"]),
                "ranges": [
                    {"layer": int(L), "offset": int(o), "size": int(s)}
                    for (L, o, s) in it["ranges"]
                ],
            }
            if "generation" in it and it["generation"] is not None:
                body["generation"] = int(it["generation"])
            if "sealed" in it:
                body["sealed"] = bool(it["sealed"])
            if "metadata" in it and it["metadata"] is not None:
                body["metadata"] = it["metadata"]
            payload_items.append(body)
        resp = self._ok(
            {
                "op": "register_content_addresses_batch",
                "items": payload_items,
            }
        )
        return (int(resp.get("total_bytes", 0)), bool(resp.get("skipped", False)))

    def staging_fail(self, reservation_id: str, reason: str = "") -> None:
        """Cleanup on transfer failure. Called by the router if the
        sender reports an error before bytes arrive."""
        self._ok(
            {
                "op": "staging_fail",
                "reservation_id": str(reservation_id),
                "reason": str(reason),
            }
        )

    @staticmethod
    def hash_bytes(data: bytes, mode: str = "sha256") -> bytes:
        """Compute a 32-byte content hash using the daemon's
        supported modes:
          sha256       - cryptographic, slowest (default)
          blake2b_256  - full 256-bit BLAKE2b

        The caller (engine connector or test driver) must use the
        SAME mode the daemon's staging_tier was configured with.
        Mode is set via `GMS_HASH_MODE` env var on daemon start."""
        from gms_kv_ring.daemon.staging_tier import hash_fn_for_mode

        return hash_fn_for_mode(mode)(data)

    def register_bootstrap_handle(
        self,
        items: "list[dict]",
    ) -> int:
        """Engine-direct path: register `items = [{content_hash:
        bytes, ptr: int, size: int}, ...]` so that peer engines can
        NIXL-read these regions from this daemon's NIXL agent. Returns
        the count successfully registered."""
        body_items = []
        for it in items:
            body = {
                "content_hash": it["content_hash"].hex(),
                "ptr": int(it["ptr"]),
                "size": int(it["size"]),
            }
            if "generation" in it and it["generation"] is not None:
                body["generation"] = int(it["generation"])
            if "sealed" in it:
                body["sealed"] = bool(it["sealed"])
            body_items.append(body)
        resp = self._ok(
            {
                "op": "register_bootstrap_handle",
                "items": body_items,
            }
        )
        return int(resp.get("registered", 0))

    def get_bootstrap_info(self, hashes: "list[bytes]") -> dict:
        """Engine-direct path: ask this daemon "where can I NIXL-read
        these hashes from?" Returns `{nixl_agent_name, listen_port,
        agent_metadata_b64, descriptors=[{ptr,size,tier,ranges?,generation?,
        sealed?} or None, ...]}`."""
        resp = self._ok(
            {
                "op": "get_bootstrap_info",
                "hashes": [h.hex() for h in hashes],
            }
        )
        return resp

    def notify_kv_arrived(
        self,
        items: "list[dict]",
    ) -> int:
        """Engine-direct path: tell this daemon "I just NIXL-read
        these hashes; please publish PlacementEvent::Stored". Off the
        critical path — for the indexer, not for data flow."""
        body_items = []
        for it in items:
            body = {
                "content_hash": it["content_hash"].hex(),
                "size": int(it.get("size", 0)),
            }
            if "metadata" in it and it["metadata"] is not None:
                body["metadata"] = it["metadata"]
            body_items.append(body)
        resp = self._ok(
            {
                "op": "notify_kv_arrived",
                "items": body_items,
            }
        )
        return int(resp.get("published", 0))

    def read_bootstrap_into_staging(
        self,
        source_nixl_name: str,
        source_agent_metadata_hex: str,
        hashes: "list[bytes]",
        descriptors: "list[dict | None]",
        timeout_s: float = 30.0,
        batch_size: int = 0,
    ) -> dict:
        """Destination-side router placement path.

        The router forwards source NIXL metadata and per-hash descriptors
        through the normal request plane. The destination worker asks its
        local daemon to NIXL-READ those descriptors into the staging tier,
        after which the existing engine connector consumes the local staged
        bytes.
        """
        body = {
            "op": "read_bootstrap_into_staging",
            "source_nixl_name": str(source_nixl_name),
            "source_agent_metadata_hex": str(source_agent_metadata_hex),
            "hashes": [h.hex() for h in hashes],
            "descriptors": descriptors,
            "timeout_s": float(timeout_s),
        }
        if batch_size > 0:
            body["batch_size"] = int(batch_size)
        resp = self._ok(body)
        return {
            "accepted": int(resp.get("accepted", 0)),
            "already_ready": int(resp.get("already_ready", 0)),
            "coalesced": int(resp.get("coalesced", 0)),
            "failed": int(resp.get("failed", 0)),
            "skipped": int(resp.get("skipped", 0)),
            "bytes_read": int(resp.get("bytes_read", 0)),
        }

    def fetch_remote(
        self,
        source_uds_path: str,
        source_nixl_name: str,
        source_ip: str,
        source_port: int,
        hashes: "list[bytes]",
        bytes_per_hash: int,
        timeout_s: float = 30.0,
        batch_size: int = 0,
    ) -> dict:
        """Pattern C: ask the DESTINATION daemon (self) to orchestrate
        an inbound transfer. The daemon opens a control connection to
        the source daemon (no router-side coordination) and drives a
        sequence of `transfer_blocks_batch` calls. Each batch becomes
        one multi-region NIXL xfer; the destination publishes
        per-block `PlacementEvent::Stored` events as each batch lands.

        `batch_size` controls pipelining granularity:
          * 0 or omitted → one xfer for all hashes (max wire
            efficiency, decode waits for all bytes)
          * = blocks-per-layer → Dynamo-style per-layer streaming
            (decode pipelines layer compute against in-flight xfers)
          * 1 → per-block xfer (max pipelining; subject to NIXL
            per-peer cap for non-batched mode — typically not needed
            in batched mode)

        Returns `{accepted, already_ready, coalesced, failed}`."""
        body = {
            "op": "fetch_remote",
            "source_uds_path": str(source_uds_path),
            "source_nixl_name": str(source_nixl_name),
            "source_ip": str(source_ip),
            "source_port": int(source_port),
            "hashes": [h.hex() for h in hashes],
            "bytes_per_hash": int(bytes_per_hash),
            "timeout_s": float(timeout_s),
        }
        if batch_size > 0:
            body["batch_size"] = int(batch_size)
        resp = self._ok(body)
        return {
            "accepted": int(resp.get("accepted", 0)),
            "already_ready": int(resp.get("already_ready", 0)),
            "coalesced": int(resp.get("coalesced", 0)),
            "failed": int(resp.get("failed", 0)),
        }

    def restore_host_blocks(
        self,
        engine_id: str,
        src_engine_id: str,
        block_hits: "list[tuple[int, int, int]]",
    ) -> bool:
        """Synchronously restore CPU-mirrored host-tier blocks.

        ``block_hits`` entries are ``(src_block_id, dest_block_id,
        expected_generation)``. A nonzero expected generation must match the
        host-tier slot generation for every layer, otherwise the daemon returns
        ``False`` and the engine falls back to recompute.
        """
        resp = self._ok(
            {
                "op": "restore_host_blocks",
                "engine_id": str(engine_id),
                "src_engine_id": str(src_engine_id),
                "items": [
                    {
                        "src_block": int(src),
                        "dest_block": int(dst),
                        "generation": int(gen),
                    }
                    for src, dst, gen in block_hits
                ],
            }
        )
        return bool(resp.get("success", False))

    def restore_staging_blocks(
        self,
        engine_id: str,
        block_hits: "list[tuple[bytes, int, int]]",
    ) -> bool:
        """Synchronously restore staged blocks into an attached engine.

        ``block_hits`` entries are ``(content_hash, dest_block_id,
        generation)``. Returns True iff the daemon copied every block
        and synchronized its restore stream.
        """
        resp = self._ok(
            {
                "op": "restore_staging_blocks",
                "engine_id": str(engine_id),
                "items": [
                    {
                        "content_hash": h.hex(),
                        "dest_block": int(dst),
                        "generation": int(gen),
                    }
                    for h, dst, gen in block_hits
                ],
            }
        )
        return bool(resp.get("success", False))

    def restore_staging_ranges(
        self,
        engine_id: str,
        range_hits: "list[tuple[bytes, int, list[tuple[int, int, int]]]]",
    ) -> bool:
        """Synchronously restore staged payloads into explicit ranges.

        ``range_hits`` entries are ``(content_hash, generation,
        ranges)`` where ranges are ordered ``(layer, offset, size)``
        tuples in the destination engine's attached pool. This covers
        engines such as SGLang where one cross-node content hash spans
        several per-token KV slots rather than one daemon block id.
        """
        resp = self._ok(
            {
                "op": "restore_staging_ranges",
                "engine_id": str(engine_id),
                "items": [
                    {
                        "content_hash": h.hex(),
                        "generation": int(gen),
                        "ranges": [
                            {
                                "layer": int(layer),
                                "offset": int(offset),
                                "size": int(size),
                            }
                            for layer, offset, size in ranges
                        ],
                    }
                    for h, gen, ranges in range_hits
                ],
            }
        )
        return bool(resp.get("success", False))

    def register_staging_restore_handles(
        self,
        items: "list[dict]",
    ) -> "list[Optional[int]]":
        """Register one-shot restore-ring handles for staging hits.

        Each item is ``{"content_hash": bytes, "generation": int}``.
        Returned handles are aligned with ``items`` and can be used as
        the src field in a FLAG_SOURCE_STAGING restore record.
        ``None`` means that item was malformed or no handle was
        available.
        """
        payload_items = []
        for it in items:
            payload_items.append(
                {
                    "content_hash": it["content_hash"].hex(),
                    "generation": int(it["generation"]),
                }
            )
        resp = self._ok(
            {
                "op": "register_staging_restore_handles",
                "items": payload_items,
            }
        )
        handles = []
        for h in resp.get("handles", []) or []:
            handles.append(None if h is None else int(h))
        return handles

    def release_staging_restore_handles(self, handles: "list[int]") -> int:
        """Release staging restore handles that were registered but not
        pushed to the restore ring, for example when the ring is full."""
        resp = self._ok(
            {
                "op": "release_staging_restore_handles",
                "handles": [int(h) for h in handles],
            }
        )
        return int(resp.get("released", 0))

    def staging_scan(
        self,
        content_hashes: list[bytes],
    ) -> dict[bytes, dict]:
        """Batch lookup against the daemon's staging tier (Phase 2 of
        cross-node design, see docs/CROSS_NODE_DESIGN.md §4.3).
        Returns a dict mapping `content_hash` (bytes) → metadata dict
        with keys `bytes_size`, `crc32`, `generation`. Hashes not
        currently READY in staging are omitted. An empty dict means
        either no hits or the daemon was built without a staging tier
        (the common case for standalone GMS without dynamo)."""
        if not content_hashes:
            return {}
        resp = self._ok(
            {
                "op": "staging_scan",
                "hashes": [h.hex() for h in content_hashes],
            }
        )
        raw = resp.get("hits", {}) or {}
        return {bytes.fromhex(k): dict(v) for k, v in raw.items()}

    def attach_restore_ring(
        self,
        engine_id: str,
        ring_path: str,
        counter_path: str,
        num_counters: int = 512,
        counter_host_addr: int = 0,
    ) -> None:
        """`counter_host_addr` is the engine's already-pinned VA of the
        counter file (UVA → device pointer). Pass it when the daemon
        runs in the same process so we can skip a redundant mmap +
        cuMemHostRegister round."""
        self._ok(
            {
                "op": "attach_restore_ring",
                "engine_id": engine_id,
                "ring_path": ring_path,
                "counter_path": counter_path,
                "num_counters": int(num_counters),
                "counter_host_addr": int(counter_host_addr),
            }
        )
