# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL-backed cross-node KV transport.

Each daemon owns a `NixlTransport` that wraps a NIXL agent with the
UCX backend. The agent:

  - Listens for inbound transfers (`enable_listen_thread=True`).
  - Registers a staging-tier buffer pool with NIXL for RDMA targeting.
  - Issues outbound `WRITE` transfers to peer agents.
  - Drains notification messages on a background thread; each notif
    carries `{reservation_id, content_hash}` and triggers the
    receiver-side `StagingTier.commit_or_reject` path.

Single-host loopback works via UCX-over-SHM (no hardware needed).
Cross-host production uses UCX-over-InfiniBand or UCX-over-RoCE.

Wire protocol (per transfer):

  1. Router commands sender via UDS: `transfer_block(target, hash, payload_bytes_or_ptr)`.
  2. Sender → reserves a sender-side memory slot, copies bytes in,
     registers if not already, calls `NixlTransport.send(peer, hash, src_ptr, size)`.
  3. NixlTransport.send:
        a. Allocates receiver-side reservation via the existing UDS
           control channel (peer's `staging_tier_reserve` op). Receiver
           pre-allocates a staging-tier slot and replies with
           `(reservation_id, dst_ptr)`.
        b. `initialize_xfer("WRITE", local_descs, remote_descs,
                            peer_name, notif_msg={hash, reservation_id})`.
        c. `transfer(handle)`. On completion, receiver's notif arrives.
  4. Receiver's notif drain thread:
        a. Decodes notif → (reservation_id, content_hash).
        b. Reads bytes from staging-tier dst buffer (already populated
           by the NIXL WRITE) and calls
           `staging_tier.commit_or_reject(reservation_id, payload)`.
        c. StagingTier verifies hash (I-9) and transitions to READY
           or CORRUPT.

Why pre-reservation: NIXL needs both sides' descriptors before the
WRITE. The sender can't write into staging without the receiver having
allocated and registered the destination buffer first. This adds one
UDS round-trip (the pre-reserve RPC) on top of the RDMA transfer; for
typical block sizes (~MB) the overhead is amortized.

Invariants enforced here:

  - I-8 (atomic reservation): the receiver's `staging_tier_reserve`
    RPC goes through `StagingTier.reserve_or_wait`; concurrent
    transfers for the same hash coalesce as waiters in the staging tier.
  - I-9 (hash-on-receive): the receiver's commit path verifies the
    payload hash before transitioning to READY.

Failure modes:

  - Peer unreachable / metadata exchange failed → `TransportClosed`
    from `add_peer`; caller can retry or fall back.
  - Transfer fails → sender's `transfer()` returns non-DONE status;
    sender calls `staging_tier_fail_reservation` RPC to release the
    receiver-side reservation; waiters get notified of failure.
  - Receiver agent dies mid-transfer → metadata invalidation; sender
    sees ERROR on next `transfer()`; reservation stale-times out on
    receiver if it restarts before the sender retries.
"""

from __future__ import annotations

import logging
import os
import struct
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---- Exception types --------------------------------------------------------


class TransportNotAvailable(RuntimeError):
    """NIXL or its UCX plugin is not importable / not constructible."""


class TransportClosed(RuntimeError):
    """Operation attempted on a closed transport, or peer is unreachable."""


# ---- Public API -------------------------------------------------------------


@dataclass(frozen=True)
class PeerHandle:
    """Identifies a peer NIXL agent reachable from this transport.

    `nixl_name` is the agent's identity in NIXL (returned by
    `add_remote_agent`). `ip_addr` + `port` are the listen endpoint
    used for metadata exchange (NIXL's own listen thread)."""

    nixl_name: str
    ip_addr: str
    port: int


@dataclass
class _AsyncSend:
    """One in-flight NIXL send tracked by the completion loop.

    `handle` is the opaque NIXL xfer handle. `on_complete` is invoked
    with `(success: bool, error_msg: str)` from the completion thread
    once NIXL reports DONE/ERR or `deadline_monotonic` passes. After
    `on_complete` runs, the NIXL handle is released."""

    handle: object
    peer_name: str
    reservation_id: str
    deadline_monotonic: float
    on_complete: object = None


_BATCH_NOTIF_MAGIC = b"BNTF"  # 4-byte prefix for multi-record notifs


def _encode_batch_notif(records: "list[tuple[str, bytes]]") -> bytes:
    """Multi-record notif carried by a single NIXL xfer covering N
    block WRITEs (the per-request-batched path that matches Dynamo's
    one-xfer-per-request shape). One transfer, one notif, N records.

    Layout:
        [4: "BNTF" magic]
        [4 LE: n_records]
        for each record:
            [4 LE: rid_len][rid_len: reservation_id utf-8][32: content_hash]

    For a single-block legacy path, `_encode_notif` (no magic prefix)
    is still used; `_decode_batch_notif` handles both."""
    parts: list[bytes] = [_BATCH_NOTIF_MAGIC, struct.pack("<I", len(records))]
    for rid, content_hash in records:
        if len(content_hash) != 32:
            raise ValueError(
                f"content_hash must be 32 bytes, got {len(content_hash)}",
            )
        rid_b = rid.encode("utf-8")
        parts.append(struct.pack("<I", len(rid_b)))
        parts.append(rid_b)
        parts.append(content_hash)
    return b"".join(parts)


def _decode_batch_notif(msg: bytes) -> "list[tuple[str, bytes]]":
    """Decode either the batch format (with BNTF magic) or the legacy
    single-record format. Returns a list of (reservation_id,
    content_hash) tuples — length 1 for the legacy format, ≥1 for
    batched."""
    if len(msg) >= 4 and msg[:4] == _BATCH_NOTIF_MAGIC:
        if len(msg) < 8:
            raise ValueError("batch notif truncated")
        n = struct.unpack_from("<I", msg, 4)[0]
        out: list[tuple[str, bytes]] = []
        pos = 8
        for _ in range(n):
            if pos + 4 > len(msg):
                raise ValueError("batch notif truncated mid-record")
            rid_len = struct.unpack_from("<I", msg, pos)[0]
            pos += 4
            if pos + rid_len + 32 > len(msg):
                raise ValueError("batch notif truncated mid-record body")
            rid = msg[pos : pos + rid_len].decode("utf-8")
            pos += rid_len
            content_hash = msg[pos : pos + 32]
            pos += 32
            out.append((rid, content_hash))
        return out
    # Legacy single-record fallback.
    rid, h = _decode_notif(msg)
    return [(rid, h)]


def _encode_notif(reservation_id: str, content_hash: bytes) -> bytes:
    """Wire format for inbound transfer notifications (legacy
    single-record). See `_encode_batch_notif` for the modern
    per-request multi-record format.

    Layout:
        [4 LE: rid_len][rid_len: reservation_id utf-8][32: content_hash]
    """
    rid_bytes = reservation_id.encode("utf-8")
    if len(content_hash) != 32:
        raise ValueError(
            f"content_hash must be 32 bytes, got {len(content_hash)}",
        )
    return struct.pack("<I", len(rid_bytes)) + rid_bytes + content_hash


def _decode_notif(msg: bytes) -> tuple[str, bytes]:
    if len(msg) < 4:
        raise ValueError(f"notif too short ({len(msg)} bytes)")
    rid_len = struct.unpack("<I", msg[:4])[0]
    expected = 4 + rid_len + 32
    if len(msg) != expected:
        raise ValueError(
            f"notif length mismatch: got {len(msg)}, expected {expected}",
        )
    rid = msg[4 : 4 + rid_len].decode("utf-8")
    h = msg[4 + rid_len : 4 + rid_len + 32]
    return rid, h


class NixlTransport:
    """Per-daemon NIXL agent + connection manager + notif drain.

    Lifecycle: construct, `add_peer()` for each known peer, `send()`
    to push bytes; `close()` to tear down."""

    def __init__(
        self,
        agent_name: str,
        listen_port: int,
        *,
        ucx_backend: str = "UCX",
        notif_poll_interval_s: float = 0.005,
        on_inbound_notif=None,
    ) -> None:
        """`on_inbound_notif(peer_name, reservation_id, content_hash)` is
        invoked on the drain thread when a remote agent finishes
        WRITE-ing to one of our registered slots. The caller (Daemon)
        wires this to `StagingTier.commit_or_reject` via a closure
        that knows how to read the destination bytes."""
        try:
            from nixl._api import nixl_agent, nixl_agent_config
        except ImportError as exc:
            raise TransportNotAvailable(
                f"pynixl not importable: {exc}",
            ) from exc

        self._agent_name = agent_name
        self._listen_port = int(listen_port)
        backend = os.environ.get("GMS_KVR_NIXL_BACKEND") or ucx_backend
        self._ucx_backend = backend
        self._on_inbound_notif = on_inbound_notif

        try:
            cfg = nixl_agent_config(
                enable_prog_thread=True,
                enable_listen_thread=True,
                listen_port=self._listen_port,
                backends=[backend],
            )
            self._agent = nixl_agent(agent_name, cfg)
        except Exception as exc:
            raise TransportNotAvailable(
                f"failed to construct nixl_agent: {exc}",
            ) from exc

        # Peer registry: nixl_name → PeerHandle. Populated by add_peer.
        self._peers: dict[str, PeerHandle] = {}
        # Memory registrations: list of ctypes-managed buffers (held
        # so refcount keeps them alive). NIXL's internal registration
        # is by (ptr, size); we just remember what we've registered to
        # avoid double-registration.
        self._registered: set[tuple[int, int]] = set()

        # Notif drain thread. All state the thread reads (stop event,
        # poll interval, callback) MUST be initialized before .start()
        # — otherwise the thread can race ahead and AttributeError.
        self._stop = threading.Event()
        self._poll_s = float(notif_poll_interval_s)
        self._lock = threading.Lock()  # guards _peers, _registered
        self._closed = False
        # Async-send tracking: (handle_id → _AsyncSend record). The
        # background completion loop polls each pending xfer, releases
        # its NIXL handle on DONE/ERR, and fires `on_complete`. This is
        # what removes the synchronous waiter that previously stalled
        # `send()` under bidirectional load.
        self._async_lock = threading.Lock()
        self._async_pending: dict[int, "_AsyncSend"] = {}
        self._notif_thread = threading.Thread(
            target=self._notif_loop,
            name=f"nixl-notif-{agent_name}",
            daemon=True,
        )
        self._notif_thread.start()
        self._completion_thread = threading.Thread(
            target=self._completion_loop,
            name=f"nixl-completion-{agent_name}",
            daemon=True,
        )
        self._completion_thread.start()

    # ----- peer registration -----

    def add_peer(self, ip_addr: str, port: int, label: str = "") -> PeerHandle:
        """Fetch the remote agent's metadata over its listen socket and
        register it locally. Returns the PeerHandle for later send()
        calls. Idempotent: re-adding the same peer is a no-op.

        `label` is an optional NIXL-side label for metadata fetch
        bookkeeping; can be empty for the common case."""
        if self._closed:
            raise TransportClosed("transport is closed")
        # NIXL fetch_remote_metadata takes the remote agent's nixl
        # name as the first argument. The first connect doesn't know
        # it, so we use the empty-string form which negotiates via
        # the listen socket.
        try:
            self._agent.fetch_remote_metadata(
                remote_agent="",
                ip_addr=ip_addr,
                port=int(port),
                label=label,
            )
        except Exception as exc:
            raise TransportClosed(
                f"fetch_remote_metadata({ip_addr}:{port}) failed: {exc}",
            ) from exc

        # After fetch, the new remote agent name appears in the
        # agent's local registry. We don't have a clean API to list
        # them, so the caller must pass us the expected name; for now
        # we assume the convention that nixl_name == f"gms-{port}" or
        # similar agreed at construction. This is brittle; cleaner
        # alternative is to send_local_metadata from peer with a
        # pre-arranged name. Documented as future work.
        # Workaround: poll for the new peer's name by checking
        # check_remote_metadata against a placeholder. For now we
        # rely on the caller passing it.
        raise NotImplementedError(
            "add_peer: name resolution is pending — use add_peer_by_name "
            "with the known agent_name in this build",
        )

    def add_peer_by_name(
        self,
        nixl_name: str,
        ip_addr: str,
        port: int,
        label: str = "",
    ) -> PeerHandle:
        """Same as add_peer but the caller already knows the peer's
        NIXL agent name (out-of-band: e.g., daemon socket exchanges
        it before transport setup). This is the production path until
        we wire up an agent-name discovery RPC."""
        if self._closed:
            raise TransportClosed("transport is closed")
        with self._lock:
            existing = self._peers.get(nixl_name)
            if existing is not None:
                return existing
        try:
            # send_local_metadata pushes OUR metadata to peer's listen
            # socket. The peer registers us. Then we fetch theirs.
            self._agent.send_local_metadata(ip_addr=ip_addr, port=int(port))
            self._agent.fetch_remote_metadata(
                remote_agent=nixl_name,
                ip_addr=ip_addr,
                port=int(port),
                label=label,
            )
        except Exception as exc:
            raise TransportClosed(
                f"metadata exchange with {nixl_name}@{ip_addr}:{port} "
                f"failed: {exc}",
            ) from exc
        handle = PeerHandle(
            nixl_name=nixl_name,
            ip_addr=ip_addr,
            port=int(port),
        )
        with self._lock:
            self._peers[nixl_name] = handle
        return handle

    def add_peer_inproc(
        self,
        peer_transport: "NixlTransport",
    ) -> PeerHandle:
        """Loopback (same-process) metadata exchange path. Avoids the
        listen-thread metadata roundtrip by snapshotting the peer's
        metadata directly. Bidirectional: registers `self` on `peer`
        AND `peer` on `self`. Intended for tests and single-host
        multi-GPU deployments where both transports live in the same
        Python process. Production cross-host deployments use
        `add_peer_by_name` over the listen socket."""
        if self._closed:
            raise TransportClosed("transport is closed")
        with self._lock:
            existing = self._peers.get(peer_transport._agent_name)
            if existing is not None:
                return existing
        # NIXL metadata is point-in-time — must be captured AFTER
        # any register_memory calls on the peer that we want to be
        # visible. Caller is responsible for the ordering.
        peer_md = peer_transport._agent.get_agent_metadata()
        my_md = self._agent.get_agent_metadata()
        peer_transport._agent.add_remote_agent(my_md)
        self._agent.add_remote_agent(peer_md)
        handle = PeerHandle(
            nixl_name=peer_transport._agent_name,
            ip_addr="inproc",
            port=peer_transport._listen_port,
        )
        with self._lock:
            self._peers[peer_transport._agent_name] = handle
        # Also register US on peer (symmetric) so peer can issue notifs back.
        with peer_transport._lock:
            if self._agent_name not in peer_transport._peers:
                peer_transport._peers[self._agent_name] = PeerHandle(
                    nixl_name=self._agent_name,
                    ip_addr="inproc",
                    port=self._listen_port,
                )
        return handle

    def remove_peer(self, nixl_name: str) -> bool:
        if self._closed:
            return False
        with self._lock:
            handle = self._peers.pop(nixl_name, None)
        if handle is None:
            return False
        try:
            self._agent.remove_remote_agent(nixl_name)
        except Exception:
            logger.exception(
                "[NixlTransport] remove_remote_agent(%s) failed",
                nixl_name,
            )
        return True

    def add_peer_from_metadata(
        self,
        nixl_name: str,
        metadata: bytes,
    ) -> PeerHandle:
        """Register a peer from an in-band NIXL metadata blob.

        Router-orchestrated decode-to-decode placement does not expose the
        source daemon's UDS path to the destination worker. Instead, the
        source worker returns a fresh metadata blob through Dynamo's request
        plane. The destination daemon registers that blob locally and can then
        issue NIXL READs against the source agent.
        """
        if self._closed:
            raise TransportClosed("transport is closed")
        peer_name = str(nixl_name)
        if not peer_name:
            raise TransportClosed("peer NIXL agent name is empty")
        with self._lock:
            existing = self._peers.get(peer_name)
        metadata_bytes = bytes(metadata or b"")
        if not metadata_bytes:
            raise TransportClosed(
                f"metadata for peer {peer_name!r} is empty",
            )
        try:
            self._agent.add_remote_agent(metadata_bytes)
        except Exception as exc:
            if existing is None:
                raise TransportClosed(
                    f"add_remote_agent({peer_name}) failed: {exc}",
                ) from exc
            try:
                self._agent.remove_remote_agent(peer_name)
                self._agent.add_remote_agent(metadata_bytes)
            except Exception as refresh_exc:
                logger.debug(
                    "[NixlTransport] peer metadata refresh for %s failed; "
                    "using existing metadata: %s",
                    peer_name,
                    refresh_exc,
                )
                return existing
        handle = PeerHandle(nixl_name=peer_name, ip_addr="metadata", port=0)
        with self._lock:
            self._peers[peer_name] = handle
        return handle

    # ----- memory registration -----

    def register_buffer(self, ptr: int, size: int, label: str = "") -> None:
        """Register a host-pinned buffer with NIXL so peers can RDMA
        into / out of it. Idempotent for the same (ptr, size) pair."""
        if self._closed:
            raise TransportClosed("transport is closed")
        key = (int(ptr), int(size))
        with self._lock:
            if key in self._registered:
                return
        try:
            self._agent.register_memory(
                [(int(ptr), int(size), 0, label or f"buf@{ptr:x}")],
                mem_type="DRAM",
            )
        except Exception as exc:
            raise TransportClosed(
                f"register_memory({ptr:x}, {size}) failed: {exc}",
            ) from exc
        with self._lock:
            self._registered.add(key)

    # ----- outbound transfer -----

    def send(
        self,
        peer: PeerHandle,
        local_ptr: int,
        size: int,
        remote_ptr: int,
        reservation_id: str,
        content_hash: bytes,
        timeout_s: float = 30.0,
    ) -> None:
        """Push `size` bytes from local memory at `local_ptr` to peer's
        memory at `remote_ptr` via NIXL UCX. Sender's `local_ptr` must
        already be `register_buffer`-ed. Receiver-side preconditions:
        peer has reserved a staging slot, registered the destination
        buffer with NIXL, and given us `remote_ptr` out-of-band.

        On successful WRITE completion, NIXL delivers a notif to the
        peer carrying `(reservation_id, content_hash)`. The peer's
        notif drain calls into its StagingTier.commit_or_reject.

        Raises TransportClosed on metadata-resolution failure;
        TimeoutError on timeout; RuntimeError on transfer-level
        failures."""
        if self._closed:
            raise TransportClosed("transport is closed")

        # Build local + remote descriptor lists. Each is a single
        # contiguous range for the typical case.
        local_descs = self._agent.get_xfer_descs(
            [(int(local_ptr), int(size), 0)],
            mem_type="DRAM",
        )
        remote_descs = self._agent.get_xfer_descs(
            [(int(remote_ptr), int(size), 0)],
            mem_type="DRAM",
        )
        # Confirm peer's descriptor metadata is reachable; gives a
        # cleaner error message than initialize_xfer's
        # NIXL_ERR_NOT_FOUND when metadata exchange is incomplete.
        if not self._agent.check_remote_metadata(peer.nixl_name, remote_descs):
            raise TransportClosed(
                f"remote metadata for {peer.nixl_name} doesn't cover "
                f"ptr={remote_ptr:x} size={size}"
            )

        notif = _encode_notif(reservation_id, content_hash)
        try:
            handle = self._agent.initialize_xfer(
                "WRITE",
                local_descs,
                remote_descs,
                peer.nixl_name,
                notif_msg=notif,
            )
        except Exception as exc:
            raise TransportClosed(
                f"initialize_xfer to {peer.nixl_name} failed: {exc}",
            ) from exc

        try:
            status = self._agent.transfer(handle)
            # NIXL returns "DONE" immediately if synchronous; otherwise
            # we poll. Either way `release_xfer_handle` is required.
            deadline = time.monotonic() + float(timeout_s)
            while status not in ("DONE", "ERR"):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"transfer to {peer.nixl_name} did not "
                        f"complete within {timeout_s}s",
                    )
                time.sleep(0.001)
                status = self._agent.transfer(handle)
            if status == "ERR":
                raise RuntimeError(
                    f"transfer to {peer.nixl_name} reported ERR",
                )
        finally:
            try:
                self._agent.release_xfer_handle(handle)
            except Exception:
                logger.exception(
                    "[NixlTransport] release_xfer_handle failed",
                )

    def read_batch(
        self,
        source_nixl_name: str,
        items: "list[tuple[int, int, int]]",
        timeout_s: float = 30.0,
    ) -> None:
        """Pull one or more remote regions into local registered memory.

        ``items`` entries are ``(local_ptr, size, remote_ptr)``. The local
        pointers must lie inside a buffer registered on this agent, and the
        remote pointers must be covered by the source agent metadata previously
        registered with :meth:`add_peer_from_metadata`.
        """
        if self._closed:
            raise TransportClosed("transport is closed")
        if not items:
            raise ValueError("read_batch: items must be non-empty")
        peer_name = str(source_nixl_name)
        if not peer_name:
            raise TransportClosed("source NIXL agent name is empty")

        local_descs = self._agent.get_xfer_descs(
            [(int(lp), int(sz), 0) for (lp, sz, _rp) in items],
            mem_type="DRAM",
        )
        remote_descs = self._agent.get_xfer_descs(
            [(int(rp), int(sz), 0) for (_lp, sz, rp) in items],
            mem_type="DRAM",
        )
        if not self._agent.check_remote_metadata(peer_name, remote_descs):
            raise TransportClosed(
                f"remote metadata for {peer_name} doesn't cover one or more "
                f"of {len(items)} READ regions"
            )

        try:
            handle = self._agent.initialize_xfer(
                "READ",
                local_descs,
                remote_descs,
                peer_name,
            )
        except Exception as exc:
            raise TransportClosed(
                f"initialize_xfer READ from {peer_name} failed: {exc}",
            ) from exc

        try:
            status = self._agent.transfer(handle)
            deadline = time.monotonic() + float(timeout_s)
            while status not in ("DONE", "ERR"):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"READ from {peer_name} did not complete within "
                        f"{timeout_s}s",
                    )
                time.sleep(0.001)
                status = self._agent.transfer(handle)
            if status == "ERR":
                raise RuntimeError(f"READ from {peer_name} reported ERR")
        finally:
            try:
                self._agent.release_xfer_handle(handle)
            except Exception:
                logger.exception(
                    "[NixlTransport] release_xfer_handle failed",
                )

    def _completion_loop(self) -> None:
        """Poll outstanding async sends; release handles + fire
        callbacks when they reach DONE/ERR or the deadline passes."""
        while not self._stop.is_set():
            try:
                with self._async_lock:
                    items = list(self._async_pending.items())
                if not items:
                    time.sleep(self._poll_s)
                    continue
                for hid, rec in items:
                    try:
                        status = self._agent.transfer(rec.handle)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[NixlTransport async] transfer() poll for "
                            "rid=%s peer=%s raised: %s — treating as ERR",
                            rec.reservation_id,
                            rec.peer_name,
                            exc,
                        )
                        status = "ERR"
                    now = time.monotonic()
                    if status == "DONE":
                        self._finalize_async(hid, rec, True, "")
                    elif status == "ERR":
                        self._finalize_async(
                            hid,
                            rec,
                            False,
                            f"transfer to {rec.peer_name} reported ERR",
                        )
                    elif now > rec.deadline_monotonic:
                        self._finalize_async(
                            hid,
                            rec,
                            False,
                            f"transfer to {rec.peer_name} timed out",
                        )
                time.sleep(self._poll_s)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[NixlTransport async] completion loop iteration "
                    "raised; continuing",
                )
                time.sleep(self._poll_s)

    def send_async_batch(
        self,
        peer: PeerHandle,
        items: "list[tuple[int, int, int, str, bytes]]",
        timeout_s: float = 30.0,
        on_complete=None,
    ) -> int:
        """Submit ONE NIXL xfer covering N WRITEs in a single
        operation — the per-request batched path matching Dynamo's
        one-xfer-per-request shape.

        `items` is a list of `(local_ptr, size, remote_ptr,
        reservation_id, content_hash)` tuples. NIXL builds one
        multi-region xfer; the notif carries all N (rid, hash)
        records. Receiver's notif drain decodes them and processes
        each.

        This eliminates the per-WRITE concurrency limit: instead of
        N concurrent WRITEs (which UCX-TCP can't reliably handle), we
        issue ONE multi-region WRITE. Bytes for different blocks
        cannot mix because they're one xfer with deterministic
        descriptor ordering.

        Returns the synthetic handle id for tracking.
        """
        if self._closed:
            raise TransportClosed("transport is closed")
        if not items:
            raise ValueError("send_async_batch: items must be non-empty")
        local_descs = self._agent.get_xfer_descs(
            [(int(lp), int(sz), 0) for (lp, sz, _rp, _rid, _h) in items],
            mem_type="DRAM",
        )
        remote_descs = self._agent.get_xfer_descs(
            [(int(rp), int(sz), 0) for (_lp, sz, rp, _rid, _h) in items],
            mem_type="DRAM",
        )
        if not self._agent.check_remote_metadata(peer.nixl_name, remote_descs):
            raise TransportClosed(
                f"remote metadata for {peer.nixl_name} doesn't cover one or "
                f"more of {len(items)} remote regions"
            )
        records = [(rid, h) for (_lp, _sz, _rp, rid, h) in items]
        notif = _encode_batch_notif(records)
        try:
            handle = self._agent.initialize_xfer(
                "WRITE",
                local_descs,
                remote_descs,
                peer.nixl_name,
                notif_msg=notif,
            )
        except Exception as exc:
            raise TransportClosed(
                f"initialize_xfer (batch, n={len(items)}) to "
                f"{peer.nixl_name} failed: {exc}",
            ) from exc
        try:
            self._agent.transfer(handle)
        except Exception as exc:
            try:
                self._agent.release_xfer_handle(handle)
            except Exception:
                pass
            raise TransportClosed(
                f"transfer() submission (batch) to {peer.nixl_name} " f"failed: {exc}",
            ) from exc
        rec = _AsyncSend(
            handle=handle,
            peer_name=peer.nixl_name,
            reservation_id=f"batch:{len(items)}",
            deadline_monotonic=time.monotonic() + float(timeout_s),
            on_complete=on_complete,
        )
        hid = id(handle)
        with self._async_lock:
            self._async_pending[hid] = rec
        return hid

    def _finalize_async(
        self,
        hid: int,
        rec: _AsyncSend,
        success: bool,
        err_msg: str,
    ) -> None:
        with self._async_lock:
            if hid not in self._async_pending:
                return  # already finalized
            del self._async_pending[hid]
        try:
            self._agent.release_xfer_handle(rec.handle)
        except Exception:
            logger.exception(
                "[NixlTransport async] release_xfer_handle failed for rid=%s",
                rec.reservation_id,
            )
        if rec.on_complete is not None:
            try:
                rec.on_complete(success, err_msg)
            except Exception:
                logger.exception(
                    "[NixlTransport async] on_complete callback raised " "for rid=%s",
                    rec.reservation_id,
                )

    # ----- inbound notification drain -----

    def _notif_loop(self) -> None:
        while not self._stop.is_set():
            try:
                notifs = self._agent.get_new_notifs()
            except Exception:
                logger.exception(
                    "[NixlTransport] get_new_notifs failed; sleeping",
                )
                self._stop.wait(self._poll_s)
                continue
            if not notifs:
                self._stop.wait(self._poll_s)
                continue
            for peer_name, msgs in notifs.items():
                pn = peer_name.decode() if isinstance(peer_name, bytes) else peer_name
                for msg in msgs:
                    try:
                        records = _decode_batch_notif(msg)
                    except Exception:
                        logger.exception(
                            "[NixlTransport] notif decode failed (peer=%s, %d bytes)",
                            pn,
                            len(msg),
                        )
                        continue
                    cb = self._on_inbound_notif
                    if cb is None:
                        logger.warning(
                            "[NixlTransport] received notif but no callback "
                            "registered; dropping %d record(s)",
                            len(records),
                        )
                        continue
                    for rid, content_hash in records:
                        try:
                            cb(pn, rid, content_hash)
                        except Exception:
                            logger.exception(
                                "[NixlTransport] inbound notif callback raised",
                            )

    # ----- lifecycle -----

    def agent_name(self) -> str:
        return self._agent_name

    def listen_port(self) -> int:
        return self._listen_port

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        try:
            self._completion_thread.join(timeout=2.0)
        except Exception:
            pass
        # If shutdown raced with submitted async transfers, release any
        # handles the completion loop did not get to finalize.
        try:
            with self._async_lock:
                pending = list(self._async_pending.items())
                self._async_pending.clear()
            for _hid, rec in pending:
                try:
                    self._agent.release_xfer_handle(rec.handle)
                except Exception:
                    logger.exception(
                        "[NixlTransport] close() release_xfer_handle failed "
                        "for rid=%s",
                        rec.reservation_id,
                    )
                if rec.on_complete is not None:
                    try:
                        rec.on_complete(False, "transport closed")
                    except Exception:
                        logger.exception(
                            "[NixlTransport] close() on_complete callback "
                            "raised for rid=%s",
                            rec.reservation_id,
                        )
        except Exception:
            logger.exception("[NixlTransport] close() async cleanup")
        try:
            self._notif_thread.join(timeout=2.0)
        except Exception:
            pass
        # NIXL agent cleans up its own state when garbage-collected;
        # we drop the reference here. UCX backend teardown happens
        # via NIXL's internal lifecycle.
        try:
            with self._lock:
                for name in list(self._peers.keys()):
                    try:
                        self._agent.remove_remote_agent(name)
                    except Exception:
                        pass
                self._peers.clear()
        except Exception:
            logger.exception("[NixlTransport] close() peer cleanup")
        self._agent = None
