# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS KV ring daemon — server + ring consumers.

One daemon process serves N engines. Control plane is a tiny Unix
socket protocol (msgpack-free, JSON-only for simplicity):

  attach_engine_pool(engine_id, layers[]):
      Caller has already CUDA-mapped the engine's HBM pool. Daemon
      records the per-layer (VA, size, stride) so consumers know
      where to DMA into. Real engine integration (cuMemImport via
      VMM-IPC shareable handles) is the engine hook's job; for tests
      and same-process use, pass pre-mapped VAs directly.

  attach_evict_ring(engine_id, ring_path):
      Spawn an evict-ring consumer for this engine. Records pushed
      to `ring_path` become D2H copies into the daemon's host tier.

  attach_restore_ring(engine_id, ring_path, counter_path, num_counters):
      Spawn a restore-ring consumer. Records become H2D copies +
      cuStreamWriteValue32 signal on the shared counter array.

  detach_engine_pool(engine_id):
      Stop consumers, free host-tier slots.

  ping():
      Liveness check.

Each engine has its OWN dedicated CUDA stream in the daemon process.
Engines do NOT serialize on each other — only on the asyncio control
loop, which is only used for setup/teardown.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
from typing import Optional

from gms_kv_ring.daemon.consumers import EnginePool, LayerDesc
from gms_kv_ring.daemon.kv_cache_manager import GmsKvCacheManager
from gms_kv_ring.daemon.rpc_content import HANDLERS as CONTENT_HANDLERS
from gms_kv_ring.daemon.rpc_lifecycle import HANDLERS as LIFECYCLE_HANDLERS

logger = logging.getLogger(__name__)

__all__ = ["Daemon", "EnginePool", "LayerDesc"]


class Daemon:
    """Unix-socket process shell for :class:`GmsKvCacheManager`."""

    def __init__(self, listen_socket: str, storage_dir=None, **manager_kwargs) -> None:
        self.listen_socket = listen_socket
        self.kv_cache_manager = GmsKvCacheManager(
            storage_dir=storage_dir,
            **manager_kwargs,
        )
        self._server: Optional[asyncio.AbstractServer] = None
        self._stop_event = None

    def __getattr__(self, name):
        # Preserve the established Daemon surface while ownership moves to
        # the dedicated manager. New code should use ``kv_cache_manager``.
        return getattr(self.kv_cache_manager, name)

    @property
    def transport(self):
        return self.kv_cache_manager.transport

    @transport.setter
    def transport(self, value) -> None:
        self.kv_cache_manager.transport = value

    @property
    def placement_publisher(self):
        return self.kv_cache_manager.placement_publisher

    @placement_publisher.setter
    def placement_publisher(self, value) -> None:
        self.kv_cache_manager.placement_publisher = value

    async def serve(self) -> None:
        try:
            os.unlink(self.listen_socket)
        except FileNotFoundError:
            pass
        self._stop_event = asyncio.Event()
        self._server = await asyncio.start_unix_server(
            self._handle,
            path=self.listen_socket,
        )
        logger.info("daemon listening on %s", self.listen_socket)
        self.kv_cache_manager.start()
        try:
            await self._stop_event.wait()
        finally:
            self._server.close()
            await self._server.wait_closed()
            self.kv_cache_manager.close()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        self.kv_cache_manager.close_connections()

    async def _handle(self, reader, writer) -> None:
        try:
            while True:
                msg = await _read_frame(reader)
                if msg is None:
                    return
                resp = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._dispatch,
                    msg,
                )
                # Stamp every response with the daemon epoch. The
                # connector reads this on each RPC; a change means
                # the daemon restarted and its in-memory state
                # (slot_generations, host_tier slots, etc.) has been
                # zeroed — the connector must drop its prefix index
                # to avoid restoring against slots the new daemon
                # doesn't know about.
                if isinstance(resp, dict):
                    resp["daemon_epoch"] = self.epoch
                await _write_frame(writer, resp)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    def _dispatch(self, msg: dict) -> dict:
        op = msg.get("op")
        try:
            handler = LIFECYCLE_HANDLERS.get(op)
            if handler is not None:
                return handler(self, msg)
            handler = CONTENT_HANDLERS.get(op)
            if handler is not None:
                return handler(self, msg)
            if op == "transport_info":
                # Sender uses this to learn the receiver's NIXL agent
                # name + listen port for metadata exchange via
                # send_local_metadata/fetch_remote_metadata. Returns
                # zeros when transport isn't enabled.
                if self.transport is None:
                    return {
                        "ok": True,
                        "agent_name": "",
                        "listen_port": 0,
                        "receive_buffer_bytes": 0,
                    }
                return {
                    "ok": True,
                    "agent_name": self.transport.agent_name(),
                    "listen_port": self.transport.listen_port(),
                    "receive_buffer_bytes": (
                        self.staging_receive_buffer.capacity()
                        if self.staging_receive_buffer
                        else 0
                    ),
                }
            if op == "transport_add_peer":
                # Out-of-band peer registration. Router calls this on
                # each daemon to teach it about its peers' NIXL agent
                # names + IPs + ports. Uses send_local_metadata +
                # fetch_remote_metadata internally.
                if self.transport is None:
                    return {"ok": False, "error": "transport not enabled"}
                try:
                    self.transport.add_peer_by_name(
                        nixl_name=str(msg["nixl_name"]),
                        ip_addr=str(msg["ip_addr"]),
                        port=int(msg["port"]),
                        label=str(msg.get("label", "")),
                    )
                except Exception as exc:
                    return {"ok": False, "error": str(exc)}
                return {"ok": True}
            if op == "transfer_blocks_batch":
                # Per-request batched cross-node push (matches Dynamo's
                # one-xfer-per-request shape). Caller supplies N items:
                #   { content_hash, target_reservation_id,
                #     target_remote_ptr, size }
                # We look up each hash in our host_tier registration,
                # copy bytes into the send buffer (N independent
                # offsets), and issue ONE multi-region NIXL xfer with
                # a multi-record notif. Eliminates per-WRITE
                # concurrency entirely — one logical xfer per request,
                # no queueing, no per-peer cap matters here.
                if self.transport is None or self.staging_send_buffer is None:
                    return {
                        "ok": False,
                        "error": "transport or send buffer not enabled",
                    }
                items = msg.get("items") or []
                if not items:
                    return {
                        "ok": False,
                        "error": "transfer_blocks_batch: items must be non-empty",
                    }
                target_nixl_name = str(msg["target_nixl_name"])
                target_ip = str(msg.get("target_ip", ""))
                target_port = int(msg.get("target_port", 0))
                timeout_s = float(msg.get("timeout_s", 30.0))
                import ctypes as _ct

                # Prepare per-item: pull bytes from host_tier, copy
                # into send buffer.
                xfer_items: list = []  # (local_ptr, size, remote_ptr, rid, hash)
                send_offsets: list = []  # (offset, size) for rollback/free
                resolve_errors: list = []
                for it in items:
                    try:
                        content_hash = bytes.fromhex(str(it["content_hash"]))
                    except Exception:
                        resolve_errors.append("invalid content_hash")
                        continue
                    target_rid = str(it["target_reservation_id"])
                    target_ptr = int(it["target_remote_ptr"])
                    with self._content_hash_lock:
                        entry = self._content_hash_index.get(content_hash)
                    if entry is None:
                        resolve_errors.append(
                            f"hash not registered: {content_hash.hex()[:16]}",
                        )
                        continue
                    if not entry.get("sealed", True):
                        resolve_errors.append(
                            f"hash not sealed: {content_hash.hex()[:16]}",
                        )
                        continue
                    engine_id = entry["engine_id"]
                    ranges = entry["ranges"]
                    total_size = sum(sz for _, _, sz in ranges)
                    declared_size = int(it.get("size", total_size))
                    if declared_size != total_size:
                        resolve_errors.append(
                            f"size mismatch for hash "
                            f"{content_hash.hex()[:16]}: "
                            f"declared={declared_size} actual={total_size}"
                        )
                        continue
                    parts: list[bytes] = []
                    any_missing = False
                    for layer, offset, size in ranges:
                        lease = self.host_tier.pin(engine_id, layer, offset)
                        if lease is None:
                            resolve_errors.append(
                                f"host_tier missing slot "
                                f"({engine_id}, {layer}, {offset})"
                            )
                            any_missing = True
                            break
                        with lease as slot:
                            view = (_ct.c_ubyte * size).from_address(slot.host_ptr)
                            parts.append(bytes(view))
                    if any_missing:
                        continue
                    payload = b"".join(parts)
                    send_offset = self.staging_send_buffer.alloc(total_size)
                    if send_offset is None:
                        resolve_errors.append(
                            "send buffer out of capacity",
                        )
                        continue
                    src_ptr = self.staging_send_buffer.ptr_at(send_offset)
                    _ct.memmove(src_ptr, payload, total_size)
                    xfer_items.append(
                        (src_ptr, total_size, target_ptr, target_rid, content_hash),
                    )
                    send_offsets.append((send_offset, total_size))
                if not xfer_items:
                    return {
                        "ok": False,
                        "error": "transfer_blocks_batch: no submittable items",
                        "resolve_errors": resolve_errors[:5],
                    }
                # Build PeerHandle.
                from gms_kv_ring.daemon.transport import PeerHandle, TransportClosed

                peer = PeerHandle(
                    nixl_name=target_nixl_name,
                    ip_addr=target_ip,
                    port=target_port,
                )
                send_buffer = self.staging_send_buffer

                # Capture send_offsets by value (Python closures are
                # late-binding by name).
                def _on_done(
                    success: bool,
                    err: str,
                    _offsets: list = list(send_offsets),
                    _peer_name: str = peer.nixl_name,
                    _n: int = len(xfer_items),
                ) -> None:
                    for off, sz in _offsets:
                        try:
                            send_buffer.free(off, sz)
                        except Exception:  # noqa: BLE001
                            logger.exception(
                                "[Daemon] batch send: free failed (n=%d)",
                                _n,
                            )
                    if not success:
                        logger.warning(
                            "[Daemon] batch send to %s (n=%d) failed: %s",
                            _peer_name,
                            _n,
                            err,
                        )

                try:
                    self.transport.send_async_batch(
                        peer=peer,
                        items=xfer_items,
                        timeout_s=timeout_s,
                        on_complete=_on_done,
                    )
                except (TransportClosed, RuntimeError) as exc:
                    for off, sz in send_offsets:
                        self.staging_send_buffer.free(off, sz)
                    return {"ok": False, "error": str(exc)}
                resp: dict = {
                    "ok": True,
                    "accepted": True,
                    "submitted": len(xfer_items),
                    "bytes_sent": sum(sz for _, sz, _, _, _ in xfer_items),
                }
                if resolve_errors:
                    resp["resolve_errors"] = resolve_errors[:5]
                    resp["resolve_error_count"] = len(resolve_errors)
                return resp
            if op == "read_bootstrap_into_staging":
                return self._read_bootstrap_into_staging(msg)

            if op == "fetch_remote":
                # Pattern C orchestration with multi-stage batched NIXL
                # xfers (matches Dynamo's per-layer pipelining shape).
                #
                # Splits N hashes into ⌈N/batch_size⌉ batches. Each
                # batch is ONE NIXL xfer carrying ≤batch_size WRITEs
                # plus one multi-record notif. Decode-side engine sees
                # per-block PlacementEvent::Stored as each batch lands
                # (because the dest decodes the multi-record notif
                # into N separate events). This gives decode-side
                # incremental visibility without per-WRITE NIXL
                # concurrency issues.
                #
                # Tuning:
                #   batch_size = N           → one big xfer; latency =
                #     slowest block; highest wire efficiency
                #   batch_size = layer_size  → Dynamo-style per-layer
                #     pipelining; decode starts attention on layer N
                #     while layer N+1 is still in flight
                #   batch_size = 1           → per-block xfer; max
                #     pipelining (subject to NIXL per-peer cap)
                if (
                    self.staging_tier is None
                    or self.staging_receive_buffer is None
                    or self.transport is None
                ):
                    return {
                        "ok": False,
                        "error": "fetch_remote: staging/transport not enabled",
                    }
                src_uds = str(msg["source_uds_path"])
                src_nixl_name = str(msg["source_nixl_name"])
                src_ip = str(msg.get("source_ip", "127.0.0.1"))
                src_port = int(msg.get("source_port", 0))
                hashes_hex = msg["hashes"]
                bytes_per_hash = int(msg["bytes_per_hash"])
                timeout_s = float(msg.get("timeout_s", 30.0))
                # Default batch_size = full request (one xfer). Caller
                # can set lower for per-stage pipelining.
                batch_size = int(msg.get("batch_size", len(hashes_hex) or 1))
                if batch_size <= 0:
                    batch_size = len(hashes_hex) or 1
                local_nixl_name = self.transport.agent_name()
                from gms_kv_ring.daemon.staging_tier import (
                    AlreadyReady,
                    Rejected,
                    Reservation,
                    Waiter,
                )

                # Local reservation pass — vectorized: one batched
                # reserve_or_wait_many + alloc_many call instead of N
                # per-block Python loops. This is the optimization that
                # closes most of the GMS-vs-Dynamo benchmark gap.
                accepted = 0
                already_ready = 0
                coalesced = 0
                failed = 0
                batch_items: list = []
                local_state: list = []
                hash_bytes_list = [bytes.fromhex(h) for h in hashes_hex]
                rresults = self.staging_tier.reserve_or_wait_many(
                    hash_bytes_list,
                    src_nixl_name,
                )
                # Collect indexes of items that got Reservation (need
                # receive-buffer allocation) and tally non-Reservation
                # outcomes in a single pass.
                reservation_idxs: list = []
                for idx, rresult in enumerate(rresults):
                    if isinstance(rresult, AlreadyReady):
                        already_ready += 1
                    elif isinstance(rresult, Waiter):
                        coalesced += 1
                    elif isinstance(rresult, Rejected):
                        failed += 1
                    else:
                        assert isinstance(rresult, Reservation)
                        reservation_idxs.append(idx)
                # Vectorized receive buffer allocation.
                offsets = self.staging_receive_buffer.alloc_many(
                    [bytes_per_hash] * len(reservation_idxs),
                )
                # Build batch_items and active_xfers map under one lock.
                new_xfers: dict = {}
                for k, idx in enumerate(reservation_idxs):
                    offset = offsets[k]
                    rresult = rresults[idx]
                    if offset is None:
                        self.staging_tier.fail_reservation(
                            rresult.reservation_id,
                            "recv buf full",
                        )
                        failed += 1
                        continue
                    content_hash = hash_bytes_list[idx]
                    h_hex = hashes_hex[idx]
                    new_xfers[rresult.reservation_id] = (
                        offset,
                        bytes_per_hash,
                        content_hash,
                    )
                    remote_ptr = self.staging_receive_buffer.ptr_at(offset)
                    batch_items.append(
                        {
                            "content_hash": h_hex,
                            "target_reservation_id": rresult.reservation_id,
                            "target_remote_ptr": remote_ptr,
                            "size": bytes_per_hash,
                        }
                    )
                    local_state.append(
                        (rresult.reservation_id, offset, content_hash),
                    )
                if new_xfers:
                    with self._xfers_lock:
                        self._active_xfers.update(new_xfers)
                # Split into batches of `batch_size` and dispatch them
                # CONCURRENTLY to the source via a connection pool.
                # Each batch = one transfer_blocks_batch RPC = one
                # multi-region NIXL xfer = one batched notif on dest.
                # Pipelining means multiple xfers in flight at once
                # to the same source — same model as Dynamo's engine
                # connector firing N NIXL READs in parallel for
                # per-layer pipelining.
                if batch_items:
                    pool = self._get_source_pool(src_uds)
                    chunks: list = []
                    for start in range(0, len(batch_items), batch_size):
                        chunks.append((start, batch_items[start : start + batch_size]))

                    from concurrent.futures import ThreadPoolExecutor

                    def _submit_chunk(start_chunk):
                        start_idx, chunk = start_chunk
                        client = pool.get()
                        return (
                            start_idx,
                            chunk,
                            client._ok(
                                {
                                    "op": "transfer_blocks_batch",
                                    "target_nixl_name": local_nixl_name,
                                    "target_ip": src_ip,
                                    "target_port": src_port,
                                    "timeout_s": timeout_s,
                                    "items": chunk,
                                }
                            ),
                        )

                    try:
                        # Workers = pool size (each socket can carry
                        # one in-flight RPC at a time). More workers
                        # than sockets gives no extra parallelism;
                        # fewer queues batches at the pool entry.
                        with ThreadPoolExecutor(
                            max_workers=len(pool._clients),
                        ) as ex:
                            results = list(ex.map(_submit_chunk, chunks))
                        for start_idx, chunk, resp in results:
                            submitted = int(resp.get("submitted", 0))
                            accepted += submitted
                            unsubmitted = len(chunk) - submitted
                            if unsubmitted > 0:
                                base = start_idx + submitted
                                for rid, off, _h in local_state[
                                    base : start_idx + len(chunk)
                                ]:
                                    with self._xfers_lock:
                                        self._active_xfers.pop(rid, None)
                                    self.staging_receive_buffer.free(
                                        off, bytes_per_hash
                                    )
                                    self.staging_tier.fail_reservation(
                                        rid,
                                        "source resolve failed",
                                    )
                                failed += unsubmitted
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[Daemon] fetch_remote: batch RPC to %s failed: %s",
                            src_uds,
                            exc,
                        )
                        # Source might be dead; drop the cached pool
                        # so the next call retries with fresh sockets.
                        self._drop_source_pool(src_uds)
                        for rid, off, _h in local_state:
                            with self._xfers_lock:
                                if self._active_xfers.pop(rid, None) is None:
                                    continue  # already drained
                            self.staging_receive_buffer.free(off, bytes_per_hash)
                            self.staging_tier.fail_reservation(
                                rid,
                                "source batch RPC failed",
                            )
                        failed += len(batch_items) - accepted
                        accepted = 0
                return {
                    "ok": True,
                    "accepted": accepted,
                    "already_ready": already_ready,
                    "coalesced": coalesced,
                    "failed": failed,
                }
            if op == "register_bootstrap_handle":
                # Engine-direct path: engine connector tells us
                # "these hashes live at these (ptr, size) regions in
                # memory I want exposed over NIXL". The daemon
                # NIXL-registers the regions (idempotent) and stores
                # the mapping. A peer daemon's get_bootstrap_info call
                # for any of these hashes returns this daemon's NIXL
                # agent name + the descriptors, letting the peer
                # engine NIXL-read directly. No daemon involvement
                # in the wire path.
                if self.transport is None:
                    return {
                        "ok": False,
                        "error": "transport not enabled",
                    }
                items = msg.get("items") or []
                if not items:
                    return {"ok": False, "error": "items must be non-empty"}
                registered = 0
                placement_events: list[tuple[bytes, int, Optional[dict]]] = []
                with self._bootstrap_lock:
                    for it in items:
                        try:
                            content_hash = bytes.fromhex(str(it["content_hash"]))
                            ptr = int(it["ptr"])
                            size = int(it["size"])
                            sealed_raw = it.get("sealed", True)
                            if isinstance(sealed_raw, str):
                                sealed = sealed_raw.lower() not in (
                                    "0",
                                    "false",
                                    "no",
                                    "off",
                                    "",
                                )
                            else:
                                sealed = bool(sealed_raw)
                            generation_raw = it.get("generation")
                            generation = (
                                None if generation_raw is None else int(generation_raw)
                            )
                        except (KeyError, ValueError, TypeError):
                            continue
                        if not sealed:
                            self._bootstrap_handles.pop(content_hash, None)
                            continue
                        # Idempotent NIXL register; transport caches.
                        try:
                            self.transport.register_buffer(
                                ptr,
                                size,
                                label=f"bs:{content_hash.hex()[:8]}",
                            )
                        except Exception:  # noqa: BLE001
                            logger.exception(
                                "[Daemon] register_bootstrap_handle: NIXL register failed"
                            )
                            continue
                        entry = {"ptr": ptr, "size": size, "sealed": True}
                        if generation is not None:
                            entry["generation"] = generation
                        self._bootstrap_handles[content_hash] = entry
                        descriptor = self._hbm_descriptor(ptr, size, generation)
                        placement_events.append(
                            (
                                content_hash,
                                size,
                                self._gms_source_metadata(descriptor),
                            )
                        )
                        registered += 1
                if self.placement_publisher is not None:
                    for content_hash, size, metadata in placement_events:
                        try:
                            self.placement_publisher.publish_stored(
                                content_hash=content_hash,
                                tier="hbm",
                                bytes_size=size,
                                metadata=metadata,
                            )
                        except TypeError:
                            self.placement_publisher.publish_stored(
                                content_hash=content_hash,
                                tier="hbm",
                                bytes_size=size,
                            )
                        except Exception:  # noqa: BLE001
                            logger.exception(
                                "[Daemon] publish_stored failed for bootstrap handle",
                            )
                return {"ok": True, "registered": registered}

            if op == "get_bootstrap_info":
                # Engine on the decode side asks "where can I NIXL-read
                # these hashes?". We return our NIXL agent name +
                # listen port + descriptors per hash. A descriptor keeps
                # legacy top-level ptr/size/tier fields and, for
                # multi-layer host blocks, also carries a ranges[] vector.
                if self.transport is None:
                    return {"ok": False, "error": "transport not enabled"}
                hashes_hex = msg.get("hashes") or []
                with self._bootstrap_lock:
                    snapshot = dict(self._bootstrap_handles)
                descriptors = []
                for h_hex in hashes_hex:
                    try:
                        content_hash = bytes.fromhex(str(h_hex))
                    except ValueError:
                        descriptors.append(None)
                        continue
                    entry = snapshot.get(content_hash)
                    if entry is not None:
                        if isinstance(entry, dict):
                            ptr = int(entry["ptr"])
                            size = int(entry["size"])
                            generation = entry.get("generation")
                        else:
                            ptr, size = entry
                            generation = None
                        descriptor = {
                            "ptr": ptr,
                            "size": int(size),
                            "tier": "hbm",
                            "ranges": [
                                {
                                    "ptr": ptr,
                                    "size": int(size),
                                    "tier": "hbm",
                                }
                            ],
                            "sealed": True,
                        }
                        if generation is not None:
                            descriptor["generation"] = int(generation)
                        descriptors.append(descriptor)
                        continue
                    # Fallback: host_tier (was advertised via
                    # register_content_address — daemon has the ranges).
                    with self._content_hash_lock:
                        ca = self._content_hash_index.get(content_hash)
                    if ca is not None and ca.get("sealed", True):
                        engine_id = ca["engine_id"]
                        ranges = ca["ranges"]
                        regions = []
                        missing = False
                        total_size = 0
                        for layer, offset, size in ranges:
                            lease = self.host_tier.pin(
                                engine_id,
                                layer,
                                offset,
                            )
                            if lease is None:
                                missing = True
                                break
                            try:
                                with lease as slot:
                                    self.transport.register_buffer(
                                        slot.host_ptr,
                                        int(size),
                                        label=(
                                            f"host:{content_hash.hex()[:8]}:"
                                            f"{int(layer)}"
                                        ),
                                    )
                                    regions.append(
                                        {
                                            "ptr": slot.host_ptr,
                                            "size": int(size),
                                            "tier": "host",
                                            "layer": int(layer),
                                            "offset": int(offset),
                                        }
                                    )
                                    total_size += int(size)
                            except Exception:  # noqa: BLE001
                                logger.exception(
                                    "[Daemon] get_bootstrap_info: host "
                                    "range NIXL registration failed",
                                )
                                missing = True
                                break
                        if not missing and regions:
                            descriptor = {
                                "ptr": int(regions[0]["ptr"]),
                                "size": int(total_size),
                                "tier": "host",
                                "ranges": regions,
                                "sealed": True,
                            }
                            if ca.get("generation") is not None:
                                descriptor["generation"] = int(ca["generation"])
                            descriptors.append(descriptor)
                            continue
                    descriptors.append(None)
                return {
                    "ok": True,
                    "nixl_agent_name": self.transport.agent_name(),
                    "listen_port": self.transport.listen_port(),
                    "agent_metadata_b64": self.transport._agent.get_agent_metadata().hex(),
                    "descriptors": descriptors,
                }

            return {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ---- minimal length-prefixed JSON framing ----


async def _read_frame(reader) -> Optional[dict]:
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        return None
    n = struct.unpack("<I", header)[0]
    body = await reader.readexactly(n)
    return json.loads(body.decode("utf-8"))


async def _write_frame(writer, msg: dict) -> None:
    body = json.dumps(msg).encode("utf-8")
    writer.write(struct.pack("<I", len(body)) + body)
    await writer.drain()
