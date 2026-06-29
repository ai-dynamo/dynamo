# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport and bootstrap RPC handlers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from gms_kv_ring.daemon.rpc_types import Handler, Message, Response

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gms_kv_ring.daemon.kv_cache_manager import GmsKvCacheManager


def handle_transport_info(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    # Sender uses this to learn the receiver's NIXL agent
    # name + listen port for metadata exchange via
    # send_local_metadata/fetch_remote_metadata. Returns
    # zeros when transport isn't enabled.
    if daemon.transport is None:
        return {
            "ok": True,
            "agent_name": "",
            "listen_port": 0,
            "receive_buffer_bytes": 0,
        }
    return {
        "ok": True,
        "agent_name": daemon.transport.agent_name(),
        "listen_port": daemon.transport.listen_port(),
        "receive_buffer_bytes": (
            daemon.staging_receive_buffer.capacity()
            if daemon.staging_receive_buffer
            else 0
        ),
    }


def handle_transport_add_peer(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    # Out-of-band peer registration. Router calls this on
    # each daemon to teach it about its peers' NIXL agent
    # names + IPs + ports. Uses send_local_metadata +
    # fetch_remote_metadata internally.
    if daemon.transport is None:
        return {"ok": False, "error": "transport not enabled"}
    try:
        daemon.transport.add_peer_by_name(
            nixl_name=str(msg["nixl_name"]),
            ip_addr=str(msg["ip_addr"]),
            port=int(msg["port"]),
            label=str(msg.get("label", "")),
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


def handle_transfer_blocks_batch(daemon: "GmsKvCacheManager", msg: Message) -> Response:
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
    if daemon.transport is None or daemon.staging_send_buffer is None:
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
        with daemon._content_hash_lock:
            entry = daemon._content_hash_index.get(content_hash)
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
            lease = daemon.host_tier.pin(engine_id, layer, offset)
            if lease is None:
                resolve_errors.append(
                    f"host_tier missing slot ({engine_id}, {layer}, {offset})"
                )
                any_missing = True
                break
            with lease as slot:
                view = (_ct.c_ubyte * size).from_address(slot.host_ptr)
                parts.append(bytes(view))
        if any_missing:
            continue
        payload = b"".join(parts)
        send_offset = daemon.staging_send_buffer.alloc(total_size)
        if send_offset is None:
            resolve_errors.append(
                "send buffer out of capacity",
            )
            continue
        src_ptr = daemon.staging_send_buffer.ptr_at(send_offset)
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
    send_buffer = daemon.staging_send_buffer

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
        daemon.transport.send_async_batch(
            peer=peer,
            items=xfer_items,
            timeout_s=timeout_s,
            on_complete=_on_done,
        )
    except (TransportClosed, RuntimeError) as exc:
        for off, sz in send_offsets:
            daemon.staging_send_buffer.free(off, sz)
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


def handle_read_bootstrap_into_staging(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    return daemon._read_bootstrap_into_staging(msg)


def handle_fetch_remote(daemon: "GmsKvCacheManager", msg: Message) -> Response:
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
        daemon.staging_tier is None
        or daemon.staging_receive_buffer is None
        or daemon.transport is None
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
    local_nixl_name = daemon.transport.agent_name()
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
    rresults = daemon.staging_tier.reserve_or_wait_many(
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
    offsets = daemon.staging_receive_buffer.alloc_many(
        [bytes_per_hash] * len(reservation_idxs),
    )
    # Build batch_items and active_xfers map under one lock.
    new_xfers: dict = {}
    for k, idx in enumerate(reservation_idxs):
        offset = offsets[k]
        rresult = rresults[idx]
        if offset is None:
            daemon.staging_tier.fail_reservation(
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
        remote_ptr = daemon.staging_receive_buffer.ptr_at(offset)
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
        with daemon._xfers_lock:
            daemon._active_xfers.update(new_xfers)
    # Split into batches of `batch_size` and dispatch them
    # CONCURRENTLY to the source via a connection pool.
    # Each batch = one transfer_blocks_batch RPC = one
    # multi-region NIXL xfer = one batched notif on dest.
    # Pipelining means multiple xfers in flight at once
    # to the same source — same model as Dynamo's engine
    # connector firing N NIXL READs in parallel for
    # per-layer pipelining.
    if batch_items:
        pool = daemon._get_source_pool(src_uds)
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
                    for rid, off, _h in local_state[base : start_idx + len(chunk)]:
                        with daemon._xfers_lock:
                            daemon._active_xfers.pop(rid, None)
                        daemon.staging_receive_buffer.free(off, bytes_per_hash)
                        daemon.staging_tier.fail_reservation(
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
            daemon._drop_source_pool(src_uds)
            for rid, off, _h in local_state:
                with daemon._xfers_lock:
                    if daemon._active_xfers.pop(rid, None) is None:
                        continue  # already drained
                daemon.staging_receive_buffer.free(off, bytes_per_hash)
                daemon.staging_tier.fail_reservation(
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


def handle_register_bootstrap_handle(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    # Engine-direct path: engine connector tells us
    # "these hashes live at these (ptr, size) regions in
    # memory I want exposed over NIXL". The daemon
    # NIXL-registers the regions (idempotent) and stores
    # the mapping. A peer daemon's get_bootstrap_info call
    # for any of these hashes returns this daemon's NIXL
    # agent name + the descriptors, letting the peer
    # engine NIXL-read directly. No daemon involvement
    # in the wire path.
    if daemon.transport is None:
        return {
            "ok": False,
            "error": "transport not enabled",
        }
    items = msg.get("items") or []
    if not items:
        return {"ok": False, "error": "items must be non-empty"}
    registered = 0
    placement_events: list[tuple[bytes, int, Optional[dict]]] = []
    with daemon._bootstrap_lock:
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
                generation = None if generation_raw is None else int(generation_raw)
            except (KeyError, ValueError, TypeError):
                continue
            if not sealed:
                daemon._bootstrap_handles.pop(content_hash, None)
                continue
            # Idempotent NIXL register; transport caches.
            try:
                daemon.transport.register_buffer(
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
            daemon._bootstrap_handles[content_hash] = entry
            descriptor = daemon._hbm_descriptor(ptr, size, generation)
            placement_events.append(
                (
                    content_hash,
                    size,
                    daemon._gms_source_metadata(descriptor),
                )
            )
            registered += 1
    if daemon.placement_publisher is not None:
        for content_hash, size, metadata in placement_events:
            try:
                daemon.placement_publisher.publish_stored(
                    content_hash=content_hash,
                    tier="hbm",
                    bytes_size=size,
                    metadata=metadata,
                )
            except TypeError:
                daemon.placement_publisher.publish_stored(
                    content_hash=content_hash,
                    tier="hbm",
                    bytes_size=size,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[Daemon] publish_stored failed for bootstrap handle",
                )
    return {"ok": True, "registered": registered}


def handle_get_bootstrap_info(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    # Engine on the decode side asks "where can I NIXL-read
    # these hashes?". We return our NIXL agent name +
    # listen port + descriptors per hash. A descriptor keeps
    # legacy top-level ptr/size/tier fields and, for
    # multi-layer host blocks, also carries a ranges[] vector.
    if daemon.transport is None:
        return {"ok": False, "error": "transport not enabled"}
    hashes_hex = msg.get("hashes") or []
    with daemon._bootstrap_lock:
        snapshot = dict(daemon._bootstrap_handles)
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
        with daemon._content_hash_lock:
            ca = daemon._content_hash_index.get(content_hash)
        if ca is not None and ca.get("sealed", True):
            engine_id = ca["engine_id"]
            ranges = ca["ranges"]
            regions = []
            missing = False
            total_size = 0
            for layer, offset, size in ranges:
                lease = daemon.host_tier.pin(
                    engine_id,
                    layer,
                    offset,
                )
                if lease is None:
                    missing = True
                    break
                try:
                    with lease as slot:
                        daemon.transport.register_buffer(
                            slot.host_ptr,
                            int(size),
                            label=(f"host:{content_hash.hex()[:8]}:{int(layer)}"),
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
        "nixl_agent_name": daemon.transport.agent_name(),
        "listen_port": daemon.transport.listen_port(),
        "agent_metadata_b64": daemon.transport._agent.get_agent_metadata().hex(),
        "descriptors": descriptors,
    }


HANDLERS: dict[str, Handler] = {
    "fetch_remote": handle_fetch_remote,
    "get_bootstrap_info": handle_get_bootstrap_info,
    "read_bootstrap_into_staging": handle_read_bootstrap_into_staging,
    "register_bootstrap_handle": handle_register_bootstrap_handle,
    "transfer_blocks_batch": handle_transfer_blocks_batch,
    "transport_add_peer": handle_transport_add_peer,
    "transport_info": handle_transport_info,
}
