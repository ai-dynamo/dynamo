# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-address, staging, and restore RPC handlers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from gms_kv_ring.daemon.rpc_types import (
    Handler,
    Message,
    Response,
    required_digest,
    required_int,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gms_kv_ring.daemon.kv_cache_manager import GmsKvCacheManager


def handle_staging_reserve(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    # Receiver-side RPC for cross-node transfer (Phase 4b).
    # Router calls this on the DESTINATION daemon to
    # pre-allocate a slot + register a write target before
    # the SOURCE daemon issues the NIXL WRITE.
    #
    # Returns (reservation_id, remote_ptr). Sender uses
    # remote_ptr in `initialize_xfer(..., remote_descs)`
    # and reservation_id+content_hash in the notif payload.
    if daemon.staging_tier is None or daemon.staging_receive_buffer is None:
        return {
            "ok": False,
            "error": "staging not enabled",
        }
    content_hash = required_digest(msg)
    size = required_int(msg, "size")
    source_daemon = str(msg.get("source_daemon", "unknown"))
    # Step 1: reserve the StagingTier slot. May coalesce
    # if another peer is already delivering this hash.
    from gms_kv_ring.daemon.staging_tier import (
        AlreadyReady,
        Rejected,
        Reservation,
        Waiter,
    )

    result = daemon.staging_tier.reserve_or_wait(
        content_hash,
        source_daemon,
    )
    if isinstance(result, AlreadyReady):
        return {
            "ok": True,
            "outcome": "already_ready",
            "bytes_size": result.hit.bytes_size,
            "generation": result.hit.generation,
        }
    if isinstance(result, Waiter):
        # Another transfer is in flight. Sender skips.
        # (Real waiter mechanics require a follow-up
        # event channel for cross-process notification;
        # current contract: router retries with backoff.)
        return {
            "ok": True,
            "outcome": "coalesced",
        }
    if isinstance(result, Rejected):
        return {
            "ok": False,
            "error": f"rejected: {result.reason}",
        }
    # Reservation — allocate receive buffer offset
    assert isinstance(result, Reservation)
    offset = daemon.staging_receive_buffer.alloc(size)
    if offset is None:
        # Out of receive-buffer capacity; release the
        # StagingTier reservation we just made.
        daemon.staging_tier.fail_reservation(
            result.reservation_id,
            "receive buffer full",
        )
        return {
            "ok": False,
            "error": "receive buffer out of capacity",
        }
    with daemon._xfers_lock:
        daemon._active_xfers[result.reservation_id] = (
            offset,
            size,
            content_hash,
        )
    remote_ptr = daemon.staging_receive_buffer.ptr_at(offset)
    return {
        "ok": True,
        "outcome": "reserved",
        "reservation_id": result.reservation_id,
        "remote_ptr": remote_ptr,
    }


def handle_staging_fail(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    # Cleanup on transfer failure. Frees the receive
    # buffer offset and the StagingTier reservation.
    rid = str(msg["reservation_id"])
    reason = str(msg.get("reason", "external"))
    with daemon._xfers_lock:
        entry = daemon._active_xfers.pop(rid, None)
    if entry is not None and daemon.staging_receive_buffer:
        offset, size, _hash = entry
        daemon.staging_receive_buffer.free(offset, size)
    if daemon.staging_tier is not None:
        daemon.staging_tier.fail_reservation(rid, reason)
    return {"ok": True}


def handle_register_content_addresses_batch(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    # Batched form of register_content_address — connector
    # emits one RPC per request_finished instead of one
    # per block. Each item: {content_hash, engine_id, ranges}.
    # Optional fields: {generation, sealed, metadata}. A
    # sealed=False item is deliberately not advertised.
    items = msg.get("items", []) or []
    if daemon.transport is None:
        return {"ok": True, "total_bytes": 0, "skipped": True}
    total_bytes = 0
    unsealed = 0
    registered: list[tuple[bytes, int, Optional[dict], dict]] = []
    with daemon._content_hash_lock:
        for item in items:
            try:
                ch = bytes.fromhex(str(item["content_hash"]))
                eng = str(item["engine_id"])
                ranges = [
                    (int(r["layer"]), int(r["offset"]), int(r["size"]))
                    for r in (item.get("ranges") or [])
                ]
                sealed_raw = item.get("sealed", True)
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
                generation_raw = item.get("generation")
                generation = None if generation_raw is None else int(generation_raw)
                metadata = item.get("metadata")
                if metadata is not None and not isinstance(metadata, dict):
                    metadata = None
            except (KeyError, ValueError, TypeError) as exc:
                return {
                    "ok": False,
                    "error": f"malformed item: {exc}",
                }
            if not sealed:
                daemon._content_hash_index.pop(ch, None)
                unsealed += 1
                continue
            entry = {"engine_id": eng, "ranges": ranges}
            if generation is not None:
                entry["generation"] = generation
            if metadata is not None:
                entry["metadata"] = metadata
            daemon._content_hash_index[ch] = entry
            sz = sum(sz for _, _, sz in ranges)
            total_bytes += sz
            registered.append((ch, sz, metadata, dict(entry)))
    # Publish Stored events outside the lock to avoid
    # blocking other content-hash lookups.
    if daemon.placement_publisher is not None:
        for ch, sz, metadata, entry in registered:
            placement_metadata = daemon._content_address_placement_metadata(
                ch,
                entry,
                metadata,
            )
            try:
                daemon.placement_publisher.publish_stored(
                    content_hash=ch,
                    tier="host_pinned",
                    bytes_size=sz,
                    metadata=placement_metadata,
                )
            except TypeError:
                daemon.placement_publisher.publish_stored(
                    content_hash=ch,
                    tier="host_pinned",
                    bytes_size=sz,
                )
            except Exception:
                logger.exception(
                    "[Daemon] publish_stored failed for batch item",
                )
    return {
        "ok": True,
        "total_bytes": total_bytes,
        "skipped": False,
        "unsealed": unsealed,
    }


def handle_register_content_address(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    # Connector calls this after a successful spill to
    # advertise the content_hash → host_tier address
    # mapping. The router uses this index to drive
    # cross-node transfers (P4c). Multi-range payload
    # because one logical block spans N layers.
    content_hash = required_digest(msg)
    engine_id = str(msg["engine_id"])
    ranges_raw = msg.get("ranges", []) or []
    ranges = [(int(r["layer"]), int(r["offset"]), int(r["size"])) for r in ranges_raw]
    sealed_raw = msg.get("sealed", True)
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
    if not sealed:
        with daemon._content_hash_lock:
            daemon._content_hash_index.pop(content_hash, None)
        return {"ok": True, "total_size": 0, "unsealed": 1}
    generation_raw = msg.get("generation")
    generation = None if generation_raw is None else int(generation_raw)
    metadata = msg.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        metadata = None
    entry = {"engine_id": engine_id, "ranges": ranges}
    if generation is not None:
        entry["generation"] = generation
    if metadata is not None:
        entry["metadata"] = metadata
    with daemon._content_hash_lock:
        daemon._content_hash_index[content_hash] = entry
    total_size = sum(sz for _, _, sz in ranges)
    if daemon.placement_publisher is not None:
        placement_metadata = daemon._content_address_placement_metadata(
            content_hash,
            entry,
            metadata,
        )
        try:
            daemon.placement_publisher.publish_stored(
                content_hash=content_hash,
                tier="host_pinned",
                bytes_size=total_size,
                metadata=placement_metadata,
            )
        except TypeError:
            daemon.placement_publisher.publish_stored(
                content_hash=content_hash,
                tier="host_pinned",
                bytes_size=total_size,
            )
        except Exception:
            logger.exception(
                "[Daemon] publish_stored failed",
            )
    return {"ok": True, "total_size": total_size}


def handle_notify_kv_arrived(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    # Engine on the decode side tells its local daemon
    # "I just NIXL-read these hashes; please publish
    # PlacementEvent::Stored". Off the critical path —
    # this is for the indexer, not for the data flow.
    if daemon.placement_publisher is None:
        return {"ok": True, "published": 0}
    items = msg.get("items") or []
    published = 0
    for it in items:
        try:
            content_hash = bytes.fromhex(str(it["content_hash"]))
            size = int(it.get("size", 0))
        except (KeyError, ValueError):
            continue
        metadata = it.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            metadata = None
        try:
            daemon.placement_publisher.publish_stored(
                content_hash=content_hash,
                tier="external",
                bytes_size=size,
                metadata=metadata,
            )
            published += 1
        except Exception:  # noqa: BLE001
            logger.exception("[Daemon] notify_kv_arrived: publish_stored failed")
    return {"ok": True, "published": published}


def handle_restore_staging_ranges(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    # Blocking control-plane restore from StagingTier into
    # explicit destination ranges. Unlike restore_staging_blocks,
    # this does not assume one content hash maps to one daemon
    # block stride. SGLang uses it for page hashes made from
    # several per-token KV slots.
    engine_id = str(msg["engine_id"])
    with daemon._lock:
        pool = daemon._pools.get(engine_id)
    if pool is None:
        return {
            "ok": False,
            "error": "engine pool not attached",
        }
    if daemon.staging_tier is None:
        return {
            "ok": False,
            "error": "staging not enabled",
        }
    raw_items = msg.get("items") or []
    from cuda.bindings import driver as drv
    from gms_kv_ring.common import metrics

    dsts: list[int] = []
    srcs: list[int] = []
    sizes: list[int] = []
    consume_handles: list = []
    valid_items = 0
    success = True
    try:
        for it in raw_items:
            try:
                content_hash = bytes.fromhex(
                    str(it["content_hash"]),
                )
                generation = int(it["generation"])
                raw_ranges = it.get("ranges") or []
                ranges = [
                    (
                        int(r["layer"]),
                        int(r["offset"]),
                        int(r["size"]),
                    )
                    for r in raw_ranges
                ]
            except (KeyError, ValueError, TypeError):
                success = False
                break
            if not ranges:
                success = False
                break
            consume = daemon.staging_tier.begin_consume(
                content_hash,
                generation,
            )
            if consume is None:
                logger.warning(
                    "restore-staging-ranges: hash=%s generation=%d not READY",
                    content_hash.hex()[:16],
                    generation,
                )
                success = False
                break
            consume_handles.append(consume)
            ptr_info = daemon.staging_tier.consume_pointer(
                consume,
            )
            if ptr_info is None:
                logger.warning(
                    "restore-staging-ranges: hash=%s disappeared after pin",
                    content_hash.hex()[:16],
                )
                success = False
                break
            src_ptr, bytes_size, _crc32 = ptr_info
            cursor = 0
            for layer_idx, offset, size in ranges:
                ld = pool.layers.get(int(layer_idx))
                if ld is None:
                    logger.warning(
                        "restore-staging-ranges: unknown layer=%d",
                        int(layer_idx),
                    )
                    success = False
                    break
                offset_i = int(offset)
                size_i = int(size)
                if offset_i < 0 or size_i <= 0 or offset_i + size_i > int(ld.size):
                    logger.warning(
                        "restore-staging-ranges: invalid "
                        "range layer=%d offset=%d size=%d "
                        "layer_size=%d",
                        int(layer_idx),
                        offset_i,
                        size_i,
                        int(ld.size),
                    )
                    success = False
                    break
                if cursor + size_i > int(bytes_size):
                    logger.warning(
                        "restore-staging-ranges: payload too "
                        "small for layer=%d offset=%d "
                        "cursor=%d size=%d payload=%d",
                        int(layer_idx),
                        offset_i,
                        cursor,
                        size_i,
                        int(bytes_size),
                    )
                    success = False
                    break
                dsts.append(int(ld.va) + offset_i)
                srcs.append(int(src_ptr) + cursor)
                sizes.append(size_i)
                cursor += size_i
            if not success:
                break
            if cursor != int(bytes_size):
                logger.warning(
                    "restore-staging-ranges: payload size "
                    "mismatch for hash=%s consumed=%d "
                    "payload=%d",
                    content_hash.hex()[:16],
                    cursor,
                    int(bytes_size),
                )
                success = False
                break
            valid_items += 1
        if success and dsts:
            for d, s, sz in zip(dsts, srcs, sizes):
                drv.cuMemcpyAsync(
                    drv.CUdeviceptr(d),
                    drv.CUdeviceptr(s),
                    int(sz),
                    int(pool.stream),
                )
            metrics.restore_h2d_bytes.inc(
                engine_id=engine_id,
                n=sum(sizes),
            )
            drv.cuStreamSynchronize(int(pool.stream))
        else:
            success = False
    except Exception:  # noqa: BLE001
        logger.warning(
            "restore-staging-ranges: copy failed",
            exc_info=True,
        )
        success = False
    finally:
        for consume in consume_handles:
            try:
                daemon.staging_tier.end_consume(consume)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "restore-staging-ranges: end_consume failed",
                    exc_info=True,
                )
    return {
        "ok": True,
        "success": bool(success),
        "requested": len(raw_items),
        "restored": valid_items if success else 0,
    }


def handle_restore_host_blocks(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    engine_id = str(msg["engine_id"])
    src_engine_id = str(msg["src_engine_id"])
    with daemon._lock:
        dest_pool = daemon._pools.get(engine_id)
        src_pool = daemon._pools.get(src_engine_id)
    if dest_pool is None or src_pool is None or dest_pool.restore_consumer is None:
        return {
            "ok": False,
            "error": "engine restore consumer or source pool not attached",
        }
    raw_items = msg.get("items") or []
    block_pairs: list[tuple[int, int]] = []
    expected_generations: dict[int, int] = {}
    for it in raw_items:
        try:
            src_block = int(it["src_block"])
            dest_block = int(it["dest_block"])
            generation = int(it.get("generation", 0))
        except (KeyError, ValueError, TypeError):
            continue
        block_pairs.append((src_block, dest_block))
        if generation:
            expected_generations[src_block] = generation
    if not block_pairs:
        return {
            "ok": False,
            "error": "no valid host restore items",
        }
    success = dest_pool.restore_consumer._process_host_tier(
        {
            "src_engine_id": src_engine_id,
            "block_pairs": block_pairs,
            "expected_generations": expected_generations,
        },
        dest_pool,
        src_pool,
    )
    return {
        "ok": True,
        "success": bool(success),
        "requested": len(block_pairs),
        "restored": len(block_pairs) if success else 0,
    }


def handle_restore_staging_blocks(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    # Blocking control-plane restore from StagingTier. Used
    # by connectors whose load hook is synchronous (for
    # example TRT-LLM): the caller supplies content hashes,
    # destination block ids, and staging generations. The
    # daemon copies bytes into HBM and returns only after
    # its stream is synchronized.
    engine_id = str(msg["engine_id"])
    with daemon._lock:
        pool = daemon._pools.get(engine_id)
    if pool is None or pool.restore_consumer is None:
        return {
            "ok": False,
            "error": "engine restore consumer not attached",
        }
    if daemon.staging_tier is None:
        return {
            "ok": False,
            "error": "staging not enabled",
        }
    raw_items = msg.get("items") or []
    block_pairs: list = []
    allocated: list[int] = []
    with daemon._staging_restore_lock:
        for it in raw_items:
            try:
                content_hash = bytes.fromhex(
                    str(it["content_hash"]),
                )
                dest_block = int(it["dest_block"])
                generation = int(it["generation"])
            except (KeyError, ValueError, TypeError):
                continue
            for _ in range(0xFFFF_FFFE):
                hid = daemon._next_staging_restore_handle
                daemon._next_staging_restore_handle += 1
                if daemon._next_staging_restore_handle > 0xFFFF_FFFF:
                    daemon._next_staging_restore_handle = 1
                if hid not in daemon._staging_restore_handles:
                    break
            else:
                continue
            daemon._staging_restore_handles[int(hid)] = (
                content_hash,
                generation,
            )
            allocated.append(int(hid))
            block_pairs.append((int(hid), dest_block))
    if not block_pairs:
        return {
            "ok": False,
            "error": "no valid staging restore items",
        }
    try:
        success = pool.restore_consumer._process_staging(
            {"block_pairs": block_pairs},
            pool,
        )
    finally:
        # _process_staging consumes handles as it reaches
        # them. Release any handles after a partial failure.
        with daemon._staging_restore_lock:
            for hid in allocated:
                daemon._staging_restore_handles.pop(hid, None)
    return {
        "ok": True,
        "success": bool(success),
        "requested": len(block_pairs),
        "restored": len(block_pairs) if success else 0,
    }


def handle_register_staging_restore_handles(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    # Worker-side connector is about to push a
    # FLAG_SOURCE_STAGING restore ring record. The ring's
    # src field is only u32, so we create one-shot local
    # handles that resolve to (content_hash, generation).
    if daemon.staging_tier is None:
        return {
            "ok": False,
            "error": "staging not enabled",
        }
    items = msg.get("items") or []
    handles: list = []
    with daemon._staging_restore_lock:
        for it in items:
            try:
                content_hash = bytes.fromhex(
                    str(it["content_hash"]),
                )
                generation = int(it["generation"])
            except (KeyError, ValueError, TypeError):
                handles.append(None)
                continue
            # Monotonic u32 handle allocation. Skip 0 so
            # tests and logs can treat it as invalid.
            for _ in range(0xFFFF_FFFE):
                hid = daemon._next_staging_restore_handle
                daemon._next_staging_restore_handle += 1
                if daemon._next_staging_restore_handle > 0xFFFF_FFFF:
                    daemon._next_staging_restore_handle = 1
                if hid not in daemon._staging_restore_handles:
                    break
            else:
                handles.append(None)
                continue
            daemon._staging_restore_handles[int(hid)] = (
                content_hash,
                generation,
            )
            handles.append(int(hid))
    return {"ok": True, "handles": handles}


def handle_release_staging_restore_handles(
    daemon: "GmsKvCacheManager", msg: Message
) -> Response:
    handles = msg.get("handles") or []
    released = 0
    with daemon._staging_restore_lock:
        for hid in handles:
            try:
                hid_i = int(hid)
            except (ValueError, TypeError):
                continue
            if daemon._staging_restore_handles.pop(hid_i, None) is not None:
                released += 1
    return {"ok": True, "released": released}


def handle_staging_scan(daemon: "GmsKvCacheManager", msg: Message) -> Response:
    # Phase 2 of cross-node design (see docs/CROSS_NODE_DESIGN.md
    # §4.3 — batch RPC chosen over SHM ring). Hashes are
    # hex-encoded for JSON transport. Disabled (returns empty)
    # if staging_tier wasn't enabled at daemon construction.
    if daemon.staging_tier is None:
        return {"ok": True, "hits": {}}
    req_hashes = msg.get("hashes", []) or []
    hashes_bytes = [bytes.fromhex(h) for h in req_hashes]
    raw_hits = daemon.staging_tier.scan(hashes_bytes)
    hits = {
        h.hex(): {
            "bytes_size": hit.bytes_size,
            "crc32": hit.crc32,
            "generation": hit.generation,
        }
        for h, hit in raw_hits.items()
    }
    return {"ok": True, "hits": hits}


HANDLERS: dict[str, Handler] = {
    "notify_kv_arrived": handle_notify_kv_arrived,
    "register_content_address": handle_register_content_address,
    "register_content_addresses_batch": handle_register_content_addresses_batch,
    "register_staging_restore_handles": handle_register_staging_restore_handles,
    "release_staging_restore_handles": handle_release_staging_restore_handles,
    "restore_host_blocks": handle_restore_host_blocks,
    "restore_staging_blocks": handle_restore_staging_blocks,
    "restore_staging_ranges": handle_restore_staging_ranges,
    "staging_fail": handle_staging_fail,
    "staging_reserve": handle_staging_reserve,
    "staging_scan": handle_staging_scan,
}
