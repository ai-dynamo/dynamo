# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine pool state, ring consumers, and peer-client pooling."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gms_kv_ring.daemon.kv_cache_manager import GmsKvCacheManager

logger = logging.getLogger(__name__)


_TLS = threading.local()


def _ensure_cuda_context(device_ordinal: int = 0) -> None:
    """Retain + push the primary CUDA context on this thread, once.

    The daemon's asyncio executor and consumer threads need a current
    context for any cu* call. Each thread retains the primary context
    and pushes it the first time it runs CUDA work."""
    if getattr(_TLS, "cuda_pushed", False):
        return
    from cuda.bindings import driver as drv

    err = drv.cuInit(0)[0]
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed: {err}")
    err, dev = drv.cuDeviceGet(device_ordinal)
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGet failed: {err}")
    err, ctx = drv.cuDevicePrimaryCtxRetain(dev)
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDevicePrimaryCtxRetain failed: {err}")
    err = drv.cuCtxPushCurrent(ctx)[0]
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxPushCurrent failed: {err}")
    _TLS.cuda_pushed = True


@dataclass
class LayerDesc:
    """Per-layer info the daemon needs to DMA in/out."""

    layer_idx: int
    va: int  # device VA in this daemon's address space (post cuMemMap)
    size: int  # total bytes of the layer
    stride: int  # bytes per block within the layer


@dataclass
class EnginePool:
    """One engine's attached state."""

    engine_id: str
    layers: dict[int, LayerDesc]
    stream: int = 0  # daemon-side CUDA stream for this engine
    evict_consumer: Optional["_EvictConsumer"] = None
    restore_consumer: Optional["_RestoreConsumer"] = None

    def __post_init__(self) -> None:
        # Per-block-id generation counter for Race #3 (cross-step
        # async-restore vs. evict): the scheduler-side connector is
        # the source of truth for generations and passes the new
        # value into the demote RPC. The daemon stores it (monotonic
        # max) so the restore consumer can read it back and reject
        # records whose `expected_generation` is stale. Keyed by
        # block_id (not (layer, offset)) because the connector
        # evicts ALL layers of a block atomically with the SAME
        # generation; multiple per-layer RPCs all set the same value.
        self.slot_generations: dict[int, int] = {}

    def set_block_generation(self, block_id: int, generation: int) -> None:
        """Set the slot's generation. Monotonic — never goes
        backwards (defensive against out-of-order RPCs in the
        rare reattach-mid-evict edge case). Called by demote RPCs
        with the caller-supplied generation; the per-block-id
        generation lifts in lockstep across layer-level demote
        RPCs that share the same block."""
        old = self.slot_generations.get(int(block_id), 0)
        if int(generation) > old:
            self.slot_generations[int(block_id)] = int(generation)

    def current_block_generation(self, block_id: int) -> int:
        """Read the current generation for `block_id` (0 if never
        evicted). Called by the restore consumer to verify the
        record's expected_generation matches reality."""
        return self.slot_generations.get(int(block_id), 0)


# ---------------------------------------------------------------- consumers


class _EvictConsumer:
    """Drains one engine's evict ring → cuMemcpyAsync D2H → host tier."""

    POLL_S = 1e-4

    def __init__(
        self,
        daemon: "GmsKvCacheManager",
        engine_id: str,
        ring_path: str,
        counter_host_addr: int = 0,
        num_counters: int = 0,
        counter_path: str = "",
    ) -> None:
        self.daemon = daemon
        self.engine_id = engine_id
        self.ring_path = ring_path
        self.counter_host_addr = int(counter_host_addr)
        self.num_counters = int(num_counters)
        self.counter_path = counter_path
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._reader = None
        self._counters = None

    def start(self) -> None:
        from gms_kv_ring.common.counter import DaemonCounterArray
        from gms_kv_ring.common.evict_ring import attach_reader

        self._reader = attach_reader(self.ring_path)
        # Same counter array as the restore consumer — engine reserves
        # slots from a single shared pool so num_counters caps total
        # in-flight (evict + restore) per engine.
        if self.counter_host_addr and self.num_counters:
            # Same-process: reuse engine's already-pinned VA.
            self._counters = DaemonCounterArray.attach_in_process(
                self.counter_host_addr,
                num_counters=self.num_counters,
            )
        elif self.counter_path and self.num_counters:
            # Cross-process: open the file and pin our own VA.
            self._counters = DaemonCounterArray.attach(
                self.counter_path,
                num_counters=self.num_counters,
            )
        self._thread = threading.Thread(
            target=self._run,
            name=f"evict-{self.engine_id[:12]}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        # Signal + join only. The consumer thread owns the reader's
        # lifetime and closes it on its way out — we must NOT munmap
        # here, because if join times out, the consumer thread is
        # still polling the buffer.
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning(
                    "evict consumer for %r failed to exit within 5s; "
                    "ring/buffer will leak until process exit (safer than "
                    "munmap during in-flight access)",
                    self.engine_id,
                )

    def _run(self) -> None:
        _ensure_cuda_context()
        logger.info("evict consumer: engine_id=%r", self.engine_id)
        try:
            while not self._stop.is_set():
                rec = self._reader.try_pop()
                if rec is None:
                    time.sleep(self.POLL_S)
                    continue
                try:
                    self._process(rec)
                except Exception:  # noqa: BLE001
                    logger.warning("evict: process failed", exc_info=True)
                    # Engine waiting on evict-ack would hang if we don't
                    # signal. Write target+1 (failure indicator) so the
                    # engine's evict_succeeded() returns False.
                    self._signal_evict_failed(rec)
        finally:
            if self._reader is not None:
                try:
                    self._reader.close()
                except Exception:
                    pass
            if self._counters is not None:
                try:
                    self._counters.close()
                except Exception:
                    pass

    def _signal_evict_ack(self, rec: dict, success: bool) -> None:
        """If the record carried a counter_slot/target (engine wants
        an ack), write the appropriate value: target on success,
        target+1 on failure. No-op if 0/0 (fire-and-forget)."""
        slot = rec.get("counter_slot", 0)
        target = rec.get("counter_target", 0)
        if not target or self._counters is None:
            return
        value = target if success else target + 1
        try:
            self._counters.write_host(slot, value)
        except Exception:  # noqa: BLE001
            logger.warning("evict: counter ack write failed", exc_info=True)
        from gms_kv_ring.common import metrics

        if not success:
            metrics.evict_errors.inc(engine_id=self.engine_id)

    def _signal_evict_failed(self, rec: dict) -> None:
        self._signal_evict_ack(rec, success=False)

    def _process(self, rec: dict) -> None:
        from cuda.bindings import driver as drv

        pool = self.daemon._pools.get(self.engine_id)
        if pool is None:
            self._signal_evict_failed(rec)
            return
        # Optional cuStreamWaitEvent on the engine's IPC event so the
        # D2H sees a quiesced source.
        if rec.get("ipc_event"):
            try:
                from cuda.bindings import runtime as rt

                handle = rt.cudaIpcEventHandle_t()
                import ctypes

                ctypes.memmove(
                    ctypes.addressof(ctypes.c_char.from_address(handle.getPtr())),
                    rec["ipc_event"],
                    64,
                )
                err, ev = rt.cudaIpcOpenEventHandle(handle)
                if err == rt.cudaError_t.cudaSuccess:
                    rt.cudaStreamWaitEvent(int(pool.stream), int(ev), 0)
                    rt.cudaEventDestroy(int(ev))
            except Exception:  # noqa: BLE001
                logger.debug("evict: ipc event import failed", exc_info=True)

        from gms_kv_ring.common import metrics

        block_id = rec["block_id"]
        writes = []
        bytes_copied = 0
        any_error = False
        for layer_idx, size, offset in rec["ranges"]:
            ld = pool.layers.get(int(layer_idx))
            if ld is None:
                any_error = True
                continue
            src_va = ld.va + int(offset)
            write = self.daemon.host_tier.reserve(
                self.engine_id,
                layer_idx,
                offset,
                size,
            )
            writes.append((int(layer_idx), int(offset), int(size), write))
            err = drv.cuMemcpyAsync(
                drv.CUdeviceptr(write.host_ptr),
                drv.CUdeviceptr(src_va),
                int(size),
                int(pool.stream),
            )[0]
            if err != drv.CUresult.CUDA_SUCCESS:
                _, m = drv.cuGetErrorString(err)
                logger.warning(
                    "evict D2H failed (eng=%r blk=%d layer=%d): %s",
                    self.engine_id,
                    block_id,
                    layer_idx,
                    m.decode() if m else err,
                )
                metrics.evict_errors.inc(engine_id=self.engine_id)
                any_error = True
            else:
                bytes_copied += int(size)
        # Drain the stream — only after this returns are the host
        # buffers actually populated. THIS is the critical sync that
        # makes the host_tier slot safe to read.
        sync_ok = True
        try:
            drv.cuStreamSynchronize(int(pool.stream))
        except Exception:  # noqa: BLE001
            logger.warning("evict: stream sync failed", exc_info=True)
            sync_ok = False
        if sync_ok and not any_error:
            from gms_kv_ring.common.checksum import crc32_at_ptr

            protected = set()
            for layer_idx, offset, size, write in writes:
                crc = crc32_at_ptr(write.host_ptr, size)
                if self.daemon.host_tier.commit(
                    write,
                    crc,
                    generation=int(rec.get("generation", 0)),
                ):
                    protected.add((self.engine_id, layer_idx, offset))
                else:
                    any_error = True
            if protected:
                self.daemon._enforce_host_tier_quota(protected)
        elif sync_ok:
            # The stream is drained, so provisional buffers are safe to free.
            for _layer_idx, _offset, _size, write in writes:
                self.daemon.host_tier.abort(write)

        if sync_ok and not any_error:
            self._signal_evict_ack(rec, success=True)
        else:
            # A failed sync may still have DMA in flight; retain its write pins
            # until process teardown rather than freeing DMA targets.
            self._signal_evict_ack(rec, success=False)
        metrics.evict_records.inc(engine_id=self.engine_id)
        if bytes_copied:
            metrics.evict_d2h_bytes.inc(engine_id=self.engine_id, n=bytes_copied)
        metrics.host_tier_slots.set(
            self.daemon.host_tier.n_slots(),
            engine_id=self.engine_id,
        )


class _RestoreConsumer:
    """Drains one engine's restore ring → cuMemcpyBatchAsync H2D →
    cuStreamWriteValue32 on counter slot."""

    POLL_S = 1e-4

    def __init__(
        self,
        daemon: "GmsKvCacheManager",
        engine_id: str,
        ring_path: str,
        counter_path: str,
        num_counters: int,
        counter_host_addr: int = 0,
    ) -> None:
        self.daemon = daemon
        self.engine_id = engine_id
        self.ring_path = ring_path
        self.counter_path = counter_path
        self.num_counters = num_counters
        self.counter_host_addr = int(counter_host_addr)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._reader = None
        self._counters = None

    def start(self) -> None:
        from gms_kv_ring.common.counter import DaemonCounterArray
        from gms_kv_ring.common.restore_ring import attach_reader

        self._reader = attach_reader(self.ring_path)
        if self.counter_host_addr:
            # Same-process: reuse the engine's already-pinned VA. No
            # second mmap, no second register, no risk of phantom
            # pinning entries surviving across tests.
            self._counters = DaemonCounterArray.attach_in_process(
                self.counter_host_addr,
                num_counters=self.num_counters,
            )
        else:
            self._counters = DaemonCounterArray.attach(
                self.counter_path,
                num_counters=self.num_counters,
            )
        self._thread = threading.Thread(
            target=self._run,
            name=f"restore-{self.engine_id[:12]}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        # Same contract as _EvictConsumer.stop: signal + join only.
        # Consumer thread owns reader + counters and cleans them up
        # in its own finally, so a join-timeout doesn't munmap memory
        # the consumer is still reading from.
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning(
                    "restore consumer for %r failed to exit within 5s; "
                    "ring/counter buffers will leak until process exit",
                    self.engine_id,
                )

    def _run(self) -> None:
        _ensure_cuda_context()
        logger.info("restore consumer: engine_id=%r", self.engine_id)
        try:
            while not self._stop.is_set():
                rec = self._reader.try_pop()
                if rec is None:
                    time.sleep(self.POLL_S)
                    continue
                try:
                    self._process(rec)
                except Exception:  # noqa: BLE001
                    # If _process raised before writing the counter, the
                    # engine's cuStreamWaitValue32 will hang forever.
                    # Unblock it: write the target value anyway and bump
                    # an error metric. The engine reads whatever bytes
                    # made it into HBM — typically wrong if H2D failed —
                    # but the request fails loudly downstream rather
                    # than silently wedging the GPU.
                    logger.warning("restore: process failed", exc_info=True)
                    self._signal_counter_after_error(rec)
        finally:
            # Consumer-owned cleanup.
            if self._reader is not None:
                try:
                    self._reader.close()
                except Exception:
                    pass
            if self._counters is not None:
                try:
                    self._counters.close()
                except Exception:
                    pass

    def _signal_counter_after_error(self, rec: dict) -> None:
        """Last-resort signal from _run's exception handler. Writes the
        FAILURE indicator (target+1) — the engine's GEQ wait satisfies
        but `handle.restore_succeeded(slot, target)` returns False."""
        try:
            if self._counters is not None:
                self._counters.write_host(
                    rec["counter_slot"],
                    rec["counter_target"] + 1,
                )
            from gms_kv_ring.common import metrics

            metrics.restore_records.inc(engine_id=self.engine_id)
            metrics.restore_failures.inc(engine_id=self.engine_id)
        except Exception:
            logger.error(
                "restore: failed to even signal error counter slot=%r",
                rec.get("counter_slot"),
                exc_info=True,
            )

    def _write_counter(self, rec: dict, success: bool) -> None:
        """Signal the engine. On success: write target. On failure:
        write target+1. The engine's `cuStreamWaitValue32(GEQ target)`
        satisfies on either; the engine's host-side
        `restore_succeeded(slot, target)` check distinguishes."""
        from gms_kv_ring.common import metrics

        try:
            value = rec["counter_target"] if success else rec["counter_target"] + 1
            self._counters.write_host(rec["counter_slot"], value)
        except Exception:  # noqa: BLE001
            logger.warning("restore counter write failed", exc_info=True)
            metrics.restore_failures.inc(engine_id=self.engine_id)
            return
        metrics.restore_records.inc(engine_id=self.engine_id)
        if not success:
            metrics.restore_failures.inc(engine_id=self.engine_id)

    def _process(self, rec: dict) -> None:
        from gms_kv_ring.common.restore_ring import FLAG_GDS_DIRECT, FLAG_SOURCE_STAGING

        flags = int(rec.get("flags", 0))

        # Host-tier transition path: a shadow engine can restore CPU-mirrored
        # blocks from a primary engine while both pools are attached. The
        # source pool supplies the source block geometry; the destination pool
        # supplies the target HBM addresses. Missing source pools or slots fail
        # closed so the engine falls back to recompute.
        src_engine_id = rec["src_engine_id"]
        dest_pool = self.daemon._pools.get(self.engine_id)
        if dest_pool is None:
            self._write_counter(rec, success=False)
            return
        if flags & FLAG_SOURCE_STAGING:
            if flags & FLAG_GDS_DIRECT:
                logger.warning(
                    "restore: invalid flags combine SOURCE_STAGING "
                    "and GDS_DIRECT; signaling failure",
                )
                self._write_counter(rec, success=False)
                return
            ok = self._process_staging(rec, dest_pool)
            self._write_counter(rec, success=ok)
            return
        src_pool = self.daemon._pools.get(src_engine_id)
        if src_pool is None:
            logger.warning(
                "restore: src_engine_id=%r is not attached; signaling failure",
                src_engine_id,
            )
            self._write_counter(rec, success=False)
            return

        # GDS-direct branch: bypass host_tier and pull bytes from
        # the storage backend straight into engine HBM via
        # `backend.promote_into_gpu` (cuFile). Used by the V1
        # KVCacheConnector's cache-hit restore-on-hit path so the
        # engine can issue the wait on the compute stream and
        # overlap cuFile reads with forward-pass compute, instead
        # of blocking in bind_connector_metadata.
        if flags & FLAG_GDS_DIRECT:
            if src_engine_id != self.engine_id:
                logger.warning(
                    "restore-gds: src_engine_id=%r mismatches daemon's "
                    "engine_id=%r; signaling failure",
                    src_engine_id,
                    self.engine_id,
                )
                self._write_counter(rec, success=False)
                return
            ok = self._process_gds(rec, dest_pool)
            self._write_counter(rec, success=ok)
            return

        ok = self._process_host_tier(rec, dest_pool, src_pool)
        self._write_counter(rec, success=ok)

    def _process_host_tier(
        self,
        rec: dict,
        dest_pool: "EnginePool",
        src_pool: "EnginePool",
    ) -> bool:
        from cuda.bindings import driver as drv
        from gms_kv_ring.common import metrics
        from gms_kv_ring.common.cuda_batch import batch_h2d, has_batch_h2d

        src_engine_id = rec["src_engine_id"]
        expected_generations = {
            int(k): int(v)
            for k, v in dict(rec.get("expected_generations") or {}).items()
        }
        leases = []
        try:
            # Collect (dst_va, src_host_ptr, size) for every layer that
            # has a valid host_tier slot. We count expected vs found —
            # if any pair is missing a slot, the restore is incomplete and
            # the engine must fall back to recompute. Slots are leased until
            # after H2D sync, so bounded eviction and re-spill cannot free or
            # overwrite a pinned source pointer.
            dsts: list[int] = []
            srcs: list[int] = []
            sizes: list[int] = []
            source_slots = []
            expected = 0
            for src_blk, dest_blk in rec["block_pairs"]:
                expected_gen = expected_generations.get(int(src_blk), 0)
                for layer_idx, ld in src_pool.layers.items():
                    expected += 1
                    src_off = int(src_blk) * ld.stride
                    lease = self.daemon.host_tier.pin(
                        src_engine_id,
                        int(layer_idx),
                        src_off,
                        expected_generation=expected_gen,
                    )
                    if lease is None:
                        continue
                    dest_ld = dest_pool.layers.get(int(layer_idx))
                    if dest_ld is None:
                        lease.release()
                        continue
                    slot = lease.slot
                    dst_va = dest_ld.va + int(dest_blk) * dest_ld.stride
                    dsts.append(dst_va)
                    srcs.append(slot.host_ptr)
                    sizes.append(slot.size)
                    source_slots.append(
                        (
                            (src_engine_id, int(layer_idx), src_off),
                            slot,
                        )
                    )
                    leases.append(lease)

            if expected == 0 or len(dsts) != expected:
                return False

            # Integrity check: re-CRC every source slot and compare to the
            # CRC captured when the CPU mirror was committed.
            from gms_kv_ring.common.checksum import crc32_at_ptr

            for key, slot in source_slots:
                if slot.crc == 0:
                    continue
                actual = crc32_at_ptr(slot.host_ptr, slot.size)
                if actual != slot.crc:
                    _engine_id, layer_idx, src_off = key
                    logger.warning(
                        "restore: CRC mismatch eng=%r layer=%d "
                        "off=%d expected=%#x actual=%#x",
                        self.engine_id,
                        layer_idx,
                        src_off,
                        slot.crc,
                        actual,
                    )
                    metrics.checksum_mismatches.inc(
                        engine_id=self.engine_id,
                    )
                    return False

            try:
                if has_batch_h2d():
                    batch_h2d(dsts, srcs, sizes, int(dest_pool.stream))
                else:
                    for d, s, sz in zip(dsts, srcs, sizes):
                        drv.cuMemcpyAsync(
                            drv.CUdeviceptr(d),
                            drv.CUdeviceptr(s),
                            int(sz),
                            int(dest_pool.stream),
                        )
                metrics.restore_h2d_bytes.inc(
                    engine_id=self.engine_id,
                    n=sum(sizes),
                )
            except Exception:  # noqa: BLE001
                logger.warning("restore H2D batch failed", exc_info=True)
                return False

            try:
                drv.cuStreamSynchronize(int(dest_pool.stream))
            except Exception:  # noqa: BLE001
                logger.warning(
                    "restore: dest stream sync failed",
                    exc_info=True,
                )
                return False
            return True
        finally:
            for lease in reversed(leases):
                lease.release()

    def _take_staging_restore_handle(
        self,
        handle_id: int,
    ) -> Optional[tuple[bytes, int]]:
        with self.daemon._staging_restore_lock:
            return self.daemon._staging_restore_handles.pop(
                int(handle_id),
                None,
            )

    def _process_staging(self, rec: dict, dest_pool: "EnginePool") -> bool:
        """Restore bytes from the destination daemon's StagingTier.

        Each restore-ring pair is ``(staging_handle_id, dest_block)``.
        The handle resolves to ``(content_hash, generation)``; the
        consumer pins that READY staging slot, copies the concatenated
        per-layer payload into the destination block, synchronizes the
        daemon stream, then releases the staging refcount.
        """
        from cuda.bindings import driver as drv
        from gms_kv_ring.common import metrics

        if self.daemon.staging_tier is None:
            logger.warning(
                "restore-staging: staging tier is disabled for engine=%r",
                self.engine_id,
            )
            return False

        dsts: list[int] = []
        srcs: list[int] = []
        sizes: list[int] = []
        consume_handles: list = []
        ok = True

        try:
            for handle_id, dest_blk in rec["block_pairs"]:
                entry = self._take_staging_restore_handle(int(handle_id))
                if entry is None:
                    logger.warning(
                        "restore-staging: unknown handle_id=%s",
                        int(handle_id),
                    )
                    ok = False
                    break
                content_hash, generation = entry
                consume = self.daemon.staging_tier.begin_consume(
                    content_hash,
                    int(generation),
                )
                if consume is None:
                    logger.warning(
                        "restore-staging: hash=%s generation=%d not READY",
                        content_hash.hex()[:16],
                        int(generation),
                    )
                    ok = False
                    break
                consume_handles.append(consume)
                ptr_info = self.daemon.staging_tier.consume_pointer(consume)
                if ptr_info is None:
                    logger.warning(
                        "restore-staging: hash=%s disappeared after pin",
                        content_hash.hex()[:16],
                    )
                    ok = False
                    break
                src_ptr, bytes_size, _crc32 = ptr_info
                cursor = 0
                for layer_idx, ld in sorted(dest_pool.layers.items()):
                    size = int(ld.stride)
                    if cursor + size > bytes_size:
                        logger.warning(
                            "restore-staging: payload too small for "
                            "layer=%d dest_blk=%d cursor=%d size=%d "
                            "payload=%d",
                            int(layer_idx),
                            int(dest_blk),
                            cursor,
                            size,
                            bytes_size,
                        )
                        ok = False
                        break
                    dst_va = int(ld.va) + int(dest_blk) * int(ld.stride)
                    dsts.append(dst_va)
                    srcs.append(int(src_ptr) + cursor)
                    sizes.append(size)
                    cursor += size
                if not ok:
                    break
                if cursor != bytes_size:
                    # The staging payload layout must match the
                    # destination daemon's known per-layer block layout.
                    # A mismatch is a correctness failure, not a partial
                    # hit. The engine falls back to recompute.
                    logger.warning(
                        "restore-staging: payload size mismatch for "
                        "hash=%s consumed=%d payload=%d",
                        content_hash.hex()[:16],
                        cursor,
                        bytes_size,
                    )
                    ok = False
                    break

            if not ok or not dsts:
                return False

            for d, s, sz in zip(dsts, srcs, sizes):
                drv.cuMemcpyAsync(
                    drv.CUdeviceptr(d),
                    drv.CUdeviceptr(s),
                    int(sz),
                    int(dest_pool.stream),
                )
            metrics.restore_h2d_bytes.inc(
                engine_id=self.engine_id,
                n=sum(sizes),
            )
            drv.cuStreamSynchronize(int(dest_pool.stream))
            return True
        except Exception:  # noqa: BLE001
            logger.warning("restore-staging: copy failed", exc_info=True)
            return False
        finally:
            for consume in consume_handles:
                try:
                    self.daemon.staging_tier.end_consume(consume)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "restore-staging: end_consume failed",
                        exc_info=True,
                    )

    def _process_gds(self, rec: dict, dest_pool: "EnginePool") -> bool:
        """GDS-direct restore: pull bytes from the storage backend
        straight into engine HBM via `promote_into_gpu` (cuFile).
        Returns True iff every (block_pair × layer) read verified.

        Single-threaded by design (one consumer thread per engine);
        cuFile reads are issued sequentially. Concurrency with the
        engine's forward pass is the win: while the consumer is
        reading, the engine's compute kernels are running and will
        cuStreamWaitValue32 only when they need the restored
        bytes."""
        from gms_kv_ring.common import metrics

        backend = (
            self.daemon.storage_tier.backend
            if hasattr(self.daemon.storage_tier, "backend")
            else self.daemon.storage_tier
        )
        if not backend.supports_gpu_direct():
            logger.warning(
                "restore-gds: backend %r doesn't support GPU-direct; "
                "signaling failure for engine=%r",
                getattr(backend, "name", type(backend).__name__),
                self.engine_id,
            )
            return False

        # `_process` already enforces src_engine_id == self.engine_id for
        # GDS-direct records, so the backend storage key uses this engine id.
        ok_all = True
        n_layers_done = 0
        for src_blk, dest_blk in rec["block_pairs"]:
            for layer_idx, ld in dest_pool.layers.items():
                src_off = int(src_blk) * ld.stride
                dst_va = ld.va + int(dest_blk) * ld.stride
                try:
                    verified = backend.promote_into_gpu(
                        self.engine_id,
                        int(layer_idx),
                        src_off,
                        dest_gpu_ptr=int(dst_va),
                        max_size=int(ld.stride),
                    )
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "restore-gds: backend raise for layer=%d src_blk=%d dst_blk=%d",
                        int(layer_idx),
                        int(src_blk),
                        int(dest_blk),
                        exc_info=True,
                    )
                    ok_all = False
                    continue
                if verified is None:
                    # Backend reports verify-failed (CRC mismatch,
                    # missing slot, etc.). Engine falls to cold
                    # compute when restore_succeeded() returns False.
                    ok_all = False
                    continue
                n_layers_done += 1
        if n_layers_done:
            metrics.promote_hbm_ok.inc(engine_id=self.engine_id, n=n_layers_done)
        return ok_all


# ---------------------------------------------------------------- daemon


class _SourceClientPool:
    """Pool of long-lived `DaemonClient` sockets to one peer daemon.
    Used by `fetch_remote` to pipeline `transfer_blocks_batch` RPCs.

    `DaemonClient` serializes calls on its single socket via an
    internal lock — to actually issue batches in parallel we need
    multiple sockets. `pool_size` is small (default 4) because each
    batch RPC is short (just submission ACK, not byte transfer)."""

    def __init__(self, uds_path: str, pool_size: int = 4) -> None:
        from gms_kv_ring.daemon.client import DaemonClient

        self.uds_path = uds_path
        self._clients: list = []
        self._next = 0
        self._lock = threading.Lock()
        # Eagerly open. If the source isn't up yet, raise — caller
        # retries the whole `fetch_remote` (rare).
        for _ in range(max(1, pool_size)):
            self._clients.append(DaemonClient(uds_path))

    def get(self):
        """Round-robin a client. Caller is free to invoke RPCs on it
        from any thread — DaemonClient is internally thread-safe."""
        with self._lock:
            c = self._clients[self._next % len(self._clients)]
            self._next += 1
        return c

    def close(self) -> None:
        for c in self._clients:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        self._clients.clear()
