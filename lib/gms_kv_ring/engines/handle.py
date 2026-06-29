# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMSKvRing — the single engine-side handle the engine hook owns.

Engine adapters (engines/sglang/, engines/vllm/) build one of these
at startup and monkey-patch their engine's eviction/restore hooks
to call:

    handle.record_evict(block_id, ranges, ipc_event=None)
    slot, target = handle.record_restore(src_engine_id, pairs)
    handle.wait_restore(stream, slot, target)

The handle creates + owns:
  • the engine-side evict ring (writer)
  • the engine-side restore ring (writer)
  • the engine counter array (cuStreamWaitValue32 source)
  • a control-socket connection to the daemon

It does NOT own the engine's KV pool — the engine builds that and
hands the layer descriptors in. Decoupling pool ownership from the
ring/handle keeps the integration boundary thin (~20 LOC inside the
engine).
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class _PathSet:
    evict_ring: str
    restore_ring: str
    counter: str


def _alloc_paths(engine_id: str) -> _PathSet:
    """All daemon-shared files live on tmpfs (/dev/shm)."""
    base = f"/dev/shm/gms-kvr-{engine_id}-{uuid.uuid4().hex[:8]}"
    return _PathSet(
        evict_ring=f"{base}.evict.ring",
        restore_ring=f"{base}.restore.ring",
        counter=f"{base}.counter.bin",
    )


class GMSKvRing:
    """Engine-side handle. One per engine process."""

    def __init__(
        self,
        *,
        engine_id: str,
        daemon_socket: str,
        layers: list[dict],
        evict_ring_capacity: int = 4096,
        restore_ring_capacity: int = 4096,
        num_counters: int = 512,
    ) -> None:
        from gms_kv_ring.common.counter import EngineCounterArray
        from gms_kv_ring.common.evict_ring import create_ring as create_evict
        from gms_kv_ring.common.restore_ring import create_ring as create_restore
        from gms_kv_ring.daemon.client import DaemonClient

        self.engine_id = engine_id
        self._paths = _alloc_paths(engine_id)
        self._closed = False
        self._engine_pool_attached = False
        self.evict_writer = None
        self.restore_writer = None
        self.counters = None
        self._client = None
        self._capabilities = {}
        # The ring buffers are SPSC: a single producer per ring is the
        # data-structure contract. Engine adapters typically own a
        # single eviction thread, but custom adapters might call from
        # multiple threads — these locks serialize pushes per-ring so
        # the SPSC invariant is held even under misuse.
        self._evict_lock = threading.Lock()
        self._restore_lock = threading.Lock()

        try:
            self.evict_writer = create_evict(
                self._paths.evict_ring,
                capacity=evict_ring_capacity,
            )
            self.restore_writer = create_restore(
                self._paths.restore_ring,
                capacity=restore_ring_capacity,
            )
            self.counters = EngineCounterArray.create(
                self._paths.counter,
                num_counters=num_counters,
            )

            self._client = DaemonClient(daemon_socket)
            self._client.attach_engine_pool(engine_id, layers)
            self._engine_pool_attached = True
            # The engine process and daemon may be separate processes.
            # Always pass the counter shm path so the daemon maps and
            # registers its own valid VA instead of using this process's
            # counter_host_addr.
            self._client.attach_evict_ring(
                engine_id,
                self._paths.evict_ring,
                counter_path=self._paths.counter,
                num_counters=num_counters,
            )
            self._client.attach_restore_ring(
                engine_id,
                self._paths.restore_ring,
                self._paths.counter,
                num_counters,
            )

            # Capability probe — done once at attach. Engine adapters
            # check `gpu_direct_storage_available()` to decide whether
            # to enable the demote_hbm_to_storage / promote_storage_to_hbm
            # paths. Cached for the lifetime of the handle (the daemon's
            # storage backend doesn't switch at runtime).
            try:
                self._capabilities = self._client.capabilities()
            except Exception:  # noqa: BLE001
                # Older daemon without the capabilities RPC — assume
                # no GPU-direct. Engine adapters should `.get` defaults.
                self._capabilities = {}
        except Exception:
            self._cleanup_resources(detach=True)
            raise

        logger.info(
            "GMSKvRing attached: engine_id=%r daemon=%s "
            "evict=%s restore=%s counter=%s capabilities=%s",
            engine_id,
            daemon_socket,
            self._paths.evict_ring,
            self._paths.restore_ring,
            self._paths.counter,
            self._capabilities,
        )

    def _cleanup_resources(self, *, detach: bool) -> None:
        client = getattr(self, "_client", None)
        if detach and client is not None and self._engine_pool_attached:
            try:
                client.detach_engine_pool(self.engine_id)
            except Exception:  # noqa: BLE001
                logger.warning("detach_engine_pool failed", exc_info=True)
            finally:
                self._engine_pool_attached = False
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
            self._client = None

        for attr in ("evict_writer", "restore_writer", "counters"):
            resource = getattr(self, attr, None)
            if resource is None:
                continue
            try:
                resource.close()
            except Exception:
                pass
            setattr(self, attr, None)

        for path in (
            self._paths.evict_ring,
            self._paths.restore_ring,
            self._paths.counter,
        ):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except Exception:
                logger.debug("unlink %s failed", path, exc_info=True)

    def _reserve_counter_slot(self) -> Optional[tuple[int, int]]:
        return self.counters.try_reserve_completed_slot()

    # ---- hot-path engine API ----

    def record_evict(
        self,
        block_id: int,
        ranges: list[tuple[int, int, int]],
        ipc_event_handle: bytes = b"",
        generation: int = 0,
    ) -> Optional[tuple[int, int]]:
        """Engine calls this from its eviction hook.

        Returns (slot, target) on successful ring push. Engine MUST
        call `evict_succeeded(slot, target)` host-side after sync
        before reusing the source HBM — until then, the bytes may not
        have landed in host_tier yet (DMA still in flight), so the
        slot must not be overwritten.

        Returns None if the ring was full (engine should drop the
        offload or retry — spills are advisory).

        Producer-side lock enforces the ring's SPSC invariant under
        multi-threaded callers, and atomically pairs the counter slot
        reservation with the push that consumes it."""
        with self._evict_lock:
            reserved = self._reserve_counter_slot()
            if reserved is None:
                return None
            slot, target = reserved
            ok = self.evict_writer.push(
                block_id=block_id,
                ranges=ranges,
                ipc_event_handle=ipc_event_handle,
                counter_slot=slot,
                counter_target=target,
                generation=int(generation),
            )
            if not ok:
                self.counters.mark_slot_failed(slot, target)
                return None
            return slot, target

    def wait_evict(self, stream: int, slot: int, target: int) -> None:
        """Optional: engine queues `cuStreamWaitValue32(GEQ target)` on
        a stream to gate downstream work on spill completion. The wait
        satisfies on success (counter == target) OR failure
        (counter == target+1) — engine MUST call `evict_succeeded()`
        after sync to distinguish.

        Most engines won't need this — the typical pattern is host-side
        polling of `evict_succeeded` from the allocator before slot
        reuse, since allocator decisions are host-side anyway."""
        self.counters.wait_value32(stream, slot, target)

    def evict_succeeded(self, slot: int, target: int) -> bool:
        """Host-side check: True iff the spill at (slot, target) is
        committed to host_tier. Caller MUST have host-synced first
        (or run after wait_evict + cudaStreamSynchronize).

        Daemon writes:
          counter[slot] == target      → success, bytes are in host_tier
          counter[slot] == target + 1  → failure, slot will not appear in host_tier
          counter[slot]  > target + 1  → slot has been reused (failure)

        Engine integration pattern for allocator-driven async eviction:

            slot, target = handle.record_evict(block_id, ranges)
            pending[block_id] = (slot, target)
            # engine continues — does NOT reuse the HBM slot yet

            # at next scheduling tick, after host sync:
            slot, target = pending.pop(block_id)
            if handle.evict_succeeded(slot, target):
                allocator.free_hbm_slot(block_id)
            else:
                # spill failed — either retry, or accept the block
                # is gone (will recompute on next access)
                pass"""
        return self.counters.read_slot(slot) == int(target)

    def restore_host_blocks_sync(
        self,
        src_engine_id: str,
        src_to_dst: list[tuple[int, int, int]],
    ) -> bool:
        """Blocking host-tier restore with source generation checks."""
        return self._client.restore_host_blocks(
            self.engine_id,
            str(src_engine_id),
            src_to_dst,
        )

    def record_restore(
        self,
        src_engine_id: str,
        block_pairs: list[tuple[int, int]],
        *,
        flags: int = 0,
    ) -> Optional[tuple[int, int]]:
        """Engine calls this on cache hit. Returns (slot, target) that
        the engine then passes to wait_restore() to gate consumer
        kernels on. Returns None if the restore ring is full — caller
        should fall through to a cold-compute path.

        `src_engine_id`: under the single-active-engine model (see
        project_single_active_engine.md) this must equal
        `self.engine_id`. The field is retained in the wire format
        for compat but the daemon enforces equality and signals
        failure on mismatch. Pass `self.engine_id` for production
        callers.

        `flags`: per-record flags byte (see common/restore_ring.py).
        Default 0 = host_tier → HBM via cudaMemcpyAsync (the
        existing path). Pass FLAG_GDS_DIRECT to read direct from
        the storage tier via the backend's `promote_into_gpu`
        (cuFile read straight into HBM, no host_tier hop).

        Producer-side lock enforces the ring's SPSC invariant under
        multi-threaded callers, and atomically pairs the slot
        reservation with the push that consumes it."""
        with self._restore_lock:
            reserved = self._reserve_counter_slot()
            if reserved is None:
                return None
            slot, target = reserved
            ok = self.restore_writer.push(
                src_engine_id=src_engine_id,
                block_pairs=block_pairs,
                counter_slot=slot,
                counter_target=target,
                flags=int(flags) & 0xFF,
            )
            if not ok:
                self.counters.mark_slot_failed(slot, target)
                return None
            return slot, target

    def record_restore_gds(
        self,
        src_engine_id: str,
        block_pairs: list[tuple[int, int]],
    ) -> Optional[tuple[int, int]]:
        """Async GPU-direct restore via the restore ring. Daemon
        pops the record, issues per-layer
        `backend.promote_into_gpu(...)` (cuFile read direct into
        engine HBM), writes the counter. Engine queues
        `wait_restore(stream, slot, target)` on its compute stream
        to gate downstream kernels — no host-side wait.

        Convenience wrapper over `record_restore(..., flags=
        FLAG_GDS_DIRECT)`. Returns None on ring-full so the caller
        can fall through to a synchronous remap or cold compute."""
        from gms_kv_ring.common.restore_ring import FLAG_GDS_DIRECT

        return self.record_restore(
            src_engine_id,
            block_pairs,
            flags=FLAG_GDS_DIRECT,
        )

    def restore_staging_blocks_sync(
        self,
        block_hits: "list[tuple[bytes, int, int]]",
    ) -> bool:
        """Blocking restore from the daemon's StagingTier.

        ``block_hits`` entries are ``(content_hash, dest_block_id,
        generation)``. This is intended for engine integrations whose
        load hook is synchronous and runs before forward execution.
        """
        if not block_hits:
            return True
        return self._client.restore_staging_blocks(self.engine_id, block_hits)

    def restore_staging_ranges_sync(
        self,
        range_hits: "list[tuple[bytes, int, list[tuple[int, int, int]]]]",
    ) -> bool:
        """Blocking restore from StagingTier into explicit engine ranges.

        ``range_hits`` entries are ``(content_hash, generation, ranges)``.
        This is used by engines whose cross-node content hash covers a
        collection of lower-level slots rather than one daemon block id.
        """
        if not range_hits:
            return True
        return self._client.restore_staging_ranges(self.engine_id, range_hits)

    def record_restore_staging(
        self,
        block_hits: "list[tuple[bytes, int, int]]",
    ) -> Optional[tuple[int, int]]:
        """Async restore from the daemon's StagingTier into engine HBM.

        ``block_hits`` entries are ``(content_hash, dest_block_id,
        generation)``. The daemon returns one-shot u32 source
        handles, then the restore ring carries ``(handle, dest)``
        pairs with FLAG_SOURCE_STAGING. Returns ``None`` if any handle
        cannot be registered or the ring is full; callers should fall
        back to recompute.
        """
        if not block_hits:
            return None
        handles = self._client.register_staging_restore_handles(
            [{"content_hash": h, "generation": gen} for h, _dst, gen in block_hits]
        )
        if len(handles) != len(block_hits) or any(h is None for h in handles):
            self._client.release_staging_restore_handles(
                [int(h) for h in handles if h is not None],
            )
            return None
        from gms_kv_ring.common.restore_ring import FLAG_SOURCE_STAGING

        pairs = [
            (int(h), int(dst))
            for h, (_content_hash, dst, _gen) in zip(handles, block_hits)
        ]
        try:
            result = self.record_restore(
                self.engine_id,
                pairs,
                flags=FLAG_SOURCE_STAGING,
            )
        except Exception:
            self._client.release_staging_restore_handles(
                [int(h) for h in handles if h is not None],
            )
            raise
        if result is None:
            self._client.release_staging_restore_handles(
                [int(h) for h in handles if h is not None],
            )
        return result

    def wait_restore(self, stream: int, slot: int, target: int) -> None:
        """Engine queues this on its compute stream BEFORE any kernel
        that consumes the restored bytes. Uses GEQ semantics: satisfies
        on either the success target (T) or the failure indicator (T+1),
        so the engine's compute stream never hangs if the daemon errors.

        Engine MUST call `restore_succeeded(slot, target)` host-side
        after syncing the stream to verify the restored bytes are
        valid — otherwise it may consume garbage on the daemon-error
        path."""
        self.counters.wait_value32(stream, slot, target)

    def restore_succeeded(self, slot: int, target: int) -> bool:
        """Returns True iff the most recent restore at (slot, target)
        completed successfully.

        Caller MUST have synced the stream that waited on this slot
        (e.g., torch.cuda.synchronize()) before calling — otherwise
        the read races the daemon's write.

        Implementation: daemon writes `target` on success, `target+1`
        on failure. We read the host-mapped counter byte and compare.
        Returns False also if the slot has been reused by a later
        record_restore (counter has advanced past target+1) — caller
        should treat that as failure and fall back to a cold path,
        since the bytes at the destination VA are no longer this
        request's bytes.

        Engine integration pattern:

            slot, target = handle.record_restore(src_id, pairs)
            handle.wait_restore(stream, slot, target)
            # ... queue downstream kernels on the stream ...
            stream.synchronize()
            if not handle.restore_succeeded(slot, target):
                # fall back to cold compute path
                ..."""
        return self.counters.read_slot(slot) == int(target)

    def current_daemon_epoch(self) -> Optional[int]:
        """Last `daemon_epoch` value seen on any RPC response, or
        None if no RPC has completed yet.

        Polled by the scheduler-side connector between RPCs to
        detect daemon restart. A change between two reads means the
        daemon process has been replaced and its in-memory state
        (slot generations, host-tier slots, restore consumer state)
        has been zeroed — every entry in the connector's prefix
        index points at slots the new daemon doesn't know about and
        must be dropped to avoid serving wrong KV bytes."""
        return self._client.current_daemon_epoch()

    def staging_scan(
        self,
        content_hashes: "list[bytes]",
    ) -> "dict[bytes, dict]":
        """Batch-query the daemon's staging tier (cross-node design
        Phase 2). Returns the subset of `content_hashes` currently
        READY in staging, each mapped to `{bytes_size, crc32,
        generation}`. Empty if staging isn't enabled on this daemon
        (standalone GMS without dynamo) or no hits.

        Called by the scheduler-side connector at
        `get_num_new_matched_tokens` time to extend a local prefix
        match with cross-node-fetched blocks. See
        `docs/CROSS_NODE_DESIGN.md` §4.3 for why this is a batch RPC
        instead of an SHM ring."""
        return self._client.staging_scan(content_hashes)

    # ---- GPU-direct storage path (GDS-capable backends only) ----

    def gpu_direct_storage_available(self) -> bool:
        """True iff the daemon's storage backend supports a
        GPU-direct data plane (e.g., NixlBackend with GDS/GDS_MT).
        Engine adapters call this once at attach time to decide
        whether to enable the demote_hbm_to_storage /
        promote_storage_to_hbm hot paths."""
        return bool(
            self._capabilities.get(
                "supports_gpu_direct",
                False,
            )
        )

    def demote_hbm_to_storage(
        self,
        layer: int,
        offset: int,
        size: int,
        *,
        generation: int = 0,
    ) -> bool:
        """Spill `size` bytes from HBM at (layer, offset) in this
        engine's pool directly to durable storage, skipping
        host_tier entirely. Requires
        `gpu_direct_storage_available()`; raises clearly otherwise.

        `generation`: caller-supplied generation tag for this slot.
        The daemon stores it; a later `promote_storage_to_hbm` with
        `expected_generation` checks for equality and refuses stale
        reads. `0` means "untagged" (legacy callers); they get no
        race-#3 protection on subsequent restores.

        Use this in the eviction policy when you have a block that
        won't be touched again soon AND you want it on durable
        storage rather than just host RAM. Compared to the
        host_tier path (record_evict + demote_to_storage), this
        avoids one pinned-RAM allocation and one host-tier index
        slot per spilled block.
        """
        return self._client.demote_hbm_to_storage(
            self.engine_id,
            layer,
            offset,
            size,
            generation=int(generation),
        )

    def promote_storage_to_hbm(
        self,
        layer: int,
        offset: int,
        size: int,
        dest_offset: Optional[int] = None,
        *,
        expected_generation: int = 0,
    ) -> bool:
        """Restore `size` bytes from durable storage at
        (engine_id, layer, offset) directly into the engine's HBM.
        Symmetric to `demote_hbm_to_storage`. Returns True on
        verified CRC; False on any failure (engine falls to cold
        compute on False).

        `dest_offset` (default: same as `offset`) lets the caller
        remap the destination: storage key remains `(engine_id,
        layer, offset)` but the bytes land at `(layer, dest_offset)`
        in the engine's pool. The connector's restore-on-hit path
        uses this to write a prefix-cached block into a freshly
        allocated block_id.

        `expected_generation`: if non-zero, the daemon enforces that
        the slot's stored generation equals this value before
        reading. Mismatch → returns False (the slot was overwritten
        by a later evict; reading would produce stale bytes).
        Protects against the cross-step async-restore-vs-evict race.
        """
        return self._client.promote_storage_to_hbm(
            self.engine_id,
            layer,
            offset,
            size,
            dest_offset=dest_offset,
            expected_generation=int(expected_generation),
        )

    # ---- lifecycle ----

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cleanup_resources(detach=True)

    def __enter__(self) -> "GMSKvRing":
        return self

    def __exit__(self, *_a) -> None:
        self.close()
