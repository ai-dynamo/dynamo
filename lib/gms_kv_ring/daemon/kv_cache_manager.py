# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS-managed KV-cache lifecycle and data-plane orchestration.

This component owns KV-specific state for N engines. The Unix-socket
server is a separate process shell which delegates RPCs here:

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
Engines do NOT serialize on each other — only on the server control
loop, which is only used for setup/teardown.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

from gms_kv_ring.daemon.consumers import (
    EnginePool,
    LayerDesc,
    _ensure_cuda_context,
    _EvictConsumer,
    _RestoreConsumer,
    _SourceClientPool,
)

logger = logging.getLogger(__name__)


class GmsKvCacheManager:
    """One-GPU KV-cache manager serving at most one active engine.

    The `_pools` dict remains keyed by engine_id to support
    failover replay: engine A active, attached as `_pools["A"]`;
    A dies; engine B starts up and attaches with the SAME id "A"
    (continuing from A's KV) or a fresh id (cold start). The
    dict can transiently hold two entries during teardown of A
    and attach of B, but only one engine actively drives the
    request path at any moment. Concurrent multi-active-engine
    is OUT OF SCOPE (see project_single_active_engine.md)."""

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        *,
        storage_backend=None,
        supervise_backend: bool = True,
        sweeper_interval_s: float = 0.0,
        sweeper_max_age_s: Optional[float] = None,
        sweeper_max_bytes: Optional[int] = None,
        sweeper_max_bytes_per_engine: Optional[int] = None,
        sweeper_compact_manifest_at_bytes: Optional[int] = None,
        staging_capacity_bytes: int = 0,
        staging_allocator: Optional["StagingAllocator"] = None,  # noqa: F821
        transport_listen_port: int = 0,
        transport_agent_name: Optional[str] = None,
        staging_receive_buffer_bytes: int = 0,
        staging_send_buffer_bytes: int = 0,
        placement_publisher: Optional["PlacementPublisher"] = None,  # noqa: F821
        daemon_id_for_placement: Optional[str] = None,
        host_tier_max_bytes: Optional[int] = None,
        host_tier_eviction_policy: Optional[str] = None,
    ) -> None:
        """`storage_backend`: any `StorageBackend` instance (NIXL,
        Mooncake, custom). Defaults to a `StorageTier` (local FS) at
        `storage_dir`.

        `supervise_backend`: wrap the backend in a `BackendSupervisor`
        so transient Python-level errors don't take down the daemon.
        Disable for tests that need direct access to backend internals.
        Note: this catches Python exceptions only — a C-level SIGSEGV
        from a transport library still kills the process."""
        from gms_kv_ring.daemon.host_tier import HostTier
        from gms_kv_ring.daemon.storage_tier import StorageSweeper, StorageTier
        from gms_kv_ring.daemon.supervisor import BackendSupervisor

        if host_tier_eviction_policy is None:
            host_tier_eviction_policy = os.environ.get(
                "GMS_KVR_HOST_TIER_EVICTION_POLICY",
                "lru",
            )
        try:
            self.host_tier = HostTier(
                eviction_policy=host_tier_eviction_policy,
            )
        except ValueError:
            logger.warning(
                "[Daemon] invalid GMS_KVR_HOST_TIER_EVICTION_POLICY=%r; using lru",
                host_tier_eviction_policy,
            )
            self.host_tier = HostTier(eviction_policy="lru")
        self.host_tier_eviction_policy = self.host_tier.eviction_policy
        if host_tier_max_bytes is None:
            try:
                host_tier_max_bytes = int(
                    os.environ.get("GMS_KVR_HOST_TIER_MAX_BYTES", "0"),
                )
            except ValueError:
                logger.warning(
                    "[Daemon] invalid GMS_KVR_HOST_TIER_MAX_BYTES=%r; "
                    "host-tier quota disabled",
                    os.environ.get("GMS_KVR_HOST_TIER_MAX_BYTES"),
                )
                host_tier_max_bytes = 0
        self.host_tier_max_bytes = max(0, int(host_tier_max_bytes or 0))
        # Optional staging tier (Phase 1 of cross-node). Enabled when
        # `staging_capacity_bytes > 0`. Owns its own state machine + scrub.
        # See `docs/CROSS_NODE_DESIGN.md` for the invariants it enforces
        # (I-8 atomic reservation, I-9 hash-on-receive verify, I-3' staging
        # generation, I-6' staging CRC scrub). Standalone GMS deployments
        # without dynamo's router don't benefit from staging_capacity > 0
        # — no one publishes inbound transfers.
        self.staging_tier = None
        if staging_capacity_bytes > 0:
            # Content-hash verification mode. `GMS_HASH_MODE` env var:
            #   sha256           - cryptographic; sync verify on critical path
            #   blake2b_256      - cryptographic; sync verify
            import os as _os

            from gms_kv_ring.daemon.staging_tier import StagingTier, hash_fn_for_mode

            hash_mode = _os.environ.get("GMS_HASH_MODE", "sha256")
            # Fail closed rather than silently selecting an algorithm which
            # may disagree with peers. Weak legacy modes are not identities.
            hash_fn = hash_fn_for_mode(hash_mode)
            self._hash_mode = hash_mode
            self.staging_tier = StagingTier(
                capacity_bytes=int(staging_capacity_bytes),
                allocator=staging_allocator,
                hash_fn=hash_fn,
                verify_on_receive=True,
            )
            logger.info(
                "[Daemon] staging_tier hash_mode=%s verify_on_receive=true",
                hash_mode,
            )

        # Cross-node transport stack (Phase 3 + 4). All three are
        # constructed together or not at all — they only make sense as
        # a set: transport carries bytes into receive_buf, callback
        # bridges into staging_tier, publisher advertises READY slots
        # to dynamo's indexer. Off by default for standalone GMS
        # (no router to drive transfers, no indexer to publish to).
        self.transport = None
        self.staging_receive_buffer = None
        self.placement_publisher = None
        # Active transfers: reservation_id -> (recv_offset, recv_size,
        # content_hash). Drained in _on_nixl_inbound.
        self._active_xfers: dict[str, tuple[int, int, bytes]] = {}
        self._xfers_lock = threading.Lock()
        # Cached DaemonClient pools per source UDS, used by
        # `fetch_remote` to issue `transfer_blocks_batch` RPCs to peer
        # daemons. Each pool holds N persistent sockets so we can
        # pipeline batches concurrently to one source. Lazily
        # populated; closed on daemon shutdown.
        self._source_pools: "dict[str, _SourceClientPool]" = {}
        self._source_pools_lock = threading.Lock()
        # Engine-direct path: hash → (ptr, size) for NIXL-registered
        # memory that peer engines can READ directly. Populated by
        # `register_bootstrap_handle`; consumed by `get_bootstrap_info`.
        # Key is the engine's content_hash, usually the shared stable
        # prefix hash rather than a digest of the raw KV bytes. ptr
        # points into engine HBM (via VMM-IPC) or daemon-owned tier
        # memory. NIXL-registered once at register time, then served
        # to peer engines as-is.
        self._bootstrap_handles: "dict[bytes, dict]" = {}
        self._bootstrap_lock = threading.Lock()
        # Source-pool size: per-source UDS socket count. 4 is enough
        # to pipeline ~4 transfer_blocks_batch RPCs in parallel; each
        # RPC is short (only carries submission ACK), so a small pool
        # serves a request rate of thousands/sec.
        self._source_pool_size = 4
        # Cross-node SEND side (P4c). Content-hash → host_tier address
        # index, populated by `register_content_address` RPC. The router
        # commands a transfer by passing a content hash; the daemon
        # looks up the addresses, reads from host_tier, memcpys into
        # `staging_send_buffer`, and pushes via NixlTransport.send.
        # Stores: hash → {"engine_id": str, "ranges": [(layer, offset, size), ...]}
        self._content_hash_index: dict[bytes, dict] = {}
        self._content_hash_lock = threading.Lock()
        # Destination restore path: small integer handle ->
        # (content_hash, staging_generation). The worker-side
        # connector registers handles just before pushing a
        # FLAG_SOURCE_STAGING restore record, because the restore ring
        # only has a u32 source field. Handles are one-shot and are
        # consumed by _RestoreConsumer._process_staging.
        self._staging_restore_handles: dict[int, tuple[bytes, int]] = {}
        self._staging_restore_lock = threading.Lock()
        self._next_staging_restore_handle = 1
        self.staging_send_buffer = None
        if (
            transport_listen_port > 0
            and staging_receive_buffer_bytes > 0
            and self.staging_tier is not None
        ):
            from gms_kv_ring.daemon.placement_publisher import LoggingPlacementPublisher
            from gms_kv_ring.daemon.staging_receive_buffer import StagingReceiveBuffer
            from gms_kv_ring.daemon.transport import (
                NixlTransport,
                TransportNotAvailable,
            )

            agent_name = (
                transport_agent_name or f"gms-{os.getpid()}-{transport_listen_port}"
            )
            daemon_id = daemon_id_for_placement or agent_name
            try:
                self.transport = NixlTransport(
                    agent_name=agent_name,
                    listen_port=int(transport_listen_port),
                    on_inbound_notif=self._on_nixl_inbound,
                )
            except TransportNotAvailable as exc:
                logger.warning(
                    "[Daemon] NIXL transport unavailable; cross-node disabled. %s",
                    exc,
                )
            if self.transport is not None:
                self.staging_receive_buffer = StagingReceiveBuffer(
                    capacity_bytes=int(staging_receive_buffer_bytes),
                )
                # One-time registration: peers see this buffer as soon
                # as they exchange metadata with us.
                self.transport.register_buffer(
                    self.staging_receive_buffer.base_ptr(),
                    self.staging_receive_buffer.capacity(),
                    label="staging_recv",
                )
                self.placement_publisher = (
                    placement_publisher
                    or LoggingPlacementPublisher(
                        daemon_id=daemon_id,
                        daemon_epoch=0,  # set by daemon-epoch logic later
                    )
                )
                # Sender-side scratch (P4c): pre-registered staging
                # area for outbound transfers. Same allocator shape as
                # the receive buffer; bump-then-coalesce. Size = max
                # in-flight outbound bytes. Off when 0.
                if staging_send_buffer_bytes > 0:
                    self.staging_send_buffer = StagingReceiveBuffer(
                        capacity_bytes=int(staging_send_buffer_bytes),
                    )
                    self.transport.register_buffer(
                        self.staging_send_buffer.base_ptr(),
                        self.staging_send_buffer.capacity(),
                        label="staging_send",
                    )
        # Storage tier lives under storage_dir if provided, otherwise
        # under a per-daemon tmpdir (tests). Production deployments
        # pass a stable path on persistent disk.
        if storage_backend is None:
            if storage_dir is None:
                import tempfile

                storage_dir = tempfile.mkdtemp(prefix="gms-kvr-storage-")
                self._owns_storage_dir = True
            else:
                self._owns_storage_dir = False
            storage_backend = StorageTier(storage_dir)
            self._backend_factory = lambda: StorageTier(storage_dir)
        else:
            self._owns_storage_dir = False
            self._backend_factory = None  # caller-supplied; no restart
        # `storage_tier` is the engine-facing name (it IS the storage
        # tier, regardless of which backend implements it). Type is
        # StorageBackend or BackendSupervisor wrapping one.
        if supervise_backend:
            self.storage_tier = BackendSupervisor(
                storage_backend,
                factory=self._backend_factory,
            )
        else:
            self.storage_tier = storage_backend
        # Optional background cleanup. Off by default; turn on with
        # interval > 0 AND at least one of max_age_s / max_bytes.
        # We construct the sweeper here but only start() it from
        # serve() so the thread lifetime is bounded by daemon serve.
        self._sweeper: Optional[StorageSweeper] = None
        if sweeper_interval_s > 0 and (
            sweeper_max_age_s is not None
            or sweeper_max_bytes is not None
            or sweeper_max_bytes_per_engine is not None
            or sweeper_compact_manifest_at_bytes is not None
        ):
            self._sweeper = StorageSweeper(
                self.storage_tier,
                interval_s=sweeper_interval_s,
                max_age_s=sweeper_max_age_s,
                max_bytes=sweeper_max_bytes,
                max_bytes_per_engine=sweeper_max_bytes_per_engine,
                compact_manifest_at_bytes=sweeper_compact_manifest_at_bytes,
                on_sweep=self._sweeper_metric_hook,
            )
        self._pools: dict[str, EnginePool] = {}
        # Durable-storage slots survive engine detach/reattach, so the
        # generation guards for those slots must survive too. This map is
        # intentionally daemon-memory only: daemon restart changes the daemon
        # epoch, which makes engine-side prefix indexes drop stale entries.
        self._storage_slot_generations: dict[str, dict[int, int]] = {}
        self._lock = threading.Lock()
        # Daemon epoch: a per-process-startup token piggybacked on
        # every RPC response. The connector tracks the last-seen
        # value and invalidates its in-memory `_PrefixIndex` when it
        # changes — which means "you're talking to a different daemon
        # instance than the one that wrote the indexed slots." This
        # closes the silent-wrong-output gap for daemon-restart and
        # for scheduler-restart-with-stale-snapshot (the snapshot
        # embeds the epoch it was written under). The clock-derived
        # value is unique per restart at microsecond resolution; the
        # connector compares for inequality, never order.
        self.epoch: int = int(time.time() * 1_000_000)
        # Background scrub thread for host_tier slots. Re-CRCs slots
        # at a low rate; drops any whose in-RAM bytes have diverged
        # from the stored CRC (proactive at-rest corruption
        # detection). Off the engine's critical path — the engine
        # forward pass never waits on this thread. Disabled when
        # rate=0 (default for tests).
        try:
            scrub_interval = float(
                os.environ.get(
                    "GMS_KVR_SCRUB_INTERVAL_S",
                    "0.0",
                ),
            )
        except ValueError:
            scrub_interval = 0.0
        try:
            scrub_idle_s = float(
                os.environ.get(
                    "GMS_KVR_SCRUB_IDLE_S",
                    "0.05",
                ),
            )
        except ValueError:
            scrub_idle_s = 0.05
        self._scrub_interval_s: float = max(0.0, scrub_interval)
        self._scrub_idle_s: float = max(0.0, scrub_idle_s)
        self._scrub_thread: Optional[threading.Thread] = None
        self._scrub_stop = threading.Event()
        # Backend scrub: reads each durable slot back from storage,
        # CRC-verifies against the stored CRC, drops mismatches.
        # The CRITICAL path for NIXL-GDS production: GDS reads go
        # disk → GPU HBM, bypassing host-side CRC verify; backend
        # scrub is the only proactive at-rest corruption detector
        # for those deployments. Off by default (it's I/O-heavy and
        # most deployments use the host_tier path which already
        # verifies on every read).
        try:
            backend_scrub_interval = float(
                os.environ.get(
                    "GMS_KVR_BACKEND_SCRUB_INTERVAL_S",
                    "0.0",
                ),
            )
        except ValueError:
            backend_scrub_interval = 0.0
        try:
            backend_scrub_idle_s = float(
                os.environ.get(
                    "GMS_KVR_BACKEND_SCRUB_IDLE_S",
                    "0.1",
                ),
            )
        except ValueError:
            backend_scrub_idle_s = 0.1
        self._backend_scrub_interval_s: float = max(
            0.0,
            backend_scrub_interval,
        )
        self._backend_scrub_idle_s: float = max(
            0.0,
            backend_scrub_idle_s,
        )
        self._backend_scrub_thread: Optional[threading.Thread] = None
        self._backend_scrub_stop = threading.Event()
        # Reusable host buffer for backend scrub. Grown lazily to the
        # largest slot size seen; freed at process exit. Avoids
        # cudaHostAlloc churn per slot (scrub doesn't need pinned
        # memory — the backend's host-staged read works with malloc).
        self._scrub_buf_ptr: int = 0
        self._scrub_buf_size: int = 0
        self._started = False
        self._closed = False
        self._connections_closed = False

    # ---- host-tier content-hash index maintenance ----

    def _drop_content_hashes_for_host_keys(
        self,
        keys: set[tuple[str, int, int]],
        *,
        tier: str = "host_pinned",
    ) -> int:
        """Remove content-hash entries that reference released host slots."""
        if not keys:
            return 0
        normalized = {(str(e), int(layer), int(o)) for e, layer, o in keys}
        removed: list[bytes] = []
        with self._content_hash_lock:
            for content_hash, entry in list(self._content_hash_index.items()):
                engine_id = str(entry.get("engine_id", ""))
                ranges = entry.get("ranges") or []
                if any(
                    (engine_id, int(layer), int(offset)) in normalized
                    for layer, offset, _size in ranges
                ):
                    self._content_hash_index.pop(content_hash, None)
                    removed.append(content_hash)
        if self.placement_publisher is not None:
            for content_hash in removed:
                try:
                    self.placement_publisher.publish_removed(
                        content_hash=content_hash,
                        tier=tier,
                    )
                except Exception:
                    logger.exception(
                        "[Daemon] publish_removed failed for host index drop",
                    )
        return len(removed)

    def _drop_content_hashes_for_engine(
        self,
        engine_id: str,
        *,
        tier: str = "host_pinned",
    ) -> int:
        removed: list[bytes] = []
        with self._content_hash_lock:
            for content_hash, entry in list(self._content_hash_index.items()):
                if str(entry.get("engine_id", "")) == str(engine_id):
                    self._content_hash_index.pop(content_hash, None)
                    removed.append(content_hash)
        if self.placement_publisher is not None:
            for content_hash in removed:
                try:
                    self.placement_publisher.publish_removed(
                        content_hash=content_hash,
                        tier=tier,
                    )
                except Exception:
                    logger.exception(
                        "[Daemon] publish_removed failed for engine drop",
                    )
        return len(removed)

    def _enforce_host_tier_quota(
        self,
        protected_keys: Optional[set[tuple[str, int, int]]] = None,
    ) -> int:
        if self.host_tier_max_bytes <= 0:
            return 0
        evicted = self.host_tier.evict_until_under(
            self.host_tier_max_bytes,
            protected_keys=protected_keys,
        )
        if not evicted:
            return 0
        dropped = self._drop_content_hashes_for_host_keys(set(evicted))
        logger.info(
            "[Daemon] host-tier quota evicted %d slots (%d content hashes); "
            "policy=%s bytes=%d max=%d",
            len(evicted),
            dropped,
            self.host_tier_eviction_policy,
            self.host_tier.total_bytes(),
            self.host_tier_max_bytes,
        )
        return len(evicted)

    # ---- host-tier scrub (background CRC re-verification) ----

    def _scrub_loop(self) -> None:  # Invariant I-6 (see docs/ARCHITECTURE.md)
        """Walk every ready host_tier slot at a low cadence,
        recompute crc32, drop slots whose stored CRC has diverged
        from the in-RAM bytes. Slot eviction here AND the metric
        bump are the only side effects — the engine forward pass
        never waits on this thread.

        Enforces **invariant I-6** (at-rest bytes match captured CRC
        or are dropped) at the host tier. The backend (durable) half
        of I-6 is enforced by `_backend_scrub_loop` below.

        Each scrubbed slot incurs ~1ms of CPU per MB; in steady
        state the thread spends most of its time in `Event.wait`.
        Between slots we sleep `_scrub_idle_s` to keep CPU usage
        bounded; between full scans we sleep `_scrub_interval_s`.
        """
        from gms_kv_ring.common import metrics
        from gms_kv_ring.common.checksum import crc32_at_ptr

        while not self._scrub_stop.is_set():
            keys = self.host_tier.snapshot_keys()
            for key in keys:
                if self._scrub_stop.is_set():
                    return
                engine_id, layer, offset = key
                lease = self.host_tier.pin(engine_id, layer, offset)
                if lease is None:
                    continue
                actual = None
                expected_crc = 0
                try:
                    with lease as slot:
                        if slot.crc == 0 or slot.size == 0:
                            continue
                        expected_crc = int(slot.crc)
                        actual = crc32_at_ptr(slot.host_ptr, slot.size)
                except Exception:  # noqa: BLE001
                    # Pointer might have been retired by a concurrent
                    # release before we pinned it; skip rather than
                    # crash the scrubber.
                    continue
                metrics.daemon_scrub_scanned.inc(engine_id=engine_id)
                if actual != expected_crc:
                    logger.error(
                        "[GMS scrub] CRC drift detected eng=%r "
                        "layer=%d off=%d expected=%#x actual=%#x; "
                        "dropping slot to prevent serving wrong "
                        "bytes on future cache-hit reads.",
                        engine_id,
                        layer,
                        offset,
                        int(expected_crc),
                        int(actual),
                    )
                    if self.host_tier.release_if_current(lease):
                        self._drop_content_hashes_for_host_keys(
                            {
                                (engine_id, layer, offset),
                            }
                        )
                    metrics.daemon_scrub_corruptions.inc(
                        engine_id=engine_id,
                    )
                # Pace ourselves so a single scrub pass doesn't
                # monopolize a CPU on a host with millions of slots.
                if self._scrub_idle_s > 0:
                    if self._scrub_stop.wait(self._scrub_idle_s):
                        return
            # End-of-pass sleep before the next full scan. A
            # zero-slot pool also lands here, so we still wake up
            # periodically for new slots.
            if self._scrub_stop.wait(self._scrub_interval_s):
                return

    def scrub_once(self) -> tuple[int, int]:
        """One-shot scrub pass over every ready host_tier slot.
        Returns `(scanned, corruptions)`. Test-callable handle on
        the same logic the background thread runs — without
        requiring a thread + wall-clock wait. Production code
        should rely on `_start_scrub()` instead."""
        from gms_kv_ring.common import metrics
        from gms_kv_ring.common.checksum import crc32_at_ptr

        scanned = 0
        corruptions = 0
        for key in self.host_tier.snapshot_keys():
            engine_id, layer, offset = key
            lease = self.host_tier.pin(engine_id, layer, offset)
            if lease is None:
                continue
            actual = None
            expected_crc = 0
            try:
                with lease as slot:
                    if slot.crc == 0 or slot.size == 0:
                        continue
                    expected_crc = int(slot.crc)
                    actual = crc32_at_ptr(slot.host_ptr, slot.size)
            except Exception:  # noqa: BLE001
                continue
            scanned += 1
            metrics.daemon_scrub_scanned.inc(engine_id=engine_id)
            if actual != expected_crc:
                logger.error(
                    "[GMS scrub] CRC drift detected eng=%r "
                    "layer=%d off=%d expected=%#x actual=%#x; "
                    "dropping slot.",
                    engine_id,
                    layer,
                    offset,
                    int(expected_crc),
                    int(actual),
                )
                if self.host_tier.release_if_current(lease):
                    self._drop_content_hashes_for_host_keys(
                        {
                            (engine_id, layer, offset),
                        }
                    )
                metrics.daemon_scrub_corruptions.inc(
                    engine_id=engine_id,
                )
                corruptions += 1
        return scanned, corruptions

    def _start_scrub(self) -> None:
        if self._scrub_interval_s <= 0:
            return
        if self._scrub_thread is not None and self._scrub_thread.is_alive():
            return
        self._scrub_stop.clear()
        self._scrub_thread = threading.Thread(
            target=self._scrub_loop,
            name="gms-host-tier-scrub",
            daemon=True,
        )
        self._scrub_thread.start()
        logger.info(
            "host_tier scrub thread started (interval=%.1fs idle=%.3fs)",
            self._scrub_interval_s,
            self._scrub_idle_s,
        )

    def _stop_scrub(self) -> None:
        self._scrub_stop.set()
        t = self._scrub_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._scrub_thread = None

    # ---- backend scrub (durable storage CRC re-verification) ----

    def _ensure_scrub_buf(self, size: int) -> int:
        """Grow `self._scrub_buf_ptr` to at least `size` bytes. Plain
        malloc — the backend's `promote()` reads via POSIX/cuFile-
        to-host into this buffer; pinned memory isn't required for
        the scrub use case (no engine-side consumption of the
        bytes), and avoiding cudaHostAlloc churn matters when
        scrubbing millions of slots."""
        import ctypes

        if size <= self._scrub_buf_size and self._scrub_buf_ptr != 0:
            return self._scrub_buf_ptr
        # Free old (if any) and re-allocate.
        if self._scrub_buf_ptr != 0:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.free(ctypes.c_void_p(self._scrub_buf_ptr))
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.malloc.restype = ctypes.c_void_p
        new_size = max(size, 4096)
        ptr = libc.malloc(ctypes.c_size_t(new_size))
        if not ptr:
            raise MemoryError(
                f"scrub buffer malloc({new_size}) failed",
            )
        self._scrub_buf_ptr = int(ptr)
        self._scrub_buf_size = int(new_size)
        return self._scrub_buf_ptr

    def scrub_backend_once(self) -> tuple[int, int]:
        """One-shot scrub over every backend slot. Returns
        `(scanned, corruptions)`. Each slot is read into a host
        buffer via the backend's existing `promote()` (which
        already does header + payload CRC verify); on a returned
        None the slot is dropped via `release_slot()`.

        Test-callable handle on the same logic the background
        thread runs. Production code uses `_start_backend_scrub()`.
        """
        from gms_kv_ring.common import metrics

        scanned = 0
        corruptions = 0
        try:
            keys = self.storage_tier.snapshot_keys()
        except Exception:  # noqa: BLE001
            return 0, 0
        for key in keys:
            engine_id, layer, offset = key
            try:
                slot = self.storage_tier.get(engine_id, layer, offset)
            except Exception:  # noqa: BLE001
                continue
            if slot is None or int(slot.size) <= 0:
                continue
            try:
                buf = self._ensure_scrub_buf(int(slot.size))
            except MemoryError:
                logger.warning(
                    "scrub_backend_once: malloc failed for slot size=%d; aborting pass",
                    int(slot.size),
                )
                return scanned, corruptions
            try:
                verified = self.storage_tier.promote(
                    engine_id,
                    layer,
                    offset,
                    buf,
                    int(slot.size),
                )
            except Exception:  # noqa: BLE001
                # A backend exception during promote is treated as
                # corruption: the slot is unreadable. Drop it so
                # cache-hit reads don't retry forever.
                verified = None
            scanned += 1
            metrics.daemon_backend_scrub_scanned.inc(
                engine_id=engine_id,
            )
            if verified is None:
                logger.error(
                    "[GMS backend-scrub] CRC verify failed for "
                    "eng=%r layer=%d off=%d size=%d; dropping "
                    "slot to prevent serving wrong bytes on future "
                    "cache-hit reads. THIS IS THE KEY GUARD FOR "
                    "NIXL-GDS PRODUCTION: the GDS read path "
                    "bypasses host-side CRC verify, so without "
                    "backend scrub a corrupt slot would silently "
                    "serve wrong KV bytes to the engine.",
                    engine_id,
                    layer,
                    offset,
                    int(slot.size),
                )
                try:
                    self.storage_tier.release_if_current(
                        engine_id,
                        layer,
                        offset,
                        slot,
                    )
                except Exception:  # noqa: BLE001
                    pass
                metrics.daemon_backend_scrub_corruptions.inc(
                    engine_id=engine_id,
                )
                corruptions += 1
        return scanned, corruptions

    def _backend_scrub_loop(self) -> None:  # Invariant I-6 (see docs/ARCHITECTURE.md)
        """Background loop. One pass over backend slots per cycle;
        sleeps `_backend_scrub_interval_s` between cycles and
        `_backend_scrub_idle_s` between slots to keep daemon I/O
        load bounded.

        Enforces the durable-storage half of **invariant I-6** (at-rest
        bytes match captured CRC or are dropped). The only at-rest
        corruption guard for NIXL-GDS production, where reads go
        disk→GPU directly and bypass host-side CRC verify.

        Bandwidth model: scrubbing a 10 GB backend at idle=0.1s
        with 1 MB slots takes ~17 minutes per pass. Tune
        `GMS_KVR_BACKEND_SCRUB_IDLE_S` down for faster coverage on
        SSD-only deployments; up for HDD/networked storage to
        avoid stealing bandwidth from cache-hit reads."""
        from gms_kv_ring.common import metrics

        while not self._backend_scrub_stop.is_set():
            try:
                keys = self.storage_tier.snapshot_keys()
            except Exception:  # noqa: BLE001
                keys = []
            for key in keys:
                if self._backend_scrub_stop.is_set():
                    return
                engine_id, layer, offset = key
                try:
                    slot = self.storage_tier.get(
                        engine_id,
                        layer,
                        offset,
                    )
                except Exception:  # noqa: BLE001
                    continue
                if slot is None or int(slot.size) <= 0:
                    continue
                try:
                    buf = self._ensure_scrub_buf(int(slot.size))
                except MemoryError:
                    logger.warning(
                        "_backend_scrub_loop: malloc failed; skipping pass",
                    )
                    break
                try:
                    verified = self.storage_tier.promote(
                        engine_id,
                        layer,
                        offset,
                        buf,
                        int(slot.size),
                    )
                except Exception:  # noqa: BLE001
                    verified = None
                metrics.daemon_backend_scrub_scanned.inc(
                    engine_id=engine_id,
                )
                if verified is None:
                    logger.error(
                        "[GMS backend-scrub] CRC verify failed "
                        "eng=%r layer=%d off=%d size=%d; dropping.",
                        engine_id,
                        layer,
                        offset,
                        int(slot.size),
                    )
                    try:
                        self.storage_tier.release_if_current(
                            engine_id,
                            layer,
                            offset,
                            slot,
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    metrics.daemon_backend_scrub_corruptions.inc(
                        engine_id=engine_id,
                    )
                # Per-slot pacing.
                if self._backend_scrub_idle_s > 0:
                    if self._backend_scrub_stop.wait(
                        self._backend_scrub_idle_s,
                    ):
                        return
            # End-of-pass sleep.
            if self._backend_scrub_stop.wait(
                self._backend_scrub_interval_s,
            ):
                return

    def _start_backend_scrub(self) -> None:
        if self._backend_scrub_interval_s <= 0:
            return
        if (
            self._backend_scrub_thread is not None
            and self._backend_scrub_thread.is_alive()
        ):
            return
        self._backend_scrub_stop.clear()
        self._backend_scrub_thread = threading.Thread(
            target=self._backend_scrub_loop,
            name="gms-backend-scrub",
            daemon=True,
        )
        self._backend_scrub_thread.start()
        logger.info(
            "backend scrub thread started (interval=%.1fs idle=%.3fs)",
            self._backend_scrub_interval_s,
            self._backend_scrub_idle_s,
        )

    def _stop_backend_scrub(self) -> None:
        self._backend_scrub_stop.set()
        t = self._backend_scrub_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._backend_scrub_thread = None
        # Free the scrub buffer.
        if self._scrub_buf_ptr != 0:
            import ctypes

            try:
                libc = ctypes.CDLL("libc.so.6", use_errno=True)
                libc.free(ctypes.c_void_p(self._scrub_buf_ptr))
            except Exception:  # noqa: BLE001
                pass
            self._scrub_buf_ptr = 0
            self._scrub_buf_size = 0

    def _sweeper_metric_hook(
        self,
        ttl_n: int,
        per_eng_n: int,
        quota_n: int,
    ) -> None:
        """Bump the same eviction metric `prune_storage` uses, so a
        Prometheus scrape sees background sweeps and explicit RPCs
        in one unified counter (distinguished by `reason` label).
        Also refreshes the `storage_tier_bytes` gauge."""
        from gms_kv_ring.common import metrics

        if ttl_n:
            metrics.storage_tier_evictions.inc(reason="ttl", n=ttl_n)
        if per_eng_n:
            metrics.storage_tier_evictions.inc(
                reason="per_engine_quota",
                n=per_eng_n,
            )
        if quota_n:
            metrics.storage_tier_evictions.inc(reason="quota", n=quota_n)
        metrics.storage_tier_bytes.set(self.storage_tier.total_bytes())

    # ---- engine pool management ----

    def attach_engine_pool(
        self,
        engine_id: str,
        layers: list[LayerDesc],
    ) -> EnginePool:
        from cuda.bindings import driver as drv
        from gms_kv_ring.common import metrics

        _ensure_cuda_context()
        # An attach for an engine_id that already has a pool means the
        # engine restarted (process died, supervisor respawned with the
        # same id). Fully tear down the prior incarnation first —
        # otherwise host-tier slots from the dead process's HBM layout
        # are still keyed by this engine_id and would feed garbage into
        # the new process on a subsequent restore. Stops consumers,
        # destroys the prior stream, frees its host-tier slots.
        with self._lock:
            had_prior = engine_id in self._pools
            persisted_generations = dict(
                self._storage_slot_generations.get(engine_id, {})
            )
        if had_prior:
            metrics.daemon_engine_reattaches.inc(engine_id=engine_id)
            logger.warning(
                "reattaching engine %r while a pool is still attached; "
                "detaching prior incarnation before accepting new pool",
                engine_id,
            )
            self.detach_engine_pool(engine_id)
        try:
            storage_bytes = int(self.storage_tier.bytes_by_engine().get(engine_id, 0))
        except Exception:  # noqa: BLE001
            storage_bytes = -1
            logger.debug(
                "attach %r: failed to snapshot storage bytes for observability",
                engine_id,
                exc_info=True,
            )
        with self._lock:
            err, stream = drv.cuStreamCreate(0)
            if err != drv.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"cuStreamCreate failed for engine {engine_id!r}: {err}",
                )
            pool = EnginePool(
                engine_id=engine_id,
                layers={ld.layer_idx: ld for ld in layers},
                stream=int(stream),
            )
            pool.slot_generations.update(persisted_generations)
            self._pools[engine_id] = pool
            metrics.daemon_engine_attaches.inc(engine_id=engine_id)
            metrics.daemon_persisted_generations.set(
                len(pool.slot_generations),
                engine_id=engine_id,
            )
            logger.info(
                "attached engine %r: layers=%d stream=0x%x reattach=%s "
                "persisted_generations=%d storage_tier_total_slots=%d "
                "storage_bytes_for_engine=%d",
                engine_id,
                len(pool.layers),
                int(stream),
                had_prior,
                len(pool.slot_generations),
                self.storage_tier.n_slots(),
                storage_bytes,
            )
            return pool

    def detach_engine_pool(self, engine_id: str) -> bool:
        from gms_kv_ring.common import metrics

        with self._lock:
            pool = self._pools.pop(engine_id, None)
        if pool is None:
            logger.debug("detach engine %r: no attached pool", engine_id)
            return False
        if pool.evict_consumer is not None:
            pool.evict_consumer.stop()
        if pool.restore_consumer is not None:
            pool.restore_consumer.stop()
        sync_ok = True
        if pool.stream:
            from cuda.bindings import driver as drv

            try:
                # Drain pending work BEFORE destroy + free. cuStreamDestroy
                # is non-blocking and leaves in-flight ops to complete in
                # the background; without an explicit sync, the next
                # daemon's freshly-created stream can wedge waiting on
                # this stream's residual work.
                drv.cuStreamSynchronize(pool.stream)
            except Exception:  # noqa: BLE001
                sync_ok = False
                logger.warning(
                    "detach %r: cuStreamSynchronize failed — leaking "
                    "host_tier slots to avoid cudaFreeHost while DMA "
                    "may still be in flight (process restart will reclaim)",
                    engine_id,
                    exc_info=True,
                )
            try:
                drv.cuStreamDestroy(pool.stream)
            except Exception:  # noqa: BLE001
                logger.debug("cuStreamDestroy failed", exc_info=True)
        if sync_ok:
            freed = self.host_tier.release_engine(engine_id)
            self._drop_content_hashes_for_engine(engine_id)
            # Storage tier survives detach by design — that's the
            # whole point of a persistent tier. Engine restart /
            # daemon restart re-indexes the files via reconcile and
            # the engine can promote durable blocks back. Operators
            # cull stale storage out of band (quota / TTL / explicit
            # release_engine_storage RPC).
            logger.info(
                "detached engine %r: freed %d host-tier slots; "
                "storage-tier files preserved across detach (%d)",
                engine_id,
                freed,
                self.storage_tier.n_slots(),
            )
        else:
            # Pop the slots from the index so they aren't reused, but
            # don't actually cudaFreeHost — the buffer may still be the
            # target of an in-flight cuMemcpyAsync we couldn't drain.
            leaked = self.host_tier.forget_engine(engine_id)
            self._drop_content_hashes_for_engine(engine_id)
            logger.warning(
                "detached engine %r: forgot (leaked) %d host_tier slots",
                engine_id,
                leaked,
            )
            from gms_kv_ring.common import metrics

            if leaked:
                metrics.host_tier_leaked_slots.inc(
                    engine_id=engine_id,
                    n=leaked,
                )
        metrics.daemon_engine_detaches.inc(engine_id=engine_id)
        metrics.daemon_persisted_generations.set(
            len(self._storage_slot_generations.get(engine_id, {})),
            engine_id=engine_id,
        )
        return True

    # ---- tier transitions: host_tier <-> storage_tier ----

    def demote_to_storage(
        self,
        engine_id: str,
        layer: int,
        offset: int,
    ) -> bool:
        """AT_REST_HOST -> AT_REST_STORAGE.

        Reads the host_tier slot (must be ready, must have a stored
        CRC), writes a CRC-stamped file to the storage tier, then
        frees the host_tier slot. Returns False if the host slot is
        missing or not-ready (no demote — block is not durably in
        host_tier yet).

        Note: this is a SYNCHRONOUS RPC. File I/O blocks the calling
        executor thread; the asyncio control loop is unaffected
        because we run dispatch via run_in_executor. For
        production-scale demotion volumes, layer a worker pool above
        this — the wire shape doesn't change."""
        from gms_kv_ring.common import metrics

        lease = self.host_tier.pin(engine_id, layer, offset)
        if lease is None:
            return False
        # The slot's stored CRC came from the evict consumer's
        # CRC32-over-pinned-bytes after the D2H sync. We hand it
        # through unchanged — single source of truth.
        with lease as slot:
            self.storage_tier.demote(
                engine_id,
                layer,
                offset,
                host_ptr=slot.host_ptr,
                size=slot.size,
                crc=slot.crc,
            )
        released = self.host_tier.release_if_current(lease)
        if released:
            self._drop_content_hashes_for_host_keys(
                {
                    (engine_id, int(layer), int(offset)),
                }
            )
        metrics.storage_tier_slots.set(
            self.storage_tier.n_slots(),
            engine_id=engine_id,
        )
        return True

    def promote_from_storage(
        self,
        engine_id: str,
        layer: int,
        offset: int,
    ) -> bool:
        """AT_REST_STORAGE -> AT_REST_HOST.

        Allocates a host_tier slot, reads the storage file into it,
        verifies CRC. On success, marks the host_tier slot ready
        with the verified CRC and releases the storage slot. On
        failure (file missing/corrupted/CRC mismatch), releases the
        provisional host_tier slot and bumps a metric — the engine
        observing a missing host_tier slot will fall to cold compute.

        Bytes-in-flight ordering: the slot is marked ready only AFTER
        the memcpy + CRC verify, so a concurrent restore consumer
        that races us sees not-ready and signals failure rather than
        consuming partial bytes."""
        from gms_kv_ring.common import metrics

        ss = self.storage_tier.get(engine_id, layer, offset)
        if ss is None:
            return False
        write = self.host_tier.reserve(
            engine_id,
            layer,
            offset,
            ss.size,
        )
        verified_crc = self.storage_tier.promote(
            engine_id,
            layer,
            offset,
            dest_host_ptr=write.host_ptr,
            max_size=ss.size,
        )
        if verified_crc is None:
            self.host_tier.abort(write)
            metrics.storage_tier_promote_failures.inc(
                engine_id=engine_id,
            )
            return False
        if not self.host_tier.commit(write, verified_crc):
            return False
        self._enforce_host_tier_quota(
            {
                (engine_id, int(layer), int(offset)),
            }
        )
        # Storage slot is now redundant — free it. Future demote
        # rewrites if the block is spilled again.
        self.storage_tier.release_if_current(engine_id, layer, offset, ss)
        metrics.storage_tier_slots.set(
            self.storage_tier.n_slots(),
            engine_id=engine_id,
        )
        return True

    # ---- HBM-direct demote (GPUDirect Storage path) ----

    def demote_hbm_to_storage(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        size: int,
        *,
        generation: int = 0,
    ) -> bool:
        """HBM_FRESH -> AT_REST_STORAGE in one hop, skipping
        host_tier.

        Available only when the storage backend supports a
        GPU-direct data plane (e.g., NixlBackend with GDS/GDS_MT
        plugins). For everything else, raises a clear error
        directing callers at the standard 2-step path
        (engine evict -> host_tier -> demote_to_storage).

        Fast path (overlapped): when the backend exposes
        `demote_from_gpu_deferred_crc` + `commit_deferred_crc`, we
        kick off the GDS transfer with a CRC=0 placeholder
        concurrently with an async D2H into a pinned scratch buffer.
        While the GDS PCIe DMA is in flight we hash the host copy
        and commit the real CRC by patching the file header before
        the atomic rename — so no reader ever sees a CRC=0 slot.
        Net wall-clock approaches `max(transfer, CRC)` instead of
        their sum.

        Slow path: a plain backend that only exposes
        `demote_from_gpu` does the CRC pre-compute synchronously
        (D2H + zlib) before the GDS write. Correct but not
        overlapped — kept as fallback."""
        from gms_kv_ring.common import metrics
        from gms_kv_ring.common.checksum import crc32_at_ptr

        backend = (
            self.storage_tier.backend
            if hasattr(self.storage_tier, "backend")
            else self.storage_tier
        )
        if not backend.supports_gpu_direct():
            raise RuntimeError(
                f"demote_hbm_to_storage requires a GPU-direct storage "
                f"backend (e.g., NixlBackend with plugin in "
                f"{{GDS, GDS_MT}}); current backend {backend.name!r} "
                f"doesn't support it. Use the standard host_tier path "
                f"(record_evict + demote_to_storage) instead."
            )
        with self._lock:
            pool = self._pools.get(engine_id)
        if pool is None:
            raise ValueError(f"engine {engine_id!r} not attached")
        ld = pool.layers.get(int(layer))
        if ld is None:
            raise ValueError(f"layer {layer} not in engine {engine_id!r}'s pool")
        generation_int = int(generation or 0)
        block_id = int(offset) // int(ld.stride)
        if generation_int:
            current_generation = pool.current_block_generation(block_id)
            if current_generation > generation_int:
                logger.warning(
                    "demote_hbm_to_storage: stale generation rejected "
                    "eng=%r layer=%d block=%d offset=%d size=%d "
                    "generation=%d current=%d",
                    engine_id,
                    int(layer),
                    block_id,
                    int(offset),
                    int(size),
                    generation_int,
                    current_generation,
                )
                metrics.daemon_stale_demotes.inc(engine_id=engine_id)
                return False
        src_va = ld.va + int(offset)

        from cuda.bindings import runtime as rt
        from gms_kv_ring.common import pinned_scratch

        # Preferred path: overlap GDS transfer with async D2H+CRC.
        if hasattr(backend, "demote_from_gpu_deferred_crc"):
            with pinned_scratch.acquire(int(size)) as scratch_ptr:
                # 1. Async D2H on a fresh stream so the GDS write
                #    (which runs on the backend's own internal
                #    queue / cuFile path) is free to proceed in
                #    parallel.
                err, stream = rt.cudaStreamCreate()
                if err != rt.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"cudaStreamCreate failed: {err}")
                try:
                    err = rt.cudaMemcpyAsync(
                        int(scratch_ptr),
                        int(src_va),
                        int(size),
                        rt.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        int(stream),
                    )[0]
                    if err != rt.cudaError_t.cudaSuccess:
                        raise RuntimeError(
                            f"cudaMemcpyAsync (D2H for CRC) failed: {err}"
                        )
                    # 2. Kick the GDS write with placeholder CRC.
                    #    This blocks until the NIXL transfer is
                    #    DONE, but the async D2H above runs
                    #    concurrently on stream + the GDS PCIe DMA
                    #    on whatever channel cuFile uses.
                    handle = backend.demote_from_gpu_deferred_crc(
                        engine_id,
                        layer,
                        offset,
                        gpu_ptr=src_va,
                        size=int(size),
                    )
                    # 3. By now (or sooner) the D2H is done. Sync
                    #    the stream and compute the real CRC.
                    err = rt.cudaStreamSynchronize(int(stream))[0]
                    if err != rt.cudaError_t.cudaSuccess:
                        raise RuntimeError(f"cudaStreamSynchronize failed: {err}")
                    real_crc = crc32_at_ptr(
                        int(scratch_ptr),
                        int(size),
                    )
                    # 4. Patch the header and atomically commit.
                    backend.commit_deferred_crc(handle, real_crc)
                    metrics.demote_hbm_overlapped.inc(
                        engine_id=engine_id,
                    )
                finally:
                    rt.cudaStreamDestroy(int(stream))
        else:
            # Fallback: pre-compute CRC, then plain demote_from_gpu.
            with pinned_scratch.acquire(int(size)) as scratch_ptr:
                err = rt.cudaMemcpy(
                    int(scratch_ptr),
                    int(src_va),
                    int(size),
                    rt.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )[0]
                if err != rt.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"D2H for CRC compute failed: {err}")
                crc = crc32_at_ptr(int(scratch_ptr), int(size))
                backend.demote_from_gpu(
                    engine_id,
                    layer,
                    offset,
                    gpu_ptr=src_va,
                    size=int(size),
                    crc=crc,
                )
                metrics.demote_hbm_sync.inc(engine_id=engine_id)

        metrics.storage_tier_slots.set(
            backend.n_slots(),
            engine_id=engine_id,
        )
        # Race #3: record the caller-supplied generation for this
        # block_id. Multiple per-layer RPCs for the same block all
        # carry the same generation; `set_block_generation` is
        # monotonic-max so they're idempotent.
        if generation_int:
            pool.set_block_generation(block_id, generation_int)
            with self._lock:
                generations = self._storage_slot_generations.setdefault(engine_id, {})
                generations[block_id] = max(
                    generations.get(block_id, 0), generation_int
                )
                persisted_generation_count = len(generations)
            metrics.daemon_persisted_generations.set(
                persisted_generation_count,
                engine_id=engine_id,
            )
        return True

    def capabilities(self) -> dict:
        """Daemon's runtime feature map for engine adapters.

        Engine adapters call this once at startup to decide whether
        to enable the GPU-direct demote/promote path. The shape is
        intentionally a plain dict so adding new flags later is
        backward-compatible — callers should `.get(key, default)`."""
        backend = (
            self.storage_tier.backend
            if hasattr(self.storage_tier, "backend")
            else self.storage_tier
        )
        return {
            "backend_name": backend.name,
            "supports_gpu_direct": bool(backend.supports_gpu_direct()),
            "supports_manifest_compaction": (
                hasattr(backend, "compact_manifest")
                and backend.manifest_size_bytes() >= 0
            ),
        }

    def promote_storage_to_hbm(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        size: int,
        dest_offset: Optional[int] = None,
        expected_generation: int = 0,
    ) -> bool:
        """AT_REST_STORAGE -> HBM_CACHED in one hop, skipping
        host_tier. Symmetric to `demote_hbm_to_storage`: backed by
        the storage tier's `promote_into_gpu` data plane (GDS reads
        file → engine's HBM directly), with CRC verify against the
        stored header.

        Available only when the storage backend supports a
        GPU-direct data plane. For non-GPU-direct backends, the
        existing `promote_from_storage` route (storage → host_tier;
        the engine then does H2D itself) remains the standard path.

        Returns True on successful CRC-verified read; False on any
        verification failure (slot missing, header mismatch, CRC
        mismatch). The engine should fall to cold compute on False
        just like a `restore_succeeded()=False` outcome.

        No overlap opportunity here: the CRC must be checked AFTER
        the bytes land in HBM. The win over the previous code path
        is `pinned scratch` for the post-GDS-read D2H+CPU CRC
        (~5x speedup on that segment vs the pageable D2H it used
        to do)."""
        from gms_kv_ring.common import metrics

        backend = (
            self.storage_tier.backend
            if hasattr(self.storage_tier, "backend")
            else self.storage_tier
        )
        if not backend.supports_gpu_direct():
            raise RuntimeError(
                f"promote_storage_to_hbm requires a GPU-direct storage "
                f"backend; current backend {backend.name!r} doesn't. "
                f"Use the 2-step path: promote_from_storage to "
                f"populate host_tier, then engine does H2D."
            )
        with self._lock:
            pool = self._pools.get(engine_id)
        if pool is None:
            raise ValueError(f"engine {engine_id!r} not attached")
        ld = pool.layers.get(int(layer))
        if ld is None:
            raise ValueError(f"layer {layer} not in engine {engine_id!r}'s pool")
        # `offset` keys the storage slot; `dest_offset` (default
        # same as offset) is where in the engine's pool we want
        # the bytes to land. They differ when the connector's
        # restore-on-hit path remaps a cached prefix into a freshly
        # allocated block_id.
        if dest_offset is None:
            dest_offset = offset

        # Race #3: if the caller supplied an expected_generation,
        # the storage slot's current generation must match.
        # Mismatch → slot was overwritten between the hash lookup
        # and this read; the bytes we'd return are NOT what the
        # hash points to. Signal failure; engine falls back to
        # cold compute. `expected_generation=0` means the caller
        # opted out of the check (e.g., legacy path or first-attach).
        if expected_generation:
            src_block_id = int(offset) // int(ld.stride)
            cur = pool.current_block_generation(src_block_id)
            if cur != int(expected_generation):
                from gms_kv_ring.common import metrics

                logger.info(
                    "promote_storage_to_hbm: generation mismatch rejected "
                    "eng=%r layer=%d block=%d offset=%d dest_offset=%d "
                    "expected=%d current=%d size=%d",
                    engine_id,
                    int(layer),
                    src_block_id,
                    int(offset),
                    int(dest_offset),
                    int(expected_generation),
                    cur,
                    int(size),
                )
                metrics.daemon_generation_mismatches.inc(engine_id=engine_id)
                metrics.promote_hbm_failures.inc(engine_id=engine_id)
                return False

        dest_va = ld.va + int(dest_offset)

        verified = backend.promote_into_gpu(
            engine_id,
            layer,
            offset,
            dest_gpu_ptr=dest_va,
            max_size=int(size),
        )
        if verified is None:
            metrics.promote_hbm_failures.inc(engine_id=engine_id)
            return False
        metrics.promote_hbm_ok.inc(engine_id=engine_id)
        return True

    # ---- operator-driven storage cleanup ----

    def release_engine_storage(self, engine_id: str) -> int:
        """Explicit: drop every storage-tier slot for `engine_id`.
        Use when an engine has been permanently retired."""
        from gms_kv_ring.common import metrics

        n = self.storage_tier.release_engine(engine_id)
        with self._lock:
            generation_records = len(self._storage_slot_generations.get(engine_id, {}))
            self._storage_slot_generations.pop(engine_id, None)
            pool = self._pools.get(engine_id)
            if pool is not None:
                pool.slot_generations.clear()
        metrics.daemon_storage_releases.inc(engine_id=engine_id)
        metrics.daemon_persisted_generations.set(0, engine_id=engine_id)
        if n:
            metrics.storage_tier_evictions.inc(reason="explicit", n=n)
            metrics.storage_tier_bytes.set(
                self.storage_tier.total_bytes(),
            )
        logger.info(
            "released engine storage %r: released_slots=%d "
            "cleared_generation_records=%d storage_tier_total_slots=%d "
            "storage_tier_total_bytes=%d",
            engine_id,
            n,
            generation_records,
            self.storage_tier.n_slots(),
            self.storage_tier.total_bytes(),
        )
        return n

    def prune_storage(
        self,
        max_age_seconds: Optional[float] = None,
        max_bytes: Optional[int] = None,
        max_bytes_per_engine: Optional[int] = None,
    ) -> int:
        """Operator-driven cleanup. One or more of:
          - max_age_seconds: TTL eviction (drop slots older than this)
          - max_bytes_per_engine: per-engine LRU until each engine
            is under the cap. Prevents one noisy engine from
            starving others on shared storage.
          - max_bytes: GLOBAL LRU eviction until total <= the quota.
        Returns total slots evicted across all phases.

        Order: TTL → per-engine → global. Each phase reads the state
        left by the prior, so a per-engine pass that satisfies the
        global budget makes the global phase a no-op."""
        from gms_kv_ring.common import metrics

        total = 0
        if max_age_seconds is not None:
            n = self.storage_tier.prune_older_than(float(max_age_seconds))
            if n:
                metrics.storage_tier_evictions.inc(reason="ttl", n=n)
            total += n
        if max_bytes_per_engine is not None:
            n = self.storage_tier.enforce_per_engine_byte_quota(
                int(max_bytes_per_engine),
            )
            if n:
                metrics.storage_tier_evictions.inc(
                    reason="per_engine_quota",
                    n=n,
                )
            total += n
        if max_bytes is not None:
            n = self.storage_tier.enforce_byte_quota(int(max_bytes))
            if n:
                metrics.storage_tier_evictions.inc(reason="quota", n=n)
            total += n
        metrics.storage_tier_bytes.set(self.storage_tier.total_bytes())
        return total

    def storage_stats(self) -> dict:
        """Read-only snapshot of storage-tier resident state."""
        return self.storage_tier.stats()

    def attach_evict_ring(
        self,
        engine_id: str,
        ring_path: str,
        counter_host_addr: int = 0,
        num_counters: int = 0,
        counter_path: str = "",
    ) -> None:
        """Attach the evict ring for an engine. To enable evict-ack:
        - same-process: pass `counter_host_addr` (engine's pinned VA)
          and `num_counters`. Daemon shares the engine's mmap.
        - cross-process: pass `counter_path` (filesystem path to the
          counter file the engine created) and `num_counters`. Daemon
          does its own mmap+register.

        Pass nothing for legacy fire-and-forget mode (unsafe if the
        engine reuses HBM slots immediately after the push)."""
        with self._lock:
            pool = self._pools.get(engine_id)
        if pool is None:
            raise ValueError(f"engine {engine_id!r} not attached")
        if pool.evict_consumer is not None:
            pool.evict_consumer.stop()
        c = _EvictConsumer(
            self,
            engine_id,
            ring_path,
            counter_host_addr=counter_host_addr,
            num_counters=num_counters,
            counter_path=counter_path,
        )
        c.start()
        pool.evict_consumer = c

    def attach_restore_ring(
        self,
        engine_id: str,
        ring_path: str,
        counter_path: str,
        num_counters: int,
        counter_host_addr: int = 0,
    ) -> None:
        with self._lock:
            pool = self._pools.get(engine_id)
        if pool is None:
            raise ValueError(f"engine {engine_id!r} not attached")
        if pool.restore_consumer is not None:
            pool.restore_consumer.stop()
        c = _RestoreConsumer(
            self,
            engine_id,
            ring_path,
            counter_path,
            num_counters,
            counter_host_addr=counter_host_addr,
        )
        c.start()
        pool.restore_consumer = c

    # ---- component lifecycle ----

    def start(self) -> None:
        """Start background maintenance owned by this manager."""
        if self._started:
            return
        if self._closed:
            raise RuntimeError("cannot restart a closed KV-cache manager")
        self._started = True
        if self._sweeper is not None:
            self._sweeper.start()
        self._start_scrub()
        self._start_backend_scrub()

    def close(self) -> None:
        """Stop background work and release manager-owned resources."""
        if self._closed:
            return
        self._closed = True
        if self._sweeper is not None:
            self._sweeper.stop()
        self._stop_scrub()
        self._stop_backend_scrub()
        self.close_connections()
        self._shutdown_all_pools()

    def _shutdown_all_pools(self) -> None:
        with self._lock:
            engine_ids = list(self._pools.keys())
        if not engine_ids:
            return
        # detach_engine_pool does CUDA work (cuStreamSynchronize +
        # cuStreamDestroy). Ensure this thread has the primary context
        # pushed — otherwise those calls silently fail. Skipped if no
        # pools attached, so daemons that never saw CUDA work don't
        # try to init the driver at shutdown.
        try:
            _ensure_cuda_context()
        except Exception:  # noqa: BLE001
            logger.debug("shutdown: cuda context push failed", exc_info=True)
            return
        for eid in engine_ids:
            try:
                self.detach_engine_pool(eid)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "shutdown: detach %r failed",
                    eid,
                    exc_info=True,
                )

    def close_connections(self) -> None:
        if self._connections_closed:
            return
        self._connections_closed = True
        # Tear down cross-node transport if it was constructed.
        if self.transport is not None:
            try:
                self.transport.close()
            except Exception:
                logger.exception("[Daemon] transport.close failed")
        if self.placement_publisher is not None:
            try:
                self.placement_publisher.close()
            except Exception:
                logger.exception("[Daemon] placement_publisher.close failed")
        # Close any cached source-daemon connection pools.
        with self._source_pools_lock:
            pools = list(self._source_pools.values())
            self._source_pools.clear()
        for p in pools:
            try:
                p.close()
            except Exception:  # noqa: BLE001
                logger.exception("[Daemon] source_pool.close failed")

    def _get_source_pool(self, uds_path: str) -> "_SourceClientPool":
        """Get-or-create a `_SourceClientPool` for `uds_path`. Pools
        are cached across `fetch_remote` calls — same shape as
        Dynamo's engine-direct path, which keeps a persistent NIXL
        agent + peer registry per worker. New pool is created under
        the lock; the actual socket connects happen there too.
        Callers should wrap their use in a try/except and call
        `_drop_source_pool` on failure (source restarted, etc.)."""
        with self._source_pools_lock:
            pool = self._source_pools.get(uds_path)
            if pool is not None:
                return pool
        # Open outside the lock — connect is mildly slow (ping
        # round-trip) and we don't want to hold the registry lock.
        new_pool = _SourceClientPool(uds_path, pool_size=self._source_pool_size)
        with self._source_pools_lock:
            existing = self._source_pools.get(uds_path)
            if existing is not None:
                # Someone else won the race; close ours and use theirs.
                new_pool.close()
                return existing
            self._source_pools[uds_path] = new_pool
            return new_pool

    def _drop_source_pool(self, uds_path: str) -> None:
        """Forget the cached pool for `uds_path`. Used when a batch
        RPC fails (the source may have restarted)."""
        with self._source_pools_lock:
            pool = self._source_pools.pop(uds_path, None)
        if pool is not None:
            pool.close()

    def _on_nixl_inbound(
        self,
        peer_name: str,
        reservation_id: str,
        content_hash: bytes,
    ) -> None:
        """Inbound NIXL notif handler. The sender's WRITE has landed
        in our staging_receive_buffer at the offset we returned from
        `staging_reserve`. Read the bytes, call StagingTier.commit_or_reject
        (which enforces I-9 hash-verify), then free the receive offset
        and publish a Stored event if the commit succeeded.

        Runs on the NixlTransport's drain thread. Exceptions here are
        logged but do not propagate — the drain thread keeps polling
        regardless."""
        with self._xfers_lock:
            entry = self._active_xfers.pop(reservation_id, None)
        if entry is None:
            logger.warning(
                "[Daemon] inbound notif for unknown reservation_id=%s "
                "from peer=%s (stale-reclaimed, double-notif?)",
                reservation_id,
                peer_name,
            )
            return
        offset, size, expected_hash = entry
        if expected_hash != content_hash:
            logger.warning(
                "[Daemon] inbound notif hash mismatch reservation=%s: "
                "expected=%s got=%s",
                reservation_id,
                expected_hash.hex()[:16],
                content_hash.hex()[:16],
            )
            # Drop. StagingTier reservation will stale-reclaim itself;
            # we explicitly fail it now so waiters get notified faster.
            if self.staging_tier is not None:
                self.staging_tier.fail_reservation(
                    reservation_id,
                    "notif hash mismatch",
                )
            if self.staging_receive_buffer is not None:
                self.staging_receive_buffer.free(offset, size)
            return
        try:
            payload = self.staging_receive_buffer.read(offset, size)
        except Exception:
            logger.exception(
                "[Daemon] failed to read inbound bytes (rid=%s, offset=%d, size=%d)",
                reservation_id,
                offset,
                size,
            )
            if self.staging_tier is not None:
                self.staging_tier.fail_reservation(
                    reservation_id,
                    "receive buffer read failed",
                )
            self.staging_receive_buffer.free(offset, size)
            return
        # Bridge into StagingTier — recomputes hash internally (I-9).
        result = self.staging_tier.commit_or_reject(reservation_id, payload)
        # Free the receive offset regardless of commit outcome; the
        # bytes have been copied into StagingTier's own allocator.
        self.staging_receive_buffer.free(offset, size)
        # Publish Stored event on success.
        from gms_kv_ring.daemon.staging_tier import CommitOk

        if isinstance(result, CommitOk) and self.placement_publisher is not None:
            metadata = self._staging_hit_placement_metadata(result.hit)
            try:
                self.placement_publisher.publish_stored(
                    content_hash=content_hash,
                    tier="external",
                    bytes_size=size,
                    metadata=metadata,
                )
            except TypeError:
                self.placement_publisher.publish_stored(
                    content_hash=content_hash,
                    tier="external",
                    bytes_size=size,
                )
            except Exception:
                logger.exception(
                    "[Daemon] placement_publisher.publish_stored failed",
                )

    def _gms_source_metadata(
        self, descriptor: dict, extra: Optional[dict] = None
    ) -> Optional[dict]:
        if self.transport is None:
            return None
        try:
            metadata = dict(extra or {})
            metadata.update(
                {
                    "source_nixl_agent_name": self.transport.agent_name(),
                    "source_nixl_listen_port": self.transport.listen_port(),
                    "source_nixl_agent_metadata_hex": self.transport._agent.get_agent_metadata().hex(),
                    "gms_descriptor": descriptor,
                }
            )
            return metadata
        except Exception:  # noqa: BLE001
            logger.exception("[Daemon] failed to build GMS placement metadata")
            return extra

    def _staging_hit_placement_metadata(
        self, hit, extra: Optional[dict] = None
    ) -> Optional[dict]:
        if self.transport is None:
            return extra
        try:
            self.transport.register_buffer(
                int(hit.bytes_ptr),
                int(hit.bytes_size),
                label=f"staging:{hit.content_hash.hex()[:8]}",
            )
            descriptor = {
                "remote_ptr": int(hit.bytes_ptr),
                "ptr": int(hit.bytes_ptr),
                "size": int(hit.bytes_size),
                "tier": "external",
                "ranges": [
                    {
                        "remote_ptr": int(hit.bytes_ptr),
                        "ptr": int(hit.bytes_ptr),
                        "size": int(hit.bytes_size),
                        "tier": "external",
                    }
                ],
                "generation": int(hit.generation),
                "sealed": True,
            }
            return self._gms_source_metadata(descriptor, extra)
        except Exception:  # noqa: BLE001
            logger.exception(
                "[Daemon] failed to build staging GMS placement descriptor"
            )
            return extra

    @staticmethod
    def _hbm_descriptor(ptr: int, size: int, generation: Optional[int] = None) -> dict:
        descriptor = {
            "remote_ptr": int(ptr),
            "ptr": int(ptr),
            "size": int(size),
            "tier": "hbm",
            "ranges": [
                {
                    "remote_ptr": int(ptr),
                    "ptr": int(ptr),
                    "size": int(size),
                    "tier": "hbm",
                }
            ],
            "sealed": True,
        }
        if generation is not None:
            descriptor["generation"] = int(generation)
        return descriptor

    def _host_content_descriptor(self, content_hash: bytes, ca: dict) -> Optional[dict]:
        if self.transport is None or self.host_tier is None:
            return None
        try:
            engine_id = ca["engine_id"]
            regions = []
            total_size = 0
            for layer, offset, size in ca.get("ranges") or []:
                lease = self.host_tier.pin(engine_id, layer, offset)
                if lease is None:
                    return None
                with lease as slot:
                    self.transport.register_buffer(
                        slot.host_ptr,
                        int(size),
                        label=f"host:{content_hash.hex()[:8]}:{int(layer)}",
                    )
                    regions.append(
                        {
                            "remote_ptr": int(slot.host_ptr),
                            "ptr": int(slot.host_ptr),
                            "size": int(size),
                            "tier": "host",
                            "layer": int(layer),
                            "offset": int(offset),
                        }
                    )
                    total_size += int(size)
            if not regions:
                return None
            descriptor = {
                "remote_ptr": int(regions[0]["remote_ptr"]),
                "ptr": int(regions[0]["remote_ptr"]),
                "size": int(total_size),
                "tier": "host",
                "ranges": regions,
                "sealed": True,
            }
            if ca.get("generation") is not None:
                descriptor["generation"] = int(ca["generation"])
            return descriptor
        except Exception:  # noqa: BLE001
            logger.exception("[Daemon] failed to build host GMS placement descriptor")
            return None

    def _content_address_placement_metadata(
        self,
        content_hash: bytes,
        entry: dict,
        extra: Optional[dict] = None,
    ) -> Optional[dict]:
        descriptor = self._host_content_descriptor(content_hash, entry)
        if descriptor is None:
            return extra
        return self._gms_source_metadata(descriptor, extra)

    @staticmethod
    def _read_descriptor_regions(
        descriptor: dict,
    ) -> tuple[list[tuple[int, int]], int] | None:
        """Return remote READ regions in copy order for one descriptor."""
        if not isinstance(descriptor, dict):
            return None
        sealed_raw = descriptor.get("sealed", True)
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
            return None

        raw_ranges = descriptor.get("ranges") or []
        if not raw_ranges:
            raw_ranges = [descriptor]

        regions: list[tuple[int, int]] = []
        for region in raw_ranges:
            if not isinstance(region, dict):
                return None
            ptr_raw = region.get("remote_ptr", region.get("ptr", 0))
            try:
                ptr = int(ptr_raw)
                size = int(region.get("size", 0))
            except (TypeError, ValueError):
                return None
            if ptr <= 0 or size <= 0:
                return None
            regions.append((ptr, size))
        if not regions:
            return None
        return regions, sum(size for _ptr, size in regions)

    def _read_bootstrap_into_staging(self, msg: dict) -> dict:
        """NIXL-READ router placement descriptors into local staging.

        This is the production decode-to-decode data path when the router only
        orchestrates over Dynamo's request plane. The router stamps source
        NIXL metadata and descriptors onto the request; the destination worker
        forwards them to its local daemon; this daemon reads bytes into its
        staging tier; the engine connector then restores from local staging.
        """
        if (
            self.staging_tier is None
            or self.staging_receive_buffer is None
            or self.transport is None
        ):
            return {
                "ok": False,
                "error": "read_bootstrap_into_staging: staging/transport not enabled",
            }

        source_nixl_name = str(
            msg.get("source_nixl_name") or msg.get("source_nixl_agent_name") or ""
        )
        metadata_hex = str(
            msg.get("source_agent_metadata_hex")
            or msg.get("source_nixl_agent_metadata_hex")
            or ""
        )
        if not source_nixl_name:
            return {"ok": False, "error": "source NIXL agent name is empty"}
        if not metadata_hex:
            return {"ok": False, "error": "source NIXL metadata is empty"}
        try:
            source_metadata = bytes.fromhex(metadata_hex)
        except ValueError:
            return {"ok": False, "error": "source NIXL metadata is not hex"}

        hashes_hex = msg.get("hashes") or []
        descriptors = msg.get("descriptors") or []
        if len(hashes_hex) != len(descriptors):
            return {
                "ok": False,
                "error": (
                    "read_bootstrap_into_staging: hashes/descriptors "
                    f"length mismatch ({len(hashes_hex)} != {len(descriptors)})"
                ),
            }
        timeout_s = float(msg.get("timeout_s", 30.0))
        batch_size = int(msg.get("batch_size", len(hashes_hex) or 1))
        if batch_size <= 0:
            batch_size = len(hashes_hex) or 1

        valid_items: list[dict] = []
        skipped = 0
        for h_hex, descriptor in zip(hashes_hex, descriptors):
            try:
                content_hash = bytes.fromhex(str(h_hex))
            except ValueError:
                skipped += 1
                continue
            parsed = self._read_descriptor_regions(descriptor)
            if parsed is None:
                skipped += 1
                continue
            regions, total_size = parsed
            metadata = None
            if isinstance(descriptor, dict):
                metadata_raw = descriptor.get("metadata")
                if isinstance(metadata_raw, dict):
                    metadata = metadata_raw
            valid_items.append(
                {
                    "content_hash": content_hash,
                    "regions": regions,
                    "total_size": int(total_size),
                    "metadata": metadata,
                }
            )

        accepted = 0
        already_ready = 0
        coalesced = 0
        failed = 0
        bytes_read = 0
        if not valid_items:
            return {
                "ok": True,
                "accepted": 0,
                "already_ready": 0,
                "coalesced": 0,
                "failed": 0,
                "skipped": skipped,
                "bytes_read": 0,
            }

        from gms_kv_ring.daemon.staging_tier import (
            AlreadyReady,
            CommitOk,
            Rejected,
            Reservation,
            Waiter,
        )

        try:
            self.transport.add_peer_from_metadata(
                source_nixl_name,
                source_metadata,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "error": f"source metadata registration failed: {exc}",
            }

        rresults = self.staging_tier.reserve_or_wait_many(
            [item["content_hash"] for item in valid_items],
            source_nixl_name,
        )
        reservation_items: list[tuple[dict, Reservation]] = []
        for item, rresult in zip(valid_items, rresults):
            if isinstance(rresult, AlreadyReady):
                already_ready += 1
            elif isinstance(rresult, Waiter):
                coalesced += 1
            elif isinstance(rresult, Rejected):
                failed += 1
            else:
                assert isinstance(rresult, Reservation)
                reservation_items.append((item, rresult))

        offsets = self.staging_receive_buffer.alloc_many(
            [item["total_size"] for item, _reservation in reservation_items]
        )
        blocks: list[dict] = []
        for (item, reservation), offset in zip(reservation_items, offsets):
            if offset is None:
                self.staging_tier.fail_reservation(
                    reservation.reservation_id,
                    "receive buffer full",
                )
                failed += 1
                continue
            blocks.append(
                {
                    "reservation_id": reservation.reservation_id,
                    "content_hash": item["content_hash"],
                    "regions": item["regions"],
                    "total_size": item["total_size"],
                    "metadata": item.get("metadata"),
                    "offset": int(offset),
                }
            )

        for start in range(0, len(blocks), batch_size):
            chunk = blocks[start : start + batch_size]
            xfer_items: list[tuple[int, int, int]] = []
            for block in chunk:
                local_ptr = self.staging_receive_buffer.ptr_at(block["offset"])
                cursor = 0
                for remote_ptr, size in block["regions"]:
                    xfer_items.append((local_ptr + cursor, size, remote_ptr))
                    cursor += size
                if cursor != block["total_size"]:
                    raise AssertionError("descriptor region size accounting bug")
            try:
                self.transport.read_batch(
                    source_nixl_name,
                    xfer_items,
                    timeout_s=timeout_s,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[Daemon] read_bootstrap_into_staging: READ from %s "
                    "failed for %d block(s): %s",
                    source_nixl_name,
                    len(chunk),
                    exc,
                )
                for block in chunk:
                    self.staging_receive_buffer.free(
                        block["offset"],
                        block["total_size"],
                    )
                    self.staging_tier.fail_reservation(
                        block["reservation_id"],
                        "source READ failed",
                    )
                failed += len(chunk)
                continue

            for block in chunk:
                payload = None
                try:
                    payload = self.staging_receive_buffer.read(
                        block["offset"],
                        block["total_size"],
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "[Daemon] read_bootstrap_into_staging: receive "
                        "buffer read failed",
                    )
                    self.staging_tier.fail_reservation(
                        block["reservation_id"],
                        "receive buffer read failed",
                    )
                    failed += 1
                finally:
                    self.staging_receive_buffer.free(
                        block["offset"],
                        block["total_size"],
                    )
                if payload is None:
                    continue
                result = self.staging_tier.commit_or_reject(
                    block["reservation_id"],
                    payload,
                    verify_content_hash=False,
                )
                if isinstance(result, CommitOk):
                    accepted += 1
                    bytes_read += block["total_size"]
                    if self.placement_publisher is not None:
                        try:
                            self.placement_publisher.publish_stored(
                                content_hash=block["content_hash"],
                                tier="external",
                                bytes_size=block["total_size"],
                                metadata=block.get("metadata"),
                            )
                        except TypeError:
                            self.placement_publisher.publish_stored(
                                content_hash=block["content_hash"],
                                tier="external",
                                bytes_size=block["total_size"],
                            )
                        except Exception:
                            logger.exception(
                                "[Daemon] read_bootstrap_into_staging: "
                                "publish_stored failed",
                            )
                else:
                    failed += 1

        return {
            "ok": True,
            "accepted": accepted,
            "already_ready": already_ready,
            "coalesced": coalesced,
            "failed": failed,
            "skipped": skipped,
            "bytes_read": bytes_read,
        }
