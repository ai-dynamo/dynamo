# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from packaging.version import Version
from vllm import __version__ as _vllm_version
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

# Async-load support requires the WAITING_FOR_EMBEDDINGS state machine and
# the get_finished_loads/request_async_load hooks. These land on top of the
# upstream EC base API in this branch.
MINIMUM_VLLM_VERSION = "0.17.0"

logger = init_logger("vllm.dynamo_ec_connector")


@dataclass
class MultimodalEmbeddingCacheConnectorMetadata(ECConnectorMetadata):
    """Commands from scheduler to worker for CPU embedding cache management."""

    loads: list[str] = field(default_factory=list)
    saves: list[str] = field(default_factory=list)
    evicts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # ECConnectorMetadata base class declares ``evict_orphan: set[str]``;
        # ensure it's initialized even when the dataclass-generated __init__
        # bypasses the base ``__init__``.
        if not hasattr(self, "evict_orphan"):
            self.evict_orphan = set()


@dataclass
class _CpuEntry:
    """Worker-side CPU-resident embedding with async-save / async-load lifetime tracking.

    save_done resolves when the DtoH copy on _save_stream has finished writing
    cpu_tensor; pending_loads carries one event per in-flight HtoD reading from
    cpu_tensor on _load_stream. Both are queried (never synchronized) by the
    reaper so eviction never blocks the hot path. is_pinned distinguishes the
    async/pinned path from the sync/pageable cap-fallback path; load and reap
    branch on it.
    """

    cpu_tensor: torch.Tensor
    save_done: torch.cuda.Event
    pending_loads: list[torch.cuda.Event] = field(default_factory=list)
    nbytes: int = 0
    is_pinned: bool = True


class DynamoMultimodalEmbeddingCacheConnector(ECConnectorBase):
    """EC connector with scheduler-authoritative CPU embedding cache, async
    DtoH save, and an async-load fast path.

    Scheduler maintains a logical LRU (OrderedDict) and issues load/save/evict
    commands to the worker via ECConnectorMetadata. Worker holds a plain
    ``dict[str, _CpuEntry]`` and obeys commands without independent caching
    decisions.

    Worker-side DtoH (save) runs on a dedicated CUDA stream with pinned host
    buffers; HtoD (load) runs on a second dedicated stream, gated on the save
    event. Loads do NOT block compute — the H2D overlaps with the prior
    step's compute and the scheduler parks the request in
    ``WAITING_FOR_EMBEDDINGS`` until ``get_finished_loads`` reports the
    mm_hash done.

    Evict is non-blocking — entries move to a retired list and are reaped
    lazily once their save_done and any pending_loads have queried True.
    """

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole) -> None:
        if Version(_vllm_version) < Version(MINIMUM_VLLM_VERSION):
            logger.warning(
                "DynamoMultimodalEmbeddingCacheConnector requires vLLM >= %s, "
                "but found %s. Some features may not work correctly.",
                MINIMUM_VLLM_VERSION,
                _vllm_version,
            )
        super().__init__(vllm_config=vllm_config, role=role)

        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is None:
            raise ValueError(
                "ec_transfer_config must be set for DynamoMultimodalEmbeddingCacheConnector"
            )

        extra_config = transfer_config.ec_connector_extra_config or {}
        if "multimodal_embedding_cache_capacity_gb" not in extra_config:
            raise ValueError(
                "multimodal_embedding_cache_capacity_gb must be set in "
                "ec_connector_extra_config for DynamoMultimodalEmbeddingCacheConnector"
            )
        capacity_gb: float = extra_config["multimodal_embedding_cache_capacity_gb"]

        # --- Scheduler-side: logical LRU for CPU embedding cache ---
        # Mirrors EncoderCacheManager but for the CPU tier, tracking bytes.
        hidden_size = vllm_config.model_config.get_hidden_size()
        dtype_bytes = torch.tensor(
            [], dtype=vllm_config.model_config.dtype
        ).element_size()
        self._bytes_per_embed = hidden_size * dtype_bytes
        self._capacity_bytes = int(capacity_gb * 1024**3)

        self._cache_order: OrderedDict[str, int] = OrderedDict()  # hash → size_bytes
        self._num_used_bytes: int = 0

        self._loads_this_step: set[str] = set()
        self._saves_this_step: set[str] = set()
        self._evicts_this_step: set[str] = set()

        # --- Scheduler-side cumulative counters (for periodic logging) ---
        self._total_hits: int = 0
        self._total_misses: int = 0
        self._total_evictions: int = 0
        self._total_loads: int = 0
        self._total_saves: int = 0
        self._log_step: int = 0

        # --- Worker-side: async DtoH state ---
        self._pin_memory: bool = bool(extra_config.get("pin_memory", True))
        self._pinned_overhead_pct: float = float(
            extra_config.get("pinned_overhead_pct", 25.0)
        )
        self._pinned_cap_bytes: int = int(
            self._capacity_bytes * (1.0 + self._pinned_overhead_pct / 100.0)
        )
        self._log_every_n_steps: int = int(
            extra_config.get("ec_log_every_n_steps", 100)
        )

        self._cpu_store: dict[str, _CpuEntry] = {}
        self._retired: list[_CpuEntry] = []
        self._save_stream: torch.cuda.Stream | None = None
        self._device: torch.device | None = None
        self._pinned_bytes_active: int = 0
        self._pinned_bytes_retired: int = 0
        self._already_done_event: torch.cuda.Event | None = None
        self._step_counter: int = 0

        # --- Worker-side: async-load (HtoD) state ---
        # H2D copies run on _load_stream and don't block compute. Each load
        # records an event into _inflight_loads; get_finished_loads polls
        # the events, moves completed tensors into encoder_cache, and
        # reports their hashes to the scheduler so it can re-admit the
        # request. _just_resurrected tracks same-hash orphan tensors that
        # are already encoder_cache-resident at start_load_caches time.
        self._inflight_loads: dict[str, tuple[torch.Tensor, torch.cuda.Event]] = {}
        self._just_resurrected: set[str] = set()
        self._load_stream: torch.cuda.Stream | None = None

        logger.info(
            "DynamoMultimodalEmbeddingCacheConnector initialized: "
            "role=%s, capacity_gb=%.2f, capacity_bytes=%d, bytes_per_embed=%d, "
            "pin_memory=%s, pinned_cap_bytes=%d",
            role.name,
            capacity_gb,
            self._capacity_bytes,
            self._bytes_per_embed,
            self._pin_memory,
            self._pinned_cap_bytes,
        )

    # ==============================
    # Scheduler-side methods
    #
    # vLLM scheduler call sequence per multimodal feature:
    #
    #   1. encoder_cache_manager.check_and_update_cache(request, i)
    #      → if True (GPU hit): skip entirely.
    #
    #   2. has_cache_item(identifier)
    #      → returns (True, True) on a CPU hit, opting into async-load:
    #         scheduler parks the request in WAITING_FOR_EMBEDDINGS and
    #         calls request_async_load(request, i) instead of the sync
    #         update_state_after_alloc.
    #      → (False, _) routes to the normal compute / save path.
    #
    #   3. update_state_after_alloc(request, i) is called for sync-allocated
    #      paths (current step's encoder compute or any sync external load).
    #      For async loads, request_async_load is called instead — both
    #      ultimately funnel into _loads_this_step / _saves_this_step.
    # ==============================

    def has_cache_item(self, identifier: str) -> "tuple[bool, bool]":
        """CPU-cache lookup, MRU-promoting on hit.

        Returns ``(hit, load_async)``:
          - miss: ``(False, False)``.
          - hit:  ``(True, True)`` — opt the request into async-load.
        """
        if identifier in self._cache_order:
            self._cache_order.move_to_end(identifier)
            self._total_hits += 1
            return True, True
        self._total_misses += 1
        return False, False

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """Sync-path hook (encoder compute, or sync external load).

        Called by the scheduler after ``encoder_cache_manager.allocate(request,
        i)`` for sync external_load_encoder_input AND for the encoder-compute
        path (where the connector records a save command). Idempotent under
        repeated ``(request, mm_hash)`` calls; the connector deduplicates by
        hash.
        """
        mm_hash: str = request.mm_features[index].identifier
        num_embeds: int = request.get_num_encoder_embeds(index)
        size_bytes: int = num_embeds * self._bytes_per_embed

        if mm_hash in self._cache_order:
            # Hash already in CPU store; sync external-load path requested.
            self._cache_order.move_to_end(mm_hash)
            self._loads_this_step.add(mm_hash)
            return

        # Miss → schedule encoder compute, mark for save.
        if size_bytes > self._capacity_bytes:
            return

        self._saves_this_step.add(mm_hash)

        while (
            self._num_used_bytes + size_bytes > self._capacity_bytes
            and self._cache_order
        ):
            evicted_hash, evicted_bytes = self._cache_order.popitem(last=False)
            self._num_used_bytes -= evicted_bytes
            self._evicts_this_step.add(evicted_hash)
            self._total_evictions += 1

        self._cache_order[mm_hash] = size_bytes
        self._num_used_bytes += size_bytes

    def request_async_load(self, request: "Request", index: int) -> None:
        """Async-path hook: the scheduler is parking the request in
        ``WAITING_FOR_EMBEDDINGS`` for this mm_hash and has reserved its
        encoder-cache slot via the inflight-async budget. The connector
        records a load command for the worker; the worker dispatches the
        H2D in the next ``start_load_caches`` call.

        Idempotent under repeated calls for the same mm_hash (set semantics).
        """
        mm_hash: str = request.mm_features[index].identifier
        # The hash MUST already be in the CPU store — that's the precondition
        # has_cache_item returned True for. MRU-promote.
        if mm_hash in self._cache_order:
            self._cache_order.move_to_end(mm_hash)
        else:
            # Defensive: scheduler called us for a hash we don't have. Log
            # and skip; the scheduler's WAITING_FOR_EMBEDDINGS will time out
            # waiting for finished_loading and the user will see the warning.
            logger.warning(
                "request_async_load: %s not in CPU cache; load will not fire",
                mm_hash,
            )
            return
        self._loads_this_step.add(mm_hash)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        """Flush load/save/evict commands into metadata for the worker.

        ``evict_orphan`` is appended by the scheduler after this returns
        (see ``Scheduler._pending_orphan_evicts`` flush in ``_make_step_output``).
        """
        meta = MultimodalEmbeddingCacheConnectorMetadata(
            loads=list(self._loads_this_step),
            saves=list(self._saves_this_step),
            evicts=list(self._evicts_this_step),
        )

        self._total_loads += len(self._loads_this_step)
        self._total_saves += len(self._saves_this_step)

        self._loads_this_step.clear()
        self._saves_this_step.clear()
        self._evicts_this_step.clear()

        self._log_step += 1
        if self._log_step % self._log_every_n_steps == 0:
            total_lookups = self._total_hits + self._total_misses
            hit_rate = (
                100.0 * self._total_hits / total_lookups if total_lookups else 0.0
            )
            used_gb = self._num_used_bytes / 1024**3
            cap_gb = self._capacity_bytes / 1024**3
            logger.info(
                "ec_connector stats: hits=%d misses=%d hit_rate=%.1f%% "
                "loads=%d saves=%d evicts=%d entries=%d used=%.2f/%.2f GB",
                self._total_hits,
                self._total_misses,
                hit_rate,
                self._total_loads,
                self._total_saves,
                self._total_evictions,
                len(self._cache_order),
                used_gb,
                cap_gb,
            )

        return meta

    # ==============================
    # Worker-side methods
    #
    # Called by the model runner each step with the metadata produced by
    # build_connector_meta. start_load_caches dispatches H2Ds on
    # _load_stream; get_finished_loads polls events and moves completed
    # tensors into encoder_cache before reporting their mm_hashes.
    # ==============================

    def _ensure_streams(self, device: torch.device) -> None:
        """Lazily create save/load streams and lock the connector to a device.

        The sentinel `_already_done_event` is recorded on the save stream and
        synchronized once so subsequent .query() calls always return True. We
        use it as save_done for sync-DtoH (pageable fallback) entries so the
        load path branches uniformly without an `is_pinned` check on every
        load.
        """
        if self._save_stream is None:
            self._device = device
            self._save_stream = torch.cuda.Stream(device=device)
            self._load_stream = torch.cuda.Stream(device=device)
            sentinel = torch.cuda.Event()
            sentinel.record(self._save_stream)
            self._save_stream.synchronize()
            self._already_done_event = sentinel
        else:
            assert device == self._device, (
                f"EC connector bound to device {self._device} "
                f"but received tensor on {device}"
            )

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        """Dispatch async H2D loads, evict requested entries.

        For each mm_hash in metadata.loads:
          - already in flight  → piggyback (no-op).
          - resurrected orphan → already in encoder_cache, queue immediate
            completion in get_finished_loads.
          - fresh             → enqueue H2D on _load_stream after waiting
            on the entry's save_done event; record load_done into both
            ``_inflight_loads`` (for completion polling) and
            ``entry.pending_loads`` (so the reaper waits for the read to
            complete before freeing the cpu_tensor).

        H2D does NOT call ``compute.wait_event(load_done)`` — that would
        kill the overlap. The scheduler gates re-admission on
        ``event.query() == True`` via get_finished_loads; once that returns
        true, the data is GPU-globally visible per CUDA programming guide
        §3.2.5.

        ``evicts`` moves entries from the CPU store to the retired list;
        ``evict_orphan`` removes entries from the worker's encoder_cache.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MultimodalEmbeddingCacheConnectorMetadata)

        compute = torch.cuda.current_stream()
        self._ensure_streams(compute.device)
        load_stream = self._load_stream
        assert load_stream is not None

        for mm_hash in metadata.loads:
            if mm_hash in self._inflight_loads:
                continue  # piggyback dedup
            if mm_hash in encoder_cache:
                # Same-hash orphan resurrection: a previously-completed
                # async load whose evict_orphan was cancelled by the
                # scheduler (because a new request hit the same hash
                # before evict shipped). Tensor is still GPU-resident in
                # encoder_cache. Report immediate completion.
                self._just_resurrected.add(mm_hash)
                continue
            entry = self._cpu_store.get(mm_hash)
            if entry is None:
                logger.warning(
                    "start_load_caches: %s not in cpu_store, skipping", mm_hash
                )
                continue

            # Make load_stream wait for the prior DtoH on save_stream. For
            # pageable entries the sentinel event is already-resolved, so
            # this is a no-op. Pinned entries gate the read so we never
            # H2D from a buffer the save_stream hasn't finished writing.
            load_stream.wait_event(entry.save_done)
            with torch.cuda.stream(load_stream):
                gpu_tensor = entry.cpu_tensor.to(
                    device=compute.device, non_blocking=entry.is_pinned
                )
            load_done = torch.cuda.Event()
            load_done.record(load_stream)
            entry.pending_loads.append(load_done)
            self._inflight_loads[mm_hash] = (gpu_tensor, load_done)

        for mm_hash in metadata.evicts:
            entry = self._cpu_store.pop(mm_hash, None)
            if entry is not None:
                if entry.is_pinned:
                    self._pinned_bytes_active -= entry.nbytes
                    self._pinned_bytes_retired += entry.nbytes
                self._retired.append(entry)

        # Drop orphan GPU tensors whose abandoned load completed in a
        # prior step. The scheduler queues these via _pending_orphan_evicts
        # in _update_from_ec_xfer_finished.
        for mm_hash in metadata.evict_orphan:
            encoder_cache.pop(mm_hash, None)

        self._prune_active_load_events()
        self._reap_retired()

        self._step_counter += 1
        if self._step_counter % self._log_every_n_steps == 0:
            logger.info(
                "EC pinned bytes: active=%d retired=%d cap=%d "
                "(entries=%d retired_count=%d inflight=%d)",
                self._pinned_bytes_active,
                self._pinned_bytes_retired,
                self._pinned_cap_bytes,
                len(self._cpu_store),
                len(self._retired),
                len(self._inflight_loads),
            )

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        """Copy a newly computed embedding from GPU encoder_cache to CPU store.

        Pinned path: queue an async DtoH on _save_stream after waiting on
        compute, record save_done, hand the entry to _cpu_store immediately
        (load path will gate on save_done). The pinned-bytes accounting tracks
        active vs retired separately so the reaper can release retired bytes
        once both save and any pending loads have drained.

        Sync fallback (pin_memory=False or over-cap): a pageable .cpu() copy
        with the always-resolved sentinel as save_done, accounted as
        is_pinned=False so the budget is not double-counted.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MultimodalEmbeddingCacheConnectorMetadata)

        if mm_hash not in metadata.saves:
            return
        if mm_hash in self._cpu_store:
            return
        if mm_hash not in encoder_cache:
            logger.warning(
                "save_caches: hash %s in metadata.saves but not in encoder_cache",
                mm_hash,
            )
            return

        src = encoder_cache[mm_hash]
        if not src.is_contiguous():
            logger.debug(
                "save_caches: non-contiguous source for hash %s, materializing",
                mm_hash,
            )
            src = src.contiguous()

        self._ensure_streams(src.device)
        nbytes = src.numel() * src.element_size()

        over_cap = (
            self._pinned_bytes_active + self._pinned_bytes_retired + nbytes
            > self._pinned_cap_bytes
        )
        use_pinned = self._pin_memory and not over_cap

        if use_pinned:
            cpu_buf = torch.empty(
                src.shape, dtype=src.dtype, device="cpu", pin_memory=True
            )
            compute = torch.cuda.current_stream(src.device)
            assert self._save_stream is not None
            self._save_stream.wait_stream(compute)
            with torch.cuda.stream(self._save_stream):
                cpu_buf.copy_(src, non_blocking=True)
                # Symmetric record_stream on the source: protects the GPU
                # source memory in case vLLM pops encoder_cache[h] before the
                # async DtoH finishes.
                src.record_stream(self._save_stream)
            save_done = torch.cuda.Event()
            save_done.record(self._save_stream)
            self._cpu_store[mm_hash] = _CpuEntry(
                cpu_tensor=cpu_buf,
                save_done=save_done,
                nbytes=nbytes,
                is_pinned=True,
            )
            self._pinned_bytes_active += nbytes
        else:
            if over_cap:
                logger.warning(
                    "save_caches: pinned budget exceeded "
                    "(active=%d retired=%d new=%d cap=%d); "
                    "falling back to sync DtoH for hash %s",
                    self._pinned_bytes_active,
                    self._pinned_bytes_retired,
                    nbytes,
                    self._pinned_cap_bytes,
                    mm_hash,
                )
            assert self._already_done_event is not None
            self._cpu_store[mm_hash] = _CpuEntry(
                cpu_tensor=src.cpu(),
                save_done=self._already_done_event,
                nbytes=nbytes,
                is_pinned=False,
            )

        self._reap_retired()

    def get_finished_loads(
        self,
        encoder_cache: dict[str, torch.Tensor] | None = None,
    ) -> set[str] | None:
        """Per-mm_hash async-load completion signal.

        Returns the set of mm_hashes whose H2D has retired since the last
        call PLUS any hashes resurrected from orphan tensors during this
        step's start_load_caches. Each completed tensor is moved into
        ``encoder_cache[mm_hash]`` BEFORE its hash is added to the returned
        set, so the next consumer step's _gather_mm_embeddings sees the
        GPU-resident tensor.
        """
        finished: set[str] = set(self._just_resurrected)
        self._just_resurrected.clear()

        if self._inflight_loads and encoder_cache is not None:
            ready = [h for h, (_, ev) in self._inflight_loads.items() if ev.query()]
            for h in ready:
                tensor, _ = self._inflight_loads.pop(h)
                # Per design: move the tensor into encoder_cache BEFORE
                # reporting completion. If the request that prompted this
                # load has aborted, the scheduler will have transitioned
                # state to ABANDONED; on receipt of finished_loading={h}
                # it will pop the entry and queue evict_orphan.
                # We optimistically place the tensor here; ABANDONED clean
                # up happens on the next step's start_load_caches via
                # evict_orphan, which will pop encoder_cache[h] there.
                encoder_cache[h] = tensor
                finished.add(h)

        return finished if finished else None

    def _prune_active_load_events(self) -> None:
        """Drop resolved load events from active entries to bound the list size."""
        for entry in self._cpu_store.values():
            if entry.pending_loads:
                entry.pending_loads = [e for e in entry.pending_loads if not e.query()]

    def _reap_retired(self) -> None:
        """Release retired entries whose save and all pending loads have drained.

        Never blocks — uses event.query() only. Safe to call from save_caches
        and start_load_caches.
        """
        if not self._retired:
            return
        still_pending: list[_CpuEntry] = []
        for entry in self._retired:
            if entry.save_done.query() and all(e.query() for e in entry.pending_loads):
                if entry.is_pinned:
                    self._pinned_bytes_retired -= entry.nbytes
                continue
            still_pending.append(entry)
        self._retired = still_pending
