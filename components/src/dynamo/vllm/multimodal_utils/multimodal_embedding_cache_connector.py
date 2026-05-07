# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import logging
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
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

MINIMUM_VLLM_VERSION = "0.17.0"

logger = logging.getLogger(__name__)


@dataclass
class MultimodalEmbeddingCacheConnectorMetadata(ECConnectorMetadata):
    """Commands from scheduler to worker for CPU embedding cache management."""

    loads: list[str] = field(default_factory=list)
    saves: list[str] = field(default_factory=list)
    evicts: list[str] = field(default_factory=list)


@dataclass
class _CpuEntry:
    """Worker-side CPU-resident embedding view into the pinned arena.

    cpu_view is a typed view (dtype/shape) over a slice of the connector's
    single pinned uint8 arena. save_done resolves when the DtoH copy on
    _save_stream has finished writing the slice; pending_loads carries one
    event per in-flight HtoD reading from the slice on the compute stream.
    Both are queried (never synchronized) by the non-blocking reaper so
    eviction never blocks the hot path.
    """

    cpu_view: torch.Tensor
    offset: int
    nbytes: int
    save_done: torch.cuda.Event
    pending_loads: list[torch.cuda.Event] = field(default_factory=list)


@dataclass
class _PendingFree:
    """Slice waiting to return to the allocator. Reaped non-blocking when
    save_done and every pending load have queried True; reaped blocking
    on the allocation-failure fallback path."""

    offset: int
    nbytes: int
    save_done: torch.cuda.Event
    pending_loads: list[torch.cuda.Event]


class DynamoMultimodalEmbeddingCacheConnector(ECConnectorBase):
    """EC connector with scheduler-authoritative CPU embedding cache.

    The scheduler maintains a logical LRU cache (OrderedDict) and issues
    load/save/evict commands to the worker via ECConnectorMetadata. The
    worker holds a plain dict[str, _CpuEntry] over a single pre-allocated
    pinned host arena and obeys commands without independent caching
    decisions.

    Worker-side DtoH (save) runs on a dedicated CUDA stream, copying into
    a slice of the pinned arena. HtoD (load) runs on the compute stream
    and waits on the save event before reading. Evict is non-blocking —
    slices move to a deferred-free queue and return to the allocator
    once their save_done and any pending_loads have resolved.

    The arena's `_bytes_per_embed` allocator granularity matches the
    scheduler's accounting unit, so `_used_bytes == _num_used_bytes`
    always holds (no alignment overhead).
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
        # Round capacity down to a chunk multiple so the arena can be exactly
        # filled by chunk-sized saves without partial-trailing-region waste.
        self._capacity_bytes -= self._capacity_bytes % self._bytes_per_embed

        self._cache_order: OrderedDict[str, int] = OrderedDict()  # hash → size_bytes
        self._num_used_bytes: int = 0

        self._loads_this_step: set[str] = set()
        self._saves_this_step: set[str] = set()
        self._evicts_this_step: set[str] = set()

        # --- Worker-side: pinned arena, allocator, async-save state ---
        self._cpu_store: dict[str, _CpuEntry] = {}
        # Arena is allocated lazily on the first save_caches call so the
        # SCHEDULER-role connector instance never pins host memory.
        self._pinned_arena: torch.Tensor | None = None
        self._arena_bytes: int = 0
        # Free regions, sorted ascending by offset; both fields are
        # _bytes_per_embed multiples.
        self._free_regions: list[tuple[int, int]] = []
        self._used_bytes: int = 0
        self._pending_free: list[_PendingFree] = []
        self._save_stream: torch.cuda.Stream | None = None
        self._device: torch.device | None = None

        logger.info(
            "DynamoMultimodalEmbeddingCacheConnector initialized: "
            "role=%s, capacity_gb=%.6f, capacity_bytes=%d, bytes_per_embed=%d",
            role.name,
            capacity_gb,
            self._capacity_bytes,
            self._bytes_per_embed,
        )

    # ==============================
    # Scheduler-side methods
    #
    # vLLM scheduler call sequence per multimodal feature:
    #
    #   1. encoder_cache_manager.check_and_update_cache(request, i)
    #      → if True (GPU hit): skip entirely, neither method below is called.
    #
    #   2. has_cache_item(identifier)
    #      → if True (CPU hit):  item goes to external_load_encoder_input
    #      → if False (CPU miss): item goes to encoder_inputs_to_schedule
    #
    #   3. update_state_after_alloc(request, i) is called for both paths.
    #      The two paths are mutually exclusive per hash within a step:
    #      - external_load_encoder_input → mm_hash IN _cache_order  → load path
    #      - encoder_inputs_to_schedule  → mm_hash NOT in _cache_order → save path
    # ==============================

    def has_cache_item(self, identifier: str) -> bool:
        """Check if an embedding is in the CPU cache, promoting it to MRU on hit.

        Called by the scheduler only after the GPU encoder_cache_manager reports
        a miss. A True return tells the scheduler to skip encoder compute and
        load the embedding from the CPU store instead.
        """
        if identifier in self._cache_order:
            self._cache_order.move_to_end(identifier)
            return True
        return False

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """Record a load or save command for a multimodal feature.

        Called by the scheduler after has_cache_item has already determined
        the path. The _cache_order check here mirrors that decision:

        CPU hit  (mm_hash in _cache_order):  mark for CPU→GPU load.
        CPU miss (mm_hash not in _cache_order): evict LRU entries if needed,
            then mark for GPU→CPU save so the worker persists the newly
            computed embedding. Silently skips items larger than total capacity.
        """
        mm_hash: str = request.mm_features[index].identifier
        num_embeds: int = request.get_num_encoder_embeds(index)
        size_bytes: int = num_embeds * self._bytes_per_embed

        if mm_hash in self._cache_order:
            self._cache_order.move_to_end(mm_hash)
            self._loads_this_step.add(mm_hash)
            return

        if size_bytes > self._capacity_bytes:
            return

        self._saves_this_step.add(mm_hash)

        if self._num_used_bytes + size_bytes > self._capacity_bytes:
            while (
                self._num_used_bytes + size_bytes > self._capacity_bytes
                and self._cache_order
            ):
                evicted_hash, evicted_bytes = self._cache_order.popitem(last=False)
                self._num_used_bytes -= evicted_bytes
                self._evicts_this_step.add(evicted_hash)

        self._cache_order[mm_hash] = size_bytes
        self._num_used_bytes += size_bytes

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        """Flush accumulated load/save/evict commands into metadata for the worker."""
        meta = MultimodalEmbeddingCacheConnectorMetadata(
            loads=list(self._loads_this_step),
            saves=list(self._saves_this_step),
            evicts=list(self._evicts_this_step),
        )

        self._loads_this_step.clear()
        self._saves_this_step.clear()
        self._evicts_this_step.clear()

        return meta

    # ==============================
    # Worker-side methods
    #
    # Called by the model runner each step with the metadata produced by
    # build_connector_meta. The worker has no caching logic of its own;
    # it simply obeys the scheduler's load/save/evict commands.
    # ==============================

    def _ensure_streams(self, device: torch.device) -> None:
        if self._save_stream is None:
            self._device = device
            self._save_stream = torch.cuda.Stream(device=device)
        else:
            assert device == self._device, (
                f"EC connector bound to device {self._device} "
                f"but received tensor on {device}"
            )

    def _ensure_arena(self) -> None:
        """Lazy: pin _capacity_bytes of host memory on the first save call.
        Pays cudaHostRegister once at first use rather than at __init__,
        so SCHEDULER-role instances never pin."""
        if self._pinned_arena is not None:
            return
        self._arena_bytes = self._capacity_bytes
        self._pinned_arena = torch.empty(
            self._arena_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        self._free_regions = [(0, self._arena_bytes)]

    def _validate_chunk(self, mm_hash: str, src: torch.Tensor, nbytes: int) -> None:
        """nbytes must be a _bytes_per_embed multiple. Raises with full
        diagnostic context so scheduler/worker drift is debuggable."""
        if nbytes % self._bytes_per_embed != 0:
            raise RuntimeError(
                f"EC connector: nbytes {nbytes} is not a multiple of "
                f"_bytes_per_embed {self._bytes_per_embed} for hash={mm_hash} "
                f"shape={tuple(src.shape)} dtype={src.dtype}"
            )

    def _alloc(self, nbytes: int) -> int | None:
        """First-fit on _free_regions. Returns offset or None."""
        for i, (off, length) in enumerate(self._free_regions):
            if length >= nbytes:
                if length == nbytes:
                    self._free_regions.pop(i)
                else:
                    self._free_regions[i] = (off + nbytes, length - nbytes)
                self._used_bytes += nbytes
                return off
        return None

    def _free(self, offset: int, nbytes: int) -> None:
        """Insert (offset, nbytes) into _free_regions and coalesce neighbors."""
        self._used_bytes -= nbytes

        lo, hi = 0, len(self._free_regions)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._free_regions[mid][0] < offset:
                lo = mid + 1
            else:
                hi = mid
        self._free_regions.insert(lo, (offset, nbytes))

        # Merge with right neighbor first (otherwise indices shift).
        if lo + 1 < len(self._free_regions):
            a_off, a_len = self._free_regions[lo]
            b_off, b_len = self._free_regions[lo + 1]
            if a_off + a_len == b_off:
                self._free_regions[lo] = (a_off, a_len + b_len)
                self._free_regions.pop(lo + 1)
        # Merge with left neighbor.
        if lo > 0:
            a_off, a_len = self._free_regions[lo - 1]
            b_off, b_len = self._free_regions[lo]
            if a_off + a_len == b_off:
                self._free_regions[lo - 1] = (a_off, a_len + b_len)
                self._free_regions.pop(lo)

    def _reap_pending_free(self) -> None:
        """Non-blocking: free slices whose save_done and pending_loads have
        all resolved. Safe to call from save and load hot paths."""
        if not self._pending_free:
            return
        still: list[_PendingFree] = []
        for pf in self._pending_free:
            if pf.save_done.query() and all(e.query() for e in pf.pending_loads):
                self._free(pf.offset, pf.nbytes)
            else:
                still.append(pf)
        self._pending_free = still

    def _alloc_with_fallback(self, nbytes: int) -> int | None:
        """Two-step fallback when _alloc fails:
        1. Synchronize all _pending_free events and free their slices.
        2. Compact live entries and retry.
        """
        if self._pending_free:
            for pf in self._pending_free:
                pf.save_done.synchronize()
                for e in pf.pending_loads:
                    e.synchronize()
                self._free(pf.offset, pf.nbytes)
            self._pending_free.clear()
            offset = self._alloc(nbytes)
            if offset is not None:
                return offset

        self._compact()
        return self._alloc(nbytes)

    def _compact(self) -> None:
        """Defragment by repacking live entries to [0, _used_bytes).

        Sorted by ascending offset so left-shifts never alias unread
        source bytes; ctypes.memmove handles overlapping ranges safely
        (Tensor.copy_ on overlapping slices of the same storage is not
        memmove-defined). Synchronizes save_stream and every entry's
        pending_loads first so no in-flight DtoH/HtoD touches arena
        bytes during the memmove.
        """
        assert self._save_stream is not None and self._pinned_arena is not None
        self._save_stream.synchronize()
        for entry in self._cpu_store.values():
            for e in entry.pending_loads:
                e.synchronize()
            entry.pending_loads.clear()

        arena_storage_ptr = self._pinned_arena.data_ptr()
        cursor = 0
        for entry in sorted(self._cpu_store.values(), key=lambda x: x.offset):
            if entry.offset != cursor:
                ctypes.memmove(
                    arena_storage_ptr + cursor,
                    arena_storage_ptr + entry.offset,
                    entry.nbytes,
                )
                entry.offset = cursor
                entry.cpu_view = (
                    self._pinned_arena.narrow(0, cursor, entry.nbytes)
                    .view(entry.cpu_view.dtype)
                    .view(entry.cpu_view.shape)
                )
            cursor += entry.nbytes
        if cursor < self._arena_bytes:
            self._free_regions = [(cursor, self._arena_bytes - cursor)]
        else:
            self._free_regions = []

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        """Copy cached embeddings from the pinned arena to GPU encoder_cache,
        then retire the slices the scheduler marked for removal."""
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MultimodalEmbeddingCacheConnectorMetadata)

        compute = torch.cuda.current_stream()
        self._ensure_streams(compute.device)

        for mm_hash in metadata.loads:
            if mm_hash in encoder_cache:
                continue
            entry = self._cpu_store.get(mm_hash)
            if entry is None:
                continue

            # Make compute wait for the prior DtoH on save_stream.
            compute.wait_event(entry.save_done)

            # HtoD on compute. The slice is pinned (it views the arena)
            # so non_blocking is genuine.
            gpu_tensor = entry.cpu_view.to(device=compute.device, non_blocking=True)

            # Symmetric record_stream on the destination: tells the caching
            # allocator the new tensor's storage is consumed by compute, so
            # if vLLM pops encoder_cache[h] before compute is done, the GPU
            # buffer is not reused early.
            gpu_tensor.record_stream(compute)

            load_done = torch.cuda.Event()
            load_done.record(compute)
            # Prune resolved events to bound list growth on hot entries.
            entry.pending_loads = [e for e in entry.pending_loads if not e.query()]
            entry.pending_loads.append(load_done)

            encoder_cache[mm_hash] = gpu_tensor

        if metadata.evicts:
            for mm_hash in metadata.evicts:
                entry = self._cpu_store.pop(mm_hash, None)
                if entry is not None:
                    self._pending_free.append(
                        _PendingFree(
                            offset=entry.offset,
                            nbytes=entry.nbytes,
                            save_done=entry.save_done,
                            pending_loads=entry.pending_loads,
                        )
                    )

        self._reap_pending_free()

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        """Copy a newly computed embedding from GPU encoder_cache into a
        slice of the pinned arena.

        Allocates a slice via the byte-level free-list allocator, queues
        an async DtoH on _save_stream after waiting on compute, records
        save_done, and hands the entry to _cpu_store. The load path gates
        on save_done before reading.

        If allocation fails (arena fragmentation), falls back to a
        synchronous reap of _pending_free, then to compaction. If both
        fail, raises RuntimeError — a save under the scheduler cap that
        cannot be placed is a programming bug, not a normal cache miss.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MultimodalEmbeddingCacheConnectorMetadata)

        if mm_hash not in metadata.saves:
            return
        if mm_hash in self._cpu_store:
            # Per ECConnector contract, a hash is saved at most once
            # per cache lifetime; treat repeats as no-op.
            return
        if mm_hash not in encoder_cache:
            return

        src = encoder_cache[mm_hash]
        if not src.is_contiguous():
            src = src.contiguous()

        self._ensure_streams(src.device)
        self._ensure_arena()

        nbytes = src.numel() * src.element_size()
        self._validate_chunk(mm_hash, src, nbytes)

        self._reap_pending_free()
        offset = self._alloc(nbytes)
        if offset is None:
            offset = self._alloc_with_fallback(nbytes)
        if offset is None:
            raise RuntimeError(
                f"EC arena exhausted for hash={mm_hash}: "
                f"nbytes={nbytes} used={self._used_bytes} "
                f"arena={self._arena_bytes} "
                f"free_regions={len(self._free_regions)} "
                f"pending_free={len(self._pending_free)}"
            )

        assert self._pinned_arena is not None
        cpu_view = (
            self._pinned_arena.narrow(0, offset, nbytes).view(src.dtype).view(src.shape)
        )

        compute = torch.cuda.current_stream(src.device)
        assert self._save_stream is not None
        self._save_stream.wait_stream(compute)
        with torch.cuda.stream(self._save_stream):
            cpu_view.copy_(src, non_blocking=True)
            # Symmetric record_stream on the source: protects the GPU
            # source memory in case vLLM pops encoder_cache[h] before
            # the async DtoH finishes.
            src.record_stream(self._save_stream)
        save_done = torch.cuda.Event()
        save_done.record(self._save_stream)

        self._cpu_store[mm_hash] = _CpuEntry(
            cpu_view=cpu_view,
            offset=offset,
            nbytes=nbytes,
            save_done=save_done,
        )
