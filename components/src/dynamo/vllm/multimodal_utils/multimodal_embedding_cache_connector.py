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

MINIMUM_VLLM_VERSION = "0.17.0"

logger = init_logger("vllm.dynamo_ec_connector")


@dataclass
class MultimodalEmbeddingCacheConnectorMetadata(ECConnectorMetadata):
    """Commands from scheduler to worker for CPU embedding cache management."""

    loads: list[str] = field(default_factory=list)
    saves: list[str] = field(default_factory=list)
    evicts: list[str] = field(default_factory=list)


@dataclass
class _CpuEntry:
    """Worker-side CPU-resident embedding with async-save / async-load lifetime tracking.

    save_done resolves when the DtoH copy on _save_stream has finished writing
    cpu_tensor; pending_loads carries one event per in-flight HtoD reading from
    cpu_tensor on the compute stream. Both are queried (never synchronized) by
    the reaper so eviction never blocks the hot path. is_pinned distinguishes
    the async/pinned path from the sync/pageable cap-fallback path; load and
    reap branch on it.
    """

    cpu_tensor: torch.Tensor
    save_done: torch.cuda.Event
    pending_loads: list[torch.cuda.Event] = field(default_factory=list)
    nbytes: int = 0
    is_pinned: bool = True


class DynamoMultimodalEmbeddingCacheConnector(ECConnectorBase):
    """EC connector with scheduler-authoritative CPU embedding cache.

    The scheduler maintains a logical LRU cache (OrderedDict) and issues
    load/save/evict commands to the worker via ECConnectorMetadata. The
    worker holds a plain dict[str, _CpuEntry] on CPU and obeys commands
    without independent caching decisions.

    Worker-side DtoH (save) runs on a dedicated CUDA stream with pinned
    host buffers; HtoD (load) runs on the compute stream and waits on the
    save event before reading. Evict is non-blocking — entries are
    retired and reaped lazily once their save_done and any pending_loads
    have resolved.
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

        # --- Worker-side: async DtoH/HtoD state ---
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
            self._total_hits += 1
            return True
        self._total_misses += 1
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

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        """Flush accumulated load/save/evict commands into metadata for the worker."""
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
    # build_connector_meta. The worker has no caching logic of its own;
    # it simply obeys the scheduler's load/save/evict commands.
    # ==============================

    def _ensure_streams(self, device: torch.device) -> None:
        """Lazily create the save stream and lock the connector to a device.

        The sentinel `_already_done_event` is recorded on the save stream and
        synchronized once so subsequent .query() calls always return True. We
        use it as save_done for sync-DtoH (pageable fallback) entries so the
        load path branches uniformly without an `is_pinned` check on every
        load.
        """
        if self._save_stream is None:
            self._device = device
            self._save_stream = torch.cuda.Stream(device=device)
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
        """Copy cached embeddings from CPU store to GPU encoder_cache, and evict
        entries the scheduler marked for removal.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MultimodalEmbeddingCacheConnectorMetadata)

        compute = torch.cuda.current_stream()
        self._ensure_streams(compute.device)

        for mm_hash in metadata.loads:
            if mm_hash in encoder_cache:
                continue
            entry = self._cpu_store.get(mm_hash)
            if entry is None:
                logger.warning(
                    "start_load_caches: hash %s not in cpu_store, skipping", mm_hash
                )
                continue

            # Make compute wait for the prior DtoH on save_stream. For pageable
            # entries the sentinel event is already-resolved, so this is a no-op.
            compute.wait_event(entry.save_done)

            # HtoD on compute. Genuinely async for pinned; PyTorch silently
            # falls back to sync for pageable, which is correct for that path.
            gpu_tensor = entry.cpu_tensor.to(
                device=compute.device, non_blocking=entry.is_pinned
            )

            # Symmetric record_stream on the destination: tells the caching
            # allocator the new tensor's storage is consumed by compute, so
            # if vLLM pops encoder_cache[h] before compute is done, the GPU
            # buffer is not reused early.
            gpu_tensor.record_stream(compute)

            load_done = torch.cuda.Event()
            load_done.record(compute)
            entry.pending_loads.append(load_done)

            encoder_cache[mm_hash] = gpu_tensor

        for mm_hash in metadata.evicts:
            entry = self._cpu_store.pop(mm_hash, None)
            if entry is not None:
                if entry.is_pinned:
                    self._pinned_bytes_active -= entry.nbytes
                    self._pinned_bytes_retired += entry.nbytes
                self._retired.append(entry)

        self._prune_active_load_events()
        self._reap_retired()

        self._step_counter += 1
        if self._step_counter % self._log_every_n_steps == 0:
            logger.info(
                "EC pinned bytes: active=%d retired=%d cap=%d "
                "(entries=%d retired_count=%d)",
                self._pinned_bytes_active,
                self._pinned_bytes_retired,
                self._pinned_cap_bytes,
                len(self._cpu_store),
                len(self._retired),
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
