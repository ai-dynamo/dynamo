# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import atexit
import ctypes
import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from multiprocessing import shared_memory
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

# Architectures whose LLM forward is verified to issue a same-stream TP
# all-reduce after `_execute_mm_encoder` returns. The Plan-B fence relies
# on that collective acting as the cross-rank visibility point: every
# worker reaches the all-reduce on its compute_stream, and the all-reduce
# can't complete on any rank until every rank's compute_stream is past
# the prior `wait_event(latest_save_event)`. Add new families only after
# tracing their forward path (or after a CI canary verifies it).
KNOWN_TP_NCCL_FORWARD_MODELS: frozenset[str] = frozenset(
    {
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
    }
)

# Use vllm.* name so logs surface inside the EngineCore subprocess where
# only the `vllm` logger has a handler attached (DEFAULT_LOGGING_CONFIG with
# propagate=False). A bare logging.getLogger(__name__) lands in the
# dynamo.* namespace and is silently dropped.
logger = init_logger("vllm.dynamo_ec_connector")


# ---------------------------------------------------------------------------
# Wire-format commands. The scheduler embeds offset/nbytes/num_embeds in each
# command so the worker can address its slot in the shared arena without
# maintaining its own hash → slot map.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _LoadCmd:
    mm_hash: str
    offset: int
    nbytes: int
    num_embeds: int


@dataclass(slots=True)
class _SaveCmd:
    mm_hash: str
    offset: int
    nbytes: int
    num_embeds: int
    writer_rank: int


@dataclass(slots=True)
class _EvictCmd:
    mm_hash: str
    offset: int
    nbytes: int


@dataclass
class MultimodalEmbeddingCacheConnectorMetadata(ECConnectorMetadata):
    """Commands from scheduler to worker for one step."""

    loads: list[_LoadCmd] = field(default_factory=list)
    saves: list[_SaveCmd] = field(default_factory=list)
    evicts: list[_EvictCmd] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scheduler-side state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _SchedulerEntry:
    offset: int
    nbytes: int
    num_embeds: int
    writer_rank: int
    save_step: int
    retire_step: int | None = None
    state: str = "PENDING"  # PENDING | READY | RETIRING


def _writer_rank_for(mm_hash: str, world_size: int) -> int:
    """Stable rank assignment so the same hash always lands in the same
    partition across requests, regardless of insertion order."""
    if world_size <= 1:
        return 0
    digest = hashlib.blake2b(mm_hash.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % world_size


class _PartitionAllocator:
    """First-fit byte-granularity allocator over one rank's partition,
    with lazy reclaim of RETIRING entries whose retire_step < scheduling_step.
    """

    def __init__(self, base_offset: int, capacity: int) -> None:
        self._free_regions: list[tuple[int, int]] = (
            [(base_offset, capacity)] if capacity > 0 else []
        )
        # (mm_hash, offset, nbytes, retire_step) — eligible for reclaim once
        # retire_step < current scheduling_step.
        self._retiring: list[tuple[str, int, int, int]] = []

    def add_retiring(
        self, mm_hash: str, offset: int, nbytes: int, retire_step: int
    ) -> None:
        self._retiring.append((mm_hash, offset, nbytes, retire_step))

    def alloc(self, nbytes: int, scheduling_step: int) -> tuple[int | None, list[str]]:
        """Try first-fit; if it fails, lazy-reclaim retiring entries with
        retire_step < scheduling_step and retry. Returns (offset_or_None,
        reclaimed_hashes)."""
        offset = self._first_fit(nbytes)
        if offset is not None:
            return offset, []
        reclaimed = self._reclaim(scheduling_step)
        if not reclaimed:
            return None, []
        return self._first_fit(nbytes), reclaimed

    def _first_fit(self, nbytes: int) -> int | None:
        for i, (off, length) in enumerate(self._free_regions):
            if length >= nbytes:
                if length == nbytes:
                    self._free_regions.pop(i)
                else:
                    self._free_regions[i] = (off + nbytes, length - nbytes)
                return off
        return None

    def _reclaim(self, scheduling_step: int) -> list[str]:
        reclaimed: list[str] = []
        survivors: list[tuple[str, int, int, int]] = []
        for h, off, nb, rs in self._retiring:
            if rs < scheduling_step:
                self._add_to_free_list(off, nb)
                reclaimed.append(h)
            else:
                survivors.append((h, off, nb, rs))
        self._retiring = survivors
        return reclaimed

    def _add_to_free_list(self, offset: int, nbytes: int) -> None:
        lo, hi = 0, len(self._free_regions)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._free_regions[mid][0] < offset:
                lo = mid + 1
            else:
                hi = mid
        self._free_regions.insert(lo, (offset, nbytes))
        # right merge
        if lo + 1 < len(self._free_regions):
            a_off, a_len = self._free_regions[lo]
            b_off, b_len = self._free_regions[lo + 1]
            if a_off + a_len == b_off:
                self._free_regions[lo] = (a_off, a_len + b_len)
                self._free_regions.pop(lo + 1)
        # left merge
        if lo > 0:
            a_off, a_len = self._free_regions[lo - 1]
            b_off, b_len = self._free_regions[lo]
            if a_off + a_len == b_off:
                self._free_regions[lo - 1] = (a_off, a_len + b_len)
                self._free_regions.pop(lo)


class _GpuPoolAllocator:
    """Pre-allocated GPU buffer with a first-fit free list over byte offsets.

    `alloc(nbytes)` returns a contiguous byte offset; the connector then
    narrows the backing buffer into a per-group slab. `free(offset, nbytes)`
    returns the slab to the free list and coalesces with adjacent runs.

    Lifetime is owned externally by the connector — see `_PoolGroup`. The
    pool slab is freed only when ALL `mm_hash` views from the original
    coalesced load group have dropped out of vLLM's `encoder_cache` dict
    (per-group lifetime, not per-hash). The buffer is allocated once at
    `_setup_worker` time, sized at 2x vLLM's encoder-cache token budget
    so chunked-prefill workloads have headroom.
    """

    def __init__(self, pool_bytes: int, device: torch.device) -> None:
        self._buffer: torch.Tensor = torch.empty(
            pool_bytes, dtype=torch.uint8, device=device
        )
        self._capacity: int = pool_bytes
        self._free_regions: list[tuple[int, int]] = (
            [(0, pool_bytes)] if pool_bytes > 0 else []
        )

    @property
    def buffer(self) -> torch.Tensor:
        return self._buffer

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    def alloc(self, nbytes: int) -> int | None:
        for i, (off, length) in enumerate(self._free_regions):
            if length >= nbytes:
                if length == nbytes:
                    self._free_regions.pop(i)
                else:
                    self._free_regions[i] = (off + nbytes, length - nbytes)
                return off
        return None

    def free(self, offset: int, nbytes: int) -> None:
        lo, hi = 0, len(self._free_regions)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._free_regions[mid][0] < offset:
                lo = mid + 1
            else:
                hi = mid
        self._free_regions.insert(lo, (offset, nbytes))
        if lo + 1 < len(self._free_regions):
            a_off, a_len = self._free_regions[lo]
            b_off, b_len = self._free_regions[lo + 1]
            if a_off + a_len == b_off:
                self._free_regions[lo] = (a_off, a_len + b_len)
                self._free_regions.pop(lo + 1)
        if lo > 0:
            a_off, a_len = self._free_regions[lo - 1]
            b_off, b_len = self._free_regions[lo]
            if a_off + a_len == b_off:
                self._free_regions[lo - 1] = (a_off, a_len + b_len)
                self._free_regions.pop(lo)

    def stats(self) -> tuple[int, int, int, int]:
        """Returns (alive_bytes, free_bytes, free_runs, largest_free_bytes).

        `largest_free` distinguishes fragmentation-OOM (large `free_bytes`,
        small `largest_free`) from true capacity exhaustion in the warn log.
        """
        free_bytes = 0
        largest = 0
        for _, length in self._free_regions:
            free_bytes += length
            if length > largest:
                largest = length
        return self._capacity - free_bytes, free_bytes, len(self._free_regions), largest


@dataclass(slots=True)
class _PoolGroup:
    """One pre-allocated pool slab backing a coalesced H2D group.

    `nbytes` is the H2D-coalesced span (PR #9304's `group_total`); `hashes`
    holds every `_LoadCmd.mm_hash` whose view shares this slab. The slab
    is returned to `_GpuPoolAllocator` only when ALL hashes have left
    `encoder_cache` — per-hash freeing would invalidate sibling views.
    """

    offset: int
    nbytes: int
    hashes: set[str]


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class DynamoMultimodalEmbeddingCacheConnector(ECConnectorBase):
    """Embedding-cache connector with one shared pinned arena per engine.

    Per-rank partitioning replaces the prior per-rank private arena: each TP
    rank owns one slice of the shared shm region (writer_rank assigned via
    blake2b(mm_hash) % world_size). All ranks DMA-read every slot, so
    capacity_bytes of host RAM is locked once per engine instead of once per
    rank.

    Two synchronization modes:

    Plan B (NCCL fence — default for allowlisted models on TP>1): the worker
    records each save's CUDA event on save_stream, tracks the latest one for
    the step, and at the encoder context exit (`clear_connector_metadata`)
    issues `compute_stream.wait_event(latest_save_event)`. The next op on
    compute_stream is the LLM forward's first TP all-reduce — that NCCL
    collective is the cross-rank fence: it cannot complete on any rank
    until every rank's compute_stream has cleared `wait_event`, which
    means every rank's prior-step D2H has retired. CPU never blocks.
    Trade-off: D2H from the *last* save in the step is gated against
    LLM-forward overlap (~100–200 µs), but stays overlapped with the
    encoder forward.

    Plan A (CPU sync — fallback): when the model isn't in the allowlist
    or TP=1 or `mm_encoder_only=True`, the worker drains save_stream +
    compute_stream and runs a TP barrier at the start of each multimodal-
    active step (`_on_step_begin`). CPU-blocking but model-agnostic.

    Scheduler-side authority:
        - LRU OrderedDict of _SchedulerEntry, keyed by mm_hash.
        - Per-rank _PartitionAllocator handles offsets and lazy reclaim.
        - State machine (PENDING → READY → RETIRING) gates load issuance.
          PENDING is promoted lazily inside `has_cache_item` once
          save_step < scheduling_step (i.e. at least one full step elapsed
          since the save was emitted, so `_on_step_begin` of *this* step
          has already drained every rank's save_stream).

    Worker-side has no per-hash state: every command in metadata carries
    (offset, nbytes, num_embeds), and the arena is a shared memoryview
    cudaHostRegister'd into each rank's CUDA context.
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
                "ec_transfer_config must be set for "
                "DynamoMultimodalEmbeddingCacheConnector"
            )

        extra_config = transfer_config.ec_connector_extra_config or {}
        if "multimodal_embedding_cache_capacity_gb" not in extra_config:
            raise ValueError(
                "multimodal_embedding_cache_capacity_gb must be set in "
                "ec_connector_extra_config"
            )
        capacity_gb: float = extra_config["multimodal_embedding_cache_capacity_gb"]

        # Encoder cache stores tensors of shape (num_embeds, feature_dim).
        # feature_dim is the *encoder output* width, which differs from the
        # text hidden_size for models that concatenate deepstack features
        # post-encoder (e.g. Qwen3-VL: out_hidden_size × (1 + len(deepstack))).
        self._model_dtype: torch.dtype = vllm_config.model_config.dtype
        dtype_bytes = torch.tensor([], dtype=self._model_dtype).element_size()
        self._feature_dim: int = self._compute_feature_dim(vllm_config.model_config)
        self._bytes_per_embed: int = self._feature_dim * dtype_bytes
        self._capacity_bytes: int = int(capacity_gb * 1024**3)
        # Round capacity down so the arena holds an integer number of
        # bytes-per-embed; partitions further round to be a multiple of it.
        self._capacity_bytes -= self._capacity_bytes % self._bytes_per_embed

        # Worker-side GPU pool sizing — 2x vLLM's encoder-cache token
        # budget. Mirrors `max_num_batched_tokens` (the underlying knob
        # vLLM derives `encoder_cache_size` from) rather than introducing
        # a new CLI flag. The 2x headroom absorbs chunked-prefill cases
        # where an encoder_cache entry survives across steps while new
        # entries are loaded. `_pool_disabled_reason` makes a 0-byte pool
        # operator-visible at the init banner.
        self._pool_size_bytes: int
        self._pool_disabled_reason: str
        sched_cfg = getattr(vllm_config, "scheduler_config", None)
        if sched_cfg is None:
            self._pool_size_bytes = 0
            self._pool_disabled_reason = "scheduler_config_missing"
        elif self._bytes_per_embed <= 0:
            self._pool_size_bytes = 0
            self._pool_disabled_reason = "bytes_per_embed_zero"
        else:
            max_batched = getattr(sched_cfg, "max_num_batched_tokens", 0)
            try:
                max_batched_int = int(max_batched)
            except (TypeError, ValueError):
                max_batched_int = -1
            if max_batched_int < 0:
                self._pool_size_bytes = 0
                self._pool_disabled_reason = "max_num_batched_tokens_non_numeric"
            elif max_batched_int == 0:
                self._pool_size_bytes = 0
                self._pool_disabled_reason = "max_num_batched_tokens_zero"
            else:
                self._pool_size_bytes = 2 * max_batched_int * self._bytes_per_embed
                self._pool_disabled_reason = ""

        # Discover TP shape early; used for partition layout on both
        # scheduler and worker. The scheduler also needs world_size to
        # compute writer_rank consistently with the worker. We read it from
        # parallel_config because the scheduler runs in P1 EngineCore where
        # the TP group isn't initialized.
        self._tp_world_size: int = max(
            int(getattr(vllm_config.parallel_config, "tensor_parallel_size", 1)), 1
        )
        self._partition_bytes: int = self._capacity_bytes // self._tp_world_size
        self._partition_bytes -= self._partition_bytes % self._bytes_per_embed

        # Scheduler-side state (only used when role == SCHEDULER).
        self._entries: OrderedDict[str, _SchedulerEntry] = OrderedDict()
        self._partitions: list[_PartitionAllocator] = [
            _PartitionAllocator(
                base_offset=r * self._partition_bytes,
                capacity=self._partition_bytes,
            )
            for r in range(self._tp_world_size)
        ]
        self._scheduling_step: int = 0
        self._loads_this_step: list[_LoadCmd] = []
        self._saves_this_step: list[_SaveCmd] = []
        self._evicts_this_step: list[_EvictCmd] = []

        # Worker-side state (filled in by _setup_worker on first call).
        self._engine_id: str = transfer_config.engine_id or ""
        self._shared_arena_nonce: str = extra_config.get("shared_arena_nonce", "")
        self._tp_rank: int = 0
        self._shm: shared_memory.SharedMemory | None = None
        self._shm_addr: int = 0
        self._arena_uint8: torch.Tensor | None = None
        self._save_stream: torch.cuda.Stream | None = None
        self._device: torch.device | None = None
        self._worker_initialized: bool = False
        self._released: bool = False

        # Plan B: one CUDA event per step instead of one per save. The
        # flag is set in `save_caches` whenever a D2H is enqueued on
        # `_save_stream`; `clear_connector_metadata` then records a
        # single event at the stream's tail and waits on it from
        # compute_stream. Equivalent to per-save events because every
        # D2H this step is enqueued serially on the same stream, but
        # collapses N cudaEventCreate+Record down to 1.
        self._save_pending_this_step: bool = False

        # O(1) per-step `mm_hash -> _SaveCmd` lookup, populated in
        # `bind_connector_metadata`. Replaces the per-call linear scan
        # in `save_caches` (model_runner invokes save_caches once per
        # produced encoder output → was O(N²) per step). Filtered to
        # `writer_rank == _tp_rank` once `_setup_worker` has run; the
        # `writer_rank != _tp_rank` early-return in `save_caches`
        # remains the correctness boundary.
        self._save_cmd_by_hash: dict[str, _SaveCmd] = {}

        # Pre-allocated worker-side GPU pool. `_gpu_pool` is populated
        # lazily in `_setup_worker` (needs `_device`). `_pool_groups`
        # tracks per-coalesced-H2D-group slab metadata (one entry per
        # PR #9304-coalesced load group); `_hash_to_group` is the
        # mm_hash → owning slab index for save-side / debug lookups.
        # `_pool_oom_log_step` is the rate-limit cursor; init to
        # -(_LOG_EVERY_N_STEPS + 1) so the first OOM always logs.
        # `_pool_oom_count` is bumped before the rate-limit gate, so the
        # periodic stats line reflects total OOMs even when the warn
        # log is suppressed.
        self._gpu_pool: _GpuPoolAllocator | None = None
        self._pool_groups: list[_PoolGroup] = []
        self._hash_to_group: dict[str, _PoolGroup] = {}
        # Init to a value that guarantees the first OOM log fires
        # regardless of `_mm_active_steps` starting at 0.
        self._pool_oom_log_step: int = -(self._LOG_EVERY_N_STEPS + 1)
        self._pool_oom_count: int = 0

        # Plan B activation gate: the model's LLM forward must be known
        # to issue a same-stream TP NCCL all-reduce after _execute_mm_encoder.
        # Disabled for TP=1 (no cross-rank concern), encoder-only configs
        # (no LLM forward to fence on), and unrecognized architectures.
        arch = (
            (vllm_config.model_config.architectures or [None])[0]
            if vllm_config.model_config is not None
            else None
        )
        mm_cfg = getattr(vllm_config.model_config, "multimodal_config", None)
        mm_encoder_only = bool(getattr(mm_cfg, "mm_encoder_only", False))
        env_disabled = os.environ.get("DYN_EC_NCCL_FENCE", "1").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        )
        self._nccl_fence_active: bool = (
            role == ECConnectorRole.WORKER
            and self._tp_world_size > 1
            and not mm_encoder_only
            and not env_disabled
            and arch in KNOWN_TP_NCCL_FORWARD_MODELS
        )
        # Reasoning trace for the init banner so an operator can tell at a
        # glance which sync path took effect.
        self._fence_reason: str
        if role != ECConnectorRole.WORKER:
            self._fence_reason = "scheduler-role"
        elif self._tp_world_size <= 1:
            self._fence_reason = "tp_world_size==1"
        elif mm_encoder_only:
            self._fence_reason = "mm_encoder_only"
        elif env_disabled:
            self._fence_reason = "DYN_EC_NCCL_FENCE_disabled"
        elif arch not in KNOWN_TP_NCCL_FORWARD_MODELS:
            self._fence_reason = f"arch_not_allowlisted:{arch}"
        else:
            self._fence_reason = "nccl_fence"

        # Per-rank counters for the periodic distribution log. Only the
        # writer rank increments _saves_issued; every rank increments
        # _loads_issued for the loads that targeted its compute stream.
        self._saves_issued: int = 0
        self._loads_issued: int = 0
        # Step counter — bumped once per `start_load_caches` call where
        # metadata was non-empty. Logging fires every _LOG_EVERY_N_STEPS.
        self._mm_active_steps: int = 0

        if role == ECConnectorRole.WORKER:
            # Defer the actual SHM/cudaHostRegister setup to first metadata
            # cycle: at __init__ time we may not know which CUDA device this
            # worker owns yet, and we want all ranks to enter the setup at
            # the same step (driven by metadata being non-empty on every
            # rank). This is correct because metadata is built by a single
            # scheduler and broadcast identically to every TP rank.
            atexit.register(self._release_shm)

        logger.info(
            "DynamoMultimodalEmbeddingCacheConnector initialized: "
            "role=%s tp_world_size=%d capacity_gb=%.6f capacity_bytes=%d "
            "partition_bytes=%d bytes_per_embed=%d sync_mode=%s (%s) "
            "gpu_pool_bytes=%d pool_disabled_reason=%s",
            role.name,
            self._tp_world_size,
            capacity_gb,
            self._capacity_bytes,
            self._partition_bytes,
            self._bytes_per_embed,
            "PlanB" if self._nccl_fence_active else "PlanA",
            self._fence_reason,
            self._pool_size_bytes,
            self._pool_disabled_reason or "none",
        )

    @staticmethod
    def _compute_feature_dim(model_config) -> int:
        """Encoder-output width per visual token.

        For Qwen3-VL: vision_config.out_hidden_size × (1 + len(deepstack)).
        For models without deepstack: out_hidden_size or text hidden_size.
        Falls back conservatively when neither is exposed."""
        fallback = int(model_config.get_hidden_size())
        hf = getattr(model_config, "hf_config", None)
        if hf is None:
            return fallback
        vc = getattr(hf, "vision_config", None)
        if vc is None:
            return fallback
        base = getattr(vc, "out_hidden_size", None) or fallback
        deepstack = getattr(vc, "deepstack_visual_indexes", None) or []
        try:
            multiplier = 1 + len(deepstack)
        except TypeError:
            multiplier = 1
        return int(base) * multiplier

    # ==================================================================
    # Scheduler-side
    # ==================================================================

    def has_cache_item(self, identifier: str) -> bool:
        entry = self._entries.get(identifier)
        if entry is None:
            return False
        # Lazy promotion: an entry is considered safely loaded once at least
        # one full scheduling step has elapsed since it was saved. By that
        # point the prior step's worker fence has run — under Plan B this
        # is `clear_connector_metadata`'s wait_event chained into the LLM
        # forward's TP NCCL all-reduce; under Plan A it is `_on_step_begin`'s
        # CPU drain + `dist.barrier`. Either way the D2H is globally visible
        # across ranks before any H2D in this step issues.
        if entry.state == "PENDING" and entry.save_step < self._scheduling_step:
            entry.state = "READY"
        if entry.state == "READY":
            self._entries.move_to_end(identifier)
            return True
        return False

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        mm_hash: str = request.mm_features[index].identifier
        num_embeds: int = request.get_num_encoder_embeds(index)
        nbytes: int = num_embeds * self._bytes_per_embed

        existing = self._entries.get(mm_hash)
        if existing is not None and existing.state == "READY":
            self._entries.move_to_end(mm_hash)
            self._loads_this_step.append(
                _LoadCmd(
                    mm_hash=mm_hash,
                    offset=existing.offset,
                    nbytes=existing.nbytes,
                    num_embeds=existing.num_embeds,
                )
            )
            return

        # CPU miss: must have come through has_cache_item returning False,
        # so existing is None or state is PENDING/RETIRING. Either way we
        # do not double-insert.
        if existing is not None:
            return
        if nbytes <= 0 or nbytes > self._partition_bytes:
            # Doesn't fit in any partition; cache is best-effort, skip.
            return

        writer_rank = _writer_rank_for(mm_hash, self._tp_world_size)
        partition = self._partitions[writer_rank]

        offset, reclaimed = partition.alloc(nbytes, self._scheduling_step)
        for h in reclaimed:
            self._entries.pop(h, None)

        if offset is None:
            # Mark the partition's LRU READY entry as RETIRING so it can be
            # reclaimed *next* step. Same-step reclaim is intentionally not
            # eligible (rs < scheduling_step gate prevents reuse-while-reads-
            # may-still-be-in-flight). This save just gets skipped, which is
            # fine — cache is best-effort.
            self._evict_lru_in_partition(writer_rank)
            return

        entry = _SchedulerEntry(
            offset=offset,
            nbytes=nbytes,
            num_embeds=num_embeds,
            writer_rank=writer_rank,
            save_step=self._scheduling_step,
            state="PENDING",
        )
        self._entries[mm_hash] = entry
        self._saves_this_step.append(
            _SaveCmd(
                mm_hash=mm_hash,
                offset=offset,
                nbytes=nbytes,
                num_embeds=num_embeds,
                writer_rank=writer_rank,
            )
        )

    def _evict_lru_in_partition(self, writer_rank: int) -> bool:
        """Mark the LRU READY entry from this partition as RETIRING."""
        for h, entry in self._entries.items():
            if entry.writer_rank != writer_rank or entry.state != "READY":
                continue
            entry.state = "RETIRING"
            entry.retire_step = self._scheduling_step
            self._partitions[writer_rank].add_retiring(
                h, entry.offset, entry.nbytes, self._scheduling_step
            )
            self._evicts_this_step.append(
                _EvictCmd(mm_hash=h, offset=entry.offset, nbytes=entry.nbytes)
            )
            return True
        return False

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        meta = MultimodalEmbeddingCacheConnectorMetadata(
            loads=self._loads_this_step,
            saves=self._saves_this_step,
            evicts=self._evicts_this_step,
        )
        self._loads_this_step = []
        self._saves_this_step = []
        self._evicts_this_step = []
        # _scheduling_step ticks at the end of metadata build so the
        # *next* call to `has_cache_item` sees save_step < scheduling_step
        # for entries written this step (lazy promotion gate).
        self._scheduling_step += 1
        return meta

    # ==================================================================
    # Worker-side
    # ==================================================================

    def bind_connector_metadata(self, connector_metadata: ECConnectorMetadata) -> None:
        """Build the per-step `mm_hash -> _SaveCmd` lookup so `save_caches`
        runs in O(1). The model_runner invokes `save_caches` once per
        produced encoder output, so the previous linear scan over
        `metadata.saves` was O(N²) per step.

        On the very first multimodal-active step `_setup_worker` has
        not yet run, so `_tp_rank` is still its default 0 — build an
        unfiltered dict in that case (slightly more dict entries on
        non-writer ranks for one step). The `writer_rank != _tp_rank`
        early-return in `save_caches` remains the correctness gate
        regardless.
        """
        super().bind_connector_metadata(connector_metadata)
        if not isinstance(
            connector_metadata, MultimodalEmbeddingCacheConnectorMetadata
        ):
            self._save_cmd_by_hash = {}
            return
        if self._worker_initialized:
            self._save_cmd_by_hash = {
                cmd.mm_hash: cmd
                for cmd in connector_metadata.saves
                if cmd.writer_rank == self._tp_rank
            }
        else:
            self._save_cmd_by_hash = {
                cmd.mm_hash: cmd for cmd in connector_metadata.saves
            }

    def _setup_worker(self) -> None:
        """Allocate shm, cudaHostRegister, build the arena view, and create
        save_stream. Invoked lazily on the first non-empty metadata cycle so
        every rank reaches it together (metadata is identical across ranks).

        Lazy-setup correctness depends on `ec_role == ec_both`: every rank
        runs both `start_load_caches` (gated by `is_consumer`) and
        `save_caches` (gated by `is_producer`), and `MultiprocExecutor`
        broadcasts metadata identically to all ranks, so all ranks call
        `_setup_worker` on the same step. If this connector is ever wired
        as `ec_producer`-only, the lazy setup must move to an unconditional
        engine-startup hook to avoid rank divergence on save-free steps.
        """
        if self._worker_initialized:
            return

        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            # TP=1 single-rank fallback.
            self._tp_rank = 0
            self._tp_world_size = 1
        else:
            try:
                from vllm.distributed.parallel_state import get_tp_group

                tp_group = get_tp_group()
                self._tp_rank = tp_group.rank_in_group
                self._tp_world_size = tp_group.world_size
            except Exception:
                self._tp_rank = torch.distributed.get_rank()
                self._tp_world_size = torch.distributed.get_world_size()

        if not self._shared_arena_nonce:
            raise ValueError(
                "shared_arena_nonce missing from ec_connector_extra_config; "
                "main.py must populate it"
            )
        shm_name = f"dynamo-ec-{self._engine_id}-{self._shared_arena_nonce[:8]}"
        # Trim down to a portable shm name (POSIX limits are usually ~31 chars
        # on macOS, much higher on Linux; we expect Linux but truncate
        # defensively).
        shm_name = shm_name.replace(".", "_")[-200:]

        if self._tp_rank == 0:
            # Best-effort cleanup if a prior crashed engine left a stale shm.
            try:
                stale = shared_memory.SharedMemory(name=shm_name)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            self._shm = shared_memory.SharedMemory(
                name=shm_name, create=True, size=self._capacity_bytes
            )

        if self._tp_world_size > 1:
            # TP-group barrier so non-zero ranks don't try to attach before
            # rank 0 has created the shm. Using the TP group (not world)
            # keeps DP siblings from blocking each other at startup.
            from vllm.distributed.parallel_state import get_tp_group

            get_tp_group().barrier()

        if self._tp_rank != 0:
            self._shm = shared_memory.SharedMemory(name=shm_name)

        assert self._shm is not None
        addr = ctypes.addressof(ctypes.c_char.from_buffer(self._shm.buf))
        self._shm_addr = addr
        # cudaHostRegisterPortable = 0x01.
        err = torch.cuda.cudart().cudaHostRegister(addr, self._capacity_bytes, 1)
        if err.value != 0:
            raise RuntimeError(
                f"cudaHostRegister failed (rank={self._tp_rank}): err={err}"
            )

        self._arena_uint8 = torch.frombuffer(self._shm.buf, dtype=torch.uint8)

        device_idx = torch.cuda.current_device()
        self._device = torch.device(f"cuda:{device_idx}")
        self._save_stream = torch.cuda.Stream(device=self._device)

        # Allocate the per-rank pre-allocated GPU pool. Skipped when
        # sizing was 0 — `start_load_caches` falls through to the
        # caching-allocator path verbatim. Emit a warning when disabled
        # so an operator catches missing scheduler_config / config-shape
        # drift before nsys shows the regression.
        if self._pool_size_bytes > 0:
            self._gpu_pool = _GpuPoolAllocator(
                pool_bytes=self._pool_size_bytes, device=self._device
            )
        else:
            logger.warning(
                "EC GPU pool DISABLED: rank=%d/%d reason=%s; load path will "
                "use the PyTorch caching allocator (per-step cudaMallocAsync)",
                self._tp_rank,
                self._tp_world_size,
                self._pool_disabled_reason,
            )

        self._worker_initialized = True
        logger.info(
            "EC shared arena ready: rank=%d/%d shm=%s bytes=%d device=%s "
            "gpu_pool_bytes=%d",
            self._tp_rank,
            self._tp_world_size,
            shm_name,
            self._capacity_bytes,
            self._device,
            self._gpu_pool.capacity_bytes if self._gpu_pool is not None else 0,
        )

    def _on_step_begin(self) -> None:
        """Plan A only: drain streams and TP-barrier so prior-step D2Hs are
        globally visible before any H2D in this step. No-op when Plan B's
        NCCL fence is active — the previous step's LLM forward already
        served as the cross-rank fence."""
        if self._nccl_fence_active:
            return
        assert self._save_stream is not None
        self._save_stream.synchronize()
        torch.cuda.current_stream().synchronize()
        if self._tp_world_size > 1:
            from vllm.distributed.parallel_state import get_tp_group

            get_tp_group().barrier()

    _LOG_EVERY_N_STEPS: int = 100

    def _maybe_log_distribution(self) -> None:
        """Periodic per-rank counter dump. Cheap (one int compare) on
        every-other step; logs once per _LOG_EVERY_N_STEPS multimodal-
        active steps. Lets an operator confirm saves spread across ranks
        and validate pool steady-state (alive + free + fragmentation)."""
        self._mm_active_steps += 1
        if self._mm_active_steps % self._LOG_EVERY_N_STEPS != 0:
            return
        if self._gpu_pool is not None:
            p_alive, p_free, p_runs, p_largest = self._gpu_pool.stats()
            pool_field = (
                f" pool_alive_mb={p_alive / 1024 / 1024:.2f}"
                f" pool_free_mb={p_free / 1024 / 1024:.2f}"
                f" pool_largest_free_mb={p_largest / 1024 / 1024:.2f}"
                f" pool_free_runs={p_runs}"
                f" pool_groups={len(self._pool_groups)}"
                f" pool_oom_count={self._pool_oom_count}"
            )
        else:
            pool_field = " pool=disabled"
        logger.info(
            "ec_connector stats: rank=%d/%d step=%d saves_issued=%d "
            "loads_issued=%d sync_mode=%s%s",
            self._tp_rank,
            self._tp_world_size,
            self._mm_active_steps,
            self._saves_issued,
            self._loads_issued,
            "PlanB" if self._nccl_fence_active else "PlanA",
            pool_field,
        )

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MultimodalEmbeddingCacheConnectorMetadata)

        # Pool alive sweep — runs BEFORE the empty-metadata early return
        # so pool accounting doesn't go stale across MM-inactive steps.
        # Per-group lifetime: a slab is freed only when ALL its hashes
        # have dropped out of `encoder_cache`. The model runner pops
        # entries from `encoder_cache` once the LLM forward consumes
        # them — that's the alive-set source of truth.
        if self._gpu_pool is not None and self._pool_groups:
            survivors: list[_PoolGroup] = []
            for grp in self._pool_groups:
                dead = grp.hashes - encoder_cache.keys()
                for h in dead:
                    # Only clear the mapping if this group still owns h.
                    # Defensive against duplicate-hash loads where a later
                    # alloc may have remapped the hash to a newer group.
                    if self._hash_to_group.get(h) is grp:
                        self._hash_to_group.pop(h, None)
                grp.hashes -= dead
                if not grp.hashes:
                    self._gpu_pool.free(grp.offset, grp.nbytes)
                else:
                    survivors.append(grp)
            self._pool_groups = survivors

        if not (metadata.loads or metadata.saves or metadata.evicts):
            return

        self._setup_worker()
        self._on_step_begin()

        compute = torch.cuda.current_stream()
        # Coalesce contiguous arena spans into one cudaMemcpyAsync each.
        # For hit-heavy steps in big-model sweeps (e.g. 397B / conc=30 /
        # 5 imgs), the alternative is many small H2Ds. Sort by offset,
        # walk groups of contiguous cmds, issue one H2D per group, then
        # expose per-cmd tensors as PyTorch views into the packed
        # allocation. With a pre-allocated GPU pool the destination is
        # a slab from the pool (no per-step `cudaMallocAsync`). Pool-OOM
        # falls back to PR #9304's `.to(...)` path so the coalescing
        # win is preserved even when the pool can't serve the group.
        # De-duplicate by mm_hash (defensive against pathological
        # metadata): the first occurrence wins, so a single slab backs
        # the hash and `_pool_groups` accounting stays unique.
        seen_hashes: set[str] = set()
        pending: list[_LoadCmd] = []
        for cmd in metadata.loads:
            if cmd.mm_hash in encoder_cache or cmd.mm_hash in seen_hashes:
                continue
            seen_hashes.add(cmd.mm_hash)
            pending.append(cmd)
        pending.sort(key=lambda c: c.offset)
        i, n = 0, len(pending)
        while i < n:
            j = i + 1
            while (
                j < n
                and pending[j - 1].offset + pending[j - 1].nbytes == pending[j].offset
            ):
                j += 1

            assert self._arena_uint8 is not None
            group = pending[i:j]
            group_start = group[0].offset
            group_total = group[-1].offset + group[-1].nbytes - group_start
            cpu_uint8 = self._arena_uint8.narrow(0, group_start, group_total)

            pool_offset: int | None = None
            if self._gpu_pool is not None:
                pool_offset = self._gpu_pool.alloc(group_total)
                if pool_offset is None:
                    self._maybe_log_pool_oom(group_total)

            if pool_offset is not None:
                packed = self._gpu_pool.buffer.narrow(0, pool_offset, group_total)
                packed.copy_(cpu_uint8, non_blocking=True)
                grp = _PoolGroup(
                    offset=pool_offset,
                    nbytes=group_total,
                    hashes={c.mm_hash for c in group},
                )
                self._pool_groups.append(grp)
                for c in group:
                    self._hash_to_group[c.mm_hash] = grp
            else:
                # Fallback: PR #9304's group-packed cudaMallocAsync path.
                packed = cpu_uint8.to(device=compute.device, non_blocking=True)

            for cmd in group:
                rel = cmd.offset - group_start
                view = (
                    packed.narrow(0, rel, cmd.nbytes)
                    .view(self._model_dtype)
                    .view(cmd.num_embeds, self._feature_dim)
                )
                # `record_stream(compute)` is correctness-critical for
                # the FALLBACK packed allocation: the caching allocator
                # may recycle the storage as soon as `encoder_cache.pop(h)`
                # drops the last ref. For pool-backed views, backing
                # storage is held by `_GpuPoolAllocator._buffer` and
                # suballocation lifetime is enforced by `_pool_groups`,
                # so this call is harmless. Single-path code keeps the
                # diff minimal.
                view.record_stream(compute)
                encoder_cache[cmd.mm_hash] = view
                self._loads_issued += 1

            i = j

        # Evict commands carry no worker-side state to clean up — the
        # scheduler-side allocator owns host-arena slot lifetime; pool
        # slabs are freed by the alive sweep at the next step's entry.

        self._maybe_log_distribution()

    def _maybe_log_pool_oom(self, needed_bytes: int) -> None:
        """Rate-limited warn on pool exhaustion.

        Bumps `_pool_oom_count` BEFORE the rate-limit gate so the
        periodic `ec_connector stats:` line reflects total OOMs even
        when the warn log is suppressed.
        """
        self._pool_oom_count += 1
        if (self._mm_active_steps - self._pool_oom_log_step) < self._LOG_EVERY_N_STEPS:
            return
        self._pool_oom_log_step = self._mm_active_steps
        if self._gpu_pool is None:
            return
        alive, free, runs, largest = self._gpu_pool.stats()
        logger.warning(
            "ec_connector pool exhausted rank=%d/%d step=%d need=%d B "
            "alive=%d B free=%d B free_runs=%d largest_free=%d B; "
            "falling back to caching allocator",
            self._tp_rank,
            self._tp_world_size,
            self._mm_active_steps,
            needed_bytes,
            alive,
            free,
            runs,
            largest,
        )

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MultimodalEmbeddingCacheConnectorMetadata)

        save_cmd = self._save_cmd_by_hash.get(mm_hash)
        if save_cmd is None:
            return
        if not self._worker_initialized:
            # start_load_caches normally runs first, but if the model runner
            # ever flips the order on a save-only step, set up here too.
            self._setup_worker()
        if save_cmd.writer_rank != self._tp_rank:
            # Other ranks rely on the cudaHostRegister'd shared mapping;
            # only the assigned writer rank issues the D2H.
            return
        if mm_hash not in encoder_cache:
            logger.warning(
                "save_caches: hash %s in metadata.saves but not in encoder_cache",
                mm_hash,
            )
            return

        src = encoder_cache[mm_hash]
        if not src.is_contiguous():
            src = src.contiguous()

        actual = src.numel() * src.element_size()
        if actual != save_cmd.nbytes:
            raise RuntimeError(
                f"EC save_caches size mismatch for {mm_hash}: "
                f"src={actual} cmd={save_cmd.nbytes} "
                f"(num_embeds={save_cmd.num_embeds} feature_dim={self._feature_dim})"
            )

        # Byte-level copy: reinterpret src as flat uint8 and copy into the
        # arena slot. Shape-agnostic — works for any encoder output layout
        # as long as numel() * element_size() matches the reserved nbytes.
        assert self._arena_uint8 is not None
        cpu_uint8 = self._arena_uint8.narrow(0, save_cmd.offset, save_cmd.nbytes)
        src_uint8 = src.view(torch.uint8).reshape(-1)

        compute = torch.cuda.current_stream(src.device)
        assert self._save_stream is not None
        self._save_stream.wait_stream(compute)
        with torch.cuda.stream(self._save_stream):
            cpu_uint8.copy_(src_uint8, non_blocking=True)
            # vLLM may pop encoder_cache[h] before our async D2H finishes;
            # protect the source storage with a record_stream on save_stream.
            src.record_stream(self._save_stream)

        self._saves_issued += 1

        if self._nccl_fence_active:
            # Plan B: just mark that a D2H was enqueued on save_stream
            # this step. `clear_connector_metadata` records exactly one
            # event at the stream's tail, which strictly happens-after
            # every save enqueued serially on the same stream. One
            # cudaEventCreate+Record per step instead of one per save.
            self._save_pending_this_step = True

    def clear_connector_metadata(self) -> None:
        """Plan B fence point + per-step state reset.

        Plan B: if any D2H was enqueued on `_save_stream` this step,
        record one CUDA event at the stream's tail and have
        compute_stream wait on it. The next op on compute_stream is
        the LLM forward's first TP all-reduce — that NCCL collective
        is the cross-rank visibility point. Recording the event here
        instead of inside `save_caches` collapses N events per step
        down to 1; equivalent because every D2H this step is enqueued
        serially on the same stream.

        Plan A path leaves `_save_pending_this_step` False, so the
        fence branch is a no-op.

        Always reset the per-step `_save_cmd_by_hash` dict so we
        don't pin _SaveCmd objects across non-multimodal steps where
        `bind_connector_metadata` may not be re-called with our type.
        """
        if self._save_pending_this_step:
            assert self._save_stream is not None
            done = torch.cuda.Event()
            done.record(self._save_stream)
            torch.cuda.current_stream().wait_event(done)
            self._save_pending_this_step = False
        self._save_cmd_by_hash = {}
        super().clear_connector_metadata()

    # ==================================================================
    # Cleanup
    # ==================================================================

    def _release_shm(self) -> None:
        if self._released:
            return
        self._released = True
        # Drop the torch view first so the underlying buffer reference goes
        # away before we close/unlink the shm.
        self._arena_uint8 = None
        try:
            if self._shm_addr != 0 and self._capacity_bytes > 0:
                torch.cuda.cudart().cudaHostUnregister(self._shm_addr)
        except Exception:  # noqa: BLE001 — atexit, never propagate
            pass
        try:
            if self._shm is not None:
                self._shm.close()
                if self._tp_rank == 0:
                    self._shm.unlink()
        except Exception:  # noqa: BLE001 — atexit, never propagate
            pass
