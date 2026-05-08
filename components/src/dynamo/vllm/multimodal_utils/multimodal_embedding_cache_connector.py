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

        # Plan B: latest save_done event recorded this step, drained at
        # `clear_connector_metadata` against compute_stream so the LLM
        # forward's TP NCCL all-reduce serves as the cross-rank fence.
        self._latest_save_event: torch.cuda.Event | None = None

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
            "partition_bytes=%d bytes_per_embed=%d sync_mode=%s (%s)",
            role.name,
            self._tp_world_size,
            capacity_gb,
            self._capacity_bytes,
            self._partition_bytes,
            self._bytes_per_embed,
            "PlanB" if self._nccl_fence_active else "PlanA",
            self._fence_reason,
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

        self._worker_initialized = True
        logger.info(
            "EC shared arena ready: rank=%d/%d shm=%s bytes=%d device=%s",
            self._tp_rank,
            self._tp_world_size,
            shm_name,
            self._capacity_bytes,
            self._device,
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
        active steps. Lets an operator confirm saves spread across ranks."""
        self._mm_active_steps += 1
        if self._mm_active_steps % self._LOG_EVERY_N_STEPS != 0:
            return
        logger.info(
            "ec_connector stats: rank=%d/%d step=%d saves_issued=%d loads_issued=%d sync_mode=%s",
            self._tp_rank,
            self._tp_world_size,
            self._mm_active_steps,
            self._saves_issued,
            self._loads_issued,
            "PlanB" if self._nccl_fence_active else "PlanA",
        )

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MultimodalEmbeddingCacheConnectorMetadata)

        if not (metadata.loads or metadata.saves or metadata.evicts):
            return

        self._setup_worker()
        self._on_step_begin()

        compute = torch.cuda.current_stream()
        for cmd in metadata.loads:
            if cmd.mm_hash in encoder_cache:
                continue
            assert self._arena_uint8 is not None
            # H2D as raw bytes, then reinterpret on the GPU side.
            cpu_uint8 = self._arena_uint8.narrow(0, cmd.offset, cmd.nbytes)
            gpu_uint8 = cpu_uint8.to(device=compute.device, non_blocking=True)
            gpu_tensor = gpu_uint8.view(self._model_dtype).view(
                cmd.num_embeds, self._feature_dim
            )
            # Tell the caching allocator the destination is consumed by
            # compute; vLLM may pop encoder_cache[h] before compute drains.
            gpu_tensor.record_stream(compute)
            encoder_cache[cmd.mm_hash] = gpu_tensor
            self._loads_issued += 1

        # Evict commands carry no worker-side state to clean up — the
        # scheduler-side allocator owns slot lifetime.

        self._maybe_log_distribution()

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MultimodalEmbeddingCacheConnectorMetadata)

        save_cmd: _SaveCmd | None = None
        for cmd in metadata.saves:
            if cmd.mm_hash == mm_hash:
                save_cmd = cmd
                break
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
            # Plan B: record the save_done event on save_stream and keep the
            # most recent one for this step. clear_connector_metadata will
            # drain it against compute_stream so the next LLM-forward NCCL
            # all-reduce serves as the cross-rank fence.
            done = torch.cuda.Event()
            done.record(self._save_stream)
            self._latest_save_event = done

    def clear_connector_metadata(self) -> None:
        """Plan B fence point. After every metadata-bearing step, drain
        the latest save_done event against compute_stream. The next op on
        compute_stream is the LLM forward's first TP all-reduce — that NCCL
        collective is the cross-rank visibility point.

        Plan A path leaves _latest_save_event as None (we never record it
        in save_caches when the fence is inactive), so this is a no-op."""
        if self._latest_save_event is not None:
            torch.cuda.current_stream().wait_event(self._latest_save_event)
            self._latest_save_event = None
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
