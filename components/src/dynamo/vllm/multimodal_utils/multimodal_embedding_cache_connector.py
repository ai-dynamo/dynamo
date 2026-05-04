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
from vllm.v1.utils import record_function_or_nullcontext

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

MINIMUM_VLLM_VERSION = "0.17.0"

# init_logger so we land in the vllm-handler tree inside the EngineCore
# subprocess; logging.getLogger(__name__) drops INFO records there.
logger = init_logger("vllm.dynamo_ec_connector")


@dataclass
class MultimodalEmbeddingCacheConnectorMetadata(ECConnectorMetadata):
    """Commands from scheduler to worker for CPU embedding cache management."""

    loads: list[str] = field(default_factory=list)
    saves: list[str] = field(default_factory=list)
    evicts: list[str] = field(default_factory=list)


class DynamoMultimodalEmbeddingCacheConnector(ECConnectorBase):
    """EC connector with scheduler-authoritative CPU embedding cache.

    The scheduler maintains a logical LRU cache (OrderedDict) and issues
    load/save/evict commands to the worker via ECConnectorMetadata. The
    worker holds a plain dict[str, Tensor] on CPU and obeys commands
    without independent caching decisions.

    This mirrors vLLM's EncoderCacheManager pattern: the scheduler is the
    single source of truth for cache state; the worker is a plain dict storage.
    """

    _log_every_n_steps: int = 100

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

        if "multimodal_embedding_cache_capacity_gb" not in (
            transfer_config.ec_connector_extra_config or {}
        ):
            raise ValueError(
                "multimodal_embedding_cache_capacity_gb must be set in "
                "ec_connector_extra_config for DynamoMultimodalEmbeddingCacheConnector"
            )
        capacity_gb: float = transfer_config.ec_connector_extra_config[
            "multimodal_embedding_cache_capacity_gb"
        ]

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

        # --- Worker-side: dumb CPU tensor store ---
        self._cpu_store: dict[str, torch.Tensor] = {}
        self._step_counter: int = 0

        logger.info(
            "DynamoMultimodalEmbeddingCacheConnector initialized: "
            "role=%s, capacity_gb=%.2f, capacity_bytes=%d, bytes_per_embed=%d",
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
        with record_function_or_nullcontext("ec_connector: scheduler_has_cache_item"):
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
        with record_function_or_nullcontext(
            "ec_connector: scheduler_update_state_after_alloc"
        ):
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
                with record_function_or_nullcontext("ec_connector: scheduler_evict"):
                    while (
                        self._num_used_bytes + size_bytes > self._capacity_bytes
                        and self._cache_order
                    ):
                        evicted_hash, evicted_bytes = self._cache_order.popitem(
                            last=False
                        )
                        self._num_used_bytes -= evicted_bytes
                        self._evicts_this_step.add(evicted_hash)
                        self._total_evictions += 1

            self._cache_order[mm_hash] = size_bytes
            self._num_used_bytes += size_bytes

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        """Flush accumulated load/save/evict commands into metadata for the worker."""
        with record_function_or_nullcontext("ec_connector: build_connector_meta"):
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

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        """Copy cached embeddings from CPU store to GPU encoder_cache, and evict
        entries the scheduler marked for removal.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MultimodalEmbeddingCacheConnectorMetadata)

        for mm_hash in metadata.loads:
            with record_function_or_nullcontext("ec_connector: load_item"):
                if mm_hash in encoder_cache:
                    continue
                if mm_hash in self._cpu_store:
                    with record_function_or_nullcontext("ec_connector: load_h2d"):
                        encoder_cache[mm_hash] = self._cpu_store[mm_hash].to(
                            "cuda", non_blocking=True
                        )
                else:
                    logger.warning(
                        "start_load_caches: hash %s not in cpu_store, skipping",
                        mm_hash,
                    )

        if metadata.evicts:
            with record_function_or_nullcontext("ec_connector: load_evict"):
                for mm_hash in metadata.evicts:
                    self._cpu_store.pop(mm_hash, None)

        self._step_counter += 1
        if self._step_counter % self._log_every_n_steps == 0:
            logger.info(
                "ec_connector worker: cpu_store entries=%d", len(self._cpu_store)
            )

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        """Copy a newly computed embedding from GPU encoder_cache to CPU store."""
        with record_function_or_nullcontext("ec_connector: save_caches"):
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
            with record_function_or_nullcontext("ec_connector: save_d2h"):
                self._cpu_store[mm_hash] = encoder_cache[mm_hash].cpu()
