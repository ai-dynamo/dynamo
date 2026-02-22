# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


class DynamoMultimodalEmbeddingCacheConnector(ECConnectorBase):
    """EC connector with scheduler-authoritative CPU embedding cache.

    The scheduler maintains a logical LRU cache (OrderedDict) and issues
    load/save/evict commands to the worker via ECConnectorMetadata. The
    worker holds a plain dict[str, Tensor] on CPU and obeys commands
    without independent caching decisions.

    This mirrors vLLM's EncoderCacheManager pattern: the scheduler is the
    single source of truth for cache state; the worker is dumb storage.
    """

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole) -> None:
        if Version(_vllm_version) < Version(MINIMUM_VLLM_VERSION):
            logger.warning(
                "DynamoMultimodalEmbeddingCacheConnector requires vLLM >= %s, "
                "but found %s. Some features may not work correctly.",
                MINIMUM_VLLM_VERSION,
                _vllm_version,
            )
        logger.info("DynamoMultimodalEmbeddingCacheConnector __init__ START")
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
        self._saves_this_step: dict[str, int] = {}
        self._evicts_this_step: set[str] = set()

        self._has_cache_item_hits: int = 0
        self._has_cache_item_misses: int = 0

        # --- Worker-side: dumb CPU tensor store ---
        self._cpu_store: dict[str, torch.Tensor] = {}

        logger.info(
            "DynamoMultimodalEmbeddingCacheConnector initialized: "
            "capacity_gb=%.2f, capacity_bytes=%d, bytes_per_embed=%d",
            capacity_gb,
            self._capacity_bytes,
            self._bytes_per_embed,
        )

    # ==============================
    # Scheduler-side methods
    # ==============================

    def has_cache_item(self, identifier: str) -> bool:
        if identifier in self._cache_order:
            self._cache_order.move_to_end(identifier)
            self._has_cache_item_hits += 1
            total = self._has_cache_item_hits + self._has_cache_item_misses
            if self._has_cache_item_hits == 1 or total % 50 == 0:
                logger.info(
                    "has_cache_item HIT: id=%s… hits=%d misses=%d "
                    "cache_size=%d used_bytes=%d",
                    identifier[:16],
                    self._has_cache_item_hits,
                    self._has_cache_item_misses,
                    len(self._cache_order),
                    self._num_used_bytes,
                )
            return True
        self._has_cache_item_misses += 1
        total = self._has_cache_item_hits + self._has_cache_item_misses
        if self._has_cache_item_misses == 1 or total % 50 == 0:
            logger.info(
                "has_cache_item MISS: id=%s… hits=%d misses=%d "
                "cache_size=%d used_bytes=%d",
                identifier[:16],
                self._has_cache_item_hits,
                self._has_cache_item_misses,
                len(self._cache_order),
                self._num_used_bytes,
            )
        return False

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        mm_hash: str = request.mm_features[index].identifier
        num_embeds: int = request.get_num_encoder_embeds(index)
        size_bytes: int = num_embeds * self._bytes_per_embed

        if mm_hash in self._cache_order:
            self._cache_order.move_to_end(mm_hash)
            self._loads_this_step.add(mm_hash)
            return

        if size_bytes > self._capacity_bytes:
            return

        self._saves_this_step[mm_hash] = size_bytes

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
        meta = MultimodalEmbeddingCacheConnectorMetadata(
            loads=list(self._loads_this_step),
            saves=list(self._saves_this_step.keys()),
            evicts=list(self._evicts_this_step),
        )

        if meta.loads or meta.saves or meta.evicts:
            logger.info(
                "build_connector_meta: loads=%d, saves=%d, evicts=%d",
                len(meta.loads),
                len(meta.saves),
                len(meta.evicts),
            )

        self._loads_this_step.clear()
        self._saves_this_step.clear()
        self._evicts_this_step.clear()
        return meta

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MultimodalEmbeddingCacheConnectorMetadata)

        for mm_hash in metadata.loads:
            if mm_hash in encoder_cache:
                continue
            if mm_hash in self._cpu_store:
                encoder_cache[mm_hash] = self._cpu_store[mm_hash].to(
                    "cuda", non_blocking=True
                )
            else:
                logger.warning(
                    "start_load_caches: hash %s not in cpu_store, skipping", mm_hash
                )

        for mm_hash in metadata.evicts:
            self._cpu_store.pop(mm_hash, None)

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
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
        self._cpu_store[mm_hash] = encoder_cache[mm_hash].cpu()
