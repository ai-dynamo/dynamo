# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
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


def _get_device(vllm_config: "VllmConfig") -> str:
    device_config = getattr(vllm_config, "device_config", None)
    for field_name in ("device", "device_type"):
        device = getattr(device_config, field_name, None)
        if isinstance(device, torch.device):
            device = device.type
        if isinstance(device, str) and device in ("cuda", "xpu"):
            return device

    target_device = os.environ.get("VLLM_TARGET_DEVICE")
    if target_device in ("cuda", "xpu"):
        return target_device

    return "cuda"


@dataclass
class MultimodalEmbeddingCacheConnectorMetadata(ECConnectorMetadata):
    """Commands from scheduler to worker for CPU embedding cache management."""

    loads: list[str] = field(default_factory=list)
    saves: list[str] = field(default_factory=list)
    evicts: list[str] = field(default_factory=list)


class _SchedulerCacheMetrics:
    """Prometheus metrics for the scheduler-side logical CPU cache.

    Emits the same EmbeddingCacheMetrics family as
    register_embedding_cache_metrics (the worker-layer
    MultimodalEmbeddingCacheManager used by encode-routing deployments), so
    dashboards see one embedding-cache family regardless of which
    implementation serves the model.

    The scheduler-side connector runs in vLLM's EngineCore process, not the
    process serving /metrics. Values reach the frontend through Dynamo's
    multiprocess Prometheus setup: PROMETHEUS_MULTIPROC_DIR is set before the
    engine starts (see setup_vllm_engine), so prometheus_client mmap-persists
    values that setup_metrics_collection's MultiProcessCollector exposes. The
    private CollectorRegistry keeps these metrics out of this process's global
    REGISTRY so they are never exported twice when the engine core runs
    in-process.
    """

    def __init__(
        self, model_name: str, component_name: str, capacity_bytes: int, dp_rank: int
    ) -> None:
        # Deferred imports: prometheus_client must be imported after
        # PROMETHEUS_MULTIPROC_DIR is set (inherited from the parent process).
        from prometheus_client import CollectorRegistry, Counter, Gauge

        from dynamo.common.utils.prometheus import EmbeddingCacheMetrics as ECM
        from dynamo.prometheus_names import labels

        self._capacity_bytes = capacity_bytes
        self.registry = CollectorRegistry()
        # dp_rank partitions series per data-parallel EngineCore (each rank has
        # its own scheduler-side cache), so gauges never collide across ranks
        # and "mostrecent" only arbitrates between a dead pre-restart pid and
        # its live replacement within one rank. Same pattern as the kvstats
        # gauges in LLMBackendMetrics.
        labelnames = [labels.MODEL, labels.COMPONENT, labels.DP_RANK]
        labelvalues = {
            labels.MODEL: model_name,
            labels.COMPONENT: component_name,
            labels.DP_RANK: str(dp_rank),
        }

        def _counter(name: str, doc: str):
            return Counter(name, doc, labelnames, registry=self.registry).labels(
                **labelvalues
            )

        def _gauge(name: str, doc: str):
            # "mostrecent" so MultiProcessCollector reports this process's
            # latest snapshot instead of aggregating across dead pids.
            return Gauge(
                name,
                doc,
                labelnames,
                registry=self.registry,
                multiprocess_mode="mostrecent",
            ).labels(**labelvalues)

        self._hits = _counter(ECM.HITS_TOTAL, "Total embedding cache hits.")
        self._misses = _counter(ECM.MISSES_TOTAL, "Total embedding cache misses.")
        self._evictions = _counter(
            ECM.EVICTIONS_TOTAL, "Total embedding cache evictions."
        )
        self._utilization = _gauge(
            ECM.UTILIZATION, "Cache memory utilization ratio (0.0-1.0)."
        )
        self._current_bytes = _gauge(
            ECM.CURRENT_BYTES, "Current cache memory usage in bytes."
        )
        self._entries = _gauge(ECM.ENTRIES, "Number of entries in the cache.")
        self.update_usage(0, 0)

    def record_hit(self) -> None:
        self._hits.inc()

    def record_miss(self) -> None:
        self._misses.inc()

    def record_evictions(self, count: int) -> None:
        self._evictions.inc(count)

    def update_usage(self, used_bytes: int, entries: int) -> None:
        self._current_bytes.set(used_bytes)
        self._entries.set(entries)
        self._utilization.set(
            used_bytes / self._capacity_bytes if self._capacity_bytes else 0.0
        )


class DynamoMultimodalEmbeddingCacheConnector(ECConnectorBase):
    """EC connector with scheduler-authoritative CPU embedding cache.

    The scheduler maintains a logical LRU cache (OrderedDict) and issues
    load/save/evict commands to the worker via ECConnectorMetadata. The
    worker holds a plain dict[str, Tensor] on CPU and obeys commands
    without independent caching decisions.

    This mirrors vLLM's EncoderCacheManager pattern: the scheduler is the
    single source of truth for cache state; the worker is a plain dict storage.
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
        self._device = _get_device(vllm_config)

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

        # Only the scheduler role does cache accounting, so only it emits
        # metrics; worker-role instances would just publish idle zeros.
        extra_config = transfer_config.ec_connector_extra_config
        self._metrics: _SchedulerCacheMetrics | None = (
            _SchedulerCacheMetrics(
                model_name=extra_config.get("model_name", ""),
                component_name=extra_config.get("component", ""),
                capacity_bytes=self._capacity_bytes,
                dp_rank=getattr(vllm_config.parallel_config, "data_parallel_rank", 0),
            )
            if role == ECConnectorRole.SCHEDULER
            else None
        )

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
            if self._metrics is not None:
                self._metrics.record_hit()
            return

        if self._metrics is not None:
            self._metrics.record_miss()

        if size_bytes > self._capacity_bytes:
            return

        self._saves_this_step.add(mm_hash)

        num_evicted = 0
        while (
            self._num_used_bytes + size_bytes > self._capacity_bytes
            and self._cache_order
        ):
            evicted_hash, evicted_bytes = self._cache_order.popitem(last=False)
            self._num_used_bytes -= evicted_bytes
            self._evicts_this_step.add(evicted_hash)
            num_evicted += 1

        self._cache_order[mm_hash] = size_bytes
        self._num_used_bytes += size_bytes

        if self._metrics is not None:
            if num_evicted:
                self._metrics.record_evictions(num_evicted)
            self._metrics.update_usage(self._num_used_bytes, len(self._cache_order))

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

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        """Copy cached embeddings from CPU store to GPU encoder_cache, and evict
        entries the scheduler marked for removal.
        """
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, MultimodalEmbeddingCacheConnectorMetadata)

        for mm_hash in metadata.loads:
            if mm_hash in encoder_cache:
                continue
            if mm_hash in self._cpu_store:
                encoder_cache[mm_hash] = self._cpu_store[mm_hash].to(
                    self._device, non_blocking=True
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
        """Copy a newly computed embedding from GPU encoder_cache to CPU store."""
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
